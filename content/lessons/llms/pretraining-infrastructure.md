---
summary: What a frontier training run actually involves — parallelism layout, failure handling, and the metric that tells you whether the cluster is earning its cost.
prereqs: [scaling-laws, training-loop, mixed-precision]
---

A large pretraining run is a distributed systems problem with a model attached. The
mathematics is settled; what decides success is whether thousands of GPUs stay busy for
weeks without losing work.

## Sizing the run

Start from the FLOP rule of lesson 1.03 and the allocation of lesson 5.03:

```python
def plan_run(n_params, n_tokens, n_gpus, gpu_flops=989e12, mfu=0.45):
    flops = 6 * n_params * n_tokens
    achieved = n_gpus * gpu_flops * mfu
    seconds = flops / achieved
    return {
        'total_flops': f'{flops:.2e}',
        'gpu_days': f'{flops / (gpu_flops * mfu) / 86400:,.0f}',
        'wall_clock_days': f'{seconds / 86400:.1f}',
    }

print(plan_run(70e9, 1.4e12, n_gpus=1024))
```

**MFU — model FLOPs utilisation** — is the fraction of theoretical peak actually achieved.
It is the single number that tells you whether the cluster is earning its cost.

::: key
| MFU | Verdict |
|---|---|
| Under 25% | Something is badly wrong — profile before spending more |
| 35–45% | Typical for a well-tuned large run |
| 50–60% | Very good |
| Above 65% | Rare outside carefully tuned dense models |

A run at 20% MFU costs twice what one at 40% costs, for identical results. Measuring MFU
before committing to a multi-week run is the highest-value hour in the project.
:::

## The parallelism layout

Four axes, combined (track 7 covers each in depth):

| Axis | Splits | Communication | Where |
|---|---|---|---|
| Data (7.05) | Batch | All-reduce of gradients | Across nodes |
| Tensor (7.06) | Within a layer | All-reduce per layer | Within a node |
| Pipeline (7.06) | Layers into stages | Point-to-point | Across nodes |
| Expert (4.14) | MoE experts | All-to-all | Across nodes |

The assignment follows bandwidth. Tensor parallelism communicates twice per layer, so it
must stay inside a node where NVLink gives ~900 GB/s. Data parallelism communicates once
per step and tolerates slower InfiniBand between nodes.

```python
# A typical 1024-GPU layout for a 70B model.
config = {
    'tensor_parallel':   8,    # within a node, over NVLink
    'pipeline_parallel': 4,    # across nodes
    'data_parallel':    32,    # 8 * 4 * 32 = 1024
    'micro_batch':       1,
    'global_batch': 4_000_000, # tokens
}
```

::: check
A run reports 20% MFU. What does that tell you?

- [x] It costs twice what a 40% run costs for identical results — profile before spending more
  > MFU is the single number saying whether the cluster is earning its keep. 35–45% is typical for a well-tuned large run; under 25% means something is badly wrong.
- [ ] It is normal for large distributed runs
  > 35–45% is normal. 20% is a problem worth a day of investigation before a multi-week commitment.
- [ ] It means 20% of GPUs are idle
  > It means the GPUs that are running are achieving a fifth of their peak arithmetic rate, which can happen with every device busy.
- [ ] It is a measure of model quality, not utilisation
  > It measures achieved FLOPs against theoretical peak, and says nothing about quality.
:::

## Batch size

Large batches reduce gradient noise as $1/\sqrt{B}$ (lesson 1.11) and improve hardware
utilisation. Both have limits.

The **critical batch size** is where further increases stop reducing the steps needed to
reach a loss. Beyond it you are spending compute for nothing. It grows during training —
gradient noise falls as the model improves — which is why large runs ramp batch size over
the first several thousand steps rather than fixing it.

Frontier runs use 4M–16M tokens per batch, reached through a ramp.

## Failures are routine

At 1,024 GPUs running for a month, hardware failures are not exceptional:

- A GPU falls off the bus or throws an Xid error.
- A node loses network.
- NCCL deadlocks and every rank hangs, with no error.
- Silent data corruption produces `nan` in one rank's gradients.

::: warning
**NCCL hangs are the worst failure mode**, because nothing crashes. Every rank waits for a
collective that will never complete, the job holds its allocation, and utilisation reads
zero while the bill continues.

Defences: set `NCCL_ASYNC_ERROR_HANDLING=1` and a watchdog timeout so a stuck collective
raises instead of hanging; have the launcher kill and restart the job when heartbeats stop.
:::

What a production run needs:

**Frequent checkpoints.** Every 15–30 minutes. Write asynchronously so training continues,
and write atomically (lesson 2.11) so a preempted job never leaves a truncated file.

**Automatic restart.** A supervisor that detects failure, drains the node, and restarts
from the last checkpoint on healthy hardware. Manual restarts do not scale to a month of
nightly failures.

**Loss-spike handling.** Spikes happen. The standard recipe: on a spike above a threshold,
roll back to the last good checkpoint, skip the next several data batches, and resume.
Automate it — the alternative is losing hours to a spike that occurred while everyone was
asleep.

**Deterministic data ordering.** The data loader must be resumable to the exact sample, or
a restart silently re-trains on data already seen and skips data never seen.

## What to monitor

```python
metrics = {
    'loss':            loss.item(),
    'grad_norm':       grad_norm.item(),      # spikes precede divergence
    'lr':              sched.get_last_lr()[0],
    'mfu':             achieved_flops / peak_flops,
    'tokens_seen':     step * global_batch,
    'param_norm':      sum(p.norm() ** 2 for p in model.parameters()) ** 0.5,
    'update_ratio':    (lr * grad_norm / param_norm),   # should sit near 1e-3
    'gpu_mem':         torch.cuda.max_memory_allocated() / 1e9,
}
```

`update_ratio` is the most underused of these. It measures how far parameters move per
step relative to their size. Healthy training sits around $10^{-3}$; much larger means the
learning rate is too high, much smaller means training has effectively stalled. It catches
problems earlier than the loss curve does.

## The phases

1. **Warmup** — a few thousand steps ramping the learning rate, with batch size also
   ramping. Most divergences happen here.
2. **Main run** — the bulk, on a cosine or warmup–stable–decay schedule (lesson 2.09).
3. **Mid-training** — the last few percent of tokens on a higher-quality mixture. Reliably
   improves benchmarks and is now standard (lesson 5.02).
4. **Decay** — anneal the learning rate to near zero. With a WSD schedule this is a short
   final phase, which lets you branch several final models from one stable checkpoint.

::: exercise
Your run is at 22% MFU on 512 H100s. Walk through diagnosis in order.
:::

::: solution
**Compute the expected number first.** A well-tuned 70B run at this scale should reach
35–45%. At 22% you are paying roughly twice what you should, so this is worth a day of
investigation before continuing.

**1. Is it the data pipeline?** Time one step with real data and one with a cached tensor
repeated. If synthetic data is much faster, the loader is starving the GPUs (lesson 2.10).
Check `num_workers`, whether shards are on local NVMe rather than network storage, and
whether tokenization is happening online when it should be offline. This is the most common
cause and the cheapest to fix.

**2. Is it communication?** Profile with `torch.profiler` and look at the fraction of time
in NCCL kernels. Above 20% points at the parallelism layout. Specific checks:
- Is tensor parallelism crossing node boundaries? It must stay within NVLink.
- Is gradient all-reduce overlapping with the backward pass, or serialised after it?
- Under gradient accumulation, are you calling `no_sync()` on all but the last microbatch
  (lesson 4.09's exercise)? Without it you all-reduce once per microbatch.

**3. Is the microbatch too small?** Microbatch 1 with pipeline parallelism means large
pipeline bubbles. Compute the bubble fraction: $(p-1)/(m+p-1)$ for $p$ stages and $m$
microbatches. With 4 stages and 4 microbatches that is 43% idle. Raising microbatches per
step to 16 cuts it to 16%.

**4. Are the kernels efficient?** Confirm you are on FlashAttention rather than a fallback
(lesson 4.12), that `torch.compile` is applied and not silently recompiling (lesson 2.14),
and that bf16 autocast is actually taking effect — print activation dtypes to be sure.

**5. Is it recomputation?** Full gradient checkpointing adds about 30% FLOPs. If you are
checkpointing every layer when memory would allow checkpointing every fourth, you are
paying MFU for headroom you do not need (lesson 7.08).

**The usual answer** at this level is a combination of 1 and 3: a starved loader plus
pipeline bubbles. Both are configuration changes rather than code changes, and together
they typically recover MFU into the high 30s.
:::

## What to carry forward

- MFU is the number that says whether the cluster is earning its cost; under 25% means stop and profile.
- Tensor parallelism inside a node, data parallelism across — assignment follows bandwidth.
- Batch size has a critical point that grows during training; ramp it.
- Failures are routine; automate checkpointing, restart and spike rollback.
- Watch `update_ratio` near $10^{-3}$ — it catches problems before the loss does.
