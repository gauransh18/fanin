---
summary: Estimating what a training run or a serving deployment will cost, before committing — and the mistakes that make estimates wrong by an order of magnitude.
prereqs: [scaling-laws, pretraining-infrastructure, inference-serving]
---

Compute is the largest line item in most machine learning work, and it is one of the few that
can be estimated accurately in advance. The arithmetic is not hard; the errors come from
forgetting terms.

## Training cost

From lesson 1.03, training costs about $6N$ FLOPs per parameter per token:

```python
def training_cost(n_params, n_tokens, gpu_flops=989e12, mfu=0.40,
                  gpu_hourly=2.50, n_gpus=64, overhead=1.35):
    flops = 6 * n_params * n_tokens
    gpu_seconds = flops / (gpu_flops * mfu)
    wall_hours = gpu_seconds / n_gpus / 3600
    compute = gpu_seconds / 3600 * gpu_hourly
    return {
        'flops': f'{flops:.2e}',
        'gpu_hours': f'{gpu_seconds/3600:,.0f}',
        'wall_clock_days': f'{wall_hours/24:.1f}',
        'compute_usd': f'${compute:,.0f}',
        'total_usd': f'${compute * overhead:,.0f}',   # with failures and restarts
    }

print(training_cost(7e9, 2e12))     # Llama-2 7B scale
```

::: key
**The `overhead` factor is where naive estimates go wrong.** A frontier run does not spend
100% of its wall clock making forward progress:

| Source | Typical |
|---|---|
| Failed runs and restarts | 10–20% |
| Checkpointing pauses | 2–5% |
| Idle time between jobs | 5–10% |
| Debugging and profiling runs | 10–30% |
| Hyperparameter search | 20–100%+ |

A 1.35× multiplier on the compute estimate is conservative for a well-run project, and
budgets that omit it are routinely wrong by 2×. Hyperparameter search in particular can
exceed the final run's cost.
:::

**MFU is the other large lever.** At 20% instead of 40%, the run costs twice as much for
identical results. Measuring MFU (lesson 5.04) before committing to a multi-week run is the
highest-return hour in the project.

## Inference cost

```python
def inference_cost(n_params, prompt_tokens, output_tokens, requests_per_day,
                   gpu_hourly=2.50, gpu_flops=989e12, gpu_bandwidth=3.35e12,
                   bytes_per_param=2, batch_size=32):
    # Prefill: compute-bound, 2N FLOPs per token.
    prefill_s = 2 * n_params * prompt_tokens / (gpu_flops * 0.4)

    # Decode: memory-bound. Time per token is weights read / bandwidth,
    # amortised across the batch (lesson 7.09).
    per_token_s = (n_params * bytes_per_param / gpu_bandwidth) / batch_size
    decode_s = output_tokens * per_token_s

    daily_gpu_hours = requests_per_day * (prefill_s + decode_s) / 3600
    return {
        'ms_per_request': f'{(prefill_s + decode_s) * 1000:.0f}',
        'daily_usd': f'${daily_gpu_hours * gpu_hourly:,.2f}',
        'per_1k_requests_usd': f'${daily_gpu_hours * gpu_hourly / requests_per_day * 1000:.3f}',
    }

print(inference_cost(7e9, 500, 200, 1_000_000))
```

The batch size in the denominator is the whole serving story (lesson 7.09). At batch 1 the
cost per request is 32× higher than at batch 32, for identical hardware.

## The crossover

For a model that will be served heavily, inference dominates:

```python
def crossover(n_params, training_tokens, tokens_per_request, requests_per_day):
    train_flops = 6 * n_params * training_tokens
    daily_inference = 2 * n_params * tokens_per_request * requests_per_day
    return train_flops / daily_inference          # days to match training cost

print(f'{crossover(7e9, 2e12, 700, 1_000_000):.0f} days to match training cost')
```

For a 7B model on 2T tokens serving a million requests a day, inference FLOPs match training
FLOPs in about twelve days. After that, everything is inference — which is the argument for
overtraining a smaller model (lesson 5.03) and for quantization (lesson 5.12).

::: check
Where do naive training cost estimates most often go wrong?

- [x] They omit the overhead factor
  > A frontier run does not spend 100% of its wall clock making forward progress. Failures, restarts from checkpoints, stragglers and debugging time add up, and a multiplier around 1.35 is typical. The FLOP arithmetic itself is the easy part.
- [ ] They use $6N$ instead of $2N$ per token
  > $6N$ is right for a training step — one unit forward and two backward. $2N$ is the forward-only inference figure.
- [ ] They assume 100% MFU
  > Anyone writing the estimate usually knows to discount MFU. It is the *non-compute* time that gets forgotten.
- [ ] They forget that GPUs are billed per second rather than per hour
  > Billing granularity is a rounding detail next to a 35% overhead.
:::

## What actually reduces cost

Ordered by typical impact:

| Lever | Typical saving | Effort |
|---|---|---|
| Raise serving batch size | 2–10× | Low — configuration |
| Fix MFU from 20% to 40% | 2× training | Medium — profiling |
| Quantize to int4 | 2–4× serving | Low — a library call |
| Use spot or preemptible instances | 60–70% | Medium — needs checkpointing |
| Distil to a smaller model | 2–10× serving | High |
| Prompt caching for shared prefixes | 2–5× on prefill | Low |
| Speculative decoding | 2–3× latency | Medium |

::: warning
**Spot instances are the largest single discount available** — typically 60–70% off — and
they require the reliability engineering of lesson 5.04 to be in place: frequent
checkpointing, automatic restart, and a data loader resumable to the exact sample.

Attempting spot without that machinery turns a 70% discount into repeated lost work. Build
the reliability first, then take the discount.
:::

::: check
You are deciding between an API and self-hosting. What determines the crossover?

- [x] Sustained throughput
  > A self-hosted GPU costs the same whether it is busy or idle, so the break-even is about utilisation rather than per-token price. Bursty or low-volume traffic favours an API; steady high volume favours self-hosting. Add the engineering time, which is usually the larger hidden cost.
- [ ] Model quality, since self-hosted models are always weaker
  > Open-weight models are competitive for many tasks, and the question here is economic.
- [ ] The size of the model, since large models cannot be self-hosted
  > Large models can be served on multiple GPUs; that raises the fixed cost rather than ruling it out.
- [ ] Latency, which is always better self-hosted
  > It can be, and network proximity and batching policy matter more than ownership.
:::

## Buy or build

```python
def build_vs_buy(requests_per_day, tokens_per_request, api_per_1m_tokens=3.0,
                 self_hosted_daily=60.0):
    daily_tokens = requests_per_day * tokens_per_request
    api_daily = daily_tokens / 1e6 * api_per_1m_tokens
    print(f'API ${api_daily:,.2f}/day   self-hosted ${self_hosted_daily:,.2f}/day')
    if api_daily > self_hosted_daily:
        print(f'self-hosting breaks even at '
              f'{self_hosted_daily / api_per_1m_tokens * 1e6 / tokens_per_request:,.0f} req/day')
```

The compute arithmetic is the easy part. The terms usually omitted:

- **Engineering time.** Self-hosting is measured in engineer-months, not GPU-hours, and an
  engineer costs more than a GPU.
- **Utilisation.** A reserved GPU costs the same at 10% load as at 90%. API pricing is
  per-token, so it scales down with traffic and self-hosting does not.
- **Peak capacity.** You must provision for peak; an API absorbs it.
- **Model updates.** A provider improves the model continuously; self-hosting means you do
  that work.

Self-hosting wins at sustained high volume, when you need a fine-tuned model, when data
cannot leave your infrastructure, or when latency requires co-location. Below roughly a
million requests a day, an API is usually cheaper once engineering time is counted honestly.

## Estimating before committing

::: key
Run a scaled-down version and extrapolate rather than estimating from first principles:

1. Train a small model — 100M parameters, 1B tokens — with the **real** pipeline.
2. Measure MFU, tokens per second, and the failure rate.
3. Extrapolate: compute scales as $6ND$, and MFU is roughly constant across scales for the
   same architecture and parallelism layout.
4. Apply the overhead multiplier.

This takes a day and it is far more accurate than any estimate built from datasheet numbers,
because it measures your pipeline rather than the hardware's peak.
:::

::: exercise
You must decide between training a 13B model from scratch and fine-tuning an existing 70B
with LoRA, for a domain-specific application serving 100k requests a day. Work the numbers.
:::

::: solution
**Training cost.**

*13B from scratch*, Chinchilla-optimal at 260B tokens: $6 \times 13\times10^9 \times
2.6\times10^{11} \approx 2\times10^{22}$ FLOPs. At 40% MFU on H100s that is roughly 14,000
GPU-hours, about **\$35,000** at \$2.50/hour — before the 1.35× overhead, so call it
**\$47,000**. And that assumes you have 260B tokens of appropriate data, which for a
domain-specific application you almost certainly do not (lesson 5.02).

*LoRA on 70B*, 50M examples-worth is unrealistic; say 50k examples at 1k tokens each,
3 epochs: $6 \times 70\times10^9 \times 1.5\times10^8 \approx 6.3\times10^{19}$ FLOPs.
Roughly 45 GPU-hours, about **\$110**. LoRA trains under 1% of parameters (lesson 5.06), so
memory fits on a small number of GPUs.

**A 400× difference in training cost.**

**Inference cost** is where the 13B wins back ground. At 700 tokens per request and 100k
requests/day:

- 13B: $2 \times 13\times10^9 \times 700 \times 10^5 = 1.8\times10^{18}$ FLOPs/day.
- 70B: $9.8\times10^{18}$ FLOPs/day — **5.4× more**.

Memory matters more than FLOPs here. 13B in bf16 is 26 GB, fitting on one 40 GB GPU with room
for a KV cache. 70B is 140 GB, needing two 80 GB GPUs — or one at int4 (lesson 5.12).

Estimating serving cost at batch 32: the 13B might run on 2 GPUs for redundancy (\$120/day);
the 70B on 4 (\$240/day). Over a year that is a **\$44,000 difference** — comparable to the
entire training cost difference.

**The decision:**

**Start with LoRA on the 70B.** Training cost is negligible, you get a working system in
days, and you learn what the task actually requires. This is almost always the right first
move.

**Then consider distillation** (lesson 5.13), not from-scratch training. Use the fine-tuned
70B to generate data, and distil into a 7B or 13B. You get the 13B's serving economics
without the from-scratch training cost or the data requirement.

**From-scratch training is justified only if** the domain is far enough from the base model's
distribution that fine-tuning genuinely fails — a different language, a different modality —
*and* you have the data. Verify that with a LoRA experiment first; it costs \$110 to find
out.

**The mistake to avoid:** treating this as a pure compute comparison. The 400× training gap
is dwarfed by the engineering time for a from-scratch run, and by the risk that after
\$47,000 the result is worse than the LoRA.
:::

## What to carry forward

- Training is $6ND$ FLOPs; apply a 1.35× overhead for failures, debugging and search.
- MFU is a 2× lever — measure it before committing to a long run.
- Serving cost scales inversely with batch size; that is the first thing to fix.
- Inference FLOPs overtake training FLOPs within weeks at real traffic.
- Estimate by running a scaled-down version of the real pipeline, not from datasheets.
