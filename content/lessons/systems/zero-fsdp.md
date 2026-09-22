---
summary: Sharding optimizer state, gradients and parameters across data-parallel ranks — the three stages, what each costs in communication, and how to configure it.
prereqs: [data-parallelism, tensors-dtypes-devices, model-parallelism]
---

Plain data parallelism replicates everything on every rank. For a 7B model that is 126 GB per
GPU of redundant state (lesson 2.01). ZeRO removes the redundancy by sharding it, and FSDP is
PyTorch's implementation.

## The memory being wasted

Per parameter, in mixed-precision AdamW:

| Component | Bytes | Redundant across ranks? |
|---|---|---|
| bf16 parameters | 2 | Needed for the forward |
| fp32 master copy | 4 | **Yes** |
| fp32 gradients | 4 | **Yes** — identical after all-reduce |
| Adam first moment | 4 | **Yes** |
| Adam second moment | 4 | **Yes** |

16 of the 18 bytes are identical on every rank. With $N$ ranks that is $16(N-1)/N$ bytes per
parameter of pure waste.

## The three stages

::: key
**ZeRO-1** shards the optimizer state. Each rank updates only its slice of parameters, then
an all-gather distributes the results. Memory per rank: $2 + 4 + 12/N$ bytes per parameter.

**ZeRO-2** also shards gradients. The all-reduce becomes a **reduce-scatter** — each rank
ends up with only the gradient slice it needs. Memory: $2 + 4/N + 12/N$.

**ZeRO-3** also shards parameters. Each rank holds $1/N$ of the weights and all-gathers a
layer's parameters just before using it, then frees them. Memory: $(2 + 4 + 12)/N$.

Communication volume: ZeRO-1 and ZeRO-2 are the **same as plain data parallelism** — a
reduce-scatter plus an all-gather is exactly an all-reduce. They are free memory savings.

ZeRO-3 adds an all-gather of parameters per layer in the forward pass and again in the
backward, roughly **1.5× the communication**. That is the real cost, and it is worth paying
only when the model does not otherwise fit.
:::

For a 7B model on 8 GPUs:

| | Per-GPU memory |
|---|---|
| Plain DDP | 126 GB — does not fit |
| ZeRO-1 | 55 GB |
| ZeRO-2 | 44 GB |
| ZeRO-3 | 16 GB |

## FSDP

```python
import functools
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},      # wrap each block separately
)

model = FSDP(
    model,
    auto_wrap_policy=policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,     # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,                    # reduce in fp32 for accuracy
        buffer_dtype=torch.bfloat16,
    ),
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,       # bound prefetch depth to control peak memory
    use_orig_params=True,         # needed for torch.compile and param groups
)
```

::: warning
**The wrap policy is the most consequential setting.** FSDP shards at the granularity of
wrapped units, and it all-gathers a whole unit before using it.

Wrap the entire model as one unit and you all-gather every parameter at once — no memory
saving at all during the forward pass. Wrap every individual `nn.Linear` and you get hundreds
of tiny collectives whose per-call overhead dominates.

Wrap per **transformer block**. That is the right granularity: large enough that collectives
are efficient, small enough that only one block's parameters are materialised at a time.
:::

The strategies map onto the ZeRO stages:

| `ShardingStrategy` | ZeRO stage |
|---|---|
| `NO_SHARD` | Plain DDP |
| `SHARD_GRAD_OP` | ZeRO-2 |
| `FULL_SHARD` | ZeRO-3 |
| `HYBRID_SHARD` | ZeRO-3 inside a node, replicate across nodes |

`HYBRID_SHARD` is worth knowing: it shards over NVLink where bandwidth is plentiful and
replicates across the slower inter-node links, which often beats full sharding on multi-node
runs.

## Prefetching

FSDP overlaps the all-gather for layer $i+1$ with the computation of layer $i$. This is what
makes ZeRO-3's extra communication tolerable, and it is the same overlap principle as
DDP's bucketed gradient reduction (lesson 7.05).

`limit_all_gathers=True` caps how far ahead it prefetches. Without a limit, aggressive
prefetching can materialise several layers' parameters at once and defeat the memory saving —
the setting exists because peak memory, not average, is what makes a run fail.

::: check
ZeRO-1 and ZeRO-2 move the same number of bytes as plain data parallelism. Why?

- [x] A reduce-scatter plus an all-gather is exactly an all-reduce — so the memory savings are free in communication terms
  > ZeRO-3 is different: it adds an all-gather of parameters per layer in the forward pass and again in the backward, which is real extra traffic for a much larger saving.
- [ ] They compress gradients before sending them
  > No compression is involved. The decomposition of the collective is what makes it free.
- [ ] They communicate less often, in larger messages
  > Bucketing does batch small messages, and that is orthogonal to the ZeRO stages.
- [ ] They overlap communication with compute, hiding the cost
  > Overlap hides latency and is used at every stage. The claim here is about volume, which is genuinely identical.
:::

## CPU offload

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(model, cpu_offload=CPUOffload(offload_params=True), ...)
```

Moves sharded parameters and optimizer state to host memory, fetching them over PCIe when
needed. This lets genuinely large models train on limited GPU memory, and it is slow — PCIe
is ~32 GB/s against HBM's 3,350 GB/s (lesson 7.01), a hundredfold gap.

Use it when the alternative is not training at all. Expect a 2–5× slowdown.

## Checkpointing a sharded model

```python
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

# Distributed: every rank writes its own shard. Fast, and the only option at scale.
state = {'model': model, 'optim': optimizer}
dcp.save(state, checkpoint_id='ckpt/step-1000')

# Consolidated: gather to rank 0 as a single file. Slow, and what you want for
# distribution -- but it must fit in one rank's CPU memory.
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    if rank == 0:
        torch.save(model.state_dict(), 'model.pt')
```

Use distributed checkpoints during training — they are fast and they resume onto the same
topology. Consolidate once at the end for release.

## Choosing a configuration

| Situation | Configuration |
|---|---|
| Model fits comfortably | DDP — simplest, fastest |
| Optimizer state is the problem | ZeRO-1 / `SHARD_GRAD_OP` — free |
| Model barely fits | ZeRO-2 |
| Model does not fit | ZeRO-3 / `FULL_SHARD` |
| Multi-node with good intra-node links | `HYBRID_SHARD` |
| Nothing else works | ZeRO-3 + CPU offload + gradient checkpointing |

Always combine with gradient checkpointing (lesson 7.08) when memory-constrained — the two
attack different terms, parameters versus activations, and compose cleanly.

::: exercise
You train a 13B model with FSDP `FULL_SHARD` on 8×A100-40GB. It runs but throughput is half
of what a 7B model achieves per parameter. Where is the loss?
:::

::: solution
**ZeRO-3's parameter all-gathers are the prime suspect**, but measure before assuming.

**Quantify the expected overhead.** `FULL_SHARD` all-gathers each block's parameters in the
forward pass and again in the backward, plus the reduce-scatter of gradients. That is roughly
1.5× DDP's communication volume. Overlapped perfectly, it should cost little; overlapped
poorly, it is exposed.

**Check, in this order:**

1. **Is the wrap policy right?** If FSDP wrapped the whole model as one unit, every forward
   all-gathers all 13B parameters — no overlap possible and no memory saving during compute.
   *Check:* `print(model)` and confirm you see `FullyShardedDataParallel` around each
   transformer block, not just at the top. This is the single most common cause.

2. **Is communication overlapping?** Profile with `nsys` and look at the timeline: NCCL
   kernels should run concurrently with compute kernels. If they are serialised, prefetching
   is not working. *Fix:* confirm `forward_prefetch` is enabled and that
   `limit_all_gathers` is not set so tightly that it prevents overlap.

3. **Is the batch size too small to hide it?** Overlap needs compute to hide communication
   behind. A 13B model sharded 8 ways holds 1.6B parameters per rank — the all-gather volume
   is fixed, but the compute per step scales with batch. *Fix:* raise the per-GPU batch, using
   gradient checkpointing (lesson 7.08) to free the activation memory for it. This often
   recovers most of the gap.

4. **Do you actually need `FULL_SHARD`?** 13B in bf16 is 26 GB, which does not fit in 40 GB
   alongside optimizer state — but `SHARD_GRAD_OP` (ZeRO-2) gives $2 + 4/8 + 12/8 = 4$ bytes
   per parameter, about 52 GB total, still too much. So ZeRO-3 is likely necessary. Confirm
   by computing it rather than assuming.

5. **Is `use_orig_params=True` set?** Without it, `torch.compile` cannot trace the model and
   you lose the fusion gains of lesson 2.14 entirely — which alone could account for a large
   fraction of the gap.

**Also worth ruling out:** `HYBRID_SHARD`. On a single node with NVLink the full-shard
all-gathers are cheap, but if these 8 GPUs are split across two nodes, they are not. Check
the topology (lesson 7.01).

**The expected outcome:** after fixing the wrap policy and raising the batch, ZeRO-3 should
cost 10–20% against DDP, not 50%. A 50% gap means something is configured wrong rather than
inherent to sharding.
:::

## What to carry forward

- 16 of 18 bytes per parameter are redundant across data-parallel ranks.
- ZeRO-1 and ZeRO-2 are free; ZeRO-3 costs about 1.5× communication for a much larger saving.
- Wrap per transformer block — the granularity decides whether sharding works at all.
- Use distributed checkpoints during training, consolidated ones for release.
- Combine with gradient checkpointing; they attack different memory terms.
