---
summary: Splitting one model across devices — tensor, pipeline and sequence parallelism, and the bandwidth argument that assigns each to a level of the hierarchy.
prereqs: [data-parallelism, transformer-block, matrix-multiplication-cost]
---

Data parallelism needs the model to fit on one device. When it does not, you split the model
itself — and the three ways of doing that have very different communication patterns.

## Tensor parallelism

Split individual weight matrices across devices. For the transformer MLP (lesson 4.05), the
clever part is choosing which dimension to split.

**Column-parallel** for the first matrix: split $W_1 \in \mathbb{R}^{d\times 4d}$ by columns,
so each rank computes a slice of the hidden activation. The nonlinearity is elementwise, so
it applies independently — **no communication needed**.

**Row-parallel** for the second: split $W_2 \in \mathbb{R}^{4d\times d}$ by rows, so each
rank produces a partial sum of the output. One all-reduce combines them.

$$
\text{GeLU}(XW_1)W_2 = \sum_{i} \text{GeLU}(XW_1^{(i)})\,W_2^{(i)}
$$

::: key
Column-then-row is the pairing that matters: it gives **one all-reduce per MLP block**
instead of two. Splitting both by columns, or both by rows, would require communicating
between them as well.

The same pattern applies to attention: split by **heads** (columns of $W_Q$, $W_K$, $W_V$),
compute attention independently per head, then row-split $W_O$ and all-reduce once. Multi-head
attention is naturally tensor-parallel because the heads are already independent
(lesson 4.03).
:::

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """Splits output features. No communication in forward."""
    def __init__(self, in_features, out_features, world, rank):
        super().__init__()
        assert out_features % world == 0
        self.weight = nn.Parameter(torch.empty(out_features // world, in_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x @ self.weight.T                   # (..., out_features // world)


class RowParallelLinear(nn.Module):
    """Splits input features. One all-reduce in forward."""
    def __init__(self, in_features, out_features, world, rank):
        super().__init__()
        assert in_features % world == 0
        self.weight = nn.Parameter(torch.empty(out_features, in_features // world))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        out = x @ self.weight.T                    # partial sum
        dist.all_reduce(out)                       # combine across ranks
        return out
```

**The cost**: two all-reduces per transformer layer (one for attention, one for the MLP) in
the forward pass, and two more in the backward. For a 32-layer model that is 128 collectives
per step, each of size $BTd$.

::: warning
That volume is why **tensor parallelism must stay within a node**. At $B=8$, $T=4096$,
$d=8192$ in bf16, each all-reduce moves 537 MB. Over NVLink at 900 GB/s that is 0.6 ms; over
InfiniBand at 25 GB/s it is 21 ms — and you pay it 128 times per step.

Crossing a node boundary with tensor parallelism turns a fast run into a communication-bound
one. Check the topology (lesson 7.01) before setting the degree.
:::

## Pipeline parallelism

Split the model by **layers**. Rank 0 holds layers 1–8, rank 1 holds 9–16, and activations
are passed point-to-point between stages.

Communication is tiny — one activation tensor per stage boundary, not per layer. The problem
is idleness.

**The bubble.** With $p$ stages processing one batch, stage 2 waits for stage 1, stage 3
waits for stage 2, and so on. Utilisation is $1/p$.

The fix is to split the batch into $m$ microbatches and pipeline them:

$$
\text{bubble fraction} = \frac{p-1}{m+p-1}
$$

```python
def bubble(stages, microbatches):
    return (stages - 1) / (microbatches + stages - 1)

for p in (4, 8):
    for m in (1, 4, 16, 64):
        print(f'{p} stages, {m} microbatches: {bubble(p, m):.1%} idle')
# 4 stages,  1 microbatch : 75.0% idle
# 4 stages, 16 microbatches: 15.8% idle
# 8 stages, 64 microbatches: 9.9% idle
```

**Rule of thumb: $m \ge 4p$.** Below that the bubble is large enough to dominate.

**Interleaved scheduling** (1F1B with virtual stages) assigns each rank several
non-contiguous layer groups, reducing the bubble further at the cost of more communication.

::: check
Tensor parallelism splits the transformer MLP column-parallel then row-parallel. Why that pairing specifically?

- [x] It gives one all-reduce per MLP block instead of two — the elementwise nonlinearity needs no communication after a column split, and the row split produces partial sums that combine in a single reduce
  > Splitting both by columns, or both by rows, would need communication between the two matrices as well. The same pattern applies to attention: split by heads, then row-split $W_O$ and all-reduce once.
- [ ] It keeps the activation memory balanced across ranks
  > Memory balance is a consequence; the communication count is the reason.
- [ ] Row-parallel matrices cannot be first in a chain
  > They can, at the cost of an extra all-reduce to assemble their input.
- [ ] It avoids materialising the $4d$ hidden activation
  > The hidden activation is materialised, sharded across ranks.
:::

## Sequence parallelism

Split the **sequence** dimension across devices. Attention needs all positions, so each rank
must gather the keys and values it does not hold.

**Ring attention** does this without materialising the full sequence anywhere: pass KV blocks
around a ring while each rank computes partial attention against them, using the same online
softmax as FlashAttention (lesson 4.12). This is what makes million-token contexts feasible.

Sequence parallelism also composes cheaply with tensor parallelism for the parts of a layer
that are elementwise — LayerNorm and dropout are replicated work under pure tensor
parallelism, and splitting them by sequence removes that redundancy along with its activation
memory.

## Combining them

Real runs use all of these at once. The assignment follows bandwidth:

```python
config = {
    'tensor_parallel':   8,     # within a node, over NVLink
    'pipeline_parallel': 4,     # across nodes, low volume
    'data_parallel':    32,     # across nodes, once per step
}
# Total GPUs = 8 * 4 * 32 = 1024
```

| Axis | Volume | Frequency | Where |
|---|---|---|---|
| Tensor | High | Per layer | Inside a node |
| Sequence | Medium | Per attention | Inside a node |
| Pipeline | Low | Per stage boundary | Across nodes |
| Data | High | Per step | Across nodes |
| Expert (4.14) | High | Per MoE layer | Across nodes, all-to-all |

Data parallelism has high volume but communicates once per step and overlaps with the
backward pass (lesson 7.05), which is why it tolerates slower links.

::: check
Pipeline parallelism splits the model by layers across devices. What is the bubble?

- [x] The idle time at the start and end of each batch while the pipeline fills and drains — with $p$ stages and $m$ microbatches it is about $(p-1)/(m+p-1)$ of the time
  > More microbatches shrink it, which is why pipeline parallelism wants large batches, and why interleaved schedules exist to shrink it further.
- [ ] The memory overhead of holding activations for in-flight microbatches
  > That overhead is real and is a separate cost from the idle time.
- [ ] The communication between adjacent stages
  > Stage-to-stage communication is small — one activation tensor — which is what makes pipelining attractive across slow links.
- [ ] The recomputation needed when a stage's activations are discarded
  > Recomputation is gradient checkpointing, an orthogonal technique.
:::

## Choosing degrees

::: key
A practical ordering:

1. **Use the largest micro-batch that fits.** Compute efficiency first.
2. **Add tensor parallelism** until the model fits, capped at the node size — 8 on most
   hardware.
3. **Add pipeline parallelism** if it still does not fit, keeping $m \ge 4p$.
4. **Fill the rest with data parallelism.**
5. **Add ZeRO/FSDP sharding** (lesson 7.07) on top of data parallelism to cut optimizer
   memory.

Tensor parallelism above the node size is almost always a mistake. Pipeline parallelism with
too few microbatches is the second most common one.
:::

::: exercise
You must train a 70B model on 64 A100-80GB GPUs. Propose a layout and justify it.
:::

::: solution
**Memory first.** From lesson 2.01, mixed-precision AdamW costs about 18 bytes per parameter:

$$
70 \times 10^9 \times 18 = 1.26\ \text{TB}
$$

Total memory available is $64 \times 80 = 5.1$ TB, so it fits in aggregate with room for
activations — but 1.26 TB cannot sit on one GPU, so the state must be split.

**Proposed layout: TP=8, PP=2, DP=4.**

- **TP=8** — a full node. Splits parameters 8-fold, so each rank holds 8.75B parameters'
  worth of weights (17.5 GB in bf16). Stays on NVLink, which the per-layer all-reduces
  require.
- **PP=2** — splits the 80 layers into two stages of 40. Halves per-rank parameter memory
  again. Communication is one activation per boundary, so crossing nodes is fine. Needs
  $m \ge 8$ microbatches; use 16 for a 6% bubble.
- **DP=4** — $8 \times 2 \times 4 = 64$. Gives an effective batch 4× the per-replica batch.
- **FSDP/ZeRO-1 across the DP dimension** — shards optimizer state 4-fold, cutting the
  dominant memory term.

**Per-rank memory check.** Parameters after TP and PP: $70/(8 \times 2) = 4.4$B, so 8.8 GB in
bf16. Gradients another 8.8 GB. Optimizer state (fp32 master, two moments = 12 bytes/param)
is $4.4\text{B} \times 12 / 4 = 13.1$ GB after ZeRO-1 sharding. Total about **31 GB**,
leaving roughly 45 GB for activations and fragmentation. Comfortable.

**Alternative worth considering: TP=8, PP=1, DP=8 with full FSDP.** Simpler — no pipeline
bubble, no microbatch tuning — and FSDP's all-gather of parameters during the forward pass
costs bandwidth that pipeline parallelism avoids. Worth benchmarking both; the answer depends
on inter-node bandwidth.

**What not to do:** TP=16 spanning two nodes. The per-layer all-reduces would cross
InfiniBand 320 times per step, and the run would be communication-bound. And PP=8 with 8
microbatches, which gives a 47% bubble.

**Validate by measuring MFU** (lesson 5.04). Below 30% means the layout is wrong, not that
the model is slow.
:::

## What to carry forward

- Column-then-row splitting gives one all-reduce per MLP block; attention splits by head.
- Tensor parallelism must stay inside a node — its volume is too high for InfiniBand.
- Pipeline bubble is $(p-1)/(m+p-1)$; keep $m \ge 4p$.
- Ring attention splits the sequence and is what enables very long context.
- Order: micro-batch, then TP to node size, then PP, then DP, then sharding.
