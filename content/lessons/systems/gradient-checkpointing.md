---
summary: Trading recomputation for activation memory — the square-root result, what to checkpoint, and the memory budget worked out end to end.
prereqs: [chain-rule-backprop, memory-hierarchy-roofline, zero-fsdp]
---

Lesson 1.08 established that every layer's backward pass needs its forward activations, and
that this — not parameters — is the memory term that scales with batch size. Gradient
checkpointing discards most of them and recomputes what is needed.

## The tradeoff

**Without checkpointing**: store every intermediate activation. Memory $O(L)$ in depth,
compute $1\times$ forward $+ 1\times$ backward.

**With checkpointing**: store activations only at chosen boundaries. Recompute the rest
during the backward pass. Memory falls; compute rises by roughly one extra forward pass.

$$
\text{overhead} \approx \frac{1 \text{ extra forward}}{1 \text{ forward} + 2 \text{ backward}} \approx 33\%
$$

::: key
33% more FLOPs sounds expensive and usually is not. From lesson 7.02, transformer training is
compute-bound on the matmuls — but the memory freed lets you **raise the batch size**, which
improves arithmetic intensity on everything else and improves hardware utilisation.

In practice, full checkpointing plus a 2–4× larger batch is frequently *faster* end to end
than no checkpointing at the smaller batch. Measure throughput in tokens per second, not
FLOPs.
:::

## The square-root result

Checkpoint every $k$-th layer in a network of depth $L$:

- Stored activations: $L/k$ checkpoints.
- Recomputation: up to $k$ layers per segment.
- Total memory: $O(L/k + k)$.

Minimised at $k = \sqrt{L}$, giving $O(\sqrt{L})$ memory. For a 100-layer network, checkpoint
every 10 layers and store 10× fewer activations for one extra forward.

## Using it

```python
import torch
from torch.utils.checkpoint import checkpoint

class Block(torch.nn.Module):
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Model(torch.nn.Module):
    def forward(self, x, use_checkpoint=True):
        for block in self.blocks:
            if use_checkpoint and self.training:
                # use_reentrant=False: the older implementation interacts badly
                # with anything needing a well-formed autograd graph.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x
```

For FSDP, use the integrated version so checkpointing and sharding compose:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
        m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
    check_fn=lambda m: isinstance(m, TransformerBlock),
)
```

## Selective checkpointing

Not all activations cost the same to store or to recompute. The useful ratio is
**memory saved per FLOP of recomputation**:

| Operation | Memory | Recompute cost | Checkpoint? |
|---|---|---|---|
| Attention scores ($T^2$) | Very high | High | Already handled by FlashAttention |
| MLP intermediate ($4d$) | High | Low — one matmul | **Yes** |
| LayerNorm output | Medium | Very low | **Yes** |
| Activation function output | Medium | Trivial | **Yes** |
| Matmul output | Medium | High | No |

::: key
The principle: **recompute cheap operations, store expensive ones.**

An elementwise activation costs almost nothing to recompute and saves $BTd$ of memory —
always recompute it. A matrix product is the expensive thing you are trying to fit, so
storing its output is usually the better trade.

PyTorch supports this directly with a policy function, which is meaningfully better than the
all-or-nothing choice:

```python
from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy

def policy(ctx, op, *args, **kwargs):
    # Save matmul outputs; recompute everything else.
    if op in {torch.ops.aten.mm.default, torch.ops.aten.bmm.default}:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

context_fn = lambda: create_selective_checkpoint_contexts(policy)
x = checkpoint(block, x, use_reentrant=False, context_fn=context_fn)
```
:::

Selective checkpointing typically recovers most of full checkpointing's memory saving at
10–15% overhead instead of 33%.

## The memory budget

Estimate before running, so you know which lever to pull:

```python
def memory_budget(n_params, batch, seq, d_model, n_layers, bytes_per=2,
                  checkpointing=False, zero_stage=3, world=8):
    # Parameters and optimizer state (lesson 2.01), sharded by ZeRO stage.
    per_param = {0: 18, 1: 2 + 4 + 12 / world,
                 2: 2 + (4 + 12) / world, 3: 18 / world}[zero_stage]
    state_gb = n_params * per_param / 1e9

    # Activations: roughly 10 saved tensors per layer without checkpointing,
    # or one per layer boundary with it.
    per_layer = 10 if not checkpointing else 1
    act_gb = per_layer * batch * seq * d_model * bytes_per * n_layers / 1e9

    print(f'state {state_gb:7.1f} GB   activations {act_gb:7.1f} GB   '
          f'total {state_gb + act_gb:7.1f} GB')

memory_budget(7e9, batch=8, seq=4096, d_model=4096, n_layers=32, checkpointing=False)
memory_budget(7e9, batch=8, seq=4096, d_model=4096, n_layers=32, checkpointing=True)
```

The two rows show the split clearly: state is fixed by the model and the sharding, while
activations scale with $B \times T$. If memory grows when you raise the batch, activations
are the term and checkpointing is the lever.

## Related techniques

- **FlashAttention** (lesson 4.12) is checkpointing applied inside attention — it recomputes
  the score matrix during the backward pass rather than storing $T^2$.
- **Activation offloading** moves activations to CPU instead of discarding them. Trades PCIe
  bandwidth for recomputation; usually worse, occasionally right when compute is saturated.
- **CPU offload for parameters** (lesson 7.07) is the same idea applied to the other memory
  term.

::: warning
Checkpointing changes what is stored, not what is computed — so results must be identical.
They are, with two exceptions worth knowing:

**Randomness.** Dropout inside a checkpointed region runs twice with different masks unless
RNG state is preserved. PyTorch handles this by default (`preserve_rng_state=True`), and
disabling it for speed makes the backward pass inconsistent with the forward.

**Non-determinism.** Recomputation may sum in a different order and give bit-different
results (lesson 2.15). Expected, and not a bug.
:::

::: exercise
You train a 7B model with a batch of 4 and sequence 2048, using 68 GB of 80 GB. You need
batch 16. What do you change, in order?
:::

::: solution
**Split the memory first.** From the budget function: with ZeRO-3 on 8 ranks, state is
$7\times10^9 \times 18/8 \approx 16$ GB. So activations are about 52 GB at batch 4 — roughly
13 GB per unit of batch.

Batch 16 would need $16 + 4 \times 52 = 224$ GB. You are 144 GB over.

**In order:**

1. **Full gradient checkpointing.** Cuts activations by roughly 10× — from 52 GB to about 5
   GB at batch 4, so about 21 GB at batch 16. Total $16 + 21 = 37$ GB. **This alone solves
   it**, at 33% more FLOPs.

2. **Selective checkpointing instead.** If full checkpointing costs too much throughput, save
   matmul outputs and recompute the rest. Expect roughly 5× activation reduction — about 42
   GB at batch 16, total 58 GB, still fitting — at 10–15% overhead rather than 33%. Try this
   second and keep it if the throughput is better.

3. **Gradient accumulation instead of a larger physical batch.** If you want batch 16 for the
   *optimisation* effect rather than for throughput, run 4 microbatches of 4 and accumulate
   (lesson 2.11). Identical gradients, no extra memory, and it works today with no other
   change. Frequently this is the right answer and people reach for checkpointing
   unnecessarily.

4. **Check you are on FlashAttention.** If attention is materialising $T^2$ scores, that is
   $4 \times 32 \times 2048^2 \times 2$ bytes per layer — a large share of the 52 GB.
   Switching to the fused kernel (lesson 4.12) may free more than checkpointing does, at no
   compute cost at all.

**Verify by measuring, not predicting:**

```python
torch.cuda.reset_peak_memory_stats()
train_step()
print(f'{torch.cuda.max_memory_allocated()/1e9:.1f} GB')
```

**And measure throughput in tokens per second**, not FLOPs. Checkpointing plus batch 16 may
well beat no checkpointing at batch 4, because the larger batch raises arithmetic intensity
on every operation in the model.
:::

## What to carry forward

- Activations, not parameters, scale with batch and sequence — that is what checkpointing attacks.
- Checkpoint every $\sqrt{L}$ layers for $O(\sqrt{L})$ memory; per transformer block in practice.
- Recompute cheap operations, store expensive ones — selective checkpointing is usually the better trade.
- 33% more FLOPs often buys a larger batch that more than pays for itself.
- Use `use_reentrant=False`, and keep RNG state preserved.
