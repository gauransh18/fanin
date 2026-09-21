---
summary: Why one attention pattern is not enough, the shape choreography that splits and rejoins heads, and what different heads actually learn.
prereqs: [scaled-dot-product-attention, shapes-views-strides]
---

A single attention head produces one distribution over positions per query. But a token
often needs several different things at once — its syntactic head, its coreferent, the
topic of the paragraph. Multi-head attention runs several attention operations in parallel
and concatenates the results.

## The construction

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W_O,
\qquad
\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)
$$

The crucial detail: each head operates in dimension $d_h = d/h$, not $d$. With $d = 512$
and $h = 8$, each head works in 64 dimensions. **Total compute is unchanged** — you are
partitioning the representation, not multiplying the work.

## The implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.d_h = n_heads, d // n_heads
        # One projection producing Q, K and V together: a single larger
        # matmul beats three smaller ones (lesson 1.03).
        self.qkv  = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x, is_causal=True):
        B, T, d = x.shape

        qkv = self.qkv(x)                                  # (B, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split the LAST dimension, then move heads to a batch axis.
        # The order matters -- lesson 2.02's exercise covers why.
        q = q.view(B, T, self.h, self.d_h).transpose(1, 2)  # (B, h, T, d_h)
        k = k.view(B, T, self.h, self.d_h).transpose(1, 2)
        v = v.view(B, T, self.h, self.d_h).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0)

        # Back to (B, T, d). transpose leaves it non-contiguous, so view
        # would fail -- hence the explicit contiguous().
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.proj(out)
```

Two details that are easy to get wrong and hard to notice:

**The fused `qkv` projection.** Three separate `nn.Linear(d, d)` calls do the same
arithmetic in three kernels; one `nn.Linear(d, 3d)` does it in one, with better arithmetic
intensity. Every production implementation fuses them.

**`dropout_p` only in training.** `F.scaled_dot_product_attention` does not read
`self.training`. Forgetting the conditional means dropout stays active at inference, which
makes generation stochastic in a way that is genuinely confusing to debug.

## Why not one big head?

With $h=1$ and $d_k = d$, you have a single distribution over positions. Multi-head gives
three things a single head cannot.

**Multiple relationships at once.** Head 1 attends to the previous token, head 2 to the
subject of the sentence, head 3 to the matching bracket. A single softmax must choose one
distribution, so these compete for the same weight budget.

**Rank.** From lesson 1.04, the effective map $W_Q W_K^\top$ has rank at most $d_h$. This
sounds like a limitation of multi-head, and in practice the restriction acts as a useful
constraint — each head is forced to find a low-dimensional relationship rather than an
unstructured one.

**Averaging reduces the damage from a bad head.** The output is a sum over heads, so one
head attending to noise contributes noise at $1/h$ scale rather than dominating.

::: insight
Multi-head attention factorises "find relevant positions" into $h$ independent
subproblems. It is the same idea as the grouped convolutions of lesson 3.09 — partition
the channels, process each group independently, then mix — and it works for the same
reason: structured sparsity in the weight matrix is cheaper than a dense one and often
better.
:::

## What heads learn

Studies of trained transformers find recurring, nameable head types:

- **Previous-token heads** attend to position $i-1$. Nearly every model has several.
- **Positional heads** attend at a fixed offset, or to the first token.
- **Syntactic heads** attend from a word to its syntactic head — direct object to verb,
  adjective to noun.
- **Induction heads** are the interesting ones. They implement "if the sequence contained
  `A B` earlier and we just saw `A`, attend to that `B`". Two heads in adjacent layers
  compose to do this, and their emergence during training coincides with a sharp jump in
  in-context learning ability.
- **Attention sinks.** Many heads dump most of their weight on the first token, which
  functions as a no-op — softmax must sum to 1, so a head with nothing to retrieve needs
  somewhere to put its mass.

::: warning
The sink behaviour has a practical consequence. Naive KV-cache eviction that drops the
oldest tokens (lesson 4.10) removes the sink, and model quality collapses — not because
those tokens carried information, but because heads lost their null option and were forced
to attend somewhere meaningful. Keeping the first few tokens permanently in the cache
fixes it, which is what StreamingLLM does.
:::

## Head count and dimension

| Model | $d$ | heads | $d_h$ |
|---|---|---|---|
| BERT-base | 768 | 12 | 64 |
| GPT-2 medium | 1024 | 16 | 64 |
| Llama-2 7B | 4096 | 32 | 128 |
| Llama-3 70B | 8192 | 64 | 128 |

Note that $d_h$ is essentially always 64 or 128. This is not a tuned hyperparameter so
much as a hardware constraint: those sizes match tensor-core tile dimensions, and heads
much smaller than 64 leave the matrix units underfilled.

Empirically, head count matters less than you would expect. Many heads can be pruned after
training with negligible quality loss — which suggests the redundancy is useful during
*training* rather than at inference, much like the ensemble reading of residual networks
in lesson 3.08.

::: exercise
You have $d = 512$. Compare $h = 8$ (with $d_h = 64$) against $h = 64$ (with $d_h = 8$) on
parameter count, FLOPs, and expressiveness.
:::

::: solution
**Parameters: identical.** $W_Q, W_K, W_V$ are each $d \times d$ regardless of how the
output is partitioned — $h \cdot d \cdot d_h = h \cdot d \cdot (d/h) = d^2$. The splitting
into heads is a `view`, not a different set of weights. $W_O$ is $d\times d$ either way.
Both configurations have $4d^2 = 1{,}048{,}576$ parameters.

**FLOPs: nearly identical.** Projections are the same. The attention itself is
$h \cdot 2T^2 d_h = 2T^2 d$ in both cases. The only real difference is overhead: 64 small
$8$-dimensional matrix products have far worse arithmetic intensity (lesson 1.03) than 8
of dimension 64, so $h=64$ is **slower in wall-clock** despite the same FLOP count.

**Expressiveness — a genuine tradeoff.**

With $h=64$, $d_h=8$: each head's $W_QW_K^\top$ has rank at most 8, so each attention
pattern is extremely constrained. More patterns, each much weaker. And the scores now have
variance 8 rather than 64, so they are less discriminative — more diffuse softmax
distributions.

With $h=8$, $d_h=64$: fewer patterns, each able to express a rank-64 relationship.

**Which is better:** $h=8$, for two reasons beyond the above. First, $d_h=8$ is well below
the tensor-core tile size, so the hardware runs at a fraction of peak. Second, empirical
work on head pruning suggests models do not use anywhere near 64 distinct relationship
types per layer — the extra heads would be redundant while each is individually weaker.

**The rule of thumb:** fix $d_h$ at 64 or 128 and derive $h = d/d_h$. That is what every
model in the table above does.
:::

## What to carry forward

- $h$ heads at $d_h = d/h$ each: same parameters, same FLOPs, more relationships.
- Fuse Q, K, V into one projection; gate attention dropout on `self.training`.
- Split the last dimension, then transpose — the other order scrambles the data.
- Induction heads implement in-context copying and their emergence is visible in the loss.
- Attention sinks are a no-op mechanism; evicting them from a KV cache breaks the model.
