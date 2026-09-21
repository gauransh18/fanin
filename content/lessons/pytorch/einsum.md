---
summary: One notation that replaces reshape, transpose, matmul and sum — and makes tensor code readable a year later.
prereqs: [shapes-views-strides, matrix-multiplication-cost]
---

`einsum` lets you name every axis and state the operation you want, instead of
choreographing reshapes to coerce tensors into `matmul`'s expected layout. For anything
beyond a plain matrix product, it is both clearer and harder to get silently wrong.

## The convention

Write the input subscripts, an arrow, and the output subscripts. Two rules do everything:

1. An index appearing in the inputs but **not** the output is **summed over**.
2. An index appearing in multiple inputs is **multiplied** elementwise across them.

```python
import torch

A = torch.randn(3, 4)
B = torch.randn(4, 5)

torch.einsum('ij,jk->ik', A, B)      # matrix product: j summed away
torch.einsum('ij->ji', A)            # transpose
torch.einsum('ii->i', torch.randn(4, 4))   # diagonal
torch.einsum('ii->',  torch.randn(4, 4))   # trace
torch.einsum('i,j->ij', torch.randn(3), torch.randn(4))   # outer product
torch.einsum('ij->', A)              # sum everything
```

The matrix product reads directly: "$i$ and $k$ survive, $j$ is contracted". That is the
definition $C_{ik} = \sum_j A_{ij}B_{jk}$ transcribed.

## Why it is worth the habit

Compare batched attention scores written both ways:

```python
import torch

B, h, T, d = 2, 8, 16, 64
q = torch.randn(B, h, T, d)
k = torch.randn(B, h, T, d)

# With matmul: you must know that matmul batches over leading dims, and
# you must transpose the right pair of axes.
scores_a = q @ k.transpose(-2, -1) / d ** 0.5

# With einsum: the contraction is stated, not implied.
scores_b = torch.einsum('bhqd,bhkd->bhqk', q, k) / d ** 0.5

torch.allclose(scores_a, scores_b, atol=1e-5)    # True
```

The second version names the query and key position axes separately (`q` and `k`), so
the output `bhqk` is unambiguous about which axis is which. Six months later, that is the
difference between reading the line and re-deriving it.

::: insight
The most valuable property of `einsum` is that **it cannot silently transpose your
data**. A wrong subscript produces either a shape error or a visibly wrong output shape.
A wrong `transpose(-2, -1)` produces the right shape and wrong values — the failure mode
from lesson 2.02's exercise.
:::

## Ellipsis for leading dimensions

`...` stands for any number of leading axes, broadcast as usual. It lets you write an
operation once and apply it at any rank.

```python
import torch

def attention_scores(q, k):
    """Works for (T, d), (B, T, d), or (B, h, T, d) alike."""
    return torch.einsum('...qd,...kd->...qk', q, k) / q.shape[-1] ** 0.5

attention_scores(torch.randn(16, 64)).shape          # (16, 16)
attention_scores(torch.randn(2, 8, 16, 64)).shape    # (2, 8, 16, 16)
```

## Contractions that are awkward otherwise

**Multi-head projection in one step**, without splitting heads by hand:

```python
import torch

B, T, d, h, dh = 2, 16, 512, 8, 64
x  = torch.randn(B, T, d)
Wq = torch.randn(h, d, dh)                      # a weight per head

q = torch.einsum('btd,hdk->bhtk', x, Wq)
q.shape                                         # (2, 8, 16, 64)
```

**Weighted sum over experts**, the core of MoE dispatch:

```python
import torch

tokens  = torch.randn(32, 512)        # (N, d)
experts = torch.randn(4, 512, 512)    # (E, d, d)
weights = torch.rand(32, 4)           # (N, E) routing weights

# d is the input feature axis, D the output one; e is contracted twice --
# once against the expert weights, once against the routing weights.
out = torch.einsum('nd,edD,ne->nD', tokens, experts, weights)
out.shape                             # (32, 512)
```

**Bilinear forms and attention biases**, where three tensors contract at once:

```python
import torch

q = torch.randn(2, 16, 64)
W = torch.randn(64, 64)
k = torch.randn(2, 16, 64)

torch.einsum('bqd,de,bke->bqk', q, W, k).shape    # (2, 16, 16)
```

## The cost trap

`einsum` does not automatically choose the cheapest contraction order for three or more
operands. With two operands it lowers to a `matmul` and is as fast as anything; with
three it may materialise a large intermediate — exactly the bracketing problem from
lesson 1.03.

::: warning
```python
# Contracts to a (b, q, k) intermediate first: could be enormous.
torch.einsum('bqd,de,bke->bqk', q, W, k)

# Explicitly cheaper: project q through W first, then one matmul.
torch.einsum('bqd,de->bqe', q, W)  # then  ... @ k.transpose(-2, -1)
```

For performance-critical multi-operand contractions, use `opt_einsum` (which searches
for an optimal order) or bracket it yourself. Benchmark before assuming `einsum` is
free — lesson 2.13 shows how.
:::

## einops, when you want more

The `einops` library extends the same idea to rearrangement and reduction with named
axis sizes:

```python
# pip install einops
from einops import rearrange, reduce

# The multi-head split from lesson 2.02, stated rather than choreographed.
q = rearrange(x, 'b t (h d) -> b h t d', h=8)
out = rearrange(attn_out, 'b h t d -> b t (h d)')

# Mean-pool over the sequence.
pooled = reduce(x, 'b t d -> b d', 'mean')
```

The parenthesised group `(h d)` states that the last axis factorises — which is precisely
the fact that made `view(B, T, h, dh)` correct and `view(B, h, T, dh)` wrong. Making it
explicit removes the ambiguity entirely.

::: exercise
Write, using `einsum`, the per-head attention output $\text{softmax}(QK^\top/\sqrt{d})V$
for $Q, K, V$ of shape $(B, h, T, d)$, and explain each subscript.
:::

::: solution
```python
import torch
import torch.nn.functional as F

def attention(q, k, v):
    d = q.shape[-1]
    scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / d ** 0.5
    attn   = F.softmax(scores, dim=-1)
    return torch.einsum('bhqk,bhkd->bhqd', attn, v)
```

**First contraction, `bhqd,bhkd->bhqk`.** `b` and `h` appear in both inputs and the
output, so they are batch axes — carried through untouched. `d` appears in both inputs
and not the output, so it is summed: that is the dot product between each query and each
key. `q` and `k` each appear in one input and in the output, so they become the two axes
of the score matrix. Naming them separately is what makes the direction unambiguous.

**Second contraction, `bhqk,bhkd->bhqd`.** Now `k` is the contracted index — summing over
key positions, weighted by the attention probabilities. `q` survives from the attention
matrix and `d` survives from the values, giving one $d$-dimensional output vector per
query position.

Reading them together: the first contracts the feature axis to compare positions; the
second contracts the position axis to mix features. That is the whole of attention, and
the subscripts say so.
:::

## What to carry forward

- Indices missing from the output are summed; shared indices are multiplied.
- `einsum` cannot silently transpose you — a wrong subscript errors or changes the shape.
- `...` handles arbitrary leading dimensions.
- Three-operand contractions may pick a bad order; bracket them yourself when it matters.
