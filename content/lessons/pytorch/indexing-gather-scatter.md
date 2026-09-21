---
summary: Basic indexing returns views, advanced indexing copies, and gather/scatter are the primitives behind embeddings, top-k and mixture routing.
prereqs: [shapes-views-strides, broadcasting]
---

PyTorch has two indexing systems with different semantics and different costs. Knowing
which one you triggered explains both your memory usage and whether your writes stick.

## Basic indexing returns views

Integers and slices produce a **view** — new metadata, same buffer, $O(1)$.

```python
import torch

x = torch.arange(24).reshape(4, 6)

x[1]            # a view: shape (6,)
x[:, 2:5]       # a view: shape (4, 3)
x[::2]          # a view with stride 2 on dim 0

v = x[1]
v[0] = 999
x[1, 0]         # tensor(999) -- the write went through
```

## Advanced indexing copies

Index with a tensor or a list and you get a **copy**, because the elements you asked for
need not be evenly spaced.

```python
import torch

x = torch.arange(24).reshape(4, 6)

y = x[[0, 2, 3]]           # copy: shape (3, 6)
y[0, 0] = 999
x[0, 0]                    # tensor(0) -- unchanged, y is independent

mask = x > 10
x[mask].shape              # boolean masking: always a 1-D copy
```

::: warning
Because advanced indexing copies, `x[idx] += 1` does **not** update `x` in the way you
might expect when `idx` contains duplicates: the read produces a copy, the increment
happens there, and the write-back keeps only the last value per position. Use
`x.index_add_(0, idx, values)` when duplicates must accumulate. This bites in sparse
gradient accumulation and in scatter-based routing.
:::

## gather: read along one axis

`gather(dim, index)` picks elements along `dim`, with the index tensor supplying a
coordinate for every output position. Output shape equals index shape.

$$
\text{out}[i][j][k] = \text{input}[\,\text{index}[i][j][k]\,][j][k] \quad \text{for dim}=0
$$

```python
import torch

logits = torch.randn(4, 10)                 # 4 examples, 10 classes
labels = torch.tensor([3, 7, 1, 9])

# The logit of each example's true class.
picked = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
picked.shape                                 # torch.Size([4])

# Equivalent, and clearer for this particular case:
torch.allclose(picked, logits[torch.arange(4), labels])   # True
```

The `unsqueeze(1)` is required because the index must have the same number of dimensions
as the input. This is the most common `gather` stumbling block.

Gather is how cross-entropy is implemented: compute log-softmax over the vocabulary, then
gather the true token's entry.

```python
import torch
import torch.nn.functional as F

logits = torch.randn(4, 10)
labels = torch.tensor([3, 7, 1, 9])

manual = -F.log_softmax(logits, dim=1).gather(1, labels[:, None]).mean()
torch.allclose(manual, F.cross_entropy(logits, labels))   # True
```

## scatter: write along one axis

`scatter_(dim, index, src)` is gather's inverse — it writes `src` into the positions
named by `index`.

```python
import torch

# One-hot encoding, without a lookup table.
labels = torch.tensor([3, 7, 1, 9])
onehot = torch.zeros(4, 10)
onehot.scatter_(1, labels.unsqueeze(1), 1.0)
onehot[0]        # tensor([0,0,0,1,0,0,0,0,0,0])

# scatter_add_ accumulates instead of overwriting -- this is the one you
# want whenever indices can repeat.
counts = torch.zeros(5)
idx    = torch.tensor([0, 2, 2, 4, 2])
counts.scatter_add_(0, idx, torch.ones(5))
counts           # tensor([1., 0., 3., 0., 1.])
```

::: key
`scatter_` with duplicate indices is **non-deterministic**: which duplicate wins is
unspecified and can differ between runs on GPU. `scatter_add_` is well-defined in value
but its floating-point summation order is also non-deterministic, so results can differ
in the last bits. Lesson 2.15 covers how to force determinism when you need it.
:::

## Embeddings are index_select

A token embedding lookup is exactly a row gather, and `nn.Embedding` is a thin wrapper
over it:

```python
import torch
import torch.nn as nn

vocab, d = 1000, 64
emb = nn.Embedding(vocab, d)
tokens = torch.tensor([[5, 92, 3], [17, 5, 0]])      # (B, T)

out = emb(tokens)
out.shape                                            # (2, 3, 64)

torch.allclose(out, emb.weight[tokens])              # True -- same operation
```

The backward pass of an embedding is a `scatter_add_`: each occurrence of token $i$ adds
its gradient to row $i$. A token appearing five times in a batch accumulates five
gradients, which is why frequent tokens' embeddings move faster — and why
`nn.Embedding(..., sparse=True)` exists for very large vocabularies.

## Top-k routing, end to end

Mixture-of-experts routing (lesson 4.14) is gather and scatter in sequence, and it is
worth seeing the whole shape dance once:

```python
import torch
import torch.nn.functional as F

B_T, n_exp, k = 8, 4, 2               # 8 tokens, 4 experts, top-2 routing
router_logits = torch.randn(B_T, n_exp)

weights, experts = torch.topk(router_logits, k, dim=-1)   # (8, 2) each
weights = F.softmax(weights, dim=-1)                      # renormalise over k

# Which tokens go to expert 0?
for e in range(n_exp):
    hits = (experts == e).nonzero(as_tuple=True)          # (token_idx, slot_idx)
    print(f'expert {e}: {hits[0].tolist()}')

# Load per expert, for the balancing loss.
load = torch.zeros(n_exp).scatter_add_(
    0, experts.reshape(-1), weights.reshape(-1))
print('load', load)
```

::: exercise
You want each sequence's **last non-padding token** representation from a tensor of shape
$(B, T, d)$, given `lengths` of shape $(B,)$. Write it with `gather`, and say why a
Python loop is wrong.
:::

::: solution
```python
import torch

B, T, d = 4, 10, 64
h = torch.randn(B, T, d)
lengths = torch.tensor([7, 3, 10, 5])

idx = (lengths - 1)                       # (B,) -- last valid position
idx = idx[:, None, None].expand(B, 1, d)  # (B, 1, d) to match h's rank
last = h.gather(1, idx).squeeze(1)        # (B, d)

# Check against the obvious version.
torch.allclose(last, torch.stack([h[b, lengths[b]-1] for b in range(B)]))  # True
```

The index must have the same rank as the input, so `[:, None, None]` adds the missing
axes and `expand` broadcasts across the feature dimension at stride 0 — no copy.

**Why not a loop.** The loop launches $B$ separate kernels and forces host–device
synchronisation on every `lengths[b]` read, which serialises the GPU. At $B = 512$ that
is 512 stalls per step. The `gather` version is one kernel and no sync. The loop is also
not traceable by `torch.compile` or `torch.export`, since the trip count depends on a
runtime value — lesson 2.14 returns to that.
:::

## What to carry forward

- Slices and integers give views; tensor and boolean indices give copies.
- `gather(dim, index)` output has the index's shape; the index needs the input's rank.
- `scatter_add_` accumulates and is what you want when indices repeat.
- Embedding forward is a gather; its backward is a scatter-add.
- Replace per-example Python loops with gather — they serialise the GPU otherwise.
