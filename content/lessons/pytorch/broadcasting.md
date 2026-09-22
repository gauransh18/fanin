---
summary: The exact rule, the silent bugs it causes, and how to write shape code that fails loudly instead of quietly.
prereqs: [shapes-views-strides]
---

Broadcasting lets operations combine tensors of different shapes without materialising
copies. It is indispensable and it is the largest single source of silently wrong
numerical code in PyTorch.

## The rule, precisely

Align shapes **from the right**. For each dimension pair:

1. Equal sizes — fine.
2. One of them is 1 — it is stretched to match the other.
3. One tensor has no such dimension — treat it as size 1, then apply rule 2.
4. Anything else — error.

```
(3, 1, 5)          (3, 1, 5)          (3, 4, 5)
   (4, 5)   →   (1, 4, 5)   →   result (3, 4, 5)
```

```python
import torch

a = torch.randn(3, 1, 5)
b = torch.randn(   4, 5)
(a + b).shape                      # torch.Size([3, 4, 5])

torch.broadcast_shapes((3, 1, 5), (4, 5))    # check without allocating
```

Stretching is **virtual**: PyTorch sets the stride of the broadcast dimension to 0, so
every step along it reads the same element. Nothing is copied, and a broadcast operation
costs no extra memory for the inputs — only the output is materialised.

## The classic silent bug

```python
import torch

pred   = torch.randn(32)       # (32,)   one prediction per example
target = torch.randn(32, 1)    # (32, 1) a stray trailing dimension

loss = ((pred - target) ** 2).mean()
(pred - target).shape          # torch.Size([32, 32])  -- NOT (32,)
```

Aligning from the right pairs `(32,)` against `(32, 1)`: the 1 broadcasts against 32, and
the missing leading dimension broadcasts too. You get a 32×32 matrix of **every pairwise
difference**, and `.mean()` happily reduces it to a scalar.

No error. The loss is a real number. It decreases during training. The model learns
something, just not the thing you asked for. This exact shape mismatch — a target with a
trailing singleton from a DataLoader or a `.unsqueeze` — is responsible for an enormous
amount of wasted compute.

::: key
**Broadcasting turns shape bugs into wrong numbers instead of exceptions.** Assert
shapes at the boundaries of your code, especially around losses:

```python
assert pred.shape == target.shape, f'{pred.shape} vs {target.shape}'
```

One line, and the class of bug disappears.
:::

::: check
`pred` has shape `(32,)` and `target` has shape `(32, 1)`. What does `((pred - target) ** 2).mean()` compute?

- [x] The mean over a 32×32 matrix of every pairwise difference
  > A real number that is not the loss you wanted. Aligned from the right, the 1 stretches against 32 and the missing leading dimension stretches too. No exception is raised, the number decreases during training, and the model learns the wrong thing.
- [ ] The correct mean squared error; the trailing dimension is ignored
  > A trailing singleton is never ignored. It is exactly what triggers the broadcast.
- [ ] A shape error, since the tensors have different ranks
  > Different ranks are legal — the shorter one is padded with leading 1s. That permissiveness is the whole problem.
- [ ] The mean of 32 values, each squared twice
  > There is no double squaring. The count is 1,024 values, not 32.
:::

## Deliberate broadcasting

Used on purpose, it is elegant. Pairwise squared distances between two sets of points,
with no loop and no intermediate tiling:

```python
import torch

A = torch.randn(100, 3)      # 100 points in 3-D
B = torch.randn(50, 3)       # 50 points

# (100, 1, 3) against (1, 50, 3) -> (100, 50, 3), reduced over the last axis.
d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
d2.shape                     # torch.Size([100, 50])

torch.allclose(d2, torch.cdist(A, B) ** 2, atol=1e-4)   # True
```

`None` in an index is `unsqueeze` — it inserts a size-1 dimension at that position, which
is the idiomatic way to set up a broadcast.

::: warning
The example above allocates a $100\times50\times3$ intermediate. At realistic scale
that is fatal: 10,000 points against 10,000 points in 768 dimensions would need
$10^8 \times 768 \times 4$ bytes = **307 GB**. Use `torch.cdist`, or expand
$\lVert a - b\rVert^2 = \lVert a\rVert^2 - 2a\!\cdot\! b + \lVert b\rVert^2$ so the
cross term is a matrix product. Broadcasting hides allocations — always ask what the
output shape is before you write it.
:::

## Attention masking is broadcasting

The causal mask in lesson 4.06 is a broadcast, and reading it as one removes the mystery:

```python
import torch

B, h, T = 2, 8, 6
scores = torch.randn(B, h, T, T)

# One (T, T) mask broadcasts across every batch item and every head.
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask, float('-inf'))
scores.shape                 # still (2, 8, 6, 6) -- mask cost nothing

# A padding mask varies per batch item but not per head or query position.
pad = torch.tensor([[1,1,1,1,0,0], [1,1,1,1,1,1]], dtype=torch.bool)
scores = scores.masked_fill(~pad[:, None, None, :], float('-inf'))
```

The `[:, None, None, :]` places the length-$T$ mask on the **key** axis and leaves head
and query axes to broadcast. Getting those `None`s in the wrong place produces a mask
that is transposed — the model attends to the future and the loss looks suspiciously
good. Lesson 4.07 covers how to catch that.

::: check
Computing pairwise distances as `((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)` is elegant. When does it stop being usable?

- [x] Once the intermediate gets large — 10,000 × 10,000 points in 768 dimensions is 307 GB
  > Broadcasting costs nothing for the *inputs*, but the elementwise difference is materialised in full before the reduction. `torch.cdist` avoids it by never forming the three-dimensional tensor.
- [ ] Once either tensor is non-contiguous, since broadcasting requires contiguity
  > Broadcasting works on any strides; it sets the stretched dimension's stride to 0.
- [ ] Once the two point sets have different dimensionality
  > That would be a genuine shape error, and it would be raised immediately rather than being a scaling problem.
- [ ] It never stops; broadcasting allocates nothing
  > Nothing is allocated for the inputs. The output is the whole cost, and here the output of the subtraction is three-dimensional.
:::

## In-place operations refuse to broadcast the target

```python
import torch

a = torch.randn(3, 1)
b = torch.randn(3, 4)

a + b            # fine: shape (3, 4)
a += b           # RuntimeError: output with shape [3, 1] doesn't match
                 # the broadcast shape [3, 4]
```

An in-place op must write into the existing buffer, so the *result* shape has to equal
the target's shape. The operands may still broadcast into it. This asymmetry is easy to
forget when converting code to in-place for memory reasons.

::: exercise
You have per-token logits of shape $(B, T, V)$ and labels of shape $(B, T)$. You write
`F.cross_entropy(logits, labels)` and get an error. Why, and what is the fix — and what
happens if you instead reshape the labels to $(B, T, 1)$?
:::

::: solution
`F.cross_entropy` expects logits of shape $(N, C)$ with targets $(N,)$, or the
image-style $(N, C, d_1, \ldots)$ with targets $(N, d_1, \ldots)$. Your $(B, T, V)$ puts
the class dimension **last**, where it expects it second.

**The standard fix** is to flatten the batch and time axes together:

```python
loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
```

This is what every language-model training loop does. The alternative,
`logits.transpose(1, 2)` to get $(B, V, T)$, also works and avoids the reshape, but
leaves a non-contiguous tensor that the kernel may copy anyway.

**Reshaping labels to $(B, T, 1)$** does not help and makes things worse. Cross-entropy
would either reject the extra dimension or, in the hand-rolled `gather`-based
implementations people write, broadcast it — producing a loss averaged over the wrong
axes with no error raised. This is the same failure as the `(32,)` versus `(32,1)` case
above: adding a singleton dimension to satisfy a shape check is almost always the wrong
move. Make the shapes genuinely match instead.
:::

## What to carry forward

- Align from the right; size 1 stretches; missing dimensions are size 1.
- Broadcasting is free for inputs (stride 0) but the output is fully materialised.
- Assert `pred.shape == target.shape` before every loss.
- `None` in an index inserts an axis — that is how masks are positioned.
- In-place ops cannot broadcast the destination.
