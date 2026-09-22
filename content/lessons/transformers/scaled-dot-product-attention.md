---
summary: The equation derived line by line — why queries and keys are separate projections, why the scale is the square root of the head dimension, and where the memory goes.
prereqs: [seq2seq-to-attention, matrix-multiplication-cost, entropy-cross-entropy-kl]
---

This is the central equation of the curriculum:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Every term is there for a reason. This lesson derives each one.

## Three projections from one input

Each position's representation $\mathbf{x}_i$ produces three vectors:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

- **Query** — what this position is looking for.
- **Key** — what this position offers to others.
- **Value** — what it contributes when selected.

Why three rather than using $X$ directly? Because $XX^\top$ is symmetric, so position $i$
would attend to $j$ exactly as much as $j$ attends to $i$. Relationships in language are
not symmetric: a pronoun should attend strongly to its antecedent without the antecedent
attending equally back. Separate $W_Q$ and $W_K$ break that symmetry.

Separating $V$ from $K$ matters for a different reason: *what makes a position worth
retrieving* and *what you get when you retrieve it* are different questions. A token might
be findable by its syntactic role and useful for its semantic content.

## The scores

$$
S = \frac{QK^\top}{\sqrt{d_k}}, \qquad S_{ij} = \frac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{d_k}}
$$

$S_{ij}$ is the dot product of query $i$ with key $j$ — the alignment measure from lesson
1.01, and nothing more. $S$ is $T \times T$: every position scored against every position.

## Why $\sqrt{d_k}$

This is the part worth deriving properly.

Suppose $\mathbf{q}$ and $\mathbf{k}$ have independent components with mean 0
and variance 1. Then by lesson 1.11's rules,

$$
\mathbb{E}[\mathbf{q}\cdot\mathbf{k}] = 0, \qquad
\text{Var}[\mathbf{q}\cdot\mathbf{k}] = \sum_{i=1}^{d_k}\text{Var}[q_i k_i] = d_k
$$

So the raw scores have standard deviation $\sqrt{d_k}$. At $d_k = 128$ that is about 11.3,
so typical scores span roughly $\pm 23$.

Now feed that into softmax. The gap between the largest and second-largest score is on the
order of $\sqrt{d_k}$, and softmax exponentiates: $e^{11} \approx 60{,}000$. **One position
receives essentially all the weight**, the distribution becomes one-hot, and its gradient
— $p_i(\delta_{ij}-p_j)$ from lesson 3.02 — vanishes because $p(1-p) \approx 0$ at both
extremes.

Dividing by $\sqrt{d_k}$ returns the scores to unit variance, keeping the softmax in its
responsive region.

::: check
A head uses $d_k = 256$. Someone removes the $1/\sqrt{d_k}$ scaling and finds training
stalls almost immediately. What has gone wrong?

- [x] Softmax has saturated, so its gradient is near zero
  > Raw scores now have standard deviation $16$, and $e^{16}$ against $e^{0}$ puts
  > essentially all the weight on one position. At $p \approx 1$ the softmax Jacobian
  > $p_i(\delta_{ij} - p_j)$ vanishes, and nothing upstream receives a gradient.
- [ ] The dot products overflow to infinity in float32
  > Scores near $\pm 48$ are nowhere close to the float32 range. The damage is to the
  > *shape* of the distribution, not to the arithmetic.
- [ ] $Q$ and $K$ are no longer distinguishable
  > The projections are untouched by the scale factor; only the magnitude of their
  > dot products changes.
- [ ] The value matrix is now the wrong shape
  > $V$ is not involved in the scores at all, and no shape depends on the scaling.
:::

```python
import torch
import torch.nn.functional as F

for d_k in (16, 64, 256):
    q, k = torch.randn(1000, d_k), torch.randn(1000, d_k)
    raw = q @ k.T
    print(f'd_k={d_k:4d}  raw std {raw.std():6.2f}  '
          f'scaled std {(raw / d_k ** 0.5).std():.2f}  '
          f'max softmax weight {F.softmax(raw[0], -1).max():.3f} '
          f'-> {F.softmax(raw[0] / d_k ** 0.5, -1).max():.3f}')
```

::: key
Without the scale, attention saturates and gradients die — and it gets worse as head
dimension grows, so the problem appears exactly when you scale the model up. This is also
why **QK-Norm** exists: over a long training run, learned $W_Q$ and $W_K$ can grow, making
the scores large again even with the $\sqrt{d_k}$ division. Normalising queries and keys
before the dot product bounds the logits permanently, and it has become standard for large
runs.
:::

## Softmax and the weighted average

$$
A = \text{softmax}(S), \qquad A_{ij} \ge 0, \qquad \sum_j A_{ij} = 1
$$

Row $i$ is a probability distribution over which positions to read. Then

$$
Z = AV, \qquad \mathbf{z}_i = \sum_j A_{ij}\mathbf{v}_j
$$

Each output is a convex combination of value vectors. That convexity matters: the output
lives in the convex hull of the values, so attention **cannot** produce a representation
outside the span of what the positions offer. All the transformation happens in the
projections and the MLP; attention only routes.

## The implementation

```python
import torch
import torch.nn.functional as F

def attention(q, k, v, mask=None):
    """q, k, v: (..., T, d_k). mask: (..., T, T) boolean, True = attend."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5          # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights
```

`-inf` before the softmax, not zero after it. Setting weights to zero after softmax leaves
the remaining weights un-normalised; `-inf` exponentiates to exactly 0 and the
normalisation accounts for it.

::: warning
If an entire row is masked — a fully padded query position — every score is `-inf`,
softmax produces $0/0$, and the result is `nan` that propagates through the whole batch.
Guard it:

```python
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
```

Using the dtype's minimum instead of `-inf` keeps the softmax finite: a uniform
distribution over the row, whose output is then discarded anyway.
:::

::: check
Why are $Q$ and $K$ separate projections rather than using $X$ directly?

- [x] $XX^\top$ is symmetric
  > Position $i$ would attend to $j$ exactly as much as $j$ attends to $i$. Relationships in language are not symmetric — a pronoun should attend strongly to its antecedent without the antecedent attending equally back. Separate $W_Q$ and $W_K$ break that symmetry.
- [ ] Using $X$ directly would make the scores too large for softmax
  > Magnitude is what the $1/\sqrt{d_k}$ scaling handles, and it would apply either way.
- [ ] The projections reduce dimension, making attention cheaper
  > In a single head $W_Q$ and $W_K$ are square; the dimension reduction belongs to the multi-head split.
- [ ] Without them the gradient with respect to $X$ would be undefined
  > $XX^\top$ is perfectly differentiable.
:::

## The cost

| Operation | FLOPs | Memory |
|---|---|---|
| $QK^\top$ | $2T^2 d_k$ | $T^2$ scores |
| softmax | $O(T^2)$ | $T^2$ weights |
| $AV$ | $2T^2 d_k$ | $T d_k$ output |

The $T^2$ **memory** is the binding constraint, not the FLOPs. For a batch of 8, 32 heads
and $T = 8192$ in bf16:

$$
8 \times 32 \times 8192^2 \times 2 \text{ bytes} = 34\ \text{GB}
$$

for one layer's attention matrix. This is the problem FlashAttention (lesson 4.12) solves —
not by reducing the arithmetic, but by never materialising $S$ at all.

## Use the fused kernel

```python
import torch.nn.functional as F

# Dispatches to FlashAttention or a memory-efficient kernel automatically.
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

In production, always call this rather than the four-line version above. Write the explicit
form to understand it; ship the fused one.

::: exercise
What would go wrong if $W_Q = W_K$, so queries and keys were the same projection?
:::

::: solution
The score matrix becomes $S = XW(XW)^\top/\sqrt{d_k}$, which is **symmetric**: $S_{ij} =
S_{ji}$ for all pairs. Three consequences.

**1. Attention becomes mutual.** If "it" attends strongly to "the cat", then "the cat"
attends equally strongly to "it". Linguistic relationships are directional — a pronoun
needs its antecedent far more than the antecedent needs the pronoun — and the model can no
longer express that.

**2. Self-attention dominates.** $S_{ii} = \lVert W^\top\mathbf{x}_i\rVert^2 \ge 0$ and is
typically the largest entry in its row, by Cauchy–Schwarz: a vector's dot product with
itself exceeds its dot product with anything of equal norm. So every position attends
mostly to itself, and attention degenerates toward the identity.

**3. The rank is halved in effect.** $S$ is symmetric positive semi-definite, so its
eigenvalues are real and non-negative. The space of attainable score matrices is
substantially smaller than the full $T\times T$ space an asymmetric pair can reach.

**The caveat worth knowing:** shared QK projections are not *useless*. Some efficient
architectures tie them deliberately to halve parameters, accepting the symmetry cost, and
they work acceptably. But the causal masking of lesson 4.06 already breaks symmetry
structurally in a decoder — position $i$ can see $j < i$ but not the reverse — which
partially compensates. In an encoder with bidirectional attention, tying them hurts much
more.
:::

## What to carry forward

- Q, K, V are three projections because relationships are asymmetric and retrievability differs from content.
- Scores have variance $d_k$; dividing by $\sqrt{d_k}$ keeps softmax out of saturation.
- Mask with the dtype minimum before softmax, never zero the weights after.
- The output is a convex combination of values — attention routes, it does not transform.
- $T^2$ memory, not FLOPs, is what limits context length.
