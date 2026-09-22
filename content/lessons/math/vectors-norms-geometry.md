---
summary: What a vector actually is, why we measure it three different ways, and where each norm shows up later in the curriculum.
---

Everything downstream of this page — gradients, embeddings, attention scores, weight
decay — is arithmetic on vectors. This lesson fixes the vocabulary and, more
importantly, fixes the *geometric picture*, because that picture is what lets you read
a paper's equation and guess what it does before working through the algebra.

## A vector is a list with rules

A vector in $\mathbb{R}^n$ is an ordered list of $n$ real numbers:

$$
\mathbf{x} = (x_1, x_2, \ldots, x_n)
$$

The list is not the point. The point is the two operations it supports, which have to
behave sensibly together:

- **Addition**, componentwise: $(\mathbf{x} + \mathbf{y})_i = x_i + y_i$
- **Scaling** by a real number: $(c\mathbf{x})_i = c\,x_i$

Anything that supports those two operations consistently is a vector space, and every
result in this track applies to it. That abstraction is why the same mathematics
describes a 3-D point, a 768-dimensional token embedding, and a whole neural network's
parameters flattened into one long list.

::: insight
When you read "the gradient is a vector in parameter space", it means exactly this: a
model with 7 billion parameters has a gradient that is one point in $\mathbb{R}^{7\times10^9}$.
Every optimizer in track 2 is a rule for moving that single point.
:::

In code, a vector is a rank-1 tensor:

```python
import torch

x = torch.tensor([3.0, -4.0, 12.0])
y = torch.tensor([1.0,  0.0, -2.0])

x + y           # tensor([ 4., -4., 10.])
2.5 * x         # tensor([ 7.5, -10.,  30.])
x.shape         # torch.Size([3])  -- rank 1, length 3
```

## The dot product is the only product you need

The dot product of two vectors of the same length is

$$
\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{n} x_i y_i
$$

It takes two vectors and returns one number. That is unglamorous until you notice
what the number means:

$$
\mathbf{x} \cdot \mathbf{y} = \lVert \mathbf{x} \rVert \, \lVert \mathbf{y} \rVert \cos\theta
$$

where $\theta$ is the angle between them. So the dot product measures **alignment**,
scaled by both lengths. Three cases are worth memorising:

| Dot product | Angle | Meaning |
|---|---|---|
| Large positive | Near $0°$ | Pointing the same way |
| Zero | $90°$ | Orthogonal — no shared direction |
| Large negative | Near $180°$ | Pointing opposite ways |

This single fact is why attention works. In lesson 4.02 a query vector is dotted
against every key vector, and the resulting numbers are exactly "how much does this
key point the same way as my query". Nothing more sophisticated is happening.

::: check
Two token embeddings have a dot product of exactly zero. What does that tell you about them?

- [x] Neither has any component pointing along the other
  > That is what orthogonality means: the alignment between them is nil. In lesson 4.02 this is a query that finds nothing to attend to in that key.
- [ ] They are the same vector up to a sign
  > Identical vectors give a large positive dot product, and opposite ones a large negative. Zero is the case in between.
- [ ] At least one of them must be the zero vector
  > The zero vector does give zero against everything, but so does any orthogonal pair of perfectly ordinary vectors.
- [ ] They have the same length
  > Length never enters. A vector of norm 1000 can be orthogonal to one of norm 0.001.
:::

## Three ways to measure length

A **norm** $\lVert \cdot \rVert$ assigns a non-negative size to a vector. Any norm must
satisfy three conditions: it is zero only for the zero vector, it scales as
$\lVert c\mathbf{x} \rVert = |c| \lVert \mathbf{x} \rVert$, and it obeys the triangle
inequality $\lVert \mathbf{x} + \mathbf{y} \rVert \le \lVert \mathbf{x} \rVert + \lVert \mathbf{y} \rVert$.

Three satisfy those conditions and show up constantly.

### The $\ell_2$ norm — ordinary length

$$
\lVert \mathbf{x} \rVert_2 = \sqrt{\sum_i x_i^2} = \sqrt{\mathbf{x} \cdot \mathbf{x}}
$$

This is Euclidean distance from the origin. It is the default: when a paper writes
$\lVert \cdot \rVert$ with no subscript, assume $\ell_2$.

It is smooth everywhere except at the origin, which is why it is the norm of choice
whenever you need to differentiate — weight decay, gradient clipping, and the
normalization layers in lesson 3.06 all use it.

### The $\ell_1$ norm — total absolute displacement

$$
\lVert \mathbf{x} \rVert_1 = \sum_i |x_i|
$$

Sometimes called the taxicab or Manhattan norm: the distance you walk on a grid where
diagonal moves are forbidden. Its corners are the interesting part. Penalising
$\lVert \mathbf{w} \rVert_1$ during training pushes weights *exactly* to zero rather
than merely small, which is why $\ell_1$ produces sparse models and $\ell_2$ does not.

### The $\ell_\infty$ norm — the worst component

$$
\lVert \mathbf{x} \rVert_\infty = \max_i |x_i|
$$

Ignores everything except the largest entry. This is the norm used for adversarial
robustness ("no pixel changed by more than $\epsilon$") and for the per-element
bounds in quantization, lesson 5.12.

::: warning
The three norms disagree enormously in high dimensions. For a vector of $n$ ones,
$\lVert \mathbf{x} \rVert_1 = n$, $\lVert \mathbf{x} \rVert_2 = \sqrt{n}$, and
$\lVert \mathbf{x} \rVert_\infty = 1$. At $n = 4096$ that is 4096 versus 64 versus 1.
When someone reports "the gradient norm", always ask which one.
:::

```python
import torch

x = torch.tensor([3.0, -4.0, 12.0])

torch.linalg.vector_norm(x, ord=2)         # tensor(13.)   sqrt(9+16+144)
torch.linalg.vector_norm(x, ord=1)         # tensor(19.)   3+4+12
torch.linalg.vector_norm(x, ord=float('inf'))  # tensor(12.)
```

::: check
A gradient vector in $\mathbb{R}^{4096}$ has every component equal to $1$. What is its $\ell_2$ norm?

= 64
> $\sqrt{4096} = 64$. The same vector has $\ell_1$ norm $4096$ and $\ell_\infty$ norm $1$ — three answers for one gradient, which is why "the gradient norm was 64" means nothing until someone says which norm.
:::

## Unit vectors and cosine similarity

Dividing a vector by its own $\ell_2$ norm gives a **unit vector** — same direction,
length one:

$$
\hat{\mathbf{x}} = \frac{\mathbf{x}}{\lVert \mathbf{x} \rVert_2}
$$

Dot two unit vectors together and the length terms vanish, leaving just $\cos\theta$.
That is **cosine similarity**, bounded in $[-1, 1]$:

$$
\text{sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\lVert \mathbf{x} \rVert_2 \lVert \mathbf{y} \rVert_2}
$$

Every embedding-based retrieval system in lesson 5.14 ranks documents by this number.
It is preferred over raw dot product when you want to compare *direction* without
letting a long vector win purely by being long.

```python
import torch
import torch.nn.functional as F

a = torch.randn(768)
b = torch.randn(768)

F.cosine_similarity(a, b, dim=0)     # near 0.0 -- see the exercise below
F.normalize(a, dim=0).norm()         # tensor(1.0000)
```

## The part that breaks intuition

In three dimensions, two random directions are often fairly aligned. In 768
dimensions they are almost never aligned. The cosine similarity of two random
Gaussian vectors in $\mathbb{R}^n$ concentrates around zero with standard deviation
approximately $1/\sqrt{n}$ — so at $n = 768$, typical similarity is about $0.036$.

This is **concentration of measure**, and it has a practical consequence you will meet
repeatedly: a high-dimensional space has room for an enormous number of nearly
orthogonal directions. A 768-dimensional embedding can store far more than 768
distinguishable concepts, because "nearly orthogonal" is cheap. That observation is
the foundation of the superposition hypothesis in interpretability work, and it is
why increasing model width helps more than a naive dimension count suggests.

::: exercise
Two vectors are drawn with each component sampled independently from
$\mathcal{N}(0, 1)$ in $\mathbb{R}^n$. Explain why their cosine similarity
concentrates near zero as $n$ grows, and why the spread shrinks like $1/\sqrt{n}$.
:::

::: solution
The numerator is $\sum_i x_i y_i$. Each term has mean $\mathbb{E}[x_i y_i] =
\mathbb{E}[x_i]\mathbb{E}[y_i] = 0$ by independence, and variance $1$. Summing $n$
independent terms gives a numerator with mean $0$ and standard deviation $\sqrt{n}$.

The denominator is $\lVert \mathbf{x} \rVert \lVert \mathbf{y} \rVert$, and each norm
concentrates tightly around $\sqrt{n}$ (the sum of $n$ squared standard normals has
mean $n$), so the denominator is approximately $n$.

The ratio therefore has mean $0$ and standard deviation approximately
$\sqrt{n}/n = 1/\sqrt{n}$. Lesson 1.11 makes the concentration step precise.
:::

::: check
You widen a model's embeddings from 768 dimensions to 3072, changing nothing else. What happens to the typical cosine similarity between two unrelated embeddings?

- [x] It roughly halves, from about $0.036$ to about $0.018$
  > The spread scales like $1/\sqrt{n}$. Quadrupling $n$ doubles $\sqrt{n}$, so the typical similarity halves — and the space gets correspondingly roomier for nearly orthogonal concepts.
- [ ] It is unchanged, because cosine similarity is scale-free
  > Cosine similarity is invariant to each vector's *length*, not to the number of dimensions. Dimension is precisely what sets the spread.
- [ ] It roughly doubles
  > It moves the other way: more dimensions means random directions are *less* aligned, not more.
- [ ] It becomes exactly zero
  > It concentrates towards zero but never reaches it. The spread shrinks like $1/\sqrt{n}$, which is small, not nothing.
:::

## What to carry forward

- A vector is defined by addition and scaling, not by being a list of numbers.
- The dot product measures alignment; attention scores are dot products and nothing more.
- $\ell_2$ is smooth and default; $\ell_1$ has corners and makes things sparse; $\ell_\infty$ reports the worst case.
- Normalize before comparing directions, or long vectors win for the wrong reason.
- High-dimensional spaces are mostly orthogonal, and that is a feature.

Next: [matrices as the maps that move these vectors around](/learn/math/matrices-as-linear-maps/).
