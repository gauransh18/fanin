---
summary: A matrix is not a grid of numbers — it is a function that moves vectors. Reading it that way makes every layer in a neural network legible.
prereqs: [vectors-norms-geometry]
---

The grid-of-numbers view of a matrix is where most people stop, and it makes deep
learning look like bookkeeping. The useful view is that a matrix **is a function**: it
eats a vector and returns a vector, and it is the only kind of function that respects
addition and scaling. Once you hold that view, a linear layer stops being `W @ x + b`
and becomes "rotate, stretch, and project into a new space".

## The defining property

A function $f$ is **linear** if, for all vectors and all scalars,

$$
f(\mathbf{x} + \mathbf{y}) = f(\mathbf{x}) + f(\mathbf{y})
\qquad\text{and}\qquad
f(c\mathbf{x}) = c\,f(\mathbf{x})
$$

That is a severe restriction, and it buys something enormous: a linear map is
completely determined by what it does to the basis vectors. If you know where
$\mathbf{e}_1 = (1,0,\ldots)$ goes, and where $\mathbf{e}_2 = (0,1,\ldots)$ goes, and
so on, you know where *every* vector goes, because every vector is a combination of
basis vectors and $f$ commutes with combinations.

::: insight
**A matrix is just a table of where the basis vectors land.** Column $j$ of $A$ is
$A\mathbf{e}_j$ — the image of the $j$-th basis vector. That is the entire content of
matrix notation. Everything else is consequence.
:::

So for $A \in \mathbb{R}^{m \times n}$ acting on $\mathbf{x} \in \mathbb{R}^n$:

$$
A\mathbf{x} = \sum_{j=1}^{n} x_j \, \mathbf{a}_j
$$

where $\mathbf{a}_j$ is the $j$-th column. Matrix–vector multiplication is a **weighted
sum of the columns**, with the vector supplying the weights. Read it this way once and
you will never mis-remember which dimension has to match.

```python
import torch

A = torch.tensor([[2.0, -1.0],
                  [0.0,  3.0],
                  [1.0,  1.0]])      # 3x2: maps R^2 -> R^3
x = torch.tensor([4.0, 5.0])

A @ x                                # tensor([3., 15., 9.])
4.0 * A[:, 0] + 5.0 * A[:, 1]        # tensor([3., 15., 9.])  -- same thing
```

## Shapes tell you the map

An $m \times n$ matrix maps $\mathbb{R}^n \to \mathbb{R}^m$. Note the reversal: the
*second* dimension is the input. This trips people up forever unless you tie it to the
column picture — there are $n$ columns because there are $n$ input basis vectors, and
each column lives in $\mathbb{R}^m$ because that is where outputs live.

| Shape | Map | What it usually is |
|---|---|---|
| $n \times n$ | $\mathbb{R}^n \to \mathbb{R}^n$ | A change of basis, a rotation, an attention mixing matrix |
| $m \times n$, $m < n$ | Squashes down | A projection, a bottleneck, a pooling layer |
| $m \times n$, $m > n$ | Lifts up | An expansion — the up-projection in an MLP block |

A transformer's feed-forward block is exactly this pattern: a $4d \times d$ lift, a
nonlinearity, then a $d \times 4d$ squash back. Lesson 4.05 uses nothing more than
this.

::: check
A layer holds a weight $W \in \mathbb{R}^{512 \times 768}$. Which is true?

- [x] It maps $\mathbb{R}^{768} \to \mathbb{R}^{512}$, and $W$ has 768 columns
  > Input dimension is second. There are 768 columns because there are 768 input basis vectors, and each column lives in $\mathbb{R}^{512}$ because that is where outputs go.
- [ ] It maps $\mathbb{R}^{512} \to \mathbb{R}^{768}$, lifting into a wider space
  > The reversal is exactly the trap. The second dimension is the input, so this squashes rather than lifts.
- [ ] It maps $\mathbb{R}^{768} \to \mathbb{R}^{512}$, and $W$ has 512 columns
  > The direction is right but the count is not: a column per input basis vector means 768 of them.
- [ ] The direction depends on whether you write $W\mathbf{x}$ or $\mathbf{x}W$
  > Only one of those is even defined for these shapes. $W\mathbf{x}$ needs $\mathbf{x}$ to have 768 entries; $\mathbf{x}W$ would need 512.
:::

## The four transformations worth recognising

**Scaling.** A diagonal matrix stretches each axis independently. Diagonal matrices
are cheap — $O(n)$ instead of $O(n^2)$ — which is why so much of modern architecture
design is about replacing dense matrices with diagonal ones plus structure.

$$
D = \begin{bmatrix} 2 & 0 \\ 0 & \tfrac{1}{2} \end{bmatrix}
$$

**Rotation.** Preserves all lengths and all angles. In two dimensions,

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

Rotary position embeddings, lesson 4.04, are literally this matrix applied to pairs of
coordinates in a query vector, with $\theta$ depending on the token's position.

**Projection.** Collapses onto a subspace and throws the rest away. A projection
satisfies $P^2 = P$: projecting twice is the same as projecting once, because after
the first application you are already in the target subspace.

**Shear.** Slides one axis proportionally to another. Rare on its own, but every
general matrix decomposes into rotations and scalings — that decomposition is the SVD
in lesson 1.06.

## Composition is multiplication

Apply $B$ then $A$ and you get another linear map, whose matrix is the product $AB$:

$$
A(B\mathbf{x}) = (AB)\mathbf{x}
$$

Two consequences follow immediately, and both matter in practice.

First, **order matters**: $AB \neq BA$ in general, because "rotate then stretch" is a
different operation from "stretch then rotate". Second, **stacking linear layers with
no nonlinearity between them is pointless** — $W_2(W_1\mathbf{x}) = (W_2W_1)\mathbf{x}$
is a single linear map, so a ten-layer linear network has exactly the representational
power of one layer. This is precisely why activation functions exist, and lesson 3.02
picks that thread up.

::: warning
There is one important exception to "stacking is pointless": a low-rank factorisation
$W \approx BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$
and $r \ll d$ *is* a single linear map, but it has far fewer parameters. LoRA, lesson
5.06, is built entirely on this observation.
:::

::: check
A colleague stacks ten `nn.Linear` layers with no activation between them and reports that the model underfits badly. What is the underlying reason?

- [x] The ten layers compose into a single linear map, so the model has the power of one layer
  > $W_{10}(\cdots W_1\mathbf{x}) = (W_{10}\cdots W_1)\mathbf{x}$. Depth without a nonlinearity buys nothing at all in representational terms — this is the entire reason activation functions exist.
- [ ] Ten layers is too few to fit anything interesting
  > Depth is not the issue. Ten *nonlinear* layers would be a perfectly capable network.
- [ ] The gradients vanish through ten multiplications
  > Vanishing gradients would make it hard to *train* a deep net, but here there is nothing to train towards: the function class itself is only linear maps.
- [ ] Matrix multiplication is not associative, so the layers interfere
  > It is associative, and that is exactly why they collapse. If it were not, they would not fold into one matrix.
:::

## The transpose, and what it actually means

$A^\top$ swaps rows and columns, so $(A^\top)_{ij} = A_{ji}$. Mechanically trivial;
conceptually it is the map that satisfies

$$
(A\mathbf{x}) \cdot \mathbf{y} = \mathbf{x} \cdot (A^\top \mathbf{y})
$$

for every $\mathbf{x}$ and $\mathbf{y}$. The transpose is how you move a matrix to the
other side of a dot product. That identity is the whole reason backpropagation works
the way it does: the backward pass of $\mathbf{y} = W\mathbf{x}$ multiplies the
incoming gradient by $W^\top$, and lesson 1.08 derives exactly that.

```python
import torch

W = torch.randn(4, 3)
x = torch.randn(3)
y = torch.randn(4)

lhs = (W @ x) @ y        # scalar
rhs = x @ (W.T @ y)      # same scalar
torch.allclose(lhs, rhs)  # True
```

::: exercise
A linear layer has weight $W \in \mathbb{R}^{512 \times 768}$. Which direction does it
map, what does each column of $W$ represent, and how many parameters does the layer
have if it also has a bias?
:::

::: solution
It maps $\mathbb{R}^{768} \to \mathbb{R}^{512}$ — a squash, cutting dimension by a
third. $W$ has 768 columns because the input is 768-dimensional; column $j$ is the
768-dimensional input basis vector $\mathbf{e}_j$'s image, a vector in
$\mathbb{R}^{512}$.

Parameters: $512 \times 768 = 393{,}216$ in the weight, plus 512 in the bias, giving
$393{,}728$. Note that the bias makes the layer *affine*, not linear — $f(\mathbf{0})
\neq \mathbf{0}$ — which is why frameworks call it `nn.Linear` rather than
`nn.LinearMap`.
:::

::: check
In backpropagation through $\mathbf{y} = W\mathbf{x}$, the incoming gradient is multiplied by $W^\top$. Which identity is that a consequence of?

- [x] $(A\mathbf{x}) \cdot \mathbf{y} = \mathbf{x} \cdot (A^\top\mathbf{y})$
  > The transpose is the map that moves a matrix across a dot product, and a gradient is exactly a dot product waiting to happen. Lesson 1.08 works the derivation through.
- [ ] $(A^\top)^\top = A$
  > True, and useless here. It says nothing about how $A$ interacts with a dot product.
- [ ] $(AB)^\top = B^\top A^\top$
  > This one tells you how transposes compose through a product, which matters for multi-layer chains, but it is not what puts $W^\top$ in the backward pass to begin with.
- [ ] $A^\top A = I$ for any $A$
  > Only true for an orthogonal matrix, which a weight matrix is generally not.
:::

## What to carry forward

- Column $j$ of $A$ is where basis vector $j$ lands. That is all a matrix is.
- $A\mathbf{x}$ is a weighted sum of $A$'s columns.
- An $m \times n$ matrix maps $\mathbb{R}^n \to \mathbb{R}^m$ — input dimension second.
- Composing maps is multiplying matrices, so stacked linear layers collapse into one.
- $A^\top$ is the map that moves $A$ across a dot product, and it is the backward pass.
