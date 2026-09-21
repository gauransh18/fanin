---
summary: Scalar derivatives generalise to gradients and Jacobians. Getting the shapes right is most of what makes autograd comprehensible.
prereqs: [vectors-norms-geometry, matrices-as-linear-maps]
---

Backpropagation is the chain rule applied to functions between vector spaces. Before
you can follow it, you need to be fluent in what a derivative *is* when inputs and
outputs are both vectors — and specifically, what shape the answer has.

## The one idea behind every derivative

A derivative is the **best linear approximation** to a function near a point. For a
scalar function that means

$$
f(x + \delta) \approx f(x) + f'(x)\,\delta
$$

For everything that follows, only the object playing the role of $f'(x)$ changes. It
is always "the linear map that best predicts the change in output from a small change
in input".

## Gradient: many inputs, one output

For $f : \mathbb{R}^n \to \mathbb{R}$ — a loss function, say — the derivative is a
vector of partial derivatives:

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n} \right)
$$

and the linear approximation is $f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f \cdot \boldsymbol{\delta}$.

Two properties make gradients useful for optimisation, and both come from that dot
product:

- $\nabla f$ points in the direction of **steepest increase**, because a dot product is
  maximised when the two vectors align. Hence descending means stepping along $-\nabla f$.
- $\lVert \nabla f \rVert$ is the rate of change in that steepest direction, which is
  why gradient norm is the natural thing to clip.

::: insight
The gradient has the **same shape as the input**. A model with parameters $\theta \in
\mathbb{R}^N$ has $\nabla_\theta \mathcal{L} \in \mathbb{R}^N$. This is why optimizer
state costs as much memory as the parameters themselves, and why Adam's two moments
triple your memory bill — lesson 7.08 budgets it.
:::

## Jacobian: many inputs, many outputs

For $f : \mathbb{R}^n \to \mathbb{R}^m$, the derivative is an $m \times n$ matrix:

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

Row $i$ is the gradient of output $i$. The approximation is $f(\mathbf{x} + \boldsymbol{\delta})
\approx f(\mathbf{x}) + J\boldsymbol{\delta}$, which is exactly the linear-map picture
from lesson 1.02: $J$ maps input perturbations to output perturbations.

For a linear layer $f(\mathbf{x}) = W\mathbf{x} + \mathbf{b}$, the Jacobian is just $W$
— a linear function is its own best linear approximation. For an elementwise activation
$f(\mathbf{x})_i = g(x_i)$, the Jacobian is diagonal with $g'(x_i)$ on the diagonal,
which is why activations are cheap to backprop through.

| Function | Derivative | Shape |
|---|---|---|
| $\mathbb{R} \to \mathbb{R}$ | derivative | scalar |
| $\mathbb{R}^n \to \mathbb{R}$ | gradient | $n$ |
| $\mathbb{R}^n \to \mathbb{R}^m$ | Jacobian | $m \times n$ |
| $\mathbb{R}^n \to \mathbb{R}$, twice | Hessian | $n \times n$ |

## Why nobody materialises a Jacobian

A hidden layer of width 4096 has a $4096 \times 4096$ Jacobian: 16.8M entries, 67 MB in
fp32, for **one layer at one training example**. Materialising these is impossible.

The trick is that backpropagation never needs the Jacobian itself — it only ever needs
**vector–Jacobian products**, $\mathbf{v}^\top J$. And for every layer type, that
product has a closed form that costs about as much as the forward pass:

$$
\text{linear: } \mathbf{v}^\top W \qquad
\text{elementwise: } \mathbf{v} \odot g'(\mathbf{x})
$$

This is the single most important implementation fact about autograd, and lesson 2.06
shows how PyTorch encodes it.

```python
import torch

x = torch.randn(4, requires_grad=True)
W = torch.randn(3, 4)
y = torch.tanh(W @ x)

# The full Jacobian, for inspection only -- never do this in a training loop.
J = torch.autograd.functional.jacobian(lambda z: torch.tanh(W @ z), x)
J.shape                                     # torch.Size([3, 4])

# What backprop actually computes: one vector-Jacobian product.
v = torch.tensor([1.0, 0.0, -2.0])
y.backward(v)
torch.allclose(x.grad, v @ J, atol=1e-5)    # True
```

## The gradients you should know by heart

With $\mathbf{x} \in \mathbb{R}^n$, $A$ symmetric where relevant:

$$
\nabla_{\mathbf{x}} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}
\qquad
\nabla_{\mathbf{x}} (\mathbf{x}^\top A \mathbf{x}) = 2A\mathbf{x}
\qquad
\nabla_{\mathbf{x}} \lVert \mathbf{x} \rVert_2^2 = 2\mathbf{x}
$$

The last one explains weight decay in one line: adding $\tfrac{\lambda}{2}\lVert \mathbf{w}
\rVert^2$ to the loss adds $\lambda\mathbf{w}$ to the gradient, so every step multiplies
weights by $(1 - \eta\lambda)$ before doing anything else. Lesson 3.07 takes it from there.

::: warning
Layout conventions differ. This curriculum uses **denominator layout**, where
$\nabla_{\mathbf{x}} f$ has the shape of $\mathbf{x}$ and the Jacobian is
$\partial \mathbf{y} / \partial \mathbf{x}$ with shape $m \times n$. Some texts
transpose everything. If a formula's shapes do not work out, try transposing before
concluding it is wrong.
:::

::: exercise
For $\mathcal{L} = \tfrac{1}{2}\lVert W\mathbf{x} - \mathbf{y} \rVert_2^2$ with
$W \in \mathbb{R}^{m\times n}$, derive $\nabla_W \mathcal{L}$ and check its shape.
:::

::: solution
Write $\mathbf{r} = W\mathbf{x} - \mathbf{y}$, so $\mathcal{L} = \tfrac12 \mathbf{r}^\top\mathbf{r}$.

By the chain rule, $\partial \mathcal{L}/\partial \mathbf{r} = \mathbf{r}$. Since
$r_i = \sum_k W_{ik}x_k - y_i$, we get $\partial r_i / \partial W_{ij} = x_j$ and zero
for other rows. So

$$
\frac{\partial \mathcal{L}}{\partial W_{ij}} = r_i x_j
\qquad\Longrightarrow\qquad
\nabla_W \mathcal{L} = \mathbf{r}\mathbf{x}^\top
$$

The outer product of the residual with the input — shape $m \times n$, matching $W$ as
it must. This is the update rule behind linear regression, and it is also literally
what a linear layer's backward pass computes: `grad_W = grad_output.T @ input`.
:::

## What to carry forward

- A derivative is the best linear approximation; only its shape changes as dimensions grow.
- Gradients match the shape of the input, which sets your optimizer's memory cost.
- Jacobians are $m\times n$ and are never materialised — autograd computes $\mathbf{v}^\top J$.
- $\nabla \lVert \mathbf{x}\rVert^2 = 2\mathbf{x}$ is weight decay; $\mathbf{r}\mathbf{x}^\top$ is a linear layer's weight gradient.
