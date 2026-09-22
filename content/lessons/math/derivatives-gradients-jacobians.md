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

::: check
For $f:\mathbb{R}^n \to \mathbb{R}^m$, what shape is the Jacobian in this curriculum's convention, and what is row $i$?

- [x] $m \times n$, and row $i$ is the gradient of output $i$
  > It maps input perturbations to output perturbations: $f(\mathbf{x} + \boldsymbol\delta) \approx f(\mathbf{x}) + J\boldsymbol\delta$, which needs $n$ columns to eat and $m$ rows to produce.
- [ ] $n \times m$, and column $i$ is the gradient of output $i$
  > That is the transposed convention some texts use. If a formula's shapes do not work out, transposing is worth trying before concluding it is wrong.
- [ ] $n \times n$, one entry per pair of inputs
  > That is the shape of a Hessian, the second derivative of a *scalar* function.
- [ ] $m \times m$, one entry per pair of outputs
  > Nothing in a first derivative pairs outputs with outputs.
:::

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

::: check
A width-4096 layer has a $4096\times4096$ Jacobian — 67 MB in fp32, per layer per example. How does backpropagation avoid ever forming one?

- [x] It only ever needs vector–Jacobian products $\mathbf{v}^\top J$, which have a closed form per layer type
  > For a linear layer that is $\mathbf{v}^\top W$; for an elementwise activation it is $\mathbf{v} \odot g'(\mathbf{x})$. Each costs about as much as the forward pass, and nothing $4096^2$ is ever allocated.
- [ ] It computes the Jacobian in chunks and discards each chunk after use
  > Still quadratic work. The saving is that the product with $\mathbf{v}$ is computable *directly*, skipping the matrix entirely.
- [ ] It approximates the Jacobian with finite differences
  > Autograd is exact. Finite differences are what you use to *check* it, in gradcheck.
- [ ] It stores the Jacobian in fp16 to halve the memory
  > 33 MB per layer per example is still impossible. Precision is not the lever here.
:::

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

::: check
Adding $\tfrac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2$ to the loss adds what to the gradient, and what does a step then do to the weights before anything else?

- [x] It adds $\lambda\mathbf{w}$, so each step multiplies the weights by $(1 - \eta\lambda)$
  > Straight from $\nabla_\mathbf{x}\lVert\mathbf{x}\rVert_2^2 = 2\mathbf{x}$. That shrinkage is the whole of weight decay — lesson 3.07 picks up what it does and does not do in Adam.
- [ ] It adds $\lambda\mathbf{w}^2$, so large weights are punished quadratically
  > The *loss* is quadratic in $\mathbf{w}$; its gradient is linear. The step is proportional to the weight, not to its square.
- [ ] It adds $\lambda$, a constant pull towards zero
  > That would be $\ell_1$ decay, whose gradient is a constant $\lambda\,\text{sign}(\mathbf{w})$ — and which is why $\ell_1$ drives weights exactly to zero while $\ell_2$ only shrinks them.
- [ ] Nothing; weight decay is applied by the optimizer, not the gradient
  > Decoupled weight decay does move it out of the gradient, but the plain $\ell_2$ penalty described here goes through the gradient exactly as above.
:::

## What to carry forward

- A derivative is the best linear approximation; only its shape changes as dimensions grow.
- Gradients match the shape of the input, which sets your optimizer's memory cost.
- Jacobians are $m\times n$ and are never materialised — autograd computes $\mathbf{v}^\top J$.
- $\nabla \lVert \mathbf{x}\rVert^2 = 2\mathbf{x}$ is weight decay; $\mathbf{r}\mathbf{x}^\top$ is a linear layer's weight gradient.
