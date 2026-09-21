---
summary: Backpropagation derived from scratch on a two-layer network, by hand, with every shape checked.
prereqs: [derivatives-gradients-jacobians, matrix-multiplication-cost]
---

Backpropagation is not an algorithm you need to memorise. It is the chain rule, applied
right to left because that ordering is cheaper. This lesson derives it on a concrete
network and checks the result against autograd.

## The chain rule, in the shape that matters

For scalars, $\frac{d}{dx} f(g(x)) = f'(g(x))\,g'(x)$. For vector functions composed as
$\mathbf{x} \to \mathbf{h} \to \mathbf{y}$, the Jacobians multiply:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{x}}
$$

For a deep network this becomes a long chain of Jacobians. And because matrix
multiplication is associative (lesson 1.03), you may bracket that chain however you
like — which turns out to decide the entire cost of training.

## Why backward, not forward

Consider a network $\mathbb{R}^N \to \mathbb{R}^{d_1} \to \cdots \to \mathbb{R}$, with
$N$ parameters and a **scalar** loss at the end.

**Forward mode** brackets left to right, propagating $\partial \mathbf{h}_i / \partial
\theta$ forward. Each intermediate is an $d_i \times N$ matrix. You need one pass per
input parameter: $N$ passes.

**Reverse mode** brackets right to left, propagating $\partial \mathcal{L} / \partial
\mathbf{h}_i$ backward. Because the loss is scalar, every intermediate is a *vector* of
size $d_i$, not a matrix. One pass gets you all $N$ gradients.

::: key
Reverse mode costs about 2× the forward pass and produces all $N$ gradients. Forward
mode costs $N$ forward passes. For $N = 7\times10^9$ that is the difference between
training being possible and not.

The asymmetry exists **only because the loss is scalar**. This is why you can never
"backprop" a vector-valued output without first reducing it.
:::

## Deriving it by hand

Take a two-layer network with an MSE loss:

$$
\mathbf{z}_1 = W_1\mathbf{x} + \mathbf{b}_1, \quad
\mathbf{a}_1 = \sigma(\mathbf{z}_1), \quad
\mathbf{z}_2 = W_2\mathbf{a}_1 + \mathbf{b}_2, \quad
\mathcal{L} = \tfrac12\lVert \mathbf{z}_2 - \mathbf{y} \rVert^2
$$

Shapes: $\mathbf{x} \in \mathbb{R}^{d_0}$, $W_1 \in \mathbb{R}^{d_1\times d_0}$,
$W_2 \in \mathbb{R}^{d_2\times d_1}$.

Work backwards, writing $\boldsymbol{\delta}_i = \partial\mathcal{L}/\partial\mathbf{z}_i$.

**Output layer.** $\boldsymbol{\delta}_2 = \mathbf{z}_2 - \mathbf{y}$, shape $d_2$.

**Its parameters.** From the exercise in lesson 1.07,

$$
\nabla_{W_2}\mathcal{L} = \boldsymbol{\delta}_2\mathbf{a}_1^\top \;(d_2\times d_1) \qquad
\nabla_{\mathbf{b}_2}\mathcal{L} = \boldsymbol{\delta}_2 \;(d_2)
$$

**Through the second linear map.** The Jacobian of $\mathbf{z}_2$ with respect to
$\mathbf{a}_1$ is $W_2$, so by the transpose identity of lesson 1.02,

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = W_2^\top \boldsymbol{\delta}_2 \;(d_1)
$$

**Through the activation.** It is elementwise, so its Jacobian is diagonal and the
product is a Hadamard product:

$$
\boldsymbol{\delta}_1 = (W_2^\top\boldsymbol{\delta}_2) \odot \sigma'(\mathbf{z}_1) \;(d_1)
$$

**First layer parameters.** Same pattern as before:

$$
\nabla_{W_1}\mathcal{L} = \boldsymbol{\delta}_1\mathbf{x}^\top \;(d_1\times d_0) \qquad
\nabla_{\mathbf{b}_1}\mathcal{L} = \boldsymbol{\delta}_1
$$

That is the whole algorithm. Three rules, applied repeatedly:

1. **Linear layer**: multiply the incoming gradient by $W^\top$; the weight gradient is
   the outer product of the incoming gradient with the layer's input.
2. **Elementwise function**: multiply by the derivative, elementwise.
3. **Branch point**: a value used twice accumulates the sum of both gradients.

## Checking it

```python
import torch

torch.manual_seed(0)
d0, d1, d2 = 4, 5, 3
x = torch.randn(d0)
y = torch.randn(d2)
W1 = torch.randn(d1, d0, requires_grad=True)
b1 = torch.randn(d1, requires_grad=True)
W2 = torch.randn(d2, d1, requires_grad=True)
b2 = torch.randn(d2, requires_grad=True)

z1 = W1 @ x + b1
a1 = torch.tanh(z1)
z2 = W2 @ a1 + b2
loss = 0.5 * (z2 - y).pow(2).sum()
loss.backward()

# The same gradients, by hand.
d2_ = z2 - y                              # (d2,)
gW2 = torch.outer(d2_, a1)                # (d2, d1)
d1_ = (W2.T @ d2_) * (1 - a1 ** 2)        # tanh' = 1 - tanh^2
gW1 = torch.outer(d1_, x)                 # (d1, d0)

torch.allclose(gW2, W2.grad, atol=1e-6)   # True
torch.allclose(gW1, W1.grad, atol=1e-6)   # True
torch.allclose(d1_, b1.grad, atol=1e-6)   # True
```

## The memory bill

Notice that $\nabla_{W_2}$ needs $\mathbf{a}_1$, and $\nabla_{W_1}$ needs $\mathbf{x}$.
Every layer's backward pass needs its forward **activations**, so the forward pass must
keep them all alive until backward consumes them.

That is the dominant memory cost of training, and it scales with batch size × sequence
length × width × depth — not with parameter count. Gradient checkpointing (lesson 7.08)
trades recomputation for this memory, and it is the single most effective lever when you
hit an out-of-memory error.

::: warning
A branch point sums gradients, it does not average them. Residual connections mean
$\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ has $\partial\mathbf{y}/\partial\mathbf{x} =
I + \partial F/\partial\mathbf{x}$. That identity term is a gradient highway: the
signal reaches early layers undamped even if $\partial F$ vanishes. Lesson 3.08 is
built on this one line.
:::

::: exercise
In a network where a single tensor $\mathbf{h}$ feeds two downstream branches that are
later added, what is $\partial\mathcal{L}/\partial\mathbf{h}$? Why does this make
weight tying work?
:::

::: solution
The gradients **add**: $\partial\mathcal{L}/\partial\mathbf{h} =
\partial\mathcal{L}/\partial\mathbf{h}\big|_{\text{branch 1}} +
\partial\mathcal{L}/\partial\mathbf{h}\big|_{\text{branch 2}}$. This follows from the
multivariable chain rule — each path through which $\mathbf{h}$ influences
$\mathcal{L}$ contributes a term.

Weight tying is the same structure with parameters instead of activations. When an
embedding matrix is shared between the input lookup and the output projection, the
parameter appears at two points in the graph, so it receives the sum of both gradients.
PyTorch handles this automatically by accumulating into `.grad` — which is also
precisely why you must call `optimizer.zero_grad()` between steps, or you will keep
accumulating across batches.
:::

## What to carry forward

- Backprop is the chain rule bracketed right to left, which is cheap *because the loss is scalar*.
- Linear: multiply by $W^\top$, weight grad is the outer product with the input.
- Elementwise: multiply by the derivative. Branch: sum the incoming gradients.
- Activations must be stored for the backward pass — that, not parameters, is your memory bill.
- $I + \partial F$ is why residual networks train.
