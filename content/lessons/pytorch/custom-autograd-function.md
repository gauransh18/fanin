---
summary: When to write your own backward, how to write it correctly, and the gradient check that proves it.
prereqs: [autograd-graph, chain-rule-backprop]
---

Autograd handles composition automatically, so most code never needs a custom backward.
The cases that do are worth recognising, because they are where the biggest memory and
speed wins live.

## When you actually need one

- **Memory.** Autograd saves every intermediate. A fused implementation can save one
  input and recompute the rest, trading FLOPs for memory. This is gradient checkpointing
  (lesson 7.08) in miniature.
- **Numerical stability.** The analytic gradient of a composition is often better
  conditioned than the product of its pieces' gradients. Log-sum-exp and softmax
  cross-entropy are the canonical examples.
- **Non-differentiable operations.** Quantization and hard sampling have zero or
  undefined gradients; a straight-through estimator substitutes a usable surrogate.
- **Calling out to something autograd cannot see.** A custom CUDA kernel, a Triton
  kernel, a C++ extension.

## The interface

```python
import torch

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)          # keep what backward needs
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x        # one return per forward input

y = Square.apply(torch.randn(3, requires_grad=True))
```

Three rules govern this:

1. **`backward` returns one gradient per `forward` input**, in the same order. Return
   `None` for inputs that do not need one (integers, flags).
2. **`grad_output` has the shape of the forward output.** Your return values must have
   the shapes of the forward inputs. If they do not, you have a bug that autograd will
   not catch.
3. **Use `ctx.save_for_backward` for tensors**, not plain attributes — it participates in
   the version-counting that detects the in-place mutations of lesson 2.06. Non-tensors
   go on `ctx` directly.

## A real example: stable log-sum-exp

The naive composition overflows, and its gradient inherits the problem:

$$
\text{LSE}(\mathbf{x}) = \log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}, \quad m = \max_i x_i
$$

The derivative is beautifully simple — it is just softmax:

$$
\frac{\partial\,\text{LSE}}{\partial x_i} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \text{softmax}(\mathbf{x})_i
$$

So the backward pass needs no exponentials at all if you save the softmax:

```python
import torch

class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        m = x.max(dim=dim, keepdim=True).values
        shifted = x - m
        s = shifted.exp().sum(dim=dim, keepdim=True)
        out = (m + s.log()).squeeze(dim)
        ctx.save_for_backward(shifted.exp() / s)    # this is softmax(x)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (softmax,) = ctx.saved_tensors
        return grad_output.unsqueeze(ctx.dim) * softmax, None   # None for `dim`

x = torch.randn(4, 10, requires_grad=True) * 100    # large: naive exp overflows
out = LogSumExp.apply(x, 1)
out.sum().backward()
torch.allclose(x.grad, torch.softmax(x, dim=1))     # True
```

Note the `None` for `dim` — it is an input to `forward`, so `backward` must account for
it positionally.

::: check
A custom `LogSumExp` saves `softmax(x)` in the forward pass. What does that buy the backward pass?

- [x] Backward needs no exponentials at all, because $\partial\,\text{LSE}/\partial x_i$ *is* softmax
  > The analytic gradient of the whole composition is better conditioned than the product of its pieces' gradients — which is exactly why log-sum-exp and softmax cross-entropy are the canonical cases for a custom backward.
- [ ] It avoids saving the input, halving memory
  > The softmax is the same size as the input, so memory is unchanged. Stability is the win here.
- [ ] It lets the backward run in a lower precision
  > Nothing about saving softmax changes the dtype. It changes which operations backward has to perform.
- [ ] It makes the function differentiable, which it otherwise would not be
  > Log-sum-exp is smooth everywhere. The naive composition is differentiable too — just numerically worse.
:::

## Straight-through estimators

Quantization rounds, and rounding has zero gradient almost everywhere. The
straight-through estimator pretends the operation was the identity on the backward pass:

```python
import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output          # pass through unchanged

x = torch.tensor([1.3, 2.7, -0.4], requires_grad=True)
RoundSTE.apply(x).sum().backward()
x.grad                              # tensor([1., 1., 1.])
```

This is not the true gradient — there is no true gradient — it is a deliberate
approximation. It underpins quantization-aware training (lesson 5.12) and the codebook
lookup in VQ-VAEs. There is a one-line idiom for the same thing without a custom class:

```python
# Forward value from `quantized`, backward path through `x`.
y = x + (quantized - x).detach()
```

::: warning
A straight-through estimator is a **biased** gradient estimator. It works when the
forward and backward functions stay close — rounding is near-identity — and degrades
badly when they do not. Clipping the pass-through to the quantization range
(`grad_output * (x.abs() <= 1)`) is a standard improvement.
:::

::: check
Your custom `backward` returns tensors of the wrong shape. What happens?

- [x] Nothing catches it automatically — `backward` must return one gradient per forward input, with that input's shape, and getting it wrong is a bug autograd will not find
  > This is what `torch.autograd.gradcheck` exists for. Write the function, then check it against finite differences before trusting a single training step.
- [ ] Autograd raises a shape error immediately
  > It may, if the shape is incompatible with the accumulation target — but broadcasting frequently absorbs the mismatch and produces plausible wrong numbers instead.
- [ ] The gradient is silently zeroed for safety
  > There is no such safety net. Whatever you return is what accumulates.
- [ ] PyTorch falls back to numerical differentiation
  > It never falls back. `gradcheck` does numerical differentiation, and only when you call it.
:::

## Always gradcheck

`torch.autograd.gradcheck` compares your analytic backward against finite differences. It
needs `float64` — `float32` finite differences are too noisy to distinguish a real bug
from rounding.

```python
import torch

x = torch.randn(4, 10, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(lambda t: LogSumExp.apply(t, 1), (x,))   # True or raises
```

::: key
Write the custom Function, then write the gradcheck, then never think about it again. A
wrong backward does not crash — it trains to a slightly worse loss, and you will spend a
week blaming the data. Five lines of gradcheck in your test suite is the cheapest
insurance in this curriculum. Lesson 7.12 makes it a policy.
:::

## Checkpointing, the built-in version

The most common memory-for-compute trade already exists, so you rarely need to hand-roll
it:

```python
import torch
from torch.utils.checkpoint import checkpoint

def block(x):
    return torch.nn.functional.gelu(x @ W1) @ W2

# Forward stores only `x`; the activations inside `block` are recomputed
# during backward.
out = checkpoint(block, x, use_reentrant=False)
```

Pass `use_reentrant=False` — the older reentrant implementation interacts badly with
anything that needs the autograd graph to be well-formed, and is being retired.

::: exercise
You implement a custom Function whose `backward` returns a gradient of the wrong shape —
$(B, T)$ instead of $(B, T, d)$. What happens, and how would you catch it?
:::

::: solution
PyTorch raises at the accumulation step, with a message about a shape mismatch between
the gradient and the tensor it is being accumulated into. So this particular error *is*
caught — but only if that input is a leaf whose `.grad` gets written.

The dangerous variant is a gradient that is the right shape and the wrong **values**, or
one that happens to broadcast. If you returned $(B, T, 1)$ instead of $(B, T, d)$, it
would broadcast cleanly into the accumulation and produce a plausible, wrong gradient
with no error at all — the same silent failure as the broadcasting bug in lesson 2.03.

`gradcheck` catches both, because it compares against numerically computed derivatives
element by element. Run it on a small `float64` input as a unit test. For functions
where finite differences are genuinely inapplicable — a straight-through estimator, for
instance — assert the shapes explicitly in `backward` instead:

```python
assert grad_x.shape == ctx.input_shape, f'{grad_x.shape} != {ctx.input_shape}'
```
:::

## What to carry forward

- Custom Functions exist for memory, stability, non-differentiability and foreign kernels.
- One returned gradient per forward input, in order, `None` where not needed.
- Save tensors with `ctx.save_for_backward` so version checking works.
- Straight-through estimators are biased on purpose; clip them to the useful range.
- `gradcheck` in float64, always.
