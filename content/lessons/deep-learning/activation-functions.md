---
summary: Why a nonlinearity is mandatory, what each common choice does to gradients, and why the field settled on GELU and SwiGLU.
prereqs: [perceptron-linear-models, chain-rule-backprop]
---

Without a nonlinearity, depth is free of consequence: lesson 1.02 showed that
$W_2(W_1\mathbf{x})$ is just $(W_2W_1)\mathbf{x}$, so a hundred stacked linear layers have
the expressive power of one. The activation function is what makes depth mean something.

## What to judge an activation by

Four properties predict how a choice will behave:

1. **Gradient magnitude.** From lesson 1.08, backprop multiplies by $g'(z)$ at every
   layer. If $g'$ is typically much less than 1, gradients vanish exponentially with
   depth.
2. **Saturation.** Regions where $g' \approx 0$ are regions where a unit stops learning.
3. **Zero-centredness.** Consistently positive outputs bias the next layer's gradients to
   share a sign, which makes optimisation zigzag.
4. **Cost.** Activations run on every element of every activation tensor. An `exp` is far
   more expensive than a `max`.

## Sigmoid and tanh: the historical default

$$
\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z) = \sigma(z)(1-\sigma(z)) \le 0.25
$$

The derivative maxes out at $0.25$, at $z = 0$. So a 10-layer sigmoid network multiplies
gradients by at most $0.25^{10} \approx 10^{-6}$ — and typically far less, because most
units are not at zero. **This is why deep networks were considered untrainable before
2010.**

$\tanh$ is a rescaled sigmoid, $\tanh(z) = 2\sigma(2z) - 1$, with derivative up to 1 and
zero-centred outputs. Strictly better than sigmoid for hidden layers, and still
saturating at both ends.

Both remain correct in two places, and only two: as the **output** of a binary classifier
(sigmoid, because you want a probability) and inside **gates** (LSTM forget gates, lesson
3.12, where a bounded $(0,1)$ multiplier is exactly the point).

## ReLU: the one that unlocked depth

$$
\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases}1 & z > 0\\ 0 & z < 0\end{cases}
$$

The derivative is exactly 1 on the positive side. Gradients pass through undamped no
matter how deep the network, which is the entire reason it works.

It also gives free sparsity — typically half the units output zero — and costs a single
comparison.

::: warning
**Dying ReLU.** A unit whose pre-activation is negative for every input in the dataset has
zero gradient for every input, so it can never recover. It is dead permanently. Large
learning rates and large negative biases cause this, and a network can lose 40% of its
units this way without any error being raised — you simply get a smaller network than
you paid for.

Diagnose it by logging the fraction of zero activations per layer. Above about 90% is a
problem.
:::

Leaky ReLU, $\max(\alpha z, z)$ with $\alpha = 0.01$, fixes the death by leaving a small
negative slope. It works, and it is rarely used in transformers because the smooth
activations below work better.

::: check
$\sigma'(z) \le 0.25$ everywhere. What does that imply for a ten-layer sigmoid network?

- [x] Gradients are multiplied by at most $0.25^{10} \approx 10^{-6}$ through the stack, and typically far less
  > Backprop multiplies by $g'(z)$ at every layer. This is why deep networks were considered untrainable before 2010, and why ReLU's derivative of exactly 1 on the positive side was such a change.
- [ ] Gradients grow by $4^{10}$, since the derivative is a fraction in the denominator
  > The derivative multiplies; it does not divide. A factor below 1 shrinks.
- [ ] Nothing — the bias terms compensate for the shrinkage
  > Biases shift the pre-activation and can move a unit out of saturation, but they do not scale the gradient that flows through.
- [ ] It only matters if the network is wider than it is deep
  > The shrinkage compounds with *depth*. Width does not enter.
:::

## GELU: the transformer default

$$
\text{GELU}(z) = z\,\Phi(z)
$$

where $\Phi$ is the standard normal CDF. The interpretation is a *stochastic gate*: each
unit is kept with probability $\Phi(z)$, and GELU is the expectation of that. Inputs far
below zero are almost surely dropped, inputs far above are almost surely kept, and the
transition is smooth.

Three properties matter:

- **Smooth everywhere**, so second-order information is well-defined and optimisation is
  better behaved than at ReLU's kink.
- **Non-monotonic** — slightly negative inputs produce slightly negative outputs, which
  gives the function more expressive capacity than a pure gate.
- **Self-regularising**, since the gating is input-dependent.

```python
import torch
import torch.nn.functional as F

z = torch.linspace(-3, 3, 7)
F.relu(z)        # tensor([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
F.gelu(z)        # tensor([-0.004, -0.045, -0.159, 0.0, 0.841, 1.955, 2.996])
F.silu(z)        # SiLU / Swish: z * sigmoid(z)
```

## SwiGLU: what modern models actually use

Gated linear units split the up-projection in two and use one half to gate the other:

$$
\text{SwiGLU}(\mathbf{x}) = \big(\text{SiLU}(W_1\mathbf{x})\big) \odot \big(W_2\mathbf{x}\big)
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up   = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

The multiplicative interaction lets the layer represent products of features, which a
single activation cannot. The cost is a third weight matrix — so implementations shrink
the hidden dimension from $4d$ to about $\tfrac{8}{3}d$ to keep the parameter count
matched. Llama, PaLM and Mistral all use this.

::: key
The empirical ordering for transformer feed-forward blocks, best first:
**SwiGLU > GELU > ReLU ≫ tanh > sigmoid.** The gaps are small between the first three —
typically under 0.5% on downstream tasks — and enormous below them. Do not spend time
tuning the activation; spend it on data.
:::

::: check
A layer reports 94% zero activations and is not learning. What has happened, and how did it get there?

- [x] Dying ReLU: those units are negative for every input and cannot recover
  > Their gradient is zero for every input, so nothing can revive them. Large learning rates and large negative biases cause it. A network can lose 40% of its units this way with no error raised; you simply get a smaller network than you paid for. Log the zero fraction per layer to catch it.
- [ ] Ordinary ReLU sparsity, which sits around half for a healthy layer
  > Around half is normal and useful. Above about 90% is the signal that something is dead rather than sparse.
- [ ] The activations underflowed in fp16
  > Underflow would produce zeros too, but ReLU clamps at exactly zero by design and the pattern here is per-unit and permanent.
- [ ] Weight decay drove the weights to zero
  > Decay shrinks weights smoothly across the layer; it does not produce a stuck subset with a live remainder.
:::

## Softmax is not an activation

Softmax normalises across a dimension, so it is a layer, not an elementwise function. Its
Jacobian is not diagonal:

$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
$$

Every output depends on every input. This matters for two reasons: it is why softmax is
more expensive to backprop than an elementwise activation, and it is why the softmax in
attention creates dependencies between all positions — the mixing that makes attention
work at all.

::: exercise
A 20-layer network with sigmoid activations trains to no better than chance. Explain why,
quantitatively, and give two fixes that do not change the activation.
:::

::: solution
**The cause.** Each layer's backward multiplies by $\sigma'(z) \le 0.25$. With typical
activations away from zero, the realistic average is nearer $0.1$–$0.2$. Over 20 layers
the gradient reaching layer 1 is scaled by roughly $0.15^{20} \approx 3\times10^{-17}$ —
below fp32's resolution relative to the weights. The early layers receive no signal at
all, so they stay at their initialisation, and the network is effectively a shallow
network on top of random features.

**Fix 1: residual connections** (lesson 3.08). With $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$,
the Jacobian is $I + \partial F$. The identity term passes gradient through undamped
regardless of how small $\partial F$ is, so the product no longer decays geometrically.
This is the fix that actually scaled.

**Fix 2: normalization** (lesson 3.06). Placing LayerNorm before each activation keeps
pre-activations near zero, where $\sigma' \approx 0.25$ rather than $\approx 0.01$. That
raises the per-layer factor to its maximum — $0.25^{20} \approx 10^{-12}$, still bad at
20 layers, but combined with careful initialisation (lesson 3.05) it makes moderate depth
trainable, which is historically how it was done before residuals.

**Why changing the activation is the real answer.** Both fixes above are treating a
symptom. ReLU's derivative is exactly 1 on the positive branch, so the product does not
decay at all. This is why the field switched rather than engineering around sigmoid.
:::

## What to carry forward

- Without a nonlinearity, depth collapses to a single linear map.
- Sigmoid's derivative caps at 0.25 — that is the vanishing gradient, quantified.
- ReLU passes gradient at exactly 1 and can die permanently; log your zero fractions.
- GELU is a smooth stochastic gate; SwiGLU adds a multiplicative interaction.
- Softmax is a layer with a dense Jacobian, not an elementwise activation.
