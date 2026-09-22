---
summary: The variance argument that gives Xavier and He, why the wrong scale kills a deep network before step one, and what transformers do differently.
prereqs: [activation-functions, expectation-variance-concentration]
---

Initialisation looks like a detail and is not. The wrong scale makes a deep network
untrainable from the first forward pass — activations either explode to `inf` or collapse
to zero, and no optimizer recovers from either.

## The problem, stated as variance

Consider one layer, $\mathbf{y} = W\mathbf{x}$, with $n_{\text{in}}$ inputs. Assume
weights are i.i.d. with mean 0 and variance $\sigma_w^2$, and inputs are i.i.d. with
variance $\sigma_x^2$ and independent of the weights. Then for each output,

$$
\text{Var}[y_i] = \text{Var}\!\left[\sum_{j=1}^{n_{\text{in}}} W_{ij}x_j\right] = n_{\text{in}}\,\sigma_w^2\,\sigma_x^2
$$

using the rules from lesson 1.11: independent variances add, and constants come out
squared.

For the variance to be preserved across the layer — neither growing nor shrinking — we
need $n_{\text{in}}\sigma_w^2 = 1$, that is

$$
\sigma_w^2 = \frac{1}{n_{\text{in}}}
$$

::: key
Get this wrong and the effect **compounds geometrically with depth**. If each layer
multiplies variance by $c$, then after $L$ layers the scale is $c^L$.

At $c = 1.5$ and $L = 50$: $1.5^{50} \approx 6\times10^8$ — activations overflow.
At $c = 0.7$ and $L = 50$: $0.7^{50} \approx 2\times10^{-8}$ — activations vanish.

The usable window is narrow, and it narrows as networks get deeper. This is the same
spectral-radius argument as lesson 1.05, applied forward instead of backward.
:::

## Xavier and He

**Xavier (Glorot)** initialisation balances the forward and backward passes. The forward
pass wants $\sigma_w^2 = 1/n_{\text{in}}$; the backward pass, which multiplies by $W^\top$,
wants $1/n_{\text{out}}$. Xavier compromises:

$$
\sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

It assumes the activation is roughly linear near zero, which holds for $\tanh$ and not
for ReLU.

**He (Kaiming)** initialisation corrects for ReLU. ReLU zeroes half its inputs, halving
the variance, so the weights must compensate:

$$
\sigma_w^2 = \frac{2}{n_{\text{in}}}
$$

That factor of 2 is the entire difference, and at 50 layers it is the difference between
a trainable network and a dead one — $0.5^{50} \approx 10^{-15}$.

```python
import torch
import torch.nn as nn

layer = nn.Linear(512, 512)

nn.init.xavier_normal_(layer.weight)                      # tanh, sigmoid
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu') # ReLU, GELU
nn.init.zeros_(layer.bias)                                 # biases: zero
```

Biases start at zero. There is nothing to preserve — they add no variance — and a nonzero
bias only shifts units toward saturation.

::: check
A 50-layer network initialised so each layer multiplies activation variance by 0.7. What is the scale at the output?

- [x] About $2\times10^{-8}$
  > The activations have vanished before training starts. $0.7^{50}$. The effect compounds geometrically with depth, which is the same spectral-radius argument as lesson 1.05 applied forward instead of backward. At $c = 1.5$ it overflows instead.
- [ ] About 0.7, since normalization keeps each layer in range
  > There is no normalization layer in this setup. That is exactly what makes initialisation load-bearing.
- [ ] About 35, since 50 layers times 0.7
  > The factors multiply, not add.
- [ ] Unchanged, because the bias terms restore the scale
  > Biases shift the mean. They do not rescale the variance.
:::

## Seeing it fail

```python
import torch
import torch.nn as nn

def activation_scale(scale, depth=40, width=512):
    x = torch.randn(256, width)
    for _ in range(depth):
        w = torch.randn(width, width) * scale
        x = torch.relu(x @ w.T)
    return x.std().item()

n = 512
print(f'too small  {activation_scale((1.0 / n) ** 0.5):.3e}')   # vanishes
print(f'He         {activation_scale((2.0 / n) ** 0.5):.3e}')   # stable
print(f'too large  {activation_scale((4.0 / n) ** 0.5):.3e}')   # explodes
```

Run it: the middle line stays near 1 across 40 layers while the others are off by many
orders of magnitude. That is the whole argument, empirically.

::: check
He initialisation uses $\sigma_w^2 = 2/n_{\text{in}}$ where Xavier uses about $1/n_{\text{in}}$. What is the factor of 2 for?

- [x] ReLU zeroes half its inputs, halving the variance
  > The weights have to compensate. Xavier assumes the activation is roughly linear near zero, which holds for tanh and not for ReLU. At 50 layers the missing factor is $0.5^{50} \approx 10^{-15}$ — the difference between a trainable network and a dead one.
- [ ] It accounts for the backward pass as well as the forward
  > That is Xavier's compromise, $2/(n_{\text{in}} + n_{\text{out}})$, and a different concern.
- [ ] It compensates for the bias term, which adds variance
  > Biases are usually initialised to zero and add no variance.
- [ ] It doubles the effective learning rate at initialisation
  > Initialisation scale and learning rate interact, but the factor is derived from ReLU's effect on variance, not from step size.
:::

## What transformers do

Two things differ from the classical recipe.

**A small fixed standard deviation.** GPT-style models initialise most weights from
$\mathcal{N}(0, 0.02^2)$ regardless of width. With LayerNorm in every block, activation
scale is re-normalised constantly, so preserving it exactly through initialisation matters
much less.

**Residual-aware scaling on output projections.** Each residual block *adds* to the
stream, so after $L$ blocks the variance has accumulated $L$ times. Scaling the weights
that write into the residual stream by $1/\sqrt{2L}$ cancels that growth:

```python
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        self.attn_out = nn.Linear(d, d)
        self.mlp_out  = nn.Linear(4 * d, d)
        # Two residual writes per block, hence 2 * n_layers.
        for proj in (self.attn_out, self.mlp_out):
            nn.init.normal_(proj.weight, std=0.02 / (2 * n_layers) ** 0.5)
```

Without this, a 48-layer model starts with a residual stream several times larger than
intended, which forces a smaller learning rate and slows the whole run.

::: warning
**Never initialise all weights to the same value, zero included.** Every unit in a layer
would then compute the same function and receive the same gradient, so they would remain
identical forever — the layer would have the expressive power of a single unit. This is
*symmetry breaking*, and randomness is the only thing providing it.

Biases are the exception: they can all be zero, because the weights already break the
symmetry.
:::

## Special cases worth knowing

- **Embeddings**: $\mathcal{N}(0, 0.02^2)$, matching the residual stream's scale.
- **LayerNorm**: gain 1, bias 0 — start as the identity.
- **Output layer of a policy or value head**: often near-zero, so the network starts with
  near-uniform outputs and does not commit to a bad policy before learning anything
  (lesson 6.07).
- **Forget gate bias in an LSTM**: initialise to 1 or 2, so the gate starts open and
  gradients flow from the beginning (lesson 3.12).

::: exercise
You initialise a 100-layer ReLU network with Xavier instead of He. Predict quantitatively
what happens to activations by layer 100, and say what you would observe in training.
:::

::: solution
**The arithmetic.** Xavier on a square layer gives $\sigma_w^2 = 2/(n+n) = 1/n$. He
requires $2/n$. So Xavier's variance is **half** what ReLU needs, and each layer
multiplies the activation variance by $\tfrac12$.

After 100 layers: $(1/2)^{100} \approx 8\times10^{-31}$ in variance, so the standard
deviation falls by $2^{-50} \approx 10^{-15}$.

**What you would observe.** The final layer's activations are around $10^{-15}$ — at the
edge of fp32's useful range and flushed to exactly zero in bf16, whose smallest normal is
about $10^{-38}$ but which has only ~3 significant digits. The logits are then essentially
constant, so the loss sits at exactly $\ln(K)$ — 2.303 for 10 classes — and does not move.

Gradients are equally dead: the backward pass multiplies by the same factors, so early
layers receive around $10^{-15}$ times the signal they should. The loss curve is a
perfectly flat line.

**How to confirm it in 30 seconds.** Print the per-layer activation standard deviation on
one forward pass:

```python
for i, layer in enumerate(model):
    x = layer(x)
    if i % 10 == 0:
        print(i, f'{x.std():.3e}')
```

A geometric decay across layers is unambiguous, and it distinguishes this from every other
cause of a flat loss curve. Lesson 3.16 makes this check part of a standard routine.
:::

## What to carry forward

- Variance must be preserved per layer, or the error compounds as $c^L$.
- Xavier for tanh; He (the factor of 2) for ReLU and friends; biases zero.
- Transformers use a fixed small std plus $1/\sqrt{2L}$ scaling on residual writes.
- Identical initial weights means the layer never differentiates — randomness breaks symmetry.
- Print per-layer activation std on one forward pass; it diagnoses this instantly.
