---
summary: One line of arithmetic that made depth work, why the identity path matters more than the block, and what the residual stream becomes in a transformer.
prereqs: [chain-rule-backprop, normalization]
---

Before 2015, deeper networks were *worse* — not overfitting, but worse on training loss
too, which ruled out capacity as the explanation. The fix was one line, and it is the
single most consequential architectural idea in this curriculum.

## The degradation problem

A 56-layer plain CNN had higher training error than a 20-layer one. That is a strange
result: the deeper network can represent everything the shallower one can, by setting the
extra 36 layers to the identity. It simply could not *find* that solution.

The diagnosis: learning an identity map with a stack of linear-plus-nonlinear layers is
hard. Each layer must arrange its weights so the composition cancels out exactly, and
gradient descent has no particular pressure to get there.

## The fix

$$
\mathbf{y} = \mathbf{x} + F(\mathbf{x})
$$

Now the identity is the **default**. If $F$ outputs zero, the block passes its input
through unchanged, and $F = 0$ is easy — every weight initialised small already
approximates it. The block only has to learn the *residual*, the difference from identity,
which is what gives the idea its name.

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fn   = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.fn(self.norm(x))      # pre-norm, from lesson 3.06
```

## Why gradients survive

From lesson 1.08, the Jacobian of the block is

$$
\frac{\partial\mathbf{y}}{\partial\mathbf{x}} = I + \frac{\partial F}{\partial\mathbf{x}}
$$

and over $L$ blocks the backward pass multiplies $L$ of these together. Expanding:

$$
\prod_{i=1}^{L}\left(I + \frac{\partial F_i}{\partial \mathbf{x}}\right)
= I + \sum_i \frac{\partial F_i}{\partial\mathbf{x}} + \sum_{i<j}\frac{\partial F_i}{\partial\mathbf{x}}\frac{\partial F_j}{\partial\mathbf{x}} + \cdots
$$

The leading term is the identity. **The gradient reaches layer 1 undamped no matter what
the blocks do** — even if every $\partial F_i$ is near zero, which is exactly the
vanishing-gradient situation of lesson 1.05.

::: key
Contrast with a plain network, whose backward pass is $\prod_i \partial F_i$. If each
factor has spectral radius 0.8, then at $L = 50$ the product is $0.8^{50} \approx 10^{-5}$
and the early layers learn nothing.

With residuals the same network has an $I$ term that never decays. This is why
residual networks scale to hundreds of layers and plain ones do not reach fifty.
:::

## The ensemble reading

Expanding the product above has another interpretation: a residual network of depth $L$
behaves like an **ensemble of $2^L$ paths** of varying length. Each term in the expansion
corresponds to a subset of blocks being "active".

Two pieces of evidence support this. Deleting a single block from a trained ResNet barely
changes its output — an ensemble is robust to losing one member, while a serial pipeline
is not. And the *effective* depth is much shorter than the nominal depth: most of the
gradient comes from paths through relatively few blocks.

This also explains stochastic depth, where blocks are randomly dropped during training: it
is dropout at the level of paths, and it works for the same reason.

## The transformer residual stream

In a pre-norm transformer, every block reads from and writes to one shared vector per
position:

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads)
        self.mlp  = FeedForward(d)

    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)    # attention writes into the stream
        x = x + self.mlp(self.n2(x))           # so does the MLP
        return x
```

That shared vector — the **residual stream** — is best understood as a communication
channel rather than a representation that each layer transforms. Blocks *read* from it,
compute something, and *add* their result back. Nothing is ever overwritten; contributions
accumulate.

This picture is the basis of most mechanistic interpretability work. Because the stream is
additive, you can attribute a logit to individual block contributions by simply decomposing
the sum — a technique called logit attribution, which is only possible because of the
architecture's linearity along the stream.

::: warning
The residual stream's magnitude **grows with depth**, since every block adds to it and
nothing rescales it. Two consequences:

- The final LayerNorm before the output head is doing real work, not decoration.
- Initialising residual output projections at $1/\sqrt{2L}$ (lesson 3.05) keeps the growth
  bounded. Without it, a 48-layer model starts with a stream several times larger than
  intended and needs a smaller learning rate.
:::

## Variants

- **Gated residuals**, $\mathbf{y} = \mathbf{x} + \alpha F(\mathbf{x})$ with learned
  $\alpha$. LayerScale initialises $\alpha$ near zero so blocks start as near-identities,
  which stabilises very deep vision transformers.
- **Highway networks** predate ResNets and use a learned gate $\mathbf{y} = g\odot F(\mathbf{x}) + (1-g)\odot\mathbf{x}$.
  Strictly more general and strictly worse in practice — the ungated identity is the point.
- **DenseNet** concatenates rather than adds, giving every layer access to all previous
  features. More expressive, far more memory.

::: exercise
Why can a residual block not simply be $\mathbf{y} = F(\mathbf{x})$ with $F$ initialised
near the identity? What does the explicit skip connection give that a good initialisation
does not?
:::

::: solution
Initialising $F$ near the identity helps at step zero and only at step zero. The
difference is what happens afterwards.

**The skip is architectural, not initial.** With $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$,
the Jacobian is $I + \partial F$ **for every value the weights ever take**. The gradient
highway cannot be destroyed by training. With $\mathbf{y} = F(\mathbf{x})$ initialised
near identity, the Jacobian is $\partial F \approx I$ at step zero, and there is nothing
preventing training from moving it somewhere with small singular values — at which point
the vanishing gradient returns, potentially thousands of steps into a run.

**Representing the identity exactly is hard.** For $F(\mathbf{x}) = g(W\mathbf{x}+\mathbf{b})$
with ReLU, exact identity requires $W = I$ *and* all pre-activations positive — so it
holds only on part of the input space. The composition over many layers compounds the
error. The skip gives the identity for free, exactly, everywhere.

**Zero is an easier target than the identity.** Weight decay and small initialisation both
push $F$ toward zero, which with a skip means "toward identity". Without a skip, those
same pressures push toward the *zero function*, which destroys the signal. The
regularisation and the architecture agree in one case and fight in the other.

**Empirically:** this was tried. Plain networks with careful identity-like initialisation
train better than naive plain networks and still lose decisively to residual networks past
about 30 layers.
:::

## What to carry forward

- $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ makes the identity the default, so blocks learn a correction.
- The Jacobian $I + \partial F$ gives an undamped gradient path at any weight value.
- A residual network behaves like an ensemble over $2^L$ paths, with short effective depth.
- The transformer's residual stream is an additive communication channel every block reads and writes.
- Stream magnitude grows with depth; the final norm and $1/\sqrt{2L}$ scaling manage it.
