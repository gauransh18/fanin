---
summary: What the universal approximation theorem says, what it does not say, and why depth beats width in practice anyway.
prereqs: [activation-functions, perceptron-linear-models]
---

An MLP is alternating linear maps and nonlinearities:

$$
\mathbf{h}_1 = g(W_1\mathbf{x}+\mathbf{b}_1), \quad
\mathbf{h}_2 = g(W_2\mathbf{h}_1+\mathbf{b}_2), \quad \ldots, \quad
\hat{\mathbf{y}} = W_L\mathbf{h}_{L-1}+\mathbf{b}_L
$$

```python
import torch.nn as nn

mlp = nn.Sequential(
    nn.Linear(784, 256), nn.GELU(),
    nn.Linear(256, 256), nn.GELU(),
    nn.Linear(256, 10),                  # no activation on the output
)
```

No activation on the final layer: it produces logits, and `F.cross_entropy` applies
log-softmax internally. Applying softmax yourself and then passing it to `cross_entropy`
applies it twice — a common bug that produces a model which trains, badly.

## The theorem

**Universal approximation** (Cybenko 1989, Hornik 1991): a feedforward network with one
hidden layer and a non-polynomial activation can approximate any continuous function on a
compact set to any desired accuracy, given enough hidden units.

It is worth being precise about what this does and does not establish.

::: warning
The theorem says a good approximation **exists**. It says nothing about:

- **How many units** you need. The bound can be exponential in the input dimension.
- **Whether gradient descent finds it.** Existence is not reachability.
- **Whether it generalises.** Fitting the training set is not the goal.
- **Whether you have enough data** to identify it.

So the theorem does not explain why neural networks work. It only rules out the
objection that they are fundamentally too limited — an objection nobody serious has
raised since 1989. Treat it as a floor, not an explanation.
:::

::: check
The universal approximation theorem says a one-hidden-layer network can approximate any continuous function on a compact set. What does it therefore explain about why neural networks work?

- [x] Nothing: it establishes existence and nothing more
  > It says nothing about how many units you need, whether gradient descent finds one, or whether it generalises. The bound can be exponential in the input dimension, and existence is not reachability. Treat the theorem as a floor that rules out an objection nobody serious has raised since 1989, not as an explanation.
- [ ] That depth is unnecessary, since width suffices
  > In principle width suffices; in practice there are functions a depth-$k$ network computes with $O(n)$ units that need $\Omega(2^n)$ at depth $k-1$.
- [ ] That networks will generalise if they are wide enough
  > The theorem is about fitting a *given* function on a compact set. Generalisation is not in its statement at all.
- [ ] That gradient descent will find the approximating network
  > Existence says nothing about reachability by any particular algorithm.
:::

## Why depth wins

If one hidden layer suffices in principle, why is every useful model deep?

**Exponential separation.** There are functions computable by a depth-$k$ network with
$O(n)$ units that require $\Omega(2^n)$ units at depth $k-1$. Compositional structure is
cheap to express with composition and expensive to express by enumeration.

**Feature reuse.** Each layer builds on the last. Recognising a face from edges, then
contours, then parts, then whole objects reuses intermediate results. A wide shallow
network must rediscover the edges independently for every output.

**Matched to the data.** Real signals are compositional — language has morphemes, words,
phrases, sentences; images have edges, textures, parts. A deep architecture encodes that
prior. This is the same "inductive bias" argument that lesson 3.09 makes for convolutions.

A concrete instance: a 4-layer network of width 256 and a 2-layer network of width 2048
have similar parameter counts, and the deep one performs substantially better on
structured data.

::: check
Why is there no activation on the final `nn.Linear` in the example MLP?

- [x] It produces logits, and `F.cross_entropy` applies log-softmax internally
  > Adding softmax yourself applies it twice. The double-softmax bug is insidious: the model still trains, just badly, because the gradients are flattened rather than wrong in an obvious way.
- [ ] Activations are never applied to the last layer of any network
  > A regression head with a bounded target may well end in a sigmoid or tanh. It depends on the output's meaning.
- [ ] It would break backpropagation through the loss
  > Backprop would work fine. The objective would simply be the wrong one.
- [ ] Softmax is too expensive to apply twice
  > Cost is not the issue; correctness is.
:::

## Counting parameters and FLOPs

For layer widths $d_0, d_1, \ldots, d_L$:

$$
\text{params} = \sum_{i=1}^{L} (d_{i-1} d_i + d_i), \qquad
\text{FLOPs} \approx 2\sum_{i=1}^{L} d_{i-1}d_i \;\;\text{per example}
$$

```python
import torch.nn as nn

def summarise(model, batch=1):
    total = sum(p.numel() for p in model.parameters())
    flops = sum(2 * m.in_features * m.out_features
                for m in model.modules() if isinstance(m, nn.Linear))
    print(f'{total/1e6:.2f}M params, {flops*batch/1e6:.1f} MFLOPs forward')

summarise(nn.Sequential(nn.Linear(784, 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(),
                        nn.Linear(256, 10)))
# 0.27M params, 0.5 MFLOPs forward
```

Note the ratio: FLOPs per example is about $2\times$ parameters, exactly the rule from
lesson 1.03. That rule holds for any architecture dominated by matrix products, which is
all of them.

## The MLP inside a transformer

Every transformer block contains an MLP, and it is where most of the parameters live:

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        self.up   = nn.Linear(d, expansion * d)
        self.down = nn.Linear(expansion * d, d)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.down(self.act(self.up(x)))
```

Two structural choices worth understanding.

**It is applied position-wise.** The same MLP runs independently on every token. All
mixing between positions happens in attention; the MLP only transforms each position's
representation. That separation of concerns is the transformer's central design decision.

**The $4\times$ expansion.** The wide middle gives the nonlinearity room to work: a
projection into a higher-dimensional space where features become separable, followed by a
projection back. With $4d$ hidden units, the MLP holds $8d^2$ parameters against
attention's $4d^2$ — so **roughly two-thirds of a transformer's parameters are in its
MLPs**, which is why MoE (lesson 4.14) targets them specifically.

::: exercise
A 2-layer MLP with width $h$ maps $\mathbb{R}^{d} \to \mathbb{R}^{d}$. Under what
condition on $h$ can it represent the identity function, and what does that say about
bottlenecks?
:::

::: solution
The network computes $W_2\,g(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$. For this to equal
$\mathbf{x}$ for all $\mathbf{x}$ in a region, the composite map must have rank $d$.

By lesson 1.04, the rank of a product is bounded by every factor, so
$\text{rank} \le \min(h, d)$. If $h < d$, the composite has rank at most $h < d$ and
cannot be the identity — at least $d - h$ input directions are destroyed and cannot be
reconstructed.

So **$h \ge d$ is necessary**. It is not quite sufficient, since $g$ must also be
invertible on the relevant range: with ReLU, $W_1$ must map the inputs into the positive
orthant for the nonlinearity to act as an identity there, which needs care with
$\mathbf{b}_1$.

**What this says about bottlenecks.** Any layer narrower than its input permanently
discards information. That is useful when you want compression (autoencoders, lesson
3.14) and harmful when you want to preserve a signal. It is precisely why transformer
MLPs expand to $4d$ rather than contracting, and why the residual connection of lesson
3.08 routes around the block entirely — the identity path costs nothing and guarantees
the representation survives even if the block destroys it.
:::

## What to carry forward

- Universal approximation guarantees existence only — not learnability, size or generalisation.
- Depth buys exponential parameter savings on compositional functions, which real data is.
- FLOPs per example ≈ 2 × parameters, for any matmul-dominated architecture.
- A transformer's MLP is position-wise; attention does all the mixing.
- Two-thirds of a transformer's parameters sit in its MLPs.
