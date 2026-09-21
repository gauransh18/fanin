---
summary: What normalization actually fixes, why LayerNorm replaced BatchNorm in sequence models, and why RMSNorm dropped half of it.
prereqs: [initialization, hessians-and-curvature]
---

Normalization layers rescale activations to a controlled distribution. They are the
reason modern networks tolerate high learning rates, and the reason initialisation
matters less than it used to.

## BatchNorm

Normalise each feature across the **batch**:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2+\epsilon}}, \qquad y_i = \gamma\hat{x}_i + \beta
$$

where $\mu_B$ and $\sigma_B^2$ are computed over the batch dimension for each feature.
The learnable $\gamma$ and $\beta$ let the layer undo the normalization if that is what
the network wants — without them, you have removed capacity rather than added stability.

At evaluation time there is no batch to normalise over, so BatchNorm keeps running
averages and uses those instead. This is the `train()`/`eval()` distinction from lesson
2.08, and forgetting to switch is a classic source of "great training accuracy, terrible
validation accuracy".

::: warning
BatchNorm creates a **dependency between examples in a batch**. Three consequences:

- Small batches give noisy statistics, and performance degrades badly below about 16.
- Train and eval compute different functions, so a bug in the running statistics shows
  up only at evaluation.
- Under distributed training the statistics are per-device unless you use `SyncBatchNorm`,
  which adds a synchronisation to every layer.

For sequence models with variable lengths, padding positions corrupt the statistics
outright. This is why transformers do not use it.
:::

## LayerNorm

Normalise across the **feature** dimension, per example:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j, \qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2, \qquad
y = \gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

Every example is normalised independently. No batch dependency, no running statistics,
identical behaviour in training and evaluation, and no problem with variable-length
sequences.

```python
import torch
import torch.nn as nn

x = torch.randn(4, 16, 512)          # (batch, tokens, features)
ln = nn.LayerNorm(512)               # normalises the LAST dimension
out = ln(x)

out.mean(-1).abs().max()             # ~0 -- each token is centred
out.std(-1).mean()                   # ~1 -- and scaled
```

## RMSNorm

Drop the mean subtraction and the bias:

$$
y = \gamma \frac{x}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}}
$$

The empirical finding was that **re-centring contributes almost nothing** — the rescaling
is doing the work. Dropping it removes one reduction pass over the feature dimension and
one parameter vector, which is a measurable saving at scale.

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # Compute in fp32: the sum of squares over 4096 elements loses
        # too much precision in bf16.
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight
```

The `.float()` is not optional. Summing $d$ squared values in bf16 accumulates rounding
error proportional to $\sqrt{d}$, and at $d = 8192$ that is visible in the loss. Every
production implementation upcasts.

Llama, Mistral, Gemma and most recent models use RMSNorm.

## Pre-norm versus post-norm

Where the normalization sits relative to the residual connection changes trainability
fundamentally.

**Post-norm** (the original transformer): $\mathbf{x} \leftarrow \text{Norm}(\mathbf{x}+F(\mathbf{x}))$.

**Pre-norm** (everything since): $\mathbf{x} \leftarrow \mathbf{x} + F(\text{Norm}(\mathbf{x}))$.

::: key
Pre-norm leaves a **clean residual path** from input to output with no normalization on
it. Gradients flow from the loss to layer 1 through nothing but additions — the
$I + \partial F$ Jacobian of lesson 1.08, uninterrupted.

Post-norm puts a normalization between every pair of blocks, so the gradient is rescaled
$L$ times on the way back. Deep post-norm transformers require careful warmup and often
diverge without it; pre-norm ones train reliably at 100+ layers.

The cost is that the residual stream's magnitude grows through depth, since nothing
rescales it. Modern models add a single final norm before the output head to fix that.
:::

```python
import torch.nn as nn

class PreNormBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.mlp  = FeedForward(d)

    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)     # norm inside, add outside
        x = x + self.mlp(self.n2(x))
        return x
```

## Why it works

The original "internal covariate shift" explanation has not held up. Two better accounts:

**It smooths the loss surface.** Normalization reduces the Lipschitz constant of the loss
and its gradient, which lowers $\lambda_{\max}$ and therefore raises the stable learning
rate from lesson 1.09's bound $\eta < 2/\lambda_{\max}$. This is the effect you actually
feel — normalized networks tolerate learning rates an order of magnitude higher.

**It decouples direction from magnitude.** After normalization, scaling a weight matrix by
$c$ leaves the output unchanged, so the weight's *norm* stops affecting the function.
Gradient descent then operates effectively on direction alone, which interacts with weight
decay in a way worth knowing: decay shrinks the norm, which increases the effective
learning rate, giving an implicit schedule.

## Others you will meet

- **GroupNorm**: normalise over groups of channels. Batch-independent, used in vision when
  batches are small.
- **InstanceNorm**: GroupNorm with one channel per group. Style transfer.
- **QK-Norm**: normalise queries and keys before the attention dot product. Prevents
  attention logits from growing during long training runs, which is a real source of
  instability at scale.

::: exercise
You replace LayerNorm with BatchNorm in a transformer. Training loss looks fine but
validation is terrible. Give the two mechanisms and say which dominates.
:::

::: solution
**Mechanism 1: padding corrupts the statistics.** BatchNorm averages each feature across
the batch *and* sequence positions. Padded positions contribute zeros (or garbage) to
$\mu_B$ and $\sigma_B^2$, so the normalization depends on how much padding a batch
happens to contain. Batches with different length distributions get different
normalizations of the same content. Training tolerates this as noise; evaluation with a
different batching strategy sees a systematically different function.

**Mechanism 2: train/eval mismatch.** During training, each token is normalised by its own
batch's statistics. At evaluation, the running averages are used. Those averages were
accumulated over batches whose padding fraction and content distribution differ from the
evaluation set — so the eval-time normalization is simply wrong. With a batch size of 1
at inference, this is the entire behaviour.

**Which dominates:** mechanism 2, and it is easy to confirm. Run evaluation with
`model.train()` (so batch statistics are used) and compare. If validation loss jumps back
to reasonable, the running statistics are the problem. If it stays bad, padding is
corrupting both paths.

**The fix is not to patch BatchNorm.** Use LayerNorm. It has no batch dependency, so both
mechanisms disappear, and it is the reason every transformer uses it.
:::

## What to carry forward

- BatchNorm ties examples together: bad for small batches, variable lengths and inference.
- LayerNorm normalises per example over features — no batch dependency, no running stats.
- RMSNorm drops re-centring; compute its reduction in fp32.
- Pre-norm keeps a clean residual path and is why deep transformers train.
- Normalization works by lowering $\lambda_{\max}$, which raises the usable learning rate.
