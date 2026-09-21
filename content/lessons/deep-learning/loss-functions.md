---
summary: Choosing a loss is choosing a noise model. The standard menu, when each breaks, and the class-imbalance traps.
prereqs: [maximum-likelihood, entropy-cross-entropy-kl]
---

Lesson 1.14 established that every standard loss is a negative log-likelihood. This
lesson is the practical consequence: which one to reach for, and what goes wrong.

## Regression

| Loss | Formula | Implied noise | Behaviour |
|---|---|---|---|
| MSE | $(y-\hat y)^2$ | Gaussian | Fits the **mean**; outliers dominate |
| MAE | $\lvert y-\hat y\rvert$ | Laplace | Fits the **median**; robust, non-smooth at 0 |
| Huber | quadratic then linear | Gaussian core, Laplace tail | Robust with smooth gradients |

That MSE fits the mean and MAE the median is not folklore — it follows from setting the
derivative of the expected loss to zero. It is also the practical difference: on
right-skewed data (house prices, latencies, token counts), the mean and median can be far
apart, and you should know which one you asked for.

```python
import torch
import torch.nn.functional as F

pred = torch.tensor([1.0, 2.0, 3.0])
targ = torch.tensor([1.5, 2.0, 10.0])          # one outlier

F.mse_loss(pred, targ)                          # tensor(16.42) -- outlier dominates
F.l1_loss(pred, targ)                           # tensor(2.50)
F.huber_loss(pred, targ, delta=1.0)             # tensor(2.29)
```

## Classification

**Cross-entropy** is the default and almost always correct:

```python
import torch.nn.functional as F

# Pass LOGITS. cross_entropy applies log_softmax internally.
loss = F.cross_entropy(logits, labels)          # labels are class indices
```

::: warning
The two most common classification loss bugs:

**Applying softmax before cross_entropy.** It is applied twice, the gradients become
tiny, and the model trains to a mediocre plateau without ever erroring.

**Using `BCELoss` instead of `BCEWithLogitsLoss`.** The `WithLogits` version fuses the
sigmoid and computes the loss in a numerically stable form. The unfused version overflows
for logits beyond about ±80 and produces `inf`.
:::

## Class imbalance

At 99:1 imbalance, predicting the majority class always gives 99% accuracy and learns
nothing. Three standard responses, in increasing order of sophistication:

**Class weights.** Weight the loss inversely to class frequency:

```python
import torch
import torch.nn.functional as F

counts = torch.tensor([9900.0, 100.0])
weights = counts.sum() / (len(counts) * counts)   # [0.5, 49.5]
loss = F.cross_entropy(logits, labels, weight=weights)
```

**Focal loss.** Down-weight examples the model already gets right, so training focuses on
the hard minority:

$$
\mathcal{L}_{\text{focal}} = -(1-p_t)^\gamma \log p_t
$$

With $\gamma = 2$, an example at $p_t = 0.9$ contributes $0.01\times$ its usual loss,
while one at $p_t = 0.1$ contributes $0.81\times$. It was introduced for dense object
detection, where background boxes outnumber objects by a thousand to one.

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    p_t = torch.exp(-ce)                        # the true class's probability
    return ((1 - p_t) ** gamma * ce).mean()
```

**Fix the base rate instead.** Often the real problem is that the deployment base rate
differs from the training one — the trap from lesson 1.13. Reweighting the loss changes
the implied prior; sometimes what you want is to leave the loss alone and adjust the
decision threshold at inference, which is cheaper and reversible.

## Ranking and contrastive losses

When you care about relative order rather than absolute values:

**InfoNCE**, the workhorse of contrastive learning, is cross-entropy over similarities:

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{q},\mathbf{k}^+)/\tau)}{\sum_i \exp(\text{sim}(\mathbf{q},\mathbf{k}_i)/\tau)}
$$

```python
import torch
import torch.nn.functional as F

def info_nce(q, k, tau=0.07):
    """q, k: (N, d) matched pairs. Other rows are the negatives."""
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    logits = (q @ k.T) / tau                    # (N, N)
    labels = torch.arange(len(q), device=q.device)
    return F.cross_entropy(logits, labels)      # the diagonal is the positive
```

The temperature $\tau$ controls how sharply the loss discriminates. Small $\tau$ produces
hard negatives mining implicitly; too small and gradients concentrate on a handful of
pairs. This is the CLIP objective, and it is also the shape of the reward model loss in
lesson 5.07.

## Combining losses

Multi-objective training needs weights, and getting them wrong wastes the run:

$$
\mathcal{L} = \lambda_1\mathcal{L}_1 + \lambda_2\mathcal{L}_2
$$

::: key
Scale the terms to comparable **magnitudes** before weighting, not after. If
$\mathcal{L}_1 \approx 5$ and $\mathcal{L}_2 \approx 0.001$, then $\lambda_2 = 1$ means
the second objective is invisible. Log each term separately for the whole run — a
combined loss that goes down while one component goes up is the most common silent
failure in multi-task training, and you cannot see it from the total.
:::

## Label smoothing, revisited

From lesson 1.15: replacing a one-hot target with $(1-\epsilon)$ and $\epsilon/(K-1)$
raises the attainable minimum above zero, so logits cannot diverge.

```python
import torch.nn.functional as F

loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

It reliably improves calibration and usually improves accuracy slightly. It is standard
in vision, and used less in language-model pretraining — where the target distribution is
genuinely uncertain already, so the regularisation is redundant.

::: exercise
You train a segmentation model with cross-entropy. It reaches 97% pixel accuracy but the
predicted masks are empty. What happened, and what loss would you use instead?
:::

::: solution
**What happened:** the foreground occupies about 3% of pixels. Predicting "background"
everywhere gives 97% accuracy, and per-pixel cross-entropy is minimised almost as well by
that degenerate solution as by a correct one — the imbalance is 32:1 and every pixel
contributes equally.

**Why accuracy hid it:** pixel accuracy is the wrong metric for the same reason. It is
dominated by the majority class, exactly as in lesson 1.13's base-rate trap.

**What to use instead.** Dice loss optimises overlap directly and is invariant to the
background's size:

$$
\mathcal{L}_{\text{dice}} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
$$

```python
def dice_loss(logits, targets, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    g = targets.float().flatten(1)
    num = 2 * (p * g).sum(1) + eps
    den = p.sum(1) + g.sum(1) + eps
    return (1 - num / den).mean()
```

The empty prediction now scores the worst possible value, because the numerator is zero.

**In practice** the standard recipe is a sum of Dice and cross-entropy: Dice supplies the
overlap objective, cross-entropy supplies well-behaved per-pixel gradients — Dice alone
has unstable gradients when the prediction is nearly empty. And change the metric to IoU
or Dice score, not pixel accuracy, or you will not notice the next version of this bug
either.
:::

## What to carry forward

- MSE fits the mean, MAE the median; pick the one matching your data's skew.
- Pass logits, never probabilities, and use the `WithLogits` variants.
- Class imbalance: weights, focal loss, or fix the threshold — and change the metric too.
- Scale multi-task terms to comparable magnitudes and log each one separately.
