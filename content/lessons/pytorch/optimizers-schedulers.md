---
summary: SGD, Adam and AdamW as concrete update rules, what each state tensor costs, and how to pick a learning-rate schedule.
prereqs: [hessians-and-curvature, nn-module]
---

An optimizer is a rule for turning gradients into parameter updates. Every rule in common
use is a cheap approximation to the Newton step of lesson 1.09, and the differences
between them are entirely about what curvature information they keep.

## SGD, with and without momentum

$$
\theta_{t+1} = \theta_t - \eta \, g_t
$$

No state, no assumptions, no curvature information. Momentum adds a velocity term:

$$
v_{t+1} = \mu v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

With $\mu = 0.9$, the velocity is an exponential average over roughly $1/(1-\mu) = 10$
steps. That averaging cancels oscillation across a narrow valley while accumulating
progress along it — an implicit attack on the condition number from lesson 1.09.

```python
import torch

opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
```

## Adam

Adam keeps two exponential moving averages, of the gradient and of its square:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Both start at zero and are therefore biased toward it early on, so they are corrected:

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}
$$

and the update divides by the root of the second moment:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

That division is the whole point. $\sqrt{v}$ estimates the typical gradient magnitude per
parameter, so each parameter gets a step scaled to its own history — the diagonal
curvature approximation of lesson 1.09.

::: key
The bias correction is not cosmetic. Without it, $\hat v_1 = (1-\beta_2)g_1^2 \approx
0.001 g_1^2$, so the first update would be about $\sqrt{1000} \approx 32$ times larger
than intended. Uncorrected Adam diverges in the first few steps. This is also why warmup
and Adam interact: both are managing the same early-training instability.
:::

Defaults worth knowing: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. For
large-batch transformer training $\beta_2 = 0.95$ is common, because the longer window
adapts too slowly when gradients shift quickly.

## AdamW, and why the W matters

Classic L2 regularisation adds $\lambda\theta$ to the gradient. In Adam that penalty then
passes through the $1/\sqrt{\hat v}$ scaling, so parameters with large gradients get
*less* decay — the opposite of the intent.

AdamW decouples them, applying decay directly to the parameters:

$$
\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_t\right)
$$

```python
import torch

# AdamW is the default for transformers. Note the decay value is much
# larger than a typical L2 coefficient, because it is not being scaled.
opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                        betas=(0.9, 0.95), weight_decay=0.1)
```

## What the state costs

| Optimizer | State tensors | Bytes per parameter (fp32) |
|---|---|---|
| SGD | none | 0 |
| SGD + momentum | 1 | 4 |
| Adam / AdamW | 2 | 8 |
| 8-bit Adam | 2 (int8 + scales) | ~2 |
| Adafactor | factored second moment | ~0.01 |

For a 7B model, Adam's state alone is 56 GB. This is why lesson 7.07's sharding exists,
and why Adafactor — which stores only row and column sums of the second moment rather
than the full tensor — is still used for the largest models.

## Learning-rate schedules

A constant learning rate converges to a noise ball of radius proportional to $\eta\sigma$
(lesson 1.16). Shrinking $\eta$ shrinks the ball. That is the entire theoretical
justification for decay.

**Warmup** ramps $\eta$ from near zero over the first few hundred to few thousand steps.
Early in training, $\lambda_{\max}$ is large and Adam's second-moment estimate is built
from too few samples, so a full-size step can diverge. Warmup is not optional for
transformers.

**Cosine decay** is the standard body of the schedule:

$$
\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)
$$

```python
import math
import torch

def lr_lambda(step, warmup=2000, total=100_000, min_ratio=0.1):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return min_ratio + (1 - min_ratio) * cosine

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

for batch in loader:
    loss = model(batch).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()                 # per step, not per epoch
    opt.zero_grad(set_to_none=True)
```

::: warning
Cosine decay bakes the total step count into the schedule. Stop early and you stop at a
high learning rate, leaving performance on the table; run longer than planned and the
rate has already bottomed out. If you might extend a run, use a **warmup–stable–decay**
schedule instead: constant in the middle, with a short decay phase appended whenever you
decide to stop. It also lets you branch multiple final models from one stable checkpoint.
:::

## Gradient clipping

Clip by global norm, not per-parameter:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This rescales all gradients by $\min(1, \text{max\_norm}/\lVert g \rVert)$, preserving the
*direction* and bounding only the magnitude. Per-parameter clipping changes the direction
and is almost never what you want. Clipping must happen after `backward()` and after
unscaling under AMP — lesson 2.12 covers the ordering.

::: exercise
Your loss is stable for 5,000 steps, then spikes and never recovers. Learning rate is
constant. What are the three most likely causes and what would you check first?
:::

::: solution
**1. A bad batch.** A duplicated or corrupted document produces an outsized gradient that
moves parameters far outside the region where the current second-moment estimates are
valid. Check by logging the gradient norm every step — a spike an order of magnitude
above the running median is the signature. Gradient clipping usually prevents this
entirely, which is why it is standard.

**2. fp16 overflow.** If you are using fp16 rather than bf16, an activation exceeding
65,504 becomes `inf`, then `nan`, and every parameter it touches is destroyed
permanently. Check whether the loss became exactly `nan` rather than merely large, and
whether the AMP scaler was repeatedly halving its scale beforehand. Switching to bf16
solves it outright (lesson 2.01).

**3. Learning rate too high for the current curvature.** Even at constant $\eta$, the
loss surface's $\lambda_{\max}$ changes during training. Crossing $\eta > 2/\lambda_{\max}$
begins an oscillation that diverges. Check by plotting the gradient norm over a window —
a growing oscillation rather than a single spike points here.

**What to check first:** log the gradient norm. It distinguishes all three cases in one
plot, costs nothing, and should be in every training run you write. The standard recovery
is to restart from the last good checkpoint, skip the offending data shard, and lower the
maximum clip norm.
:::

## What to carry forward

- Momentum averages ~$1/(1-\mu)$ gradients; Adam additionally scales per parameter by $1/\sqrt{v}$.
- Bias correction is essential, not cosmetic.
- AdamW decouples weight decay from the adaptive scaling — use it for transformers.
- Adam costs 8 bytes per parameter of state; that is what sharding exists to fix.
- Warmup then cosine, stepped per iteration; clip by global norm.
