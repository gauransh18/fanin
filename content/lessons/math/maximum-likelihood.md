---
summary: Where loss functions come from. Every standard loss is the negative log-likelihood of some distributional assumption.
prereqs: [distributions-in-ml, bayes-and-independence]
---

Loss functions are not arbitrary design choices. Nearly every one in common use is the
negative log-likelihood of a specific distributional assumption, and recognising which
assumption you have made is how you decide whether the loss fits the problem.

## The principle

Given data $D = \{x_1,\ldots,x_n\}$ and a model $p(x\mid\theta)$, maximum likelihood
picks the parameters that make the observed data most probable:

$$
\hat\theta_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n p(x_i\mid\theta)
$$

Products of many small numbers underflow, so take logs and negate to get a minimisation:

$$
\hat\theta_{\text{MLE}} = \arg\min_\theta \; -\sum_{i=1}^n \log p(x_i\mid\theta)
$$

That expression — the **negative log-likelihood** — *is* your loss function. Everything
else is a special case.

## Deriving the standard losses

**Gaussian residuals → MSE.** Assume $y_i \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$:

$$
-\log p(y_i\mid\theta) = \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2} + \log(\sigma\sqrt{2\pi})
$$

The second term is constant in $\theta$, so minimising NLL is minimising squared error.
Note $\sigma^2$ only scales the loss — which is why MSE never estimates noise magnitude,
and why a model trained with plain MSE cannot tell you its own uncertainty.

**Bernoulli → binary cross-entropy.** Assume $y_i \sim \text{Bernoulli}(p_\theta(x_i))$:

$$
-\log p(y_i\mid\theta) = -\left[y_i\log p_\theta + (1-y_i)\log(1 - p_\theta)\right]
$$

**Categorical → cross-entropy.** With one-hot $y$ and softmax probabilities $p_\theta$:

$$
-\log p(y\mid\theta) = -\log p_{\theta, y_{\text{true}}}
$$

Only the true class's log-probability appears. Language model training is exactly this,
summed over every token position.

**Laplace residuals → MAE.** $p(y) \propto e^{-|y-\mu|/b}$ gives $-\log p \propto
|y - \mu|$ — mean absolute error, and with it robustness to outliers.

::: insight
Picking a loss *is* picking a noise model. If your MSE-trained regressor is being
dragged around by outliers, the honest diagnosis is that you asserted Gaussian residuals
for data that is not Gaussian. Change the assumption rather than patching the symptom.
:::

## What MLE guarantees, and when

Under regularity conditions, MLE is **consistent** (converges to the true parameter as
$n\to\infty$), **asymptotically normal**, and **asymptotically efficient** — no
unbiased estimator has lower variance in the limit.

Those are asymptotic statements, and the conditions include a correctly specified model
and i.i.d. data. Neural networks violate essentially all of them. The guarantees are
why MLE is the default starting point, not a promise about your training run.

The concrete failure at finite $n$ is **overfitting**: MLE will happily drive the
likelihood of training data to its maximum by memorising it.

## MAP: MLE with a prior

Maximum a posteriori adds the prior term from lesson 1.13:

$$
\hat\theta_{\text{MAP}} = \arg\min_\theta \left[-\sum_i \log p(x_i\mid\theta) - \log p(\theta)\right]
$$

A Gaussian prior $\mathcal{N}(0, 1/\lambda)$ contributes $\tfrac{\lambda}{2}\lVert\theta\rVert^2$
— **weight decay is MAP estimation with a Gaussian prior**, exactly. A Laplace prior
contributes $\lambda\lVert\theta\rVert_1$ and produces sparsity.

As $n$ grows, the likelihood term grows with $n$ while the prior stays fixed, so MAP
converges to MLE. Regularisation matters most when data is scarce — which is precisely
the empirical finding that weight decay helps fine-tuning far more than it helps
pretraining.

```python
import torch

# Fit a Gaussian by MLE, and check against the closed form.
data = torch.randn(1000) * 2.5 + 7.0

mu = torch.zeros(1, requires_grad=True)
log_sigma = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([mu, log_sigma], lr=0.05)

for _ in range(2000):
    sigma = log_sigma.exp()
    nll = (0.5 * ((data - mu) / sigma) ** 2 + log_sigma).mean()
    opt.zero_grad(); nll.backward(); opt.step()

print(f'fitted  mu={mu.item():.3f}  sigma={log_sigma.exp().item():.3f}')
print(f'closed  mu={data.mean():.3f}  sigma={data.std(unbiased=False):.3f}')
```

::: warning
The MLE for a Gaussian's variance divides by $n$, not $n-1$, and so is **biased low**.
Unbiasedness and maximum likelihood are different criteria, and MLE does not promise the
former. In deep learning the distinction almost never matters, but it explains why
`torch.std` defaults to the unbiased $n-1$ version while a hand-rolled MLE does not.
:::

::: exercise
You train a model with MSE and want calibrated uncertainty. Show how to extend it to
predict variance too, and identify the failure mode.
:::

::: solution
Have the network output both $\mu_\theta(x)$ and $\log\sigma_\theta(x)$ — predicting the
log keeps $\sigma$ positive without a constraint. Then minimise the full Gaussian NLL:

$$
\mathcal{L} = \frac{1}{n}\sum_i \left[ \frac{(y_i - \mu_\theta(x_i))^2}{2\sigma_\theta(x_i)^2} + \log\sigma_\theta(x_i) \right]
$$

This is **heteroscedastic regression**: the model learns where it is uncertain, and the
$\log\sigma$ term stops it from claiming infinite uncertainty to zero out the first term.

**The failure mode.** The residual term is divided by $\sigma^2$, so the model can reduce
loss on hard examples by inflating their predicted variance instead of fitting them. In
the extreme it gives up on difficult regions entirely, reporting huge uncertainty rather
than learning. The standard fixes are to warm up with plain MSE for the first epochs
before enabling the variance head, or to clamp $\log\sigma$ to a sensible range.
:::

## What to carry forward

- Loss = negative log-likelihood. MSE ⇔ Gaussian, cross-entropy ⇔ categorical, MAE ⇔ Laplace.
- Choosing a loss is choosing a noise model; mismatch shows up as outlier sensitivity.
- MLE is consistent and efficient asymptotically, and overfits at finite $n$.
- MAP = MLE + prior; weight decay is a Gaussian prior, L1 is a Laplace prior.
