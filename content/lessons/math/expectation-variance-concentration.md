---
summary: Why averaging reduces noise by exactly the square root of the batch size, and when that guarantee stops holding.
prereqs: [probability-spaces]
---

Every training run is an exercise in estimating an expectation from samples. How wrong
that estimate is, and how fast the error shrinks, determines your batch size, your
learning rate, and how much you should trust an eval number.

## Variance and its algebra

The variance of $X$ measures spread about the mean:

$$
\text{Var}[X] = \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

The second form is usually how you compute it, and it is also a numerical trap: for
large means, $\mathbb{E}[X^2]$ and $\mathbb{E}[X]^2$ are two big numbers whose
difference is small, so catastrophic cancellation can produce a negative "variance".
Use Welford's algorithm in production code.

Two rules cover most uses:

$$
\text{Var}[aX + b] = a^2\text{Var}[X]
\qquad
\text{Var}[X + Y] = \text{Var}[X] + \text{Var}[Y] \;\;\text{(if independent)}
$$

The constant $b$ vanishes — shifting does not change spread. The scale comes out
**squared**, which is why standard deviation, being in the same units as $X$, is
usually the more interpretable number.

## The $1/\sqrt{n}$ law

Take $n$ i.i.d. samples with mean $\mu$ and variance $\sigma^2$, and average them:

$$
\text{Var}\!\left[\frac{1}{n}\sum_i X_i\right] = \frac{1}{n^2}\sum_i \text{Var}[X_i] = \frac{\sigma^2}{n}
$$

Standard deviation of the mean is therefore $\sigma/\sqrt{n}$.

::: key
**Four times the batch gives half the gradient noise.** This is the single most
important scaling fact in optimisation, and the reason large-batch training has
diminishing returns: you pay 4× the compute for a 2× noise reduction.

It is also why the linear scaling rule — multiply the learning rate by $k$ when you
multiply batch size by $k$ — eventually breaks. It holds while gradient noise dominates
and fails once you approach the curvature limit $\eta < 2/\lambda_{\max}$ from
lesson 1.09.
:::

::: check
You raise the batch size from 256 to 1024. By what factor does the standard deviation of the gradient estimate fall?

= 2
> Variance of a mean goes like $\sigma^2/n$, so the standard deviation goes like $\sigma/\sqrt{n}$. Four times the batch is twice the precision — 4× the compute for a 2× noise reduction, which is exactly why large-batch training has diminishing returns.
:::

## Bias and variance

An estimator $\hat{\theta}$ has

$$
\mathbb{E}\!\left[(\hat\theta - \theta)^2\right] = \underbrace{(\mathbb{E}[\hat\theta] - \theta)^2}_{\text{bias}^2} + \underbrace{\text{Var}[\hat\theta]}_{\text{variance}}
$$

Mean squared error splits cleanly into these two parts, and the split is the frame for
several tradeoffs later in the curriculum:

- A minibatch gradient is **unbiased but high variance**. Larger batches cut variance
  and leave bias at zero.
- Advantage estimation in lesson 6.08 deliberately introduces bias (via bootstrapping)
  to cut variance, and the $\lambda$ in GAE dials the tradeoff.
- Early stopping increases bias and decreases variance — the classic regularisation
  trade in lesson 3.07.

## Covariance and correlation

For two variables,

$$
\text{Cov}[X, Y] = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]
$$

Normalising by both standard deviations gives correlation $\rho \in [-1, 1]$.

Independence implies zero covariance. **The converse is false**: $X \sim \mathcal{N}(0,1)$
and $Y = X^2$ have zero covariance and are maximally dependent. Covariance only detects
*linear* relationships, which is the whole reason mutual information exists as a
separate concept in lesson 1.15.

```python
import torch

x = torch.randn(100_000)
y = x ** 2
torch.corrcoef(torch.stack([x, y]))[0, 1]   # ~0.00 -- yet y is a function of x
```

::: check
$X \sim \mathcal{N}(0,1)$ and $Y = X^2$. Their correlation is about zero. What does that tell you?

- [x] Nothing about independence — covariance only detects *linear* relationships, and $Y$ is a deterministic function of $X$
  > Independence implies zero covariance; the converse is false. This gap is the whole reason mutual information exists as a separate concept in lesson 1.15.
- [ ] They are independent, since zero correlation is the definition
  > The definition of independence is $p(x,y) = p(x)p(y)$. Zero correlation is a much weaker consequence.
- [ ] The sample was too small to detect the relationship
  > The correlation is zero in the limit too. It is not a sampling artefact — the symmetry of the Gaussian makes $\mathbb{E}[X^3] = 0$ exactly.
- [ ] $Y$ must have infinite variance
  > $X^2$ for a standard normal has variance 2. Finite, and beside the point.
:::

## Concentration: the actual guarantee

Variance bounds average error. Concentration inequalities bound *tail* probability —
the chance of being far off at all.

**Chebyshev**, assuming only finite variance:

$$
P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}
$$

Weak, but assumption-free. **Hoeffding**, for bounded variables $X_i \in [a, b]$, is
exponentially stronger:

$$
P\!\left(\left|\bar{X} - \mu\right| \ge t\right) \le 2\exp\!\left(\frac{-2nt^2}{(b-a)^2}\right)
$$

This is the inequality behind honest eval error bars. For accuracy — bounded in $[0,1]$ —
on $n$ examples, a 95% confidence interval needs $t \approx \sqrt{\ln(40)/(2n)}$.

| Eval set size | 95% interval on accuracy |
|---|---|
| 100 | $\pm 13.6$ points |
| 1,000 | $\pm 4.3$ points |
| 10,000 | $\pm 1.4$ points |

::: warning
A model scoring 71.2% against 69.8% on a 500-example benchmark has not been shown to be
better. The interval is roughly $\pm 6$ points. Lesson 5.16 returns to this, because the
practice of reporting three-decimal benchmark numbers on small eval sets is endemic and
mostly meaningless.
:::

## When the guarantees fail

All of the above assumes finite variance and independence. Both fail in practice.

**Heavy tails.** Gradient distributions in transformer training are frequently heavy
tailed — the occasional batch produces a gradient orders of magnitude larger than
typical. For sufficiently heavy tails, variance is infinite and $1/\sqrt{n}$ simply does
not apply. This is the real justification for gradient clipping: not that large
gradients are "wrong", but that the averaging guarantee does not cover them.

**Dependence.** Correlated samples carry less information than their count suggests. The
*effective* sample size is smaller than $n$, so error bars computed from $n$ are too
narrow. Near-duplicate documents in a training set do exactly this.

::: exercise
You compare two models on a 500-example benchmark: 71.2% against 69.8%. Is the
difference meaningful? What changes if both models are evaluated on the *same* 500
examples?
:::

::: solution
**Independently.** Each estimate has standard error $\sqrt{p(1-p)/n} \approx
\sqrt{0.21/500} \approx 2.0$ points. The difference has standard error
$\sqrt{2.0^2 + 2.0^2} \approx 2.9$ points. Observed difference 1.4 points is under half
a standard error — entirely consistent with noise.

**Paired.** Evaluating both on the same examples lets you use a paired test, which
cancels the shared difficulty of the items. Only the examples where the models
*disagree* carry information (McNemar's test). If they disagree on, say, 40 examples
split 27–13, that is significant at $p < 0.05$ even though the marginal difference
looked like noise.

The practical lesson: always evaluate on identical examples and report paired
statistics. It can turn an underpowered comparison into a decisive one at no extra cost.
:::

::: check
Which of these estimators is unbiased but high variance, the pairing that a larger batch improves?

- [x] A minibatch gradient
  > Monte Carlo estimation of $\nabla_\theta\mathcal{L}$ is unbiased for any batch size; enlarging the batch cuts variance and leaves bias at zero. That clean split is unusual — most variance reductions cost bias.
- [ ] A bootstrapped advantage estimate
  > Bootstrapping deliberately *introduces* bias to cut variance. GAE's $\lambda$ is the dial between them, in lesson 6.08.
- [ ] An early-stopped model's predictions
  > Early stopping is the classic trade in the other direction: more bias, less variance.
- [ ] A sample variance computed as $\mathbb{E}[X^2] - \mathbb{E}[X]^2$
  > That form is a numerical trap rather than a statistical one — catastrophic cancellation at large means can even return a negative variance.
:::

## What to carry forward

- $\text{Var}[aX] = a^2\text{Var}[X]$; independent variances add.
- Averaging $n$ samples cuts standard deviation by $\sqrt{n}$ — 4× batch for 2× less noise.
- MSE = bias² + variance, and most ML tradeoffs are trades along that line.
- Zero correlation is not independence.
- Hoeffding gives real eval error bars; heavy tails and correlated data void the warranty.
