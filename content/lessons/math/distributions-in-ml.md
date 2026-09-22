---
summary: Six distributions cover almost everything in machine learning. What each one models, its parameters, and where it appears.
prereqs: [probability-spaces]
---

You do not need a catalogue of distributions. You need six, and you need to recognise
which one a paper is implicitly assuming when it picks a loss function.

## Bernoulli and categorical

**Bernoulli** models a single binary outcome with probability $p$:

$$
P(X = 1) = p, \quad P(X = 0) = 1 - p, \quad \mathbb{E}[X] = p, \quad \text{Var}[X] = p(1-p)
$$

Variance peaks at $p = 0.5$ and vanishes at the extremes — a confident model produces
low-variance predictions, which matters for the gradient noise of lesson 1.11.

**Categorical** generalises to $K$ outcomes with probabilities $p_1,\ldots,p_K$ summing
to 1. A language model's output over a 128,000-token vocabulary is a categorical
distribution, produced by applying softmax to logits:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax is the canonical way to turn arbitrary reals into a categorical distribution, and
its negative log-likelihood is exactly cross-entropy (lesson 1.15).

## Gaussian

The workhorse:

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Three reasons it is everywhere:

1. **The central limit theorem.** Sums of many independent contributions converge to
   Gaussian regardless of the components' distributions. Since almost every quantity in
   a network is a sum over many inputs, Gaussians appear whether or not you invite them.
2. **Maximum entropy.** Among all distributions with a given mean and variance, the
   Gaussian has the highest entropy — it assumes the least beyond those two moments.
3. **Closure.** Sums of Gaussians are Gaussian, linear maps of Gaussians are Gaussian,
   conditionals and marginals of joint Gaussians are Gaussian. The algebra stays closed.

The multivariate form is parameterised by mean $\boldsymbol{\mu}$ and covariance
$\Sigma$:

$$
p(\mathbf{x}) \propto \exp\!\left(-\tfrac12 (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)
$$

::: insight
Assuming Gaussian noise and maximising likelihood gives **exactly** mean squared error:
$-\log p \propto (x - \mu)^2$. Whenever you choose MSE you have assumed Gaussian
residuals with constant variance, whether or not you meant to. Lesson 3.04 makes this
the basis for choosing losses deliberately.
:::

## Uniform

Constant density on $[a, b]$: maximum entropy when all you know is the range. Its main
appearance is in initialisation — He and Xavier both come in uniform variants — and in
the $\epsilon$-greedy exploration of lesson 6.11.

## Exponential family, and why it recurs

Bernoulli, categorical, Gaussian, Poisson, Beta and Gamma all share the form

$$
p(x \mid \theta) = h(x)\exp\!\left(\eta(\theta)^\top T(x) - A(\theta)\right)
$$

Membership is not trivia. It guarantees a fixed-size sufficient statistic $T(x)$ — no
matter how much data you collect, you can summarise it in a constant number of numbers
— and it makes the log-likelihood **concave** in the natural parameters, so maximum
likelihood is a convex problem with a unique optimum. Logistic regression and softmax
regression inherit that guarantee; neural networks lose it the moment you add a hidden
layer.

::: check
You choose mean squared error for a regression task. What distributional assumption have you made, whether or not you meant to?

- [x] Gaussian residuals with constant variance
  > $-\log p \propto (x-\mu)^2$ for a Gaussian, so minimising MSE *is* maximum likelihood under that noise model. If outliers are dragging the fit around, the honest diagnosis is that the assumption is wrong, not that the loss needs patching.
- [ ] Laplace residuals, which is why MSE is robust to outliers
  > Laplace residuals give mean *absolute* error, and that is the robust one. MSE squares the outlier's influence.
- [ ] Uniform residuals over a bounded range
  > A uniform likelihood is flat wherever it is non-zero, so it produces no gradient towards any particular fit.
- [ ] None; MSE is assumption-free because it only measures distance
  > Every loss in common use is a negative log-likelihood of something. Squared distance is the Gaussian's.
:::

## Heavy tails: the one to watch for

A Gaussian's tail decays like $e^{-x^2}$, so a 5σ event has probability $3\times10^{-7}$.
Many real quantities decay polynomially instead, like $x^{-\alpha}$, making extremes
vastly more likely.

Where this bites in practice:

- **Token frequencies** follow a Zipf law: rank $r$ has frequency about $1/r$. This is
  why vocabularies need subword tokenization (lesson 4.08) — the tail of rare words never
  ends.
- **Gradient magnitudes** in transformer training are often heavy-tailed, which is why
  clipping is standard rather than optional.
- **Attention weights** after softmax are typically heavy-tailed over positions, which
  is what makes sparse-attention approximations viable at all.

::: warning
If a quantity is heavy-tailed, the sample mean is a poor summary and the sample variance
may not converge at all. Report medians and quantiles. A "mean sequence length" over a
web corpus is nearly meaningless; the 50th, 90th and 99th percentiles tell you what you
actually need for bucketing.
:::

```python
import torch

# Gaussian vs. heavy-tailed (Student-t, df=2) -- same centre, different world.
g = torch.randn(200_000)
t = torch.distributions.StudentT(df=2.0).sample((200_000,))

for name, s in (('gaussian', g), ('student-t', t)):
    q = torch.tensor([0.5, 0.9, 0.99, 0.9999])
    print(name, 'max', f'{s.abs().max():8.1f}',
          'quantiles', [f'{v:.2f}' for v in torch.quantile(s.abs(), q)])
# The Student-t median is comparable; its maximum is orders of magnitude larger.
```

## Choosing a likelihood chooses a loss

| Output | Distribution | Negative log-likelihood |
|---|---|---|
| Real value | Gaussian | Mean squared error |
| Real value, outliers expected | Laplace | Mean absolute error |
| Binary | Bernoulli | Binary cross-entropy |
| One of $K$ classes | Categorical | Cross-entropy |
| Count | Poisson | Poisson loss |

Reading this table in reverse is the useful direction: a loss function is a distributional
assumption in disguise. If MSE is giving you trouble on data with outliers, the real
problem is that you assumed Gaussian residuals for data that is Laplace-ish.

::: exercise
A model predicts house prices with MSE loss. Prices are right-skewed with a long upper
tail. What goes wrong, and what are two fixes?
:::

::: solution
MSE assumes Gaussian residuals with constant variance. Right-skewed prices violate both:
the tail produces large residuals, and squaring them makes a handful of mansions dominate
the gradient. The model under-predicts typical houses in order to reduce error on a few
extreme ones.

**Fix one: transform the target.** Predict $\log(\text{price})$. Log-normal data becomes
approximately Gaussian, MSE becomes appropriate, and the implied error model becomes
multiplicative — "within 10%" rather than "within \$50,000" — which is usually what you
wanted for prices anyway.

**Fix two: change the likelihood.** Use MAE (Laplace residuals), which is far less
sensitive to outliers, or Huber loss, which is quadratic near zero and linear in the
tails. Huber is the pragmatic default when you want MSE's smooth gradients near the
optimum and MAE's robustness far from it.
:::

::: check
Why do token frequencies make subword tokenization necessary?

- [x] They follow a Zipf law, decaying polynomially, so the tail of rare words never ends
  > Rank $r$ has frequency about $1/r$. No finite word vocabulary covers it, and the out-of-vocabulary rate stays stubbornly non-zero — lesson 4.08 is the response.
- [ ] Because a Gaussian tail decays too fast to cover rare words
  > Word frequencies are not Gaussian at all. The problem is the opposite: a much *heavier* tail than exponential decay would give.
- [ ] Because vocabularies must be a power of two for GPU efficiency
  > Alignment is a real minor concern, and unrelated to why the tail needs subwords.
- [ ] Because rare words carry more information and need more parameters
  > Rare words do carry more information per occurrence, but that argues for representing them well, not for the specific decomposition into subwords.
:::

## What to carry forward

- Softmax turns logits into a categorical distribution; its NLL is cross-entropy.
- Gaussians dominate because of the CLT, maximum entropy, and algebraic closure.
- MSE ⇔ Gaussian residuals; MAE ⇔ Laplace; cross-entropy ⇔ categorical.
- Exponential-family likelihoods are concave, so their MLE is a convex problem.
- Heavy tails are common in this field; report quantiles, not means.
