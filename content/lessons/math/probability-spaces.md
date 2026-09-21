---
summary: The minimum probability vocabulary needed to read a machine learning paper, built around what a random variable actually is.
---

Machine learning is applied probability, but the probability it needs is narrow. This
lesson covers the vocabulary precisely enough that later lessons — maximum likelihood,
cross-entropy, policy gradients — can be stated without hand-waving.

## The three pieces

A probability space is a triple $(\Omega, \mathcal{F}, P)$:

- $\Omega$, the **sample space** — everything that could happen.
- $\mathcal{F}$, the **events** — the subsets of $\Omega$ you can assign probability to.
- $P$, the **measure** — a function from events to $[0,1]$ with $P(\Omega) = 1$, additive
  over disjoint events.

For a coin flip, $\Omega = \{H, T\}$. For a language model, $\Omega$ is the set of all
token sequences. The formalism rarely appears in papers, but one consequence does: you
can only speak about the probability of an *event*, not of an individual outcome in a
continuous space. $P(X = 3.7) = 0$ for any continuous $X$; only $P(3.6 < X < 3.8)$ is
meaningful.

## Random variables are functions

A **random variable** is not a variable. It is a function $X : \Omega \to \mathbb{R}$
that assigns a number to each outcome.

This distinction matters when you read $p(\mathbf{x}, \mathbf{y})$ in a paper. There is
one underlying source of randomness, and $\mathbf{x}$ and $\mathbf{y}$ are two
measurements of it. That is why they can be dependent — they are reading the same coin.

**Discrete** random variables take countably many values, described by a probability
mass function $p(x) = P(X = x)$ with $\sum_x p(x) = 1$.

**Continuous** random variables are described by a density $p(x)$ with $\int p(x)\,dx = 1$.

::: warning
A density is **not** a probability. $p(x)$ can exceed 1 — a uniform distribution on
$[0, 0.1]$ has density 10 everywhere on that interval. Only the integral over a region
is a probability. This is why log-likelihoods of continuous variables can be positive,
which surprises people the first time a normalizing-flow model reports one.
:::

## Joint, marginal, conditional

Given a joint distribution $p(x, y)$, two operations recover everything else.

**Marginalisation** sums or integrates out a variable:

$$
p(x) = \sum_y p(x, y) \qquad\text{or}\qquad p(x) = \int p(x, y)\,dy
$$

**Conditioning** restricts to a known value and renormalises:

$$
p(y \mid x) = \frac{p(x, y)}{p(x)}
$$

The denominator is exactly what makes it sum to one again. Rearranged, this is the
**product rule** $p(x,y) = p(y\mid x)p(x)$, and chaining it gives the factorisation
every autoregressive model is built on:

$$
p(x_1, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})
$$

That identity is exact — no assumption, no approximation. A language model is a
parameterised estimate of each conditional on the right. Lesson 5.01 starts here.

## Independence

$X$ and $Y$ are **independent** if $p(x,y) = p(x)p(y)$, equivalently $p(y\mid x) = p(y)$
— knowing one tells you nothing about the other.

Most machine learning assumes training examples are **i.i.d.**: independent and
identically distributed. This assumption is doing real work, and it is routinely
violated in ways that matter:

- Consecutive frames of a video are not independent, which is why replay buffers in
  lesson 6.06 shuffle experience.
- Documents scraped from the same site share style and content, which is why
  deduplication in lesson 5.02 is not a nicety.
- Any distribution shift between training and deployment breaks "identically
  distributed" outright.

## Expectation

The expected value of a function of a random variable is

$$
\mathbb{E}[f(X)] = \sum_x p(x) f(x) \qquad\text{or}\qquad \int p(x) f(x)\,dx
$$

Expectation is **linear**, always: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$,
whether or not $X$ and $Y$ are independent. That unconditional linearity is used
constantly, usually without comment.

When the integral is intractable — which it essentially always is — you estimate it by
sampling:

$$
\mathbb{E}[f(X)] \approx \frac{1}{n}\sum_{i=1}^{n} f(x_i), \qquad x_i \sim p
$$

This is the **Monte Carlo estimator**, and it is unbiased for any $n$. A minibatch
gradient is exactly this estimator applied to $\nabla_\theta \mathcal{L}$; so is every
policy gradient in track 6.

```python
import torch

# A minibatch gradient is a Monte Carlo estimate of the full-dataset gradient.
population = torch.randn(100_000)
true_mean = population.mean()

for n in (8, 64, 512, 4096):
    batches = torch.stack([population[torch.randint(0, 100_000, (n,))].mean()
                           for _ in range(200)])
    print(f'n={n:5d}  bias {(batches.mean()-true_mean).abs():.2e}  '
          f'std {batches.std():.4f}')
# Bias stays near zero at every n; the standard deviation falls like 1/sqrt(n).
```

::: exercise
A language model assigns $p(x_t \mid x_{<t})$ at each step. Show that maximising the
log-likelihood of a corpus decomposes into a sum of per-token terms, and say why that
matters computationally.
:::

::: solution
By the chain rule of probability, $p(x_1,\ldots,x_T) = \prod_t p(x_t\mid x_{<t})$.
Taking logs turns the product into a sum:

$$
\log p(x_1,\ldots,x_T) = \sum_{t=1}^{T} \log p(x_t \mid x_{<t})
$$

Three consequences follow. Numerically, the sum avoids the underflow that multiplying
thousands of probabilities below 1 would cause. Computationally, the terms are
independent given the prefixes, so a causal mask lets you evaluate all $T$ of them in
one parallel forward pass rather than $T$ sequential ones — the core efficiency of
transformer training, lesson 4.06. And statistically, the gradient is a sum over
tokens, so a batch of sequences gives you a Monte Carlo estimate with variance falling
like $1/(\text{batch} \times T)$.
:::

## What to carry forward

- A random variable is a function on outcomes, not a variable.
- Densities are not probabilities and may exceed 1.
- Marginalise by summing out; condition by dividing and renormalising.
- The autoregressive factorisation is exact, not an assumption.
- Expectation is always linear; minibatch gradients are unbiased Monte Carlo estimates.
