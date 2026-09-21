---
summary: One line of algebra that reframes every inference problem, and the conditional independence assumptions that make models tractable.
prereqs: [probability-spaces]
---

Bayes' rule is three symbols of rearrangement with outsized consequences. It is how you
invert a conditional — turning "probability of evidence given a cause" into "probability
of a cause given evidence", which is almost always the direction you need.

## The rule

From the product rule $p(x, y) = p(y \mid x)p(x) = p(x \mid y)p(y)$, divide:

$$
p(y \mid x) = \frac{p(x \mid y)\,p(y)}{p(x)}
$$

The names matter because papers use them without definition:

- $p(y \mid x)$ — **posterior**: belief after seeing evidence.
- $p(x \mid y)$ — **likelihood**: how well hypothesis $y$ explains the evidence.
- $p(y)$ — **prior**: belief before seeing anything.
- $p(x) = \sum_y p(x\mid y)p(y)$ — **evidence**: a normalising constant.

The evidence term is usually intractable — it sums over every hypothesis — which is why
most practical work uses the proportional form:

$$
p(y \mid x) \propto p(x \mid y)\,p(y)
$$

Posterior ∝ likelihood × prior. That is the sentence to remember.

## The base-rate trap

A test is 99% accurate for a disease affecting 1 in 10,000 people. You test positive.
What is the probability you have it?

$$
p(\text{sick}\mid +) = \frac{0.99 \times 0.0001}{0.99\times0.0001 + 0.01\times0.9999} \approx 0.0098
$$

Under 1%. The prior is so small that false positives from the vast healthy population
swamp the true positives — there are roughly 100 false positives for every true one.

::: warning
This is the exact failure mode of rare-event classifiers. A 99%-accurate detector for a
1-in-10,000 event is nearly useless in deployment, and no amount of test-set accuracy
will reveal it if your test set is balanced. Lesson 5.16 returns to this: **always check
whether your eval distribution matches the deployment base rate.**
:::

## Conditional independence

$X$ and $Y$ are conditionally independent given $Z$, written $X \perp Y \mid Z$, if

$$
p(x, y \mid z) = p(x\mid z)\,p(y\mid z)
$$

Once you know $Z$, learning $Y$ tells you nothing further about $X$.

This is a *different* claim from marginal independence, and neither implies the other.
Ice-cream sales and drowning deaths are marginally dependent (both rise in summer) and
conditionally independent given temperature. Conversely, two independent coin flips
become dependent once you condition on their sum — this is "explaining away", and it is
why conditioning is not always clarifying.

## Where the assumptions get made

**Naive Bayes** assumes all features are conditionally independent given the class:

$$
p(y \mid x_1,\ldots,x_n) \propto p(y)\prod_i p(x_i \mid y)
$$

Almost always false, and still often useful — the decision boundary can be right even
when the probabilities are badly calibrated.

**Markov assumption.** A sequence model that conditions only on the last $k$ tokens
assumes $x_t \perp x_{<t-k} \mid x_{t-k:t-1}$. This is exactly what an $n$-gram model
does, and exactly what a transformer **refuses** to do — full attention over the context
is a deliberate rejection of the Markov assumption, which is most of why transformers
replaced RNNs.

**Sliding-window and sparse attention** (lesson 4.13) reintroduce a Markov-like
assumption to buy back compute. Understanding that trade as "which conditional
independences am I willing to assume" is the right way to evaluate a long-context method.

## Priors as regularisation

Maximum a posteriori estimation maximises $\log p(\theta \mid D) = \log p(D\mid\theta) +
\log p(\theta) + \text{const}$. The prior becomes an additive penalty:

| Prior on $\theta$ | $-\log p(\theta)$ | Familiar name |
|---|---|---|
| Gaussian, $\mathcal{N}(0, \sigma^2)$ | $\tfrac{1}{2\sigma^2}\lVert\theta\rVert_2^2$ | L2 / weight decay |
| Laplace, scale $b$ | $\tfrac{1}{b}\lVert\theta\rVert_1$ | L1 / sparsity |
| Uniform | constant | No regularisation |

So weight decay is a Gaussian prior centred at zero, and L1's sparsity comes from the
Laplace density's sharp peak at the origin. Lesson 1.14 works this out properly.

```python
import torch

# Bayesian updating of a coin's bias, one flip at a time.
# Beta(a, b) is conjugate to Bernoulli: the posterior stays Beta.
a, b = 1.0, 1.0                                   # uniform prior
flips = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])

for i, flip in enumerate(flips, 1):
    a, b = a + flip, b + (1 - flip)               # the entire update
    mean, var = a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1))
    print(f'after {i:2d} flips  E[p]={mean:.3f}  sd={var.sqrt():.3f}')
# The mean moves toward 0.8 and the spread shrinks as evidence accumulates.
```

::: exercise
A spam filter has $p(\text{spam}) = 0.3$. The word "meeting" appears in 2% of spam and
30% of legitimate mail. An email contains "meeting" — what is the posterior? Then it also
contains "invoice" (40% of spam, 5% of legitimate). What does the naive Bayes assumption
let you do, and what could go wrong?
:::

::: solution
**After "meeting":**

$$
p(\text{spam}\mid m) = \frac{0.02\times0.3}{0.02\times0.3 + 0.30\times0.7} = \frac{0.006}{0.216} \approx 0.028
$$

The word is strong evidence *against* spam — the prior drops from 30% to 2.8%.

**Adding "invoice"**, naive Bayes assumes the two words are conditionally independent
given the class, so their likelihoods multiply:

$$
p(\text{spam}\mid m, i) = \frac{0.02\times0.40\times0.3}{0.02\times0.40\times0.3 + 0.30\times0.05\times0.7} = \frac{0.0024}{0.0129} \approx 0.186
$$

**What could go wrong:** "meeting" and "invoice" plausibly co-occur in legitimate
business mail far more often than independence predicts. Naive Bayes treats the pair as
two independent pieces of evidence when they are largely one, so it over-updates. This
is the standard failure of the assumption — correlated features are double-counted,
producing posteriors that are far too confident in whichever direction the correlated
group points.
:::

## What to carry forward

- Posterior ∝ likelihood × prior; the evidence term is usually intractable and usually ignorable.
- Rare events make accurate classifiers useless — always check the base rate.
- Conditional and marginal independence are unrelated claims.
- Architectural choices are conditional-independence assumptions in disguise.
- A prior is a regulariser: Gaussian gives L2, Laplace gives L1.
