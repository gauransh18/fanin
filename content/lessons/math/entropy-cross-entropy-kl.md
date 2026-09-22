---
summary: Information theory as it is actually used: what a loss of 2.3 nats means, why KL is not a distance, and where each quantity appears in training.
prereqs: [distributions-in-ml, maximum-likelihood]
---

Cross-entropy is the loss almost every model in this curriculum is trained with. Knowing
what its value *means* — not just that lower is better — lets you read a training curve
and know whether the number is good.

## Surprise and entropy

Define the **surprise** of an outcome with probability $p$ as $-\log p$. It is zero for
a certain event, large for a rare one, and additive over independent events because
logs turn products into sums. Those three properties pin it down uniquely up to the base
of the logarithm.

**Entropy** is expected surprise:

$$
H(p) = -\sum_x p(x)\log p(x) = \mathbb{E}_{x\sim p}[-\log p(x)]
$$

It measures how uncertain a distribution is, in **nats** with natural log or **bits**
with $\log_2$. A deterministic distribution has $H = 0$; a uniform distribution over $K$
outcomes has the maximum $H = \log K$.

::: key
Entropy is the average number of nats needed to encode a sample from $p$ with an optimal
code. This is not a metaphor — it is Shannon's source coding theorem, and it is why a
language model's loss is directly interpretable as compression.
:::

## Cross-entropy

Encode samples from $p$ using a code optimised for $q$, and the average cost is the
**cross-entropy**:

$$
H(p, q) = -\sum_x p(x)\log q(x)
$$

You always pay at least $H(p)$, with equality only when $q = p$. In supervised learning
$p$ is the one-hot label and $q$ is the model, so the sum collapses to a single term:

$$
H(p, q) = -\log q(y_{\text{true}})
$$

which is exactly the negative log-likelihood of lesson 1.14. Cross-entropy loss and
maximum likelihood are the same objective written in two vocabularies.

### Reading the number

A language model's loss is the cross-entropy per token in nats. Convert it:

$$
\text{perplexity} = e^{\mathcal{L}}
$$

Perplexity is the effective number of equally likely choices the model is deciding
between at each step.

| Loss (nats) | Perplexity | Interpretation |
|---|---|---|
| 11.7 | 120,000 | Uniform over a 120k vocabulary — untrained |
| 4.6 | 100 | Roughly a decent $n$-gram model |
| 2.3 | 10 | Choosing among ~10 plausible next tokens |
| 1.6 | 5 | Strong modern model on general text |
| 0 | 1 | Perfect prediction — memorisation or a trivial corpus |

So "the loss went from 2.5 to 2.4" means perplexity fell from 12.2 to 11.0 — a 10%
reduction in effective branching. Small-looking loss deltas at low loss are large.

::: check
A language model's loss falls from 2.5 nats to 2.4. What happened to perplexity?

- [x] It fell from about 12.2 to about 11.0 — roughly a 10% cut in effective branching
  > Perplexity is $e^{\mathcal{L}}$, so a difference in loss is a *ratio* in perplexity. Small-looking deltas at low loss are large, which is why late-training curves look flat and are not.
- [ ] It fell by 0.1, since perplexity tracks loss linearly
  > The relationship is exponential. Near a loss of 2.5 a 0.1 change moves perplexity by more than a full point.
- [ ] It fell by about 4%, in proportion to the loss
  > $2.4/2.5$ is 4%, but that arithmetic is on the log scale. The quantity readers care about moves by $e^{-0.1} \approx 0.905$.
- [ ] Nothing measurable; 0.1 nats is within noise
  > Whether it is within noise depends on the eval, not on the size of the number. In effective-choice terms it is a real move.
:::

## KL divergence

The **excess** cost of using $q$ when the truth is $p$:

$$
D_{\text{KL}}(p \parallel q) = H(p,q) - H(p) = \sum_x p(x)\log\frac{p(x)}{q(x)}
$$

Two properties hold and one does not.

- $D_{\text{KL}} \ge 0$ always, with equality only when $p = q$ (Gibbs' inequality).
- $D_{\text{KL}}$ is **not symmetric**: $D_{\text{KL}}(p\parallel q) \neq D_{\text{KL}}(q\parallel p)$.
- It is therefore **not a distance**, and it violates the triangle inequality too.

The asymmetry is not a technicality — it changes model behaviour:

**Forward KL, $D_{\text{KL}}(p\parallel q)$** is *mass-covering*. Wherever $p$ has mass,
$q$ must too, or the $\log(p/q)$ term blows up. Minimising it spreads $q$ over all of
$p$'s modes, producing blurry averages. This is what maximum likelihood minimises.

**Reverse KL, $D_{\text{KL}}(q\parallel p)$** is *mode-seeking*. Wherever $q$ has mass,
$p$ must too, but $q$ may ignore regions of $p$ entirely. Minimising it picks one mode
and commits. Variational inference and the RLHF objective in lesson 5.08 both use this
direction, which is part of why RLHF reduces output diversity.

```python
import torch
import torch.nn.functional as F

p = torch.tensor([0.5, 0.5, 0.0, 0.0])       # a two-mode truth
q1 = torch.tensor([0.25, 0.25, 0.25, 0.25])  # mass-covering
q2 = torch.tensor([0.98, 0.01, 0.005, 0.005])# mode-seeking

def kl(a, b, eps=1e-10):
    return (a * ((a + eps) / (b + eps)).log()).sum()

print(f'forward  cover {kl(p, q1):.3f}   seek {kl(p, q2):.3f}')   # cover wins
print(f'reverse  cover {kl(q1, p):.3f}   seek {kl(q2, p):.3f}')   # seek wins
```

::: check
You want a generative model that commits to one mode of the data cleanly rather than blurring across all of them. Which direction of KL should you minimise?

- [x] Reverse KL, $D_{\text{KL}}(q\parallel p)$, which is mode-seeking
  > Wherever $q$ puts mass, $p$ must too — but $q$ is free to ignore whole regions of $p$. That is exactly "pick one mode and be sharp".
- [ ] Forward KL, $D_{\text{KL}}(p\parallel q)$, which is mode-seeking
  > Forward KL is the mass-*covering* one: wherever $p$ has mass, $q$ must too, or $\log(p/q)$ blows up. It produces blurry averages, and it is what maximum likelihood minimises.
- [ ] Either; KL is symmetric
  > It is not, and the asymmetry is not a technicality — it changes what the trained model does.
- [ ] Neither; use the Jensen–Shannon divergence, which is the only symmetric option
  > JS is symmetric and a reasonable choice, but the question is which *direction* buys mode-seeking, and reverse KL is the answer.
:::

## Mutual information

How much knowing $Y$ reduces uncertainty about $X$:

$$
I(X; Y) = D_{\text{KL}}\big(p(x,y) \parallel p(x)p(y)\big) = H(X) - H(X\mid Y)
$$

It is zero exactly when $X$ and $Y$ are independent, and — unlike correlation — it
detects **any** dependence, linear or not. That is the gap lesson 1.11 flagged with the
$Y = X^2$ example.

It is also symmetric and non-negative, and it is notoriously hard to estimate in high
dimensions from samples, which is why contrastive objectives (InfoNCE and friends)
optimise lower bounds on it rather than the thing itself.

## Label smoothing, in one line

Replacing a one-hot target with $(1-\epsilon)$ on the true class and $\epsilon/(K-1)$
elsewhere raises the target's entropy from 0 to something positive. Since the attainable
minimum loss is now $H(p) > 0$, the model can no longer drive logits to infinity chasing
probability 1. That is the whole mechanism: it caps confidence by making perfect
confidence suboptimal.

::: exercise
Your language model reports validation loss 2.1 nats/token. A colleague's reports 3.0
bits/token. Which is better, and what does the difference mean in perplexity?
:::

::: solution
Convert to a common unit. $3.0$ bits $= 3.0 \times \ln 2 \approx 2.079$ nats.

So the colleague's model is slightly **better**: 2.079 against 2.1 nats.

In perplexity, $e^{2.1} \approx 8.17$ versus $e^{2.079} \approx 8.00$ — about a 2%
difference in effective branching factor. Close enough that tokenizer differences would
dominate: a model with a larger vocabulary compresses more text per token and will show
higher per-token loss for the same underlying quality. Comparing loss across tokenizers
is meaningless; compare **bits per byte** instead, which normalises it away. Lesson 5.16
covers the rest of the eval traps.
:::

::: check
Cross-entropy loss with a one-hot label collapses to $-\log q(y_{\text{true}})$. What is that the same thing as?

- [x] The negative log-likelihood of lesson 1.14 — the same objective in a different vocabulary
  > $p$ being one-hot kills every term but the true class. Cross-entropy and maximum likelihood are not merely related; for this setup they are identical.
- [ ] The KL divergence between the label and the model
  > Close: $D_{\text{KL}}(p\parallel q) = H(p,q) - H(p)$, and $H(p) = 0$ for a one-hot label, so they happen to coincide *here*. They are not the same object in general.
- [ ] The entropy of the model's output distribution
  > That would be $-\sum_x q(x)\log q(x)$, which measures the model's own uncertainty and does not involve the label at all.
- [ ] The perplexity of the model
  > Perplexity is $e^{\mathcal{L}}$ — a transformation of this number, not the number.
:::

## What to carry forward

- Entropy is expected surprise, and it is a real compression bound.
- Cross-entropy loss is negative log-likelihood; $e^{\mathcal{L}}$ is perplexity.
- KL is non-negative and asymmetric: forward covers modes, reverse picks one.
- Mutual information catches nonlinear dependence that correlation misses.
- Never compare per-token loss across different tokenizers.
