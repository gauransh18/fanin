---
summary: Generating several tokens per forward pass without changing the output distribution — the rejection sampling proof, and when the speedup is real.
prereqs: [inference-serving, kv-caching, decoding-strategies]
---

Decode is memory-bound: the GPU reads every weight to produce one token and its arithmetic
units sit idle (lesson 7.09). Speculative decoding fills that idle capacity by **verifying
several candidate tokens in one forward pass**.

The remarkable property is that it is **exact** — the output distribution is identical to
standard sampling, not an approximation.

## The mechanism

1. A cheap **draft** model proposes $k$ tokens autoregressively.
2. The **target** model scores all $k$ in **one** forward pass, because they are already
   known — the same parallelism that makes training efficient (lesson 4.07).
3. A rejection-sampling rule accepts a prefix of the draft and samples a correction.

::: key
Step 2 is where the win comes from. Scoring $k$ known tokens costs almost exactly what
scoring 1 costs, because the operation is memory-bound and the weights are read once either
way.

So if the draft is right $\alpha$ of the time, you get roughly $1/(1-\alpha)$ tokens per
target forward pass instead of 1.
:::

## The acceptance rule

For each drafted token $x$ at position $i$, with draft probability $q(x)$ and target
probability $p(x)$:

- Accept with probability $\min\left(1, \frac{p(x)}{q(x)}\right)$.
- On rejection, sample from the **residual** distribution
  $p'(x) \propto \max(0, p(x) - q(x))$ and stop.

```python
import torch

def speculative_step(target, draft, prefix, k=4, temperature=1.0):
    # 1. Draft k tokens autoregressively (cheap model, k small forward passes).
    drafted, q_probs = [], []
    seq = prefix
    for _ in range(k):
        logits = draft(seq)[:, -1] / temperature
        q = torch.softmax(logits, -1)
        tok = torch.multinomial(q, 1)
        drafted.append(tok); q_probs.append(q.gather(-1, tok))
        seq = torch.cat([seq, tok], -1)

    # 2. ONE target forward over prefix + all k drafted tokens.
    p_logits = target(seq)[:, -(k + 1):] / temperature
    p = torch.softmax(p_logits, -1)

    # 3. Accept/reject left to right.
    accepted = []
    for i in range(k):
        p_i = p[:, i].gather(-1, drafted[i])
        if torch.rand(1).item() < min(1.0, (p_i / q_probs[i]).item()):
            accepted.append(drafted[i])
        else:
            # Residual distribution: the part of p that q under-weighted.
            q_full = torch.softmax(draft(prefix)[:, -1] / temperature, -1)
            residual = torch.clamp(p[:, i] - q_full, min=0)
            residual = residual / residual.sum()
            accepted.append(torch.multinomial(residual, 1))
            return accepted                       # stop at the first rejection

    # All k accepted: the target's own next-token prediction is free.
    accepted.append(torch.multinomial(p[:, k], 1))
    return accepted
```

## Why it is exact

The proof is short and worth seeing, because "free speedup with identical output" sounds like
it cannot be true.

For a token $x$, the probability it is emitted is the probability the draft proposes it and
it is accepted, plus the probability of a rejection followed by sampling $x$ from the
residual:

$$
P(x) = \underbrace{q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right)}_{\text{proposed and accepted}} + \underbrace{P(\text{reject})\cdot\frac{\max(0, p(x)-q(x))}{\sum_{x'}\max(0, p(x')-q(x'))}}_{\text{rejected, then corrected}}
$$

The first term is $\min(q(x), p(x))$. The rejection probability is
$1 - \sum_{x'}\min(q(x'),p(x')) = \sum_{x'}\max(0, p(x')-q(x'))$, which cancels the
denominator of the second term.

So $P(x) = \min(q(x),p(x)) + \max(0, p(x)-q(x)) = p(x)$. Exactly the target distribution, for
any draft model whatsoever.

## The speedup

With acceptance rate $\alpha$ and draft length $k$, the expected accepted tokens per target
pass is

$$
\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{k+1}}{1-\alpha}
$$

```python
def expected_speedup(alpha, k, draft_cost_ratio=0.1):
    tokens = (1 - alpha ** (k + 1)) / (1 - alpha)
    cost = 1 + k * draft_cost_ratio          # one target pass + k draft passes
    return tokens / cost

for alpha in (0.6, 0.8, 0.9):
    best = max(range(1, 12), key=lambda k: expected_speedup(alpha, k))
    print(f'alpha={alpha}: best k={best}, speedup {expected_speedup(alpha, best):.2f}x')
```

| $\alpha$ | Best $k$ | Speedup |
|---|---|---|
| 0.6 | 3 | 1.7× |
| 0.8 | 5 | 2.6× |
| 0.9 | 8 | 3.8× |

Acceptance rate is everything, and it depends on how well the draft matches the target. A
draft from the same family and training data does far better than a generic small model.

## Getting a draft model

- **A smaller model from the same family** — Llama 7B drafting for Llama 70B. Typical $\alpha$
  of 0.7–0.85.
- **Self-speculation (Medusa)** — extra prediction heads on the target model itself, trained
  to predict tokens 2, 3, 4 ahead. No separate model to serve.
- **EAGLE** — drafts in the target's feature space rather than token space, which raises
  acceptance substantially.
- **Prompt lookup / n-gram** — no model at all: search the context for a matching n-gram and
  propose its continuation. Astonishingly effective for summarisation, editing and code
  completion, where output copies from input.

```python
def prompt_lookup(tokens, k=10, n=3):
    """Find the most recent occurrence of the last n tokens, propose what followed."""
    pattern = tokens[-n:]
    for i in range(len(tokens) - n - 1, -1, -1):
        if tokens[i:i + n] == pattern:
            return tokens[i + n:i + n + k]
    return []
```

Zero cost, zero training, and on copy-heavy tasks it beats a draft model.

## Tree attention

Rather than one linear draft, propose a **tree** of candidates and verify them all in one
pass with a carefully constructed attention mask. This raises the chance that some path is
accepted, at the cost of a larger verification batch.

Medusa and EAGLE both use this. It is the difference between a 2× and a 3× speedup in
practice.

::: warning
**Speculative decoding helps latency, not throughput.**

The win comes from idle arithmetic capacity during decode. At batch 1 there is plenty. At
batch 64 the GPU is already busy (lesson 7.09), the verification passes compete with real
work, and the speedup shrinks toward nothing — or goes negative once draft overhead is
counted.

Use it for low-latency single-stream serving. Do not expect it to raise aggregate throughput
on a saturated server, and measure rather than assuming.
:::

::: exercise
Your speculative setup with a 1B draft and 70B target gives only 1.3× speedup. Diagnose.
:::

::: solution
**Measure acceptance rate first.** It is the one number that determines everything:

```python
accepted_tokens / (n_target_passes * k)
```

**If $\alpha$ is low (below ~0.6):**

- **Are the tokenizers identical?** A draft with a different vocabulary cannot propose tokens
  the target scores, and any adapter layer between them costs acceptance. This is the most
  common cause with mismatched model families.
- **Are sampling parameters matched?** The draft should sample at the same temperature and
  top-$p$ as the target. A draft at temperature 0 proposing greedy tokens while the target
  samples at 0.8 gives systematically low acceptance.
- **Is the draft from the same family?** Llama-1B drafting for Llama-70B does much better
  than a generic 1B. If they were trained on different data, expect $\alpha$ around 0.5.
- **Is $k$ too large?** Acceptance decays geometrically along the draft, so a long draft
  wastes work on tokens that will be rejected. Tune $k$ against measured $\alpha$ using the
  formula above.

**If $\alpha$ is high (above 0.8) and the speedup is still 1.3×:**

- **The draft is too expensive.** A 1B model is 1/70 of the target's parameters but its
  forward pass is not 1/70 of the cost — at batch 1 both are memory-bound, so the ratio is
  closer to the weight-size ratio, but kernel launch overhead and Python are fixed costs that
  do not shrink. With $k=5$ you run 5 draft passes per target pass. *Check:* time the draft
  and target passes separately. If the draft is more than about 15% of the target, use a
  smaller draft, or switch to Medusa heads which add almost nothing.
- **The batch size is too large.** Re-read the warning above. Log the server's mean batch
  size — above about 16 and the idle capacity speculation exploits is already being used.
- **Verification is not actually one pass.** A buggy implementation that runs the target once
  per drafted token gets no speedup at all, by construction. *Check:* count target forward
  passes per generated token; it should be well below 1.

**The fastest discriminator:** log $\alpha$, draft-to-target time ratio, and mean batch size.
Those three numbers identify which of the above applies.
:::

## What to carry forward

- The target scores $k$ known tokens in one pass because decode is memory-bound.
- The rejection rule makes the output distribution exactly the target's, for any draft.
- Expected tokens per pass is $(1-\alpha^{k+1})/(1-\alpha)$; tune $k$ to measured $\alpha$.
- Prompt lookup needs no model and wins on copy-heavy tasks.
- It helps single-stream latency and does little for a saturated server.
