---
summary: Spending compute at inference instead of training, why chain of thought works mechanistically, and what verifiable rewards changed.
prereqs: [decoding-strategies, rlhf-ppo, scaling-laws]
---

Scaling laws (lesson 5.03) describe training compute. A separate axis exists: spending more
compute **per query** at inference. The tradeoff is different and in some domains far more
favourable.

## Why chain of thought works

A transformer does a fixed amount of computation per token: $L$ layers, regardless of how
hard the question is. There is no mechanism for thinking longer about a harder problem
within one forward pass.

Generating intermediate tokens changes that. Each generated token is fed back as input, so
the model can perform a sequence of computations with the results of earlier steps
available in context.

::: key
**Chain of thought converts depth into length.** A problem requiring $k$ sequential steps
cannot be solved in $L$ layers if $k > L$ — but it can be solved in $k$ forward passes, with
intermediate results written into the context.

This is a claim about computational capability, not about prompting. It is why "think step
by step" improves multi-step problems and does essentially nothing for lookup questions:
lookup needs no sequential depth.
:::

## The simple methods

**Chain of thought** — prompt for intermediate steps, or fine-tune on data containing them.

**Self-consistency** — sample $n$ chains at temperature and take the majority answer.
Different reasoning paths make different errors; the correct answer is the mode.

```python
from collections import Counter

def self_consistency(model, prompt, n=16, temperature=0.8):
    answers = [extract_answer(model.generate(prompt, temperature=temperature))
               for _ in range(n)]
    counts = Counter(a for a in answers if a is not None)
    answer, votes = counts.most_common(1)[0]
    return answer, votes / len(answers)        # the vote share is a confidence signal
```

Accuracy rises roughly logarithmically in $n$, with most of the gain by $n=16$. It requires
a well-defined answer to vote on, so it works for maths and multiple choice and not for
open-ended generation.

**Best-of-$n$ with a verifier** — sample $n$, score each with a reward or process model, and
take the best. Better than majority voting when a good verifier exists, and it degrades
badly when the verifier is exploitable (lesson 5.07).

::: check
Why does "think step by step" help on multi-step problems and do essentially nothing for lookup questions?

- [x] Chain of thought converts depth into length
  > A problem needing $k$ sequential steps cannot fit in $L$ layers when $k > L$, but can be done in $k$ forward passes with intermediates in context. It is a claim about computational capability, not about prompting style. A lookup needs no sequential depth, so there is nothing for the extra passes to do.
- [ ] It gives the model more chances to sample the right answer
  > That is self-consistency, which is a different method — sample $n$ chains and take the mode.
- [ ] It shifts the output distribution towards training data that contains reasoning
  > That is part of why the prompt works at all, and it does not explain why the gain is specific to multi-step problems.
- [ ] It increases the effective context, letting the model attend to more of the question
  > The question is the same length either way.
:::

## Process supervision

An outcome reward says only whether the final answer was right. A **process reward model**
scores each reasoning step.

The difference matters because a correct answer reached by faulty reasoning is rewarded
identically to one reached correctly — and the model learns whatever produced the reward.
Process supervision penalises the faulty path directly.

```python
def prm_guided_search(model, prm, prompt, beam=4, max_steps=10):
    """Beam search over reasoning steps, scored by a process reward model."""
    beams = [(prompt, 0.0)]
    for _ in range(max_steps):
        candidates = []
        for text, score in beams:
            for step in model.sample_next_step(text, n=beam):
                candidates.append((text + step, score + prm.score(text, step)))
        beams = sorted(candidates, key=lambda b: -b[1])[:beam]
        if all(is_complete(t) for t, _ in beams):
            break
    return max(beams, key=lambda b: b[1])[0]
```

Process supervision outperforms outcome supervision on mathematical reasoning by a
substantial margin. Its cost is annotation: someone must label each step.

## Verifiable rewards

The development that changed reasoning training: in domains where correctness is
**checkable by a program**, you do not need a reward model at all.

- Maths — compare against a known answer, or check with a symbolic system.
- Code — run the tests.
- Formal proofs — run the proof checker.

The reward is then exact and unhackable. Every failure mode from lesson 5.07 — length bias,
formatting bias, sycophancy — disappears, because the verifier does not care about style.

This is what makes RL on reasoning work where RLHF is fragile. The model generates long
chains, the verifier scores them, and RL (typically GRPO, lesson 6.14) optimises the policy
against a signal that cannot be gamed.

The observed behaviour is that models trained this way learn to generate much longer chains,
to backtrack when a path fails, and to check their own work — none of which was explicitly
demonstrated. They emerge because they raise the probability of a verified-correct answer.

::: warning
**Verifiable domains are a small fraction of what people want models to do.** Maths, code
and formal logic have checkable answers. Writing, analysis, advice and judgement do not.

Capability gained through verifiable-reward training transfers to neighbouring tasks
partially and unevenly. Treat "we solved reasoning in maths" as a statement about maths
until evidence says otherwise.
:::

::: check
Self-consistency samples $n$ chains at temperature and takes the majority answer. Why does that work?

- [x] Different reasoning paths make different errors
  > Wrong answers scatter while the correct one is the mode. It needs an answer that can be compared for equality, which is why it applies cleanly to maths and code and awkwardly to open-ended generation.
- [ ] Higher temperature produces better reasoning
  > Temperature is there to produce *diverse* chains. Each individual chain is typically worse than a greedy one.
- [ ] The majority answer has the highest likelihood under the model
  > The highest-likelihood answer is what greedy decoding gives, and self-consistency frequently beats it.
- [ ] Averaging reduces variance, as it does for gradients
  > The mechanism is voting over discrete answers, not averaging a continuous quantity.
:::

## The inference-compute tradeoff

| Method | Cost | Typical gain |
|---|---|---|
| Greedy | 1× | baseline |
| Chain of thought | 3–10× tokens | large on multi-step |
| Self-consistency, $n=16$ | 16× | moderate |
| Best-of-$n$ with a verifier, $n=64$ | 64× | large where verification is good |
| Long RL-trained reasoning | 10–100× tokens | large in verifiable domains |

The finding that reframed the field: for a fixed total compute budget, spending it on
inference for a smaller model can beat spending it on training a larger one — in domains
where verification is cheap relative to generation.

That last clause is the whole condition. Verification must be cheaper than generation for
the tradeoff to work, which is exactly when a verifier exists.

## Practical notes

- **Do not use temperature 0 with self-consistency.** Identical samples carry no
  information; you need diversity for the vote to mean anything. Temperature 0.7–1.0.
- **Budget the reasoning length.** Long chains cost latency linearly. Many deployments cap
  it or let the caller choose.
- **Cache the prompt.** With a long shared system prompt, prefill dominates (lesson 4.13) —
  prompt caching makes repeated reasoning queries far cheaper.
- **Measure per-query cost, not per-token cost.** A model generating 4,000 reasoning tokens
  to answer correctly may be cheaper than three failed attempts at 200 tokens each.

::: exercise
Your model scores 45% on a maths benchmark greedily and 62% with self-consistency at
$n=32$. Where is the remaining 38%, and what would you try next?
:::

::: solution
**Decompose the gap first.** Self-consistency can only help when a correct chain is among
the samples. Measure two things:

- **pass@32**: the fraction of problems where *at least one* of the 32 samples is correct.
- **maj@32**: what you measured, 62%.

If pass@32 is, say, 80%, then for 18% of problems the model *can* find the answer but the
majority vote does not select it. That is a **selection** problem and it is much easier to
fix than a capability problem.

If pass@32 is 63%, voting is already extracting nearly everything available, and the
remaining 37% is a genuine capability gap.

**If it is selection (pass@32 ≫ maj@32):**

- **Train a verifier or process reward model** and use best-of-$n$ instead of majority
  voting. This reliably captures most of the pass@$n$–maj@$n$ gap.
- **Weight votes by chain confidence** — average log-probability, or a PRM score — rather
  than counting equally.
- **Raise $n$**, though returns are logarithmic and 32 is already past the steep part.

**If it is capability (pass@32 ≈ maj@32):**

- **Check the failure modes by hand.** Read 30 wrong answers. Arithmetic slips, misread
  problems, and genuinely unknown techniques need completely different fixes, and you
  cannot tell which dominates without looking.
- **If arithmetic slips dominate**, give the model a calculator tool (lesson 5.15). This is
  a tokenization limitation (lesson 4.08), not a reasoning one.
- **If the model does not know the technique**, more inference compute cannot help. That
  needs training — RL with verifiable rewards on similar problems is the current best
  approach.
- **Check the extraction.** A surprising share of apparent failures are answer-parsing bugs:
  the model was right and your regex did not find it. Sample 20 "wrong" answers and read
  them before doing anything else.

**Order of operations:** measure pass@32, then read 30 failures. Those two steps cost an
hour and determine which of the above is worth doing.
:::

## What to carry forward

- Chain of thought converts fixed depth into variable length — that is why it helps multi-step problems.
- Self-consistency needs temperature and a well-defined answer to vote on.
- Process rewards beat outcome rewards because a right answer from wrong reasoning is still rewarded.
- Verifiable rewards are unhackable, and exist only in a narrow set of domains.
- Measure pass@$n$ alongside maj@$n$ to separate selection from capability.
