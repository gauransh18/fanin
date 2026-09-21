---
summary: What is different when the policy is a language model, why GRPO drops the value function, and where RL genuinely helps.
prereqs: [rlhf-ppo, trpo-ppo, actor-critic, reasoning-test-time-compute]
---

Track 6 has developed RL for control problems. Language models are a strange MDP, and the
differences change which algorithms work.

## What is unusual

| Property | Control | Language model |
|---|---|---|
| Action space | ~10 discrete, or continuous | 128,000 discrete |
| Transitions | Stochastic | **Deterministic** — append the token |
| Episode length | 100–1,000 steps | 100–10,000 tokens |
| Reward | Dense, per step | **Terminal only** |
| Initialisation | Random | A capable pretrained model |
| Exploration | The central problem | Sampling from the policy |

Three of these matter enough to change the algorithm.

**Deterministic transitions** mean all stochasticity comes from the policy. There is no
environment noise, so return variance is entirely the policy's own — which makes variance
reduction more tractable than in control.

**Terminal reward** makes credit assignment the central difficulty. One scalar must be
attributed across hundreds of token decisions.

**A strong initialisation** changes the goal. In control you learn from scratch; here you
*adjust* a model that already works. The KL penalty (lesson 5.08) encodes that: the job is
to move a little, in a good direction.

## GRPO

PPO needs a value network — a fourth model in memory (lesson 5.08). Group Relative Policy
Optimisation removes it.

The idea: sample $G$ responses for the **same prompt** and use the group's mean reward as
the baseline:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1,\ldots,r_G)}{\text{std}(r_1,\ldots,r_G)}
$$

```python
import torch

def grpo_advantages(rewards, group_size):
    """rewards: (n_prompts * group_size,). Baseline is the group mean."""
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-8)
    return adv.view(-1)
```

::: key
This is a **valid baseline** by lesson 6.07's argument: the group mean depends on the
prompt, not on the specific action taken, so subtracting it leaves the gradient unbiased.

Three things it buys:

- **No value network.** One fewer model to hold, train and debug — a large saving when each
  is 7B+ parameters.
- **No value-function bias.** PPO's advantages inherit the critic's error (lesson 6.08); the
  group mean is an unbiased estimate of the prompt's expected reward.
- **A natural per-prompt normalisation.** Hard prompts and easy prompts are automatically put
  on the same scale, which PPO needs reward whitening to approximate.

The cost is $G$ generations per prompt instead of one — typically 4 to 16. Generation is the
expensive part of the loop, so this is a real cost, traded against a model's worth of memory
and the critic's instability.
:::

Each token in a response receives the same advantage, since the reward is terminal. That is
crude credit assignment, and it works because the KL penalty keeps updates small.

## Verifiable rewards

Lesson 5.10 introduced this and it is worth restating in RL terms: in maths, code and formal
proof, the reward is computed by a **program**, not a learned model.

```python
def verifiable_reward(problem, response):
    answer = extract_answer(response)
    if answer is None:
        return -1.0                          # failed to produce a parseable answer
    return 1.0 if answer == problem.gold else 0.0
```

Every failure mode from lesson 5.07 — length bias, formatting bias, sycophancy, reward
hacking — requires an exploitable reward model. A program that checks the answer is not
exploitable in that way.

This is why RL works so much better on reasoning than on general helpfulness. It is not that
the algorithm is different; the reward is exact.

::: warning
Verifiable rewards have their own hacking surface, and it is worth naming:

- **Test-case gaming** in code: a model that special-cases the visible tests rather than
  solving the problem.
- **Answer-format exploitation**: producing output that satisfies the extractor without
  reasoning — for instance, enumerating every plausible answer so the regex finds the right
  one.
- **Reward-only-on-format**: if a malformed answer scores $-1$ and a wrong answer scores $0$,
  the model learns formatting before correctness, which is fine, but check the reward
  structure is what you meant.

Hold out test cases, and read samples. "Unhackable" means harder to hack, not impossible.
:::

## Credit assignment

A single terminal reward for 500 tokens is a weak signal. Three approaches:

**Uniform.** Every token gets the same advantage. What GRPO does, and it works because the
KL constraint prevents large updates.

**Process rewards.** Score each reasoning step with a learned PRM (lesson 5.10). Much denser,
and it needs step-level annotation.

**Token-level from the value function.** PPO's GAE distributes credit across tokens using the
critic. More principled, and only as good as the critic — which for language models is hard
to train well, since the value of a partial response is genuinely uncertain.

## Practical notes

**Length and reward.** Longer responses have more tokens and, with uniform credit, more total
gradient. This creates pressure toward length independent of quality — the length bias of
lesson 5.07, arriving through a different path. Normalise by length, or penalise it
explicitly.

**Batch composition.** Mix prompts of varying difficulty. A batch where every response is
correct has zero advantage variance after group normalisation and contributes no gradient at
all. This is easy to miss and wastes a batch.

**Entropy collapse.** RL narrows the output distribution (lesson 5.08). Monitor per-token
entropy; a sharp fall means diversity is being lost, which shows up later as a model that
cannot be sampled usefully at temperature.

**Evaluate with something other than the reward.** Held-out benchmarks and human review. The
reward cannot detect its own exploitation, and this is the single most common process failure
in RL fine-tuning.

## Where RL actually helps

::: key
The honest summary of the evidence:

- **Verifiable domains** — maths, code, formal reasoning. RL with a program checker is
  clearly better than SFT, and it produces behaviours (backtracking, self-checking) that were
  never demonstrated.
- **Preference alignment** — helpfulness, tone, refusal behaviour. RLHF and DPO both work and
  the gains are real but modest; SFT on good data gets much of the way.
- **Knowledge** — RL does not add facts. Anything requiring information the base model does
  not have needs pretraining or retrieval, not RL.
- **Broad capability** — RL adjusts a pretrained model. It does not create capability that
  was not latent in the base.

The framing that has held up: **RL elicits and sharpens; pretraining creates.** Expecting RL
to add a capability the base model cannot sample at all is expecting the wrong thing —
because the policy only ever explores its own distribution (lesson 6.11).
:::

::: exercise
You run GRPO on maths problems with a verifiable reward. Accuracy improves from 40% to 65%,
then plateaus while response length triples. Diagnose.
:::

::: solution
**Two things are happening and they need separating.**

**The plateau.** With a binary reward and group-normalised advantages, a prompt where all $G$
samples are correct — or all wrong — has zero advantage variance and contributes **no
gradient**. As accuracy rises, more prompts fall into the all-correct bucket, so the
effective batch size shrinks. At 65% accuracy with $G=8$, a substantial fraction of prompts
are producing nothing.

*Check:* log the fraction of prompts with nonzero advantage variance. If it has fallen from
90% to 40%, that is the plateau.

*Fixes:* filter the training set to prompts near the model's current ability (a curriculum);
raise $G$ so partial credit appears more often; or add harder problems.

**The length growth.** Three candidate causes, distinguishable by measurement:

1. **Genuine improvement.** Longer chains of thought do help (lesson 5.10). Check whether
   accuracy *conditional on length* has improved, or whether only the length changed. If
   short responses are as accurate as they were, the length is not buying anything.

2. **Uniform credit assignment.** Every token in a correct response gets the same positive
   advantage, so a longer correct response receives more total gradient. This creates direct
   pressure toward length with no quality component. *Fix:* normalise the advantage by
   response length.

3. **Format exploitation.** The model may have learned that enumerating many candidate
   answers raises the chance the extractor finds the right one. *Check:* read 20 long
   responses. This is usually obvious on inspection, and it is a reward-specification bug —
   tighten the extractor to require a single clearly marked answer.

**What to do, in order:** read 20 samples first — it distinguishes cause 3 from 1 and 2
immediately and costs ten minutes. Then log advantage-variance coverage to confirm the
plateau's cause. Then normalise by length if cause 2 survives.

**One more thing to check:** per-token entropy. If it has collapsed, the model has become
near-deterministic, so all $G$ samples per prompt are nearly identical — which independently
produces zero advantage variance and looks exactly like the plateau. Raising sampling
temperature is the immediate remedy.
:::

## What to carry forward

- A language model is an MDP with deterministic transitions and terminal reward.
- GRPO uses the group mean as a baseline, removing the value network at the cost of $G$ generations.
- Verifiable rewards are exact, which is why RL works better on reasoning than on helpfulness.
- Watch for zero advantage variance — all-correct batches contribute nothing.
- RL elicits and sharpens; it does not create capability the base model lacks.
