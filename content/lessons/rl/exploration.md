---
summary: Why random exploration fails on hard problems, the principled alternatives, and what actually works at scale.
prereqs: [q-learning-sarsa, markov-decision-processes, bayes-and-independence]
---

An agent cannot learn about actions it never takes. Exploration is the problem of gathering
information you do not yet know you need, and $\epsilon$-greedy — the default everywhere —
is close to the worst possible approach.

## Why $\epsilon$-greedy fails

$\epsilon$-greedy takes a uniformly random action with probability $\epsilon$. Two
consequences:

**It is undirected.** Random actions are as likely to undo progress as to extend it. In a
corridor of length $n$ requiring $n$ correct actions in sequence, the probability of
reaching the end by random walk is exponentially small in $n$.

**It does not accumulate.** Each random action is independent, so the agent oscillates
around its current position rather than travelling anywhere new. This is a random walk, and
a random walk covers distance $\sqrt{t}$ in time $t$, not $t$.

The canonical demonstration is Montezuma's Revenge, where the first reward requires roughly
100 specific actions. DQN with $\epsilon$-greedy scored zero for years — not badly, exactly
zero, because it never once saw a reward.

## Optimism in the face of uncertainty

The principled framing: prefer actions you are **uncertain** about, because either they are
good (you gain reward) or they are bad (you gain information). Both outcomes are useful.

**UCB** makes this precise for bandits:

$$
a_t = \arg\max_a\left[\hat{Q}(a) + c\sqrt{\frac{\ln t}{N(a)}}\right]
$$

The bonus shrinks as an action is tried more often. UCB achieves logarithmic regret, which
is optimal — you cannot do better than $O(\ln t)$ on a bandit.

```python
import math

def ucb_select(values, counts, t, c=2.0):
    return max(range(len(values)), key=lambda a:
               float('inf') if counts[a] == 0
               else values[a] + c * math.sqrt(math.log(t) / counts[a]))
```

**Thompson sampling** takes a Bayesian route: maintain a posterior over each action's value,
sample from it, and act greedily on the sample. Actions with wide posteriors sometimes sample
high and get tried. It matches UCB's guarantees and is usually better in practice.

```python
import random

def thompson_bernoulli(successes, failures):
    """Beta posterior, conjugate to Bernoulli rewards (lesson 1.13)."""
    return max(range(len(successes)),
               key=lambda a: random.betavariate(successes[a] + 1, failures[a] + 1))
```

Both extend to full RL only approximately, because maintaining per-state uncertainty over a
large state space is the hard part.

::: check
DQN with $\epsilon$-greedy scored exactly zero on Montezuma's Revenge for years. Why exactly zero rather than merely low?

- [x] The first reward needs roughly 100 specific actions in sequence, and a random walk covers distance $\sqrt{t}$ rather than $t$
  > The agent never once saw a reward to learn from. Undirected exploration does not accumulate: each random action is independent, so the agent oscillates around where it already is. With no reward ever observed, there is nothing for any value-based method to fit.
- [ ] The reward was too small to register against the noise
  > A reward of any size would have been transformative. None was ever reached.
- [ ] The state space was too large for the network to represent
  > Capacity was not the limit — DQN handled comparably complex games.
- [ ] $\epsilon$ was annealed to zero too quickly
  > Even with a constant $\epsilon$ the random walk does not reach the first reward in any practical time.
:::

## Count-based exploration

Give a bonus for visiting rarely seen states:

$$
r^+(s,a) = r(s,a) + \frac{\beta}{\sqrt{N(s)}}
$$

In a large or continuous state space, exact counts are useless — every state is visited once.
**Pseudo-counts** estimate visitation density instead, using a density model or a hash:

```python
import numpy as np

class HashCountBonus:
    """SimHash: project states to a discrete code, count codes."""
    def __init__(self, obs_dim, k=32, beta=0.01):
        self.A = np.random.randn(k, obs_dim)
        self.counts, self.beta = {}, beta

    def bonus(self, obs):
        code = tuple(np.sign(self.A @ obs).astype(int))
        self.counts[code] = self.counts.get(code, 0) + 1
        return self.beta / np.sqrt(self.counts[code])
```

## Curiosity

Reward the agent for states where its own **prediction error** is high — the intuition being
that surprise indicates something not yet understood.

::: warning
**The noisy TV problem.** An agent with a prediction-error bonus, placed in front of a screen
showing random static, will watch it forever. The static is unpredictable, so prediction
error stays maximal, so the bonus never decays.

This is not a corner case — it is the generic failure of curiosity based on prediction
error. Any *aleatoric* uncertainty (irreducible randomness in the environment) produces
permanent reward.

The fix is to measure **epistemic** uncertainty — what the agent could learn — rather than
total uncertainty. Random Network Distillation does this: predict the output of a fixed
random network on the current state. Error falls as states are visited and is zero for
genuinely deterministic randomness, because a fixed network's output on a given input is
always the same.
:::

```python
import torch
import torch.nn as nn

class RND(nn.Module):
    """Prediction error against a frozen random target. Epistemic only."""
    def __init__(self, obs_dim, hidden=128, out=64):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, out))
        for p in self.target.parameters():
            p.requires_grad_(False)                    # frozen, never trained
        self.predictor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, out))

    def bonus(self, obs):
        with torch.no_grad():
            target = self.target(obs)
        return (self.predictor(obs) - target).pow(2).mean(-1)
```

## Parameter-space noise

Perturb the **weights** rather than the actions:

$$
\tilde{\theta} = \theta + \mathcal{N}(0, \sigma^2 I)
$$

The resulting policy is a consistent, different policy — so its behaviour is temporally
coherent rather than jittering step to step. This addresses $\epsilon$-greedy's second
failure directly.

NoisyNet makes $\sigma$ learnable per parameter, so the network decides how much exploration
each weight needs and reduces it as learning proceeds. It is one of Rainbow's six components
(lesson 6.06).

## What works at scale

::: key
The honest summary from large-scale practice: **massive parallelism plus a small amount of
undirected noise often beats sophisticated exploration.**

With 10,000 parallel environments, rare events occur somewhere. OpenAI Five and AlphaStar
used simple entropy bonuses at enormous scale rather than curiosity methods.

Sophisticated exploration matters when environment steps are expensive — robotics, real
systems — where you cannot buy coverage with parallelism. Choose based on which resource is
scarce.
:::

Two further approaches that have held up:

**Go-Explore** separates returning from exploring: archive interesting states, return to one
deterministically, then explore from there. This attacks the "cannot get back to the frontier"
problem that defeats random exploration, and it solved Montezuma's Revenge.

**Curriculum and reward shaping.** Often the most effective intervention is to make the
problem easier rather than the exploration better. Start with easier goals, or shape the
reward — carefully, per lesson 6.01's warning about degenerate policies.

::: check
UCB adds a bonus $c\sqrt{\ln t / N(a)}$ to each action's estimated value. What is the principle?

- [x] Optimism in the face of uncertainty
  > Prefer actions you are uncertain about, because either they are good (reward) or bad (information). Both outcomes are useful, which is what makes the bonus principled rather than a heuristic. UCB achieves logarithmic regret on a bandit, which is optimal.
- [ ] Prefer actions that have been tried most, since their estimates are reliable
  > The bonus shrinks as $N(a)$ grows, so it does the opposite.
- [ ] Add noise proportional to the reward scale, to escape local optima
  > That describes parameter-space or action-space noise, which is undirected.
- [ ] Decay exploration over time, as $\epsilon$-greedy schedules do
  > The $\ln t$ in the numerator means the bonus does not simply decay — it grows slowly for actions that stay untried.
:::

## Exploration in language models

RLHF (lesson 5.08) has an exploration problem that is usually not named as one. The policy
samples from its own distribution, so it only ever explores continuations it already
considers plausible. Temperature and the entropy implied by the KL penalty are the only
mechanisms.

This is one reason RLHF adjusts rather than transforms: it cannot discover a strategy the
pretrained model would never sample.

::: exercise
Your agent must open a door (needing a key from another room) to reach a reward. It never
succeeds. Rank four approaches.
:::

::: solution
**Diagnose first.** The task requires a *sequence*: find key, pick up key, cross to door,
open door, reach reward. If each stage needs ~20 correct actions, random exploration reaches
the reward with probability around $|\mathcal{A}|^{-100}$ — effectively zero. No amount of
$\epsilon$-greedy will find it.

**Ranked, best first:**

**1. Reward shaping / curriculum.** Give intermediate reward for picking up the key, and for
reaching the door while holding it. Or train on a curriculum: start the agent next to the
reward, then next to the door with the key, then progressively further back.

This is first because it is the only approach that reliably works and it is cheap. The
objection is that it requires domain knowledge — but you already have it: you know the key
matters. Use it. Apply lesson 6.01's check: confirm the shaped reward cannot be farmed (drop
and re-collect the key).

**2. Go-Explore.** Archive states by a discretised signature, return to a promising archived
state deterministically, explore from there. This directly solves the structural problem —
the agent can return to "holding the key, next to the door" instead of having to rediscover
it. It needs no reward engineering and is the strongest general method for this shape of
task.

**3. Count-based or RND bonus.** An intrinsic bonus for novel states pushes the agent toward
the unexplored, and "holding the key" is a genuinely novel state configuration, so it gets
rewarded. This works, but it explores in all directions equally — it will explore the
irrelevant corners of the map just as eagerly. Slower than 1 or 2, and it needs no domain
knowledge at all.

**4. More parallel environments.** With 10,000 parallel runs, the chance that *one* stumbles
through the sequence rises. At $10^{-30}$ per episode it is still hopeless; at $10^{-6}$ it
becomes viable. Worth trying only if the simulator is very cheap, and it is a brute-force
substitute for understanding the problem.

**What is missing from the list: $\epsilon$-greedy with more steps.** Running 100× longer
changes the probability linearly while the difficulty is exponential in sequence length. It
will not work.

**In practice:** combine 1 and 2. Shape the reward to make the first stage learnable, and
use an archive so the agent can reliably return to the frontier it has reached.
:::

## What to carry forward

- $\epsilon$-greedy is undirected and does not accumulate, so it fails on sequential tasks.
- Prefer uncertain actions: UCB and Thompson sampling are the principled forms.
- Prediction-error curiosity has the noisy-TV failure; RND measures epistemic uncertainty instead.
- Parameter-space noise gives temporally coherent exploration.
- At scale, parallelism often beats clever exploration; clever exploration matters when steps are expensive.
