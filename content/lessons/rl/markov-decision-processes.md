---
summary: The formalism every RL algorithm optimises, what the Markov property actually assumes, and how a language model fits the frame.
prereqs: [probability-spaces, expectation-variance-concentration]
---

Reinforcement learning studies sequential decision-making under uncertainty. The MDP is the
formal object; every algorithm in this track is a different way of solving one.

## The five-tuple

An MDP is $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$ — states.
- $\mathcal{A}$ — actions.
- $P(s' \mid s, a)$ — transition probabilities.
- $R(s, a, s')$ — reward.
- $\gamma \in [0,1]$ — discount factor.

The loop: observe $s_t$, take $a_t$, receive $r_t$, land in $s_{t+1}$. Repeat.

## The Markov property

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1}\mid s_t, a_t)
$$

The future depends on the past only through the present state. This is the assumption that
makes everything tractable — without it, a policy would need the entire history.

::: key
**The Markov property is a property of your state representation, not of the world.**

A single video frame is not Markov: you cannot tell velocity from one image. Stack four
frames and it is. Chess with only the board position is nearly Markov but misses castling
rights and the fifty-move counter — include those and it is.

When an RL algorithm fails, "is my state actually Markov?" is one of the first things to
check. The usual fix is to add the missing information to the state, not to change the
algorithm.
:::

When you genuinely cannot observe the full state, you have a **POMDP**, and the standard
response is to maintain a belief state or feed a history to a recurrent or attention-based
policy — which is exactly what a language model does.

::: check
An RL agent trained on single video frames cannot learn to catch a ball. What is the most likely diagnosis?

- [x] The state is not Markov — velocity is not recoverable from one image, so stacking several frames is the fix
  > The Markov property is a property of your state representation, not of the world. When an algorithm fails, "is my state actually Markov?" is one of the first things to check, and the usual fix adds the missing information to the state rather than changing the algorithm.
- [ ] The discount factor is too low
  > A low $\gamma$ shortens the horizon, which would not stop the agent seeing where the ball is going *now*.
- [ ] The reward is too sparse
  > Sparse reward makes learning slow. It does not make the task unlearnable in the way a missing state variable does.
- [ ] The action space is too large
  > Catching a ball needs few actions. The information deficit is on the observation side.
:::

## Returns and discounting

The objective is cumulative reward:

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

$\gamma$ does three things:

- **Guarantees convergence.** For bounded rewards, $\sum \gamma^k r$ converges when
  $\gamma < 1$. Without it the sum can diverge for infinite-horizon problems.
- **Sets the effective horizon**, roughly $1/(1-\gamma)$ steps. At $\gamma = 0.99$ that is
  100 steps; at $0.9$ it is 10.
- **Encodes preference for sooner rewards**, which is sometimes genuinely wanted.

| $\gamma$ | Horizon | Typical use |
|---|---|---|
| 0 | 1 step | Contextual bandits |
| 0.9 | ~10 | Short tasks |
| 0.99 | ~100 | Standard control |
| 1.0 | undiscounted | Finite episodes — including RLHF |

RLHF uses $\gamma = 1$ (lesson 5.08): a response is a finite episode and a token at position
400 matters as much as one at position 4.

## Policies

A policy maps states to actions. Deterministic, $a = \pi(s)$, or stochastic,
$\pi(a\mid s)$.

Stochastic policies are usually preferred, for three reasons: they provide exploration
without a separate mechanism; they are differentiable, which policy gradients need
(lesson 6.07); and in partially observable settings the optimal policy can genuinely be
stochastic.

## A language model as an MDP

The mapping is exact, and worth internalising because it is what makes track 6 relevant to
track 5:

| MDP | Language model |
|---|---|
| State $s_t$ | The prompt plus tokens generated so far |
| Action $a_t$ | The next token |
| Transition | Deterministic — append the token |
| Reward | 0 per token; the reward model's score at the end |
| $\gamma$ | 1 |
| Policy $\pi_\theta(a\mid s)$ | The model's next-token distribution |

Two features make this MDP unusual. **Transitions are deterministic** — there is no
environment stochasticity, only the policy's own randomness. And the **reward is terminal
and sparse**, which is why credit assignment is the central difficulty in RLHF.

The action space is enormous — 128,000 actions per step — which rules out any method
enumerating actions and forces policy gradient approaches.

::: check
What does the discount factor $\gamma$ do, beyond expressing a preference for sooner rewards?

- [x] It keeps the return finite on continuing tasks, where an undiscounted infinite sum need not converge
  > It also sets an effective horizon of roughly $1/(1-\gamma)$ steps — $\gamma = 0.99$ means about 100 steps matter. Treating it as a mere preference misses that it is also a mathematical necessity and a horizon knob.
- [ ] It normalises rewards to $[0,1]$
  > It does not rescale individual rewards at all.
- [ ] It compensates for the Markov assumption being approximate
  > Discounting and state representation are independent concerns.
- [ ] It guarantees the optimal policy is deterministic
  > A finite MDP has a deterministic optimal policy for any valid $\gamma$.
:::

## A minimal environment

```python
import numpy as np

class GridWorld:
    """4x4 grid. Start top-left, goal bottom-right, -1 per step."""
    def __init__(self, size=4):
        self.size = size
        self.goal = (size - 1, size - 1)

    def reset(self):
        self.pos = (0, 0)
        return self.pos

    def step(self, action):
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        r = min(max(self.pos[0] + dr, 0), self.size - 1)
        c = min(max(self.pos[1] + dc, 0), self.size - 1)
        self.pos = (r, c)
        done = self.pos == self.goal
        return self.pos, (0.0 if done else -1.0), done

env = GridWorld()
state = env.reset()
for _ in range(20):
    state, reward, done = env.step(np.random.randint(4))
    if done:
        break
```

The $-1$ per step encodes "reach the goal quickly" without specifying a path. That is the
appeal of RL: you specify the objective, not the solution.

## Reward design is the hard part

::: warning
**Specifying the reward is where RL projects fail.** The agent optimises exactly what you
wrote, which is rarely what you meant.

Classic cases: a boat-racing agent that learned to circle a lap-point pickup forever instead
of finishing; a cleaning robot that learned to knock objects over so it could re-clean them.
Both maximised the stated reward.

Two rules that help. **Reward outcomes, not behaviours** — rewarding a sub-behaviour you
believe is useful invites optimising the proxy rather than the goal. And **check the
degenerate policy**: before running anything, ask what the laziest possible way to maximise
your reward would be. If you can think of one, the agent will find it.

This is the same Goodhart failure as reward hacking in lesson 5.07, in its original setting.
:::

## The taxonomy

Every algorithm in this track sits somewhere on three axes:

- **Model-free or model-based** — does it learn $P$? Lesson 6.12.
- **Value-based or policy-based** — does it learn $Q$ and act greedily, or optimise $\pi$
  directly? Lessons 6.05 and 6.07.
- **On-policy or off-policy** — does it learn from its own current behaviour, or from data
  produced by another policy? Lessons 6.09 and 6.10.

::: exercise
You train an agent to play a racing game with reward = distance travelled. It learns to
drive in tight circles. What went wrong, and give two fixes.
:::

::: solution
**Distance travelled is maximised by driving continuously, and a circle is the cheapest way
to travel continuously.** The reward never mentioned the track, the direction, or the finish
line — so the agent found the policy that maximises what you actually wrote. It is behaving
correctly.

**Fix 1: reward progress along the track, not raw distance.** Project the car's position
onto the track centreline and reward the *increase* in that coordinate:

```python
reward = track_progress(pos) - track_progress(prev_pos)
```

Circling now yields zero: the projection returns to where it started. This is the right
shape of fix because it rewards the outcome you want rather than a correlate of it.

**Fix 2: reward completion, with a time penalty.** Give a large reward for crossing the
finish line and $-1$ per timestep. Circling accrues unbounded negative reward and never
reaches the bonus.

This is cleaner in specification and much harder to learn: the reward is sparse, so random
exploration may never finish a lap and the agent sees no positive signal at all. Sparse
rewards are the honest specification and the hard learning problem — lesson 6.11 covers the
exploration techniques that address it.

**In practice, combine them:** dense progress reward for learnability, plus a completion
bonus so the agent's objective is anchored to the real goal. Check the degenerate policy
again after combining — a common mistake is a progress reward that can be farmed by
oscillating back and forth across a checkpoint, which is why the increment must be
*monotonic* progress, not absolute position.

**The general procedure:** write the reward, then spend five minutes deliberately trying to
think of a degenerate policy that maximises it. If you find one, the agent will. This costs
nothing and catches most reward-design failures before they cost GPU time.
:::

## What to carry forward

- An MDP is states, actions, transitions, reward and discount; $\gamma$ sets the horizon.
- The Markov property is a property of your state representation — fix the state, not the algorithm.
- A language model is an MDP with deterministic transitions and a terminal reward.
- Reward outcomes, not behaviours, and always check the degenerate policy first.
