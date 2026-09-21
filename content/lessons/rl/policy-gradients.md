---
summary: Optimising a policy directly, the log-derivative trick that makes it possible, and why the naive estimator has crippling variance.
prereqs: [value-functions-bellman, maximum-likelihood, chain-rule-backprop]
---

Value methods learn $Q$ and act greedily, which requires enumerating actions. Policy
gradient methods optimise $\pi_\theta$ directly — which is the only option for continuous
control, and the only option for a 128,000-token action space.

## The objective and its problem

$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[R(\tau)\right]
$$

The difficulty is that $\theta$ appears in the **distribution** being averaged over, not in
the function. You cannot move $\nabla_\theta$ inside a sampling operation.

## The log-derivative trick

$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)
$$

This identity — just the chain rule on $\log$ — converts a gradient of a probability into an
expectation:

$$
\nabla_\theta J = \int \nabla_\theta p_\theta(\tau) R(\tau)\,d\tau
= \mathbb{E}_{\tau\sim\pi_\theta}\left[\nabla_\theta\log p_\theta(\tau)\,R(\tau)\right]
$$

Now expand $\log p_\theta(\tau)$. The trajectory probability is

$$
p_\theta(\tau) = p(s_0)\prod_t \pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)
$$

and taking logs turns it into a sum. The transition terms do **not depend on $\theta$**, so
they vanish under differentiation:

$$
\nabla_\theta\log p_\theta(\tau) = \sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

::: key
**The environment dynamics disappear.** The policy gradient needs no model of $P$ — only the
ability to sample from the environment and to differentiate the policy's log-probability.

That is what makes policy gradients model-free, and it is the single most useful fact in
this lesson.
:::

## REINFORCE

$$
\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\right]
$$

```python
import torch

def reinforce_step(policy, optimizer, trajectory, gamma=0.99):
    states, actions, rewards = zip(*trajectory)

    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns = torch.tensor(list(reversed(returns)))
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # baseline

    logits = policy(torch.stack(states))
    logprobs = torch.log_softmax(logits, dim=-1)
    chosen = logprobs.gather(1, torch.tensor(actions)[:, None]).squeeze(1)

    loss = -(chosen * returns).sum()          # negative: we ascend J
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

The interpretation is direct: **increase the log-probability of actions that led to high
return, decrease it for actions that led to low return.** Weight each by how good the outcome
was.

Note that this is supervised learning's cross-entropy loss with the target being the action
taken and the weight being the return. That is not a coincidence — with $G_t = 1$ for every
action it *is* behaviour cloning.

## The variance problem

REINFORCE is unbiased and has variance so high it is often unusable.

Two sources. Returns depend on every random choice for the rest of the episode, so their
variance grows with horizon (lesson 6.04). And the gradient scales with the *absolute*
return: if all returns are between 990 and 1010, the estimator is dominated by a large
common term carrying no information about which action was better.

Three reductions, in increasing order of importance:

**Causality.** An action cannot affect rewards that already happened, so use the
reward-to-go $G_t$ rather than the full-episode return. This is already in the code above
and is strictly correct.

**Baselines.** Subtract any function of state:

$$
\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\left(G_t - b(s_t)\right)\right]
$$

This is **unbiased for any $b$ that does not depend on the action**, because

$$
\mathbb{E}_{a\sim\pi}\left[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\right]
= b(s)\nabla_\theta\sum_a \pi_\theta(a\mid s) = b(s)\nabla_\theta 1 = 0
$$

The best simple baseline is $V(s)$, which turns $G_t - V(s_t)$ into an estimate of the
advantage (lesson 6.02). Learning that baseline is what makes an actor–critic (lesson 6.08).

**Normalisation.** Standardise returns within a batch. Technically this introduces bias, and
it reduces variance enough that everyone does it.

## The gradient does not exist as a "loss"

::: warning
`loss = -(logprob * returns).sum()` is **not** a loss function in the usual sense. Its value
is meaningless — it does not measure anything, and watching it go up or down tells you
nothing about whether the policy is improving.

It is a **surrogate**: an expression whose gradient equals the policy gradient. This is a
frequent source of confusion, and it has a practical consequence — monitor **episode
return**, not the surrogate loss.

Note also that `returns` must be detached. If gradients flow through the return, you are
differentiating something with no meaning.
:::

## Continuous actions

For a continuous action space, output the parameters of a distribution:

```python
import torch
import torch.nn as nn

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim))
        # State-independent log-std is standard for continuous control:
        # simpler to optimise and usually works as well as a learned head.
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        return torch.distributions.Normal(self.net(obs), self.log_std.exp())

    def act(self, obs):
        dist = self(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1)      # sum over dimensions
```

`.sum(-1)` over action dimensions is required: independent dimensions means log-probabilities
add. Forgetting it produces a gradient scaled by the action dimension.

## On-policy only

The derivation assumed $\tau \sim \pi_\theta$ — the *current* policy. So every sample must
come from the policy being updated, and data becomes stale the moment you take a step.

This is expensive: you cannot reuse a replay buffer as DQN does. Importance sampling
corrects for a different behaviour policy:

$$
\nabla_\theta J = \mathbb{E}_{\tau\sim\pi_{\text{old}}}\left[\frac{\pi_\theta(\tau)}{\pi_{\text{old}}(\tau)}\nabla_\theta\log\pi_\theta(\tau)R(\tau)\right]
$$

but the ratio's variance explodes when the policies differ. Controlling exactly that ratio is
what PPO does, in lesson 6.09.

::: exercise
Your REINFORCE agent's episode return oscillates wildly and never improves. You are using
returns-to-go with no baseline. What is happening?
:::

::: solution
**Almost certainly variance, not a bug** — but rule out the bug first.

**Check the sign.** The gradient *ascends* $J$, so the loss must be
`-(logprob * returns)`. A sign error produces a policy that reliably gets worse, which is
distinguishable from oscillation: plot the return over many episodes and see whether there is
a downward trend.

**Check for detachment.** `returns` must be detached from the graph. If they are not, you are
differentiating through the return computation, which is meaningless and produces erratic
updates.

**Then the variance diagnosis.** Without a baseline, the gradient is weighted by the raw
return. If returns are, say, 480 to 520, then every action in a good episode is reinforced
by ~500 and every action in a slightly worse episode by ~490 — the signal distinguishing
them is 2% of the magnitude, buried in an estimator whose standard deviation is far larger.
The policy is being pushed hard in a nearly random direction each update.

**Confirm it in one line:** print `returns.std() / returns.mean().abs()`. Below about 0.1
means the informative variation is a small fraction of the scale, and a baseline is
essential.

**Fixes, in order:**

1. **Normalise returns within the batch.** `(returns - mean) / (std + 1e-8)`. One line, and
   it usually resolves the oscillation outright. This is the fix to try first.
2. **Learn a value baseline.** A network $V_\phi(s)$ trained on returns, used as
   $G_t - V_\phi(s_t)$. This is an actor–critic (lesson 6.08), and it is strictly better than
   batch normalisation because the baseline is state-dependent.
3. **Larger batches.** Accumulate several episodes before each update. Variance falls as
   $1/\sqrt{n}$ (lesson 1.11), so 16 episodes per update is 4× less noise than one.
4. **Lower the learning rate.** With a high-variance gradient, small steps are the only way
   to make progress reliably.
5. **Use GAE** (lesson 6.08), which trades a little bias for a large variance reduction and
   is what every modern implementation does.

**What the oscillation looks like once fixed:** return rising with visible noise, rather than
swinging without trend. If it is still flat after all five, the problem is exploration or
reward design, not variance — and lesson 6.01's degenerate-policy check is where to look.
:::

## What to carry forward

- The log-derivative trick turns a gradient of a distribution into an expectation you can sample.
- The environment dynamics cancel, which is what makes policy gradients model-free.
- Increase the log-probability of actions with high return, weighted by how high.
- The surrogate's value is meaningless — monitor episode return.
- Any state-dependent baseline is unbiased; $V(s)$ is the best simple choice.
