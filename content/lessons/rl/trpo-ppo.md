---
summary: Why a policy gradient step can destroy a policy, the trust region that prevents it, and the clipped objective that made it practical.
prereqs: [actor-critic, policy-gradients, hessians-and-curvature]
---

Policy gradients have a failure mode supervised learning does not: a step that is too large
produces a worse policy, which then collects worse data, which makes the next step worse
still. Supervised learning recovers from a bad step because the data is fixed. RL does not.

## The problem, precisely

$$
\theta \leftarrow \theta + \alpha\nabla_\theta J
$$

The gradient is a local statement. The learning rate that is safe depends on the local
curvature (lesson 1.09), and in RL that curvature changes as the policy changes.

Worse, the step size in **parameter** space says nothing about the step size in **policy**
space. A small change to $\theta$ can produce a large change to $\pi_\theta$ — especially
when the policy is near-deterministic, where a small logit shift flips the argmax.

::: key
The right quantity to limit is the change in the policy's **distribution**, not in its
parameters. KL divergence is the natural measure:

$$
D_{\text{KL}}\!\left(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta\right) \le \delta
$$

This is a constraint in distribution space, so it is invariant to how the policy is
parameterised. That invariance is the whole idea.
:::

## TRPO

Maximise a surrogate objective subject to a KL constraint:

$$
\max_\theta \; \mathbb{E}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}\hat{A}\right]
\quad\text{s.t.}\quad \mathbb{E}\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}}\|\pi_\theta)\right]\le\delta
$$

The ratio is importance sampling (lesson 6.07), correcting for the fact that the data came
from $\pi_{\theta_{\text{old}}}$. This is what permits **multiple gradient steps on one
batch** — without it, the first step invalidates the data.

TRPO solves this with a second-order method: approximate the KL constraint by its quadratic
expansion, whose Hessian is the Fisher information matrix, and solve with conjugate gradient
plus a line search.

It has a monotonic improvement guarantee, and it is complicated: conjugate gradient,
Hessian-vector products (lesson 1.09), and a backtracking line search, none of which
compose well with anything else.

## PPO

PPO gets most of TRPO's benefit with a first-order method. Write $r_t(\theta)$ for the
probability ratio, and clip it:

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\Big(r_t\hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\Big)\right]
$$

```python
import torch

def ppo_loss(logprobs, old_logprobs, advantages, clip=0.2):
    ratio = torch.exp(logprobs - old_logprobs)          # exp of a difference: stable
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -torch.min(unclipped, clipped).mean()        # negative: ascend
```

The `min` is the part worth understanding.

- **When $\hat{A} > 0$** (a good action), the objective is capped at $(1+\epsilon)\hat{A}$.
  Increasing the probability beyond that yields no further gain, so there is no incentive to
  push hard.
- **When $\hat{A} < 0$** (a bad action), the objective is capped at $(1-\epsilon)\hat{A}$ —
  the ratio cannot be pushed below $1-\epsilon$ for credit.

::: warning
The `min` makes the objective a **pessimistic bound**. Taking the minimum of clipped and
unclipped means the clipping only ever *removes* incentive to move — it never creates
incentive to move back.

This matters: if $r_t$ has already moved outside the range, the gradient is zero rather than
negative. PPO does not pull the policy back into the trust region; it just stops pushing.
A common misunderstanding is that clipping enforces a constraint, and it does not — it
removes the objective's gradient beyond a threshold.
:::

## The full algorithm

```python
import torch
import torch.nn.functional as F

def ppo_update(actor, critic, opt, batch, epochs=4, minibatch=64,
               clip=0.2, value_coef=0.5, entropy_coef=0.01,
               max_grad_norm=0.5, target_kl=0.015):
    states, actions, old_logprobs, advantages, returns = batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(states)
    for epoch in range(epochs):
        for idx in torch.randperm(n).split(minibatch):
            dist = actor(states[idx])
            logprobs = dist.log_prob(actions[idx])
            ratio = torch.exp(logprobs - old_logprobs[idx])

            adv = advantages[idx]
            policy_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()

            value_loss = F.mse_loss(critic(states[idx]).squeeze(-1), returns[idx])
            entropy = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt.step()

        # Early stop: if the policy has moved too far, stop reusing this batch.
        with torch.no_grad():
            kl = (old_logprobs - actor(states).log_prob(actions)).mean()
        if kl > 1.5 * target_kl:
            break
```

The early-stopping check is what actually enforces a trust region, and it is frequently
omitted from implementations. Since clipping only zeroes the gradient rather than pulling
back, the KL check is the mechanism that stops a batch being over-exploited.

## What to monitor

```python
metrics = {
    'clip_fraction': ((ratio - 1).abs() > clip).float().mean(),
    'approx_kl':     ((ratio - 1) - (ratio).log()).mean(),   # the k3 estimator
    'entropy':       dist.entropy().mean(),
    'explained_var': explained_variance(values, returns),
    'episode_return': ...,                                    # the only real metric
}
```

| Signal | Healthy | What it means otherwise |
|---|---|---|
| `clip_fraction` | 0.05–0.2 | Above 0.3: steps too large, lower the learning rate |
| `approx_kl` | 0.005–0.02 | Above 0.05: policy moving too fast |
| `entropy` | slowly falling | Sharp drop: premature collapse |
| `explained_var` | rising to 0.8+ | Near 0: the critic is useless (lesson 6.08) |

## Implementation details that matter more than the algorithm

::: key
Reproducing PPO's published results depends on a list of details that are not in the paper.
The most consequential:

- **Advantage normalisation** per minibatch.
- **Observation normalisation** with a running mean and variance.
- **Reward scaling** by a running estimate of the return's standard deviation.
- **Orthogonal initialisation**, with the policy's final layer scaled by 0.01 so the initial
  policy is near-uniform.
- **Learning-rate annealing** over training.
- **Value-function clipping**, analogous to the policy clip.
- **Gradient clipping** at 0.5.

Ablation studies find these collectively account for more performance variation than the
choice between PPO and TRPO. This is uncomfortable and it is what the evidence says —
lesson 2.15's point about reporting variance applies with force here.
:::

## Where you have already seen it

RLHF (lesson 5.08) is PPO with a language model as the actor. The mapping is exact: the
ratio is over token log-probabilities, the advantage comes from GAE over the reward model's
terminal score, and the KL penalty against the SFT model plays an additional role that the
clip alone does not cover.

::: exercise
Your PPO run has `clip_fraction` at 0.45 and `approx_kl` at 0.08. Episode return is
oscillating. What is happening and what do you change?
:::

::: solution
**Both numbers are 3–5× their healthy range. The policy is moving far too fast per batch.**

A clip fraction of 0.45 means nearly half the sampled actions have ratios outside
$[0.8, 1.2]$ — so for half your data, the gradient has been zeroed by clipping and those
samples contribute nothing. You are paying full cost for the rollout and using half of it.

An approximate KL of 0.08 means the policy after the update is substantially different from
the one that collected the data. The importance-sampling correction that justifies reusing
the batch is only valid for small ratios, so by the later epochs you are optimising a
surrogate that no longer approximates the objective.

**Changes, in order:**

1. **Lower the learning rate**, by 3–10×. This is the direct cause and the first thing to
   try. Re-check both metrics; you want clip fraction near 0.1 and KL near 0.01.

2. **Reduce the number of epochs per batch**, from 4 to 2 or 1. Each epoch pushes the policy
   further from the data-collecting policy, and the last epochs are where the ratio blows
   out.

3. **Add the KL early-stopping check** if it is not present. Break out of the epoch loop when
   `approx_kl > 1.5 * target_kl`. This bounds the damage automatically rather than requiring
   you to tune epochs per environment.

4. **Check advantage normalisation.** If advantages are not normalised, their scale varies
   between batches and the effective step size varies with it. This produces exactly the
   oscillation you describe, and it is a one-line fix.

**What not to do: lower the clip range.** It is tempting — a smaller $\epsilon$ clips more —
but clipping only zeroes gradients. Tightening it means more of your data contributes
nothing, wasting rollouts without addressing why the ratios are large. The learning rate is
the lever that controls how far the policy moves.

**How to verify the fix worked:** clip fraction in 0.05–0.2, approximate KL in 0.005–0.02,
and episode return rising with noise rather than swinging. If return is still flat with
healthy metrics, the problem has moved elsewhere — check explained variance (lesson 6.08)
and then the reward design (lesson 6.01).
:::

## What to carry forward

- Limit the change in policy distribution, not in parameters — KL is parameterisation-invariant.
- The importance ratio is what permits multiple epochs on one batch.
- Clipping removes incentive to move further; it does not pull back, so keep the KL early stop.
- Watch clip fraction and approximate KL; they diagnose step size directly.
- The implementation details matter as much as the algorithm.
