---
summary: Learning a value function to reduce policy gradient variance, and GAE — the one knob that trades bias against variance explicitly.
prereqs: [policy-gradients, monte-carlo-td, value-functions-bellman]
---

Lesson 6.07 established that any state-dependent baseline is unbiased and that $V(s)$ is the
best simple choice. Learning that $V$ with a second network gives actor–critic, and it is
the structure behind every practical policy gradient method including RLHF.

## Two networks

- **Actor** $\pi_\theta(a\mid s)$ — the policy, updated by the policy gradient.
- **Critic** $V_\phi(s)$ — the value function, updated by regression on returns.

$$
\nabla_\theta J = \mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat{A}_t\right],
\qquad
\hat{A}_t = G_t - V_\phi(s_t)
$$

The critic's job is only to reduce variance. Its errors bias the gradient, but a biased
low-variance estimator beats an unbiased estimator you cannot see through — the tradeoff
from lesson 1.11, made concrete.

## Advantage estimators

The choice of $\hat{A}_t$ is the design decision:

| Estimator | Formula | Bias | Variance |
|---|---|---|---|
| Monte Carlo | $G_t - V(s_t)$ | None | High |
| TD(0) | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | High | Low |
| $n$-step | $\sum_{k<n}\gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$ | Medium | Medium |

Exactly the spectrum of lesson 6.04, applied to advantages.

## Generalised advantage estimation

GAE averages all $n$-step estimators with exponentially decaying weights, giving one
parameter that moves along the whole spectrum:

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},
\qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$\lambda = 0$ gives TD(0); $\lambda = 1$ gives Monte Carlo. Values around 0.95 are standard.

It computes in one backward pass:

```python
import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):
    """rewards, values, dones: (T,). values excludes the bootstrap; last_value
    is V(s_T). Returns advantages and value targets, both (T,)."""
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae      # the recursion
        advantages[t] = gae
    return advantages, advantages + values                      # returns = A + V
```

::: key
The recursion $\hat{A}_t = \delta_t + \gamma\lambda(1-d_t)\hat{A}_{t+1}$ is the whole
algorithm — it computes an infinite sum in $O(T)$ by working backwards.

Two details are load-bearing. The `next_nonterminal` factor resets the accumulation at
episode boundaries; without it, advantage from one episode leaks into the previous one.
And the value target is `advantages + values`, not the raw return — which keeps the critic
consistent with the same $\lambda$-weighted estimate the actor is using.
:::

## The loop

```python
import torch
import torch.nn.functional as F

def a2c_update(actor, critic, opt, batch, gamma=0.99, lam=0.95,
               value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
    states, actions, rewards, dones, last_value = batch

    with torch.no_grad():
        values = critic(states).squeeze(-1)
    advantages, returns = compute_gae(rewards, values, dones, gamma, lam, last_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dist = actor(states)
    logprobs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    policy_loss = -(logprobs * advantages).mean()
    value_loss = F.mse_loss(critic(states).squeeze(-1), returns)

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
    opt.step()
```

**The entropy bonus** deserves attention. Subtracting $-\beta H(\pi)$ from the loss rewards
high entropy, which keeps the policy stochastic and prevents premature collapse onto a
suboptimal action. Without it, policy gradient methods routinely converge to a deterministic
policy early and stop exploring — the same mode collapse as lesson 5.08's RLHF, for the same
reason.

Typical $\beta$ is 0.01 for discrete actions and near zero for continuous ones, where the
Gaussian's learned standard deviation already controls exploration.

::: check
The critic's errors bias the policy gradient. Why is that acceptable?

- [x] A biased low-variance estimator beats an unbiased one you cannot see through — the bias–variance tradeoff of lesson 1.11 made concrete
  > Monte Carlo advantages are unbiased and too noisy to learn from; TD(0) is biased and quiet. GAE's $\lambda$ is the dial between them.
- [ ] The bias cancels over many updates
  > It does not cancel; it systematically points the gradient slightly wrong. It is tolerated, not eliminated.
- [ ] The critic converges before the actor, so the bias is transient
  > The two train together and the critic chases a moving target, so its error never fully disappears.
- [ ] Policy gradients are insensitive to the advantage's scale
  > They are quite sensitive to it, which is why advantage normalisation is standard.
:::

## Shared or separate networks

Sharing a trunk between actor and critic saves compute and lets the value task regularise the
representation. It also means the two losses compete for the same parameters, and the value
loss — typically much larger in magnitude — can dominate.

If you share, tune `value_coef` carefully and watch both losses separately. For anything
other than pixel input, separate networks are simpler and usually at least as good.

## Diagnosing a critic

```python
def explained_variance(predicted, actual):
    """1.0 = perfect, 0.0 = no better than predicting the mean, <0 = worse."""
    var = actual.var()
    return 1 - (actual - predicted).var() / (var + 1e-8)
```

::: warning
**Explained variance is the diagnostic to watch.** Near zero means the critic is no better
than a constant, so your advantages are essentially raw returns and you have the variance
problem of lesson 6.07 back.

Healthy training shows explained variance rising into 0.8–0.95. If it stays near zero:

- The value learning rate may be too low, or the critic too small.
- The reward scale may be extreme — normalise returns.
- The environment may be genuinely unpredictable, in which case no critic can help and you
  need a larger $\lambda$ to rely on real rewards instead.
:::

## The family

- **A2C** — synchronous, several environments stepped in parallel, one update from the batch.
- **A3C** — asynchronous workers with a shared parameter server. Historically important,
  largely replaced by A2C, which is simpler and performs as well on a GPU.
- **PPO** (lesson 6.09) — A2C plus a clipped objective allowing multiple epochs per batch.
- **SAC** (lesson 6.10) — off-policy actor–critic with entropy in the objective itself.

::: check
GAE's $\lambda$ moves along a spectrum. What are the two ends?

- [x] $\lambda = 0$ gives TD(0) — biased and low variance; $\lambda = 1$ gives Monte Carlo — unbiased and high variance
  > It is an exponentially weighted average of all $n$-step estimators, so one parameter dials the whole bias–variance spectrum of lesson 6.04. Typical values sit around 0.95.
- [ ] $\lambda = 0$ gives Monte Carlo and $\lambda = 1$ gives TD(0)
  > Reversed: $\lambda = 0$ keeps only the first TD error.
- [ ] $\lambda$ controls the discount factor, not the estimator
  > $\gamma$ and $\lambda$ appear together as $(\gamma\lambda)^l$ and do different jobs — one sets the horizon, the other the bootstrapping depth.
- [ ] Both ends give the same estimator, differing only in computational cost
  > They differ in bias and variance, which is the entire point of having the knob.
:::

## Where it reappears

RLHF (lesson 5.08) is an actor–critic:

| Actor–critic | RLHF |
|---|---|
| Actor $\pi_\theta$ | The language model |
| Critic $V_\phi$ | The value head |
| Reward | Reward model score at the last token |
| GAE | Same code, $\gamma = 1$ |
| Entropy bonus | Replaced by the KL penalty |

The KL penalty plays the entropy bonus's role: both prevent the policy from collapsing, one
by rewarding uncertainty and the other by anchoring to a reference distribution.

::: exercise
Your actor–critic has explained variance of 0.02 after 100k steps. The policy is not
improving. Diagnose.
:::

::: solution
**The critic has learned nothing, so the advantages are effectively raw returns.** That puts
you back in REINFORCE's variance regime (lesson 6.07), which explains the lack of progress.

**Check the mechanical causes first, in this order:**

1. **Is the critic receiving gradients at all?** Print `critic.parameters()[0].grad.norm()`
   after a backward. `None` or zero means the value loss is detached from the critic — a
   common bug is computing `values` inside `torch.no_grad()` and then using that same tensor
   for the value loss, so nothing flows back. In the code above, note that `values` for GAE
   is under `no_grad` while the value loss recomputes `critic(states)` *outside* it. That
   distinction is the bug.

2. **Is `value_coef` too small?** If the policy loss is order 0.01 and the value loss is
   order 100, a coefficient of 0.5 is fine — but if you set it to 0.001 to stop the value
   loss dominating a shared trunk, the critic barely trains.

3. **Are returns on a sane scale?** If rewards are in the thousands, the value targets are
   enormous and MSE gradients are correspondingly large, which forces a tiny learning rate.
   Normalise returns, or scale rewards.

**If the mechanics are correct:**

4. **Is the environment predictable at all?** Explained variance measures how much of the
   return variance the state explains. In a highly stochastic environment, even a perfect
   critic has low explained variance — there genuinely is no signal. Test by checking
   whether returns from the *same* state vary widely. If so, the critic is not broken and the
   answer is to raise $\lambda$ toward 1, relying on real rewards rather than a value that
   cannot be learned.

5. **Is the critic large enough?** A two-layer MLP on pixel input will not learn a value
   function. Match its capacity to the observation complexity.

**Also worth checking:** that `dones` is correct in the GAE computation. If episode
boundaries are not marked, advantages leak across episodes and the value targets are wrong
in a way that caps explained variance regardless of critic capacity.

**Order of operations:** check gradients reach the critic (one line), then check the reward
scale (one line), then check whether the environment is predictable. Those three cover almost
every real case.
:::

## What to carry forward

- The critic exists to reduce variance; its bias is an accepted cost.
- GAE's $\lambda$ moves continuously between TD(0) and Monte Carlo.
- Reset the GAE accumulation at episode boundaries, and set value targets to $\hat{A} + V$.
- The entropy bonus prevents premature collapse onto a deterministic policy.
- Watch explained variance — near zero means the critic is doing nothing.
