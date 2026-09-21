---
summary: Continuous control with off-policy sample efficiency — the reparameterisation that makes it differentiable, and the three fixes that made it stable.
prereqs: [deep-q-networks, actor-critic, autoencoders]
---

DQN cannot handle continuous actions because $\arg\max_a Q(s,a)$ requires enumeration
(lesson 6.06). PPO handles them but is on-policy, so it discards every sample after one
batch. This family gets both: continuous actions and replay.

## DDPG

Learn a deterministic policy $\mu_\theta(s)$ and a critic $Q_\phi(s,a)$. The critic is
trained exactly as in DQN, and the actor is trained by **ascending the critic**:

$$
\nabla_\theta J = \mathbb{E}\left[\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)}\;\nabla_\theta\mu_\theta(s)\right]
$$

::: key
The chain rule runs *through the critic*. Because actions are continuous, $Q$ is
differentiable with respect to $a$, so you can ask "which direction in action space increases
$Q$?" and move the actor that way.

This is what replaces the $\arg\max$: gradient ascent on a differentiable $Q$ instead of
enumeration over a discrete set. It is also why it needs a deterministic policy — the
gradient flows through a specific action, not through a sampling operation.
:::

```python
import torch

def ddpg_update(actor, critic, actor_target, critic_target,
                actor_opt, critic_opt, batch, gamma=0.99, tau=0.005):
    s, a, r, s2, d = batch

    # Critic: standard DQN target, with the target actor choosing the next action.
    with torch.no_grad():
        target_q = r + gamma * (1 - d) * critic_target(s2, actor_target(s2)).squeeze(-1)
    critic_loss = torch.nn.functional.mse_loss(critic(s, a).squeeze(-1), target_q)
    critic_opt.zero_grad(set_to_none=True); critic_loss.backward(); critic_opt.step()

    # Actor: maximise Q at the actor's own action. Freeze the critic so its
    # parameters are not updated by the actor's loss.
    for p in critic.parameters():
        p.requires_grad_(False)
    actor_loss = -critic(s, actor(s)).mean()
    actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); actor_opt.step()
    for p in critic.parameters():
        p.requires_grad_(True)

    # Polyak averaging for both targets.
    with torch.no_grad():
        for net, tgt in ((actor, actor_target), (critic, critic_target)):
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.mul_(1 - tau).add_(tau * p)
```

Exploration needs an explicit mechanism, since the policy is deterministic: Gaussian noise
added to the action, or Ornstein–Uhlenbeck noise for temporally correlated exploration.

DDPG works and is notoriously unstable. Its main failure is **Q-value overestimation** — the
actor finds and exploits errors in the critic, driving $Q$ upward with no corresponding
improvement in return.

## TD3

Three fixes, each targeting a specific instability:

**Twin critics.** Learn two $Q$ networks and take the minimum for the target:

$$
y = r + \gamma\min_{i=1,2}Q_{\phi_i'}(s', \tilde{a}')
$$

The minimum is a deliberate underestimate, which counteracts the maximisation bias of
lesson 6.05. Being pessimistic is safer than being optimistic here: an underestimated action
is simply not chosen, while an overestimated one is chased.

**Delayed policy updates.** Update the actor once per two critic updates. A policy trained
against a poor critic moves in a poor direction; letting the critic settle first reduces the
error the actor exploits.

**Target policy smoothing.** Add clipped noise to the target action:

```python
noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
next_action = (actor_target(s2) + noise).clamp(-max_action, max_action)
```

This smooths $Q$ over nearby actions, so the actor cannot exploit a narrow spike in the
critic's error surface. It is a regulariser on the value function, expressed as noise on the
action.

## SAC

SAC changes the objective itself, adding entropy as a term to be maximised:

$$
J(\pi) = \mathbb{E}\left[\sum_t r_t + \alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\right]
$$

::: key
Entropy is **in the objective**, not a bonus bolted onto the loss. The consequences are
substantial:

- Exploration is built in and automatic — no noise process to tune.
- The policy is stochastic, so it naturally hedges between similarly good actions rather
  than committing to one.
- It is far more robust to hyperparameters than DDPG or TD3, which is the practical reason
  it became the default for continuous control.

$\alpha$ trades reward against entropy, and modern SAC tunes it automatically to hit a
target entropy, removing the one remaining sensitive knob.
:::

The policy is a squashed Gaussian, using the **reparameterisation trick** from lesson 3.14
so gradients flow through the sample:

```python
import torch
import torch.nn as nn

class SquashedGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu, self.log_std = nn.Linear(hidden, act_dim), nn.Linear(hidden, act_dim)
        self.max_action = max_action

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)          # bound it, or it explodes
        std = log_std.exp()

        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()                             # reparameterised: differentiable
        a = torch.tanh(u)                                # squash into [-1, 1]

        # Change of variables for tanh: log p(a) = log p(u) - log|da/du|.
        logp = normal.log_prob(u).sum(-1)
        logp -= (2 * (torch.log(torch.tensor(2.0)) - u - torch.nn.functional.softplus(-2 * u))).sum(-1)
        return a * self.max_action, logp
```

::: warning
The tanh correction term is mandatory and easy to get wrong. Squashing changes the density
by the Jacobian of the transform, so the log-probability must be corrected by
$\log(1 - \tanh^2(u))$ summed over dimensions.

The form used above is the numerically stable rewriting of that expression — the direct
version underflows when $|u|$ is large, producing `-inf` and then `nan`.

Omit the correction entirely and the entropy term is wrong, which breaks the temperature
tuning and quietly removes the exploration that makes SAC work.
:::

## Choosing

| | On/off-policy | Actions | Sample efficiency | Stability |
|---|---|---|---|---|
| PPO | On | Both | Low | High |
| DDPG | Off | Continuous | High | Low |
| TD3 | Off | Continuous | High | Medium |
| SAC | Off | Continuous | High | High |

**SAC** for continuous control with expensive environment steps — robotics, simulation with
a slow physics engine. **PPO** when the simulator is cheap and parallelisable, where its
stability and simple implementation outweigh sample inefficiency. **TD3** when you want
deterministic actions at deployment.

::: exercise
Your SAC agent learns well and then collapses after 500k steps — return drops to near zero
and does not recover. What is happening?
:::

::: solution
**Three candidate causes. Check them in this order.**

**1. The entropy temperature has gone wrong.** Automatic $\alpha$ tuning adjusts it to hit a
target entropy. If the target is set too high — the default is often $-\dim(\mathcal{A})$ —
$\alpha$ grows to force more exploration, eventually dominating the reward term. The policy
becomes nearly uniform and return collapses.

*Check:* plot $\alpha$ and policy entropy over training. $\alpha$ rising steadily before the
collapse confirms it. *Fix:* lower the target entropy, or fix $\alpha$ at a value that was
working.

**2. Q-value divergence.** Even with twin critics, off-policy bootstrapping can diverge
(lesson 6.04's deadly triad). Once $Q$ is wrong, the actor optimises against a fiction.

*Check:* plot mean $Q$ against actual discounted return. $Q$ growing while return does not
is unambiguous. *Fix:* lower the critic learning rate, reduce $\tau$ for slower target
updates, or check for the terminal-mask bug from lesson 6.06's exercise.

**3. A stale replay buffer.** If the buffer holds 1M transitions and the policy has changed
substantially, most of the data reflects a policy very different from the current one. SAC
is off-policy and tolerates this, but not without limit — the critic is fitted to a
distribution the actor no longer visits.

*Check:* does collapse coincide with the buffer filling? *Fix:* a smaller buffer, or
periodically clear the oldest fraction.

**Also worth ruling out:**

- **`log_std` clamping.** If the bounds are too loose, the standard deviation can explode and
  the policy becomes effectively random. Confirm the clamp to $[-20, 2]$ is present.
- **The tanh log-probability correction.** If it is wrong or missing, the entropy estimate is
  wrong, which makes temperature tuning drive $\alpha$ in an arbitrary direction — this
  presents exactly as cause 1 and is a bug rather than a tuning problem.

**The fastest discriminator:** plot $\alpha$, policy entropy, mean $Q$ and episode return on
one chart. The collapse's cause is almost always visible in which curve moved first.
:::

## What to carry forward

- DDPG replaces the $\arg\max$ with gradient ascent through a differentiable critic.
- TD3's three fixes each target a specific instability; the twin-critic minimum is the main one.
- SAC puts entropy in the objective, which makes exploration automatic and tuning robust.
- The tanh log-probability correction is mandatory and must use the stable form.
- SAC for expensive environments, PPO for cheap parallel ones.
