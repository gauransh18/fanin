---
summary: Learning the environment's dynamics to plan or to generate experience — the sample-efficiency gain and the compounding-error problem.
prereqs: [markov-decision-processes, dynamic-programming, autoencoders]
---

Model-free methods need millions of environment steps. Model-based methods learn $P$ and
$R$, then use them — either to plan, or to generate imagined experience — and can be orders
of magnitude more sample-efficient.

## The two uses

**Planning.** Use the model to look ahead at decision time. MCTS, model-predictive control.
No policy need be learned at all.

**Data generation.** Use the model to produce synthetic transitions, and train a model-free
algorithm on them. Dyna.

## Dyna

The simplest combination: interleave real experience with imagined experience.

```python
import random
from collections import defaultdict

def dyna_q(env, n_actions, episodes=100, planning_steps=50,
           alpha=0.1, gamma=0.95, eps=0.1):
    Q = defaultdict(float)
    model = {}                                     # (s, a) -> (r, s')
    for _ in range(episodes):
        state, done = env.reset(), False
        while not done:
            action = epsilon_greedy(Q, state, n_actions, eps)
            next_state, reward, done = env.step(action)

            # Real update.
            best = max(Q[(next_state, a)] for a in range(n_actions))
            Q[(state, action)] += alpha * (reward + gamma * best - Q[(state, action)])
            model[(state, action)] = (reward, next_state)

            # Imagined updates from the learned model.
            for _ in range(planning_steps):
                s, a = random.choice(list(model))
                r, s2 = model[(s, a)]
                best = max(Q[(s2, a2)] for a2 in range(n_actions))
                Q[(s, a)] += alpha * (r + gamma * best - Q[(s, a)])

            state = next_state
    return Q
```

With `planning_steps=50`, each real transition yields 51 updates. In a deterministic
environment the model is exact, so this is 50× sample efficiency for free.

## Learning dynamics

For continuous state spaces, learn a network. Two details matter more than the architecture.

::: key
**Predict the delta, not the next state.** $\hat{s}_{t+1} = s_t + f_\theta(s_t,a_t)$. States
change slowly, so the delta is small and centred near zero — far easier to learn than a
target that is nearly identical to the input. This alone is worth a large accuracy
improvement.

**Model uncertainty.** A deterministic model is overconfident where it has no data, and a
planner will exploit exactly those regions. Predict a distribution, and use an ensemble whose
disagreement marks where the model should not be trusted.
:::

```python
import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    """Predicts a Gaussian over the state delta."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim * 2))

    def forward(self, s, a):
        mu, log_std = self.net(torch.cat([s, a], -1)).chunk(2, -1)
        return mu, log_std.clamp(-10, 2).exp()

    def step(self, s, a):
        mu, std = self(s, a)
        return s + mu + std * torch.randn_like(mu)      # delta, plus noise
```

## Compounding error

::: warning
Model error compounds over a rollout. If one-step error is $\epsilon$ and the dynamics have
Lipschitz constant $L$, error after $k$ steps grows roughly as

$$
\epsilon\,\frac{L^k - 1}{L - 1}
$$

For $L > 1$ this is exponential. At $L = 1.1$ and $\epsilon = 0.01$, error after 50 steps is
about 1.2 — larger than the state values themselves.

The practical consequence: **long model rollouts are worthless**. Every method that works
uses short ones — typically 1 to 15 steps — branched from real states, and relies on a
model-free learner to handle the long horizon. This is exactly the bootstrapping argument of
lesson 6.04: use the model for a few steps, then trust a learned value.
:::

MBPO formalises this: generate short rollouts from states in the real replay buffer, add
them to an augmented buffer, and train SAC on the mixture.

::: check
What limits how far a learned dynamics model can be rolled out?

- [x] Compounding error — each step's prediction feeds the next, so small errors multiply and the imagined trajectory leaves the distribution the model was fitted on
  > Short rollouts from real states, model ensembles, and uncertainty-aware termination are the standard responses. The horizon, not the one-step accuracy, is what decides whether the model helps.
- [ ] Memory, since each imagined step must be stored
  > Imagined transitions are the same size as real ones and are cheap to store.
- [ ] The model cannot represent stochastic transitions
  > Probabilistic models and ensembles represent them routinely.
- [ ] Rollouts cannot be batched on a GPU
  > They batch extremely well, which is much of the appeal.
:::

## Planning with a model

**Model-predictive control.** At each step, optimise a short action sequence under the model,
execute the first action, then replan:

```python
import torch

def cem_planner(model, state, horizon=15, n_samples=500, elites=50, iters=5,
                act_dim=None, act_low=-1.0, act_high=1.0):
    """Cross-entropy method: sample sequences, keep the best, refit, repeat."""
    mu = torch.zeros(horizon, act_dim)
    sigma = torch.ones(horizon, act_dim) * 0.5

    for _ in range(iters):
        actions = (mu + sigma * torch.randn(n_samples, horizon, act_dim)).clamp(act_low, act_high)
        returns = torch.zeros(n_samples)
        s = state.expand(n_samples, -1).clone()
        for t in range(horizon):
            s = model.step(s, actions[:, t])
            returns += reward_fn(s, actions[:, t])

        best = actions[returns.topk(elites).indices]
        mu, sigma = best.mean(0), best.std(0)        # refit the sampling distribution

    return mu[0]                                     # execute only the first action
```

Replanning every step is what makes MPC robust: model error at step 10 does not matter,
because by the time you get there you have replanned from the real state.

**MCTS** builds a search tree, using UCB (lesson 6.11) to balance exploring promising
branches against unexplored ones. AlphaZero combines MCTS with learned policy and value
networks — the policy narrows the search, the value replaces rollouts to the end.

**MuZero** goes further: it learns dynamics in a **latent** space, with no reconstruction
loss at all. The model is trained only to predict reward, value and policy — so it learns
whatever representation supports planning, discarding visual detail that does not. This is
the key insight of the line of work: you do not need to model the environment, only the parts
that matter for decisions.

## World models

Dreamer learns a latent dynamics model and trains an actor–critic **entirely inside** it.
Real experience is used only to improve the model; the policy never sees the environment
directly during its own updates.

Because the latent space is compact, imagined rollouts are cheap — thousands can run in
parallel on a GPU. This is what makes it viable for continuous control from pixels, where
environment steps are expensive and model steps are not.

::: check
Dyna interleaves real experience with imagined experience from a learned model. What is it buying?

- [x] Sample efficiency — each real transition is used many times, through the model, instead of once
  > Model-free methods need millions of environment steps. When real steps are expensive, dangerous or slow, replaying them through a model is often orders of magnitude cheaper.
- [ ] Lower variance in the value estimates
  > Imagined transitions carry the model's own error, which can raise variance as well as bias.
- [ ] The ability to handle continuous actions
  > Continuity is orthogonal; Dyna as described is tabular.
- [ ] Guaranteed convergence to the optimal policy
  > Convergence now depends on the model being right, which is a weaker guarantee, not a stronger one.
:::

## When to use it

| Situation | Model-based? |
|---|---|
| Environment steps expensive (robotics) | Yes — sample efficiency is the point |
| Simulator cheap and parallel | No — model-free is simpler and ultimately better |
| Dynamics simple, reward complex | Yes |
| Dynamics chaotic or high-dimensional | Probably not — the model will be poor |
| Need to plan for a new goal at test time | Yes — a model generalises across goals |

That last row is underrated: a learned model can be replanned against a *different* reward
function without retraining, which a model-free policy cannot.

::: exercise
Your model-based agent achieves lower final performance than model-free SAC, despite being
10× more sample-efficient early. Why?
:::

::: solution
**Model bias caps asymptotic performance.** This is the standard and expected shape of the
result, not a bug.

**The mechanism.** The policy optimises against the *model*, not the environment. Where the
model is wrong, the policy exploits the error — finding actions that look excellent under the
model and are mediocre in reality. This is reward hacking (lesson 5.07) with the model
playing the role of the exploitable proxy.

Early in training the model is fitted to a small dataset and model-free methods have barely
any data, so model-based wins. Later, model-free sees millions of *real* transitions and is
limited only by optimisation, while model-based is limited by model accuracy, which plateaus.

**Confirm it:** evaluate the policy under the model and under the real environment. A large
and growing gap is the signature. Also check model prediction error as a function of rollout
length — if one-step error is small but 10-step error is large, compounding is doing the
damage.

**Fixes, in order:**

1. **Shorten the rollouts.** If you are using 50-step imagined rollouts, drop to 5. Error
   compounds exponentially (above), so this is usually the largest single improvement.
   MBPO's finding was that very short rollouts branched from real states work best.

2. **Ensemble the model and be pessimistic.** Train 5–7 dynamics models and, during planning,
   penalise states where they disagree. The policy is then discouraged from exactly the
   regions where the model is unreliable. This is the same idea as TD3's twin-critic minimum
   (lesson 6.10).

3. **Keep real data in the mix.** Train the policy on a mixture of real and imagined
   transitions rather than imagined alone. Real transitions anchor it to the true dynamics.

4. **Fine-tune model-free at the end.** Use the model-based agent as initialisation, then run
   SAC on real data. This gets the early sample efficiency *and* the asymptotic performance,
   and it is what you should do if you have the budget for both.

**The honest framing:** model-based RL buys sample efficiency and pays in asymptote. If
environment steps are cheap, that is a bad trade. If they cost a physical robot's time, it is
the only viable option — and the hybrid in fix 4 is usually the right answer.
:::

## What to carry forward

- Dyna trades compute for samples: many imagined updates per real transition.
- Predict state deltas, and model uncertainty with an ensemble.
- Error compounds exponentially, so rollouts must be short and branched from real states.
- MPC replans every step, which makes it robust to model error at long horizons.
- MuZero models only what matters for decisions, not the observation.
