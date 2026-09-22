---
summary: Learning a policy from a fixed dataset with no interaction — the distribution shift that makes it hard, and the two families of fix.
prereqs: [q-learning-sarsa, deep-q-networks, ddpg-td3-sac]
---

Offline RL learns from a logged dataset with **no environment interaction**. This is the
setting for healthcare, industrial control, recommendation and autonomous driving — anywhere
exploration is dangerous, expensive or prohibited.

It is also much harder than it looks, for one specific reason.

## Why off-policy algorithms do not just work

Q-learning is off-policy (lesson 6.05), so it should learn from any data. In practice,
running DQN or SAC on a fixed dataset usually produces a policy far worse than the one that
generated it.

::: key
**The problem is extrapolation error on out-of-distribution actions.**

The Bellman target is $r + \gamma\max_{a'}Q(s',a')$. The max ranges over **all** actions,
including ones never taken in the dataset. For those, $Q$ is an extrapolation of a network
fitted elsewhere — and neural networks extrapolate arbitrarily.

If $Q$ happens to be high for an unseen action, that value enters the target, raising $Q$ at
the previous state. There is no corrective feedback, because the agent cannot try the action
and find out it is bad. The error propagates and amplifies.

Online, this self-corrects: the agent tries the overestimated action, sees a low reward, and
the estimate falls. Offline, nothing corrects it. **Interaction is the error-correction
mechanism**, and removing it is what makes the setting hard.
:::

The symptom is unmistakable: $Q$-values grow without bound while actual policy performance
collapses.

## Policy constraint methods

Keep the learned policy close to the behaviour policy that generated the data.

**BCQ** restricts the action space to actions a generative model considers plausible under
the data. **BEAR** constrains the maximum mean discrepancy between the learned and behaviour
policies. **TD3+BC** is the simplest and works remarkably well — add a behaviour-cloning term
to TD3's actor loss:

```python
import torch
import torch.nn.functional as F

def td3_bc_actor_loss(actor, critic, states, dataset_actions, alpha=2.5):
    pi = actor(states)
    q = critic(states, pi)
    # Normalise by |Q| so the two terms stay comparable across reward scales.
    lam = alpha / q.abs().mean().detach()
    return -lam * q.mean() + F.mse_loss(pi, dataset_actions)
```

The BC term anchors the policy to actions that appear in the data, so the critic is only ever
queried where it has support. The $\lambda$ normalisation is the detail that makes it robust
— without it, the balance between the two terms depends on the environment's reward scale.

## Conservative value methods

Instead of constraining the policy, make the critic **pessimistic** about unseen actions.

**CQL** adds a term that pushes down $Q$ for actions the policy would take and pushes up $Q$
for actions in the dataset:

$$
\mathcal{L}_{\text{CQL}} = \alpha\left(\mathbb{E}_{s\sim\mathcal{D}, a\sim\pi}[Q(s,a)] - \mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)]\right) + \mathcal{L}_{\text{TD}}
$$

```python
import torch

def cql_penalty(critic, states, dataset_actions, policy, n_samples=10, alpha=5.0):
    B = states.size(0)
    # Sample actions the policy might take, plus uniform actions.
    sampled = policy.sample(states.repeat_interleave(n_samples, 0))
    q_sampled = critic(states.repeat_interleave(n_samples, 0), sampled).view(B, n_samples)

    # logsumexp is a soft maximum over sampled actions (lesson 2.07).
    q_ood = torch.logsumexp(q_sampled, dim=1).mean()
    q_data = critic(states, dataset_actions).mean()
    return alpha * (q_ood - q_data)
```

The result is a **lower bound** on the true value function. A pessimistic $Q$ means the
policy is never attracted to an action whose value is uncertain — safe by construction, and
conservative to a fault if $\alpha$ is too large.

**IQL** avoids querying unseen actions entirely. It fits an expectile regression of $V$
toward the upper range of observed returns, then extracts a policy by advantage-weighted
regression:

$$
\mathcal{L}_V = \mathbb{E}\left[L_2^\tau\big(Q(s,a) - V(s)\big)\right], \qquad
L_2^\tau(u) = |\tau - \mathbb{1}(u<0)|\,u^2
$$

With $\tau = 0.7$–$0.9$, the expectile approximates a max over the *actions present in the
data* — capturing the benefit of a max without ever evaluating $Q$ off-distribution. This is
why IQL is often the strongest and simplest choice.

::: check
Q-learning is off-policy, so it should learn from any dataset. Why does running DQN on a fixed dataset usually produce a policy worse than the one that generated it?

- [x] The Bellman max ranges over all actions, including ones never taken — $Q$ extrapolates arbitrarily there, the inflated value enters the target, and nothing corrects it
  > Online, the agent tries the overestimated action and the estimate falls. Interaction *is* the error-correction mechanism, and removing it is what makes the setting hard. The symptom is unmistakable: $Q$-values growing without bound while performance collapses.
- [ ] The dataset is not large enough to cover the state space
  > More data helps, and the failure happens even with excellent coverage of the *states* if the actions are narrow.
- [ ] Off-policy correction requires importance weights that are unavailable offline
  > Q-learning's target needs no importance weights; that is what makes it off-policy in the first place.
- [ ] The behaviour policy is unknown
  > Some constraint methods estimate it, and not knowing it is a complication rather than the core failure.
:::

## Data quality decides everything

::: warning
Offline RL cannot exceed what the data supports. Three cases:

- **Expert data only.** Behaviour cloning is a strong baseline and often wins. Offline RL
  adds little because there is nothing better to find.
- **Mixed-quality data.** This is where offline RL earns its complexity — it can stitch good
  segments from different mediocre trajectories into a better policy than any single one.
- **Narrow data.** If the dataset never visits the region where good behaviour lies, no
  algorithm can find it.

**Always run behaviour cloning as a baseline.** A substantial fraction of published offline
RL results do not beat it, and it is a one-line implementation.
:::

## Evaluation is the unsolved part

You cannot run the policy to evaluate it — that is the whole premise. **Off-policy
evaluation** estimates performance from logged data:

- **Importance sampling** reweights logged returns by the policy ratio. Unbiased, and the
  variance explodes over long horizons for the same reason as lesson 6.07's importance
  sampling.
- **Doubly robust** estimators combine importance sampling with a learned value model,
  reducing variance at some bias.
- **Fitted Q evaluation** trains a critic for the target policy on the logged data — and
  inherits exactly the extrapolation problem the whole field is about.

None is reliable enough to select hyperparameters. In practice, deployment decisions rest on
a limited online test, which is what offline RL was supposed to avoid.

## Where it meets language models

DPO (lesson 5.09) is offline preference learning, and the connection is worth seeing:

- It learns from a **fixed** preference dataset with no on-policy generation.
- Its KL term against the reference model is a **policy constraint**, exactly analogous to
  TD3+BC's behaviour-cloning term.
- Its off-policy weakness — data reflects a different policy than the one being trained — is
  the same distribution-shift problem stated in a different vocabulary.
- Iterative DPO, which regenerates data with the current policy, is the language-model
  analogue of moving back toward online RL.

::: exercise
Your offline RL agent's Q-values reach 10,000 while dataset returns are around 50. What is
happening and what do you change?
:::

::: solution
**Classic extrapolation error.** The critic is assigning high values to actions absent from
the data, and those values are feeding back through the Bellman target.

**The mechanism, step by step.** $\max_{a'}Q(s',a')$ picks whichever action the network
happens to score highest — including unseen ones where the network is extrapolating. That
inflated value raises $Q(s,a)$. Next iteration, $Q(s,a)$ is part of some other state's
target. With no interaction to falsify the estimate, the inflation compounds. This is the
deadly triad (lesson 6.04) with the third leg — off-policy — taken to its extreme.

**Confirm it in two checks:**

1. Compare $Q(s,a)$ for **dataset** actions against $Q(s,a)$ for **policy** actions at the
   same states. If policy actions score far higher, the policy has found the extrapolation
   region.
2. Plot mean $Q$ against mean dataset return over training. Divergence between them is the
   signature.

**Fixes, in order of expected effectiveness:**

1. **Switch to IQL.** It never evaluates $Q$ on actions outside the dataset, so the failure
   mode is structurally impossible rather than merely penalised. Simplest and usually the
   strongest choice.

2. **Add a CQL penalty**, with $\alpha \approx 5$–10. It pushes down $Q$ on policy actions
   and up on dataset actions, producing a lower bound. Tune $\alpha$ upward until $Q$ values
   are in the range of dataset returns — that is the diagnostic to target.

3. **Add a behaviour-cloning term** (TD3+BC). One line, keeps the policy near the data, and
   often sufficient on its own.

4. **Check the terminal mask.** `target = r + gamma * q_next * (1 - done)`. A missing
   terminal mask produces unbounded value growth for a completely different reason, and it is
   worth excluding before attributing everything to extrapolation.

**And run behaviour cloning as a baseline** before any of this. If BC matches or beats your
corrected offline RL agent, the dataset is expert-quality and the extra machinery is not
earning its complexity.

**A calibration to remember:** a correct $Q$ should be roughly bounded by
$r_{\max}/(1-\gamma)$. At $\gamma = 0.99$ and dataset returns near 50, values above a few
hundred are already suspect. 10,000 is not a tuning problem.
:::

## What to carry forward

- The Bellman max queries unseen actions, where a network extrapolates and errors compound.
- Interaction is what corrects value errors online; offline you must replace it with pessimism.
- Policy constraints keep you near the data; conservative critics lower unseen values.
- IQL avoids off-distribution queries entirely and is usually the best starting point.
- Always run behaviour cloning as a baseline, and expect off-policy evaluation to be unreliable.
