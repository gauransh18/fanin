---
summary: The full RLHF loop — four models in memory, why the KL penalty is load-bearing, and the failure modes that make it hard to run.
prereqs: [reward-modeling, supervised-finetuning]
seealso: [policy-gradients, actor-critic, trpo-ppo]
---

RLHF optimises a language model against a learned reward (lesson 5.07) using policy
gradient methods (track 6). It is the technique that turned capable base models into usable
assistants, and it is genuinely difficult to run.

## The objective

$$
\max_\theta \; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}\Big[ r(x,y) - \beta\, D_{\text{KL}}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big) \Big]
$$

Maximise reward, penalised by divergence from the SFT model. Both terms are load-bearing.

::: key
**The KL penalty is not a regulariser you can tune away.** It does three things:

1. **Keeps the policy in-distribution for the reward model.** The reward model was trained
   on SFT-like outputs. Far from that distribution its predictions are unreliable, and
   optimising against unreliable predictions is exactly reward hacking (lesson 5.07).
2. **Preserves capabilities.** Without it, the policy collapses toward whatever maximises
   reward — often a single high-scoring response pattern repeated regardless of input.
3. **Bounds the damage.** It makes RLHF an *adjustment* to the SFT model rather than a
   fresh optimisation, which is why it does not destroy what pretraining built.

$\beta$ is typically 0.01–0.1. Too low and the policy hacks the reward; too high and
nothing changes.
:::

## Four models in memory

| Model | Role | Trained? |
|---|---|---|
| **Policy** $\pi_\theta$ | Generates responses | Yes |
| **Reference** $\pi_{\text{ref}}$ | Frozen SFT model for the KL term | No |
| **Reward** $r_\phi$ | Scores completed responses | No |
| **Value** $V_\psi$ | Estimates expected return, for the advantage | Yes |

Four copies of a 7B model is roughly 56 GB in bf16 before optimizer state. This is the
practical reason RLHF is hard: memory, plus the complexity of keeping four models
synchronised across a cluster.

Common economies: share a backbone between policy and value with two heads; run the
reference and reward models as LoRA adapters over the policy's frozen base.

## The loop

```python
import torch
import torch.nn.functional as F

def rlhf_step(policy, ref, reward_model, value, prompts, beta=0.05, clip=0.2):
    # 1. Roll out the current policy.
    with torch.no_grad():
        responses, logprobs_old = policy.generate(prompts, return_logprobs=True)
        ref_logprobs = ref.logprobs(prompts, responses)
        scores = reward_model(prompts, responses)            # scalar per response
        values_old = value(prompts, responses)

    # 2. Per-token reward: KL everywhere, the reward model's score at the end.
    kl = logprobs_old - ref_logprobs                          # (B, T)
    rewards = -beta * kl
    rewards[:, -1] += scores                                  # terminal reward only

    # 3. GAE (lesson 6.08).
    advantages, returns = compute_gae(rewards, values_old, gamma=1.0, lam=0.95)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. Several PPO epochs on this batch.
    for _ in range(4):
        logprobs = policy.logprobs(prompts, responses)
        ratio = torch.exp(logprobs - logprobs_old)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()   # lesson 6.09

        value_loss = F.mse_loss(value(prompts, responses), returns)
        (policy_loss + 0.5 * value_loss).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

Two structural details:

**The reward is terminal.** The reward model scores a *complete* response, so there is no
per-token signal from it. Only the KL penalty is dense. This is what makes credit
assignment hard — the model must work out which of 500 tokens caused a good score.

**$\gamma = 1$.** No discounting. A response is a finite episode and a token at position 400
matters as much as one at position 4.

## Estimating the KL

The exact KL requires summing over the whole vocabulary. Two estimators are used in
practice:

```python
# k1: unbiased, high variance. logratio = log pi(y) - log pi_ref(y)
kl_k1 = logratio

# k3: unbiased, much lower variance, and always non-negative.
kl_k3 = torch.exp(-logratio) - 1 + logratio
```

The k3 estimator is standard. It is unbiased for the same quantity and has far lower
variance, which matters because this term is inside a gradient estimate that is already
noisy.

## The failure modes

**Reward hacking.** Covered in lesson 5.07. Watch the reward–length correlation, KL
magnitude, and human evaluation at checkpoints — never the reward model's own score, which
cannot detect that it is being gamed.

**Mode collapse.** The policy converges to a narrow set of high-reward responses. Diversity
falls sharply, which shows up as near-identical answers to related prompts. This follows
from reverse-KL's mode-seeking behaviour (lesson 1.15) and is partly intrinsic to the
objective.

**Value function lag.** If $V_\psi$ is poorly fitted, advantages are wrong and the policy
gradient points in an unhelpful direction. Symptoms: value loss not decreasing, explained
variance near zero. Warm up the value head before enabling policy updates.

**Instability.** PPO has many interacting hyperparameters. Reproducing published RLHF
results is notoriously difficult, and small implementation differences — advantage
normalisation, KL estimator, reward whitening — change outcomes materially.

## What to monitor

```python
metrics = {
    'reward_mean':      scores.mean(),
    'kl_mean':          kl.sum(-1).mean(),        # per sequence, not per token
    'response_length':  lengths.float().mean(),   # rising = probable hacking
    'value_loss':       value_loss,
    'explained_var':    1 - (returns - values).var() / returns.var(),
    'clip_fraction':    (ratio.sub(1).abs() > clip).float().mean(),
    'entropy':          -(logprobs.exp() * logprobs).sum(-1).mean(),
}
```

`clip_fraction` above about 0.2 means the policy is moving too fast for PPO's trust region
— lower the learning rate. `entropy` collapsing means mode collapse is underway.

## Simpler alternatives

RLHF's complexity motivated several alternatives:

- **DPO** (lesson 5.09) — reformulates the same objective as a supervised loss on
  preference pairs. No reward model, no rollouts, no value function.
- **RLAIF** — replace human labels with a strong model's judgements. Cheaper, scales
  further, inherits the judge's biases.
- **Best-of-$n$** — sample $n$ responses and return the highest-reward one. No training at
  all; costs $n\times$ inference. A strong baseline that is often skipped in comparisons.
- **GRPO** — drops the value model, computing advantages from the mean reward across a
  group of samples for the same prompt. Much less memory, and it works well for verifiable
  domains (lesson 6.14).

::: exercise
During RLHF the reward rises steadily while human evaluation gets worse. KL is 45 nats per
sequence. What is happening and what do you do?
:::

::: solution
**45 nats is very large.** A typical healthy run sits around 5–15 nats per sequence. At 45
the policy has moved far outside the distribution the reward model was trained on, so the
reward model's scores are extrapolations — and the policy has found a direction where those
extrapolations are high and actual quality is not.

Rising reward with falling human quality is the definition of reward hacking, and the KL
number tells you the mechanism: distribution shift, not a subtle reward-model flaw.

**Immediate actions:**

1. **Roll back to the checkpoint where human evaluation was last good.** Continuing makes
   it worse monotonically; there is nothing to gain from more steps.
2. **Raise $\beta$** by 3–5×. If it was 0.01, try 0.05. Re-run and watch KL — you want it
   to stabilise in the 5–15 range rather than growing without bound.
3. **Look at the actual outputs.** Print 20 responses from the hacked policy next to the
   SFT model's. The hack is almost always obvious once you read them: padding, repeated
   structure, a stock phrase the reward model likes, or degenerate repetition.

**Then diagnose why the KL grew unchecked:**

- **Is the KL penalty actually being applied?** A sign error, or applying it to the wrong
  tensor, is a common bug and produces exactly this. Verify that increasing $\beta$ reduces
  measured KL.
- **Is $\beta$ adaptive?** Some implementations adjust $\beta$ to hit a KL target. If yours
  does and the target is set too high, that is your answer.
- **Is the reference model the right one?** Using the base model instead of the SFT model
  as reference means the KL term is pulling toward the wrong distribution.

**Longer-term fixes:**

- **Use a reward ensemble** and subtract a multiple of the standard deviation. In
  off-distribution regions the ensemble disagrees, so the penalty grows exactly where the
  reward is unreliable.
- **Evaluate with humans at every checkpoint**, not just at the end. The reward model
  cannot tell you it is being hacked.
- **Consider DPO or best-of-$n$.** Both stay much closer to the SFT distribution by
  construction, which makes this failure mode far harder to reach.
:::

## What to carry forward

- RLHF maximises reward minus $\beta\,D_{\text{KL}}$ from the SFT policy; both terms matter.
- Four models in memory, and the reward is terminal, which makes credit assignment hard.
- Use the k3 KL estimator; normalise advantages; $\gamma = 1$.
- Rising reward with falling human quality plus large KL is reward hacking.
- Evaluate with humans at checkpoints — the reward model cannot detect its own exploitation.
