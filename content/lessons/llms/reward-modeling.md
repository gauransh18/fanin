---
summary: Learning a scalar quality signal from pairwise comparisons, the Bradley-Terry model behind it, and why reward models are so easy to overfit.
prereqs: [supervised-finetuning, maximum-likelihood, entropy-cross-entropy-kl]
---

SFT can only imitate demonstrations (lesson 5.05). To exceed them you need a signal about
*relative* quality, and the cheapest reliable source of that signal is a human choosing
between two responses.

## Why comparisons rather than ratings

Asking annotators to score a response 1–10 produces unusable data: different people anchor
differently, the same person drifts over a session, and nobody can articulate what
distinguishes a 7 from an 8.

Asking "which of these two is better" is far more reliable. It requires no absolute scale,
and inter-annotator agreement is substantially higher.

## Bradley–Terry

Model the probability that response $y_w$ is preferred to $y_l$ as depending only on the
difference of their latent scores:

$$
p(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)
$$

Maximising the likelihood of observed comparisons (lesson 1.14) gives the loss:

$$
\mathcal{L} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\big(r_\theta(x,y_w) - r_\theta(x,y_l)\big)\right]
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """An SFT model with the LM head replaced by a scalar head."""
    def __init__(self, base_model, d):
        super().__init__()
        self.backbone = base_model
        self.score = nn.Linear(d, 1, bias=False)
        nn.init.normal_(self.score.weight, std=1 / (d + 1) ** 0.5)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask, output_hidden_states=True)
        h = h.hidden_states[-1]                              # (B, T, d)
        # Score the LAST non-padding token: with causal attention only that
        # position has seen the whole response (lesson 4.06).
        last = attention_mask.sum(1) - 1
        return self.score(h[torch.arange(h.size(0)), last]).squeeze(-1)


def preference_loss(model, chosen, chosen_mask, rejected, rejected_mask, margin=0.0):
    r_w = model(chosen, chosen_mask)
    r_l = model(rejected, rejected_mask)
    return -F.logsigmoid(r_w - r_l - margin).mean(), (r_w > r_l).float().mean()
```

::: key
**Score the last non-padding token.** With causal attention, earlier positions have not
seen the full response, so their representations cannot encode overall quality. Scoring
position 0, or mean-pooling across positions, both produce a reward model that reads the
beginning of a response and ignores the rest.

This is a genuine and common bug, and its symptom is a policy that learns to write good
opening sentences followed by nonsense.
:::

## Only differences are identified

The loss depends on $r(x,y_w) - r(x,y_l)$. Adding a constant $c$ to every reward for a
given prompt leaves it unchanged, so **the absolute scale is not identified**.

Two consequences:

- Comparing raw reward values across prompts is meaningless. A reward of 3.2 on one prompt
  and 1.1 on another says nothing about which response is better in any absolute sense.
- RL against the reward model (lesson 5.08) must use **advantages**, not raw rewards —
  which is exactly what lesson 6.08 shows is necessary for variance reduction anyway.

In practice a small penalty on $\lVert r \rVert$ keeps the scale bounded, purely for
numerical stability.

## Training details

- **Initialise from the SFT model.** The reward model must understand the task before it
  can judge it. Training from a pretrained base without SFT works worse.
- **One epoch.** Reward models overfit faster than almost anything else in this pipeline.
  Validation accuracy typically peaks within one pass and falls after.
- **Low learning rate**, 1e-6 to 5e-6 — lower than SFT.
- **Accuracy, not loss, is the metric.** The fraction of pairs ranked correctly is
  interpretable; the loss is not. Human agreement on preference data is typically 70–75%,
  so a reward model at 75% validation accuracy is at the ceiling of the signal, not
  underfitting.

::: warning
**A reward model above ~78% validation accuracy is usually a warning, not a success.**
Human annotators agree with each other only about 70–75% of the time. A model exceeding
that has learned annotator idiosyncrasies or spurious correlates of the labelling process
rather than the underlying preference.

Check what it is keying on before celebrating.
:::

## Length bias, and every other shortcut

Annotators prefer longer responses, all else equal. A reward model trained on their
judgements learns "longer is better", and RL against it produces a policy that pads.

This is the clearest instance of a general problem: the reward model learns **correlates**
of quality that are easier to detect than quality itself.

```python
import numpy as np

def diagnose_length_bias(model, pairs):
    """Correlation between reward and response length is the first thing to check."""
    rewards, lengths = [], []
    for prompt, response in pairs:
        rewards.append(score(model, prompt, response))
        lengths.append(len(response.split()))
    r = np.corrcoef(rewards, lengths)[0, 1]
    print(f'reward-length correlation: {r:+.3f}')
    return r
```

A correlation above about 0.3 means length is doing significant work. Mitigations:

- **Length-controlled evaluation** — compare only responses of similar length.
- **Length penalty** in the RL objective.
- **Balanced preference data** — ensure the preferred response is not systematically
  longer, by construction.

Other shortcuts that show up: formatting (bullet points score higher), hedging language,
and sycophancy — agreeing with the user's stated position regardless of correctness.

## Ensembles and uncertainty

A single reward model is a fixed, exploitable target. Two standard improvements:

**Ensembles.** Train several reward models with different seeds and use the mean, or the
mean minus a multiple of the standard deviation. Disagreement between them marks regions
where the reward is unreliable, and penalising high-variance regions directly discourages
the policy from exploiting them.

**Process rewards.** Instead of one score for a whole response, score each reasoning step.
Denser signal, harder to game with a plausible-looking final answer, and substantially more
expensive to annotate. This is what makes the reasoning training of lesson 5.10 work.

::: exercise
Your reward model reaches 82% validation accuracy. RLHF against it produces a policy that
writes extremely long, hedged, list-formatted responses that humans rate *worse* than the
SFT model. Diagnose.
:::

::: solution
**This is reward hacking, and 82% was the warning sign.** Human agreement is 70–75%; a
model at 82% is fitting something more consistent than human preference — most likely
surface features of how the data was collected.

**Diagnose in this order:**

1. **Correlate reward with length.** Run the function above on held-out responses. Above
   0.3 and length is a major factor; the policy has found that padding raises reward
   monotonically.

2. **Correlate with formatting.** Score matched pairs differing only in presentation — the
   same content as prose versus as a bulleted list. A large gap means formatting is being
   rewarded independently of content.

3. **Check the KL divergence from the SFT policy.** If it is large, the policy has drifted
   far from the distribution the reward model was trained on, so the reward model is being
   evaluated off-distribution where it is unreliable. This is the standard failure mode and
   the KL penalty exists to prevent it (lesson 5.08).

4. **Score adversarial constructions.** Take a good response and append irrelevant hedging
   and a summary list. If the reward goes *up*, you have a direct demonstration.

**Fixes, in order of impact:**

- **Tighten the KL penalty.** If the policy has drifted, this is the immediate lever and
  costs nothing to try.
- **Retrain the reward model with length-balanced data**, so the preferred response is not
  systematically longer. This attacks the root cause.
- **Add an explicit length penalty** to the RL reward.
- **Use a reward ensemble** and penalise disagreement, so the policy cannot exploit one
  model's idiosyncrasy.
- **Stop earlier.** Reward hacking worsens monotonically with RL steps. Evaluate the policy
  with *humans* at several checkpoints, not with the reward model — which by construction
  cannot detect that it is being hacked.

**The structural lesson:** a reward model is a learned proxy, and optimising hard against a
proxy diverges from the true objective. This is Goodhart's law with a gradient. It is the
main reason DPO (lesson 5.09) and direct methods are attractive — not that they avoid the
problem, but that they keep the policy closer to the data distribution by construction.
:::

## What to carry forward

- Pairwise comparisons are reliable where absolute ratings are not.
- Bradley–Terry gives a logistic loss on the reward difference; only differences are identified.
- Score the last non-padding token, and train for one epoch at a low rate.
- Validation accuracy above ~78% means you are fitting annotator artefacts.
- Always measure the reward–length correlation before trusting the model.
