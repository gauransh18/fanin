---
summary: Training a small model from a large one's outputs, why soft targets carry more information than labels, and the variants that matter for LLMs.
prereqs: [entropy-cross-entropy-kl, supervised-finetuning, quantization]
---

Quantization shrinks a model without changing its architecture. Distillation makes a
genuinely smaller model that learns from a larger one — and it often beats training that
small model from scratch on the original data.

## Soft targets

The classical formulation matches the teacher's full output distribution, softened by a
temperature:

$$
\mathcal{L} = \alpha\,T^2 \cdot D_{\text{KL}}\!\left(p_T^{(\tau)} \,\|\, p_S^{(\tau)}\right) + (1-\alpha)\,\mathcal{L}_{\text{CE}}(y, p_S)
$$

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.9):
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.log_softmax(teacher_logits / T, dim=-1),
        reduction='batchmean', log_target=True) * (T ** 2)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard
```

The $T^2$ factor is not cosmetic. Softening by $T$ shrinks the gradients of the soft term by
roughly $1/T^2$, so without the correction the balance between the two terms would change
every time you adjusted the temperature.

::: key
**Soft targets carry more information than labels.** A one-hot label says "this is a cat".
A teacher's distribution says "cat 0.9, lynx 0.06, dog 0.03, car $10^{-8}$" — which encodes
that lynxes resemble cats and cars do not.

Hinton called this **dark knowledge**: the relative probabilities of the *wrong* answers
describe the structure of the problem, and that structure is what a small model struggles to
learn from labels alone. It is also why distillation works with *unlabelled* data — the
teacher supplies the target.
:::

## The LLM variants

**Sequence-level distillation.** Generate text with the teacher and fine-tune the student on
it as ordinary SFT data (lesson 5.05). Simple, needs no logit access, and works with an
API-only teacher. Most "distilled" open models are this.

**Token-level distillation.** Match the teacher's per-token distribution over a corpus.
Needs logits, gives a much denser signal — a distribution over 128,000 tokens at every
position rather than one sampled token.

**On-policy distillation.** The student generates, and the teacher scores the *student's*
samples. This fixes sequence-level distillation's exposure bias: the student learns on the
distribution it actually produces, not the teacher's. Generalised knowledge distillation
(GKD) formalises it and consistently beats off-policy variants.

**Reasoning distillation.** Have a strong reasoning model produce chains of thought
(lesson 5.11), filter to those reaching verified-correct answers, and fine-tune a small
model on them. This is how small models acquire reasoning behaviour they could not learn
from the original corpus.

## Which divergence

Lesson 1.15's asymmetry matters here:

- **Forward KL**, $D_{\text{KL}}(p_T \| p_S)$, is mass-covering. The student must put mass
  everywhere the teacher does, so it hedges across all the teacher's modes. This is the
  classical choice.
- **Reverse KL**, $D_{\text{KL}}(p_S \| p_T)$, is mode-seeking. The student concentrates on
  one mode and may ignore the rest.

For a student with far less capacity than the teacher, reverse KL is often better: the
student cannot represent every mode, and committing to the important ones beats blurring
across all of them. Forward KL with a small student produces an averaged, hedging model —
the same phenomenon as MSE's blur in lesson 3.14.

## Architectural choices

**Width or depth?** At matched parameter count, keeping depth and reducing width generally
distils better — reasoning depth is harder to recover than representational width.

**Layer initialisation.** Initialise the student's layers from a subset of the teacher's
(every $k$-th layer). Substantially faster convergence than random initialisation.

**Intermediate matching.** Match hidden states or attention maps as well as outputs:

```python
import torch.nn.functional as F

def feature_distillation(student_hidden, teacher_hidden, projections):
    """Map student layer i to teacher layer round(i * L_t / L_s)."""
    loss = 0.0
    for s_h, t_h, proj in zip(student_hidden, teacher_hidden, projections):
        loss = loss + F.mse_loss(proj(s_h), t_h.detach())
    return loss / len(projections)
```

The projection handles the dimension mismatch. This helps most when the student is much
smaller, where output matching alone leaves the intermediate representations unconstrained.

## What it achieves

| Approach | Typical retention of teacher quality |
|---|---|
| Train small model from scratch | baseline |
| Sequence-level distillation | +3–8 points on benchmarks |
| Token-level distillation | +5–12 points |
| On-policy distillation | +6–15 points |
| Reasoning distillation (verified chains) | Large in the target domain |

::: warning
**A distilled model inherits the teacher's errors and biases**, including ones the original
training data would not have produced. It also inherits the teacher's refusals and style,
which is usually wanted and occasionally not.

And distilling from a model whose terms prohibit it is a licensing question, not a technical
one. Check before building a product on it.
:::

## Distillation and quantization together

They are complementary and usually combined: distil to a smaller architecture, then quantize
the result. The ordering matters — quantize last, because distillation needs full precision
gradients and a quantized student trains worse.

A useful comparison when you have a memory budget:

- 70B at 4 bits = 35 GB, best quality.
- 13B distilled from 70B, at 8 bits = 13 GB, much faster, meaningfully worse.
- 13B distilled, at 4 bits = 6.5 GB, fastest, worse again.

Lesson 5.12's rule — bigger model, fewer bits — holds until the model no longer fits at all,
at which point distillation is the next lever.

::: exercise
You distil a 70B teacher into a 7B student using sequence-level distillation on 100k
generated examples. The student is good at the teacher's typical outputs and poor at
anything unusual. Explain and fix.
:::

::: solution
**The training distribution was the teacher's high-probability output**, and the student
learned exactly that.

Sequence-level distillation samples from the teacher, usually at moderate temperature. Those
samples concentrate on the teacher's modes. The student never sees what the teacher would do
on an unusual prompt, or how the teacher handles the tail of its own distribution — so on
those inputs the student is operating outside its training distribution entirely.

There is a second, compounding cause: **exposure bias**. The student is trained on the
teacher's outputs but at inference generates its own. Once it produces something the teacher
would not have, it is off-distribution and errors compound — lesson 5.05's point about
teacher forcing, in a distillation setting.

**Fixes, in order of impact:**

1. **On-policy distillation.** Have the *student* generate, and the teacher score or correct
   those generations. The student now learns on the distribution it actually produces, which
   addresses both causes directly. This is the single biggest improvement available and is
   what GKD does.

2. **Diversify the prompts.** 100k examples is not many, and if the prompts came from a
   narrow source the coverage problem is in the inputs rather than the sampling. Include
   adversarial prompts, edge cases, and out-of-domain inputs — the places you observed
   failure.

3. **Raise the sampling temperature** when generating teacher data, to 1.0–1.2, and sample
   several completions per prompt. This exposes more of the teacher's distribution. It also
   admits more teacher errors, so filter.

4. **Switch to token-level distillation if you have logit access.** Matching the full
   distribution at every position gives orders of magnitude more signal than a single
   sampled sequence, and it directly conveys what the teacher would do on low-probability
   continuations.

**How to confirm the diagnosis before acting:** measure the student's loss on teacher
outputs at temperature 0 versus temperature 1.2. A large gap confirms that coverage of the
teacher's distribution — not capacity — is the limitation. If the gap is small, the student
is capacity-limited and none of the above will help much.
:::

## What to carry forward

- Soft targets encode the structure of wrong answers, which labels do not.
- Keep the $T^2$ factor, or changing temperature silently reweights the loss.
- Reverse KL suits a much smaller student; forward KL makes it hedge.
- On-policy distillation fixes exposure bias and is consistently the best variant.
- Distil first, quantize second.
