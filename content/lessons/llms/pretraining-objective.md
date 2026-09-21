---
summary: Why next-token prediction is a stronger objective than it looks, what the loss value means, and the two-phase structure every pretraining curve shows.
prereqs: [entropy-cross-entropy-kl, causal-masking, probability-spaces]
---

The entire pretraining objective is next-token prediction:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

This looks modest. It is not, and understanding why is most of understanding why the field
went the direction it did.

## Why it is more than autocomplete

From lesson 1.10, the factorisation $p(x_1,\ldots,x_T) = \prod_t p(x_t\mid x_{<t})$ is
**exact**. So a model that predicts next tokens perfectly has learned the full joint
distribution over text.

Consider what predicting the next token well actually requires:

```
The capital of the country that borders Spain to the west is ___
```

Predicting `Lisbon` requires geography, the resolution of a relative clause, and an
inference chain. There is no separate "reasoning module" — the loss just gets lower when
the model can do it.

```
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + ___
```

Predicting `fibonacci(n-2)` requires understanding recursion and the specific function
being defined.

::: key
Any capability that reduces next-token loss will be learned, if it is learnable from the
data and the model has capacity. This is why one objective produces translation,
arithmetic, code, and in-context learning without any of them being asked for.

It is also the limitation: a capability that does **not** reduce next-token loss on the
training distribution will not emerge, however much you scale. Being truthful, being
helpful, and refusing harmful requests are all in that category — the pretraining corpus
contains plenty of confident falsehood, and predicting it accurately is rewarded. That gap
is what tracks 5.05 through 5.09 exist to close.
:::

## Reading the loss

From lesson 1.15, the loss is cross-entropy in nats per token and $e^{\mathcal{L}}$ is
perplexity.

| Loss | Perplexity | Stage |
|---|---|---|
| 11.7 | 120,000 | Random init on a 120k vocabulary |
| 6.2 | 490 | Unigram frequencies learned — a few hundred steps |
| 4.5 | 90 | Bigram and local syntax |
| 3.0 | 20 | Grammar, basic factual recall |
| 2.0 | 7.4 | A strong modern model on general text |
| 1.5 | 4.5 | Frontier scale |

A drop from 2.5 to 2.4 is a 10% reduction in effective branching factor. Small loss deltas
at low loss are large; this is why scaling-law papers plot log-loss.

::: warning
Per-token loss is **not comparable across tokenizers** (lesson 4.08). A larger vocabulary
packs more text into each token, so per-token loss is mechanically higher for the same
quality. Compare **bits per byte**:

$$
\text{bpb} = \frac{\mathcal{L} \cdot \log_2 e \cdot n_{\text{tokens}}}{n_{\text{bytes}}}
$$

Any comparison of two models' losses without this correction is measuring tokenizers.
:::

## The shape of the curve

Every pretraining run shows the same structure:

**Phase 1, the first few hundred steps.** Loss falls from $\ln V$ to roughly 6. The model
learns the marginal token distribution — which tokens are frequent — and nothing about
context. A plateau *here* means context is not reaching the prediction (lesson 3.16's final
exercise).

**Phase 2, everything after.** Loss falls as a power law in compute. No further sharp
transitions in the aggregate curve. Lesson 5.03 makes this quantitative.

The interesting structure is hidden inside phase 2. Individual capabilities appear abruptly
while the aggregate loss moves smoothly — in-context learning emerging alongside induction
heads (lesson 4.03) is the best-documented case. The aggregate is an average over many
capabilities at different stages, so smoothness in the average is compatible with
sharpness in each.

## Practical details that matter

**Document packing.** Concatenate documents and chop into fixed blocks. No padding, every
token trains. Use a block-diagonal mask so documents cannot attend across boundaries
(lesson 4.06).

**One epoch, or close to it.** Frontier runs see most data once or twice. With each token
seen once there is nothing to memorise, which is why regularisation is nearly absent from
pretraining (lesson 3.07). Repeating data past about 4 epochs gives sharply diminishing
returns and eventually hurts.

**Loss masking.** Prompt tokens in instruction data are usually masked out with
`ignore_index=-100` so only the response contributes. In pretraining, everything trains.

```python
import torch
import torch.nn.functional as F

def pretraining_loss(model, tokens):
    """tokens: (B, T+1) of packed documents."""
    logits = model(tokens[:, :-1])                      # (B, T, V)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
        ignore_index=-100)
```

## The alternatives, and why they lost

**Masked language modelling** (BERT) predicts 15% of positions from bidirectional context.
Better representations per token, but a 6–7× worse signal-to-FLOP ratio (lesson 4.07) and
no generation.

**Span corruption** (T5) masks contiguous spans and generates them. Works well, and needs
an encoder–decoder, which costs the simplicity that made scaling easy.

**Fill-in-the-middle** rearranges a document so the model learns to complete a gap given
both sides. Used as a *supplement* during pretraining for code models, because code editing
is a bidirectional task.

The winner is the simplest one, and it won for reasons of training efficiency rather than
modelling quality — which is a recurring pattern in this curriculum.

::: exercise
Your pretraining loss plateaus at 6.2 after 500 steps and stays there for 50,000 more. The
learning rate is correct and gradients flow. What is wrong?
:::

::: solution
$e^{6.2} \approx 490$, so the model is behaving as though choosing uniformly among ~490
tokens — exactly the unigram distribution. **It has learned token frequencies and nothing
about context.** Something is preventing context from reaching the prediction.

**Ordered hypotheses:**

1. **The attention mask is diagonal rather than causal.** Each position sees only itself,
   so the model is a unigram predictor by construction. Print the mask for $T=5$ and check
   it is lower-triangular with a True diagonal (lesson 4.06).

2. **Positional information is missing.** Without it, self-attention is
   permutation-equivariant (lesson 4.01) — the model sees a bag of tokens and cannot use
   order. Loss lands near bag-of-words level, which is close to this. Check that RoPE
   tables are registered as buffers and are actually being applied; a buffer left as a
   plain attribute stays on CPU and silently does nothing after `.to('cuda')` (lesson 2.08).

3. **The residual stream is being destroyed.** A missing residual connection, or a
   normalization applied in the wrong place, can leave the output nearly independent of the
   input. Check per-layer activation statistics (lesson 3.16).

4. **Targets are shifted the wrong way.** If targets are shifted by $-1$ instead of $+1$,
   the model is asked to predict the *previous* token — which it cannot do from a causal
   context any better than chance beyond frequency.

**The fastest discriminator** is lesson 3.16's check 2: try to overfit a single batch of 8
sequences. If the loss cannot reach near zero, it is a structural bug (1, 2 or 3). If it
can, the architecture is fine and the problem is in the data pipeline (4) — decode one
input/target pair and read them.

**A fifth cause worth ruling out** if the model *can* overfit a batch: the data itself. If
your corpus was tokenized with a different tokenizer than the model's, or is shuffled at
the token rather than document level, there is no learnable context. Decode a training
sample and read it.
:::

## What to carry forward

- The autoregressive factorisation is exact, so next-token prediction targets the full joint.
- Any capability that lowers the loss is learned; any that does not, is not.
- $e^{\mathcal{L}}$ is perplexity; compare bits per byte across tokenizers.
- A plateau near 6 means the model learned frequencies but not context.
- Pack documents, mask across boundaries, and train roughly one epoch.
