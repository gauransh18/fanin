---
summary: The bottleneck that motivated attention, the original alignment mechanism, and why "attention is all you need" was a claim about parallelism.
prereqs: [recurrent-networks, lstm-gru]
---

Attention was not invented for transformers. It was invented to fix one specific failure
in machine translation, and understanding that failure is what makes the mechanism feel
inevitable rather than arbitrary.

## The bottleneck

A 2014-era translation model was two RNNs. The encoder read the source sentence and
produced a final hidden state; the decoder generated the target from that state alone:

$$
\mathbf{c} = \mathbf{h}_T^{\text{enc}}, \qquad
\mathbf{s}_t = \text{RNN}(\mathbf{s}_{t-1}, \mathbf{y}_{t-1}, \mathbf{c})
$$

Everything about the source sentence had to fit in one fixed-size vector. For a
five-word sentence, fine. For a fifty-word sentence, the same 512 numbers must carry
fifty words of content — and performance degraded sharply with length, in a way that
plotting BLEU against source length made unmistakable.

::: insight
This is the fixed-state limitation of lesson 3.12, in its original setting. The decoder
needed a word from position 3 at output step 20, and by then the encoder state had been
overwritten seventeen times.
:::

## Bahdanau attention

The 2014 fix: keep **every** encoder state, and let the decoder choose which to read at
each step.

$$
e_{ti} = a(\mathbf{s}_{t-1}, \mathbf{h}_i), \qquad
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_j \exp(e_{tj})}, \qquad
\mathbf{c}_t = \sum_i \alpha_{ti}\mathbf{h}_i
$$

Three steps that recur unchanged in every attention mechanism since:

1. **Score** each source position against the current decoder state.
2. **Normalise** the scores into weights that sum to 1, with softmax.
3. **Average** the source states, weighted by those scores.

The context vector $\mathbf{c}_t$ is now different at every output step. The bottleneck
is gone: the decoder can reach any source position directly, in one hop.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """The original: a small MLP scores each (decoder state, encoder state) pair."""
    def __init__(self, d):
        super().__init__()
        self.W_s, self.W_h = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(self, s, H, mask=None):          # s: (B, d), H: (B, T, d)
        scores = self.v(torch.tanh(self.W_s(s).unsqueeze(1) + self.W_h(H)))  # (B, T, 1)
        scores = scores.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        alpha = F.softmax(scores, dim=-1)        # (B, T)
        return torch.bmm(alpha.unsqueeze(1), H).squeeze(1), alpha
```

The scoring function here is a learned MLP — **additive** attention. Luong simplified it
to a dot product a year later, which is what lesson 4.02 builds on, because a dot product
is one matrix multiply and an MLP is not.

## Attention weights as alignment

Plotting $\alpha_{ti}$ as a heatmap over source and target positions produced the first
genuinely interpretable picture of what a neural translation model was doing: a mostly
diagonal band, with off-diagonal structure exactly where the two languages reorder — French
adjectives after nouns, German verbs at the end.

::: warning
This made attention weights look like an explanation, and they are weaker evidence than
they appear. A high weight means a position contributed heavily to a weighted average; it
does not establish that the *information used* came from there, because the value vectors
have already mixed information across positions in earlier layers.

There is a substantial literature showing attention weights can be perturbed significantly
without changing predictions. Treat them as a useful signal, not a causal account.
:::

## Then remove the RNN

The 2017 observation was that if attention already lets any position reach any other in
one hop, the recurrence is doing very little — and the recurrence is what forbids
parallelism (lesson 3.11).

So: drop the RNN, apply attention *within* the sequence as well as across, and add
positional encodings to restore the word order the recurrence had implicitly supplied.

| | RNN + attention | Transformer |
|---|---|---|
| Path length between positions | $O(T)$ through the RNN | $O(1)$ |
| Parallel over positions in training | No | Yes |
| Compute per layer | $O(T \cdot d^2)$ | $O(T^2 d + T d^2)$ |
| Order information | From the recurrence | Added explicitly |

Note that the transformer is asymptotically **worse** in compute — quadratic in sequence
length. It won anyway, because $O(T^2)$ work that saturates a GPU beats $O(T)$ work that
runs at 5% utilisation. That is the arithmetic-intensity argument of lesson 1.03 deciding
an architecture.

::: key
"Attention is all you need" is easily misread as a claim that attention is a better
inductive bias. The load-bearing claim was about **hardware**: attention can be expressed
as large matrix products over all positions at once, so training parallelises across the
sequence. That is what made scale possible, and scale is what made the models good.
:::

## Self-attention

The generalisation is small. Instead of a decoder state attending over encoder states, let
every position attend over every position in the *same* sequence:

$$
\text{score}(i, j) = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}, \qquad
\mathbf{z}_i = \sum_j \text{softmax}_j(\text{score}(i,j))\,\mathbf{v}_j
$$

Each position produces three vectors from its own representation: a **query** (what am I
looking for), a **key** (what do I offer), and a **value** (what I will contribute if
selected). Lesson 4.02 derives why the scaling factor is $\sqrt{d}$ and what the three
projections buy.

::: exercise
Why does a transformer need positional encodings when an RNN does not?
:::

::: solution
Self-attention is **permutation-equivariant**. Permute the input positions and the outputs
permute identically — the operation itself contains no notion of order.

The reason is visible in the formula: $\mathbf{z}_i = \sum_j \alpha_{ij}\mathbf{v}_j$ is a
sum over $j$, and a sum does not care about the order of its terms. The score
$\mathbf{q}_i\cdot\mathbf{k}_j$ depends on the two tokens' *contents*, not their indices.
So "dog bites man" and "man bites dog" produce the same multiset of representations.

An RNN gets order for free because it is **sequential by construction**: $\mathbf{h}_t$
depends on $\mathbf{h}_{t-1}$, so position $t$ is defined by how many updates have
happened. Order is not represented — it is the mechanism.

Convolutions get it the same way, from the kernel's fixed offsets (lesson 3.09), which is
why they also need no positional input.

**The transformer's tradeoff:** removing the sequential dependency is exactly what bought
parallelism, and positional encoding is the price. Lesson 4.04 covers the options —
sinusoidal, learned, and RoPE — and the fact that they are a *choice* rather than a
built-in is why transformers can be adapted to non-sequential data (images, graphs, sets)
just by changing what "position" means.
:::

## What to carry forward

- Attention began as a fix for a fixed-size encoder bottleneck in translation.
- Score, normalise, average — that is every attention mechanism, then and now.
- Attention weights are suggestive but weak evidence for what information was used.
- The transformer's win was parallelism over positions, not a better inductive bias.
- Self-attention is permutation-equivariant, which is why positions must be added explicitly.
