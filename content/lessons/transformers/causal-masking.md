---
summary: The mask that makes parallel training of an autoregressive model possible, the off-by-one that silently breaks it, and how to test for leakage.
prereqs: [scaled-dot-product-attention, probability-spaces]
---

An autoregressive model factorises $p(x_1,\ldots,x_T) = \prod_t p(x_t\mid x_{<t})$
(lesson 1.10). Each factor conditions only on the prefix. The causal mask is what enforces
that while still computing all $T$ factors in one parallel pass — and it is the single
most important correctness property in a transformer.

## The mask

Position $i$ may attend to position $j$ only when $j \le i$:

$$
S_{ij} = \begin{cases}
\dfrac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{d_k}} & j \le i \\[1ex]
-\infty & j > i
\end{cases}
$$

```python
import torch

T = 5
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
# tensor([[ True, False, False, False, False],
#         [ True,  True, False, False, False],
#         [ True,  True,  True, False, False],
#         [ True,  True,  True,  True, False],
#         [ True,  True,  True,  True,  True]])
```

Lower triangular, **including the diagonal** — a position sees itself. Excluding the
diagonal is a real and subtle bug: position 0 would then have nothing to attend to, its
softmax row would be all `-inf`, and you would get `nan`.

## Why this buys parallelism

::: key
Without the mask you would have to run the model $T$ times, once per prefix, to get $T$
training signals. With it, one forward pass over the full sequence produces all $T$
next-token predictions simultaneously — position $i$'s output is a valid prediction of
token $i+1$ because it only saw tokens $\le i$.

This is the entire reason transformer training is efficient. It is also why the trick does
not help at **inference**: generating token $t+1$ genuinely requires token $t$, which does
not exist yet. Training is parallel, generation is sequential, and that asymmetry drives
all of lesson 4.10 and the serving work in track 7.
:::

::: check
Why must the causal mask include the diagonal?

- [x] Without it position 0 would have nothing to attend to, its softmax row would be all $-\infty$, and the output would be `nan`
  > A position sees itself. Excluding the diagonal is a real and subtle bug — and one that surfaces as a `nan` rather than as a wrong-looking mask.
- [ ] Including the diagonal is optional; it only changes how much a token weights itself
  > For every position but the first it is a modelling difference. For position 0 it is the difference between a distribution and a division by zero.
- [ ] It is needed so the mask is symmetric
  > A causal mask is deliberately not symmetric. That asymmetry is the entire point.
- [ ] It lets the model attend to future tokens at inference
  > It never permits attending forwards, at training or inference.
:::

## Teacher forcing

During training, position $i$ conditions on the **true** prefix, not on what the model
would have generated. That is teacher forcing, and it follows directly from maximising
the likelihood of the data.

The consequence is **exposure bias**: at training time the model only ever sees
ground-truth prefixes; at generation time it sees its own outputs, including its own
mistakes. Errors compound, because a single bad token puts the model in a state it never
trained on.

In practice this matters far less than early work suggested. Large models trained on
enough data are robust to their own errors, and scheduled sampling — mixing in model
predictions during training — has not proven worth its complexity. RL fine-tuning
(lesson 6.14) addresses it directly by training on the model's own generations.

## Shifting the targets

The off-by-one here is the most common bug in a from-scratch implementation:

```python
import torch
import torch.nn.functional as F

# tokens: (B, T+1)
inputs  = tokens[:, :-1]        # positions 0 .. T-1
targets = tokens[:,  1:]        # positions 1 .. T

logits = model(inputs)                        # (B, T, vocab)
loss = F.cross_entropy(logits.reshape(-1, vocab),
                       targets.reshape(-1),
                       ignore_index=-100)
```

Position $i$ of the logits predicts `targets[i]`, which is `tokens[i+1]`. Get this wrong
and one of two things happens:

- **No shift** — the model predicts the current token from itself, which the diagonal of
  the mask makes trivial. Loss collapses to near zero within a few hundred steps and the
  model is useless.
- **Shifted the wrong way** — the model predicts the previous token, which is also easy,
  and the loss again drops implausibly fast.

Both look like spectacular training success. Lesson 3.16's first check — compare the
initial loss to $\ln V$, then watch how fast it falls — catches them.

## Combining causal and padding masks

Variable-length batches need both:

```python
import torch

def build_mask(pad_mask, device):
    """pad_mask: (B, T) bool, True for real tokens."""
    B, T = pad_mask.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    # Causal over queries x keys, AND the key must be a real token.
    return causal[None, None] & pad_mask[:, None, None, :]
```

The broadcast positions matter (lesson 2.03): `pad_mask[:, None, None, :]` places the
length mask on the **key** axis and lets the batch, head and query axes broadcast. Putting
it on the query axis instead masks the wrong thing and produces a model that trains to a
mediocre loss.

::: warning
A fully padded query row is all `-inf`, and softmax of that is `nan` — which then
propagates through the entire batch via the matmul, including into positions that were
fine. Use the dtype minimum rather than `-inf`:

```python
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
```

The padded row then produces a finite garbage value that the loss discards via
`ignore_index`, rather than poisoning everything.
:::

::: check
Causal masking makes training parallel over positions. Why does it not make *generation* parallel?

- [x] Generating token $t+1$ genuinely requires token $t$, which does not exist yet
  > The mask lets you score a known sequence in parallel, not invent an unknown one. Training is parallel, generation is sequential, and that asymmetry drives KV caching in lesson 4.10 and all the serving work in track 7.
- [ ] The mask has to be rebuilt at each step, which serialises the loop
  > Rebuilding a triangular mask is trivial and is not what serialises anything.
- [ ] Softmax cannot be computed incrementally
  > It can, and FlashAttention's online softmax does exactly that.
- [ ] Generation uses a different attention pattern from training
  > It uses the same pattern. What differs is that the tokens arrive one at a time.
:::

## Document packing

Pretraining concatenates many documents into fixed-length blocks to avoid padding
entirely (lesson 2.10). That creates a new leakage problem: tokens from document 2 can
attend to document 1.

Two responses. The simple one is to accept it — with a separator token, the model learns
that content before a separator is irrelevant, and the cost is small. The correct one is a
**block-diagonal mask**:

```python
import torch

def packed_mask(doc_ids, device):
    """doc_ids: (B, T) int, the document each token belongs to."""
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]      # (B, T, T)
    T = doc_ids.size(1)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return same_doc & causal[None]
```

FlashAttention supports this efficiently through variable-length APIs (`flash_attn_varlen`),
so there is no longer a performance reason to skip it.

## Testing for leakage

Do not trust the mask by inspection. Test it — this catches every variant of the bug in
three lines:

```python
import torch

def test_causal(model, T=16, vocab=100):
    """Changing a future token must not change an earlier output."""
    x = torch.randint(0, vocab, (1, T))
    with torch.no_grad():
        base = model(x)
        x2 = x.clone()
        x2[0, T // 2] = (x2[0, T // 2] + 1) % vocab      # perturb the middle
        perturbed = model(x2)

    before = (base[0, :T // 2] - perturbed[0, :T // 2]).abs().max()
    after  = (base[0, T // 2:] - perturbed[0, T // 2:]).abs().max()
    print(f'before the edit: {before:.2e} (must be ~0)')
    print(f'after  the edit: {after:.2e} (must be > 0)')
    assert before < 1e-5, 'LEAK: a future token changed an earlier output'
    assert after > 1e-5, 'the model is ignoring the edit entirely'
```

::: exercise
You implement a transformer and the training loss drops to 0.02 within 200 steps. What
happened, and how would you confirm it?
:::

::: solution
A loss of 0.02 nats is a perplexity of 1.02 — the model is choosing correctly among
essentially one option every time. No language model achieves that on real text; the
irreducible entropy of English is around 0.6–1.2 nats per token even for the best models.
**The model is seeing the answer.**

**Three candidate mechanisms:**

1. **Targets not shifted.** `targets = tokens` instead of `tokens[:, 1:]`. The model
   predicts the current token, and the mask's diagonal means it can see it. Trivial.

2. **The mask is wrong.** Using `torch.triu` instead of `tril`, or forgetting the mask
   entirely, lets each position see the whole sequence including its own target.

3. **The mask is applied to the wrong axis.** Transposing the score matrix before masking,
   or building the mask as `(T, T)` and applying it after `q @ k.transpose(-2,-1)` in the
   wrong orientation, produces an upper-triangular effect.

**How to confirm, in order of speed:**

Run the `test_causal` function above. It distinguishes all three: if `before` is nonzero,
the mask leaks (cases 2 or 3); if `before` is zero, the mask is fine and the problem is
the target shift (case 1).

Then decode one training example and read it:

```python
print('input :', tokenizer.decode(inputs[0][:20]))
print('target:', tokenizer.decode(targets[0][:20]))
```

The target should read as the input shifted left by one token. If the two lines are
identical, it is case 1.

Finally, print the mask itself for `T=5` and confirm it is lower-triangular with a `True`
diagonal. Thirty seconds, and it rules out cases 2 and 3 conclusively.
:::

## What to carry forward

- Lower triangular **including the diagonal**; excluding it gives `nan` at position 0.
- The mask is what makes training parallel; inference stays sequential regardless.
- Shift targets by one. Loss falling implausibly fast means you did not.
- Broadcast the padding mask onto the **key** axis, and mask with the dtype minimum.
- Write the leakage test. Do not verify a mask by reading it.
