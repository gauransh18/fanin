---
summary: Turning a probability distribution into text — what each sampling parameter actually does, and why greedy decoding is not the safe default.
prereqs: [pretraining-objective, distributions-in-ml]
---

A language model outputs a distribution over the vocabulary. Decoding is the rule that
picks a token from it, and the choice changes the output more than most people expect.

## Greedy

Take the argmax at every step:

```python
next_token = logits.argmax(dim=-1)
```

Deterministic and fast, and it produces noticeably worse text than sampling on open-ended
generation. Two reasons:

**Degenerate repetition.** Greedy decoding falls into loops — "the the the", or a paragraph
repeated verbatim. Once a repeated pattern becomes locally most likely, nothing breaks the
cycle.

**Likelihood is not quality.** The highest-probability continuation of a sentence is often
bland. Human text is *not* the maximum-likelihood sequence under the model — it sits in a
region of moderate probability, and always choosing the mode leaves that region.

Greedy is correct when there is one right answer: classification, extraction, structured
output, and evaluation where reproducibility matters.

## Temperature

Divide the logits before the softmax:

$$
p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

```python
import torch

logits = torch.tensor([2.0, 1.0, 0.5, 0.1])
for T in (0.5, 1.0, 1.5):
    print(f'T={T}: {torch.softmax(logits / T, -1).numpy().round(3)}')
# T=0.5: [0.617 0.084 0.031 0.014]  -> sharper
# T=1.0: [0.514 0.189 0.115 0.077]
# T=1.5: [0.447 0.229 0.164 0.126]  -> flatter
```

$T \to 0$ approaches greedy; $T \to \infty$ approaches uniform. Note that temperature
**never assigns zero probability** — even at $T = 0.1$ the worst token retains a small
chance, and over a thousand tokens small chances accumulate. That is why truncation methods
are used alongside it rather than instead.

## Top-$k$ and top-$p$

**Top-$k$** keeps the $k$ highest-probability tokens and renormalises. Simple, and the right
$k$ depends on the distribution's shape: after "The capital of France is" the distribution
is nearly one-hot and $k=50$ admits 49 wrong answers; after "She opened the door and" it is
broad and $k=50$ may be too restrictive.

**Top-$p$ (nucleus)** keeps the smallest set whose cumulative probability exceeds $p$. The
set size adapts to the distribution — small when the model is confident, large when it is
not. This is why it became the default.

```python
import torch

def sample(logits, temperature=1.0, top_k=None, top_p=None, min_p=None):
    logits = logits / max(temperature, 1e-6)

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, -float('inf'))

    if min_p is not None:
        # Keep tokens at least min_p as likely as the best one.
        probs = logits.softmax(-1)
        threshold = min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < threshold, -float('inf'))

    if top_p is not None:
        sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        remove = cumulative - sorted_logits.softmax(-1) > top_p   # keep the first over p
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, -float('inf'))

    return torch.multinomial(logits.softmax(-1), num_samples=1)
```

**Min-$p$** is a newer alternative: keep tokens whose probability is at least $p$ times the
maximum. It adapts like top-$p$ but responds to the *peak* rather than the cumulative mass,
which handles very flat distributions better.

::: key
**Order matters.** Apply temperature first, then truncation. Truncating before scaling
means temperature operates on an already-restricted set and its effect is not what you
intended.

Most libraries do this correctly; hand-rolled samplers frequently do not.
:::

::: check
Greedy decoding produces noticeably worse open-ended text than sampling. Why?

- [x] Human text is not the maximum-likelihood sequence — it sits in a region of moderate probability, and always taking the mode leaves that region
  > It also falls into degenerate repetition: once a repeated pattern is locally most likely, nothing breaks the cycle. Greedy remains correct where there *is* one right answer — classification, extraction, structured output, reproducible evaluation.
- [ ] Greedy decoding cannot use the full vocabulary
  > It considers every logit; it simply always takes the largest.
- [ ] The argmax is undefined when two logits tie
  > Ties are broken arbitrarily and are vanishingly rare in float arithmetic.
- [ ] Sampling produces higher-likelihood sequences on average
  > Sampling produces *lower*-likelihood sequences on average. That is the point.
:::

## Beam search

Maintain $b$ partial sequences and expand each, keeping the $b$ highest-scoring:

```python
def beam_search(model, prompt, beam=4, max_len=64, length_penalty=1.0):
    beams = [(prompt, 0.0)]
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logprobs = model(seq)[:, -1].log_softmax(-1)
            top = logprobs.topk(beam)
            for lp, tok in zip(top.values[0], top.indices[0]):
                candidates.append((torch.cat([seq, tok[None, None]], -1), score + lp.item()))
        # Normalise by length, or beam search will always prefer shorter sequences.
        beams = sorted(candidates,
                       key=lambda b: -b[1] / (b[0].size(1) ** length_penalty))[:beam]
    return beams[0][0]
```

Beam search finds higher-likelihood sequences, which is what you want for translation and
summarisation — tasks with a roughly correct output. It is actively bad for open-ended
generation: higher likelihood means blander, and beams converge on near-identical
continuations.

The length penalty is not optional. Log-probabilities are negative, so a longer sequence
always scores worse, and unnormalised beam search terminates as early as it can.

## Repetition control

```python
def apply_repetition_penalty(logits, generated, penalty=1.1):
    """Divide positive logits and multiply negative ones for seen tokens."""
    for token in set(generated.tolist()):
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty
    return logits
```

Also common: **no-repeat $n$-gram** blocking (forbid any $n$-gram that already occurred) and
**frequency/presence penalties** (subtract a term proportional to how often a token has
appeared).

Use these sparingly. A repetition penalty applied to code forbids reusing variable names; a
no-repeat-3-gram constraint on prose forbids legitimate repeated phrases. Heavy repetition
usually indicates a model or prompt problem that the penalty is masking.

## What to use

| Task | Settings |
|---|---|
| Factual QA, extraction | Greedy, or $T=0$ |
| Code generation | $T = 0.2$, top-$p$ 0.95 |
| Chat and general use | $T = 0.7$, top-$p$ 0.9 |
| Creative writing | $T = 0.9{-}1.1$, top-$p$ 0.95 |
| Self-consistency voting | $T = 0.8$ — diversity is the point |
| Translation, summarisation | Beam 4–5 with length penalty |

::: warning
**"Temperature 0" is not fully deterministic in practice.** Floating-point non-associativity
(lesson 2.15) means that changing the batch composition changes reduction order, which can
flip an argmax when the top two logits are close. Serving systems batch requests
dynamically, so the same prompt can produce different output on different calls.

If you need reproducibility, fix the batch size, or accept that greedy is *nearly*
deterministic rather than exactly so.
:::

## Structured output

For output that must parse — JSON, a specific schema — constrain decoding rather than
prompting and hoping:

```python
def constrained_step(logits, allowed_token_ids):
    mask = torch.full_like(logits, -float('inf'))
    mask[allowed_token_ids] = 0
    return logits + mask
```

Libraries (Outlines, guidance, llama.cpp's grammars) compile a grammar or JSON schema into
a per-step mask of allowed tokens. The output is guaranteed valid, which removes an entire
class of retry logic.

The cost: constraining can push the model off its preferred continuation and reduce quality
on the *content*. Constrain the structure, not the substance.

::: exercise
Your chat model produces repetitive, dull responses. You are using greedy decoding. Walk
through what to change and what to check.
:::

::: solution
**Greedy is the immediate cause.** Repetition loops and blandness are its two characteristic
failures, both explained above: once a pattern is locally most likely, argmax cannot escape
it, and the mode of the distribution is blander than typical human text.

**First change:** $T = 0.7$, top-$p$ $= 0.9$. This is the standard chat configuration and
usually resolves it outright.

**If repetition persists at those settings**, the problem is not decoding:

1. **Check the chat template** (lesson 5.05). A mismatch between training and inference
   formatting produces a model operating off-distribution, and degenerate repetition is a
   common symptom. Print the exact prompt string and diff it against a training example.

2. **Check the context length.** If the conversation exceeds the trained context, the model
   is extrapolating positionally (lesson 4.13) and quality degrades sharply. Print the token
   count.

3. **Check for a damaged model.** Over-trained SFT (lesson 5.05) or DPO degeneration
   (lesson 5.09) both produce flat, repetitive output. Compare against the base model at the
   same decoding settings — if the base is more varied, the fine-tune is the problem.

**If it is dull but not repetitive**, that is a different diagnosis: the SFT data was dull.
Decoding cannot add variety the model does not have. Raising temperature to 1.2 will
produce more varied *errors*, not more interesting *content*.

**What not to reach for first:** a repetition penalty. It masks the symptom, and at the
values needed to suppress real degeneration (1.3+) it visibly damages fluency and forbids
legitimate repetition. Use it as a small safety net (1.05–1.1) after fixing the cause, not
instead of fixing it.
:::

## What to carry forward

- Greedy is right for one-answer tasks and wrong for open-ended generation.
- Temperature rescales and never zeroes; truncation methods do the zeroing.
- Top-$p$ adapts its set size to the distribution, which is why it is the default.
- Apply temperature before truncation.
- Beam search suits translation and summarisation, and is bad for open-ended text.
