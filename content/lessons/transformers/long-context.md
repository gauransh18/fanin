---
summary: What actually limits context length, why a model advertised at 128k often cannot use it, and how to evaluate the claim.
prereqs: [flash-attention, positional-encodings, kv-caching]
---

Context windows grew from 512 tokens to over a million in a few years. Understanding which
constraints were removed — and which were not — is what separates a real capability from a
number on a model card.

## The three constraints

**Memory for attention.** $O(T^2)$ score storage. **Solved** by FlashAttention (lesson
4.12): the matrix is never materialised, so this is no longer the binding limit.

**Compute.** Attention FLOPs grow as $T^2$. From lesson 4.05, attention overtakes the dense
projections only past $T \approx 6d$ — around 24k tokens for a 4096-dimensional model.
Below that, long context is close to free; above it, it dominates and the cost is real.

**KV cache memory.** $O(T)$ per sequence, and unavoidable while you want exact attention.
From lesson 4.10, a single 32k-token sequence on Llama-2 7B needs 17 GB of cache. **This is
now the binding constraint**, which is why GQA and cache quantization matter so much.

## Positional extrapolation

A model trained at 4k does not work at 32k just because the memory fits. RoPE's rotation
angles beyond the trained range are out of distribution, and attention patterns degrade.

Three standard extensions, all requiring a short fine-tune:

- **Position interpolation** — scale positions by $L_{\text{train}}/L_{\text{new}}$ so they
  land back in the trained range. Simple, effective, and it compresses local resolution.
- **NTK-aware scaling** — raise RoPE's base from 10,000. High-frequency (local) components
  are barely affected while low-frequency (global) ones stretch, preserving local
  resolution better than uniform interpolation.
- **YaRN** — combines the two with a temperature correction and per-frequency treatment.
  Currently the strongest, and what most long-context releases use.

::: warning
Extending context is not free in quality. Models fine-tuned for long context typically
regress slightly on short-context tasks, because the positional representation they were
optimised for has changed. Always evaluate both regimes after extending, not just the new
one.
:::

## Lost in the middle

The effect that matters most in practice: retrieval accuracy is **not uniform** across the
context. Models reliably use information at the start and end of a long context and
frequently miss information in the middle.

Two contributing causes. Training data has a strong bias toward the beginning of documents
being important. And attention sinks (lesson 4.03) concentrate weight on early positions,
while recency effects favour the end.

The practical consequences are immediate: in a RAG system (lesson 5.14), put the most
relevant retrieved passage **first or last**, not in the middle. And when a model fails to
use a fact you provided, check where in the context it sat before concluding the model
cannot use it.

## Evaluating the claim

**Needle in a haystack.** Insert a fact at a known position in a long document and ask for
it. Sweep over position and context length to get a two-dimensional map.

```python
def needle_test(model, tokenizer, haystack, needle, question, positions, lengths):
    results = {}
    for length in lengths:
        for frac in positions:
            ctx = tokenizer.encode(haystack)[:length]
            at = int(len(ctx) * frac)
            ctx = ctx[:at] + tokenizer.encode(needle) + ctx[at:]
            prompt = tokenizer.decode(ctx) + '\n\n' + question
            results[(length, frac)] = model.generate(prompt)
    return results
```

::: key
**Needle tests are necessary and not sufficient.** Retrieving one verbatim sentence is the
*easiest* possible long-context task — the target is lexically distinctive and the task is
pure lookup.

Harder and more representative evaluations:
- **Multi-hop** — the answer requires combining facts from two distant positions.
- **Aggregation** — "how many times does X occur", which requires reading everything.
- **Ordering** — "what happened before Y", requiring positional reasoning.
- **Negative controls** — the needle is absent, and the model must say so rather than
  confabulate.

A model can score 100% on needle-in-a-haystack at 128k and fail badly at any of these.
Lesson 5.16 makes the general version of this argument.
:::

## The cost of long prompts

| Context | Prefill FLOPs (7B) | KV cache (GQA-8) | Time to first token |
|---|---|---|---|
| 4k | $5.6\times10^{13}$ | 0.5 GB | ~0.2 s |
| 32k | $4.8\times10^{14}$ | 4.3 GB | ~1.5 s |
| 128k | $2.4\times10^{15}$ | 17.2 GB | ~8 s |

Prefill is compute-bound and grows superlinearly once attention dominates. For an
interactive application, the time to first token at 128k is often the deciding constraint
rather than memory.

**Prompt caching** is the standard mitigation: if a long prefix is shared across requests —
a system prompt, a document being asked about repeatedly — cache its KV once and reuse it.
This turns repeated long-context queries from $O(T)$ prefill into $O(1)$.

## When not to use long context

Long context competes with retrieval, and retrieval often wins:

| Situation | Better approach |
|---|---|
| One document, many questions | Long context with prompt caching |
| Large corpus, targeted questions | Retrieval (lesson 5.14) |
| Needle is lexically distinctive | Retrieval — cheaper and more reliable |
| Answer needs global synthesis | Long context — retrieval cannot see the whole thing |
| Latency-sensitive | Retrieval — prefill dominates otherwise |

The honest summary: long context is best when you genuinely need *global* reasoning over a
document that fits. For lookup over a corpus, a retrieval system is cheaper, faster, and
more accurate.

::: exercise
A model advertises 128k context. You test it and find that facts placed beyond 40k tokens
are retrieved unreliably. Give three explanations and how you would distinguish them.
:::

::: solution
**1. The long-context fine-tune was too short or too narrow.** Extending from 4k to 128k
usually involves a brief fine-tune on long documents. If that stage used few tokens, or
documents that were long but low-quality (concatenated unrelated text), the model has
positional embeddings that technically cover 128k without attention patterns that use
them.

*Distinguish:* test at several lengths — 8k, 16k, 32k, 64k, 128k. A sharp cliff at a
specific length points here, and the cliff often sits at the length actually used in
fine-tuning.

**2. Lost-in-the-middle, not a length limit.** Perhaps position matters more than length.

*Distinguish:* hold total length fixed at 128k and sweep the needle's position from 0% to
100%. If accuracy is high at both ends and poor in the middle, it is this — and the 40k
figure is an artefact of where you happened to place the needle.

**3. The positional scaling method is degrading.** Position interpolation compresses
resolution uniformly, so at high compression ratios nearby positions become hard to
distinguish. This degrades gradually with length rather than cliff-edging.

*Distinguish:* test a task requiring *fine positional discrimination* — "what word
immediately follows X" — at various lengths. Smooth degradation points here; the cliff
pattern points to explanation 1.

**A fourth possibility worth ruling out first:** your evaluation may be wrong. Confirm the
prompt is actually reaching the model unchanged — API truncation, a tokenizer mismatch
inflating the token count, or a template silently dropping content are all common. Print
the token count the model reports receiving and compare to what you sent.

**The practical response** regardless of cause: measure the *usable* context for your task
and treat that as the real number, not the advertised one. Advertised context length is a
maximum input size, not a guarantee of uniform capability across it.
:::

## What to carry forward

- FlashAttention removed the $T^2$ memory limit; the KV cache is now the binding one.
- Extending context needs positional rescaling plus a fine-tune, and costs a little short-context quality.
- Retrieval accuracy is not uniform across position — put important content first or last.
- Needle tests are the easiest long-context task; test multi-hop and aggregation too.
- For lookup over a corpus, retrieval beats long context on cost, latency and accuracy.
