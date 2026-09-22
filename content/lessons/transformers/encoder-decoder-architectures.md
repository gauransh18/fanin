---
summary: Three ways to arrange attention, what each is good for, and why decoder-only won despite being the least obvious choice.
prereqs: [transformer-block, causal-masking]
---

The same block can be wired three ways. The differences are entirely about **what each
position is allowed to see**, and that single choice determines what the model can do.

## Encoder-only

Bidirectional attention: every position attends to every other, in both directions.

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

Each token's representation is informed by its full context on both sides, which is ideal
for understanding and useless for generation — you cannot generate left to right if
producing token $t$ requires seeing token $t+1$.

Trained with **masked language modelling**: corrupt 15% of tokens and predict them from
the rest. BERT, RoBERTa, and every modern text embedding model work this way.

The efficiency cost is real: only the masked 15% of positions produce a training signal,
so 85% of the compute per sequence produces no gradient. An autoregressive model gets a
signal at every position.

## Encoder-decoder

An encoder reads the source bidirectionally; a decoder generates the target causally,
with an extra **cross-attention** sublayer reading the encoder's output.

```python
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n1, self.self_attn  = nn.RMSNorm(d), MultiHeadAttention(d, n_heads)
        self.n2, self.cross_attn = nn.RMSNorm(d), CrossAttention(d, n_heads)
        self.n3, self.mlp        = nn.RMSNorm(d), FeedForward(d)

    def forward(self, x, memory):
        x = x + self.self_attn(self.n1(x), is_causal=True)   # attend to own prefix
        x = x + self.cross_attn(self.n2(x), memory)          # attend to the source
        x = x + self.mlp(self.n3(x))
        return x
```

Cross-attention is ordinary attention with queries from the decoder and keys and values
from the encoder:

```python
q = self.W_q(decoder_x)        # (B, T_tgt, d)
k = self.W_k(memory)           # (B, T_src, d)
v = self.W_v(memory)
```

The natural fit for sequence-to-sequence tasks where source and target are genuinely
different objects: translation, summarisation, speech recognition. T5, BART and Whisper
are built this way.

## Decoder-only

One stack, causal attention throughout, no encoder. Source and target are concatenated
into one sequence and the model predicts the next token at every position.

GPT, Llama, Mistral, Qwen — everything at the frontier.

::: check
Masked language modelling corrupts 15% of tokens and predicts them. What does that cost relative to autoregressive training?

- [x] Only the masked 15% of positions produce a training signal, so 85% of the compute per sequence yields no gradient
  > An autoregressive model gets a signal at every position. That efficiency gap is one of the reasons decoder-only models scaled better, quite apart from what each architecture can do.
- [ ] Nothing — every position still contributes through the bidirectional context
  > Positions contribute as *context*, which is not the same as producing a loss term.
- [ ] It requires twice the memory, because attention is bidirectional
  > Bidirectional attention over $T$ positions costs the same as causal attention over $T$ positions, up to the masked-out half.
- [ ] It cannot use the same tokenizer as a generative model
  > Tokenization is shared freely between the two.
:::

## Why decoder-only won

It is not the obvious choice. Encoder-decoder separates the "read" and "write" roles
cleanly, and for translation it is arguably the better inductive bias. Four reasons it
lost anyway.

::: key
**1. Every position trains.** A decoder-only model gets a next-token prediction signal at
all $T$ positions. BERT gets one at 15% of them. For a fixed compute budget that is a
6–7× difference in gradient signal per FLOP.

**2. One objective covers everything.** Translation, summarisation, question answering and
chat are all "continue this text" once you put the input in the prompt. No task-specific
architecture, no separate encoder to train.

**3. KV caching is straightforward.** With causal attention, past keys and values never
change, so generation reuses them (lesson 4.10). Cross-attention caches too, but the
bookkeeping is messier.

**4. In-context learning emerged.** Nobody designed for it. A model trained to continue
text turned out to be able to infer a task from examples in its prompt, and that capability
scales with model size. It is a property of the decoder-only setup that encoder-decoder
models show much more weakly.
:::

## Prefix LM: the compromise

Allow bidirectional attention over the prompt and causal attention over the continuation —
a block-structured mask:

```python
import torch

def prefix_mask(T, prefix_len, device):
    """Bidirectional within the prefix, causal after it."""
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    causal[:, :prefix_len] = True          # everyone can see the whole prefix
    return causal
```

This gets bidirectional understanding of the input with autoregressive generation of the
output, in one stack. UL2 and PaLM used variants. It has not displaced pure causal
attention, largely because the extra complexity buys less than expected once models are
large.

## The comparison

| | Encoder-only | Encoder-decoder | Decoder-only |
|---|---|---|---|
| Attention | Bidirectional | Bi + causal + cross | Causal |
| Objective | Masked LM | Denoising / seq2seq | Next token |
| Signal per sequence | 15% of positions | Target only | All positions |
| Generation | No | Yes | Yes |
| In-context learning | No | Weak | Strong |
| Examples | BERT, embedding models | T5, Whisper | GPT, Llama, Claude |

::: check
In cross-attention, where do the queries, keys and values come from?

- [x] Queries from the decoder, keys and values from the encoder's output
  > The decoder asks a question from its own state and reads the source. Self-attention takes all three from the same stream; cross-attention is what makes it "cross".
- [ ] Queries from the encoder, keys and values from the decoder
  > That would have the source interrogating the partial translation, which is backwards.
- [ ] All three from the encoder, with the decoder only reading the result
  > Then the attention would not depend on what the decoder has generated so far, and every output step would read the same thing.
- [ ] All three from the decoder, with the encoder output added afterwards
  > That is a residual connection, not attention. Nothing would select *which* source positions to read.
:::

## Where encoders survive

Encoder-only models are not obsolete — they are the right tool for a narrower job:

- **Embeddings and retrieval.** A bidirectional encoder produces better sentence
  representations than a causal model, because the final token of a causal model has seen
  everything but earlier tokens have not. This matters for lesson 5.14's RAG systems.
- **Classification and token labelling**, where you never generate and the bidirectional
  context is free.
- **Reranking**, where a cross-encoder scoring a (query, document) pair jointly beats
  comparing two independent embeddings.

::: exercise
You need a system that translates English to French. Argue for each architecture, then
pick one.
:::

::: solution
**Encoder-decoder.** The task genuinely has two distinct sequences with different
languages and different lengths, which is exactly what the architecture models. Cross-
attention gives an explicit alignment mechanism — the direct descendant of lesson 4.01's
Bahdanau attention, which was invented for this task. The encoder reads French-relevant
features from English bidirectionally, which a causal model cannot do for the source. For
a dedicated translation system trained from scratch on parallel data, this is the
strongest choice at a given parameter count, and it is why Whisper and most production
translation systems still use it.

**Decoder-only.** Format as `English: {src}\nFrench:` and continue. The source is attended
to causally rather than bidirectionally, which is a real loss — but it is partly
compensated because the model sees the source at every generation step through attention.
The advantages: you can start from an existing pretrained LLM rather than training from
scratch, you get few-shot control of register and terminology for free, and one model
handles every language pair plus every other task.

**Encoder-only.** Not applicable — it cannot generate.

**The pick depends on the constraint.**

*If you are building a dedicated, high-throughput translation service* and have parallel
data: encoder-decoder. Better quality per parameter, cheaper inference (the encoder runs
once, the decoder is smaller), and cross-attention gives you alignments for free, which
matters for formatting and terminology enforcement.

*If translation is one capability among many*, or you want to start from a pretrained
model: decoder-only. This is what has actually happened in practice — general LLMs now
match or beat dedicated systems on most language pairs, not because the architecture is
better for translation but because they were trained on vastly more data.

**The honest summary:** encoder-decoder is the better architecture for this task in
isolation; decoder-only wins because the task is rarely in isolation.
:::

## What to carry forward

- The only difference between the three is what each position may attend to.
- Encoder-only gets a signal at 15% of positions; decoder-only at all of them.
- Cross-attention is ordinary attention with queries from one sequence and keys from another.
- Decoder-only won on training efficiency, task generality, and in-context learning.
- Bidirectional encoders remain the right choice for embeddings and reranking.
