---
summary: How text becomes integers, why BPE won, and the failure modes that explain a surprising share of model errors.
prereqs: [embeddings, distributions-in-ml]
---

Tokenization is the least glamorous part of a language model and the source of a
disproportionate number of its strange behaviours. Arithmetic errors, character-counting
failures, and inflated costs for some languages all trace back to this layer.

## Why not characters or words

**Characters** give a tiny vocabulary and no out-of-vocabulary problem, but sequences
become 4–5× longer. Since attention is $O(T^2)$ in memory, that is a 16–25× cost — and the
model must spend capacity learning that `c-a-t` is a unit.

**Words** give short sequences and a hopeless vocabulary. By the Zipf law of lesson 1.12,
the tail never ends: any fixed vocabulary leaves a long tail of unknowns, and morphology
means `run`, `runs`, `running`, `ran` are unrelated symbols.

**Subwords** are the compromise: frequent words stay whole, rare words decompose into
pieces, and nothing is ever out of vocabulary.

## Byte-pair encoding

BPE starts from bytes and repeatedly merges the most frequent adjacent pair:

```python
from collections import Counter

def train_bpe(corpus_words, num_merges):
    """corpus_words: {word: count}. Returns the ordered merge list."""
    vocab = {' '.join(w) + ' </w>': c for w, c in corpus_words.items()}
    merges = []

    for _ in range(num_merges):
        pairs = Counter()
        for word, count in vocab.items():
            symbols = word.split()
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += count
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        joined = ''.join(best)
        pattern = ' '.join(best)
        vocab = {w.replace(pattern, joined): c for w, c in vocab.items()}
    return merges

print(train_bpe({'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}, 6))
# [('e','s'), ('es','t'), ('est','</w>'), ('l','o'), ('lo','w'), ('n','e')]
```

Encoding applies the merges in the order they were learned. Because the merge list is
deterministic, the tokenizer is exactly reproducible — which matters, because a mismatch
between the tokenizer used for training and for inference produces garbage.

**Byte-level BPE** (GPT-2 onwards) starts from the 256 byte values rather than Unicode
characters. Every possible string is representable, so there is no unknown token and no
encoding-specific behaviour.

## The alternatives

**WordPiece** (BERT) merges by likelihood gain rather than raw frequency — it picks the
pair maximising $\frac{P(ab)}{P(a)P(b)}$, which favours pairs that genuinely co-occur over
pairs that are merely common. Marks continuations with `##`.

**Unigram** (SentencePiece) works backwards: start with a large candidate vocabulary and
prune tokens whose removal costs the least likelihood. It can produce multiple valid
segmentations of the same string, which enables **subword regularisation** — sampling
different segmentations during training as data augmentation.

**SentencePiece** is properly a framework rather than an algorithm. Its contribution is
treating the input as a raw byte stream with no pre-tokenization, encoding spaces as a
visible marker (`▁`). That makes it language-agnostic — critical for Chinese, Japanese and
Thai, which have no whitespace word boundaries.

## The failure modes

::: key
**Numbers.** If `1234` tokenizes as `12|34` but `1235` as `123|5`, the model sees no
consistent structure and must learn arithmetic over an inconsistent representation. This
is a large part of why LLMs are unreliable at multi-digit arithmetic. Modern tokenizers
(Llama 3, GPT-4o) split digits individually or in fixed groups of three specifically to
fix this.

**Character-level tasks.** "How many r's in strawberry?" is hard because the model never
sees characters — `strawberry` may be two or three tokens, and the letters inside them are
not accessible. This is a representational limitation, not a reasoning failure.

**Language inequity.** English averages roughly 4 characters per token. Many other
languages need 2–3× more tokens for the same content, because the vocabulary was fit on an
English-dominated corpus. Since APIs bill per token and context windows are counted in
tokens, speakers of those languages pay more and get less context for identical text.

**Glitch tokens.** Tokens present in the tokenizer's training corpus but essentially
absent from the model's training data keep their random initial embeddings. Prompting with
them produces bizarre behaviour — the famous `SolidGoldMagikarp` case. Lesson 3.13's point
about rare embeddings never moving from initialisation, made concrete.
:::

## Practical consequences

```python
# pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding('cl100k_base')

for text in ['hello world', 'こんにちは世界', '1234567', 'def fib(n):']:
    ids = enc.encode(text)
    print(f'{text!r:20s} {len(ids):3d} tokens  {[enc.decode([i]) for i in ids]}')
```

Three rules that follow:

- **Never compare per-token loss across tokenizers.** A model with a larger vocabulary
  compresses more text per token, so its per-token loss is higher for the same quality.
  Compare **bits per byte** instead — the point lesson 1.15's exercise made.
- **Count tokens, not characters**, when budgeting context or cost.
- **Pin the tokenizer with the model.** A tokenizer mismatch does not error; it produces
  fluent nonsense.

## Vocabulary size

The tradeoff is direct. Larger vocabulary means shorter sequences (cheaper attention) and
a larger embedding matrix (lesson 3.13) with rarer tokens per row.

| Model | Vocabulary |
|---|---|
| GPT-2 | 50,257 |
| Llama-2 | 32,000 |
| Llama-3 | 128,256 |
| Gemma | 256,000 |

The trend upward tracks multilingual coverage: a 32k vocabulary fit mostly on English
tokenizes other languages badly, and the fix is simply more vocabulary. At 256k, the
embedding is 2B parameters at $d=8192$ — which is why very large vocabularies untie the
input and output embeddings, since sharing becomes the binding constraint.

::: exercise
Your model is bad at arithmetic. You check and `1000` is one token while `1001` is two.
Explain the mechanism and give two fixes.
:::

::: solution
**The mechanism.** To add `1000 + 1001`, the model must recover place-value structure from
token embeddings. `1000` arrives as a single opaque vector; `1001` arrives as two vectors
whose composition is different again. There is no consistent mapping from token to digits,
so the model cannot learn a general algorithm — it must memorise individual arithmetic
facts per tokenization pattern.

This compounds: `999 + 1` requires recognising a carry across a boundary the tokenizer has
placed arbitrarily. The model is being asked to do positional arithmetic on a
representation that destroys positional structure.

**Fix 1: change the tokenizer (the real fix).** Force digits to tokenize individually, or
in consistent groups of three from the right. Llama 3 splits every digit separately;
GPT-4o groups in threes. Either gives a consistent representation where place value is
recoverable, and both substantially improve arithmetic. The cost is longer sequences for
numeric text, which is cheap.

```python
# In a BPE trainer, forbid merges that produce multi-digit tokens.
pattern = r"""[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|..."""
```

**Fix 2: work around it at inference.** Insert separators so the tokenizer is forced to
split consistently — `1 0 0 0 + 1 0 0 1` — or have the model call a calculator tool
(lesson 5.15). Both work and neither fixes the underlying model.

**What does not work:** more training data on arithmetic. The representation is the
bottleneck, not the quantity of examples. Models trained on enormous amounts of arithmetic
with a bad tokenizer still fail on unseen digit lengths, because there is no consistent
structure to generalise from.

**How to confirm the diagnosis:** test the same arithmetic with spaces between digits. If
accuracy jumps sharply, tokenization is the cause. If it does not, the problem is
elsewhere.
:::

## What to carry forward

- Subwords balance sequence length against vocabulary size; BPE merges frequent pairs.
- Byte-level BPE has no unknown token, because every string is a byte sequence.
- Inconsistent digit tokenization is a major cause of arithmetic failure.
- Non-English text costs 2–3× more tokens for identical content.
- Compare bits per byte across tokenizers, never per-token loss.
