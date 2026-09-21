---
summary: Where the data comes from, why deduplication matters more than almost any architectural choice, and the filtering pipeline that decides model quality.
prereqs: [pretraining-objective, expectation-variance-concentration]
---

Data is the largest single determinant of a pretrained model's quality, and it receives
the least attention in papers because it is unglamorous and hard to share. The pipeline
below is roughly what every serious effort runs.

## The sources

| Source | Scale | Quality | Notes |
|---|---|---|---|
| Common Crawl | ~100 TB/snapshot | Very low raw | Needs aggressive filtering |
| Curated web (FineWeb, C4) | 1–15 T tokens | Medium | Common Crawl, filtered |
| Code (GitHub, The Stack) | ~1 T tokens | High | Licence filtering required |
| Books | ~100 B tokens | High | Long-range coherence |
| Wikipedia | ~10 B tokens | High | Small but heavily upweighted |
| Papers (arXiv, PubMed) | ~50 B tokens | High | Technical and mathematical |
| Synthetic | unbounded | Variable | Increasingly significant |

Raw Common Crawl is mostly unusable: boilerplate, navigation, spam, machine translation,
and adult content. A typical pipeline keeps **under 10%** of what it ingests.

## The pipeline

```
extract text  →  language ID  →  quality filter  →  dedup  →  decontaminate  →  mix
```

**Text extraction.** HTML to text, dropping navigation and boilerplate. Trafilatura and
resiliparse are the standard tools, and the choice measurably affects downstream quality —
this is not a solved preprocessing step.

**Language identification.** fastText or CLD3 with a confidence threshold. Raising the
threshold improves quality and disproportionately removes code-switched and dialectal text,
which is a real fairness cost rather than a neutral filter.

**Quality filtering.** Two families:

- **Heuristic** (Gopher rules): drop documents outside 50–100,000 words, with mean word
  length outside 3–10 characters, with over 90% of lines starting identically, with under
  80% of words containing an alphabetic character, or missing enough stop words.
- **Model-based**: train a classifier to distinguish curated text (Wikipedia, books) from
  raw crawl and keep high-scoring documents. This is what FineWeb-Edu does, and it produces
  a measurably better corpus than heuristics alone.

```python
def gopher_filters(doc):
    words = doc.split()
    n = len(words)
    if not 50 <= n <= 100_000:
        return False
    if not 3 <= sum(len(w) for w in words) / n <= 10:
        return False
    if sum(1 for w in words if any(c.isalpha() for c in w)) / n < 0.8:
        return False
    lines = doc.split('\n')
    if lines and max(lines.count(l) for l in set(lines)) / len(lines) > 0.3:
        return False                                     # repeated boilerplate
    stop = {'the', 'be', 'to', 'of', 'and', 'that', 'have', 'it'}
    return len(stop & {w.lower() for w in words}) >= 2
```

## Deduplication is the highest-leverage step

::: key
Common Crawl contains enormous duplication — the same article on twenty sites, templated
pages, mirrored documentation. Training on duplicates means the model **memorises** rather
than generalises, and memorised text is regurgitated verbatim, which is a privacy and
copyright problem as well as a quality one.

Deduplication reliably improves downstream performance at a fixed token budget. It is
cheap, it is mechanical, and it matters more than most architectural choices you could
make instead.
:::

Two levels:

**Exact duplicates** — hash each document, keep one per hash. Trivial and catches a lot.

**Near-duplicates** — MinHash with locality-sensitive hashing. Represent each document by
the minimum hash values of its $n$-gram shingles; documents sharing many minima are likely
similar, and LSH finds candidate pairs without an all-pairs comparison.

```python
import hashlib

def minhash_signature(text, n_perm=128, shingle=5):
    words = text.lower().split()
    shingles = {' '.join(words[i:i + shingle]) for i in range(len(words) - shingle + 1)}
    sig = []
    for seed in range(n_perm):
        best = min((int(hashlib.md5(f'{seed}:{s}'.encode()).hexdigest()[:16], 16)
                    for s in shingles), default=0)
        sig.append(best)
    return sig

def jaccard_estimate(sig_a, sig_b):
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)
```

The fraction of matching signature positions estimates Jaccard similarity — that is the
MinHash guarantee. Threshold around 0.8 for near-duplicate removal.

## Decontamination

Benchmark test sets leak into web crawls constantly. A model that has seen the test set
reports inflated scores and you will not know.

```python
def is_contaminated(doc, benchmark_ngrams, n=13):
    words = doc.lower().split()
    doc_ngrams = {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}
    return bool(doc_ngrams & benchmark_ngrams)
```

13-gram overlap is the common threshold. Run it against every benchmark you intend to
report, and **report the contamination rate you found** — a paper that does not mention
decontamination has almost certainly not done it.

## Mixing

Sources are sampled at rates different from their natural proportions:

```python
mixture = {
    'filtered_web': 0.60,
    'code':         0.15,
    'books':        0.10,
    'papers':       0.08,
    'wikipedia':    0.05,
    'other':        0.02,
}
```

Two findings worth knowing. **Code improves reasoning on non-code tasks** — the leading
explanation is that code has explicit, verifiable logical structure. And upsampling small
high-quality sources helps up to about 4 epochs of that source, then hurts.

Modern practice adds **curriculum**: a final phase on a high-quality mixture — textbooks,
curated instruction data, code — for the last few percent of tokens. This "mid-training"
or "annealing" phase reliably improves benchmark performance and is now standard.

## Synthetic data

Increasingly, a large share of training data is model-generated: textbook-style explanations,
solved problems with reasoning traces, code with tests.

It works, and it has a specific failure mode. Training a model repeatedly on its own output
narrows the distribution — the tails disappear first, and after several generations the
model collapses toward a low-diversity mode. The mitigations are to keep real data in the
mixture, to filter synthetic output by an independent verifier (execution for code, a proof
checker for maths), and to generate with a **stronger** model than the one being trained.

::: warning
The scaling-law analysis of lesson 5.03 assumes fresh data. If your corpus is smaller than
the compute-optimal requirement, you are repeating data, and the returns to compute fall
well below what the laws predict. For most organisations, **data is the binding constraint,
not compute** — which is why so much effort now goes into synthetic generation.
:::

::: exercise
You train two 7B models on 1T tokens each. Corpus A is 1T unique tokens. Corpus B is 250B
unique tokens repeated 4 times. Predict the difference and explain.
:::

::: solution
**A will be meaningfully better**, but the gap is smaller than the 4× data difference
suggests.

**Why B is not catastrophic.** Empirical work on repeated data finds that up to roughly 4
epochs, repeated tokens are worth *nearly* as much as fresh ones — on the order of 80–95%
of the value per token. B sees 4 epochs, which is right at the edge of that regime. Expect
B's final loss to be a few percent higher, not dramatically so.

**Why A still wins.** Each additional epoch on the same data yields diminishing information.
The model has already extracted most of what a document teaches by the second pass;
subsequent passes contribute increasingly to memorisation rather than generalisation.

**What to measure to confirm it.** Three signals distinguish memorisation from learning:

1. **Train–validation gap.** B will show a larger gap, since it has partly memorised its
   training set. A's should be near zero at one epoch.
2. **Verbatim extraction rate.** Prompt with a 50-token prefix from the training set and
   measure how often the continuation is reproduced exactly. B's rate will be markedly
   higher — which is also a privacy and copyright liability.
3. **Held-out-domain performance.** B's narrower corpus covers less ground, so it will lag
   more on domains absent from its 250B than on domains present in it.

**What to do if you are in B's situation** — which most organisations are, since fresh
high-quality data runs out:

- Repeat, but stop around 4 epochs. Beyond that the returns go sharply negative.
- Spend the surplus compute on a **larger model** on the same data rather than more epochs;
  lesson 5.03's tradeoff shifts when data is capped.
- Generate synthetic data from a stronger model, with verification.
- Deduplicate harder — a corpus claimed as 250B unique tokens usually is not, and
  MinHash-level deduplication frequently removes another 10–30%.
:::

## What to carry forward

- Under 10% of raw crawl survives filtering, and extraction quality measurably matters.
- Deduplication is the highest-leverage step and prevents verbatim memorisation.
- Decontaminate against every benchmark you report, with 13-gram overlap.
- Code in the mixture improves non-code reasoning; upsample quality sources to ~4 epochs.
- Repeated data is worth ~80–95% per token up to 4 epochs, then falls off sharply.
