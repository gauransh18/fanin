---
summary: Giving a model access to documents it was not trained on — the pipeline, the retrieval quality problem that dominates, and how to evaluate the parts separately.
prereqs: [embeddings, long-context, encoder-decoder-architectures]
---

A pretrained model knows what was in its training data, at the time it was collected, up to
whatever it memorised. Retrieval-augmented generation supplies the rest at query time:
retrieve relevant documents, put them in the context, answer from them.

## The pipeline

```
documents → chunk → embed → index
                                ↓
query → embed → search → rerank → prompt → generate
```

Each stage is a place the system fails, and the failures look identical from the outside —
a wrong answer. Diagnosing RAG means measuring the stages separately.

## Chunking

Documents must be split to fit retrieval units. The size is a real tradeoff:

| Size | Retrieval precision | Context completeness |
|---|---|---|
| 128 tokens | High | Often missing what it needs |
| 512 tokens | Good | Usually sufficient |
| 2048 tokens | Low — dilutes the embedding | Complete |

```python
def chunk_with_overlap(text, size=512, overlap=64, tokenizer=None):
    tokens = tokenizer.encode(text)
    chunks = []
    for start in range(0, len(tokens), size - overlap):
        chunk = tokens[start:start + size]
        if len(chunk) < overlap:            # skip a trailing fragment
            break
        chunks.append(tokenizer.decode(chunk))
    return chunks
```

Overlap prevents a relevant passage being split across a boundary. Better still, chunk on
**semantic boundaries** — headings, paragraphs, function definitions — rather than a fixed
token count, since a chunk that straddles two unrelated sections embeds to their average and
matches neither.

## Embedding and search

Encode chunks with a bidirectional encoder (lesson 4.07 — a causal model's early tokens have
not seen the rest, which makes it a worse sentence encoder), and search by cosine similarity
(lesson 1.01).

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def embed(model, tokenizer, texts, batch_size=32):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True,
                          truncation=True, return_tensors='pt')
        h = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1)        # mean pool over real tokens
        out.append(F.normalize(pooled, dim=-1))         # normalise: cosine == dot
    return torch.cat(out)
```

For more than a few hundred thousand chunks, exact search is too slow and you want an
approximate index — HNSW (a navigable small-world graph) or IVF-PQ (clustering plus product
quantization). FAISS, Qdrant and pgvector all implement these.

## Hybrid search

::: key
**Dense embeddings miss exact matches.** A query for error code `ERR_4021` or function name
`parse_header_v2` embeds to a generic vector — rare tokens contribute little to a mean-pooled
representation, so the exact string you need may not be retrieved at all.

BM25, a sparse lexical method, handles exactly this and fails where dense retrieval
succeeds: paraphrase and synonymy.

Use both and fuse the rankings. Hybrid retrieval beats either alone on essentially every
benchmark, and the gap is largest on technical corpora.
:::

```python
def reciprocal_rank_fusion(rankings, k=60):
    """Combine ranked lists without needing comparable scores."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

RRF needs no score calibration between the two systems, which is why it is the default
fusion method.

## Reranking

Retrieve 50–100 candidates cheaply, then rerank the top ones with a **cross-encoder** that
reads query and document together:

```python
# Bi-encoder: embed separately, compare vectors. Fast, less accurate.
score = cosine(embed(query), embed(doc))

# Cross-encoder: one forward pass over the pair. Slow, much more accurate.
score = reranker(f'{query} [SEP] {doc}')
```

The cross-encoder sees interactions between query and document tokens that independent
embeddings cannot represent. It is too slow to run over a whole corpus and ideal over 50
candidates — which is the retrieve-then-rerank pattern.

## Assembling the prompt

```python
def build_prompt(query, chunks):
    context = '\n\n'.join(
        f'[{i+1}] {c["source"]}\n{c["text"]}' for i, c in enumerate(chunks))
    return f"""Answer using only the sources below. Cite them as [1], [2].
If the sources do not contain the answer, say so.

{context}

Question: {query}
Answer:"""
```

Two details that materially change output quality:

**Position.** From lesson 4.13, models use the start and end of a long context more
reliably than the middle. Put the highest-ranked chunk **first or last**, not in the middle.

**Explicit grounding instruction.** "Answer only from the sources" and "say so if absent"
measurably reduce fabrication. Without them the model falls back on parametric knowledge and
you cannot tell which it used.

## Evaluating the parts separately

::: warning
"The answer was wrong" is not a diagnosis. Measure each stage:

- **Retrieval recall@$k$** — is the needed chunk in the retrieved set at all? If not, no
  amount of prompting or model upgrading helps.
- **Reranking precision** — is it in the top 3 after reranking?
- **Faithfulness** — is the answer supported by the retrieved text? Check by asking a
  separate model to verify each claim against the sources.
- **Answer correctness** — the end-to-end metric.

The most common real finding is that recall@10 is around 60%, which caps the whole system at
60% regardless of the generator. People upgrade the LLM and see no improvement, because the
LLM was never the bottleneck.
:::

## Improving retrieval

- **Query rewriting** — have the model rephrase a conversational query into a search query,
  or generate several and union the results.
- **HyDE** — generate a hypothetical answer, embed *that*, and search with it. Answers are
  distributionally closer to documents than questions are.
- **Contextual chunk headers** — prepend a summary of the parent document to each chunk
  before embedding, so a chunk retains context it would otherwise lose.
- **Fine-tune the embedding model** on your domain with contrastive learning (lesson 3.04's
  InfoNCE). Reliably the largest single improvement, and it needs only query–document pairs
  you can mine from logs.

## RAG or long context

From lesson 4.13: retrieval wins for targeted lookup over a corpus, long context wins for
global reasoning over a document that fits. The practical answer is often both — retrieve
aggressively into a long context, so precision matters less.

::: exercise
Your RAG system answers correctly 60% of the time. You upgrade the generator from a 7B to a
70B model and accuracy does not move. What is happening?
:::

::: solution
**The generator was not the bottleneck.** Retrieval almost certainly is.

**Measure recall first.** Take 100 questions with known answers, and for each check whether
the chunk containing the answer appears in the retrieved set:

```python
recall = sum(gold_chunk_id in retrieved_ids(q) for q, gold_chunk_id in eval_set) / len(eval_set)
```

If recall@10 is around 60%, you have found it: for 40% of questions the information is never
in the context, and the model's only options are to say it does not know or to fabricate. No
generator can answer a question it was not given the information for.

**Then measure faithfulness on the cases where retrieval succeeded.** If the model answers
correctly on ~95% of the questions whose chunk *was* retrieved, the generator is fine and
100% of your headroom is in retrieval.

**Fix retrieval, in order of expected gain:**

1. **Add BM25 and fuse.** If the corpus contains identifiers, version numbers, error codes
   or function names, dense-only retrieval is systematically missing them. This is usually
   the largest single win on technical corpora.
2. **Add a cross-encoder reranker** over the top 50. Typically 10–20 points of precision@3.
3. **Check chunking.** Look at the chunks that *should* have been retrieved. If the answer
   straddles a boundary, or the chunk is too large and the embedding is diluted, fix the
   chunking — this is often the real cause and costs nothing to test.
4. **Fine-tune the embedder** on in-domain query–document pairs. Largest gain, most work.

**Also worth ruling out:** that your evaluation is wrong. Read 20 "incorrect" answers by
hand. A substantial fraction of reported RAG failures turn out to be answers that were
correct but phrased differently than the gold string, or questions whose gold answer is
itself wrong.

**The general principle:** in a pipeline, measure each stage. Upgrading the most expensive
component is the most expensive way to discover it was not the problem.
:::

## What to carry forward

- Chunk on semantic boundaries with overlap; 512 tokens is a reasonable default.
- Dense retrieval misses exact strings — always fuse with BM25.
- Retrieve broadly, then rerank with a cross-encoder.
- Put the best chunk first or last, and instruct the model to ground and abstain.
- Measure recall, precision, faithfulness and correctness separately, or you cannot diagnose anything.
