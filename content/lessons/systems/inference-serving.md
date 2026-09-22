---
summary: Why serving is a different problem from training, continuous batching, paged attention, and the latency-throughput tradeoff you must choose explicitly.
prereqs: [kv-caching, memory-hierarchy-roofline, attention-variants]
---

Training is compute-bound and offline. Serving is memory-bound, online, and has latency
targets. Almost every technique here exists to raise arithmetic intensity during decode.

## The problem, restated

From lesson 4.10, generation has two phases:

| | Prefill | Decode |
|---|---|---|
| Tokens per step | $p$ (whole prompt) | 1 |
| Arithmetic intensity | High | $\approx 1$ |
| Bound by | Compute | **Memory bandwidth** |

At batch 1, decode reads every weight from HBM to produce one token. For a 7B model in bf16
that is 14 GB per token. At 3.35 TB/s, the floor is 4.2 ms per token — **239 tokens/second,
no matter how fast the arithmetic is.**

::: key
Batching is the fix, and it is nearly free. Processing $B$ sequences reads the same 14 GB
once and does $B$ times the arithmetic. Arithmetic intensity rises linearly with $B$, and so
does throughput, until you cross the roofline ridge (lesson 7.02) and become compute-bound.

Everything else in this lesson is about making large batches possible.
:::

## Continuous batching

Static batching runs a batch to completion before starting the next. Since sequences finish
at different lengths, the short ones idle waiting for the longest.

Continuous batching (also called in-flight batching) evicts finished sequences and admits new
ones **every step**:

```python
class ContinuousBatchScheduler:
    def __init__(self, model, max_batch=64, max_tokens=8192):
        self.model, self.max_batch, self.max_tokens = model, max_batch, max_tokens
        self.running, self.waiting = [], []

    def step(self):
        # Retire finished sequences, freeing their cache blocks.
        for seq in [s for s in self.running if s.finished]:
            self.running.remove(seq)
            seq.free_blocks()

        # Admit new work while capacity allows.
        while self.waiting and len(self.running) < self.max_batch:
            candidate = self.waiting[0]
            if self.tokens_in_use() + candidate.prompt_len > self.max_tokens:
                break                                # no room; wait
            self.running.append(self.waiting.pop(0))

        return self.model.decode_step(self.running)  # one token for every sequence
```

The gain is 2–4× throughput over static batching on realistic traffic, and it is the single
biggest improvement available.

::: check
At batch 1, decoding a 7B bf16 model reads 14 GB per token. On 3.35 TB/s of bandwidth, what is the floor?

- [x] About 4.2 ms per token — roughly 239 tokens/second, however fast the arithmetic is
  > Batching is the fix and it is nearly free: $B$ sequences read the same 14 GB once and do $B$ times the arithmetic, so throughput rises linearly with $B$ until you cross the roofline ridge. Everything else in serving is about making large batches possible.
- [ ] About 4.2 μs per token, which is negligible
  > Three orders of magnitude out — 14 GB at 3.35 TB/s is milliseconds, not microseconds.
- [ ] It depends on the model's FLOP count, not its size
  > Decode is memory-bound, so bytes read is the binding quantity, not arithmetic.
- [ ] There is no floor; a faster GPU removes it
  > A faster GPU with the same bandwidth has the same floor. Bandwidth is what moves it.
:::

## Paged attention

A contiguous KV cache per sequence must be allocated for the *maximum* possible length. A
sequence that stops at 200 tokens with a 4,096 reservation wastes 95% of its allocation.
Measured waste in naive servers is 60–80%.

Paged attention borrows virtual memory's solution: allocate the cache in fixed-size **blocks**
(typically 16 tokens) from a shared pool, with a per-sequence block table mapping logical
positions to physical blocks.

```python
class PagedKVCache:
    def __init__(self, n_blocks, block_size, n_layers, n_kv_heads, d_head, device):
        self.block_size = block_size
        shape = (n_blocks, block_size, n_kv_heads, d_head)
        self.k = [torch.zeros(shape, device=device, dtype=torch.bfloat16)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=torch.bfloat16)
                  for _ in range(n_layers)]
        self.free = list(range(n_blocks))
        self.tables = {}                      # seq_id -> [physical block ids]

    def allocate(self, seq_id, n_tokens):
        need = (n_tokens + self.block_size - 1) // self.block_size
        if len(self.free) < need:
            return False                      # trigger preemption
        self.tables.setdefault(seq_id, []).extend(self.free[:need])
        self.free = self.free[need:]
        return True

    def release(self, seq_id):
        self.free.extend(self.tables.pop(seq_id, []))
```

Waste falls to under 4% — the fragmentation within the last partial block and nothing more.
In practice this is a 2–4× increase in concurrent sequences, which by the argument above is a
2–4× throughput increase.

**Copy-on-write sharing** falls out of the block table for free: several sequences sampled
from the same prompt share the prompt's physical blocks until they diverge. This makes
best-of-$n$ sampling (lesson 5.11) far cheaper than $n$ independent generations.

## Prefix caching

Requests often share a long prefix — a system prompt, a document, a few-shot preamble. Cache
its KV once and reuse it, turning an $O(p)$ prefill into $O(1)$.

```python
def hash_prefix(tokens, block_size=16):
    """Hash each block cumulatively so any shared prefix length matches."""
    hashes, running = [], hashlib.sha256()
    for i in range(0, len(tokens) - block_size + 1, block_size):
        running.update(bytes(tokens[i:i + block_size]))
        hashes.append(running.hexdigest())
    return hashes
```

For chat applications where every turn resends the conversation, this is an enormous saving —
each turn's prefill is proportional to the *new* tokens, not the whole history.

## The tradeoff you must choose

::: warning
**Throughput and latency are in direct tension, and you cannot optimise both.**

Larger batches raise tokens/second across all users and raise the time each individual user
waits for their next token. There is no configuration that is best for both.

| Metric | Definition | Matters for |
|---|---|---|
| TTFT | Time to first token | Perceived responsiveness |
| TPOT | Time per output token | Streaming speed |
| Throughput | Total tokens/s across users | Cost per token |

Decide which you are optimising *before* tuning. An interactive assistant needs TPOT under
about 50 ms (faster than reading speed); a batch summarisation job cares only about
throughput and should run at the largest batch that fits.
:::

**Chunked prefill** addresses the worst interaction between them: a long prefill blocks
decode for every running sequence, spiking their TPOT. Splitting the prefill into chunks and
interleaving decode steps between them keeps latency smooth at a small throughput cost.

::: check
Paged attention stores the KV cache in fixed-size blocks rather than one contiguous per-sequence buffer. What does that fix?

- [x] Fragmentation and over-allocation — a contiguous buffer must be sized for the maximum possible length, so most of it sits reserved and unused
  > Blocks are allocated on demand and shared between sequences with a common prefix, which raises the batch size you can fit and therefore throughput.
- [ ] The $O(T^2)$ cost of attention over long sequences
  > Paging is a memory-layout change and does not alter the attention computation.
- [ ] Numerical drift between cached and recomputed keys
  > There is no such drift; caching is exact.
- [ ] The need to recompute the prompt for each new request
  > That is prefix caching, which paging makes practical but is a distinct idea.
:::

## What to measure

```python
metrics = {
    'ttft_p50': ..., 'ttft_p99': ...,
    'tpot_p50': ..., 'tpot_p99': ...,
    'throughput_tok_s': ...,
    'batch_size_mean': ...,          # how full is the batch, really?
    'kv_utilisation': ...,           # fraction of cache blocks in use
    'preemption_rate': ...,          # sequences evicted for lack of cache
    'queue_depth': ...,
}
```

Report **p99, not the mean**. A mean TTFT of 200 ms with a p99 of 8 seconds is a bad
experience for one request in a hundred, and the mean conceals it entirely.

`preemption_rate` above a few percent means the cache is undersized for the traffic — raise
it, or lower `max_batch`.

## The stack

vLLM, TensorRT-LLM, SGLang and TGI all implement continuous batching and paged attention.
Differences are in kernel quality, quantization support, structured output and scheduling
policy. Build your own only to learn; use one of these in production.

::: exercise
Your server does 1,200 tokens/s at batch 8 and 1,400 at batch 32. You expected near-linear
scaling. Where is the loss?
:::

::: solution
**4× the batch for 1.17× the throughput** means you left the memory-bound regime early, or
something is capping the effective batch.

**Check, in this order:**

1. **Is the effective batch actually 32?** Log `batch_size_mean`. With static batching or a
   scheduler that cannot admit new work, the nominal maximum of 32 may correspond to a real
   average of 10 — sequences finish and their slots go unused until the whole batch drains.
   *Fix:* continuous batching. This is the most common cause and the largest available win.

2. **Is the KV cache limiting concurrency?** Log `kv_utilisation` and `preemption_rate`. If
   preemption is nonzero, the scheduler wanted batch 32 and could not hold 32 caches. *Fix:*
   paged attention to recover the 60–80% fragmentation waste, GQA if the model supports it
   (lesson 4.11), or KV quantization (lesson 5.12).

3. **Are you now compute-bound?** Compute arithmetic intensity (lesson 7.02). At batch 32,
   $I \approx 32$ — still well below the ridge of 295, so you should *not* be compute-bound
   yet. If measured TFLOP/s is near peak, something is wrong with the estimate; if it is low
   and bandwidth is also low, neither resource is saturated and the bottleneck is elsewhere —
   scheduling overhead or Python.

4. **Is per-step overhead dominating?** At batch 8 the model step might be 10 ms and the
   Python scheduling 2 ms. At batch 32 the model step is 12 ms and scheduling is 8 ms because
   it loops over sequences. *Check:* profile the host side. *Fix:* vectorise the scheduler,
   or use CUDA graphs to eliminate launch overhead.

5. **Are long sequences blocking?** One 8,000-token prefill stalls every decode step behind
   it. This shows as high p99 TPOT alongside mediocre throughput. *Fix:* chunked prefill.

**The expected shape once fixed:** throughput should scale close to linearly with batch until
$I$ approaches the ridge — so batch 32 should be near 4× batch 8, not 1.17×. A gap this large
is a configuration problem, not a hardware limit.
:::

## What to carry forward

- Decode is bandwidth-bound at batch 1; batching is the only lever that matters.
- Continuous batching admits and retires sequences every step — 2–4× over static.
- Paged attention cuts cache waste from 60–80% to under 4%, and gives prefix sharing free.
- Throughput and latency trade against each other; choose which you are optimising.
- Report p99, and watch preemption rate and mean batch size.
