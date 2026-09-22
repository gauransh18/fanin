---
summary: Why generation without a cache is quadratic, what the cache costs in memory, and the bugs that produce plausible wrong output.
prereqs: [gpt-from-scratch, causal-masking]
---

Training is parallel over positions; generation is not. Without a cache, generating $n$
tokens re-computes the entire prefix $n$ times. The KV cache removes that redundancy and
introduces a new bottleneck in its place.

## The redundancy

To generate token $t+1$ you run the model on tokens $1..t$. To generate $t+2$ you run it
on $1..t+1$. The keys and values for positions $1..t$ are **identical** in both passes —
causal attention means position $j$'s key and value depend only on tokens $\le j$, which
have not changed.

Without caching, generating $n$ tokens from a prompt of length $p$ costs

$$
\sum_{t=p}^{p+n} O(t^2) = O((p+n)^3)
$$

With caching, each step processes one new token against a cached prefix:

$$
\sum_{t=p}^{p+n} O(t) = O((p+n)^2)
$$

::: key
A factor of $(p+n)$ — for a 2,000-token generation that is roughly **2,000× less compute**.
Nobody generates without a cache. The question is only how to manage its memory.
:::

## The implementation

```python
import torch

class KVCache:
    """Pre-allocated so generation never reallocates mid-run."""
    def __init__(self, batch, n_heads, max_len, d_head, n_layers, device, dtype):
        shape = (batch, n_heads, max_len, d_head)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.length = 0

    def update(self, layer, k_new, v_new):
        T = k_new.size(2)
        self.k[layer][:, :, self.length:self.length + T] = k_new
        self.v[layer][:, :, self.length:self.length + T] = v_new
        end = self.length + T
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n):
        self.length += n
```

Pre-allocation matters: `torch.cat` on every step reallocates and copies the whole cache,
which turns an $O(1)$ update into $O(t)$ and fragments memory (lesson 2.13).

## Prefill and decode are different workloads

Generation has two phases with completely different characteristics:

| | Prefill | Decode |
|---|---|---|
| Tokens processed | $p$ (the whole prompt) | 1 |
| Parallelism | Full | None |
| Arithmetic intensity | High | Very low |
| Bound by | Compute | Memory bandwidth |

Prefill is one big matrix product — compute-bound, near peak FLOP/s. Decode is a
matrix–vector product per step — memory-bound, at a few percent of peak (lesson 1.03).

This is why every inference optimisation in lesson 7.09 is really about raising decode's
arithmetic intensity: batching many requests together turns the matrix–vector product back
into a matrix–matrix product.

::: check
Generating 2,000 tokens without a KV cache costs roughly how much more compute than with one?

- [x] About 2,000× — $O((p+n)^3)$ against $O((p+n)^2)$
  > Each step re-computes keys and values for the whole prefix, and causal attention means those are identical every time. Nobody generates without a cache; the only question is how to manage its memory.
- [ ] About 2× — the cache saves the backward pass equivalent
  > There is no backward pass in generation. The saving is the entire re-computation of the prefix at every step.
- [ ] About 4× — attention is quadratic, so caching halves the exponent
  > It does reduce the exponent, from 3 to 2, and the resulting factor is $(p+n)$, not a constant.
- [ ] Nothing, if the batch is large enough
  > A large batch improves arithmetic intensity. It does not remove the redundant work.
:::

## The memory cost

$$
\text{cache bytes} = 2 \times B \times L \times h \times T \times d_h \times \text{bytes}
$$

for keys and values, across $L$ layers and $h$ heads.

```python
def kv_bytes(batch, n_layers, n_heads, seq_len, d_head, dtype_bytes=2):
    return 2 * batch * n_layers * n_heads * seq_len * d_head * dtype_bytes

# Llama-2 7B: 32 layers, 32 heads, d_head 128
for b, t in [(1, 4096), (1, 32768), (32, 4096), (64, 8192)]:
    gb = kv_bytes(b, 32, 32, t, 128) / 1e9
    print(f'batch {b:3d}, len {t:6d}  ->  {gb:7.2f} GB')
# batch  1, len   4096  ->    2.15 GB
# batch  1, len  32768  ->   17.18 GB
# batch 32, len   4096  ->   68.72 GB
# batch 64, len   8192  ->  274.88 GB
```

::: warning
At batch 32 with 4k context, the KV cache alone is 69 GB — **more than the model's 14 GB
of bf16 weights**. On an 80 GB card, the cache, not the model, is what caps your batch
size, and batch size is what determines throughput.

This single table is the reason grouped-query attention exists (lesson 4.11), the reason
paged attention exists (lesson 7.09), and the reason long context is expensive to serve
rather than merely expensive to train.
:::

::: check
What does the KV cache become the bottleneck for, once you are using one?

- [x] Serving batch size — the cache, not the weights, is what runs out of memory first
  > Weights are a fixed cost paid once; the cache grows with batch size *and* with sequence length. Every attention variant in lesson 4.11 exists to shrink this number.
- [ ] Arithmetic throughput, since cached attention is compute-bound
  > Decode is strongly memory-bound: one new token against a large cache has almost no arithmetic per byte read.
- [ ] Model quality, since cached keys drift from recomputed ones
  > They are bitwise identical in exact arithmetic and near-identical in practice. Caching is a pure optimisation.
- [ ] Prefill latency, which grows with cache size
  > Prefill builds the cache rather than reading a large one, and is compute-bound.
:::

## The bugs

**RoPE position offset.** During cached decoding you pass one token, but its position is
$t$, not 0. Using position 0 means every generated token is rotated as though it were
first. The model still produces fluent text — it is just wrong, and nothing errors.

```python
offset = cache.length                       # NOT 0
cos, sin = self.cos[offset:offset + T], self.sin[offset:offset + T]
```

**The causal flag.** With a cache and $T=1$, the single query must attend to the entire
cached prefix. Passing `is_causal=True` with a 1×$t$ score matrix masks almost everything:

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None or T > 1))
```

**Stale cache across requests.** Reusing a cache object without resetting `length` mixes
one conversation's keys into another's. In a server this is a correctness *and* a privacy
bug.

::: key
All three produce **plausible, fluent, wrong output** and raise no exception. The only
reliable defence is the comparison from lesson 4.09:

```python
torch.manual_seed(0); cached   = generate(model, prompt, 32, use_cache=True)
torch.manual_seed(0); uncached = generate(model, prompt, 32, use_cache=False)
assert torch.equal(cached, uncached), 'KV cache changes the output'
```

Greedy decoding must produce identical tokens either way. Run this whenever you touch
caching code.
:::

## Shrinking the cache

- **GQA / MQA** (lesson 4.11) — share keys and values across query heads. An 8× reduction
  is typical and is now standard.
- **Quantize the cache** to int8 or fp8. Halves or quarters it with small quality loss;
  keys are more sensitive than values.
- **Sliding window** — keep only the last $w$ tokens. Bounded memory at the cost of losing
  long-range retrieval, and it requires keeping the first few tokens as attention sinks
  (lesson 4.03).
- **Paged attention** — allocate the cache in fixed blocks rather than contiguously,
  eliminating the fragmentation that otherwise wastes 60–80% of reserved memory
  (lesson 7.09).

::: exercise
You serve Llama-2 7B on an 80 GB GPU. Weights take 14 GB in bf16. What batch size can you
run at 4k context, and how much does GQA with 8 KV heads change it?
:::

::: solution
**With full multi-head attention (32 KV heads).**

Per-sequence cache at 4,096 tokens:

$$
2 \times 32\ \text{layers} \times 32\ \text{heads} \times 4096 \times 128 \times 2\ \text{B} = 2.15\ \text{GB}
$$

Budget: 80 GB total, minus 14 GB weights, minus roughly 6 GB for activations, CUDA context
and fragmentation, leaves about 60 GB.

$$
60 / 2.15 \approx \textbf{27 sequences}
$$

**With GQA at 8 KV heads.** Only the KV projections shrink; query heads stay at 32. The
cache is $8/32 = 1/4$ the size:

$$
2 \times 32 \times 8 \times 4096 \times 128 \times 2\ \text{B} = 0.54\ \text{GB per sequence}
$$

$$
60 / 0.54 \approx \textbf{111 sequences}
$$

**A 4× improvement in batch size.** Since decode is memory-bandwidth-bound (the table
above), throughput scales close to linearly with batch size in this regime — each weight
read from HBM is amortised over 4× as many sequences. So this is roughly **4× the tokens
per second** from an architectural change that costs about 1% in quality.

**Two caveats.** Real systems get less than the arithmetic suggests, because contiguous
cache allocation fragments badly — paged attention (lesson 7.09) recovers most of the gap.
And at very large batch sizes decode eventually becomes compute-bound again, at which
point further batching stops helping.

**Why this matters commercially:** serving cost per token falls roughly 4×. That is the
entire reason every model released since Llama-2 70B uses GQA.
:::

## What to carry forward

- Caching turns generation from cubic to quadratic — it is not optional.
- Prefill is compute-bound, decode is memory-bound; they need different optimisations.
- The cache can exceed the model's own weights and is what caps your batch size.
- RoPE offset, the causal flag, and stale state all produce fluent wrong output silently.
- Always diff cached against uncached greedy generation.
