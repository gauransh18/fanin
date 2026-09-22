---
summary: MQA, GQA and MLA — three ways to shrink the KV cache, and the quality cost of each.
prereqs: [kv-caching, multi-head-attention]
---

Lesson 4.10 established that the KV cache, not the weights, caps serving batch size. Every
variant here attacks that number. They differ in how much they shrink it and how much
quality they give back.

## Multi-query attention

Keep $h$ query heads; use **one** key head and **one** value head, shared by all queries.

$$
\text{head}_i = \text{Attention}(XW_Q^i,\; XW_K,\; XW_V)
$$

The cache shrinks by a factor of $h$ — typically 32×. The quality cost is real: every query
head now retrieves from the same key space, so heads can differ in *what they look for* but
not in *what is findable*.

## Grouped-query attention

The compromise that won. Partition the $h$ query heads into $g$ groups, each sharing one KV
head:

$$
g = h \;\Rightarrow\; \text{full MHA}, \qquad g = 1 \;\Rightarrow\; \text{MQA}
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv = n_heads, n_kv_heads
        self.d_head = d // n_heads
        self.n_rep = n_heads // n_kv_heads          # queries per KV head

        self.q_proj = nn.Linear(d, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, d = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv,    self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv,    self.d_head).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v)            # only n_kv heads are stored

        # Expand KV heads to match query heads. expand() is a stride-0 view,
        # so it costs no memory (lesson 2.03).
        k = k[:, :, None].expand(B, self.n_kv, self.n_rep, -1, self.d_head)
        v = v[:, :, None].expand(B, self.n_kv, self.n_rep, -1, self.d_head)
        k = k.reshape(B, self.n_heads, -1, self.d_head)
        v = v.reshape(B, self.n_heads, -1, self.d_head)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None))
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, d))
```

::: key
The `expand` is the detail that makes GQA cheap. It is a stride-0 view (lesson 2.03), so
the KV heads are *logically* repeated without being copied — the cache stores $g$ heads and
the attention kernel sees $h$. Materialising the expansion with `repeat` instead would
undo the entire memory saving.

Production kernels go further and consume the grouped layout directly, avoiding even the
logical expansion.
:::

| Config | KV heads | Cache size | Typical quality |
|---|---|---|---|
| MHA | 32 | 1.0× | baseline |
| GQA-8 | 8 | 0.25× | within noise |
| GQA-4 | 4 | 0.125× | ~0.3% worse |
| MQA | 1 | 0.031× | ~1–2% worse |

GQA with 8 groups is the standard choice: a 4× cache reduction for a quality difference
that does not reliably exceed seed noise (lesson 2.15). Llama-2 70B, Llama-3, Mistral and
essentially every model since use it.

::: check
Multi-query attention shrinks the KV cache by a factor of $h$, typically 32×. What does it give up?

- [x] Every query head now retrieves from one shared key space
  > Heads can differ in what they look for but not in what is findable. Grouped-query attention is the compromise that won: $g$ groups of query heads, each with its own KV head, with $g = h$ recovering full MHA and $g = 1$ recovering MQA.
- [ ] It reduces the number of query heads, cutting the model's capacity to attend
  > Query heads are untouched. Only the key and value heads are shared.
- [ ] It requires retraining from scratch and cannot be retrofitted
  > Uptraining an existing MHA model into GQA with a small amount of extra training is standard practice.
- [ ] It makes attention no longer parallelisable across heads
  > Heads remain fully parallel; they simply read from shared keys and values.
:::

## Multi-head latent attention

DeepSeek's approach takes the low-rank idea from lesson 1.04 and applies it to the cache
itself. Instead of storing $K$ and $V$, store a compressed latent $\mathbf{c}_t$ and
reconstruct on the fly:

$$
\mathbf{c}_t = W_{DKV}\,\mathbf{h}_t, \qquad
\mathbf{k}_t = W_{UK}\,\mathbf{c}_t, \qquad
\mathbf{v}_t = W_{UV}\,\mathbf{c}_t
$$

with $\dim(\mathbf{c}) \ll h \cdot d_h$. The cache holds only $\mathbf{c}$.

The clever part is that the up-projections can be **absorbed** into the neighbouring
weights at inference time. Since

$$
\mathbf{q}^\top \mathbf{k} = \mathbf{q}^\top W_{UK}\mathbf{c} = (W_{UK}^\top\mathbf{q})^\top \mathbf{c}
$$

you can fold $W_{UK}$ into $W_Q$ once and never reconstruct $\mathbf{k}$ at all. This is
exactly lesson 1.03's re-bracketing argument, applied to save memory rather than FLOPs.

MLA reports a cache roughly 1/14 the size of MHA with quality **better** than MHA — the
low-rank constraint appears to act as a useful regulariser, much as lesson 3.15 would
predict.

::: warning
MLA interacts awkwardly with RoPE. Rotation is position-dependent, so it cannot be absorbed
into a position-independent weight matrix. DeepSeek's solution splits the head dimension: a
compressed part carrying no position, and a small uncompressed part carrying RoPE. This is
fiddly, and it is the main reason MLA has not been adopted more widely despite its results.
:::

## Sliding window attention

Restrict each position to the previous $w$ tokens. Cache memory becomes $O(w)$ rather than
$O(T)$, and attention cost becomes linear in sequence length.

```python
import torch

def sliding_window_mask(T, window, device):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (j > i - window)
```

Information still propagates further than $w$ through **layer stacking**: with a window of
4,096 and 32 layers, the theoretical receptive field is $32 \times 4096 = 131$k tokens —
the same receptive-field arithmetic as lesson 3.10, including its caveat that the effective
field is much smaller than the theoretical one.

Mistral 7B uses a 4,096 window. Several models interleave sliding-window and full-attention
layers, getting bounded memory for most layers and true global retrieval in a few.

## Choosing

| Goal | Choice |
|---|---|
| Serve at high batch size | **GQA-8** — near-free 4× |
| Extreme memory pressure | MQA, or MLA if you can implement it |
| Very long context | Sliding window plus some full layers |
| Maximum quality, batch of 1 | Plain MHA |

::: exercise
You have a 7B model with 32 query heads. Compare GQA-8 against MQA on cache size,
parameter count, and the arithmetic intensity of decoding.
:::

::: solution
**Cache size.** The cache scales with KV heads. GQA-8 is $8/32 = 1/4$ of MHA; MQA is
$1/32$. So MQA's cache is **8× smaller than GQA-8's**. At 4k context on Llama-2 7B that is
0.54 GB against 0.067 GB per sequence.

**Parameters.** Only the K and V projections shrink. With $d = 4096$ and $d_h = 128$:

- $W_Q$: $4096 \times 4096 = 16.8$M in both cases.
- GQA-8: $W_K$ and $W_V$ are $4096 \times (8 \times 128) = 4.2$M each.
- MQA: $4096 \times 128 = 0.52$M each.

Per layer, GQA-8 saves 25.2M against MHA; MQA saves 32.2M. Across 32 layers that is 806M
versus 1.03B — a 3% difference in total model size. **Parameters are not the reason to
choose either**; the cache is.

**Arithmetic intensity, which is the interesting part.** During decode the attention step
reads the whole cache to process one query. Its intensity is

$$
I \approx \frac{2 \cdot h \cdot T \cdot d_h}{2 \cdot g \cdot T \cdot d_h \cdot 2\ \text{bytes}} = \frac{h}{2g}
$$

- MHA ($g = h$): $I = 0.5$ — deeply memory-bound.
- GQA-8 ($g = 8$, $h = 32$): $I = 2$.
- MQA ($g = 1$): $I = 16$.

So MQA does 32× more arithmetic per byte read. Since decode is bandwidth-bound
(lesson 4.10), that translates fairly directly into throughput.

**Which to pick.** GQA-8, unless memory is the hard binding constraint. MQA's remaining 8×
cache saving buys less than it appears — at GQA-8's batch sizes the system is often already
approaching compute-bound, so the extra headroom does not convert to throughput — and it
costs 1–2% quality, which is above seed noise and therefore real. GQA is the better point
on the curve, which is why essentially every recent model sits there.
:::

::: check
In grouped-query attention with $h$ query heads and $g$ KV heads, what do the two extremes correspond to?

- [x] $g = h$ is full multi-head attention, and $g = 1$ is multi-query attention
  > GQA is the dial between them, and the reason it won is that a handful of KV groups recovers most of MHA's quality at close to MQA's cache size.
- [ ] $g = h$ is multi-query and $g = 1$ is full multi-head
  > It is the other way around: one KV head shared by everything is the multi-*query* case.
- [ ] $g = 0$ disables the cache entirely
  > There is no $g = 0$; every query head needs keys to attend to.
- [ ] Both extremes give the same cache size, and only quality differs
  > The cache size differs by a factor of $h$, which is the whole motivation.
:::

## What to carry forward

- The cache scales with KV heads, not query heads — that is the lever.
- GQA-8 is a 4× cache reduction for quality within seed noise; it is the default.
- Expand KV heads with `expand` (stride-0), never `repeat`.
- MLA compresses the cache with a low-rank latent and absorbs the up-projection into $W_Q$.
- Sliding windows bound memory; layer stacking recovers some long-range propagation.
