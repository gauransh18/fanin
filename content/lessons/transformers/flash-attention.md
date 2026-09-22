---
summary: The algorithm that made long context affordable by never materialising the attention matrix — and the online softmax trick that makes it possible.
prereqs: [scaled-dot-product-attention, kv-caching]
seealso: [memory-hierarchy-roofline]
---

FlashAttention computes exactly the same function as lesson 4.02's four lines. It is 2–4×
faster and uses $O(T)$ memory instead of $O(T^2)$. The speedup comes entirely from
**moving less data**, not from doing less arithmetic.

## The problem is memory traffic

Standard attention writes the $T \times T$ score matrix to HBM, reads it back for the
softmax, writes the normalised weights back, and reads them again for $AV$:

| Step | Reads | Writes |
|---|---|---|
| $S = QK^\top$ | $Q, K$ | $S$ ($T^2$) |
| $A = \text{softmax}(S)$ | $S$ ($T^2$) | $A$ ($T^2$) |
| $O = AV$ | $A$ ($T^2$), $V$ | $O$ |

That is $O(T^2)$ traffic to HBM, four times over. HBM bandwidth is roughly 3 TB/s while
on-chip SRAM is around 20 TB/s — and by lesson 7.02's roofline argument, an operation that
moves this much data per FLOP is bandwidth-bound long before it is compute-bound.

::: key
The arithmetic is $O(T^2 d)$ and the traffic is $O(T^2)$, so arithmetic intensity is $O(d)$
— a constant, independent of $T$. Standard attention gets *no* better at using the hardware
as sequences grow. FlashAttention raises the intensity by keeping intermediates on-chip.
:::

## Tiling

Split $Q$, $K$ and $V$ into blocks small enough that a tile of each fits in SRAM. For each
query block, loop over key blocks, compute that tile of scores **in SRAM**, and accumulate
the output incrementally.

The score matrix is never written to HBM. It exists only as a sequence of small tiles.

The obstacle is softmax: it normalises over the whole row, and you only have one tile of
that row at a time.

## Online softmax

The trick is that a softmax can be updated incrementally, maintaining a running maximum and
a running sum.

Process block $j$ with block maximum $m^{(j)}$ and block sum $\ell^{(j)}$. Given running
state $(m, \ell, O)$, the update is:

$$
m^{\text{new}} = \max(m, m^{(j)})
$$
$$
\ell^{\text{new}} = e^{m - m^{\text{new}}}\ell + e^{m^{(j)} - m^{\text{new}}}\ell^{(j)}
$$
$$
O^{\text{new}} = e^{m - m^{\text{new}}}\,O + e^{m^{(j)}-m^{\text{new}}}\,\tilde{O}^{(j)}
$$

The exponential factors **rescale** everything accumulated so far to the new maximum. This
is exactly the numerically stable softmax of lesson 2.07, made incremental — and it is
algebraically exact, not an approximation.

```python
import torch

def flash_attention_reference(q, k, v, block=128):
    """Readable reference for the algorithm. The real kernel fuses this
    into one CUDA/Triton launch; this is a numerical demonstration."""
    B, h, T, d = q.shape
    scale = d ** -0.5
    out = torch.zeros_like(q)

    for i in range(0, T, block):
        q_i = q[:, :, i:i + block]                       # query tile
        m = torch.full((B, h, q_i.size(2), 1), -float('inf'), device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros_like(q_i)

        for j in range(0, T, block):
            if j > i + block:                             # causal: skip blocks
                break                                     # entirely above the diagonal
            k_j, v_j = k[:, :, j:j + block], v[:, :, j:j + block]

            s = (q_i @ k_j.transpose(-2, -1)) * scale
            if j + block > i:                             # partial block: mask inside
                qi = torch.arange(i, i + q_i.size(2), device=q.device)[:, None]
                kj = torch.arange(j, j + k_j.size(2), device=q.device)[None, :]
                s = s.masked_fill(kj > qi, -float('inf'))

            m_j = s.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_j)
            p = torch.exp(s - m_new)
            scale_old = torch.exp(m - m_new)

            l = scale_old * l + p.sum(dim=-1, keepdim=True)
            acc = scale_old * acc + p @ v_j
            m = m_new

        out[:, :, i:i + block] = acc / l
    return out


q, k, v = (torch.randn(1, 2, 512, 64) for _ in range(3))
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print((flash_attention_reference(q, k, v) - ref).abs().max())   # ~1e-6
```

::: check
FlashAttention is 2–4× faster than standard attention. Where does the speedup come from?

- [x] Moving less data — the $T\times T$ score matrix never reaches HBM, so arithmetic intensity rises
  > It computes exactly the same function with exactly the same FLOPs. Standard attention has intensity $O(d)$, a constant independent of $T$, so it never gets better at using the hardware as sequences grow.
- [ ] Doing fewer FLOPs by skipping low-weight score entries
  > Nothing is skipped; the result is numerically the same, not an approximation.
- [ ] Running the softmax in lower precision
  > The online softmax is carefully kept accurate. Precision is not the lever.
- [ ] Parallelising across heads, which standard attention does not
  > Standard attention already parallelises across heads.
:::

## Recomputation in the backward pass

The backward pass needs the attention weights, which were never stored. FlashAttention
**recomputes** them tile by tile from the saved $Q$, $K$, $V$ and the row statistics
$(m, \ell)$.

This is gradient checkpointing (lesson 7.08) applied inside a single operation: pay extra
FLOPs to avoid $O(T^2)$ memory. Because the operation was bandwidth-bound to begin with,
the extra arithmetic is close to free, and the backward pass is *also* faster than the
standard implementation.

## What it buys

| Sequence length | Standard memory | Flash memory | Typical speedup |
|---|---|---|---|
| 1,024 | 4 MB | 0.1 MB | 1.5× |
| 4,096 | 64 MB | 0.4 MB | 2.5× |
| 16,384 | 1 GB | 1.6 MB | 3.5× |
| 65,536 | 16 GB | 6.4 MB | 4×+ |

(Per head per batch item, bf16.) The memory column is why 100k-token context became
possible at all — the FLOPs were always affordable; the $T^2$ allocation was not.

## Using it

```python
import torch.nn.functional as F

# Dispatches to FlashAttention when shapes, dtype and hardware allow.
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Check which backend was selected:
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # Raises if the Flash kernel cannot be used for these inputs.
```

::: warning
The fused kernel silently falls back to a slower path when its preconditions are not met.
Common causes: fp32 inputs (it needs fp16 or bf16), a head dimension above 256, a custom
additive attention bias, or a non-contiguous layout.

If you added a custom bias — an ALiBi or T5-style relative position term (lesson 4.04) —
you have left the fast path. This is a large part of why RoPE won: it modifies $q$ and $k$
before the kernel rather than adding to the scores inside it.

Use `sdpa_kernel` to turn the silent fallback into an error while developing.
:::

::: check
Tiling attention runs into one obstacle. What is it, and what solves it?

- [x] Softmax normalises over a whole row, but only one tile of that row is on chip at a time — an online softmax with a running maximum and running sum fixes it
  > Each new block rescales the accumulated output by $e^{m - m^{\text{new}}}$, so the result is exact rather than approximate.
- [ ] The causal mask cannot be applied tile by tile
  > It applies cleanly per tile, and whole tiles above the diagonal can be skipped entirely.
- [ ] Query blocks and key blocks must be the same size
  > They need not be; the block sizes are tuned independently against SRAM capacity.
- [ ] Tiles cannot be written back to HBM in parallel
  > The output is accumulated per query block and written once. Parallel writes are not the difficulty.
:::

## Versions

- **FlashAttention-1** introduced tiling and online softmax.
- **FlashAttention-2** improved work partitioning across thread blocks and reduced
  non-matmul FLOPs — roughly 2× over v1.
- **FlashAttention-3** exploits Hopper's asynchrony and fp8, reaching 75% of peak on H100.

The algorithm is unchanged across versions; the improvements are all about mapping it to
hardware. That is a reasonable summary of where performance work in this field goes.

::: exercise
FlashAttention does *more* FLOPs than standard attention, because it recomputes the
weights in the backward pass. Explain why it is nonetheless faster.
:::

::: solution
**Because attention was never compute-bound.**

Standard attention on $T = 4096$, $d = 64$, per head:

- FLOPs: $4T^2d \approx 4.3\times10^9$.
- HBM traffic: $\sim 3T^2$ elements at 2 bytes $\approx 100$ MB.
- Arithmetic intensity: $4.3\times10^9 / 10^8 \approx 43$ FLOP/byte.

An A100 has about 312 TFLOP/s of bf16 and 2 TB/s of bandwidth, so its roofline ridge point
(lesson 7.02) is $312/2 = 156$ FLOP/byte. At 43, attention sits **well below** the ridge —
it is bandwidth-bound, and the GPU's matrix units are idle waiting for data.

**FlashAttention trades the abundant resource for the scarce one.** It does roughly 1.3×
the FLOPs (the recomputation) while cutting HBM traffic by a factor of order $T/\text{block}$
— from $O(T^2)$ down to $O(T^2 / B_c + Td)$ with tiles kept in SRAM. Intensity rises well
past the ridge, so the kernel becomes compute-bound and the extra arithmetic runs on units
that were previously stalled.

**The general lesson**, which lesson 7.02 makes into a method: when an operation is
bandwidth-bound, extra arithmetic is *free*. Fusing more work into a kernel, or recomputing
instead of storing, costs nothing in wall-clock until you cross the ridge point. Check
which side of it you are on before optimising — the intuitive move of "do fewer FLOPs" is
the wrong optimisation for most of the operations in a transformer that are not matrix
multiplies.
:::

## What to carry forward

- Same output, same FLOPs-plus-recompute, $O(T)$ memory instead of $O(T^2)$.
- The win is HBM traffic: tiles stay in SRAM and the score matrix is never written.
- Online softmax with a running max and sum makes tiled normalisation exact.
- A custom additive bias drops you off the fast path — RoPE does not.
- Below the roofline ridge, extra arithmetic is free.
