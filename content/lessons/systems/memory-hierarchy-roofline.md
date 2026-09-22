---
summary: The one model that tells you whether an operation is limited by arithmetic or by memory, and therefore which optimisation will help.
prereqs: [gpu-architecture, matrix-multiplication-cost, profiling]
---

Most performance work is wasted because it optimises the resource that was not the
constraint. The roofline model answers that question in one calculation, and it should be
the first thing you compute.

## Arithmetic intensity

$$
I = \frac{\text{FLOPs performed}}{\text{bytes moved to and from HBM}}
$$

Achievable performance is then

$$
P = \min\left(P_{\text{peak}},\; I \times BW_{\text{peak}}\right)
$$

Two regimes, separated by the **ridge point** $I^* = P_{\text{peak}}/BW_{\text{peak}}$:

- $I < I^*$ — **memory-bound**. Performance is $I \times BW$. More FLOPs are free.
- $I > I^*$ — **compute-bound**. Performance is capped by the arithmetic units.

For an H100: $989 \times 10^{12} / 3.35 \times 10^{12} \approx 295$ FLOP/byte.

::: key
**Below the ridge, extra arithmetic costs nothing.** The units are idle waiting for data, so
fusing more work into a kernel, or recomputing instead of storing, is free in wall-clock
terms.

This is the single most useful fact in performance engineering, and it is why FlashAttention
can do *more* FLOPs and run faster (lesson 4.12), and why gradient checkpointing (lesson 7.08)
is cheaper than its 30% FLOP overhead suggests.
:::

## Computing it for common operations

```python
def roofline(name, flops, bytes_moved, peak_flops=989e12, peak_bw=3.35e12):
    I = flops / bytes_moved
    ridge = peak_flops / peak_bw
    achievable = min(peak_flops, I * peak_bw)
    print(f'{name:28s} I={I:8.1f}  {"compute" if I > ridge else "MEMORY ":>7s}-bound'
          f'  ceiling {achievable/1e12:7.1f} TFLOP/s')

n, b = 4096, 2                               # square dim, bytes per element
roofline('matmul 4096^3',  2*n**3,      3*n*n*b)
roofline('matmul batch=1', 2*n*n,       (n*n + 2*n)*b)
roofline('elementwise add', n*n,        3*n*n*b)
roofline('layernorm',       5*n*n,      2*n*n*b)
roofline('softmax',         5*n*n,      2*n*n*b)
```

Typical output:

| Operation | $I$ | Regime |
|---|---|---|
| Large square matmul | 1365 | Compute-bound |
| Matrix–vector (batch 1) | 1.0 | **Memory-bound** |
| Elementwise add | 0.17 | **Memory-bound** |
| LayerNorm | 1.25 | **Memory-bound** |
| Softmax | 1.25 | **Memory-bound** |

::: warning
**Almost everything except large matmuls is memory-bound.** Elementwise operations,
normalizations, activations and softmax all move roughly as many bytes as they do FLOPs.

Two consequences that shape how transformers are optimised:

- Fusing a chain of elementwise operations into one kernel is nearly pure profit — you pay
  one round trip instead of $n$.
- Decode with batch 1 (lesson 4.10) has $I \approx 1$, which is 1/295 of the ridge. The GPU
  is running at well under 1% of its arithmetic peak, and no amount of FLOP reduction helps.
  Only raising the batch size does.
:::

::: check
An H100 has a ridge point around 295 FLOP/byte. What follows for an operation well below it?

- [x] Extra arithmetic is free in wall-clock terms
  > The units are idle waiting for data. This is why FlashAttention can do *more* FLOPs and run faster, and why gradient checkpointing costs far less than its 30% FLOP overhead suggests. It is the single most useful fact in performance engineering.
- [ ] Reducing FLOPs is the highest-leverage optimisation
  > Below the ridge, cutting FLOPs changes nothing — the data movement is the constraint.
- [ ] The kernel is running at peak and cannot be improved
  > It is running at $I \times BW$, which is well below peak. Raising $I$ improves it.
- [ ] Lower precision will not help
  > Lower precision halves the bytes moved, which below the ridge is a direct speedup.
:::

## Fusion, quantified

```python
import torch

n = 8192
x = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
w = torch.randn(n, device='cuda', dtype=torch.bfloat16)

def unfused(x, w):
    a = x * w          # read x, w; write a
    b = a + 1.0        # read a;    write b
    return torch.relu(b)   # read b;    write out

@torch.compile
def fused(x, w):
    return torch.relu(x * w + 1.0)     # one read, one write
```

Unfused moves roughly 6 tensor-sized transfers; fused moves 2. Since both are memory-bound,
the speedup is close to the traffic ratio — about **3×**. That is what `torch.compile`
(lesson 2.14) buys on elementwise chains, and why its gains are largest on models with many
small operations.

## Where the transformer sits

For one transformer layer at model dimension $d$, batch $B$, sequence $T$:

| Component | FLOPs | Bytes | Regime |
|---|---|---|---|
| QKV projection | $6BTd^2$ | $\sim 3d^2 + BTd$ | Compute-bound if $BT \gg d$ |
| Attention scores | $2BT^2d$ | $\sim BT^2$ | $I \approx d$ — compute-bound if $d > 295$ |
| MLP | $16BTd^2$ | $\sim 8d^2 + BTd$ | Compute-bound if $BT \gg d$ |
| LayerNorm | $5BTd$ | $2BTd$ | **Always memory-bound** |
| Residual add | $BTd$ | $3BTd$ | **Always memory-bound** |

The crossover is $BT$ versus $d$. During **training**, $BT$ is large — thousands of tokens —
so projections and the MLP are compute-bound and the GPU runs near peak. During **decode**,
$BT = B$ and if $B$ is small the same operations become memory-bound.

That single difference explains why training reaches 40–50% MFU and single-stream decode
reaches 2–5%.

::: check
You fuse three elementwise kernels into one. On a memory-bound operation, what does that buy?

- [x] Two round trips to HBM are eliminated
  > The intermediates never leave the chip, so the traffic falls by roughly two thirds. The arithmetic is identical. On a memory-bound operation, traffic is the runtime, which is why `torch.compile`'s main win on elementwise chains is fusion.
- [ ] Two kernel launches are saved, which is the main cost
  > Launch overhead is a few microseconds and matters only for very small tensors.
- [ ] The arithmetic is reduced by a factor of three
  > No arithmetic is removed at all.
- [ ] Nothing, unless the operation was compute-bound
  > It is precisely when the operation is memory-bound that fusion pays.
:::

## Deciding what to optimise

::: key
Compute $I$ before doing anything.

**Memory-bound?** Fuse operations, use a lower-precision dtype (fewer bytes per element),
raise the batch size, improve the access pattern for coalescing, keep data in shared memory.
Reducing FLOPs will not help.

**Compute-bound?** Use tensor cores, pick shapes that are multiples of 8 or 16, or find an
algorithm with fewer operations. Reducing memory traffic will not help.

Getting this backwards is the most common way performance work is wasted.
:::

## Measuring it

```python
import time
import torch

def measure(fn, flops, bytes_moved, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    s = (time.perf_counter() - t0) / iters
    print(f'{flops/s/1e12:8.1f} TFLOP/s   {bytes_moved/s/1e9:8.1f} GB/s   {s*1e3:7.3f} ms')
```

Run it and compare both numbers against the hardware's peaks. Near peak bandwidth with low
TFLOP/s confirms memory-bound; near peak TFLOP/s confirms compute-bound. If neither is near
peak, something else is wrong — launch overhead, occupancy, or a fallback kernel.

::: exercise
Your attention implementation achieves 40 TFLOP/s on an H100 during training with $B=8$,
$T=4096$, $h=32$, $d_h=128$. Where is the headroom?
:::

::: solution
**Compute the intensity for the score computation**, per head per batch item:

- FLOPs: $4T^2 d_h = 4 \times 4096^2 \times 128 \approx 8.6\times10^9$.
- HBM traffic, unfused: read $Q$, $K$, $V$ at $3Td_h$ elements; write $S$ at $T^2$; read $S$
  for softmax, write $A$, read $A$ for $AV$ — roughly $3T^2 + 3Td_h$ elements, about
  $5\times10^7$ elements, $10^8$ bytes in bf16.
- $I \approx 8.6\times10^9 / 10^8 \approx 86$ FLOP/byte.

**86 is well below the ridge of 295**, so unfused attention is **memory-bound**. Its ceiling
is $86 \times 3.35$ TB/s $\approx 288$ TFLOP/s — and you are at 40, which is 14% of even
that. So there are two separate problems.

**First: you are probably not using FlashAttention.** Confirm it:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # Raises if the Flash kernel cannot be used for these inputs.
```

If it raises, the reason is in the message — commonly an fp32 dtype, a custom additive bias
(lesson 4.12), or a non-contiguous layout. FlashAttention keeps tiles in shared memory, which
raises $I$ well past the ridge and moves you into the compute-bound regime.

**Second: 40 TFLOP/s is low even for an unfused kernel.** Check in this order:

1. **Dtype.** fp32 means no tensor cores at all — a 15× penalty (lesson 7.01). This alone
   could explain the number.
2. **Contiguity.** The `view` and `transpose` of lesson 4.03 leave tensors strided. A kernel
   that has to materialise a contiguous copy first pays the traffic twice.
3. **Is the causal mask being materialised?** Building a $T \times T$ boolean mask each call
   and writing it to HBM adds $T^2$ bytes of traffic for no arithmetic. Use `is_causal=True`
   rather than an explicit mask.

**Expected after fixing:** with FlashAttention in bf16, 400–600 TFLOP/s on this shape. The
gap from 40 is more than an order of magnitude, which is why this check is worth doing before
any other optimisation.
:::

## What to carry forward

- $I = \text{FLOPs} / \text{bytes}$; compare against the ridge point, $\sim$295 on H100.
- Below the ridge, extra arithmetic is free — fuse and recompute freely.
- Everything but large matmuls is memory-bound, including norms, activations and softmax.
- Training is compute-bound because $BT \gg d$; decode is not, which is the whole serving problem.
- Measure both TFLOP/s and GB/s and compare each to peak before optimising anything.
