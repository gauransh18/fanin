---
summary: Writing fused GPU kernels in Python — the block-level abstraction, autotuning, and a working fused softmax.
prereqs: [cuda-kernels, memory-hierarchy-roofline]
---

Triton sits between PyTorch and CUDA. You write Python, think in **blocks** rather than
threads, and the compiler handles coalescing, shared memory and vectorisation — typically
reaching 80–95% of hand-tuned CUDA.

## Thinking in blocks

A CUDA kernel is written from one thread's point of view. A Triton kernel is written from one
**block's** point of view, operating on arrays:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                       # which block am I?
    offsets = pid * BLOCK + tl.arange(0, BLOCK)       # a vector of indices
    mask = offsets < n                                # bounds guard, vectorised

    x = tl.load(x_ptr + offsets, mask=mask)           # loads a whole block
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

Two differences from CUDA worth noting. `tl.arange(0, BLOCK)` produces a **vector** of
offsets, so `tl.load` fetches a block at once and the compiler arranges coalescing. And the
mask is applied to the whole vector — no per-thread `if`, so no divergence to reason about.

`BLOCK: tl.constexpr` marks a compile-time constant. Triton specialises the kernel per value,
which is what enables autotuning.

## Fused softmax

The canonical example, because softmax is memory-bound (lesson 7.02) and PyTorch's
implementation reads the row several times:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < n_cols

    # One read of the row, into registers.
    x = tl.load(in_ptr + row * in_stride + col_offsets,
                mask=mask, other=-float('inf'))

    # Subtract the max for stability (lesson 2.07) -- all in registers.
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)

    tl.store(out_ptr + row * out_stride + col_offsets, out, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK < 2048 else 8,
    )
    return out
```

::: key
The whole row lives in registers between the load and the store. PyTorch's version makes
several passes over HBM — max, subtract, exp, sum, divide — and this makes one.

By the roofline model that is a 3–4× reduction in traffic on a memory-bound operation, and
the measured speedup is close to that. Triton kernels like this one are why
`torch.compile`'s Inductor backend generates Triton (lesson 2.14).
:::

::: check
What is the main conceptual difference between writing a CUDA kernel and a Triton kernel?

- [x] CUDA is written from one thread's point of view; Triton is written from one block's, operating on whole arrays with the compiler handling coalescing, shared memory and vectorisation
  > `tl.arange(0, BLOCK)` produces a vector of offsets, so a single `tl.load` fetches a block. That shift is what gets you to 80–95% of hand-tuned CUDA in Python.
- [ ] Triton runs on the CPU and CUDA on the GPU
  > Both compile to GPU code. Triton emits PTX.
- [ ] Triton is interpreted, so it is slower but easier to debug
  > It is JIT-compiled, and the performance is the point.
- [ ] CUDA supports fusion and Triton does not
  > Fusion is one of the main reasons to reach for Triton at all.
:::

## Autotuning

Block size, warp count and pipeline stages interact with shape and hardware in ways that are
not predictable. Let the compiler search:

```python
import triton

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],          # re-tune when these change
)
@triton.jit
def matmul_kernel(...):
    ...
```

The first call for a new shape benchmarks every config, which takes seconds. Results are
cached per `key`, so subsequent calls are free. If your shapes vary every step, autotuning
will dominate — the same recompilation trap as lesson 2.14.

## A blocked matmul

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulate in fp32 even for bf16 inputs -- this is what tensor cores do
    # internally and what keeps long K reductions accurate (lesson 2.12).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)                       # maps to tensor cores
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

`tl.dot` compiles to tensor-core instructions (lesson 7.01), which is what makes this
competitive. The accumulator stays fp32 throughout and is cast only on the store.

::: check
Triton's `@triton.autotune` tries several block sizes and caches the best. Why is that more than a convenience?

- [x] The optimal block size depends on the shapes, the dtype and the specific GPU, so a hand-picked constant is right for one configuration and wrong for the rest
  > It is the main reason a Triton kernel can reach 80–95% of hand-tuned CUDA without hand-tuning: the search is done once per shape, at runtime, on the actual hardware.
- [ ] It compiles the kernel ahead of time, removing JIT overhead
  > Autotuning adds compile passes; it does not remove them.
- [ ] It selects between Triton and cuBLAS implementations
  > It searches Triton configurations only.
- [ ] It guarantees the kernel is numerically identical across block sizes
  > Different reduction orders can differ in the last bits, which is a known caveat rather than a guarantee.
:::

## Where Triton fits

| | PyTorch | Triton | CUDA |
|---|---|---|---|
| Effort | Minimal | Moderate | High |
| Typical peak achieved | Varies | 80–95% | 100% |
| Custom fusion | No | Yes | Yes |
| Warp-level primitives | No | Limited | Full |
| Debuggability | Good | Moderate | Poor |

FlashAttention's reference implementation is written in Triton. So is most of Inductor's
generated code. It is the default choice for a custom kernel, and CUDA is the fallback when
Triton's abstractions do not fit.

::: warning
**Triton debugging is harder than PyTorch debugging.** There is no Python debugger inside a
kernel. Two things help:

`TRITON_INTERPRET=1` runs kernels in a Python interpreter — very slow, and it lets you print
and inspect intermediate values.

And always validate against a PyTorch reference:

```python
torch.testing.assert_close(my_triton_op(x), reference_op(x), rtol=1e-3, atol=1e-3)
```

A kernel that is fast and wrong is worse than no kernel. Write the test first.
:::

::: exercise
You write a Triton kernel fusing LayerNorm and a residual add. It matches PyTorch numerically
but is only 1.2× faster, where the traffic analysis predicted 2.5×. Where is the gap?
:::

::: solution
**Check the traffic arithmetic first.** Unfused: LayerNorm reads $x$ (possibly twice, for
mean and variance) and writes the output; the residual add reads two tensors and writes one.
Roughly 6 tensor passes. Fused: read $x$ and the residual, write the output — 3 passes. So 2×
is the ceiling, and 2.5× was optimistic unless PyTorch was making an extra pass.

**Then investigate the gap from 2× to 1.2×, in this order:**

1. **Is the block size right?** If `BLOCK` is much larger than the row width, most lanes are
   masked and you are moving useful data at a fraction of the achievable rate. Use
   `triton.next_power_of_2(n_cols)` and set `num_warps` accordingly — 4 for small rows, 8 or
   16 for large ones. Autotune over both.

2. **Are you actually memory-bound?** Measure achieved GB/s and compare to peak (lesson 7.02).
   At 90% of peak bandwidth, the kernel is optimal and the remaining gap is in the baseline
   measurement, not your kernel. At 40%, there is real headroom.

3. **Is the reduction inefficient?** LayerNorm needs mean and variance, which are two
   reductions over the row. A naive two-pass implementation reads the row twice. Use Welford's
   algorithm or the $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ form in one pass — with the caveat
   from lesson 1.11 about cancellation, which is why you compute in fp32.

4. **Is PyTorch already fusing it?** If the baseline is running under `torch.compile`,
   Inductor has probably generated a similar Triton kernel already, and you are comparing two
   fused implementations. Profile and check the kernel names. This is the most common
   explanation for a disappointing speedup and it costs two minutes to rule out.

5. **Launch overhead.** If the tensors are small, a 5 μs launch dominates. Check the kernel's
   duration in `nsys` — under about 20 μs and overhead is a significant fraction.

**Order:** check 4 first (is the baseline already fused?), then 2 (am I at bandwidth?). Those
two answer whether there is anything left to win before you spend time tuning.
:::

## What to carry forward

- Triton kernels operate on blocks, not threads; the compiler handles coalescing.
- Keep intermediates in registers — that is where the memory-bound speedup comes from.
- `tl.constexpr` parameters enable autotuning; cache by shape or pay for re-tuning.
- Accumulate in fp32 and cast on store.
- Validate against a PyTorch reference before measuring speed.
