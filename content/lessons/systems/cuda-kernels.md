---
summary: Writing a kernel from scratch — the thread indexing, the shared-memory reduction pattern, and when hand-writing one is actually worth it.
prereqs: [gpu-architecture, memory-hierarchy-roofline]
---

Most performance problems are solved by using a better library kernel. Occasionally you need
one that does not exist — a fused operation, an unusual layout, a custom backward. This
lesson is enough CUDA to write it and to read someone else's.

## The programming model

A kernel is code that every thread runs. Each thread computes which data it is responsible
for from its own indices:

```cpp
__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    if (i < n) {                                      // guard: n need not divide evenly
        out[i] = a[i] + b[i];
    }
}
```

The guard is not optional. You launch $\lceil n/256 \rceil$ blocks of 256 threads, so the
last block has threads beyond the array, and without the check they write out of bounds.

Launch it:

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;             // ceiling division
vector_add<<<blocks, threads>>>(d_a, d_b, d_out, n);
```

## Reductions: the pattern worth knowing

A sum over an array cannot be done independently per thread. The standard approach is a
tree reduction in shared memory:

```cpp
__global__ void reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread loads two elements and adds them -- halves the tree depth.
    float v = (i < n) ? input[i] : 0.0f;
    if (i + blockDim.x < n) v += input[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();

    // Tree reduction. Stride halves each round.
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // The last 32 elements are within one warp: use shuffles, no barrier needed.
    if (tid < 32) {
        float x = sdata[tid] + sdata[tid + 32];
        for (int offset = 16; offset > 0; offset >>= 1) {
            x += __shfl_down_sync(0xffffffff, x, offset);
        }
        if (tid == 0) output[blockIdx.x] = x;
    }
}
```

Three things here are the actual content of the lesson.

::: key
**`__syncthreads()` is a barrier for the whole block.** Every thread must reach it. Calling
it inside a divergent branch where some threads do not participate **deadlocks**. This is why
the loop condition `tid < s` wraps only the addition, not the barrier.

**Warp shuffles avoid barriers.** Threads within a warp execute in lockstep, so once the
reduction is down to 32 elements you can exchange values directly with `__shfl_down_sync`
and skip synchronisation entirely. This is worth a meaningful fraction of the kernel's time.

**Loading two elements per thread** halves the tree depth for free. Small structural choices
like this dominate reduction performance.
:::

## A fused kernel that earns its place

The case where hand-writing wins: a chain of memory-bound operations that no library fuses.

```cpp
// GELU applied in place, fused with a bias add. Two reads and one write
// instead of four reads and two writes.
__global__ void fused_bias_gelu(float* x, const float* bias, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = x[i] + bias[i % d];
    // tanh approximation of GELU: cheaper than the exact erf form.
    const float c = 0.7978845608f;               // sqrt(2/pi)
    float inner = c * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(inner));
}
```

By lesson 7.02's model, both operations are memory-bound, so halving the traffic roughly
halves the time.

## Calling it from PyTorch

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>

__global__ void square_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * in[i];
}

torch::Tensor square(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    square_kernel<<<(n + threads - 1) / threads, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
'''

module = load_inline(
    name='square_ext',
    cpp_sources='torch::Tensor square(torch::Tensor x);',
    cuda_sources=cuda_source,
    functions=['square'],
    verbose=False,
)

x = torch.randn(1_000_000, device='cuda')
torch.allclose(module.square(x), x * x)      # True
```

The `TORCH_CHECK` lines are load-bearing. A kernel handed a non-contiguous tensor reads
whatever is at those addresses and produces silently wrong results — the worst possible
failure mode, since nothing errors.

## Errors are asynchronous

::: warning
Kernel launches are asynchronous, so an error surfaces at the *next* synchronisation point —
often in an unrelated line, dozens of operations later. The stack trace points at innocent
code.

For debugging, force synchronous reporting:

```bash
CUDA_LAUNCH_BLOCKING=1 python train.py
```

and use `compute-sanitizer` (the successor to `cuda-memcheck`) to catch out-of-bounds
accesses and race conditions, which otherwise corrupt memory silently.

In production code, check after launches:

```cpp
cudaError_t err = cudaGetLastError();
TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
```
:::

## When to write one

| Situation | Write a kernel? |
|---|---|
| A library kernel exists | **No** — cuBLAS and cuDNN are years of tuning |
| `torch.compile` fuses it adequately | **No** — check first (lesson 2.14) |
| Chain of memory-bound ops, unfused | Maybe — try Triton first (lesson 7.04) |
| Novel algorithm, unusual access pattern | Yes |
| Squeezing the last 10% of a known-critical kernel | Yes, with `ncu` |

::: key
**Try Triton before CUDA.** Lesson 7.04 covers it: Python syntax, automatic handling of
coalescing and shared memory, and typically 80–95% of hand-written CUDA performance at a
fraction of the effort.

Hand-written CUDA is for the cases where Triton's abstractions get in the way — unusual
synchronisation, warp-level primitives, or the last few percent on a kernel that runs
billions of times.
:::

## Profiling a kernel

```bash
ncu --set full -o profile python script.py
```

The metrics that matter: achieved occupancy, memory throughput as a fraction of peak, tensor
core utilisation, and warp stall reasons. The stall breakdown tells you what to fix — "long
scoreboard" means waiting on memory, "barrier" means excessive `__syncthreads()`.

::: exercise
Your custom kernel is slower than the equivalent PyTorch operation. Give four causes and how
to check each.
:::

::: solution
**1. PyTorch is calling a tuned library.** For anything matmul-shaped, `torch.matmul`
dispatches to cuBLAS, which has hand-tuned kernels per architecture and shape. You will not
beat it. *Check:* profile with `nsys` and look at the kernel name PyTorch actually launched —
if it starts with `ampere_` or `sm90_`, it is a cuBLAS kernel.

**2. Uncoalesced memory access.** If consecutive threads read non-consecutive addresses, you
pay up to 32 transactions instead of 1 (lesson 7.01). *Check:* `ncu` reports
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` per request — a value well above 4 sectors
per request means poor coalescing. *Fix:* make thread $i$ read element $i$, restructuring the
loop if needed.

**3. Low occupancy from register or shared-memory pressure.** *Check:* `ncu` reports achieved
occupancy directly. *Fix:* compile with `-maxrregcount` to cap registers per thread, reduce
shared memory per block, or adjust the block size. Remember lesson 7.01's caveat — low
occupancy is only a problem if the kernel is latency-bound, which the stall reasons will tell
you.

**4. The kernel is memory-bound and you optimised arithmetic.** *Check:* compute $I$ (lesson
7.02) and compare achieved GB/s against peak. If you are at 90% of bandwidth, the kernel is
already optimal and no further work helps — the only remaining move is to move less data,
which means fusing more operations in or using a smaller dtype.

**Two more worth ruling out:**

**Launch overhead.** A kernel launch costs ~5 μs. If your kernel runs in 2 μs, you are
measuring the launch. Batch the work, or use CUDA graphs.

**You are measuring the compile.** `load_inline` compiles on first call, which takes seconds.
Warm up before timing (lesson 2.13).

**Order of checks:** compute $I$ first — it tells you which half of the optimisation space to
look in, and it takes one minute. Then profile with `ncu` and read the stall reasons.
:::

## What to carry forward

- Every thread computes its own index and must guard against running past the array.
- `__syncthreads()` must be reached by every thread in the block, or it deadlocks.
- Warp shuffles reduce the last 32 elements without barriers.
- Validate contiguity and device in the wrapper — a wrong-layout kernel fails silently.
- Errors are asynchronous; use `CUDA_LAUNCH_BLOCKING=1` and `compute-sanitizer`.
