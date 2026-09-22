---
summary: The hardware model you need to reason about performance — SMs, warps, tensor cores, and the occupancy question that decides everything else.
prereqs: [matrix-multiplication-cost, profiling]
---

You do not need to write CUDA to benefit from knowing how a GPU executes work. You do need
it to understand why some operations run at 5% of peak and others at 70%.

## The execution hierarchy

| Level | Count (H100) | What it is |
|---|---|---|
| Thread | — | One execution context |
| **Warp** | 32 threads | The real unit of scheduling |
| Block | ≤1024 threads | Shares memory, runs on one SM |
| **SM** | 132 | An independent processor |
| Grid | — | All blocks of a kernel launch |

::: key
**The warp is the fundamental unit, not the thread.** 32 threads execute the same instruction
in lockstep. If threads in a warp take different branches, the warp executes both paths with
the inactive threads masked off — **warp divergence**, and it doubles the cost of that
region.

This is why GPU code avoids data-dependent branching. Not because branches are forbidden,
but because a warp cannot take two paths at once.
:::

## The memory hierarchy

| Memory | Size (H100) | Bandwidth | Latency |
|---|---|---|---|
| Registers | 256 KB/SM | ~20 TB/s | ~1 cycle |
| Shared / L1 | 228 KB/SM | ~15 TB/s | ~30 cycles |
| L2 | 50 MB | ~7 TB/s | ~200 cycles |
| HBM3 | 80 GB | 3.35 TB/s | ~400 cycles |

Each step down is roughly 3–5× slower and much larger. The performance question for any
kernel is: **how much work can I do per byte read from HBM?**

That is arithmetic intensity, and lesson 7.02 makes it into a method. FlashAttention
(lesson 4.12) is exactly the answer "keep the tile in shared memory and never write the score
matrix to HBM".

::: check
Why does GPU code avoid data-dependent branching?

- [x] A warp of 32 threads executes one instruction in lockstep — if they take different branches, the warp runs both paths with the inactive threads masked, doubling the cost of that region
  > It is not that branches are forbidden. It is that a warp cannot take two paths at once, so divergence is paid in serialised execution.
- [ ] Branches are not supported in CUDA C
  > They are ordinary C control flow and compile fine.
- [ ] Branch prediction is absent, so every branch stalls the pipeline
  > The cost is warp divergence, not misprediction. Uniform branches across a warp are cheap.
- [ ] Branching prevents memory coalescing
  > Divergent branches can disturb access patterns as a side effect; the direct cost is executing both paths.
:::

## Coalescing

When a warp reads memory, the hardware combines the 32 requests into as few transactions as
possible. **Consecutive threads reading consecutive addresses** gives one or two
transactions. A strided or scattered access pattern gives up to 32.

```python
import torch

x = torch.randn(4096, 4096, device='cuda')

x.sum(dim=1)     # rows are contiguous: coalesced, fast
x.sum(dim=0)     # column stride is 4096 elements: much slower
```

The factor is often 2–5×, and it is the practical reason lesson 2.02's strides matter and why
`channels_last` helps convolutions.

## Tensor cores

Dedicated units that compute small matrix products — typically $16\times16\times16$ — in one
instruction. On an H100 they provide roughly $989$ TFLOP/s of bf16 against about $67$
TFLOP/s on the general-purpose units.

**A 15× gap.** Any operation that does not use tensor cores is leaving almost everything on
the table.

To use them, three conditions must hold:

- **Dtype**: fp16, bf16, fp8, or tf32. Plain fp32 does not use them.
- **Shape**: dimensions should be multiples of 8 (fp16/bf16) or 16 (fp8). A matrix with
  dimension 4095 falls back to a slower path.
- **Layout**: contiguous, in a layout the kernel expects.

```python
import torch

# This is why vocabularies are padded to a multiple of 64 (lesson 4.09).
for d in (4096, 4095):
    a = torch.randn(4096, d, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(d, 4096, device='cuda', dtype=torch.bfloat16)
    # Time both -- the 4095 case is measurably slower for a smaller problem.
```

::: check
An H100 has 3.35 TB/s of HBM bandwidth and about 15 TB/s of shared memory. What is the performance question that follows for any kernel?

- [x] How much work can I do per byte read from HBM
  > Each step down the hierarchy is roughly 3–5× slower and much larger. Keeping data on-chip and doing more with it before writing back is what every fused kernel, including FlashAttention, is about.
- [ ] How many threads can I launch
  > Occupancy matters for hiding latency and is downstream of the traffic question.
- [ ] How can I avoid using shared memory, which is scarce
  > Shared memory is what you *want* to use — it is where data goes to be reused.
- [ ] How do I keep every SM busy at all times
  > Every SM can be busy and still stalled waiting on HBM, which is exactly the memory-bound case.
:::

## Occupancy

Occupancy is the fraction of a warp scheduler's slots that are filled. It matters because
GPUs hide latency by switching warps: when one warp stalls on a memory read, another runs.
Too few resident warps and the SM idles.

Three things limit it:

- **Registers per thread.** An SM has a fixed register file; a kernel using many registers
  per thread supports fewer resident threads.
- **Shared memory per block.** Same tradeoff.
- **Block size.** Too small wastes scheduler slots; too large may not fit.

::: warning
**Higher occupancy is not automatically better.** A kernel that uses many registers to keep
data close can outperform a high-occupancy kernel that reads from memory repeatedly.
FlashAttention is deliberately low-occupancy — it uses a lot of shared memory per block so
tiles stay on chip.

Occupancy is a diagnostic, not a target. Optimise for time, and use occupancy to explain the
number.
:::

## Multi-GPU topology

Within a node, GPUs connect over **NVLink** — about 900 GB/s bidirectional on H100. Between
nodes, **InfiniBand** gives 25–50 GB/s per link.

That is a 20–40× gap, and it dictates the parallelism layout of lesson 7.06: tensor
parallelism, which communicates twice per layer, must stay inside a node; data parallelism,
which communicates once per step, can cross nodes.

```bash
nvidia-smi topo -m          # shows the interconnect between every GPU pair
```

Read this before designing a distributed run. A layout that assumes NVLink between GPUs
connected only by PCIe will be several times slower than expected.

## Reading a profile

```python
import torch

torch.cuda.reset_peak_memory_stats()
print(torch.cuda.get_device_properties(0))
# Note: multiProcessorCount, totalGlobalMem, and the clock rates.

# The number that matters most:
print(f'peak memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
```

For kernel-level detail, `ncu` (Nsight Compute) gives achieved occupancy, memory throughput
and tensor-core utilisation per kernel. `nsys` (Nsight Systems) gives the timeline view —
which is where you see the gaps that lesson 2.13 taught you to read.

## Generations

| | A100 | H100 | B200 |
|---|---|---|---|
| bf16 TFLOP/s | 312 | 989 | ~2250 |
| Memory | 40/80 GB | 80 GB | 192 GB |
| Bandwidth | 2.0 TB/s | 3.35 TB/s | ~8 TB/s |
| FLOP/byte ridge | 156 | 295 | ~280 |

The ridge point — peak FLOP/s divided by peak bandwidth — has **risen** across generations.
Compute has grown faster than bandwidth, so more operations fall on the memory-bound side of
the roofline than used to. That trend is why fusion, FlashAttention and quantization have
become more valuable over time rather than less.

::: exercise
Your matrix multiply of $(8192, 8192) \times (8192, 8192)$ in fp32 achieves 15 TFLOP/s on an
H100 rated at 989 TFLOP/s bf16 and 67 TFLOP/s fp32. Is this good?
:::

::: solution
**Compare against the right peak first.** The 989 figure is bf16 tensor cores. In fp32 you
are on the general-purpose units at 67 TFLOP/s, so 15 TFLOP/s is **22% of the achievable
peak** — poor, but not the 1.5% that comparing against 989 would suggest.

**Why fp32 is slow here.** Plain fp32 does not use tensor cores at all. The single largest
improvement available is to stop using it.

**In order of impact:**

1. **Enable TF32.** One line, no code change, and it routes fp32 matmuls onto tensor cores
   with 10-bit mantissas:

   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```
   Expect 5–8× on this operation. Precision is adequate for essentially all deep learning;
   it is the default in recent PyTorch for good reason.

2. **Use bf16.** Full tensor-core rate. For an $8192^3$ matmul, which is deeply compute-bound
   (arithmetic intensity ≈ 1365 FLOP/byte, far past the ridge of 295), you should reach
   600–800 TFLOP/s — a 40–50× improvement over the fp32 number.

3. **Check the shapes.** 8192 is a multiple of 64, so this is fine. Had it been 8191, you
   would have fallen off the tensor-core path regardless of dtype.

**After switching to bf16, what is a good number?** 70–80% of peak on a large square matmul
is achievable with cuBLAS. Below 50% suggests thermal throttling, a non-contiguous input, or
another process sharing the GPU — check `nvidia-smi` for clocks and utilisation.

**The general lesson:** always state which peak you are comparing against. "15 TFLOP/s on a
989 TFLOP/s GPU" sounds catastrophic and "22% of fp32 peak" identifies the actual problem,
which is the dtype.
:::

## What to carry forward

- The warp of 32 threads is the scheduling unit; divergence within one costs double.
- Coalesced access is 2–5× faster; this is why strides and memory format matter.
- Tensor cores are ~15× the general units, and need the right dtype, shape and layout.
- Occupancy is a diagnostic, not a target — FlashAttention is deliberately low-occupancy.
- NVLink is 20–40× InfiniBand, which decides where each parallelism axis goes.
