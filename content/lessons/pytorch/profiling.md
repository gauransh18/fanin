---
summary: Measuring where time and memory actually go, rather than guessing. Asynchrony makes naive timing wrong by default.
prereqs: [training-loop, tensors-dtypes-devices]
---

Every optimisation you make without measuring first is a guess. GPU asynchrony makes
naive timing actively misleading, so the first thing to learn is how to time anything
at all.

## Timing, correctly

```python
import time
import torch

# WRONG: the kernel has only been queued, not executed.
t0 = time.perf_counter()
y = model(x)
print(time.perf_counter() - t0)       # measures the launch, ~50us

# RIGHT: wait for the queue to drain.
torch.cuda.synchronize()
t0 = time.perf_counter()
y = model(x)
torch.cuda.synchronize()
print(time.perf_counter() - t0)
```

Better still, use CUDA events — they are timestamped on-device and do not stall the host:

```python
import torch

start, end = torch.cuda.Event(True), torch.cuda.Event(True)

for _ in range(10):        # warm up: first calls include kernel autotuning
    model(x)

start.record()
for _ in range(50):
    model(x)
end.record()
torch.cuda.synchronize()
print(f'{start.elapsed_time(end) / 50:.3f} ms per call')
```

::: warning
Always warm up. The first invocation of a kernel includes cuDNN autotuning, memory pool
growth, and — under `torch.compile` — the entire compilation. Benchmarks that skip warmup
routinely report numbers 10–100× too slow and send people optimising the wrong thing.
:::

## The PyTorch profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./trace'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        train_step()
        prof.step()

print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=15))
```

The `schedule` matters: profiling every step produces gigabytes of trace and distorts the
timing it is trying to measure. Skip one, warm one, record three.

Read the table for three things:

- **`self_cuda_time_total`** — where GPU time actually goes. If `aten::copy_` or
  `aten::contiguous` is near the top, you have a layout problem (lesson 2.02).
- **The CPU/CUDA time ratio** — if CPU time exceeds CUDA time, you are launch-bound: the
  Python side cannot queue work fast enough. Fuse operations or use CUDA graphs.
- **Call counts** — thousands of tiny kernels is the signature of an unfused elementwise
  chain, which lesson 2.14's `torch.compile` fixes directly.

The Chrome trace (`chrome://tracing` or Perfetto) is worth more than the table. Gaps in
the CUDA row are the GPU idling, and their position tells you why: gaps at the start of
each step mean data loading; gaps mid-step mean synchronisation.

## Memory

```python
import torch

torch.cuda.reset_peak_memory_stats()
train_step()
print(f'peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
print(f'reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} GB')
```

**Allocated** is what your tensors use. **Reserved** is what the caching allocator holds
from the driver. A large gap means fragmentation — many differently sized allocations
have left unusable holes.

For a visual breakdown:

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100_000)
train_step()
torch.cuda.memory._dump_snapshot('snap.pickle')
torch.cuda.memory._record_memory_history(enabled=None)
# Open snap.pickle at https://pytorch.org/memory_viz
```

This shows every allocation's size, lifetime and the stack that created it — which is how
you find the tensor that is alive for the whole step when it only needs to exist for one
line.

::: key
Training memory is roughly:

$$
\text{params} + \text{grads} + \text{optimizer state} + \text{activations} + \text{fragmentation}
$$

The first three are fixed and computable (lesson 2.01). **Activations** are the only term
that scales with batch size and sequence length, so if your memory grows when you raise
the batch, activations are the target and gradient checkpointing (lesson 7.08) is the
lever.
:::

## Is it compute-bound or memory-bound?

Apply lesson 1.03's arithmetic intensity. Compute achieved FLOP/s and achieved bandwidth,
then compare each against the hardware's peak:

```python
import torch, time

def roofline(fn, flops, bytes_moved, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    s = (time.perf_counter() - t0) / iters
    print(f'{flops/s/1e12:8.2f} TFLOP/s   {bytes_moved/s/1e9:8.1f} GB/s')

n = 8192
a = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
b = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)

roofline(lambda: a @ b, 2 * n ** 3, 3 * n * n * 2)    # compute-bound
roofline(lambda: a + b, n * n,      3 * n * n * 2)    # memory-bound
```

If you are near peak bandwidth and far from peak FLOP/s, more arithmetic is free — fuse
more work into the kernel. If you are near peak FLOP/s, only a better algorithm helps.

## The usual culprits

| Symptom | Likely cause | Fix |
|---|---|---|
| Low GPU utilisation, gaps at step start | Data loading | Lesson 2.10 |
| CPU time > CUDA time | Launch-bound | `torch.compile`, CUDA graphs |
| `aten::copy_` high in the profile | Non-contiguous layouts | Fix the strides |
| Many tiny kernels | Unfused elementwise chain | `torch.compile` |
| Memory grows across steps | Retained graph | `detach()` your accumulators |
| Allocated ≪ reserved | Fragmentation | `expandable_segments:True` |

::: exercise
Your profile shows `aten::to` consuming 30% of step time. What is happening, and how
would you find the cause?
:::

::: solution
`aten::to` is a dtype or device conversion, and 30% means it is being called constantly
on large tensors. Three common causes:

**1. Per-step host-to-device transfer of something that should live on the GPU.** A mask,
a position-index tensor, or a constant rebuilt each step. Check with `record_shapes=True`
and look at the sizes — a mask of shape $(T, T)$ moved every step is the classic. Fix by
registering it as a buffer (lesson 2.08) so it moves once.

**2. An autocast boundary in a loop.** Every entry into and exit from an autocast region
casts tensors. If you are wrapping individual layers rather than the whole forward, you
pay for a conversion at every boundary. Wrap the forward once.

**3. A dtype mismatch forcing repeated promotion.** Mixing an fp32 buffer with bf16
activations makes every operation promote the bf16 side. Check with:

```python
for name, buf in model.named_buffers():
    print(name, buf.dtype, buf.device)
```

Any buffer still in fp32 while activations are bf16 will cause this.

**How to localise it:** run the profiler with `with_stack=True` and sort by
`self_cpu_time_total` filtered to `aten::to`. The stack trace names the exact line. Then
check whether the tensor being converted changes between steps — if it does not, it
should be created once and cached.
:::

## What to carry forward

- Synchronise or use CUDA events; always warm up before timing.
- Profile with a schedule, and read the trace's gaps, not just the table.
- Allocated versus reserved distinguishes real usage from fragmentation.
- Compare against roofline to know whether fusing or a new algorithm is the answer.
- `aten::copy_` and `aten::to` near the top mean a layout or dtype bug, not slow hardware.
