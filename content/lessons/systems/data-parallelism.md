---
summary: The simplest way to use many GPUs, the all-reduce that makes it work, and the overlap that decides whether it scales.
prereqs: [training-loop, gpu-architecture, expectation-variance-concentration]
---

Data parallelism replicates the model on every device, splits the batch, and averages the
gradients. It is the first axis you should use and the only one many runs need.

## The mechanism

1. Every rank holds a full copy of the parameters.
2. Each processes a different slice of the batch.
3. Gradients are **all-reduced** — summed across ranks and divided by the world size.
4. Every rank applies the same update, so replicas stay identical.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, dist.get_world_size()

rank, world = setup()
model = DDP(model.cuda(), device_ids=[rank % torch.cuda.device_count()])

sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(epochs):
    sampler.set_epoch(epoch)        # or every rank sees the same order every epoch
    for x, y in loader:
        loss = model(x.cuda(), y.cuda())
        loss.backward()             # the all-reduce happens inside here
        opt.step()
        opt.zero_grad(set_to_none=True)
```

::: warning
`sampler.set_epoch(epoch)` is required and easy to forget. Without it, the shuffle seed is
fixed, so every epoch presents examples in the same order. Training still works and converges
worse — a silent bug, exactly the class lesson 2.15 is about.
:::

## Ring all-reduce

A naive all-reduce sends everything to one rank and broadcasts back, which makes that rank a
bottleneck and costs $O(N)$ in its bandwidth.

The ring algorithm is bandwidth-optimal. Split the gradient into $N$ chunks and run two
phases around a ring:

**Reduce-scatter** — $N-1$ steps, each rank sending one chunk to its neighbour and
accumulating what it receives. Afterwards each rank holds the fully reduced value of one
chunk.

**All-gather** — $N-1$ steps circulating the reduced chunks so everyone ends with all of
them.

$$
\text{data sent per rank} = 2\,\frac{N-1}{N}\,S \;\approx\; 2S
$$

Independent of $N$. That is what makes data parallelism scale: adding ranks does not increase
per-rank communication volume.

```python
import torch.distributed as dist

# What DDP does under the hood, once per bucket:
dist.all_reduce(grad, op=dist.ReduceOp.SUM)
grad /= dist.get_world_size()
```

## Overlap is what makes it fast

::: key
Gradients become available **during** the backward pass, layer by layer, starting from the
output. DDP exploits this: it groups parameters into buckets and launches an asynchronous
all-reduce for each bucket as soon as its gradients are ready.

So communication for the last layers overlaps with computation for the earlier ones. Done
well, the all-reduce is nearly free.

Two knobs. `bucket_cap_mb` (default 25) sets the bucket size — too small means many tiny
collectives with high per-call overhead; too large means less overlap. And parameters are
bucketed in **reverse registration order**, which is why declaring layers in forward order
matters for overlap quality.
:::

## Gradient accumulation and no_sync

With accumulation (lesson 2.11), every `backward()` triggers an all-reduce — but only the
last one is needed, since intermediate gradients are incomplete:

```python
from contextlib import nullcontext

for step in range(steps):
    opt.zero_grad(set_to_none=True)
    for micro in range(accum):
        x, y = next(data_iter)
        # Suppress the all-reduce on all but the final microbatch.
        ctx = model.no_sync() if micro < accum - 1 else nullcontext()
        with ctx:
            (model(x, y) / accum).backward()
    opt.step()
```

Without `no_sync`, you communicate `accum` times more than necessary. At `accum=8` that is 8×
the network traffic for identical results — and it is one of the most common causes of poor
scaling efficiency.

## Scaling the batch and the learning rate

$N$ ranks at batch $B$ each gives an effective batch of $NB$. Gradient noise falls as
$1/\sqrt{NB}$ (lesson 1.11), so the steps are more reliable and you can take larger ones.

**Linear scaling**: multiply the learning rate by $N$, with warmup over the first few hundred
steps. It holds while gradient noise dominates and fails once you approach the curvature
limit $\eta < 2/\lambda_{\max}$ from lesson 1.09. Beyond the **critical batch size**
(lesson 5.04), larger batches stop reducing the steps needed.

**Square-root scaling**, $\eta \propto \sqrt{N}$, is the more conservative alternative and is
often better for adaptive optimizers.

## Measuring efficiency

```python
import time
import torch
import torch.distributed as dist

def scaling_efficiency(step_fn, single_gpu_throughput, warmup=10, iters=50):
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        step_fn()
    torch.cuda.synchronize(); dist.barrier()
    per_step = (time.perf_counter() - t0) / iters

    world = dist.get_world_size()
    throughput = world * batch_size / per_step
    ideal = world * single_gpu_throughput
    if dist.get_rank() == 0:
        print(f'{throughput:.0f} samples/s, {100*throughput/ideal:.1f}% of linear')
```

| Efficiency | Verdict |
|---|---|
| Above 90% | Healthy |
| 70–90% | Investigate — usually communication or stragglers |
| Below 70% | Something is wrong |

## What goes wrong

**Stragglers.** Every all-reduce is a barrier, so the slowest rank sets the pace for all.
Causes: a thermally throttled GPU, one rank reading from slower storage, or uneven batch
sizes from variable-length data. Diagnose by timing each rank's compute separately and
comparing.

**Crossing node boundaries.** NVLink is 20–40× InfiniBand (lesson 7.01). A run where data
parallelism spans nodes is fine — it communicates once per step — but check the topology
before assuming.

**Memory.** DDP replicates parameters, gradients and optimizer state on every rank. For a
7B model that is 126 GB per GPU (lesson 2.01), which does not fit. Sharding it is what
FSDP does, in lesson 7.07.

::: exercise
Scaling from 1 to 8 GPUs gives 5.2× throughput — 65% efficiency. Diagnose.
:::

::: solution
**Measure where the time goes before guessing.** Time a step with communication disabled by
running the same model on one GPU with the same per-GPU batch. The difference is
communication plus synchronisation overhead.

**Then check, in this order:**

1. **Is `no_sync` being used with gradient accumulation?** If you accumulate over 4
   microbatches without it, you all-reduce 4× more than needed. One line, and it is the most
   common cause of exactly this efficiency number. *Check:* count all-reduces per step with
   `NCCL_DEBUG=INFO`.

2. **Are there stragglers?** Time the forward and backward on each rank separately (before
   the all-reduce) and print the spread. If one rank is 30% slower, the collective waits for
   it and overall efficiency is capped at that rank's speed. *Causes:* thermal throttling
   (check `nvidia-smi -q -d PERFORMANCE`), a slower storage path for one rank, or
   variable-length batches. *Fix:* bucket by length (lesson 2.10) so per-rank work is even.

3. **Is the topology what you think?** `nvidia-smi topo -m`. If some GPU pairs are connected
   by PCIe rather than NVLink, the ring includes slow links and the whole collective runs at
   the slowest hop. *Fix:* set `NCCL_P2P_LEVEL` appropriately, or rearrange the ring.

4. **Is the bucket size wrong?** Too small gives many small collectives with per-call
   overhead dominating; too large delays the first all-reduce until late in the backward,
   losing overlap. Try `DDP(..., bucket_cap_mb=50)` and compare.

5. **Is the model too small to hide communication?** Overlap works when there is computation
   remaining to hide the transfer behind. A small model has a short backward pass and a
   gradient tensor that still takes time to reduce. *Fix:* larger per-GPU batch, which
   increases compute per step without increasing communication volume.

**The most likely answer at 65% with 8 GPUs** is (1) or (2). Both are cheap to check and
both are configuration rather than code changes.
:::

## What to carry forward

- Every rank holds a full replica; gradients are all-reduced and the update is identical.
- Ring all-reduce sends $\approx 2S$ per rank regardless of world size.
- Overlap communication with the backward pass — that is what makes it nearly free.
- Use `no_sync()` on all but the last microbatch under gradient accumulation.
- `set_epoch()` on the sampler, or every epoch has the same order.
