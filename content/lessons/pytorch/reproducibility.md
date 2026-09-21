---
summary: What seeding does and does not guarantee, the sources of nondeterminism you cannot seed away, and how to report results honestly.
prereqs: [datasets-dataloaders, training-loop]
---

"Set the seed" is necessary and nowhere near sufficient. Understanding which sources of
variation remain is the difference between a result someone else can reproduce and a
number you cannot explain a month later.

## Seeding everything

```python
import os, random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)              # covers CUDA too
    os.environ['PYTHONHASHSEED'] = str(seed)
```

There are four independent generators, and library code uses all of them. Missing one
means an augmentation or a sampling call quietly varies between runs.

DataLoader workers need separate handling — see lesson 2.10's `worker_init_fn`, without
which every worker produces identical randomness.

## What seeding still does not fix

::: key
**Floating-point addition is not associative.** $(a + b) + c \neq a + (b + c)$ in finite
precision. GPU reductions sum in whatever order thread blocks happen to complete, which
varies run to run. Two identically seeded runs can differ in the last bits of every
reduction — and in a long training run, those differences compound into visibly different
losses.
:::

The other unseedable sources:

- **Atomics.** `scatter_add_`, `index_add_`, and the backward of embeddings and
  `grid_sample` use atomic adds whose order is nondeterministic.
- **cuDNN algorithm selection.** Benchmark mode picks the fastest convolution algorithm
  for your shapes, and the choice can differ between runs.
- **Multi-GPU reduction order.** NCCL all-reduce is not bitwise deterministic across
  different topologies or rank counts.
- **Library and driver versions.** A cuBLAS update changes kernel selection and therefore
  summation order.

## Forcing determinism, and its cost

```python
import torch

torch.use_deterministic_algorithms(True)     # errors on any nondeterministic op
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False       # fixed algorithm, not the fastest
# Required by cuBLAS for deterministic reductions:
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`use_deterministic_algorithms(True)` raises rather than silently proceeding when an
operation has no deterministic implementation. That is the point — it tells you exactly
which line is the problem.

Expect to pay 10–30% throughput for this, sometimes more for convolutional models. Use it
for debugging and for regression tests, not for production training.

::: warning
Determinism is only guaranteed for the **same hardware, same library versions, same
number of devices**. A run that is bit-identical on 8×A100 will not be on 8×H100, or on
4×A100. If you need cross-machine reproducibility, what you actually need is a checkpoint
and an archived environment, not a seed.
:::

## Reporting results honestly

A single seed is an anecdote. The variance between seeds is often larger than the
difference between the methods being compared — which is how a great deal of published
improvement turns out to be noise.

```python
import statistics

results = [run(seed=s) for s in range(5)]
mean = statistics.mean(results)
sd   = statistics.stdev(results)
sem  = sd / len(results) ** 0.5
print(f'{mean:.3f} ± {sd:.3f} (sd, n={len(results)})   sem {sem:.3f}')
```

Report the **standard deviation** across seeds and say how many you ran. Reporting a
standard error without the count hides how thin the evidence is, and reporting neither
hides everything.

Combine this with lesson 1.11's Hoeffding bound: you have two sources of uncertainty —
seed variance in training, and sampling variance in evaluation — and both need to be
smaller than the effect you are claiming.

## What to record for a reproducible run

```python
import json, subprocess, sys
import torch

def environment():
    return {
        'python': sys.version,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'devices': [torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())],
        'world_size': int(os.environ.get('WORLD_SIZE', 1)),
        'git': subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True).stdout.strip(),
        'dirty': bool(subprocess.run(['git', 'status', '--porcelain'],
                                     capture_output=True, text=True).stdout),
    }

json.dump({'env': environment(), 'config': config, 'seed': seed},
          open('run.json', 'w'), indent=2)
```

The `dirty` flag is the one people skip and later regret: a commit hash is meaningless if
the working tree had uncommitted changes. Lesson 7.11 turns this into a workflow.

## Resuming exactly

A resumed run continues identically only if you restore the RNG state too:

```python
import torch

state = {
    'model': model.state_dict(),
    'opt': opt.state_dict(),
    'sched': sched.state_dict(),
    'step': step,
    'rng': torch.get_rng_state(),
    'cuda_rng': torch.cuda.get_rng_state_all(),
    'py_rng': random.getstate(),
    'np_rng': np.random.get_state(),
}
```

Without the RNG state, a resumed run sees a different data order and different dropout
masks from that point on. The loss curve will still look fine, which is why this is
usually discovered only when someone tries to reproduce a specific number.

::: exercise
Your colleague cannot reproduce your reported accuracy of 84.2%, getting 83.1% with your
exact code and config. Walk through how you would diagnose it.
:::

::: solution
**1. Establish the seed variance first.** Run your own code across 5 seeds. If your
results span 83.0–84.5%, there is nothing to explain — 84.2% was the top of your range
and 83.1% is inside it. This is the most common answer, and the honest conclusion is that
the original number should have been reported as a mean with a spread.

**2. Compare environments.** Diff the `environment()` dict from both runs. Different
PyTorch or CUDA versions change kernel selection and therefore reduction order; different
GPU counts change the effective batch size unless the config scales it.

**3. Check the `dirty` flag.** If your recorded run had uncommitted changes, your
colleague is running different code. This is frequent and embarrassing.

**4. Check data.** Is the dataset pinned by hash? A dataset that was re-downloaded,
re-shuffled, or had a filtering step updated is a different experiment. Record a
checksum of the processed data, not just the download URL.

**5. Check evaluation.** Are you both evaluating the same checkpoint selection rule —
last step, or best validation? "Best validation over 50 evals" is itself a maximisation
over noise and biases the number upward by roughly one standard deviation. If you
reported best-of and they reported last, that alone can explain a point.

**The resolution in most real cases** is step 1 plus step 5: the difference is within
seed noise, amplified by best-checkpoint selection. Report mean ± sd over seeds and the
discrepancy usually disappears.
:::

## What to carry forward

- Seed all four generators, and seed DataLoader workers explicitly.
- Floating-point non-associativity and atomics are not seedable.
- `use_deterministic_algorithms(True)` costs throughput and tells you exactly what is nondeterministic.
- Determinism holds only for identical hardware, versions and device counts.
- Report mean ± sd across seeds, and record the git hash *and* whether the tree was dirty.
