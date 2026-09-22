---
summary: The input pipeline is the most common reason a GPU sits idle. What the knobs do and how to tell whether you are data-bound.
prereqs: [tensors-dtypes-devices]
---

A training step is only as fast as the batch it is waiting for. Data loading is where
most first-time training runs lose half their throughput, and the diagnosis is simple
once you know what to measure.

## The two dataset styles

**Map-style** implements `__len__` and `__getitem__`. The sampler decides the order, so
shuffling and distributed splitting come for free.

```python
import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, tokens, block):
        self.tokens, self.block = tokens, block

    def __len__(self):
        return len(self.tokens) - self.block

    def __getitem__(self, i):
        chunk = self.tokens[i : i + self.block + 1]
        return chunk[:-1], chunk[1:]      # inputs, next-token targets
```

**Iterable-style** implements `__iter__` and yields samples. Use it when the data does
not fit in memory or has no meaningful length — a stream of shards, a remote object
store. You then own sharding and shuffling yourself, and getting it wrong means workers
duplicate each other's data.

::: warning
With an `IterableDataset` and `num_workers > 1`, every worker runs the same `__iter__`
unless you split the work explicitly. Read `torch.utils.data.get_worker_info()` and
partition by `worker_id`, or each epoch will contain $N$ copies of everything and your
effective dataset will be $N$ times smaller than you think.
:::

## The knobs that matter

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,            # subprocesses fetching in parallel
    pin_memory=True,          # page-locked staging buffer for fast H2D copies
    prefetch_factor=2,        # batches queued per worker
    persistent_workers=True,  # do not tear workers down between epochs
    drop_last=True,           # keep every batch the same shape
)
```

- **`num_workers`** is the single most important setting. Zero means loading happens on
  the training process's main thread, so the GPU idles during every `__getitem__`. Start
  at 4–8 and measure.
- **`pin_memory=True`** with `.to(device, non_blocking=True)` lets the host-to-device
  copy overlap with compute. Nearly free throughput, as lesson 2.01 noted.
- **`persistent_workers=True`** avoids re-forking every epoch, which matters when workers
  have expensive setup (opening shard files, building an index).
- **`drop_last=True`** discards the final partial batch. With `torch.compile` this is
  important: a differently sized final batch triggers a recompilation every epoch
  (lesson 2.14).

## Collation

`collate_fn` turns a list of samples into a batch. The default stacks tensors, which
requires identical shapes. Variable-length sequences need a custom one:

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = pad_sequence(ys, batch_first=True, padding_value=-100)   # ignored by CE
    return xs, ys, lengths
```

The `-100` is not arbitrary: it is `F.cross_entropy`'s default `ignore_index`, so padded
target positions contribute no loss and no gradient. Padding inputs with token 0 while
forgetting to mask the targets is a common and quiet bug — the model learns to predict
padding.

::: key
**Sort by length before batching.** Padding a batch to its longest member wastes compute
on every shorter sequence. Bucketing similar lengths together typically cuts padding
waste from 40% to under 5% on natural text, which is a 1.5× speedup for a sampler
change. For language-model pretraining, the better answer is to concatenate documents
and chop into fixed blocks, eliminating padding entirely.
:::

::: check
You use an `IterableDataset` with `num_workers=8` and do not read `get_worker_info()`. What happens?

- [x] Every worker runs the same `__iter__`, so each epoch contains eight copies of everything
  > Your effective dataset is eight times smaller than you think, and the model sees each example eight times per "epoch". Map-style datasets avoid this because the sampler owns the ordering; with iterable style, sharding is yours to do.
- [ ] Workers deadlock waiting for a shared iterator
  > Each worker gets its own copy of the dataset object, so there is no contention — which is exactly why the duplication is silent.
- [ ] PyTorch raises an error requiring explicit sharding
  > It does not. The run proceeds and looks normal.
- [ ] Only worker 0 produces data and the others idle
  > All eight produce data. That is the problem.
:::

## Diagnosing a data-bound loop

```python
import time
import torch

def profile_loader(loader, model, opt, steps=50):
    data_time = step_time = 0.0
    it = iter(loader)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        batch = next(it)
        torch.cuda.synchronize()
        t1 = time.perf_counter(); data_time += t1 - t

        loss = model(batch).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t = time.perf_counter(); step_time += t - t1

    total = data_time + step_time
    print(f'data {data_time/total:6.1%}   compute {step_time/total:6.1%}')
```

If data time exceeds about 10%, you are leaving throughput on the table. The fixes, in
the order worth trying:

1. Raise `num_workers` until the fraction stops falling.
2. Enable `pin_memory` and `non_blocking=True`.
3. Move augmentation and tokenization offline — do it once, not every epoch.
4. Switch to a pre-tokenized binary format (memory-mapped `.bin`, WebDataset, Arrow)
   so `__getitem__` is a slice rather than a parse.

::: warning
Each worker is a **separate process** with its own copy of the dataset object. A dataset
holding a 20 GB in-memory array will be copied 8 times with `num_workers=8`, and the job
dies on OOM. Use `np.memmap` or an on-disk format so workers share page cache rather than
duplicating heap. This is also why datasets must be picklable — an open file handle or a
database connection as an attribute will fail at fork time.
:::

## Reproducible shuffling

```python
import torch
from torch.utils.data import DataLoader

g = torch.Generator()
g.manual_seed(1234)

def seed_worker(worker_id):
    # Each worker needs its own deterministic seed for numpy/random too.
    import numpy as np, random
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed); random.seed(seed)

loader = DataLoader(ds, batch_size=32, shuffle=True,
                    num_workers=4, generator=g, worker_init_fn=seed_worker)
```

Without `worker_init_fn`, workers inherit the parent's NumPy seed at fork time and
produce **identical** random augmentations — a famously subtle bug that silently reduces
your effective augmentation diversity by a factor of `num_workers`. Lesson 2.15 covers
the rest of reproducibility.

::: exercise
Your GPU utilisation hovers around 40%. `num_workers=8`, `pin_memory=True`. What do you
check next, and in what order?
:::

::: solution
**First, confirm it really is the data.** Run the profile above. If compute is already
above 90% of wall time, the problem is not loading — it is that your model is too small
to saturate the device, and the answer is a larger batch or a fused kernel, not more
workers.

**If data time is high, find which part.** Time `__getitem__` for a single sample. If it
is milliseconds, the work inside is too heavy — image decoding and tokenization belong in
an offline preprocessing pass, not in the loop.

**Check for main-process synchronisation.** `.item()` on the loss, `print(tensor)`, or a
metric computed on CPU each step stalls the pipeline regardless of worker count (lesson
2.01). This presents exactly as low utilisation with a healthy loader.

**Check worker starvation.** If workers are blocked on disk, more workers will not help;
`iostat` showing saturated disk means the fix is a faster format or a local cache, not
more processes.

**Check the batch size.** At 40% utilisation with a small model, the kernels may simply
be too small to fill the GPU — the memory-bound regime of lesson 1.03. Doubling the batch
often doubles throughput at no cost per sample.
:::

::: check
`drop_last=True` discards the final partial batch. Beyond keeping shapes uniform, why does it matter under `torch.compile`?

- [x] A differently sized final batch triggers a recompilation, once per epoch
  > Compiled graphs are specialised on shape. One odd batch per epoch is enough to pay the compile cost repeatedly for no benefit — lesson 2.14 has the details.
- [ ] `torch.compile` cannot handle variable batch sizes at all
  > It handles them by recompiling, or by marking a dimension dynamic. The cost, not the capability, is the issue.
- [ ] The partial batch produces incorrect gradients
  > Its gradients are perfectly correct; they just represent fewer examples.
- [ ] Dropping it improves convergence by keeping the batch statistics stable
  > Batch statistics do wobble on a small final batch, but that is a minor effect next to the recompilation cost.
:::

## What to carry forward

- `num_workers=0` means the GPU waits for Python; start at 4–8 and measure.
- `pin_memory` plus `non_blocking=True` overlaps transfer with compute.
- Pad targets with `-100` so padded positions contribute no loss.
- Bucket by length, or concatenate and chunk, to stop paying for padding.
- Seed workers explicitly or they will produce identical randomness.
