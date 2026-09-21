---
summary: What torch.compile does to your Python, why graph breaks and recompiles undo it, and how to find both.
prereqs: [profiling, autograd-graph]
---

Eager PyTorch launches one kernel per operation. `torch.compile` traces your Python into
a graph, fuses what it can, and generates Triton kernels — typically 1.3–2× on
transformers, and far more on elementwise-heavy code.

```python
import torch

model = torch.compile(model)      # that is the whole API
```

The rest of this lesson is about the two things that silently give the speedup back.

## What actually happens

Three stages run under the hood:

1. **TorchDynamo** traces Python bytecode, extracting an FX graph of tensor operations.
   Anything it cannot trace becomes a **graph break** — the graph is cut, that region
   runs in eager mode, and tracing resumes afterwards.
2. **AOTAutograd** traces the backward pass too, so both directions are compiled.
3. **Inductor** lowers the graph to Triton (GPU) or C++ (CPU), fusing elementwise chains
   and picking layouts.

The fusion is where most of the win comes from. A chain like
`x.add(b).relu().mul(s)` is three kernels and three round trips to HBM in eager mode; the
compiler emits one kernel with one round trip — the arithmetic-intensity argument from
lesson 1.03, applied automatically.

## Graph breaks

Anything that needs a tensor's *value* at trace time forces a break:

```python
import torch

def bad(x):
    if x.sum() > 0:              # break: needs the value to pick a branch
        return x * 2
    return x - 1

def also_bad(x):
    print(x.mean().item())       # break: .item() leaves the graph
    return x * 2

def fine(x):
    return torch.where(x.sum() > 0, x * 2, x - 1)   # no break: stays in tensors
```

Find them:

```python
import torch

torch._logging.set_logs(graph_breaks=True)
compiled = torch.compile(model)
compiled(x)

# Or fail loudly instead of degrading quietly:
compiled = torch.compile(model, fullgraph=True)   # raises on any break
```

::: key
Develop with `fullgraph=True`. A silent graph break in the middle of your attention
implementation turns a compiled model into an eager one with extra overhead, and the only
symptom is that the speedup you expected never appeared. Making it an error is how you
find out immediately.
:::

Common breaks and their fixes:

| Break | Fix |
|---|---|
| `.item()`, `.tolist()`, `print(tensor)` | Move logging outside the compiled region |
| Data-dependent `if` | `torch.where`, or `torch.cond` |
| Unsupported library call (some NumPy, custom C) | Wrap in `torch._dynamo.disable` or register a custom op |
| Mutating a Python list of tensors | Build the list, then `torch.stack` |
| `try/except` around tensor code | Hoist it out of the traced region |

## Recompilation

Dynamo specialises on tensor **shapes**, dtypes and devices. A new combination triggers a
fresh compile, and compiles are expensive — seconds each.

```python
import torch

torch._logging.set_logs(recompiles=True)

model = torch.compile(model)
model(torch.randn(32, 128))    # compile
model(torch.randn(32, 128))    # cached, fast
model(torch.randn(32, 129))    # RECOMPILE -- different sequence length
```

The default cache limit is 8; exceeding it makes Dynamo give up and fall back to eager
permanently, with a warning that is easy to miss in a busy log.

::: warning
Variable sequence lengths are the usual cause. A dataloader yielding batches padded to
each batch's own longest member produces a new shape almost every step, so you recompile
constantly and then fall back to eager — ending up **slower** than never compiling.

Three fixes:
- `drop_last=True` and bucket lengths to a few fixed sizes (lesson 2.10).
- `torch.compile(model, dynamic=True)` to compile shape-generic kernels — slower per
  call than a specialised one, but compiled once.
- Pad to a fixed block length, which for language-model pretraining you should be doing
  anyway.
:::

## Modes

```python
import torch

torch.compile(model)                              # balanced default
torch.compile(model, mode='reduce-overhead')      # CUDA graphs: best for small batches
torch.compile(model, mode='max-autotune')         # searches kernel configs; slow to compile
```

`reduce-overhead` uses CUDA graphs to eliminate launch overhead, which is exactly the
right medicine when the profiler shows CPU time exceeding CUDA time (lesson 2.13). It
requires static shapes and stable memory addresses, so it interacts badly with anything
that reallocates between steps.

`max-autotune` benchmarks multiple kernel configurations at compile time. It can take
minutes per shape and is worth it only for a long production run.

## Compiling part of a model

You do not have to compile everything. For a model with one awkward region, compile the
rest:

```python
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        x = self.compiled_blocks(x)
        return self.awkward_custom_thing(x)

# Or mark the region to leave alone:
@torch._dynamo.disable
def awkward_custom_thing(x):
    ...
```

Compiling the transformer blocks and leaving tokenization, sampling and logging in eager
mode is a common and effective split.

## When it does not help

- **Already compute-bound on large matmuls.** If a profile shows near-peak FLOP/s
  (lesson 2.13), there is nothing to fuse — cuBLAS is already optimal.
- **Data-bound loops.** Fix lesson 2.10 first; compiling a model that spends 60% of its
  time waiting for batches gains you 40% of a speedup.
- **Very short runs.** Compilation costs seconds to minutes. A 100-step debugging run may
  never amortise it.

::: exercise
You compile your model and throughput *drops* by 20%. Give three explanations and the
check for each.
:::

::: solution
**1. You are recompiling every step.** Turn on `torch._logging.set_logs(recompiles=True)`
and count. Variable shapes from a padding dataloader are the usual cause. If you exceed
the cache limit, Dynamo falls back to eager while still paying Dynamo's tracing overhead
— strictly worse than not compiling. Fix with fixed-size buckets or `dynamic=True`.

**2. Graph breaks are fragmenting the model.** Check with `fullgraph=True`; if it raises,
you have breaks. Many small compiled regions separated by eager code can be slower than
uniform eager, because each boundary costs a guard check and prevents fusion across it.
A `print` or `.item()` inside the forward is the usual culprit.

**3. You are measuring the compile.** Compilation happens on the first call and can take
tens of seconds. If your benchmark averages over 20 steps starting from step 0, the
compile dominates. Warm up with at least 3 calls at each shape before timing — the same
rule as lesson 2.13.

**Order to check them:** recompiles first (a single log line answers it), then warmup
(re-measure with more warmup), then `fullgraph=True`. The first two account for most
reported cases.
:::

## What to carry forward

- `torch.compile` fuses elementwise chains and compiles the backward too.
- Develop with `fullgraph=True` so breaks are errors rather than silent slowdowns.
- Shape changes recompile; exceeding the cache limit falls back to eager for good.
- `reduce-overhead` is the answer to launch-bound; `max-autotune` for long runs.
- Fix data loading before compiling — it will not rescue an idle GPU.
