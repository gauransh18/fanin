---
summary: The dynamic graph PyTorch builds during the forward pass, what backward does to it, and the four things that break it.
prereqs: [chain-rule-backprop, tensors-dtypes-devices]
---

Autograd is not magic and it is not a compiler. It records a graph while your forward
code runs, then walks that graph backwards applying the rules from lesson 1.08. Knowing
exactly what gets recorded explains every autograd error you will meet.

## What gets recorded

Every tensor carries three autograd fields:

```python
import torch

x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()

x.requires_grad, y.requires_grad     # (True, True) -- propagates forward
x.grad_fn                            # None: x is a leaf, created by the user
y.grad_fn                            # <MulBackward0>
z.grad_fn                            # <SumBackward0>
y.grad_fn.next_functions             # ((<AccumulateGrad>, 0),) -- points at x
```

- **`requires_grad`** — whether gradients flow through this tensor. It propagates
  forward: any operation with a `requires_grad=True` input produces a `requires_grad=True`
  output.
- **`grad_fn`** — the node that knows how to compute this tensor's backward. `None` for
  leaves.
- **`grad`** — where accumulated gradients land, populated only for leaf tensors.

The graph is built **during the forward pass** and is specific to the shapes and control
flow of that one run. A Python `if` that takes a different branch next time produces a
different graph, which is what "define-by-run" means.

## What backward does

```python
import torch

x = torch.randn(3, requires_grad=True)
z = (x ** 2).sum()
z.backward()
x.grad                    # tensor 2*x
```

`backward()` walks the graph in reverse topological order, calling each node's rule and
accumulating into leaves. Three details matter.

**It accumulates, it does not assign.** `x.grad += new` every time. This is deliberate —
it is what makes gradient accumulation over microbatches and weight tying work (lesson
1.08) — and it is why you must call `optimizer.zero_grad()` each step.

**It frees the graph.** Intermediate buffers are released as they are consumed, so a
second `backward()` on the same graph raises an error. Pass `retain_graph=True` only when
you genuinely need two backward passes through one forward, and know you are paying for
the memory.

**It requires a scalar** — or an explicit seed vector. The asymmetry from lesson 1.08 in
API form:

```python
import torch

x = torch.randn(3, requires_grad=True)
y = x * 2                                  # a vector

y.backward()                               # RuntimeError: grad can be implicitly
                                           # created only for scalar outputs
y.backward(torch.ones_like(y))             # fine: supply the seed yourself
```

## The four things that break the graph

::: key
**1. `.detach()`** returns a tensor sharing storage with `requires_grad=False`. Gradients
stop at the boundary. This is how you freeze a teacher model, stop a target network from
being trained (lesson 6.06), or log a loss value.

**2. `torch.no_grad()`** disables recording entirely inside the block. No graph is built,
so no activation memory is retained. Use it for inference and for the optimizer's
parameter update.

**3. In-place operations on tensors needed for backward.** PyTorch version-checks its
saved tensors and raises if one was mutated. The error names the operation, and the fix
is almost always to stop being clever about memory.

**4. Converting through NumPy or Python floats.** `.numpy()`, `.item()`, `float(t)` all
leave the graph. Any arithmetic done outside PyTorch is invisible to autograd.
:::

```python
import torch

x = torch.randn(3, requires_grad=True)

with torch.no_grad():
    y = x * 2
y.requires_grad            # False -- nothing was recorded

z = (x * 2).detach()
z.requires_grad            # False -- recorded, then cut

# The in-place trap.
a = torch.randn(3, requires_grad=True)
b = a.exp()                # backward for exp needs b itself
b.mul_(2)                  # mutating it invalidates the saved value
try:
    b.sum().backward()
except RuntimeError as e:
    print(str(e)[:80])     # "one of the variables needed for gradient
                           #  computation has been modified..."
```

::: check
Why must you call `optimizer.zero_grad()` every step?

- [x] `backward()` accumulates into `.grad` rather than assigning, so last step's gradient would be added to this one's
  > Accumulation is deliberate: it is what makes gradient accumulation over microbatches and weight tying work at all. Clearing is the caller's job precisely because the framework cannot know which you meant.
- [ ] Because `.grad` holds stale memory that must be freed each step
  > The buffer is reused, not leaked. Zeroing writes over it rather than releasing it.
- [ ] Because the graph is freed after backward and `.grad` becomes invalid
  > The graph being freed is a separate fact, and it does not invalidate `.grad` — leaf gradients survive precisely so the optimizer can read them.
- [ ] You do not have to; modern optimizers zero automatically
  > `set_to_none=True` changes *how* they are cleared, but somebody still has to call it.
:::

## no_grad versus detach versus inference_mode

| Tool | Graph built? | Memory saved? | Typical use |
|---|---|---|---|
| `.detach()` | Yes, up to the cut | No | Stop gradients at one point |
| `torch.no_grad()` | No | Yes | Evaluation, optimizer step |
| `torch.inference_mode()` | No | Yes, plus more | Pure inference, no training after |

`inference_mode` is stricter than `no_grad`: it also skips version counting, so tensors
created inside it can never be used in autograd later. It is faster, and it will raise if
you try to reuse its outputs in a training graph.

::: check
`y = x * 2` is a vector and `y.backward()` raises. Why, and what fixes it?

- [x] Backward needs a scalar or an explicit seed vector; `y.backward(torch.ones_like(y))` supplies one
  > This is lesson 1.08's asymmetry in API form. Reverse mode gets all gradients in one pass *because* it starts from a single number; with a vector output you have to say which combination you meant.
- [ ] `x` is not a leaf tensor, so there is nowhere to accumulate
  > `x` was created by the user with `requires_grad=True`, which makes it a leaf. That part is fine.
- [ ] The graph was already freed by a previous backward
  > That raises a different error, and only after a backward has actually run.
- [ ] Multiplication by a constant is not recorded
  > It is recorded — `y.grad_fn` is a `MulBackward0`.
:::

## Reading the graph

```python
import torch

def walk(fn, depth=0):
    if fn is None or depth > 4:
        return
    print('  ' * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1)

x = torch.randn(3, requires_grad=True)
loss = (x * 2 + 1).pow(2).sum()
walk(loss.grad_fn)
# SumBackward0
#   PowBackward0
#     AddBackward0
#       MulBackward0
#         AccumulateGrad
```

That printout is the chain rule from lesson 1.08 made visible, read from the loss
backwards to the parameter.

::: warning
**The most expensive autograd bug is accidentally keeping graphs alive.** Writing
`total_loss += loss` instead of `total_loss += loss.detach()` keeps every step's graph in
memory, because `total_loss` still references them. Memory grows linearly with steps and
the run dies with OOM at an unpredictable point. If your memory use climbs across
iterations rather than within one, this is almost certainly the cause.
:::

::: exercise
Why does `optimizer.step()` not need to be wrapped in `torch.no_grad()`, even though it
performs arithmetic on tensors that have `requires_grad=True`?
:::

::: solution
It effectively is — just not by you. Optimizer implementations wrap their update in
`@torch.no_grad()` internally, and they operate on `param.data` or use in-place ops on
tensors explicitly excluded from tracking.

The reason it matters: if the update *were* recorded, each step would extend the graph
with the previous step's update, chaining every iteration together. Memory would grow
without bound and the gradients computed on the next backward would be wrong — they would
include derivatives of the optimizer's own arithmetic.

You can see the requirement directly by writing SGD by hand:

```python
# Wrong: records the update into the graph.
for p in model.parameters():
    p -= lr * p.grad                      # RuntimeError: a leaf Variable that
                                          # requires grad is being used in an
                                          # in-place operation

# Right:
with torch.no_grad():
    for p in model.parameters():
        p -= lr * p.grad
```

The error in the first version is PyTorch specifically protecting you from this, since
mutating a leaf in-place would invalidate the graph that produced its gradient.
:::

## What to carry forward

- The graph is built during forward, per-run, and freed by backward.
- Gradients accumulate into leaves — hence `zero_grad()`.
- `backward()` needs a scalar or an explicit seed vector.
- `detach`, `no_grad`, in-place mutation and NumPy round-trips each break the graph.
- Accumulating undetached losses is the usual cause of memory growing across steps.
