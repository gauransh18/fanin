---
summary: What nn.Module actually does, how parameters and buffers are discovered, and the registration bugs that silently stop training.
prereqs: [autograd-graph]
---

`nn.Module` is bookkeeping, not mathematics. It tracks which tensors are parameters,
which are buffers, and how submodules nest — so that `.parameters()`, `.to(device)`,
`.state_dict()` and `.train()` can recurse over everything at once.

## Parameters versus buffers versus plain tensors

```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()                       # must come first
        self.w = nn.Parameter(torch.randn(d, d)) # trained
        self.register_buffer('steps', torch.zeros(1))  # state, not trained
        self.scale = 1.0 / d ** 0.5              # plain Python, not tracked

    def forward(self, x):
        self.steps += 1
        return (x @ self.w) * self.scale

m = Block(4)
[n for n, _ in m.named_parameters()]   # ['w']
[n for n, _ in m.named_buffers()]      # ['steps']
'steps' in m.state_dict()              # True -- buffers are saved
```

| Kind | In `.parameters()` | In `.state_dict()` | Moved by `.to()` | Gets gradients |
|---|---|---|---|---|
| `nn.Parameter` | yes | yes | yes | yes |
| Registered buffer | no | yes | yes | no |
| Plain attribute tensor | no | no | **no** | no |

That last row is the bug. A tensor assigned as a plain attribute stays on whatever device
it was created on, so `model.to('cuda')` leaves it behind and the forward pass fails with
a device mismatch — often only under distributed training, where the model is moved after
construction.

::: warning
Buffers are for state that must travel with the model but is not learned: running
statistics in BatchNorm, a causal mask, RoPE frequency tables, a step counter. If you
find yourself writing `self.mask = torch.tril(...)`, register it. Use
`persistent=False` if you would rather recompute it than store it in every checkpoint.
:::

## Containers, and the registration trap

```python
import torch.nn as nn

class Good(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])

class Broken(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = [nn.Linear(d, d) for _ in range(n)]     # a plain list

len(list(Good(3, 8).parameters()))     # 6 -- weights and biases
len(list(Broken(3, 8).parameters()))   # 0
```

`Broken` runs. Its forward pass computes the right thing. But the optimizer receives no
parameters for those layers, so **they never update**, and `.to('cuda')` never moves
them. The loss goes down — the remaining trained layers compensate — and the model is
quietly much worse than it should be.

Use `nn.ModuleList` for lists, `nn.ModuleDict` for dicts, `nn.Sequential` when the
forward pass is a straight chain, and `nn.ParameterList` for bare parameters.

::: key
Assert your parameter count right after building the model:

```python
n = sum(p.numel() for p in model.parameters())
print(f'{n/1e6:.1f}M parameters')
```

If that number is smaller than your architecture implies, you have an unregistered
container. It is a five-second check that catches an expensive class of bug.
:::

::: check
A model stores its layers in a plain Python list instead of an `nn.ModuleList`. The forward pass is correct and the loss goes down. What is wrong?

- [x] Those layers appear in no `.parameters()` call
  > The optimizer never updates them and `.to('cuda')` never moves them. The remaining trained layers compensate, which is why the loss still falls and nothing raises. The model is quietly much worse than it should be — the worst kind of bug, because every signal says it is fine.
- [ ] The forward pass will fail once the model is moved to a GPU
  > It often does fail on device mismatch, but not always, and by then the parameters have already been silently frozen for however long you trained on CPU.
- [ ] Python lists are not picklable, so checkpointing breaks
  > Lists pickle fine. The layers are simply absent from `state_dict()`, so the checkpoint is incomplete rather than unwritable.
- [ ] Gradients accumulate without being zeroed
  > No gradients are produced for those layers at all, since the optimizer never sees them.
:::

## train() and eval() only flip a flag

`model.train()` and `model.eval()` set `self.training` recursively. They change nothing
else. Two layer types read that flag:

- **Dropout** zeroes activations in training and is an identity in eval.
- **BatchNorm** uses batch statistics in training and its running averages in eval.

Everything else ignores it. Notably, `eval()` does **not** disable gradients — that is
`torch.no_grad()`'s job, and forgetting it means your evaluation loop builds a full graph
and uses training-level memory.

```python
import torch

model.eval()
with torch.no_grad():                 # both are needed, for different reasons
    for batch in val_loader:
        ...
```

## Hooks

Hooks let you observe or modify a module's inputs, outputs and gradients without editing
it — which makes them the right tool for debugging, interpretability and activation
checkpointing.

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
captured = {}

def grab(name):
    def hook(module, inputs, output):
        captured[name] = output.detach()     # detach, or you keep the graph
    return hook

handles = [m.register_forward_hook(grab(str(i))) for i, m in enumerate(model)]
model(torch.randn(2, 8))
{k: v.shape for k, v in captured.items()}
for h in handles:
    h.remove()                                # always remove them
```

A backward hook (`register_full_backward_hook`) sees gradients, which is how you inspect
where a gradient vanishes without inserting print statements into a library.

::: warning
Hooks leak. A forward hook that stores un-detached outputs keeps the entire graph alive,
and a hook registered inside a loop accumulates. Always keep the handle and call
`.remove()`, ideally in a `try/finally` or a context manager.
:::

::: check
You need a causal mask to travel with the model, move with `.to()`, and not be trained. What do you use?

- [x] `register_buffer('mask', ...)`, with `persistent=False` if you would rather recompute it than store it in every checkpoint
  > Buffers are exactly this: state that belongs to the model but is not learned — running statistics, RoPE tables, step counters, masks.
- [ ] `nn.Parameter(..., requires_grad=False)`
  > This works for the device question but puts the mask in `.parameters()`, where optimizers and weight decay will find it. Freezing by flag is easy to undo by accident.
- [ ] A plain attribute, `self.mask = torch.tril(...)`
  > A plain attribute tensor is not moved by `.to()`, so it stays on whichever device it was built on — often surfacing only under distributed training.
- [ ] A module-level constant outside the class
  > It would never move with the model, and one shared mask across instances of different sizes is its own bug.
:::

## Initialization and parameter groups

Two patterns worth having in your fingers.

```python
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)     # recurses over every submodule
```

```python
# Weight decay on matrices, none on biases, norms or embeddings.
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim < 2 else decay).append(p)

opt = torch.optim.AdamW([
    {'params': decay,    'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=3e-4)
```

The `p.ndim < 2` test is the standard heuristic: matrices and embeddings get decay,
one-dimensional parameters (biases, LayerNorm gains) do not. Applying decay to a
LayerNorm gain pulls it toward zero, which fights the normalization it is meant to scale.

::: exercise
You load a checkpoint with `model.load_state_dict(sd)` and it reports missing and
unexpected keys. Give three likely causes and how you would confirm each.
:::

::: solution
**1. A `DistributedDataParallel` prefix.** DDP wraps the model, so every key gains a
`module.` prefix. Confirm by printing `list(sd.keys())[:3]`. Strip it:

```python
sd = {k.removeprefix('module.'): v for k, v in sd.items()}
```

**2. Non-persistent or newly added buffers.** A mask or RoPE table registered with
`persistent=False` is absent from the checkpoint, and one added since the checkpoint was
written will be reported missing. Confirm by checking whether the missing names are
buffers rather than parameters. These are safe to ignore — pass `strict=False` once you
have verified that every missing key is a recomputable buffer.

**3. A genuine architecture change.** Renamed layers, a different depth, or a changed
hidden size. Confirm by comparing shapes, not just names:

```python
model_sd = model.state_dict()
for k in set(sd) & set(model_sd):
    if sd[k].shape != model_sd[k].shape:
        print(k, sd[k].shape, '->', model_sd[k].shape)
```

Never reach for `strict=False` before doing this. It silences the warning and leaves the
mismatched layers randomly initialised, which presents later as a model that trains but
performs inexplicably badly.
:::

## What to carry forward

- `nn.Parameter` trains, buffers travel, plain attributes do neither.
- A plain Python list of modules registers nothing — use `nn.ModuleList`.
- Print the parameter count after construction; it catches registration bugs instantly.
- `eval()` only flips a flag — you still need `no_grad()`.
- Remove hooks and detach what they capture.
