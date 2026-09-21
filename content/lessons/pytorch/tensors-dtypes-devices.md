---
summary: What a tensor is in memory, which dtype to reach for, and the device transfers that quietly dominate your training time.
---

A tensor is not a mathematical object in PyTorch. It is a **typed pointer into a flat
buffer**, plus metadata describing how to interpret it. Holding that picture makes the
whole API predictable.

## The anatomy of a tensor

Every tensor has four pieces of state:

```python
import torch

x = torch.randn(2, 3, 4)

x.shape       # torch.Size([2, 3, 4])  -- logical dimensions
x.dtype       # torch.float32          -- element type
x.device      # device(type='cpu')     -- where the buffer lives
x.stride()    # (12, 4, 1)             -- how to walk the buffer
x.storage_offset()  # 0                -- where this view starts
```

The underlying storage is one-dimensional and contiguous. Shape and stride are the lens
through which you view it. Lesson 2.02 makes strides the centrepiece; for now, note that
`x[1, 2, 3]` is found at flat index $1{\cdot}12 + 2{\cdot}4 + 3{\cdot}1 = 23$.

## Dtypes and what they cost

| dtype | Bits | Range | Precision | Use |
|---|---|---|---|---|
| `float32` | 32 | $\pm 3.4\times10^{38}$ | ~7 digits | Default; master weights |
| `float16` | 16 | $\pm 65{,}504$ | ~3 digits | Legacy mixed precision |
| `bfloat16` | 16 | $\pm 3.4\times10^{38}$ | ~2 digits | Modern training default |
| `float8_e4m3fn` | 8 | $\pm 448$ | ~1 digit | Frontier-scale training |
| `int8` | 8 | $-128..127$ | exact | Quantized inference |
| `int64` | 64 | huge | exact | Indices, token IDs |

The float16 versus bfloat16 distinction deserves attention because it causes real bugs.
Both use 16 bits, but they split them differently: fp16 gives 10 bits to the mantissa
and 5 to the exponent; bf16 gives 7 and 8. Since bf16 has the *same exponent range as
fp32*, it never overflows where fp32 would not, so it needs no loss scaling.

::: warning
fp16's maximum is 65,504. An attention logit sum or an un-scaled gradient can exceed
that and become `inf`, which then poisons everything downstream as `nan`. This is the
single most common cause of "loss became nan at step 3,000" on older hardware. bf16
trades mantissa bits for range precisely to avoid it — use bf16 if your hardware has it.
:::

```python
import torch

big = torch.tensor(70000.0)
big.half()       # tensor(inf, dtype=torch.float16)   -- overflow
big.bfloat16()   # tensor(70144., dtype=torch.bfloat16) -- fine, less precise

# bf16 rounds aggressively but never overflows in fp32's range.
torch.tensor(1.0001).bfloat16()   # tensor(1.0000)
```

## Devices, and the transfer you did not notice

A tensor lives on exactly one device. Operations require all operands on the same one.

```python
import torch

x = torch.randn(1000, 1000)
if torch.cuda.is_available():
    x = x.to('cuda')          # allocates on GPU, copies over PCIe
    y = x.cpu()               # copies back -- and synchronises
```

Two facts about transfers matter more than anything else in this lesson.

**GPU operations are asynchronous.** Launching a kernel returns immediately; the work is
queued. Any operation that needs the result on the CPU — `.item()`, `.cpu()`,
`print(tensor)`, an `if` on a tensor value — forces a **synchronisation**, stalling the
pipeline until the queue drains.

**Transfers are slow.** PCIe gives roughly 16–32 GB/s, against 2,000–3,000 GB/s of
on-device memory bandwidth on a modern accelerator. A round trip costs about 100× what
the same data movement costs on-device.

::: key
The classic training-loop bug:

```python
for batch in loader:
    loss = model(batch).mean()
    loss.backward()
    losses.append(loss.item())   # synchronises EVERY step
```

`.item()` blocks until the GPU finishes. Accumulate on-device and transfer once:

```python
running = torch.zeros((), device='cuda')
for batch in loader:
    loss = model(batch).mean()
    loss.backward()
    running += loss.detach()     # stays on GPU, no stall
print((running / n).item())      # one sync, at the end
```
:::

**Pinned memory** lets a CPU→GPU copy overlap with compute, because the pages cannot be
swapped out. `DataLoader(..., pin_memory=True)` combined with `.to(device,
non_blocking=True)` is nearly free throughput — lesson 2.10 covers the rest.

## Creating tensors without surprises

```python
import torch

torch.zeros(3, 4)                       # filled
torch.empty(3, 4)                       # uninitialised -- contains garbage
torch.arange(0, 10, 2)                  # tensor([0, 2, 4, 6, 8])
torch.linspace(0, 1, 5)                 # tensor([0.00, 0.25, 0.50, 0.75, 1.00])

# Match an existing tensor's dtype AND device in one call.
ref = torch.randn(3, 4, device='cuda', dtype=torch.bfloat16)
torch.zeros_like(ref)                   # same shape, dtype, device
ref.new_zeros(2, 2)                     # different shape, same dtype/device
```

Prefer `*_like` and `new_*` over hard-coding `device='cuda'`. Code that names a device
literally breaks the moment someone runs it on CPU, on a second GPU, or under a
distributed wrapper.

::: warning
`torch.tensor(data)` **copies**. `torch.as_tensor(data)` shares memory with a NumPy array
when it can. `torch.from_numpy(a)` always shares — so mutating `a` mutates the tensor,
and vice versa. Sharing is faster and occasionally catastrophic; know which one you
called.
:::

::: exercise
You are training a 7B-parameter model with Adam in mixed precision. Account for the
memory used by parameters and optimizer state, and say which pieces could be shed.
:::

::: solution
The standard mixed-precision recipe keeps:

- **bf16 parameters** for the forward and backward pass: $7\times10^9 \times 2$ B = **14 GB**
- **fp32 master parameters**, since bf16 has too few mantissa bits to accumulate small
  updates: $\times 4$ B = **28 GB**
- **fp32 gradients**: **28 GB**
- **Adam first moment** (fp32): **28 GB**
- **Adam second moment** (fp32): **28 GB**

Total **126 GB** — before a single activation. That is why a 7B model does not train on
one 80 GB card without sharding.

**What can be shed.** ZeRO/FSDP (lesson 7.07) shards the master weights, gradients and
optimizer state across $N$ ranks, cutting the 112 GB of fp32 state by a factor of $N$
and leaving only the 14 GB of bf16 parameters replicated. 8-bit Adam stores the moments
in int8 with block-wise scaling, cutting 56 GB to 14 GB. And SGD with momentum carries
one moment instead of two, though it converges worse on transformers.
:::

## What to carry forward

- A tensor is shape + stride + dtype + device over a flat buffer.
- bf16 has fp32's exponent range; fp16 overflows at 65,504.
- `.item()` and `print()` synchronise — keep accumulators on-device.
- Use `*_like` and `new_*` instead of hard-coding a device.
- Adam in mixed precision costs about 18 bytes per parameter before activations.
