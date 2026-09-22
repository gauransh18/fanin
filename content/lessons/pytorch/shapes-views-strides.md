---
summary: Strides explain why some reshapes are free and others copy, why transpose is instant, and what "contiguous" actually means.
prereqs: [tensors-dtypes-devices]
---

Almost every confusing PyTorch error message is a stride problem wearing a disguise.
This lesson makes strides explicit so those errors become obvious.

## Strides

A stride is **how many elements to skip in the flat buffer to advance one step along a
dimension**. For a contiguous tensor of shape $(d_0, d_1, d_2)$, the strides are
$(d_1 d_2,\; d_2,\; 1)$ — the last dimension is adjacent in memory, and each earlier
dimension steps over a whole slab.

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)
x.stride()          # (12, 4, 1)
x.is_contiguous()   # True

# Element [i,j,k] lives at flat offset i*12 + j*4 + k*1.
x[1, 2, 3].item(), x.flatten()[1*12 + 2*4 + 3].item()   # (23, 23)
```

## Views are free

A **view** shares the same buffer with different metadata. No data moves, so the
operation is $O(1)$ regardless of tensor size.

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)

y = x.transpose(0, 1)      # shape (3, 2, 4)
y.stride()                 # (4, 12, 1)  -- strides swapped, nothing copied
y.data_ptr() == x.data_ptr()   # True: same storage

y[0, 0, 0] = 999
x[0, 0, 0]                 # tensor(999) -- they share memory
```

Transpose does not rearrange anything. It swaps two numbers in the stride tuple. This is
why transposing a 10 GB tensor is instantaneous — and why the result is no longer
contiguous.

::: check
Transposing a 10 GB tensor returns instantly. What actually happened?

- [x] Two numbers in the stride tuple were swapped; the buffer was untouched
  > A view shares storage and changes only the lens. That is also why the result is no longer contiguous, and why writing through the view is visible in the original.
- [ ] The copy was queued asynchronously and will complete later
  > No copy is scheduled at all. `data_ptr()` on both tensors returns the same address.
- [ ] PyTorch stored a lazy transpose that is applied when the tensor is read
  > Strides are not lazy — they are how every read already works. There is no deferred operation to apply.
- [ ] The tensor was small enough to fit in cache
  > 10 GB is not in any cache, and size is irrelevant: a transpose is $O(1)$ at any size.
:::

## Contiguity, and why it matters

A tensor is **contiguous** when its elements sit in memory in the order a row-major
traversal would visit them — equivalently, when its strides match the canonical pattern
for its shape.

Non-contiguity is not an error. It becomes one when an operation requires a flat layout:

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(0, 1)

y.view(-1)          # RuntimeError: view size is not compatible with
                    # input tensor's size and stride
y.reshape(-1)       # works -- silently copies into a new contiguous buffer
y.contiguous().view(-1)   # explicit: copy, then a free view
```

::: key
- **`view`** requires compatible strides and never copies. It fails loudly when it cannot
  comply.
- **`reshape`** returns a view when it can and a copy when it cannot. It never fails, and
  it never tells you which happened.

Use `view` when you want a guarantee that nothing is copied. Use `reshape` when you do
not care. A `reshape` in a hot loop that silently copies a large tensor every iteration
is a classic invisible performance bug — lesson 2.13 shows how to find it.
:::

::: check
You want a guarantee that a reshape in a hot loop is not silently copying a large tensor. Which call do you reach for?

- [x] `view`, which fails loudly when the strides do not permit a free reshape
  > `reshape` returns a view when it can and a copy when it cannot, and never tells you which. A `reshape` copying a large tensor every iteration is a classic invisible performance bug.
- [ ] `reshape`, which never copies
  > It copies whenever the strides are incompatible — for instance after a transpose.
- [ ] `contiguous`, which guarantees the tensor is never copied
  > `contiguous` is the opposite: it *is* the copy, made explicit. It is a fine thing to call, just not a way to avoid copying.
- [ ] `permute`, which is always a view and never copies
  > `permute` is indeed always a view, but it reorders dimensions rather than reshaping them.
:::

## The shape-manipulation vocabulary

```python
import torch

x = torch.randn(2, 3, 4)

x.reshape(6, 4)              # merge leading dims
x.reshape(2, -1)             # -1 means "infer this one"
x.permute(2, 0, 1).shape     # (4, 2, 3) -- arbitrary reordering, a view
x.unsqueeze(1).shape         # (2, 1, 3, 4) -- insert a size-1 dim
x.squeeze().shape            # drop ALL size-1 dims -- rarely what you want
x.squeeze(1).shape           # drop dim 1 only if it is size 1 -- prefer this
x.flatten(1, 2).shape        # (2, 12) -- merge a specific range
```

::: warning
Bare `squeeze()` removes *every* size-1 dimension. With a batch size of 1 it will
silently drop your batch dimension, and the error surfaces three layers later as a
broadcasting mismatch. Always pass the dimension explicitly.
:::

## The transformer reshape you will write a hundred times

Splitting a $d$-dimensional residual stream into $h$ heads is the canonical
stride manipulation, and it is worth being able to write from memory:

```python
import torch

B, T, d, h = 2, 16, 512, 8
d_head = d // h

x = torch.randn(B, T, d)

# (B, T, d) -> (B, h, T, d_head): heads become a batch dimension.
q = x.view(B, T, h, d_head).transpose(1, 2)
q.shape                    # torch.Size([2, 8, 16, 64])
q.is_contiguous()          # False -- transpose left it strided

# ... attention happens here, over the last two dims ...

# Back to (B, T, d). transpose makes it non-contiguous, so view would fail.
out = q.transpose(1, 2).contiguous().view(B, T, d)
```

The `.contiguous()` before `.view()` is not optional and not decoration. Omit it and you
get the "view size is not compatible" error, which is the single most-asked PyTorch
question on the internet.

Note the ordering too: `view(B, T, h, d_head)` then `transpose`, **not**
`view(B, h, T, d_head)`. The first splits the last dimension in place, which is what you
want; the second reinterprets the buffer entirely and silently scrambles the data across
positions and heads. It produces the right shape and wrong values — the worst kind of bug.

## Memory format

For 4-D image tensors, PyTorch supports a second physical layout:

```python
import torch

x = torch.randn(32, 64, 56, 56)                       # NCHW, the default
y = x.to(memory_format=torch.channels_last)           # NHWC physically

x.stride()   # (200704, 3136, 56, 1)
y.stride()   # (200704, 1, 3584, 64)  -- channels adjacent in memory
y.shape == x.shape   # True: the logical shape is unchanged
```

Channels-last matches what tensor cores want for convolutions and often gives a
meaningful speedup with no code change beyond the conversion. The logical indexing is
identical; only the strides differ.

::: exercise
`x` has shape $(B, T, d)$. Compare `x.view(B, h, T, d//h)` with
`x.view(B, T, h, d//h).transpose(1, 2)`. Both produce shape $(B, h, T, d/h)$. Why is only
one correct?
:::

::: solution
The flat buffer is laid out with $d$ varying fastest, then $T$, then $B$ — element
$(b,t,i)$ sits at offset $b\cdot Td + t\cdot d + i$.

**`view(B, T, h, d//h)`** splits only the last axis: index $(b,t,\eta,j)$ maps to
$b\cdot Td + t\cdot d + \eta\cdot d_h + j$. Head $\eta$ gets the contiguous slice
$[\eta d_h, (\eta+1)d_h)$ of the feature vector at position $t$ — correct. The subsequent
`transpose(1,2)` only reorders axes, moving no data.

**`view(B, h, T, d//h)`** reinterprets the buffer as if $h$ were the *second* axis:
index $(b,\eta,t,j)$ maps to $b\cdot Td + \eta\cdot Td_h + t\cdot d_h + j$. Head 0 now
receives the first $T d_h$ elements of the buffer — which is the full feature vector of
the first $T/h$ **positions**, not a feature slice of every position. It mixes the
position and feature axes.

Both produce the same shape and no error. The second produces silently wrong values, and
the model trains to a mediocre loss without ever crashing. Assert on a known-value tensor
when you write this kind of reshape.
:::

## What to carry forward

- Strides say how to walk the buffer; shape says how to interpret the walk.
- Transpose and permute are free metadata edits that break contiguity.
- `view` never copies and can fail; `reshape` never fails and may copy silently.
- Split the last dimension *then* transpose — the other order scrambles the data.
- Always pass a dimension to `squeeze`.
