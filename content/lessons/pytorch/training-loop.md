---
summary: The loop written properly once — ordering, accumulation, evaluation, checkpointing — with the reasons for each line.
prereqs: [optimizers-schedulers, datasets-dataloaders, autograd-graph]
---

Most training loops are copied and then subtly corrupted. This is the loop written out
with every line justified, so you can modify it knowingly.

## The five steps, in order

```python
for batch in loader:
    loss = model(batch)          # 1. forward
    loss.backward()              # 2. backward
    clip_grad_norm_(...)         # 3. clip
    optimizer.step()             # 4. update
    optimizer.zero_grad()        # 5. clear
```

The ordering constraints are real:

- **Clip after backward, before step.** Clipping needs gradients to exist and must happen
  before they are consumed.
- **`zero_grad` after step, not before backward.** Both orderings work for plain training,
  but only "after step" is correct once you add gradient accumulation, because you need
  gradients to survive across microbatches.
- **Scheduler after optimizer.** `scheduler.step()` reads the optimizer's state. Calling
  it first produces an off-by-one learning rate for the whole run.

## The full loop

```python
import math
import torch
from torch.nn.utils import clip_grad_norm_

def train(model, loader, val_loader, *, steps, accum=1, lr=3e-4,
          warmup=2000, clip=1.0, device='cuda', log_every=50, eval_every=1000):

    model.to(device)

    # Decay matrices, not biases or norms -- see lesson 2.08.
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad]
    other = [p for n, p in model.named_parameters() if p.ndim <  2 and p.requires_grad]
    opt = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': 0.1},
         {'params': other, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # Accumulate on-device so nothing synchronises inside the loop.
    running = torch.zeros((), device=device)
    it, step = iter(loader), 0

    while step < steps:
        model.train()
        opt.zero_grad(set_to_none=True)

        for micro in range(accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(x, y)

            # Divide so the accumulated gradient is the MEAN over microbatches.
            (loss / accum).backward()
            running += loss.detach() / accum

        grad_norm = clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        step += 1

        if step % log_every == 0:
            avg = (running / log_every).item()        # the only sync, deliberate
            print(f'step {step:6d}  loss {avg:.4f}  '
                  f'lr {sched.get_last_lr()[0]:.2e}  gnorm {grad_norm:.2f}')
            running.zero_()

        if step % eval_every == 0:
            print(f'  val {evaluate(model, val_loader, device):.4f}')
            save({'model': model.state_dict(), 'opt': opt.state_dict(),
                  'sched': sched.state_dict(), 'step': step}, f'ckpt-{step}.pt')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            total += model(x, y) * x.size(0)
        n += x.size(0)
    model.train()
    return (total / n).item()
```

::: check
Which orderings in the training loop are genuinely load-bearing?

- [x] Clip after backward and before step
  > Clipping needs the gradients to exist, and has to happen before the optimizer consumes them.
- [x] `zero_grad` after step rather than before backward
  > Both work for plain training. Only "after step" survives gradient accumulation, where gradients must live across microbatches.
- [x] `scheduler.step()` after `optimizer.step()`
  > The scheduler reads the optimizer's state. Calling it first gives an off-by-one learning rate for the entire run.
- [ ] Forward before moving the batch to the device
  > The batch has to be on the device *before* the forward pass, not after — this ordering is not optional, it is impossible.
:::

## The lines that are easy to get wrong

::: key
**`loss / accum`.** Gradients accumulate by summing (lesson 1.08). Without the division,
accumulating over 8 microbatches gives a gradient 8× larger than the equivalent large
batch — effectively multiplying your learning rate by 8 and diverging. Divide the loss,
not the gradients.

**`zero_grad(set_to_none=True)`.** Setting gradients to `None` rather than zero skips a
kernel launch per parameter and lets the optimizer skip parameters entirely. It is the
default in recent PyTorch and is strictly better, with one caveat: code that reads
`p.grad` unconditionally now needs a `None` check.

**`running += loss.detach()`.** Without `detach`, `running` holds a reference to every
step's graph and memory grows without bound — the exact bug from lesson 2.06.

**`model.train()` restored after eval.** Forgetting this leaves dropout off and BatchNorm
in inference mode for the rest of training. The loss curve looks *better*, because dropout
is disabled, which is what makes it hard to notice.
:::

::: check
With gradient accumulation over `accum` microbatches, the loop calls `(loss / accum).backward()`. Why the division?

- [x] Gradients accumulate by summing
  > Dividing makes the total the *mean* over microbatches — matching what one large batch would give. Without it, the effective gradient scale grows with `accum`, and the learning rate you tuned at `accum=1` is suddenly `accum` times too large.
- [ ] To keep the loss value comparable when logging
  > The logged number does change, but the reason is the gradient, not the display. You would still divide if you logged nothing.
- [ ] To prevent overflow in bf16
  > bf16 has fp32's exponent range and does not overflow here. This division is about scale, not range.
- [ ] Because `backward` assigns rather than accumulates, so only the last microbatch would count
  > It is the other way around: `backward` accumulates, which is precisely why the sum needs normalising.
:::

## Checkpoint everything the run depends on

```python
def save(state, path):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)     # atomic: a killed job never leaves a half file
```

A resumable checkpoint needs the model, the optimizer state, the scheduler state, the
step counter, and the RNG state. Saving only `model.state_dict()` means resuming with
Adam's moments reset to zero, which produces a visible loss spike for hundreds of steps
while they rebuild.

The `os.replace` is not fussiness. Jobs get preempted mid-write, and a truncated
checkpoint discovered three days later is a genuinely expensive mistake.

## Sanity checks before the real run

Run these in order. Each takes minutes and each catches a different class of bug.

1. **Forward shape check.** One batch through the model; assert the output shape and that
   the initial loss is near $\ln(\text{vocab})$ — 11.7 for a 120k vocabulary. A very
   different value means the output layer or the loss reduction is wrong.
2. **Overfit one batch.** Train on a single batch for 200 steps. The loss must reach
   nearly zero. If it cannot, the model cannot learn at all and nothing else matters.
   This is the single highest-value check in this curriculum — lesson 3.16 builds on it.
3. **Gradient flow.** Print the gradient norm per layer after one backward. A layer with
   zero gradient is disconnected; one with a norm orders of magnitude above the rest will
   dominate.
4. **Short run with checkpointing.** 100 steps, save, kill, resume. Confirm the loss
   continues smoothly rather than jumping.

::: warning
Do not start a multi-day run without step 2. A model that cannot overfit a single batch
has a structural bug — a detached path, a wrong mask, a target misalignment — and no
amount of tuning will fix it. The check costs five minutes and has saved more GPU-hours
than any other item on this list.
:::

::: exercise
You add gradient accumulation with `accum=4` and the loss diverges immediately, though
the same code worked at `accum=1`. What is the most likely cause?
:::

::: solution
The loss is almost certainly not being divided by `accum`.

Gradients accumulate additively, so four microbatch backwards produce
$g_1 + g_2 + g_3 + g_4$, not their mean. That is 4× the gradient of the equivalent large
batch, so the effective learning rate is 4× what you set — and if your configured rate
was already near the stability limit $2/\lambda_{\max}$ from lesson 1.09, it now exceeds
it and diverges on the first step.

The fix is `(loss / accum).backward()`, as in the loop above.

**Two other candidates worth ruling out.** First, `zero_grad()` placed inside the
microbatch loop instead of outside it — that discards every microbatch but the last, so
you are training on a quarter of the data with a normal-size step, which degrades rather
than diverges. Second, calling `sched.step()` per microbatch rather than per optimizer
step, which advances the schedule 4× too fast and blows through warmup in a quarter of
the intended steps.

To confirm it is the division: log the gradient norm. With the bug it will be
approximately 4× the `accum=1` value from the very first step.
:::

## What to carry forward

- forward → backward → clip → step → zero, and scheduler after optimizer.
- Divide the loss by the accumulation count.
- Detach anything you accumulate for logging.
- Checkpoint optimizer and scheduler state, and write atomically.
- Overfit one batch before every real run.
