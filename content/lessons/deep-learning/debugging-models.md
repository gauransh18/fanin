---
summary: A decision procedure for a model that will not learn, ordered by how much time each check costs versus how much it rules out.
prereqs: [training-loop, initialization, normalization, generalization-double-descent]
---

Most debugging time is wasted on the wrong hypothesis. This lesson is an ordering: each
check is cheap, and each one eliminates a large class of causes. Work down the list rather
than guessing.

## Check 1: is the initial loss right?

Before anything else, run one forward pass and compare the loss to the value a *random*
model should produce.

| Task | Expected initial loss |
|---|---|
| $K$-class classification | $\ln K$ — 2.303 for 10, 11.7 for 120,000 |
| Binary cross-entropy | $\ln 2 = 0.693$ |
| MSE on standardised targets | $\approx 1.0$ |

```python
import math, torch

with torch.no_grad():
    loss = model(next(iter(loader)))
print(f'initial {loss.item():.3f}, expected {math.log(vocab_size):.3f}')
```

::: key
This one line catches an enormous share of real bugs, because almost every structural
mistake changes it:

- **Much higher** — the output layer is badly initialised, or logits are being scaled,
  or the labels are shifted so the model is being asked to predict the wrong token.
- **Much lower** — label leakage. The model can already see the answer. Check your
  causal mask (lesson 4.06) and your target alignment.
- **Exactly zero** — the loss is being computed against itself, or every target is masked
  out and you are averaging over an empty set.
:::

::: check
A 120,000-token language model reports an initial loss of 0.4 before any training. What does that indicate?

- [x] Label leakage: the model can already see the answer
  > A random model should sit near $\ln 120000 \approx 11.7$. Check the causal mask and the target alignment. A loss far *above* 11.7 points the other way, at initialisation or logit scaling or shifted labels.
- [ ] Excellent initialisation, which will speed up training
  > No initialisation scheme can beat $\ln K$ before training, because a random model has no information about the targets.
- [ ] The loss is being averaged over too few tokens
  > Averaging over few tokens makes the number noisy, not systematically a factor of thirty too small.
- [ ] The vocabulary is smaller than configured
  > Even a two-token vocabulary would start near $\ln 2 = 0.693$, which is still above 0.4.
:::

## Check 2: can it overfit one batch?

Take a single batch of 8 examples and train on it for 200 steps with no regularisation.

```python
import torch

batch = next(iter(loader))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(300):
    loss = model(batch)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f'{step:4d}  {loss.item():.6f}')
```

The loss **must** approach zero. A model that cannot memorise 8 examples has a structural
fault, and no amount of learning-rate tuning or data cleaning will fix it.

If it fails, the cause is in a short list:

- A **detached path** — some part of the computation is outside the graph (lesson 2.06).
- An **unregistered parameter** — a plain Python list instead of `nn.ModuleList` (lesson 2.08).
- **Zero gradients** reaching some layer.
- A **wrong loss**, for example a target and prediction that do not correspond.

## Check 3: where does the gradient stop?

```python
import torch

loss.backward()
for name, p in model.named_parameters():
    if p.grad is None:
        print(f'{name:50s} NO GRADIENT')
    else:
        print(f'{name:50s} {p.grad.norm():.3e}')
```

Read it for three patterns:

- **`None`** means the parameter is not in the graph at all. Unregistered, or detached.
- **Exactly zero** means the gradient reached it and was zero — a dead ReLU (lesson 3.02),
  or a fully masked output.
- **Geometric decay across layers** means vanishing gradients: wrong initialisation
  (lesson 3.05) or missing residual connections (lesson 3.08).

## Check 4: what do the activations look like?

```python
import torch

acts = {}
hooks = [m.register_forward_hook(
            lambda mod, inp, out, n=n: acts.__setitem__(n, out.detach()))
         for n, m in model.named_modules() if len(list(m.children())) == 0]

model(batch)
for n, a in acts.items():
    if a.dtype.is_floating_point:
        print(f'{n:40s} mean {a.mean():+.3f}  std {a.std():.3f}  '
              f'zeros {(a == 0).float().mean():.1%}')
for h in hooks:
    h.remove()
```

What to look for:

- **std growing or shrinking geometrically** — initialisation is wrong (lesson 3.05).
- **std at zero** — the layer is dead.
- **Zero fraction above 90%** after a ReLU — dying ReLU (lesson 3.02).
- **`nan` or `inf`** — find the *first* layer where it appears; everything after is
  downstream damage.

## Check 5: is the data what you think it is?

```python
x, y = next(iter(loader))
print('shapes ', x.shape, y.shape)
print('x range', x.min().item(), x.max().item())
print('labels ', torch.bincount(y.flatten()[y.flatten() >= 0]))
# Decode a sample back to text/pixels and LOOK at it with its label.
print(tokenizer.decode(x[0]), '->', tokenizer.decode(y[0]))
```

Common findings, all of which produce a model that trains and is bad:

- Inputs in $[0, 255]$ instead of normalised.
- Labels off by one, or shifted by one position for next-token prediction.
- A class that never appears.
- Padding counted in the loss because `ignore_index` was not set (lesson 2.10).

## The learning-rate sweep

Once the structural checks pass, find the learning rate before tuning anything else:

```python
import torch

lrs, losses = [], []
lr = 1e-7
for batch in loader:
    for g in opt.param_groups:
        g['lr'] = lr
    loss = model(batch)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    lrs.append(lr); losses.append(loss.item())
    lr *= 1.1
    if lr > 1 or loss.item() > 4 * losses[0]:
        break
# Plot losses against lrs on a log axis. Pick roughly one order of magnitude
# below where the curve turns upward.
```

::: check
Why is "overfit a single batch of eight examples" such a valuable early check?

- [x] Failing it puts the bug in the model, the loss or the optimizer
  > A single batch should reach near-zero loss, so anything that stops it is structural rather than a data or hyperparameter problem. It removes generalisation from the picture entirely, so anything that goes wrong is structural. Each check in the list eliminates a large class of causes cheaply, which is why the ordering matters more than any individual test.
- [ ] It proves the model will generalise to the full dataset
  > It proves nothing about generalisation. Memorising eight examples is the *easiest* thing a network can do.
- [ ] It measures how fast training will be at scale
  > Throughput on eight examples tells you almost nothing about a full run.
- [ ] It verifies the data loader is shuffling correctly
  > It uses one fixed batch, so shuffling never comes into it.
:::

## The symptom table

| Symptom | Most likely cause | Check |
|---|---|---|
| Loss flat at $\ln K$ | Learning rate far too low, or gradients not flowing | 3 |
| Loss `nan` immediately | LR too high, or fp16 overflow (lesson 2.12) | 4 |
| Loss `nan` after N steps | A bad batch, or an unguarded `log`/`sqrt` of zero | 4 |
| Loss decreases then spikes | LR too high for current curvature (lesson 1.09) | — |
| Train loss falls, val does not | Overfitting, or too small (lesson 3.15) | 5 |
| Val loss much lower than train | Dropout still on at eval, or a leak | 5 |
| Loss identical every run | Data not shuffled, or seed fixed inside the loop | 5 |
| Memory grows across steps | Retained graph (lesson 2.06) | — |

## The one rule

::: warning
**Change one thing at a time, and write down what you changed.** Tuning three
hyperparameters at once and getting a better result teaches you nothing, because you
cannot attribute the improvement — and the next time it regresses you have no way back.

Keep a log: hypothesis, change, result. Most debugging failures are failures of
bookkeeping rather than of insight.
:::

::: exercise
Your language model's training loss drops from 11.7 to 6.2 in 100 steps, then plateaus
there for 10,000 steps. Validation matches. What is happening?
:::

::: solution
The plateau value is the clue. $e^{6.2} \approx 493$, so the model is behaving as though
choosing uniformly among roughly 500 tokens. It has learned the **unigram distribution** —
which tokens are frequent — and nothing about context.

That is exactly the loss you get from predicting the marginal token frequency regardless
of the prefix, and it is the first thing any language model learns, usually within a few
hundred steps. Getting stuck there means context is not reaching the prediction.

**Ordered hypotheses:**

1. **The attention mask is wrong.** If every position is masked out except itself, each
   token can only see itself, and the model reduces to a unigram predictor. Check by
   printing the mask for a short sequence and confirming it is lower-triangular, not
   diagonal (lesson 4.06).

2. **The targets are misaligned.** If targets are not shifted by one, the model is being
   asked to predict the *current* token from itself — which is trivially solvable and
   would give loss near zero — or to predict something unrelated. Print an input/target
   pair decoded to text and read it.

3. **Positional information is missing or broken.** Without positional encodings the model
   is permutation-invariant over the context, so it cannot use word order. Loss then sits
   around bag-of-words level, which is close to this.

4. **The learning rate collapsed.** Check the scheduler — a cosine schedule with the wrong
   total step count, or a warmup that never ends, can leave the rate near zero. Log the
   actual learning rate every step (lesson 2.11).

**The fastest discriminator** is check 2 from this lesson: try to overfit one batch. If
the model cannot drive a single batch to near-zero loss, it is a structural bug (1, 2 or
3). If it can, the architecture is fine and the problem is optimisation (4).
:::

## What to carry forward

- Compare initial loss to $\ln K$ before anything else.
- Overfit one batch. A model that cannot has a structural bug, not a tuning problem.
- Print gradient norms per parameter; `None`, zero and geometric decay each mean something specific.
- Look at your data decoded, with its labels, at least once.
- One change at a time, written down.
