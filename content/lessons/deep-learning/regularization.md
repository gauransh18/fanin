---
summary: Weight decay, dropout and early stopping as three answers to the same question, and why pretraining needs almost none of them.
prereqs: [maximum-likelihood, bayes-and-independence]
---

Regularisation trades training fit for generalisation. From lesson 1.11's decomposition,
every technique here increases bias to decrease variance — the question is only which
form of bias is least harmful.

## Weight decay

Add a penalty on parameter magnitude:

$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2}\lVert\theta\rVert_2^2
$$

By lesson 1.07, $\nabla\lVert\theta\rVert^2 = 2\theta$, so the gradient gains $\lambda\theta$
and each step multiplies weights by $(1-\eta\lambda)$ before anything else. By lesson 1.13,
this is a Gaussian prior centred at zero.

Use **AdamW**, not `Adam(weight_decay=...)` — lesson 2.09 explains why coupling decay to
the adaptive scaling inverts its effect.

```python
import torch

# Decay matrices only. Decaying a LayerNorm gain pulls it toward zero,
# fighting the normalization it exists to scale.
decay    = [p for p in model.parameters() if p.ndim >= 2]
no_decay = [p for p in model.parameters() if p.ndim <  2]
opt = torch.optim.AdamW([
    {'params': decay,    'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=3e-4)
```

Typical values: 0.1 for transformer pretraining, 0.01–0.05 for fine-tuning,
$10^{-4}$ for older vision models with SGD. The large transformer value looks alarming
next to classical L2 coefficients; it is not comparable, because AdamW's decay is not
scaled by $1/\sqrt{v}$.

## Dropout

Zero each activation independently with probability $p$ during training, and scale the
survivors by $1/(1-p)$ so the expected value is unchanged:

```python
import torch.nn as nn

nn.Dropout(p=0.1)      # inverted dropout: scaling happens in training
```

The scaling during training — rather than at inference — is why `eval()` makes dropout a
pure identity with no correction needed.

Three ways to read what it does:

- **Ensemble.** Each forward pass samples a different subnetwork; inference approximates
  averaging over $2^n$ of them.
- **Co-adaptation prevention.** No unit can rely on a specific other unit being present,
  so features must be individually useful.
- **Noise injection.** Equivalent to adding multiplicative noise, which is a form of data
  augmentation in representation space.

::: warning
**Dropout is largely absent from modern LLM pretraining.** The reason is a change in
regime: when a model sees each token roughly once, there is nothing to memorise, so
overfitting is not the binding constraint. Dropout then only slows convergence.

It remains standard for **fine-tuning** on small datasets, where the same examples are
seen many times — and there $p = 0.1$ is typical.
:::

::: check
Why is 0.1 a normal weight decay for transformer pretraining when classical L2 coefficients are around $10^{-4}$?

- [x] AdamW's decay is applied directly to the parameters, not scaled by $1/\sqrt{v}$
  > The two numbers are not on the same scale. Coupling decay to the adaptive step, as plain `Adam(weight_decay=...)` does, changes both its magnitude and its direction of effect. The values are not comparable across the two formulations.
- [ ] Transformers overfit far more, so they need stronger regularisation
  > Pretraining on trillions of tokens is close to the opposite regime — regularisation matters most when data is scarce, as lesson 1.14 argued.
- [ ] Larger models tolerate larger decay because they have more parameters
  > Parameter count does not rescale the penalty this way.
- [ ] It is applied per layer rather than globally
  > It is applied per parameter group, and the grouping is about which tensors get decay at all, not about the scale.
:::

## Early stopping

Stop when validation loss stops improving:

```python
best, patience, waited = float('inf'), 5, 0
for epoch in range(max_epochs):
    train_one_epoch()
    val = evaluate()
    if val < best - 1e-4:
        best, waited = val, 0
        torch.save(model.state_dict(), 'best.pt')
    else:
        waited += 1
        if waited >= patience:
            break
```

For L2-regularised linear models, early stopping is provably equivalent to weight decay
with a $\lambda$ determined by the stopping time — they are the same regulariser in
different coordinates.

::: key
**Early stopping biases your reported number.** Selecting the best of $N$ validation
evaluations is a maximisation over noise, so the selected value is optimistically biased
by roughly one standard deviation of the evaluation noise — the effect flagged in lesson
2.15's exercise.

If you stop on validation, report the **test** number. If you have only one held-out set,
you cannot both select on it and report it honestly.
:::

## Data augmentation

The most effective regulariser, where it applies: rather than constraining the model,
enlarge the data with transformations that preserve the label.

```python
import torch

# Mixup: convex combinations of examples and their labels.
def mixup(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

x_mix, y_a, y_b, lam = mixup(x, y)
loss = lam * F.cross_entropy(model(x_mix), y_a) + \
       (1 - lam) * F.cross_entropy(model(x_mix), y_b)
```

Augmentation encodes an **invariance** you believe holds: a rotated cat is a cat. Its
power comes from injecting real prior knowledge rather than generic smoothness. For text
it is much harder — most edits change meaning — which is one reason language models rely
on scale instead.

::: check
Why exempt LayerNorm gains and biases from weight decay?

- [x] Pulling a normalization gain toward zero fights the very scaling the layer exists to provide
  > The conventional split is by rank: decay tensors with `ndim >= 2`, exempt the rest. Biases are similar — shrinking them toward zero constrains the model without buying generalisation.
- [ ] They have too few parameters for decay to matter
  > They are few, and the effect on them is large precisely because there is nothing else setting their scale.
- [ ] Decay is undefined for one-dimensional tensors
  > It is perfectly well defined. The reason is what it does, not whether it can.
- [ ] They are not trained parameters, so decay would have nothing to act on
  > They are trained. That is what makes decaying them harmful.
:::

## Which one, when

| Situation | Use |
|---|---|
| LLM pretraining, one epoch | Weight decay only |
| Fine-tuning on thousands of examples | Weight decay + dropout 0.1 + early stopping |
| Vision from scratch | Augmentation + weight decay + label smoothing |
| Tiny dataset, thousands of examples | All of the above, plus a smaller model |
| Validation ≫ training loss | More regularisation, or more data |
| Both losses high and flat | **Less** regularisation — you are underfitting |

That last row is the one people get backwards. If training loss itself is poor, the model
is not overfitting and adding regularisation makes it worse. Lesson 3.16 turns this into a
diagnostic.

::: exercise
You fine-tune a 7B model on 500 examples. Training loss reaches near zero in two epochs;
validation loss rises from epoch one. List four interventions in the order you would try
them.
:::

::: solution
**1. LoRA instead of full fine-tuning.** 500 examples cannot identify 7 billion
parameters — the problem is wildly under-determined. LoRA with $r=8$ trains a few million
parameters instead, which is a capacity constraint far stronger than any penalty you could
add. This is the highest-value change and it is also cheaper. Lesson 5.06 covers it.

**2. Fewer epochs, and a lower learning rate.** Validation rising from epoch one means the
optimum was somewhere inside epoch one. Try one epoch at a third of the learning rate, and
evaluate several times *within* the epoch rather than only at its end.

**3. More data, even imperfect data.** Going from 500 to 5,000 examples will beat every
regularisation technique. Synthetic augmentation, related public datasets, or mixing in a
slice of general instruction data all help — the last also mitigates catastrophic
forgetting, which at this data scale is a bigger risk than overfitting.

**4. Standard regularisation.** Weight decay 0.01–0.1, dropout 0.05–0.1, early stopping on
a proper validation split. Useful, but the smallest lever of the four.

**What not to do:** do not train longer hoping it recovers, and do not tune on the set you
intend to report. With 500 examples you likely have around 100 in validation, which by
lesson 1.11's Hoeffding bound gives roughly ±10 points of accuracy at 95% confidence —
most of the differences you will observe between these interventions are not measurable at
that size. Use cross-validation and report the spread.
:::

## What to carry forward

- Weight decay is a Gaussian prior; apply it to matrices, not to norms and biases.
- Dropout matters for fine-tuning and is largely absent from single-epoch pretraining.
- Early stopping is weight decay in disguise, and biases whatever number you select on.
- Augmentation beats penalties when you have a real invariance to encode.
- High training loss means underfitting — reduce regularisation, do not add it.
