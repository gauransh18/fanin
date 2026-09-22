---
summary: Why the classical bias-variance picture predicts the wrong thing for large models, what double descent shows, and what actually explains generalisation.
prereqs: [expectation-variance-concentration, regularization]
---

Classical learning theory says a model with more parameters than data points will
memorise and fail to generalise. Modern models have hundreds of times more parameters than
data points and generalise well. This lesson is about what the classical picture got
right, where it breaks, and what to actually rely on.

## The classical picture

From lesson 1.11, expected error decomposes as bias² + variance + irreducible noise. The
textbook story: as capacity grows, bias falls and variance rises, so test error is
U-shaped with an optimum in the middle.

This is correct — for the model classes it was derived for. It is also where the received
wisdom "more parameters means overfitting" comes from.

## Double descent

Plot test error against model size and you see something the U-shape does not predict:

$$
\text{error} \nearrow \text{at the interpolation threshold, then} \searrow \text{again}
$$

Three regimes:

1. **Underparameterised.** Fewer parameters than needed to fit the training data. Test
   error falls as capacity grows. Classical behaviour.
2. **The interpolation threshold.** Just enough capacity to fit the training set exactly.
   Test error **peaks**, often catastrophically. There is exactly one interpolating
   solution and no freedom to choose a good one.
3. **Overparameterised.** Far more capacity than needed. Test error falls again, often
   below the first regime's best.

::: key
The explanation is not about capacity but about **choice among interpolating solutions**.
In regime 3 there are infinitely many parameter settings that fit the training data
exactly. Gradient descent does not pick one at random — it has an implicit bias toward
minimum-norm, low-complexity solutions, and those generalise.

At the threshold, the single interpolating solution is whatever the data forces, typically
with enormous weights and wild behaviour between data points. That is the peak.
:::

The same shape appears along other axes: **epoch-wise** double descent (test error rises,
then falls again with longer training) and **sample-wise** double descent (more data can
temporarily *hurt*, by moving you toward the threshold).

## Implicit regularisation

Gradient descent is not a neutral search. For linear regression started at zero, it
converges to the **minimum $\ell_2$-norm** interpolating solution. For separable logistic
regression, it converges in direction to the **max-margin** classifier — the same solution
an SVM finds, without anyone asking for it.

For deep networks the characterisation is not known in general, but the empirical pattern
is robust: SGD finds solutions that are flatter and simpler than a random interpolator.
Three contributing factors, in rough order of confidence:

- **Gradient noise** biases toward flat minima, whose Hessian eigenvalues are small
  (lesson 1.09). Noise cannot persist in a sharp minimum.
- **Small initialisation** means the network starts near a simple function and gradient
  descent finds the nearest interpolator, not an arbitrary one.
- **Early stopping**, even implicitly, limits how far from the initialisation the weights
  travel.

::: check
Test error peaks at the interpolation threshold and falls again beyond it. What is the explanation?

- [x] It is about choice among interpolating solutions
  > Past the threshold there are infinitely many, and gradient descent's implicit bias picks low-norm ones. At the threshold there is essentially one interpolating solution, whatever the data forces, typically with enormous weights and wild behaviour between data points. Capacity is not the variable that matters; the freedom to choose is.
- [ ] Larger models have lower variance, so the bias–variance curve turns over
  > The classical decomposition predicts the first U-shape and does not predict the second descent at all.
- [ ] The extra parameters act as explicit regularisation
  > Nothing explicit is added. The regularisation is implicit, and it comes from the optimizer rather than the objective.
- [ ] It is an artefact of early stopping in the experiments
  > Epoch-wise double descent shows the same shape *along* training time, which is the opposite of an early-stopping artefact.
:::

## Grokking

A related phenomenon worth knowing because it breaks the usual stopping heuristics: on
small algorithmic datasets, a model can reach 100% training accuracy with chance-level
validation accuracy, sit there for **thousands** of steps, and then suddenly generalise.

The current understanding is that the network first memorises — a high-norm solution that
fits the data — and weight decay slowly pushes it toward the lower-norm general solution,
which happens to also fit. Memorisation is fast and generalisation is slow, and the
transition is abrupt.

::: warning
Grokking means early stopping on validation loss can stop you **before** the model has
learned anything general. Two guards: keep training past the point validation stops
improving when the run is cheap, and watch weight norm alongside loss — a norm that is
still falling means the solution is still changing, even if the metrics are flat.
:::

::: check
Sample-wise double descent means more data can temporarily *hurt*. How?

- [x] It moves a fixed-size model towards its interpolation threshold
  > At the threshold the single interpolating solution is whatever the data forces. The same peak, approached along a different axis. It is a striking reminder that "more data is always better" is a statement about the underparameterised regime.
- [ ] More data increases label noise, which dominates at the margin
  > Noise proportion is unchanged by collecting more of the same distribution.
- [ ] Larger datasets require smaller learning rates, slowing convergence
  > A tuning consequence, not the mechanism behind the peak.
- [ ] It only happens with corrupted labels
  > Label noise sharpens the effect and is not required for it.
:::

## What predicts generalisation

Parameter count does not. Some things that do better:

**Data scale and quality.** The single strongest lever. Lesson 5.03's scaling laws make
this quantitative: loss falls as a power law in data, with no sign of the classical
overfitting penalty when each example is seen roughly once.

**Effective capacity, not nominal.** Weight norm, margin, and sharpness correlate with
generalisation far better than parameter count. A 7B model trained with weight decay may
have lower effective capacity than a 100M model without.

**The flatness heuristic.** Flat minima generalise better (lesson 1.09's exercise) — with
the caveat stated there, that flatness is not reparameterisation-invariant, so the claim
needs care before it can be made precise.

**Single-epoch training.** When a model sees each token once, it cannot memorise it. This
is the real reason LLM pretraining needs so little regularisation (lesson 3.07) — the
overfitting regime is never entered.

## What to do in practice

| Observation | Diagnosis | Action |
|---|---|---|
| Train loss high | Underfitting | Bigger model, longer training, less regularisation |
| Train low, val high, small data | Genuine overfitting | More data, more regularisation, smaller model |
| Train low, val high, large data | Possibly the threshold | Try a **larger** model |
| Both flat, val not improving | Possibly grokking | Train longer; watch weight norm |
| Val worse with more data | Sample-wise descent | Keep adding data; it recovers |

That third row is the one that contradicts instinct, and it is why "shrink the model" is
no longer the automatic response to a train–validation gap.

::: exercise
You train models of 1M, 10M, 100M and 1B parameters on 100k examples. Test error is 0.25,
0.18, 0.31, 0.15. Explain the non-monotonicity and say what you would do next.
:::

::: solution
**The shape is double descent.** The 100M model sits near the interpolation threshold —
roughly enough capacity to fit 100k examples exactly, and no more. There is essentially
one interpolating solution available, it is high-norm and badly behaved between the
training points, and test error spikes to 0.31.

The 1B model is well past the threshold. Many interpolating solutions exist, and SGD's
implicit bias selects a low-norm one, giving the best result at 0.15.

**How to confirm it.** Three checks:
- Plot **training** error too. The threshold is where training error first reaches zero —
  if that happens around 100M, the diagnosis is confirmed.
- Measure weight norm at convergence. A spike at 100M matching the error spike is the
  signature.
- Add points at 30M, 50M, 200M, 300M. Double descent has a characteristic narrow peak; a
  single bad point could equally be a failed run.

**What to do next.** First re-run 100M with a different seed — one bad run is not a curve
(lesson 2.15). Then, if the peak is real:

- **Scale past it**, which is the modern answer: 1B already wins, and larger will likely
  win more.
- **Or regularise at the threshold.** Weight decay and augmentation shift the peak and
  flatten it, because they change which interpolating solution is selected. This rescues
  the 100M model if you are constrained to that size.
- **Do not conclude that 10M is the right size.** Its 0.18 is the classical optimum, and
  the overparameterised regime beats it.

**The broader point:** sweeping model size and picking the minimum assumes a U-shaped
curve. With double descent, a coarse sweep can land you on the peak and lead you to
discard exactly the direction that works.
:::

## What to carry forward

- Test error is not U-shaped in model size; it peaks at the interpolation threshold and falls again.
- The explanation is which interpolating solution SGD selects, not how many exist.
- Gradient descent has an implicit bias toward minimum-norm and max-margin solutions.
- Grokking can hide generalisation behind thousands of flat steps — watch weight norm.
- A train–validation gap at large data scale may mean your model is too *small*.
