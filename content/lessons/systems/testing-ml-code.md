---
summary: Tests that catch the bugs that do not raise exceptions — shape errors, silent leakage, and numerical drift.
prereqs: [debugging-models, custom-autograd-function, causal-masking]
---

Machine learning code fails differently from ordinary software. A bug rarely crashes; it
produces a model that trains to a slightly worse loss. Tests have to be designed for that.

## The shape of the problem

::: key
Ordinary code has a correct output you can assert against. ML code often does not — the model
is supposed to produce something you cannot compute independently.

So tests target **properties** that must hold regardless of the learned weights:

- Shapes and dtypes.
- Invariances and equivariances the architecture claims.
- Gradient existence and finiteness.
- Information flow — causality, masking.
- Agreement with a slow, obviously correct reference implementation.
:::

## Shape and dtype

```python
import pytest
import torch

@pytest.mark.parametrize('batch,seq', [(1, 1), (1, 128), (8, 512), (3, 17)])
def test_model_shapes(batch, seq):
    model = GPT(Config(vocab_size=1000, n_layer=2, n_embd=64, n_head=4))
    x = torch.randint(0, 1000, (batch, seq))
    logits, _ = model(x)
    assert logits.shape == (batch, seq, 1000)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
```

Include the awkward sizes: batch 1 (which `squeeze()` destroys, lesson 2.02), sequence 1, and
a non-power-of-two like 17. Those are where broadcasting and reshape bugs surface.

## Causality

The most valuable single test for a decoder, from lesson 4.06:

```python
import torch

def test_causal_mask_no_leakage():
    model = GPT(Config(vocab_size=100, n_layer=2, n_embd=32, n_head=4)).eval()
    x = torch.randint(0, 100, (1, 16))

    with torch.no_grad():
        base, _ = model(x)
        edited = x.clone()
        edited[0, 8] = (edited[0, 8] + 1) % 100
        after, _ = model(edited)

    # Editing position 8 must not change any output before it.
    assert torch.allclose(base[0, :8], after[0, :8], atol=1e-5), 'causal mask leaks'
    # And it must change something at or after it, or the model ignores input.
    assert not torch.allclose(base[0, 8:], after[0, 8:], atol=1e-5)
```

Both assertions matter. The first catches leakage; the second catches a model that is
ignoring its input entirely, which would pass the first trivially.

::: check
Why can't ML code be tested the way ordinary code is?

- [x] A bug rarely crashes
  > It produces a model that trains to a slightly worse loss, and there is usually no independently computable correct output to assert against. So tests target *properties* that must hold whatever the weights are: shapes and dtypes, claimed invariances, gradient finiteness, information flow, and agreement with a slow obviously-correct reference.
- [ ] Because training is non-deterministic, so no assertion can be stable
  > Non-determinism complicates numerical regression tests and does not stop shape, causality or gradient tests being exact.
- [ ] Because models are too large to run in CI
  > A two-layer model with a 1,000-token vocabulary tests the same code paths in milliseconds.
- [ ] Because the correct output depends on the training data
  > That is part of why there is no reference output, and the deeper issue is that the property, not the value, is what you can check.
:::

## Invariances

Test what the architecture claims:

```python
import torch

def test_attention_permutation_equivariance():
    """Without positional encoding, permuting inputs permutes outputs."""
    attn = MultiHeadAttention(d=64, n_heads=4)
    x = torch.randn(1, 8, 64)
    perm = torch.randperm(8)

    out_then_perm = attn(x, is_causal=False)[:, perm]
    perm_then_out = attn(x[:, perm], is_causal=False)
    assert torch.allclose(out_then_perm, perm_then_out, atol=1e-5)


def test_layernorm_scale_invariance():
    """LayerNorm output is unchanged by scaling its input."""
    ln = torch.nn.LayerNorm(64, elementwise_affine=False)
    x = torch.randn(4, 64)
    assert torch.allclose(ln(x), ln(x * 7.3), atol=1e-5)
```

An invariance test is strong: it holds for every weight setting, so it catches bugs that a
specific-value test would miss.

## Gradients

```python
import torch

def test_all_parameters_receive_gradient():
    model = GPT(Config(vocab_size=100, n_layer=2, n_embd=32, n_head=4))
    x = torch.randint(0, 100, (2, 16))
    _, loss = model(x[:, :-1], x[:, 1:])
    loss.backward()

    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)]
    assert not dead, f'no gradient reaches: {dead}'


def test_custom_op_gradient():
    """gradcheck in float64 -- float32 finite differences are too noisy."""
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: MyCustomOp.apply(t), (x,))
```

The first test catches unregistered parameters (lesson 2.08) and detached paths (lesson 2.06)
— two bugs that produce a model which trains, badly, with no error.

## Data pipeline

The pipeline is where the quietest bugs live:

```python
import torch

def test_targets_are_shifted():
    """The single most common from-scratch bug (lesson 4.06)."""
    dataset = TokenDataset(tokens=list(range(1000)), block=16)
    x, y = dataset[0]
    assert torch.equal(x[1:], y[:-1]), 'targets are not the inputs shifted by one'


def test_no_train_val_overlap():
    train_hashes = {hash(tuple(ex)) for ex in train_dataset}
    val_hashes = {hash(tuple(ex)) for ex in val_dataset}
    overlap = train_hashes & val_hashes
    assert not overlap, f'{len(overlap)} examples leak between splits'


def test_padding_is_masked_from_loss():
    batch = collate([make_example(5), make_example(12)])
    assert (batch['labels'] == -100).any(), 'padded targets must be ignored'
    assert batch['labels'][0, 5:].eq(-100).all()
```

## Reference implementations

For anything optimised, keep a slow obviously-correct version and test against it:

```python
import torch
import torch.nn.functional as F

def reference_attention(q, k, v, causal=True):
    """Deliberately naive. Readable, slow, obviously right."""
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if causal:
        T = q.size(-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float('-inf'))
    return F.softmax(scores, dim=-1) @ v


def test_optimised_attention_matches_reference():
    q, k, v = (torch.randn(2, 4, 32, 16, dtype=torch.float64) for _ in range(3))
    torch.testing.assert_close(my_attention(q, k, v), reference_attention(q, k, v))
```

Use float64 for the comparison so tolerance is not masking a real discrepancy.

::: check
Which test would catch a causal mask that accidentally lets position $i$ see position $i+1$?

- [x] Perturb a token at position $j$ and assert that outputs at every position $i < j$ are unchanged
  > It is an exact, weight-independent property — either information flowed backwards or it did not. A loss curve would never tell you, because a leaky mask makes training *easier*.
- [ ] Asserting the loss decreases over 100 steps
  > A model with a leaky mask trains beautifully. That is what makes the bug dangerous.
- [ ] Comparing the output shape against the expected shape
  > Shape is unaffected by which positions attend to which.
- [ ] Checking that all logits are finite
  > Finiteness would catch an all-$-\infty$ row, which is the opposite failure.
:::

## The integration test

```python
def test_overfits_single_batch():
    """Lesson 3.16's check, as a test. Catches structural bugs nothing else does."""
    model = GPT(Config(vocab_size=50, n_layer=2, n_embd=32, n_head=4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randint(0, 50, (4, 16))

    for _ in range(200):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    assert loss.item() < 0.1, f'cannot overfit one batch: loss {loss.item():.3f}'
```

::: key
**This is the highest-value test in the suite.** It takes seconds on a tiny model and it
catches every structural bug — detached paths, unregistered parameters, wrong targets, broken
masks — in one assertion.

If you write only one test, write this one.
:::

## Numerical regression

Catch unintended changes in behaviour:

```python
import torch

def test_forward_matches_golden():
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=100, n_layer=2, n_embd=32, n_head=4))
    x = torch.arange(16)[None]
    out, _ = model(x)

    golden = torch.load('tests/golden/forward.pt')
    torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)
```

::: warning
Golden-value tests are brittle in exactly the way lesson 2.15 describes: a PyTorch upgrade
changes kernel selection, which changes summation order, which changes the last bits.

Use loose tolerances, regenerate deliberately when the change is understood, and never treat
a golden-test failure as proof of a bug without checking whether the environment changed.
:::

## Cheap and expensive tests

```python
@pytest.mark.slow
def test_training_reduces_loss_on_real_data():
    ...

# pytest -m "not slow"     on every commit
# pytest                   nightly
```

Keep the fast suite under about 30 seconds so it runs on every commit. Shape, causality,
gradient and pipeline tests all belong there. Anything needing a GPU or real data goes in the
slow suite.

::: exercise
Your model trains and reaches a plausible loss, but generated text is nonsense. Which tests
would have caught it, and which would not?
:::

::: solution
**Plausible loss with nonsense output points to a train/inference mismatch** — the training
path is computing something sensible and the generation path is not doing the same thing.

**Tests that would have caught it:**

1. **Cached versus uncached generation** (lesson 4.09). If the KV cache mishandles the RoPE
   position offset or the causal flag, training is unaffected — it never uses the cache — and
   generation is subtly wrong:

   ```python
   def test_cache_matches_no_cache():
       torch.manual_seed(0); a = generate(model, prompt, 20, use_cache=True)
       torch.manual_seed(0); b = generate(model, prompt, 20, use_cache=False)
       assert torch.equal(a, b)
   ```
   This is the most likely catch.

2. **A round-trip tokenizer test.** If the tokenizer used for generation differs from the one
   used for training, loss is fine and output is nonsense:
   ```python
   def test_tokenizer_roundtrip():
       for s in ['hello world', 'def f(x):', '日本語']:
           assert tokenizer.decode(tokenizer.encode(s)) == s
   ```

3. **A chat-template consistency test** (lesson 5.05), if this is an instruction model. The
   template at inference must match training character for character.

4. **A generation smoke test** asserting the output is not degenerate — no token repeated
   more than $n$ times consecutively, and at least $k$ distinct tokens.

**Tests that would *not* have caught it:**

- **Shape tests.** Shapes are correct in both paths.
- **The overfit-one-batch test.** It exercises only the training path, which is working.
- **Gradient tests.** Gradients are fine.
- **Causality tests**, if they run the uncached path only — which is the usual way they are
  written.

**The general lesson:** a test suite that only exercises the training path cannot catch
inference bugs, and the training and inference paths of a transformer differ substantially
(caching, no teacher forcing, sampling). **Test the path you deploy**, not just the one you
train — and the cached-versus-uncached comparison is the single test that covers most of that
gap.
:::

## What to carry forward

- Test properties — shapes, invariances, causality, gradient flow — not learned values.
- The overfit-one-batch test catches every structural bug in one assertion.
- Test the data pipeline: target shift, split overlap, padding masking.
- Keep a slow reference implementation and compare in float64.
- Test the inference path separately; cached generation must match uncached.
