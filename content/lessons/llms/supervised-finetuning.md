---
summary: Turning a text continuer into an assistant, why loss masking matters, and the data quality finding that surprised everyone.
prereqs: [pretraining-objective, regularization, training-loop]
---

A pretrained model completes text. Asked "What is the capital of France?" it may well
answer — or produce three more quiz questions, because that is what follows a question in
many documents. Supervised fine-tuning teaches the format of being an assistant.

## The objective is unchanged

Same next-token loss, different data and one crucial masking change:

```python
import torch
import torch.nn.functional as F

def sft_loss(model, input_ids, labels):
    """labels == -100 wherever the loss should be ignored."""
    logits = model(input_ids[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100)
```

::: key
**Mask the prompt.** Only the assistant's response should contribute to the loss:

```python
labels = input_ids.clone()
labels[:, :prompt_length] = -100       # the model is not learning to write prompts
```

Training on prompt tokens teaches the model to *generate* user messages, which shows up as
a model that continues the conversation by inventing the user's next turn. It also dilutes
the gradient with tokens you do not care about — with a 500-token prompt and a 50-token
answer, 90% of the signal is wasted.
:::

## Chat templates

A conversation must be serialised into one token sequence with role markers:

```python
CHAT_TEMPLATE = """<|system|>
{system}<|end|>
<|user|>
{user}<|end|>
<|assistant|>
{assistant}<|end|>"""
```

Three requirements that are easy to get wrong:

- **Special tokens must be in the tokenizer** and their embeddings trained. Adding tokens
  after pretraining means rows initialised at random, which behave badly until trained
  (lesson 3.13).
- **The template must match at inference exactly.** A mismatched template — a missing
  newline, a different marker — produces a model that behaves as though it were never
  fine-tuned. This is the most common deployment bug in the whole pipeline.
- **`<|end|>` must be in the labels.** If the end-of-turn token is masked out, the model
  never learns to stop and will generate until it hits the token limit.

```python
def build_example(tokenizer, system, user, assistant):
    prompt = f'<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n'
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(assistant + '<|end|>')      # include the stop token
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return torch.tensor(ids), torch.tensor(labels)
```

## Quality beats quantity, decisively

The LIMA result is the one to know: **1,000 carefully curated examples** produced a model
competitive with one fine-tuned on 52,000 automatically generated ones.

The explanation that has held up: pretraining already contains the knowledge and the
abilities. Fine-tuning teaches *format and style* — which register to use, how long to
answer, when to refuse, how to structure a response. That is a small amount of information,
and a thousand good demonstrations convey it better than fifty thousand mediocre ones.

::: warning
Low-quality SFT data actively damages a model. Two mechanisms:

**Style over substance.** If responses are consistently confident regardless of correctness,
the model learns confidence as a style — including when it is wrong. Hallucination rates
rise measurably.

**Teaching it to guess.** If the data never contains "I don't know", the model learns that
an answer is always expected, so it fabricates. Include examples of appropriate refusal and
uncertainty, or you are training the failure in.
:::

## Hyperparameters

SFT is a small-data regime, and the settings differ sharply from pretraining:

| | Pretraining | SFT |
|---|---|---|
| Learning rate | 3e-4 | **1e-5 to 2e-5** |
| Epochs | ~1 | **2–3** |
| Weight decay | 0.1 | 0.01–0.1 |
| Dropout | 0 | **0.05–0.1** |
| Warmup | 2,000 steps | 3–10% of total |

The learning rate is 20–30× lower. At pretraining rates the model moves far enough to
destroy what it learned — **catastrophic forgetting**, visible as a model that follows
instructions well and has lost general knowledge.

Beyond about 3 epochs, overfitting is severe: training loss keeps falling while the model
memorises exact responses and generalises worse.

## Packing, carefully

Concatenating short examples to fill the context wastes no compute, but it needs the
block-diagonal mask of lesson 4.07:

```python
def pack_examples(examples, max_len, pad_id):
    packed_ids, packed_labels, doc_ids = [], [], []
    doc = 0
    for ids, labels in examples:
        if len(packed_ids) + len(ids) > max_len:
            break
        packed_ids.extend(ids); packed_labels.extend(labels)
        doc_ids.extend([doc] * len(ids)); doc += 1
    pad = max_len - len(packed_ids)
    return (packed_ids + [pad_id] * pad,
            packed_labels + [-100] * pad,
            doc_ids + [-1] * pad)          # -1 attends to nothing
```

Without the mask, example 2 attends to example 1 and learns spurious continuations across
unrelated conversations.

## Where SFT stops

SFT teaches the model to imitate demonstrations. It cannot teach it to be *better* than
them, because the objective is to match the reference response exactly.

Three consequences:

- Quality is capped by the demonstrators. A model cannot learn to write better than its
  best example.
- There is no signal about **relative** quality — every demonstration is equally correct,
  so the model cannot learn that one response is preferable to another.
- Subtle properties are hard to demonstrate. "Be helpful but do not overclaim" is easier to
  *judge* than to *write* consistently across a thousand examples.

That is the gap preference learning fills, and it is why lessons 5.07 through 5.09 exist.

::: exercise
Your SFT model gives good answers but never stops — it answers, then invents a follow-up
question and answers that too. What went wrong?
:::

::: solution
**The end-of-turn token is not being learned.** The model has no signal that a response
ends, so at inference it continues into whatever the template's next section looks like.

**Three specific causes, in order of likelihood:**

1. **`<|end|>` is masked out of the labels.** If you set `labels[prompt_len:] = ids[prompt_len:]`
   but the stop token was appended *after* that slice, or if you truncated labels before it,
   the model never receives gradient on the one token that matters most. Check directly:

   ```python
   ids, labels = build_example(...)
   print('last 5 label ids:', labels[-5:].tolist())
   print('eos id:', tokenizer.eos_token_id)
   ```
   The final label must be the stop token, not `-100`.

2. **Generation is not stopping on it.** The model may be emitting the token correctly while
   the generation loop ignores it. Check `eos_token_id` is passed to `generate`, and that
   it matches the token the template actually uses — a custom `<|end|>` is often a different
   id from the tokenizer's default `</s>`.

3. **Template mismatch at inference.** If training used `<|assistant|>\n` and inference uses
   `<|assistant|>` without the newline, the model is in a slightly different distribution
   and its stopping behaviour is unreliable. Print the exact prompt string sent at inference
   and diff it against a training example character by character.

**How to confirm which:** run a forward pass on one training example and look at the
probability assigned to the stop token at the final position.

```python
with torch.no_grad():
    logits = model(ids[None, :-1])
probs = logits[0, -1].softmax(-1)
print('P(eos) at end of response:', probs[tokenizer.eos_token_id].item())
```

Near 1.0 means the model learned it and the bug is in generation (cause 2 or 3). Near 0
means it never learned it (cause 1).

**Also check for truncation.** If `max_length` cut examples mid-response during training,
many examples ended without a stop token at all, and the model learned that responses often
just continue.
:::

## What to carry forward

- Same loss as pretraining, with the prompt masked to `-100`.
- The chat template must match between training and inference, exactly.
- Include the stop token in the labels, or the model never learns to finish.
- 1,000 good examples beat 50,000 mediocre ones; bad data teaches confident hallucination.
- Learning rate 20–30× lower than pretraining; 2–3 epochs at most.
