---
summary: Fine-tuning a fraction of the parameters, the low-rank hypothesis it rests on, and how to pick rank and target modules.
prereqs: [rank-span-subspaces, singular-value-decomposition, supervised-finetuning]
---

Full fine-tuning a 70B model needs roughly 1.1 TB of memory for parameters, gradients and
Adam state (lesson 2.01). LoRA trains under 1% of the parameters and fits on a single GPU,
with quality close to full fine-tuning.

## The hypothesis

Full fine-tuning updates $W \to W + \Delta W$. The observation behind LoRA is that
$\Delta W$ has much **lower rank** than $W$ — the adaptation lives in a small subspace even
though the weights do not.

So parameterise the update as a product of two thin matrices (lesson 1.04):

$$
W' = W + \Delta W = W + BA, \qquad B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k},\; r \ll \min(d,k)
$$

$W$ is frozen. Only $A$ and $B$ train.

## The implementation

```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                  # freeze the pretrained weight

        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout)

        # A gets a normal init; B starts at ZERO so BA = 0 and the model is
        # exactly the pretrained one at step 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Apply the factors in sequence: B(A x), never (BA) x.
        # Lesson 1.03's re-bracketing -- ~250x fewer FLOPs.
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + delta * self.scaling
```

::: key
**$B$ initialised to zero** is the detail that makes LoRA work. At step 0, $BA = 0$, so the
model is bit-identical to the pretrained one. Training starts from a known-good point
rather than from a perturbed model, which is why LoRA is stable at learning rates 10× higher
than full fine-tuning.

Initialising both at random would add noise to every layer before the first step —
equivalent to corrupting the pretrained weights.
:::

The $\alpha/r$ scaling means changing $r$ does not require retuning the learning rate:
doubling $r$ halves the scaling, keeping the update magnitude comparable.

## What it saves

For $d = k = 4096$ and $r = 8$:

| | Full | LoRA |
|---|---|---|
| Trainable per matrix | 16.8M | 65.5K (0.39%) |
| Gradient memory | 16.8M × 4 B | 65.5K × 4 B |
| Adam state | 16.8M × 8 B | 65.5K × 8 B |
| 7B model total trainable | 7B | ~4M |
| Optimizer memory (7B) | ~84 GB | ~48 MB |

The frozen base weights still need to be resident — 14 GB in bf16 — but they need no
gradient and no optimizer state. That is where the 1.1 TB becomes 20 GB.

::: check
`LoRALinear` initialises `lora_B` to zeros and `lora_A` with Kaiming uniform. Why not both randomly?

- [x] With $B = 0$ the product $BA$ is zero, so the model at step 0 is exactly the pretrained one
  > Starting from a random perturbation of a carefully pretrained model would throw away quality before training has learnt anything. Zeroing one factor makes the adapter a no-op at initialisation without freezing it.
- [ ] Zero initialisation halves the parameter count
  > Both matrices are trained and both are stored. Nothing is saved.
- [ ] It prevents the two factors from collapsing to the same subspace
  > Symmetry breaking is why $A$ is random rather than zero. If *both* were zero the gradients would be zero too and nothing would ever train.
- [ ] It makes the adapter mergeable into $W$ at deployment
  > Merging works from any values of $A$ and $B$.
:::

## Which modules

```python
def apply_lora(model, targets=('q_proj', 'k_proj', 'v_proj', 'o_proj'), r=8, alpha=16):
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'{trainable/1e6:.2f}M trainable / {total/1e9:.2f}B ({100*trainable/total:.3f}%)')
    return model
```

The original paper adapted only $W_Q$ and $W_V$. Later work found that adapting **all**
linear layers — attention projections and the MLP — works better at the same parameter
budget, because the MLP holds two-thirds of the parameters (lesson 4.05) and is where much
of the task-specific computation lives.

Rule of thumb: prefer **more modules at lower rank** over fewer modules at higher rank.

## Choosing rank

| $r$ | Use |
|---|---|
| 4–8 | Style and format adaptation; small datasets |
| 16–32 | General instruction tuning — the usual default |
| 64–128 | New domains, substantial capability change |
| 256+ | Rarely justified; consider full fine-tuning |

Larger $r$ helps when the task genuinely requires a high-rank update. If increasing $r$
stops helping, the bottleneck is data, not capacity.

## QLoRA

Quantize the frozen base to 4-bit and keep LoRA in bf16:

```python
# pip install bitsandbytes peft
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',              # normal float 4: fits a Gaussian's quantiles
    bnb_4bit_compute_dtype=torch.bfloat16,  # dequantize to bf16 for the matmul
    bnb_4bit_use_double_quant=True,         # quantize the quantization constants too
)
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b-hf',
                                             quantization_config=bnb)
```

This is what makes 70B fine-tuning possible on a single 48 GB GPU. The base weights become
35 GB; gradients flow *through* the quantized weights to the LoRA parameters without ever
updating the base. Quality loss is small — typically within a point of 16-bit LoRA.

## Merging for deployment

$$
W_{\text{merged}} = W + \frac{\alpha}{r}BA
$$

```python
@torch.no_grad()
def merge_lora(lora_layer):
    delta = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
    lora_layer.base.weight += delta
    return lora_layer.base
```

After merging, inference costs exactly what the base model costs — **zero LoRA overhead**.
This is the other half of lesson 1.03's re-bracketing argument: compute $B(Ax)$ during
training where the factors are cheap, and materialise $BA$ once at deployment where you pay
for it a single time.

Keeping adapters unmerged lets you serve many fine-tunes from one base model in memory,
swapping only the small adapter per request. That is how multi-tenant fine-tuning services
work.

## The family

- **LoRA** — the baseline described here.
- **DoRA** — decomposes the update into magnitude and direction, adapting each separately.
  Consistently a little better for a little more compute.
- **Prefix tuning** — prepend trainable vectors to the keys and values. Fewer parameters,
  generally worse than LoRA.
- **IA³** — learn per-channel rescaling vectors. Extremely few parameters, limited capacity.
- **BitFit** — train only biases. A useful baseline, rarely competitive.

::: exercise
LoRA with $r=8$ on attention projections gives noticeably worse results than full
fine-tuning on your task. Give three interventions in order.
:::

::: solution
**1. Target more modules before raising the rank.** Attention-only LoRA adapts one third of
the parameters (lesson 4.05); the MLP holds the other two thirds and is where much
task-specific transformation happens. Add `gate_proj`, `up_proj` and `down_proj`. At $r=8$
across all linear layers you roughly triple trainable parameters and typically recover most
of the gap — this is usually the whole fix.

**2. Raise the rank, and check whether it helps.** Try $r \in \{16, 32, 64\}$ with $\alpha =
2r$. The shape of the curve is the diagnostic: if quality improves through $r=64$, the
update genuinely needs high rank and LoRA's premise is weak for your task. If it plateaus
at $r=16$, capacity was never the constraint and the gap is elsewhere.

**3. Check the things that are not about capacity.** In practice these account for a large
share of reported LoRA underperformance:
- **Learning rate.** LoRA wants 1e-4 to 3e-4 — about 10× full fine-tuning's rate. Using the
  full-fine-tuning rate leaves it badly undertrained.
- **Epochs.** With fewer trainable parameters, LoRA often needs more passes.
- **Are the embeddings and output head frozen?** If your task introduces new vocabulary or
  a substantially different output distribution, those layers may need training too — LoRA
  on them, or unfreeze them outright.

**If all three fail**, the task requires a high-rank update and LoRA is the wrong tool.
That happens when adapting to a genuinely new domain — a new language, a new modality —
rather than a new task in a known domain. The options then are full fine-tuning, or
continued pretraining on domain data followed by LoRA for the task.

**One thing to rule out first:** confirm the comparison is fair. Full fine-tuning at 3
epochs against LoRA at 1 epoch, or different chat templates between the two runs, produces
exactly this symptom for reasons that have nothing to do with rank.
:::

## What to carry forward

- $\Delta W$ is empirically low-rank; LoRA parameterises it as $BA$ with $r \ll d$.
- Initialise $B$ to zero so the model starts exactly at the pretrained weights.
- Compute $B(Ax)$, never $(BA)x$ — 250× fewer FLOPs.
- Target all linear layers at low rank rather than a few at high rank.
- Merge for single-model deployment; keep adapters separate to serve many from one base.
