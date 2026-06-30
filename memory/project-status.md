---
name: llava-project-status
description: LLaVA-1.5-7B hallucination reduction project — current state and next steps
metadata:
  type: project
---

## LLaVA Project Complete Status (2026-06-30)

**Repos:**
- LLaVA: https://github.com/rainforest888/LLaVA-hallucination-reduction
- Qwen3-VL: https://github.com/rainforest888/GRPO-VLLM-hallucination-reduction

**LLaVA-1.5-7B POPE Baseline (4-bit quantized, 8GB GPU):**
| Subset | N | Acc | Prec | Rec | F1 |
|--------|---|------|------|-----|-----|
| Random | 3000 | 87.93% | 97.10% | 78.20% | 86.63% |
| Popular | 3000 | 85.87% | 92.36% | 78.20% | 84.69% |
| Adversarial | 500 | 83.20% | 85.47% | 80.00% | 82.64% |

**Strategy results (500 adversarial):**
- UAC L15 α=0.77: 83.20% (Δ0.00)
- AdaIAT-U L15 α=1.0: 83.20% (Δ0.00, byte-identical to UAC)

**Why attention correction failed:** LLaVA uses Llama-2-7B whose attention is extremely concentrated (Gini > 0.98, one position gets 60-90% of weight per head). Per-layer re-weighting cannot change the argmax of a near-one-hot distribution. Contrast with Qwen3-VL-2B (Gini ~0.96) where strategies had a narrow window for +0.2% gain.

**Next: VCD (logit-level contrastive decoding) showed +4.0% on 50-sample test — worth scaling to 500.**

**Why:** Attention re-weighting fundamentally cannot work on LLaVA — need logit-level interventions instead.
**How to apply:** Read [[llava-project-status]] and SUMMARY.md, then focus on VCD or LoRA fine-tuning.
