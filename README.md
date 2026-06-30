# LLaVA Hallucination Reduction

**LLaVA-1.5-7B POPE hallucination reduction via attention routing strategies.**

---

## Overview

This project evaluates and improves LLaVA-1.5-7B on the POPE (Polling-based Object Probing Evaluation) benchmark by applying per-layer attention correction strategies (UAC, AdaIAT, VCD, etc.) to reduce object hallucination.

The companion project for **Qwen3-VL-2B** is at [GRPO-VLLM-hallucination-reduction](https://github.com/rainforest888/GRPO-VLLM-hallucination-reduction).

## Model Architecture

| Property | Value |
|----------|-------|
| Model | LLaVA-1.5-7B-hf |
| Language Model | Llama-2-7B (32 layers, hidden=4096, 32 MHA heads) |
| Vision Encoder | CLIP-ViT-L/14 (576 vision tokens) |
| Vision-Language Connector | MLP projection |
| Total Attention Modules | 32 LM layers |

## POPE Baseline Results (500 adversarial)

| Strategy | Accuracy | Precision | Recall | F1 | Yes% | Δ |
|----------|:--------:|:---------:|:------:|:----:|:----:|:--:|
| Baseline | 83.20% | 85.47% | 80.00% | 82.64% | 46.80% | — |
| UAC L15 α=0.77 | 83.20% | 86.40% | 78.80% | 82.43% | 45.60% | +0.00% |
| AdaIAT-U L15 α=1.0 | 83.20% | 86.40% | 78.80% | 82.43% | 45.60% | +0.00% |

**Key finding**: Per-layer attention correction strategies (UAC, AdaIAT-U) are ineffective on LLaVA-1.5-7B because its self-attention is extremely concentrated (Gini > 0.98). Unlike Qwen3-VL-2B (Gini ~0.96) where these strategies had a narrow +0.2% effect, LLaVA's attention pattern is near-one-hot — one position gets 60-90% of weight per head, making re-weighting unable to change the argmax.

## Setup

```bash
source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
cd G:/claude\ code_workspace/GRPO-VLLM-hallucination-reduction/llava_project
```

### External Dependencies

| Resource | Path |
|----------|------|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` (HuggingFace cache) |
| POPE data | `G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\POPE-main\POPE-main\output\coco\` |
| COCO val2014 | `G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\val2014\val2014\` |

## Project Structure

```
llava_project/
├── README.md
├── pope_baseline.py          # Baseline POPE inference
├── pope_evaluate.py           # Metrics: TP/FP/TN/FN/Acc/Prec/Rec/F1
├── pope_inference.py          # Alternative inference script
├── pope_results/
│   └── baseline/              # POPE baseline results (JSONL)
└── router/
    ├── router_module.py       # RouterManager + LayerRouter (core)
    ├── strategies.py          # UAC / AdaIAT / VHR strategy implementations
    ├── calibration.py         # Phase 0: calibration weights
    ├── dpo_data.py            # POPE data loader
    ├── dpo_train.py           # DPO training loop
    ├── grpo_train.py          # GRPO training v1
    ├── grpo_train_v2.py       # GRPO training v2 (random exploration)
    ├── grpo_v3.py             # GRPO v3 (counterfactual baseline)
    ├── pope_inference_router.py    # Router argmax inference
    ├── pope_inference_forced.py    # Forced single-strategy inference
    ├── adaiat_inference.py         # AdaIAT-V evaluation
    ├── adaiat_u_inference.py       # AdaIAT-U evaluation
    ├── uac_inference.py            # UAC evaluation
    ├── vcd_inference.py            # VCD evaluation
    ├── cai_bracs_inference.py      # CAI+BRACS inference
    ├── cai_bracs_v2.py             # CAI+BRACS v2
    ├── oracle_test.py              # Oracle strategy search
    ├── calibrate_vhr.py            # VHR calibration
    ├── smoke_vhr.py                # VHR smoke test
    ├── calc_steering.py            # CASAL-style steering
    ├── test_casal_lime.py          # CASAL & LIME evaluation
    ├── recalibrate_u.py            # AdaIAT-U recalibration
    ├── recalibrate_uac_real.py     # UAC recalibration on real images
    ├── sweep_vhr_alpha.py          # VHR alpha sweep
    ├── overnight.py                # Automated overnight pipeline
    ├── run_overnight_cai.py        # CAI overnight run
    ├── _verify_signal.py           # Signal verification
    ├── _pope_analysis.py           # Analysis tools
    ├── _analyze_calib.py           # Calibration analysis
    ├── _eval_all.py                # Batch evaluation
    ├── _rebuild_hard_set.py        # Hard set builder
    ├── _run_3strategy_oracle.py    # 3-strategy oracle
    ├── _run_champion.py            # Champion evaluation
    └── _run_stratification.py      # Stratification analysis
```

## Environment

| Item | Config |
|------|--------|
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB) |
| CUDA | PyTorch 2.12.0 |
| Python | conda env `qwen3vl` |

## License

Academic research use. See referenced papers for their respective licenses.
