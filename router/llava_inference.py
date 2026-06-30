"""
LLaVA single-strategy inference for POPE adversarial (500 questions).

Runs UAC, AdaIAT-U on LLaVA-1.5-7B with 4-bit quantization.
Hooks into internal attention computation with a meticulous forward
replacement that mirrors LlamaAttention.forward exactly, plus the
strategy correction.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python router/llava_inference.py --strategy uac --layer 15
    python router/llava_inference.py --strategy adaiat_u --layer 15 --alpha 1.0
    python router/llava_inference.py --strategy none
"""
import json, os, sys, argparse, torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, repeat_kv, eager_attention_forward,
)

# ─── Paths ────────────────────────────────────────────────────────────
BASE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction"
POPE_DIR = os.path.join(BASE, r"Qwen3vl\POPE-main\POPE-main\output\coco")
IMAGE_DIR = os.path.join(BASE, r"Qwen3vl\val2014\val2014")
RESULTS_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pope_results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Args ─────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--strategy", type=str, default="none",
                choices=["none", "uac", "adaiat_u"])
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--alpha", type=float, default=0.77)
ap.add_argument("--outdir", type=str, default=None)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--ncalib", type=int, default=30)
args = ap.parse_args()

STRATEGY = args.strategy
LAYER = args.layer
ALPHA = args.alpha
N_SAMPLES = args.n

OUT_NAME = args.outdir or ({"none": "baseline", "uac": f"uac_L{LAYER}_a{ALPHA}",
                             "adaiat_u": f"adaiat_u_L{LAYER}_a{ALPHA}"}.get(STRATEGY, STRATEGY))
OUTPUT_DIR = os.path.join(RESULTS_BASE, OUT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"LLaVA {STRATEGY} ({OUT_NAME}) → {OUTPUT_DIR}")
EPS = 1e-8

# ─── Load model (4-bit) ──────────────────────────────────────────────
print("Loading LLaVA-1.5-7B (4-bit)...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
)
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=quant_config, device_map="auto",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", local_files_only=True)
model.eval()
for p in model.parameters():
    p.requires_grad = False

lm = model.model.language_model
N_LAYERS = len(lm.layers)
H_HEADS = lm.config.num_attention_heads
HEAD_DIM = lm.config.head_dim
KV_HEADS = lm.config.num_key_value_heads
HIDDEN = lm.config.hidden_size
N_VISION = (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2  # 576
print(f"LLaVA: {N_LAYERS} LM layers, {H_HEADS} heads, {HEAD_DIM} head_dim, {KV_HEADS} KV heads, {N_VISION} vis tokens")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ─── Answer parser ────────────────────────────────────────────────────
def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"


# ════════════════════════════════════════════════════════════════════════
# Unified hook factory
# ════════════════════════════════════════════════════════════════════════
def make_hook_forward(attn_mod, strategy_info: dict):
    """
    Return a new `forward` for attn_mod that mirrors LlamaAttention.forward
    exactly, then applies a post-softmax correction to attn_weights.

    `strategy_info` is a mutable dict with keys:
        - "strategy": "none" | "uac" | "adaiat_u"
        - "W": tensor or None
        - "M": tensor or None
        - "threshold": float
        - "alpha": float
        - "prefill_done": bool flag for each generation
        - "collect_queue": list (for calibration capture)
    """
    orig_forward = attn_mod.forward

    def hooked_forward(hidden_states, position_embeddings=None, attention_mask=None,
                       past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, HEAD_DIM)

        query_states = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, attn_mod.layer_idx)

        # ── Compute attention weights (mirrors eager_attention_forward) ──
        key_states_expanded = repeat_kv(key_states, attn_mod.num_key_value_groups)
        value_states_expanded = repeat_kv(value_states, attn_mod.num_key_value_groups)

        scaling = attn_mod.scaling if hasattr(attn_mod, 'scaling') else HEAD_DIM ** -0.5
        attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states_expanded.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # ── Strategy correction (post-softmax) ──
        si = strategy_info
        is_prefill = (past_key_values is None or past_key_values.get_seq_length() == 0)

        if is_prefill and si.get("strategy") == "uac" and si.get("W") is not None:
            W = si["W"].to(device=attn_weights.device, dtype=attn_weights.dtype)
            Hw, Lw = W.shape
            if Hw != H_HEADS:
                W = W[:H_HEADS, :] if Hw >= H_HEADS else W.expand(H_HEADS, -1)
            L_apply = min(Lw, attn_weights.shape[-1])
            last_row = attn_weights[:, :, -1:, :L_apply]
            log_w = torch.log(W[:, :L_apply].clamp_min(1e-6))
            corr = 1.0 + si["alpha"] * torch.tanh(log_w)
            corr = corr.unsqueeze(0).unsqueeze(2)
            last_row = last_row * corr
            attn_weights[:, :, -1:, :L_apply] = last_row / last_row.sum(dim=-1, keepdim=True).clamp_min(EPS)

        if is_prefill and si.get("strategy") == "adaiat_u" and si.get("M") is not None:
            n_vis = min(N_VISION, attn_weights.shape[-1])
            a_vis = attn_weights[:, :, -1, :n_vis]
            atp_current = a_vis.mean()
            if atp_current < si["threshold"]:
                M = si["M"].to(device=attn_weights.device, dtype=attn_weights.dtype)
                amp = 1.0 + si["alpha"] * M
                attn_weights[:, :, -1:, :n_vis] *= amp.view(1, H_HEADS, 1, 1)
                row_sum = attn_weights[:, :, -1:, :].sum(dim=-1, keepdim=True).clamp_min(EPS)
                attn_weights[:, :, -1:, :] = attn_weights[:, :, -1:, :] / row_sum

        # ── Capture attention (for calibration) ──
        if si.get("collect_queue") is not None and is_prefill:
            si["collect_queue"].append(attn_weights[0, :, -1, :min(N_VISION, attn_weights.shape[-1])].detach().cpu())

        # ── Output ──
        attn_output = torch.matmul(attn_weights, value_states_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_mod.o_proj(attn_output)
        return attn_output, attn_weights

    return hooked_forward


# ════════════════════════════════════════════════════════════════════════
# STRATEGY: NONE
# ════════════════════════════════════════════════════════════════════════
def strategy_none(subset="adversarial"):
    qs = [json.loads(l) for l in open(os.path.join(POPE_DIR, f"coco_pope_{subset}.json"), encoding="utf-8")]
    if N_SAMPLES > 0: qs = qs[:N_SAMPLES]
    out_file = os.path.join(OUTPUT_DIR, f"coco_pope_{subset}_answers.json")
    results = []
    for q in tqdm(qs, desc=f"POPE {subset}"):
        img = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q["text"] + " Please answer yes or no."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8)
        raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append({"question": q["text"], "answer": answer_yes_no(raw), "raw_output": raw})
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results


# ════════════════════════════════════════════════════════════════════════
# STRATEGY: UAC
# ════════════════════════════════════════════════════════════════════════
def calibrate_uac(n_calib=50):
    print(f"UAC calibration: {n_calib} blank images...")
    blank = Image.new("RGB", (336, 336), color=(0, 0, 0))
    attn_mod = lm.layers[LAYER].self_attn
    si = {"strategy": "none", "W": None, "M": None, "threshold": 0, "alpha": ALPHA,
          "collect_queue": []}
    attn_mod.forward = make_hook_forward(attn_mod, si)

    for _ in tqdm(range(n_calib), desc="UAC calibration"):
        si["collect_queue"].clear()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=blank, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=4)
        if si["collect_queue"]:
            # Last entry is prefill attention: (H, L_vis)
            # Save all prefill attention maps
            pass
        torch.cuda.empty_cache()

    attn_mod.forward = attn_mod.forward  # restore later; let's just re-load

    # Actually, let's capture differently
    # Re-do with a cleaner approach
    return _calibrate_uac_v2(n_calib)


def _calibrate_uac_v2(n_calib=50):
    """Simpler UAC calibration: capture attention via forward hook."""
    from transformers.models.llama.modeling_llama import LlamaAttention

    attn_mod = lm.layers[LAYER].self_attn
    captured_weights = []

    def collect_hook(module, args, kwargs, output):
        # output is (attn_output, attn_weights)
        aw = output[1]  # (1, H, Lq, Lk)
        captured_weights.append(aw.detach().cpu())

    handle = attn_mod.register_forward_hook(collect_hook, with_kwargs=True)

    blank = Image.new("RGB", (336, 336), color=(0, 0, 0))
    for _ in tqdm(range(n_calib), desc="UAC calibration"):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=blank, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=4)
        torch.cuda.empty_cache()

    handle.remove()

    # Use only prefill attention maps (Lq > 10 means prefill, not single-token decode)
    prefill_maps = [aw[0] for aw in captured_weights if aw.shape[2] > 10]  # each (H, Lq, Lk)
    if not prefill_maps:
        print("ERROR: No prefill attention maps captured")
        return None
    stacked = torch.stack([m.mean(dim=1) for m in prefill_maps], dim=0)  # (N, H, Lk)
    mean_all = stacked.mean(dim=0)  # (H, Lk)
    eps_val = 1e-8
    head_means = mean_all.mean(dim=1, keepdim=True)  # (H, 1)
    W = (head_means + eps_val) / (mean_all + eps_val)  # (H, Lk)
    print(f"UAC W shape: {W.shape}, range: [{W.min():.4f}, {W.max():.4f}]")
    return W


def strategy_uac(subset="adversarial", n_calib=50):
    W_uac = _calibrate_uac_v2(n_calib)
    if W_uac is None:
        return strategy_none(subset)

    attn_mod = lm.layers[LAYER].self_attn
    si = {"strategy": "uac", "W": W_uac, "M": None, "threshold": 0, "alpha": ALPHA,
          "collect_queue": None}
    attn_mod.forward = make_hook_forward(attn_mod, si)
    print(f"UAC hook installed at layer {LAYER}, alpha={ALPHA}")

    qs = [json.loads(l) for l in open(os.path.join(POPE_DIR, f"coco_pope_{subset}.json"), encoding="utf-8")]
    if N_SAMPLES > 0: qs = qs[:N_SAMPLES]
    out_file = os.path.join(OUTPUT_DIR, f"coco_pope_{subset}_answers.json")
    results = []
    for q in tqdm(qs, desc=f"POPE {subset} (UAC L{LAYER})"):
        img = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q["text"] + " Please answer yes or no."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8)
        raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append({"question": q["text"], "answer": answer_yes_no(raw), "raw_output": raw})

    attn_mod.forward = attn_mod.__class__.forward.__get__(attn_mod, attn_mod.__class__)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results


# ════════════════════════════════════════════════════════════════════════
# STRATEGY: AdaIAT-U
# ════════════════════════════════════════════════════════════════════════
def _calibrate_ada_iat_u(n_calib=30):
    print(f"AdaIAT-U calibration: {n_calib} correct + {n_calib} wrong...")
    baseline_dir = os.path.join(RESULTS_BASE, "baseline")
    correct_samples, wrong_samples = [], []
    for s in ["random", "popular", "adversarial"]:
        a = [json.loads(l) for l in open(f"{baseline_dir}/coco_pope_{s}_answers.json", encoding="utf-8")]
        b = [json.loads(l) for l in open(f"{POPE_DIR}/coco_pope_{s}.json", encoding="utf-8")]
        for ai, bi in zip(a, b):
            (correct_samples if ai["answer"] == bi["label"] else wrong_samples).append((bi["image"], bi["text"]))
    print(f"  Pool: {len(correct_samples)} correct, {len(wrong_samples)} wrong")
    correct_samples = correct_samples[:n_calib]
    wrong_samples = wrong_samples[:n_calib]

    attn_mod = lm.layers[LAYER].self_attn
    collect_queue = []
    si = {"strategy": "none", "W": None, "M": None, "threshold": 0, "alpha": ALPHA,
          "collect_queue": collect_queue}
    attn_mod.forward = make_hook_forward(attn_mod, si)

    def collect_atp(samples, label):
        ret = []
        for img_name, q_text in tqdm(samples, desc=f"  {label}"):
            collect_queue.clear()
            img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q_text + " Please answer yes or no."}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=4)
            if collect_queue:
                ret.append(collect_queue[-1])  # (H, L_vis)
            torch.cuda.empty_cache()
        return ret

    atp_correct = collect_atp(correct_samples, "Correct")
    atp_wrong = collect_atp(wrong_samples, "Wrong")
    attn_mod.forward = attn_mod.__class__.forward.__get__(attn_mod, attn_mod.__class__)

    if not atp_correct or not atp_wrong:
        print("ERROR: Calibration failed")
        return None, 0.0

    sc = torch.stack(atp_correct, 0)
    sw = torch.stack(atp_wrong, 0)
    mean_c = sc.mean(dim=-1).mean(dim=0)  # (H,)
    mean_w = sw.mean(dim=-1).mean(dim=0)  # (H,)
    M = mean_c / (mean_w + 1e-8)
    per_sample_atp_w = sw.mean(dim=-1).mean(dim=1)
    threshold = (per_sample_atp_w.mean() + 0.5 * per_sample_atp_w.std()).item()

    print(f"  M mean={M.mean():.4f}, M>1 heads={int((M>1).sum().item())}/{H_HEADS}")
    print(f"  Threshold: {threshold:.6f}")
    return M, threshold


def strategy_ada_iat_u(subset="adversarial", n_calib=30):
    M_ada, threshold = _calibrate_ada_iat_u(n_calib)
    if M_ada is None:
        return strategy_none(subset)

    attn_mod = lm.layers[LAYER].self_attn
    si = {"strategy": "adaiat_u", "W": None, "M": M_ada, "threshold": threshold,
          "alpha": ALPHA, "collect_queue": None}
    attn_mod.forward = make_hook_forward(attn_mod, si)
    print(f"AdaIAT-U hook installed at layer {LAYER}, alpha={ALPHA}")

    qs = [json.loads(l) for l in open(os.path.join(POPE_DIR, f"coco_pope_{subset}.json"), encoding="utf-8")]
    if N_SAMPLES > 0: qs = qs[:N_SAMPLES]
    out_file = os.path.join(OUTPUT_DIR, f"coco_pope_{subset}_answers.json")
    results = []
    for q in tqdm(qs, desc=f"POPE {subset} (AdaIAT-U L{LAYER})"):
        img = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q["text"] + " Please answer yes or no."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8)
        raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append({"question": q["text"], "answer": answer_yes_no(raw), "raw_output": raw})

    attn_mod.forward = attn_mod.__class__.forward.__get__(attn_mod, attn_mod.__class__)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Strategy: {STRATEGY}, Layer: {LAYER}, Alpha: {ALPHA}, N: {N_SAMPLES}")

    if STRATEGY == "none":
        results = strategy_none("adversarial")
    elif STRATEGY == "uac":
        results = strategy_uac("adversarial")
    elif STRATEGY == "adaiat_u":
        results = strategy_ada_iat_u("adversarial")

    labels = [json.loads(l) for l in open(os.path.join(POPE_DIR, "coco_pope_adversarial.json"), encoding="utf-8")][:len(results)]
    correct = sum(1 for r, l in zip(results, labels) if r["answer"] == l["label"])
    yes_count = sum(1 for r in results if r["answer"] == "yes")
    print(f"\n  Adversarial: {correct}/{len(results)} = {correct/len(results):.4f}, yes_ratio={yes_count/len(results):.3f}")
    print(f"  Results saved to {OUTPUT_DIR}/")
    print("Done.")
