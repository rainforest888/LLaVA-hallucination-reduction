"""
LLaVA CLIP-Guided Decoding for POPE yes/no evaluation.

Strategy: at each generation step, compute cosine similarity between the
image features (mean-pooled CLIP ViT patch embeddings, projected through the
multimodal projector to 4096-d) and the frozen text embeddings of "yes"/"no".
Add the similarity as a logit bias, steering the model toward the answer whose
text embedding is closer to the image content.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python clip_guided_inference.py
    python clip_guided_inference.py --subsets adversarial
    python clip_guided_inference.py --subsets adversarial --lam 1.0 --max_n 100
"""
import json, os, sys, torch, argparse
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers.generation import LogitsProcessor
from PIL import Image
import torch.nn.functional as F

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction"
POPE_DIR = os.path.join(BASE, r"Qwen3vl\POPE-main\POPE-main\output\coco")
IMAGE_DIR = os.path.join(BASE, r"Qwen3vl\val2014\val2014")

ap = argparse.ArgumentParser()
ap.add_argument("--subsets", type=str, default="random,popular,adversarial")
ap.add_argument("--max_n", type=int, default=0, help="max questions per subset (0=all)")
ap.add_argument("--lam", type=float, default=0.5, help="CLIP guidance strength (logit bias scale)")
ap.add_argument("--outdir", type=str, default=None, help="override output directory name under pope_results/ (default: clip_guided)")
args = ap.parse_args()
OUT_DIR = os.path.join(os.path.dirname(__file__), "pope_results", args.outdir or "clip_guided")
os.makedirs(OUT_DIR, exist_ok=True)

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"


class CLIPGuidedLogitsProcessor(LogitsProcessor):
    """Adds cosine-similarity bias between image features and yes/no text embeddings.

    scores[yes] += lam * cos_sim(image_feat_pooled, embed["yes"])
    scores[no]  += lam * cos_sim(image_feat_pooled, embed["no"])
    """
    def __init__(self, token_yes, token_no, yes_emb, no_emb, captured, lam=0.5):
        self.token_yes = token_yes
        self.token_no = token_no
        self.yes_emb = yes_emb   # (D,) float32, pre-normalized
        self.no_emb = no_emb     # (D,) float32, pre-normalized
        self.captured = captured  # dict with key 'img_feat', populated by forward hook
        self.lam = lam

    def __call__(self, input_ids, scores):
        if 'img_feat' not in self.captured:
            return scores
        # Mean-pool over all 576 patch positions → (4096,)
        img_feat = self.captured['img_feat'].float().mean(dim=0)
        img_feat = F.normalize(img_feat, p=2, dim=-1)

        cos_yes = torch.dot(img_feat, self.yes_emb.to(img_feat.device)).item()
        cos_no  = torch.dot(img_feat, self.no_emb.to(img_feat.device)).item()

        if self.token_yes < scores.shape[-1]:
            scores[0, self.token_yes] += self.lam * cos_yes
        if self.token_no < scores.shape[-1]:
            scores[0, self.token_no]  += self.lam * cos_no
        return scores


# ─── Load model ────────────────────────────────────────────────────────────
print("Loading LLaVA-1.5-7B with 4-bit quantization...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB / "
      f"{torch.cuda.memory_reserved()/1e9:.1f} GB reserved")

# ─── Token setup ───────────────────────────────────────────────────────────
tok = processor.tokenizer
TOK_YES = tok.encode("yes", add_special_tokens=False)[0]  # 4874
TOK_NO  = tok.encode("no",  add_special_tokens=False)[0]  # 694
print(f"Token IDs: yes={TOK_YES}, no={TOK_NO}")

# Pre-compute and pre-normalize text embeddings for yes/no
emb_weight = model.get_input_embeddings().weight  # (32064, 4096)
YES_EMB = F.normalize(emb_weight[TOK_YES].detach().float().clone(), p=2, dim=-1)
NO_EMB  = F.normalize(emb_weight[TOK_NO].detach().float().clone(),  p=2, dim=-1)

# ─── Hook to capture image features during generation ──────────────────────
# During model.generate(), the first forward pass processes pixel_values
# through the vision tower and multi-modal projector.  We capture the
# projected image features (576, 4096) via a forward hook on model.model
# (the LlavaModel sub-module), whose output dataclass carries
# .image_hidden_states.
captured = {}

def _capture_image_features(module, input, output):
    if hasattr(output, 'image_hidden_states') and output.image_hidden_states is not None:
        captured['img_feat'] = output.image_hidden_states.detach()

hook_handle = model.model.register_forward_hook(_capture_image_features, with_kwargs=False)

# ─── Run POPE subsets ──────────────────────────────────────────────────────
for subset in args.subsets.split(","):
    subset = subset.strip()
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
    out_file = os.path.join(OUT_DIR, f"coco_pope_{subset}_answers.json")

    questions = [json.loads(l) for l in open(pope_file, encoding="utf-8")]
    if args.max_n > 0:
        questions = questions[:args.max_n]

    results = []
    yes_count = 0
    for q in tqdm(questions, desc=f"POPE {subset} (λ={args.lam})"):
        img = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": q["text"] + " Please answer yes or no."},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

        # Clear captured features from the previous sample
        captured.clear()
        logits_processor = CLIPGuidedLogitsProcessor(
            TOK_YES, TOK_NO, YES_EMB, NO_EMB, captured, lam=args.lam)

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8,
                                 logits_processor=[logits_processor])

        raw = processor.decode(
            gen[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        ans = answer_yes_no(raw)
        if ans == "yes":
            yes_count += 1
        results.append({"question": q["text"], "answer": ans, "raw_output": raw})

    # Write results (one JSON object per line)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(results)} to {out_file}")
    torch.cuda.empty_cache()

hook_handle.remove()
print(f"\nCLIP-Guided (λ={args.lam}) complete. Results in {OUT_DIR}/")
print("Next: python pope_evaluate.py clip_guided")
