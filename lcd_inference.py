"""
LLaVA LCD (Language-Contrastive Decoding) for POPE yes/no evaluation.

For each question, runs TWO forward passes:
  1. Normal: with image → get logits
  2. No-image: with a black image tensor (torch.zeros) → get logits
Final logits = logits_with_image - alpha * logits_without_image

Implements as a LogitsProcessor that pre-computes no-image logits and
subtracts them during generation. This steers the model away from its
language-only prior, amplifying the visual signal.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python lcd_inference.py
    python lcd_inference.py --subsets adversarial --max_n 10
    python lcd_inference.py --subsets adversarial,popular --alpha 1.0
"""
import json, os, sys, torch, argparse
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers.generation import LogitsProcessor
from PIL import Image
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction"
POPE_DIR = os.path.join(BASE, r"Qwen3vl\POPE-main\POPE-main\output\coco")
IMAGE_DIR = os.path.join(BASE, r"Qwen3vl\val2014\val2014")
OUT_DIR = os.path.join(os.path.dirname(__file__), "pope_results", "lcd")
os.makedirs(OUT_DIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--subsets", type=str, default="random,popular,adversarial")
ap.add_argument("--max_n", type=int, default=0, help="max questions per subset (0=all)")
ap.add_argument("--alpha", type=float, default=0.5,
                help="LCD subtraction strength: logits_final = logits_img - alpha * logits_noimg")
args = ap.parse_args()


def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t:
        t = t.split(".")[0]
    t = t.replace(",", "")
    w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"


class LCDLogitsProcessor(LogitsProcessor):
    """Pre-computes no-image logits and subtracts α · logits_noimg during generation.

    The no-image forward pass is computed once in __init__ with a black pixel_values
    tensor (same input_ids as the real-image pass).  During generate(), at each step
    the corresponding position's no-image logits are subtracted from the scores.

    final_logits = logits_with_image - α · logits_without_image
    """
    def __init__(self, model, noimg_inputs, alpha=0.5):
        with torch.no_grad():
            out_noimg = model(
                input_ids=noimg_inputs["input_ids"],
                attention_mask=noimg_inputs.get("attention_mask"),
                pixel_values=noimg_inputs.get("pixel_values"),
                use_cache=False,
            )
        # out_noimg.logits shape: (1, seq_len, vocab_size)
        self.noimg_logits = out_noimg.logits[0]  # (seq_len, vocab_size)
        self.alpha = alpha

    def __call__(self, input_ids, scores):
        # input_ids shape: (batch, current_length)
        # scores shape:    (batch, vocab_size) — logits for the NEXT token
        # The scores come from output.logits[:, -1, :] which is at position
        # input_ids.shape[1] - 1 in the full sequence logits.
        pos = input_ids.shape[1] - 1
        if pos < self.noimg_logits.shape[0]:
            noimg_scores = self.noimg_logits[pos].to(
                device=scores.device, dtype=scores.dtype
            )
            scores = scores - self.alpha * noimg_scores
        return scores


# ─── Load model ────────────────────────────────────────────────────────────────
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

# ─── Run POPE subsets ──────────────────────────────────────────────────────────
for subset in args.subsets.split(","):
    subset = subset.strip()
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
    out_file = os.path.join(OUT_DIR, f"coco_pope_{subset}_answers.json")

    questions = [json.loads(l) for l in open(pope_file, encoding="utf-8")]
    if args.max_n > 0:
        questions = questions[:args.max_n]

    results = []
    yes_count = 0
    for q in tqdm(questions, desc=f"POPE {subset}"):
        img = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")

        # Build prompt and tokenize with the real image
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": q["text"] + " Please answer yes or no."},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

        # Clone inputs and replace pixel_values with a black image
        noimg_inputs = {k: v for k, v in inputs.items()}
        if "pixel_values" in noimg_inputs:
            noimg_inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])

        # Create LCD logits processor with pre-computed no-image forward
        lcd_proc = LCDLogitsProcessor(model, noimg_inputs, alpha=args.alpha)

        # Generate with LCD bias (only the first token gets LCD subtraction;
        # subsequent tokens generate normally — sufficient for yes/no extraction)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=8,
                logits_processor=[lcd_proc],
            )
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

    labels = [json.loads(l)["label"] for l in open(pope_file, encoding="utf-8")][:len(results)]
    correct = sum(1 for r, l in zip(results, labels) if r["answer"] == l)
    acc = correct / len(results)
    print(f"  {subset}: {correct}/{len(results)} = {acc:.4f}  "
          f"yes_ratio={yes_count/len(results):.3f}")
    torch.cuda.empty_cache()

print(f"\nLCD alpha={args.alpha} complete. Results in {OUT_DIR}/")
print("Next: python pope_evaluate.py lcd")
