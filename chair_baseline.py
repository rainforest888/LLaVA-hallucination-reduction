"""
CHAIR with attention strategies for LLaVA-1.5-7B.
LLaVA = CLIP-ViT + Llama-2-7B (32 layers, 32 MHA heads, hidden=4096).
Attention modules: model.language_model.model.layers[N].self_attn
"""
import json, os, sys, torch, argparse, random
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import torch.nn.functional as F
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
BASE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction"
IMAGE_DIR = os.path.join(BASE, r"Qwen3vl\val2014\val2014")
SEG_FILE = os.path.join(BASE, r"Qwen3vl\POPE-main\POPE-main\segmentation\coco_ground_truth_segmentation.json")

ap = argparse.ArgumentParser()
ap.add_argument("--n_images", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--strategy", type=str, default="none", choices=["none","uac","ada_iat_u","vcd","vhr","clip","lcd","beam","otp"])
ap.add_argument("--outdir", type=str, default=None)
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--alpha", type=float, default=0.77)
ap.add_argument("--gamma", type=float, default=1.0, help="VCD gamma")
args = ap.parse_args()

random.seed(args.seed)
outdir_name = args.outdir or f"chair_{args.strategy}"
OUT_DIR = os.path.join(os.path.dirname(__file__), "pope_results", outdir_name)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"CHAIR strategy={args.strategy} layer={args.layer} alpha={args.alpha} out={outdir_name}")

# Load seg data
seg_data = [json.loads(l) for l in open(SEG_FILE, encoding="utf-8")]
seen = set(); unique_seg = []
for e in seg_data:
    if e["image"] not in seen: seen.add(e["image"]); unique_seg.append(e)
available = [e for e in unique_seg if os.path.exists(os.path.join(IMAGE_DIR, e["image"]))]
sample = random.sample(available, min(args.n_images, len(available)))
print(f"Sampled {len(sample)} images")

# Load model
print("Loading LLaVA-1.5-7B 4-bit...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"),
    device_map="auto", local_files_only=True,
    attn_implementation="eager" if args.strategy in ("uac","ada_iat_u","vhr") else "sdpa")
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

if args.strategy in ("uac", "ada_iat_u", "vhr"):
    model.model.language_model.config.output_attentions = True
# =========== UAC Calibration ===========
if args.strategy == "uac":
    attn_mod = model.model.language_model.layers[args.layer].self_attn
    H = 32
    W_list = []
    
    def uac_calib_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            W_list.append(output[1][0, :, -1, :].detach().cpu())
    
    h = attn_mod.register_forward_hook(uac_calib_hook)
    calib_images = random.sample(available, min(20, len(available)))
    print(f"UAC calibration on {len(calib_images)} images...")
    for e in tqdm(calib_images):
        img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
        msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Describe this image."}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        with torch.no_grad(): model.generate(**inputs, max_new_tokens=2, pad_token_id=processor.tokenizer.eos_token_id)
        torch.cuda.empty_cache()
    h.remove()
    # Use first image's W (or average same-length ones)
    target_len = W_list[0].shape[1]
    same_len = [w for w in W_list if w.shape[1] == target_len]
    W_cal = torch.stack(same_len).mean(0) if same_len else W_list[0]
    W = W_cal.float()
    mean_W = W.mean() + 1e-8
    W_norm = mean_W / (W + 1e-8)
    print(f"UAC calibrated: W shape={W.shape}, range [{W_norm.min():.3f},{W_norm.max():.3f}]")

# =========== AdaIAT-U Calibration ===========
if args.strategy == "ada_iat_u":
    attn_mod = model.model.language_model.layers[args.layer].self_attn
    H = 32
    correct_attn = []; wrong_attn = []
    _last = [None]
    
    def adaiat_calib_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            _last[0] = output[1][0, :, -1, :10].mean(dim=-1).detach().cpu()
    
    h = attn_mod.register_forward_hook(adaiat_calib_hook)
    print("AdaIAT-U calibration...")
    cal_count = 0
    for e in available:
        if len(correct_attn) >= 10 and len(wrong_attn) >= 10: break
        img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
        msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Is there a person in this image? Answer yes or no."}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
            tok_id = out.logits[0,-1,:].argmax().item()
        ans = "yes" if tok_id == 4874 else "no"
        gt = "yes" if "person" in e.get("objects",[]) else "no"
        if _last[0] is not None:
            if ans == gt and len(correct_attn) < 10:
                correct_attn.append(_last[0])
            elif ans != gt and len(wrong_attn) < 10:
                wrong_attn.append(_last[0])
        cal_count += 1
        torch.cuda.empty_cache()
    h.remove()
    if correct_attn and wrong_attn:
        C = torch.stack(correct_attn).mean(0); Wr = torch.stack(wrong_attn).mean(0)
        M = (C+1e-8)/(Wr+1e-8); THR = Wr.mean() + 0.5*(C.mean()-Wr.mean())
        print(f"AdaIAT-U: M range [{M.min():.3f},{M.max():.3f}], thr={THR:.5f}")
    else:
        print("WARNING: calibration failed, disabling strategy")
        args.strategy = "none"

# =========== VHR Calibration ===========
if args.strategy == "vhr":
    attn_mod = model.model.language_model.layers[args.layer].self_attn
    H = 32
    var_list = []
    
    def vhr_calib_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            aw = output[1][0, :, :, :576].detach().cpu()  # (H, Lq, 576 visual tokens)
            var_list.append(aw.var(dim=-1).mean(dim=1))  # (H,) per-head variance
    
    h = attn_mod.register_forward_hook(vhr_calib_hook)
    calib_imgs = random.sample(available, min(20, len(available)))
    print(f"VHR calibration on {len(calib_imgs)} images...")
    for e in tqdm(calib_imgs):
        img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
        msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Describe this image."}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
        with torch.no_grad(): model.generate(**inputs, max_new_tokens=2, pad_token_id=processor.tokenizer.eos_token_id)
        torch.cuda.empty_cache()
    h.remove()
    mean_var = torch.stack(var_list).mean(0)  # (H,)
    VHR_weights = torch.softmax(-mean_var, dim=0) * H  # low-variance heads get higher weight
    print(f"VHR calibrated: head weights range [{VHR_weights.min():.3f},{VHR_weights.max():.3f}]")

# =========== Generate ===========
results = []
for e in tqdm(sample, desc=f"CHAIR {args.strategy}"):
    img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Please describe this image in detail."}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

    if args.strategy == "vcd":
        noimg_inputs = {k: v for k, v in inputs.items()}
        noimg_inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        with torch.no_grad():
            out_noimg = model(**noimg_inputs)
        noimg_logits = out_noimg.logits[0].clone()
        class VCDLP:
            def __init__(s, nl, g): s.nl, s.g = nl, g
            def __call__(s, ids, sc):
                p = ids.shape[1]-1
                if p < s.nl.shape[0]: sc = (1+s.g)*sc - s.g*s.nl[p].to(sc.device,sc.dtype)
                return sc
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, logits_processor=[VCDLP(noimg_logits, args.gamma)])
    
    elif args.strategy == "clip":
        # CLIP-Guided: bias logits using image-text embedding similarity
        # Capture image features from vision tower output
        img_feat = [None]
        def clip_hook(m, inp, out):
            if hasattr(out, 'image_hidden_states') and out.image_hidden_states is not None:
                img_feat[0] = out.image_hidden_states.detach().float().mean(dim=0)  # (576,4096)→(4096,)
        hh = model.model.register_forward_hook(clip_hook, with_kwargs=False)
        # Quick forward to capture image features
        with torch.no_grad(): model(**inputs)
        hh.remove()
        if img_feat[0] is not None:
            emb = model.get_input_embeddings().weight.float()  # (V,4096)
            img_feat_n = F.normalize(img_feat[0], p=2, dim=-1).unsqueeze(0)  # (1,4096)
            emb_n = F.normalize(emb, p=2, dim=-1)  # (V,4096)
            clip_bias = (img_feat_n @ emb_n.T).squeeze(0)  # (V,)
            class CLIPLP:
                def __init__(s, bias, lam): s.bias, s.lam = bias, lam
                def __call__(s, ids, sc): return sc + s.lam * s.bias.to(sc.device,sc.dtype)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=64, logits_processor=[CLIPLP(clip_bias, args.alpha)])
        else:
            with torch.no_grad(): gen = model.generate(**inputs, max_new_tokens=64)
    
    elif args.strategy == "lcd":
        # LCD: logits_img - alpha * logits_noimg (opposite sign from VCD)
        noimg_inputs = {k: v for k, v in inputs.items()}
        noimg_inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        with torch.no_grad(): out_noimg = model(**noimg_inputs)
        nl = out_noimg.logits[0].clone()
        class LCDLP:
            def __init__(s, nl, a): s.nl, s.a = nl, a
            def __call__(s, ids, sc):
                p = ids.shape[1]-1
                if p < s.nl.shape[0]: sc = sc - s.a * s.nl[p].to(sc.device,sc.dtype)
                return sc
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, logits_processor=[LCDLP(nl, args.alpha)])
    
    elif args.strategy == "beam":
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, num_beams=3, early_stopping=True)
    
    elif args.strategy == "otp":
        # OTP: penalize over-confident tokens (max logit >> mean logit)
        class OTPLP:
            def __init__(s, thr=3.0, penalty=2.0): s.thr, s.pen = thr, penalty
            def __call__(s, ids, sc):
                top = sc.max(dim=-1, keepdim=True)
                ratio = top.values / sc.mean(dim=-1, keepdim=True).clamp_min(1e-8)
                if ratio.item() > s.thr:
                    sc.scatter_(-1, top.indices, top.values - s.pen)
                return sc
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, logits_processor=[OTPLP()])
    
    elif args.strategy in ("uac", "ada_iat_u", "vhr"):
        attn_mod = model.model.language_model.layers[args.layer].self_attn
        orig_forward = attn_mod.forward
        
        if args.strategy == "uac":
            def uac_forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
                # Simple Llama attention forward with UAC on last row
                bsz, q_len, _ = hidden_states.size()
                q = attn_mod.q_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                k = attn_mod.k_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                v = attn_mod.v_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, attn_mod.layer_idx)
                aw = torch.matmul(q, k.transpose(-2,-1)) / (attn_mod.head_dim**0.5)
                if attention_mask is not None:
                    aw = aw + attention_mask[:,:,:,:k.shape[-2]]
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
                # UAC on last query row
                Lk_apply = min(W_norm.shape[1], aw.shape[-1])
                w = W_norm[:,:Lk_apply].to(device=aw.device, dtype=aw.dtype)
                log_w = torch.log(w.clamp_min(1e-6))
                corr = (1.0 + args.alpha * torch.tanh(log_w)).unsqueeze(0).unsqueeze(2)
                row = aw[:,:,-1:,:Lk_apply] * corr[:,:H,:,:]
                aw[:,:,-1:,:Lk_apply] = row / row.sum(dim=-1,keepdim=True).clamp_min(1e-8)
                out = torch.matmul(aw, v)
                out = out.transpose(1,2).contiguous().reshape(bsz,q_len,-1)
                return attn_mod.o_proj(out), aw
        
        elif args.strategy == "ada_iat_u":
            def adaiat_forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
                bsz, q_len, _ = hidden_states.size()
                q = attn_mod.q_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                k = attn_mod.k_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                v = attn_mod.v_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, attn_mod.layer_idx)
                aw = torch.matmul(q, k.transpose(-2,-1)) / (attn_mod.head_dim**0.5)
                if attention_mask is not None:
                    aw = aw + attention_mask[:,:,:,:k.shape[-2]]
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
                # AdaIAT-U on question tokens (first ~20 positions)
                q_tok_end = min(20, aw.shape[-1])
                atp = aw[:,:,-1,:q_tok_end].mean().item()
                if atp < THR:
                    amp = (1.0 + args.alpha * M.to(device=aw.device,dtype=aw.dtype)).view(1,H,1,1)
                    aw[:,:,-1:,:q_tok_end] = aw[:,:,-1:,:q_tok_end] * amp
                    row_sum = aw[:,:,-1:,:].sum(dim=-1,keepdim=True).clamp_min(1e-8)
                    aw[:,:,-1:,:] = aw[:,:,-1:,:] / row_sum
                out = torch.matmul(aw, v)
                out = out.transpose(1,2).contiguous().reshape(bsz,q_len,-1)
                return attn_mod.o_proj(out), aw
        
        elif args.strategy == "vhr":
            def vhr_forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
                bsz, q_len, _ = hidden_states.size()
                q = attn_mod.q_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                k = attn_mod.k_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                v = attn_mod.v_proj(hidden_states).view(bsz,q_len,H,-1).transpose(1,2)
                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, attn_mod.layer_idx)
                aw = torch.matmul(q, k.transpose(-2,-1)) / (attn_mod.head_dim**0.5)
                if attention_mask is not None:
                    aw = aw + attention_mask[:,:,:,:k.shape[-2]]
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
                # VHR: amplify high-variance (visually-sensitive) heads
                w = VHR_weights.to(device=aw.device, dtype=aw.dtype).view(1,H,1,1)
                row = aw[:,:,-1:,:] * (1.0 + args.alpha * (w - 1.0))
                aw[:,:,-1:,:] = row / row.sum(dim=-1,keepdim=True).clamp_min(1e-8)
                out = torch.matmul(aw, v)
                out = out.transpose(1,2).contiguous().reshape(bsz,q_len,-1)
                return attn_mod.o_proj(out), aw
        
        if args.strategy == "uac": attn_mod.forward = uac_forward
        elif args.strategy == "ada_iat_u": attn_mod.forward = adaiat_forward
        elif args.strategy == "vhr": attn_mod.forward = vhr_forward

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64)
    raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    results.append({"image_id": e["image_id"], "image": e["image"], "caption": raw.strip()})

    if args.strategy in ("uac", "ada_iat_u", "vhr"):
        attn_mod.forward = orig_forward
    torch.cuda.empty_cache()

out_file = os.path.join(OUT_DIR, "captions.jsonl")
with open(out_file, "w", encoding="utf-8") as f:
    for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved {len(results)} captions to {out_file}")
print(f"Next: python chair_evaluate.py {out_file}")
