"""
CHAIR Evaluation — measures object hallucination in image captions.

CHAIR_s = fraction of captions that contain at least one hallucinated object.
CHAIR_i = fraction of all mentioned objects that are hallucinated (not in ground truth).

Usage:
    python chair_evaluate.py pope_results/chair_baseline/captions.jsonl
    python chair_evaluate.py pope_results/chair_baseline/captions.jsonl \\
        G:/claude code_workspace/GRPO-VLLM-hallucination-reduction/Qwen3vl/POPE-main/POPE-main/segmentation/coco_ground_truth_segmentation.json
"""
import json, os, sys, re, argparse

# ---------------------------------------------------------------------------
# COCO 80 objects sorted by length descending (multi-word first) so that
# "dining table" matches before "table", "traffic light" before "light", etc.
# ---------------------------------------------------------------------------
COCO_OBJECTS = [
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "potted plant", "dining table",
    "cell phone", "teddy bear", "hair drier",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "kite", "skateboard", "surfboard",
    "bottle", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "pizza", "donut", "cake", "chair", "couch", "bed",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "toothbrush",
]

# Build regex patterns: whole-word, case-insensitive
# Multi-word phrases: escape each word, join with \s+, wrap with \b
# Single words: simple \b word \b
WORD_PATTERNS = []
for obj in COCO_OBJECTS:
    words = obj.split()
    if len(words) > 1:
        pat = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
    else:
        pat = r'\b' + re.escape(obj) + r'\b'
    WORD_PATTERNS.append((obj, re.compile(pat, re.IGNORECASE)))


def extract_objects(text: str) -> set:
    """Extract COCO object mentions from caption text.

    Returns a set of matched object names.  Each multi-word phrase is
    matched before any single-word term that is a subset of it, because
    the list is sorted by length descending.
    """
    found = set()
    # Replace common punctuation so "table, chair." still matches
    clean = text.replace("'s", " ").replace("'", " ")
    for obj_name, pattern in WORD_PATTERNS:
        if pattern.search(clean):
            found.add(obj_name)
    return found


# ---------------------------------------------------------------------------
def main():
    SEG_FILE_DEFAULT = os.path.join(
        r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction",
        r"Qwen3vl\POPE-main\POPE-main\segmentation\coco_ground_truth_segmentation.json",
    )

    ap = argparse.ArgumentParser(description="CHAIR evaluation")
    ap.add_argument("captions_file", help="JSONL file from chair_baseline.py")
    ap.add_argument("seg_file", nargs="?", default=SEG_FILE_DEFAULT,
                    help="COCO segmentation JSONL (default: bundled)")
    args = ap.parse_args()

    # Load captions
    captions = [json.loads(l) for l in open(args.captions_file, encoding="utf-8")]
    print(f"Loaded {len(captions)} captions from {args.captions_file}")

    # Load segmentation data → index by image filename
    seg_data = [json.loads(l) for l in open(args.seg_file, encoding="utf-8")]
    gt_by_image = {e["image"]: set(e["objects"]) for e in seg_data}
    print(f"Loaded {len(gt_by_image)} ground-truth entries from {args.seg_file}")

    # Evaluate each caption
    total_captions = 0
    hallucinated_captions = 0
    total_mentions = 0
    hallucinated_mentions = 0

    for c in captions:
        img = c["image"]
        caption = c["caption"]
        gt_objects = gt_by_image.get(img)
        if gt_objects is None:
            print(f"  Warning: no GT for image {img}, skipping")
            continue

        found = extract_objects(caption)
        hallucinations = found - gt_objects

        total_captions += 1
        total_mentions += len(found)
        hallucinated_mentions += len(hallucinations)
        if len(hallucinations) > 0:
            hallucinated_captions += 1

    # -------------------------------------------------------------------
    chair_s = hallucinated_captions / total_captions * 100 if total_captions else 0
    chair_i = hallucinated_mentions / total_mentions * 100 if total_mentions else 0

    print(f"\n{'='*50}")
    print(f"CHAIR Evaluation Results")
    print(f"{'='*50}")
    print(f"  Total captions evaluated : {total_captions}")
    print(f"  Total object mentions     : {total_mentions}")
    print(f"  Hallucinated captions     : {hallucinated_captions}")
    print(f"  Hallucinated mentions     : {hallucinated_mentions}")
    print(f"  CHAIR_s (sentence-level)  : {chair_s:.2f}%")
    print(f"  CHAIR_i (instance-level)  : {chair_i:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
