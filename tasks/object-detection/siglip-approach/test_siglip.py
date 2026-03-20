"""
SigLIP zero-shot classification test for NM i AI grocery product detection.

Tests google/siglip-base-patch16-224 on 50 random crop images from the
cached dataset, using 356 Norwegian grocery product category names.

Reports:
- Top-1 and top-5 accuracy
- Inference speed per crop
- Model size (FP32 and estimated INT8)
"""

import json
import random
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "google/siglip-base-patch16-224"
NUM_SAMPLES = 50
SEED = 42
DATASET_YAML = "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/stratified_split/dataset.yaml"
SAMPLES_JSON = "/home/me/ht/nmiai/tasks/object-detection/vlm-approach/cached_dataset/samples.json"
BATCH_SIZE = 32

random.seed(SEED)


def load_categories(yaml_path: str) -> dict[int, str]:
    """Load category id -> name mapping from dataset.yaml."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["names"]


def load_samples(json_path: str, n: int) -> list[dict]:
    """Load n random samples from the cached dataset."""
    with open(json_path) as f:
        all_samples = json.load(f)
    # Filter out unknown_product (class 355) and empty class name (class 300)
    valid = [s for s in all_samples if s["category_id"] not in (300, 355)]
    return random.sample(valid, min(n, len(valid)))


def extract_features(output):
    """Extract feature tensor from model output (handles both tensor and BaseModelOutput)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state[:, -1]  # SigLIP uses last token
    raise ValueError(f"Cannot extract features from {type(output)}")


def encode_texts(model, processor, prompts, device):
    """Encode text prompts in batches, return normalized feature matrix on device."""
    all_features = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        inputs = processor(text=batch, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            features = extract_features(out)
            all_features.append(features.cpu())
        del inputs
        torch.cuda.empty_cache()
    features = torch.cat(all_features, dim=0).to(device)
    return features / features.norm(dim=-1, keepdim=True)


def encode_image(model, processor, img, device):
    """Encode a single image, return normalized feature vector."""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        features = extract_features(out)
    return features / features.norm(dim=-1, keepdim=True)


def classify_samples(model, processor, samples, text_features, categories, device):
    """Classify samples against text features, return (top1, top5, total, mistakes, times)."""
    sorted_cat_keys = sorted(categories.keys())
    top1 = top5 = total = 0
    mistakes = []
    times = []

    for s in samples:
        img_path = s["crop_path"]
        if not Path(img_path).exists():
            continue

        img = Image.open(img_path).convert("RGB")
        t0 = time.time()
        img_feat = encode_image(model, processor, img, device)
        similarity = (img_feat @ text_features.T).squeeze(0)
        top5_ids = similarity.topk(5).indices.tolist()
        elapsed = time.time() - t0
        times.append(elapsed)

        top5_cat_ids = [sorted_cat_keys[i] for i in top5_ids]
        top5_names = [categories[cid] for cid in top5_cat_ids]
        gt_id = s["category_id"]

        if top5_cat_ids[0] == gt_id:
            top1 += 1
        if gt_id in top5_cat_ids:
            top5 += 1
        total += 1

        if top5_cat_ids[0] != gt_id:
            mistakes.append({
                "gt": s["category_name"],
                "pred_top1": top5_names[0],
                "pred_top5": top5_names,
            })

    return top1, top5, total, mistakes, times


def main():
    print(f"Loading model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # ── Model size ──────────────────────────────────────────────────────
    param_count = sum(p.numel() for p in model.parameters())
    fp32_mb = param_count * 4 / (1024 ** 2)
    int8_mb = param_count * 1 / (1024 ** 2)
    print(f"\nModel parameters: {param_count:,}")
    print(f"FP32 size: {fp32_mb:.1f} MB")
    print(f"INT8 estimated size: {int8_mb:.1f} MB")
    print(f"Budget remaining after YOLO (~230MB): {190 - int8_mb:.1f} MB {'OK' if int8_mb < 190 else 'OVER BUDGET'}")

    # ── Load categories ─────────────────────────────────────────────────
    categories = load_categories(DATASET_YAML)
    cat_names = [categories[i] for i in sorted(categories.keys())]
    print(f"\nCategories: {len(cat_names)}")

    # ── Default prompt template ─────────────────────────────────────────
    text_prompts = [f"a photo of {name}" for name in cat_names]

    print("Encoding text prompts (batched)...")
    t0 = time.time()
    text_features = encode_texts(model, processor, text_prompts, device)
    print(f"Text encoding: {time.time() - t0:.2f}s for {len(text_prompts)} prompts")

    # ── Classify samples ────────────────────────────────────────────────
    samples = load_samples(SAMPLES_JSON, NUM_SAMPLES)
    print(f"\nClassifying {len(samples)} samples...")

    top1, top5, total, mistakes, times = classify_samples(
        model, processor, samples, text_features, categories, device
    )

    # ── Results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS ({total} samples, template: 'a photo of X')")
    print(f"{'='*60}")
    print(f"Top-1 accuracy: {top1}/{total} = {100*top1/total:.1f}%")
    print(f"Top-5 accuracy: {top5}/{total} = {100*top5/total:.1f}%")
    avg_ms = 1000 * sum(times) / len(times)
    print(f"Avg inference time per crop: {avg_ms:.1f}ms")
    print(f"  (text features pre-cached, only image encoding + similarity)")

    crops_per_image = 10
    images_per_batch = 100
    total_crops = crops_per_image * images_per_batch
    est_time = total_crops * avg_ms / 1000
    print(f"\nEstimated time for {total_crops} crops: {est_time:.1f}s (budget: 300s)")

    # Show some mistakes
    print(f"\n{'='*60}")
    print(f"SAMPLE MISTAKES (first 10)")
    print(f"{'='*60}")
    for m in mistakes[:10]:
        print(f"  GT: {m['gt']}")
        print(f"  Predicted: {m['pred_top1']}")
        print(f"  Top-5: {m['pred_top5'][:3]}...")
        print()

    # ── Alternative prompt templates ────────────────────────────────────
    print(f"\n{'='*60}")
    print("TESTING ALTERNATIVE PROMPT TEMPLATES")
    print(f"{'='*60}")

    templates = {
        "bare name": cat_names,
        "et bilde av": [f"et bilde av {n}" for n in cat_names],
        "a grocery product called": [f"a grocery product called {n}" for n in cat_names],
        "a product on a shelf:": [f"a product on a store shelf: {n}" for n in cat_names],
    }

    for tmpl_name, prompts in templates.items():
        tf = encode_texts(model, processor, prompts, device)
        t1, t5, n, _, _ = classify_samples(model, processor, samples, tf, categories, device)
        print(f"  Template '{tmpl_name}': top1={100*t1/n:.1f}%, top5={100*t5/n:.1f}% ({n} samples)")

    # ── Sandbox compatibility ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SANDBOX COMPATIBILITY CHECK")
    print(f"{'='*60}")
    print(f"transformers version: {__import__('transformers').__version__}")
    print(f"torch version: {torch.__version__}")
    try:
        from transformers import SiglipModel, SiglipProcessor
        print("SiglipModel/SiglipProcessor: AVAILABLE (named import)")
    except ImportError:
        print("SiglipModel/SiglipProcessor: NOT AVAILABLE as named import")
        print("  Using AutoModel/AutoProcessor instead (works fine)")
    print(f"AutoModel.from_pretrained: works (used above)")
    print(f"AutoProcessor.from_pretrained: works (used above)")

    # ── Can we pre-compute and save text embeddings? ────────────────────
    print(f"\n{'='*60}")
    print("TEXT EMBEDDING CACHE SIZE")
    print(f"{'='*60}")
    emb_size_mb = text_features.shape[0] * text_features.shape[1] * 4 / (1024**2)
    print(f"Text embeddings shape: {text_features.shape}")
    print(f"Text embeddings size (FP32): {emb_size_mb:.2f} MB")
    print(f"Text embeddings size (FP16): {emb_size_mb/2:.2f} MB")
    print("-> Can save pre-computed embeddings, skip text encoder at inference")
    print("-> Only need vision encoder in the ZIP (~half the model)")


if __name__ == "__main__":
    main()
