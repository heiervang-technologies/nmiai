"""Evaluate SigLIP zero-shot classification on classifier crops.

Uses timm vision encoder + pre-computed text embeddings.
No transformers library needed at runtime.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_data_config

SCRIPT_DIR = Path(__file__).parent
TEXT_EMBED_PATH = SCRIPT_DIR / "text_embeddings.pth"
CROPS_DIR = SCRIPT_DIR / "data-creation/data/classifier_crops"

# Category aliases: merge umlaut spelling variants
ALIASES = {59: 61, 170: 260, 36: 201}


def load_text_embeddings(device):
    data = torch.load(TEXT_EMBED_PATH, map_location=device, weights_only=False)
    # Use ensemble embeddings (average of multiple prompts)
    embeds = data["ensemble_embeddings"].to(device).float()
    # Already L2-normalized
    return embeds, data["category_names"]


def load_crops_dataset(crops_dir):
    """Load (path, label) pairs from classifier_crops/{category_id}/*.jpg"""
    samples = []
    for cat_dir in sorted(crops_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_id = int(cat_dir.name)
        for img_path in sorted(cat_dir.glob("*.jpg")):
            samples.append((img_path, cat_id))
    return samples


@torch.inference_mode()
def evaluate(model_name="vit_so400m_patch14_siglip_384", batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load text embeddings
    text_embeds, cat_names = load_text_embeddings(device)
    num_classes = text_embeds.shape[0]
    print(f"Text embeddings: {text_embeds.shape}")

    # Load vision encoder
    print(f"Loading {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.eval().to(device)
    embed_dim = model.num_features
    print(f"Vision embed dim: {embed_dim}")
    assert embed_dim == text_embeds.shape[1], f"Dim mismatch: vision {embed_dim} vs text {text_embeds.shape[1]}"

    # Get transforms
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)
    print(f"Input size: {config.get('input_size', 'unknown')}")

    # Load dataset
    samples = load_crops_dataset(CROPS_DIR)
    print(f"Loaded {len(samples)} crops across {len(set(s[1] for s in samples))} categories")

    # SigLIP uses a learned temperature/logit_scale
    # Default SigLIP logit_scale is typically around exp(4.6) ≈ 100
    # For zero-shot we use temperature=1 on normalized embeddings (cosine sim)
    # The text embeddings are already normalized

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion_examples = []

    t0 = time.time()
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        images = []
        labels = []
        for path, label in batch_samples:
            try:
                img = Image.open(path).convert("RGB")
                images.append(transform(img))
                labels.append(label)
            except Exception:
                continue

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)
        batch_labels = torch.tensor(labels, device=device)

        # Forward through vision encoder
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            image_embeds = model(batch_tensor)
        image_embeds = image_embeds.float()
        image_embeds = F.normalize(image_embeds, dim=-1)

        # Cosine similarity as logits
        logits = image_embeds @ text_embeds.T  # [B, 356]

        # Apply aliases to predictions
        top5_preds = logits.topk(5, dim=-1).indices  # [B, 5]
        top1_preds = top5_preds[:, 0]

        # Apply aliases to ground truth too
        for j in range(len(labels)):
            gt = labels[j]
            gt_aliased = ALIASES.get(gt, gt)
            pred1 = ALIASES.get(top1_preds[j].item(), top1_preds[j].item())
            pred5 = [ALIASES.get(p.item(), p.item()) for p in top5_preds[j]]

            is_correct_1 = (pred1 == gt_aliased)
            is_correct_5 = (gt_aliased in pred5)

            correct_top1 += is_correct_1
            correct_top5 += is_correct_5
            total += 1
            per_class_total[gt_aliased] += 1
            if is_correct_1:
                per_class_correct[gt_aliased] += 1
            elif len(confusion_examples) < 50:
                confusion_examples.append({
                    "gt": gt_aliased,
                    "gt_name": cat_names[gt],
                    "pred": pred1,
                    "pred_name": cat_names[top1_preds[j].item()],
                    "score": logits[j, top1_preds[j]].item(),
                    "gt_score": logits[j, gt].item(),
                })

        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {total}/{len(samples)} ({100 * total / len(samples):.1f}%) "
                  f"top1={100 * correct_top1 / max(total, 1):.1f}% "
                  f"top5={100 * correct_top5 / max(total, 1):.1f}% "
                  f"[{elapsed:.1f}s]")

    elapsed = time.time() - t0
    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Total samples: {total}")
    print(f"Top-1 accuracy: {top1_acc:.2f}%")
    print(f"Top-5 accuracy: {top5_acc:.2f}%")
    print(f"Time: {elapsed:.1f}s ({1000 * elapsed / total:.1f}ms/sample)")
    print(f"{'=' * 60}")

    # Per-class analysis
    per_class_acc = {}
    for cat_id in per_class_total:
        acc = per_class_correct[cat_id] / per_class_total[cat_id]
        per_class_acc[cat_id] = acc

    # Worst classes
    worst = sorted(per_class_acc.items(), key=lambda x: x[1])[:20]
    print(f"\nWorst 20 classes:")
    for cat_id, acc in worst:
        name = cat_names[cat_id] if cat_id < len(cat_names) else "?"
        n = per_class_total[cat_id]
        print(f"  [{cat_id}] {name[:50]:50s} acc={100*acc:.0f}% ({per_class_correct[cat_id]}/{n})")

    # Confusion examples
    if confusion_examples:
        print(f"\nConfusion examples (first 20):")
        for ex in confusion_examples[:20]:
            print(f"  GT: [{ex['gt']}] {ex['gt_name'][:35]:35s} "
                  f"-> Pred: [{ex['pred']}] {ex['pred_name'][:35]:35s} "
                  f"(score={ex['score']:.3f}, gt_score={ex['gt_score']:.3f})")

    # Save results
    results = {
        "model": model_name,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "total_samples": total,
        "time_seconds": elapsed,
        "per_class_accuracy": {str(k): v for k, v in per_class_acc.items()},
        "worst_classes": [(cat_id, acc) for cat_id, acc in worst],
        "confusion_examples": confusion_examples[:50],
    }
    out_path = SCRIPT_DIR / f"siglip_zeroshot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_so400m_patch14_siglip_384")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    evaluate(args.model, args.batch_size)
