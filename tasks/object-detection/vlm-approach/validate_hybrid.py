"""
Validate the hybrid YOLO + DINOv2 classification pipeline.

Uses the COCO training annotations as ground truth:
1. For each annotation crop, extract DINOv2 embedding
2. Match against reference embeddings
3. Measure top-1 and top-5 classification accuracy
4. Analyze where DINOv2 helps vs hurts compared to YOLO-only

Can run on CPU with pre-cached embeddings.

Usage: python validate_hybrid.py [--cached]
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import numpy as np


DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"

# Embedding sources
DATA_AGENT_REF = DATA_ROOT / "ref_embeddings.pth"
OUR_REF = Path(__file__).parent / "ref_embeddings.pth"
TRAINING_EMB_CACHE = Path(__file__).parent / "training_embeddings.pth"


def load_all_ref_embeddings(device="cpu"):
    """Load and merge both embedding sources."""
    ref_embeddings = {}  # cat_id -> list of [embed_dim] tensors

    # Load data agent's category centroids (all 356 categories)
    if DATA_AGENT_REF.exists():
        data = torch.load(DATA_AGENT_REF, map_location=device, weights_only=False)
        cat_embs = F.normalize(data["category_embeddings"].float(), dim=-1)
        cat_ids = data.get("category_ids", list(range(cat_embs.shape[0])))

        for i, cat_id in enumerate(cat_ids):
            cat_id = int(cat_id)
            ref_embeddings.setdefault(cat_id, []).append(cat_embs[i])

        # Add reference image embeddings
        if "reference_embeddings" in data and "barcode_to_category" in data:
            ref_emb = F.normalize(data["reference_embeddings"].float(), dim=-1)
            barcodes = data.get("reference_barcodes", [])
            b2c = data["barcode_to_category"]
            for i, bc in enumerate(barcodes):
                bc_str = str(bc)
                if bc_str in b2c:
                    cat_id = int(b2c[bc_str])
                    ref_embeddings.setdefault(cat_id, []).append(ref_emb[i])

    # Load our multi-angle embeddings
    if OUR_REF.exists():
        data = torch.load(OUR_REF, map_location=device, weights_only=False)
        emb_dict = data.get("embeddings", data)
        for cat_id, emb in emb_dict.items():
            if not isinstance(emb, torch.Tensor):
                continue
            cat_id = int(cat_id)
            emb = F.normalize(emb.float(), dim=-1)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            for j in range(emb.shape[0]):
                ref_embeddings.setdefault(cat_id, []).append(emb[j])

    # Stack per category
    final = {}
    for cat_id, embs in ref_embeddings.items():
        final[cat_id] = torch.stack(embs)

    return final


def evaluate_nn_classification(train_embeddings, train_labels, ref_embeddings, split_ratio=0.8):
    """Evaluate nearest-neighbor classification on a held-out portion of training data."""
    N = train_embeddings.shape[0]
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(N * split_ratio)
    train_idx = indices[:split]
    val_idx = indices[split:]

    val_embs = F.normalize(train_embeddings[val_idx].float(), dim=-1)
    val_labels = train_labels[val_idx]

    print(f"Validation set: {len(val_idx)} crops")
    print(f"Training set: {len(train_idx)} crops")
    print()

    # Method 1: Match against reference embeddings (nearest neighbor)
    print("--- Method 1: Reference embedding nearest-neighbor ---")
    correct_top1 = 0
    correct_top5 = 0
    no_ref = 0

    for i in range(len(val_embs)):
        emb = val_embs[i].unsqueeze(0)  # [1, 384]
        true_label = val_labels[i].item()

        best_cat = -1
        best_sim = -1
        top5_cats = []

        for cat_id, ref_emb in ref_embeddings.items():
            sim = (emb @ ref_emb.T).max().item()
            if sim > best_sim:
                best_sim = sim
                best_cat = cat_id
            top5_cats.append((sim, cat_id))

        top5_cats.sort(reverse=True)
        top5_ids = [c for _, c in top5_cats[:5]]

        if best_cat == true_label:
            correct_top1 += 1
        if true_label in top5_ids:
            correct_top5 += 1

    top1_acc = correct_top1 / len(val_embs)
    top5_acc = correct_top5 / len(val_embs)
    print(f"  Top-1 accuracy: {correct_top1}/{len(val_embs)} = {top1_acc:.1%}")
    print(f"  Top-5 accuracy: {correct_top5}/{len(val_embs)} = {top5_acc:.1%}")
    print()

    # Method 2: Match against training crop centroids (leave-category-out)
    print("--- Method 2: Training crop centroid matching ---")
    train_embs = F.normalize(train_embeddings[train_idx].float(), dim=-1)
    train_labs = train_labels[train_idx]

    # Build per-category centroids from training split
    cat_centroids = {}
    for label in train_labs.unique():
        mask = train_labs == label
        centroid = train_embs[mask].mean(dim=0)
        cat_centroids[label.item()] = F.normalize(centroid.unsqueeze(0), dim=-1)

    centroid_matrix = torch.stack([cat_centroids[k] for k in sorted(cat_centroids.keys())]).squeeze(1)
    centroid_ids = torch.tensor(sorted(cat_centroids.keys()))

    sim = val_embs @ centroid_matrix.T  # [val, num_cats]
    top1_preds = centroid_ids[sim.argmax(dim=-1)]
    top5_preds = centroid_ids[sim.topk(5, dim=-1).indices]

    correct_top1 = (top1_preds == val_labels).sum().item()
    correct_top5 = sum(val_labels[i].item() in top5_preds[i].tolist() for i in range(len(val_embs)))

    print(f"  Top-1 accuracy: {correct_top1}/{len(val_embs)} = {correct_top1/len(val_embs):.1%}")
    print(f"  Top-5 accuracy: {correct_top5}/{len(val_embs)} = {correct_top5/len(val_embs):.1%}")
    print()

    # Method 3: Combined (average of ref + centroid similarities)
    print("--- Method 3: Combined ref + centroid matching ---")
    correct_top1 = 0
    for i in range(len(val_embs)):
        emb = val_embs[i].unsqueeze(0)
        true_label = val_labels[i].item()

        scores = {}

        # Reference similarity
        for cat_id, ref_emb in ref_embeddings.items():
            scores[cat_id] = (emb @ ref_emb.T).max().item() * 0.4

        # Centroid similarity
        centroid_sim = (emb @ centroid_matrix.T).squeeze(0)
        for j, cat_id in enumerate(centroid_ids.tolist()):
            scores[cat_id] = scores.get(cat_id, 0) + centroid_sim[j].item() * 0.6

        best_cat = max(scores, key=scores.get)
        if best_cat == true_label:
            correct_top1 += 1

    print(f"  Top-1 accuracy: {correct_top1}/{len(val_embs)} = {correct_top1/len(val_embs):.1%}")
    print()

    # Per-class analysis
    print("--- Per-class accuracy (worst categories) ---")
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for i in range(len(val_embs)):
        true_label = val_labels[i].item()
        emb = val_embs[i].unsqueeze(0)

        # Use centroid method
        sim = (emb @ centroid_matrix.T).squeeze(0)
        pred = centroid_ids[sim.argmax()].item()

        per_class_total[true_label] += 1
        if pred == true_label:
            per_class_correct[true_label] += 1

    # Load category names
    with open(COCO_ANNOTATIONS) as f:
        coco = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Worst classes
    class_accs = []
    for cat_id, total in per_class_total.items():
        acc = per_class_correct[cat_id] / total
        class_accs.append((acc, total, cat_id))

    class_accs.sort()
    print("  Worst 20 categories:")
    for acc, total, cat_id in class_accs[:20]:
        name = id_to_name.get(cat_id, f"cat_{cat_id}")
        print(f"    [{cat_id:3d}] {acc:.0%} ({per_class_correct[cat_id]}/{total}) {name[:50]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true", help="Use cached training embeddings")
    args = parser.parse_args()

    device = "cpu"

    # Load reference embeddings
    print("Loading reference embeddings...")
    ref_embeddings = load_all_ref_embeddings(device)
    print(f"Reference categories: {len(ref_embeddings)}")
    total_refs = sum(v.shape[0] for v in ref_embeddings.values())
    print(f"Total reference embeddings: {total_refs}")
    print()

    # Load or compute training embeddings
    if args.cached and TRAINING_EMB_CACHE.exists():
        print(f"Loading cached training embeddings from {TRAINING_EMB_CACHE}")
        cache = torch.load(TRAINING_EMB_CACHE, map_location=device, weights_only=False)
        train_embs = cache["embeddings"]
        train_labels = cache["labels"]
    else:
        # Check centurion cache
        centurion_cache = Path("/home/me/nmiai-vlm/training_embeddings.pth")
        if centurion_cache.exists():
            print(f"Loading training embeddings from centurion cache")
            cache = torch.load(centurion_cache, map_location=device, weights_only=False)
            train_embs = cache["embeddings"]
            train_labels = cache["labels"]
        else:
            print("ERROR: No cached training embeddings found.")
            print("Run train_linear_probe.py first to generate training_embeddings.pth")
            print("Or wait for centurion training to complete.")
            return

    print(f"Training embeddings: {train_embs.shape}")
    print(f"Training labels: {train_labels.shape}, unique classes: {train_labels.unique().shape[0]}")
    print()

    evaluate_nn_classification(train_embs, train_labels, ref_embeddings)


if __name__ == "__main__":
    main()
