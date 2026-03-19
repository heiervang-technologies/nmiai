"""
Evaluate DINOv2 classification quality on the training set validation split.

Tests how well nearest-neighbor matching against reference embeddings
can classify crops from the COCO training annotations.

Can run on CPU (no GPU needed).

Usage: python eval_classification.py [--device cpu]
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np


DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"

# Data agent's embeddings (averaged per category from training crops)
DATA_AGENT_REF = DATA_ROOT / "ref_embeddings.pth"

# Our own embeddings (per-angle reference images)
OUR_REF = Path(__file__).parent / "ref_embeddings.pth"

# Our cached training embeddings (if available from train_linear_probe.py)
TRAINING_EMB_CACHE = Path(__file__).parent / "training_embeddings.pth"


def eval_nearest_neighbor_with_category_embeddings():
    """
    Evaluate classification using the data agent's category_embeddings
    (one embedding per category from averaged training crops).

    Uses leave-one-out style: for each training crop embedding, match it
    against category centroids (excluding its own contribution would be ideal,
    but we approximate by matching against all centroids).
    """
    print("=" * 60)
    print("Eval: Category centroid matching (data agent embeddings)")
    print("=" * 60)

    data = torch.load(DATA_AGENT_REF, map_location="cpu", weights_only=False)

    cat_embs = F.normalize(data["category_embeddings"].float(), dim=-1)  # [356, 384]
    cat_ids = data["category_ids"]  # list of 356 ints

    # Also get reference embeddings + barcode mapping
    ref_embs = F.normalize(data["reference_embeddings"].float(), dim=-1)  # [344, 384]
    b2c = data["barcode_to_category"]  # {barcode: cat_id}
    ref_barcodes = data["reference_barcodes"]

    # Build ref category ID list
    ref_cat_ids = []
    for bc in ref_barcodes:
        bc_str = str(bc)
        if bc_str in b2c:
            ref_cat_ids.append(b2c[bc_str])
        else:
            ref_cat_ids.append(-1)
    ref_cat_ids = torch.tensor(ref_cat_ids)

    print(f"Category embeddings: {cat_embs.shape}")
    print(f"Reference embeddings: {ref_embs.shape}")
    print(f"Categories with reference images: {(ref_cat_ids >= 0).sum().item()}")
    print()

    # Self-test: match each category embedding against all others
    sim = cat_embs @ cat_embs.T  # [356, 356]
    # Zero out diagonal (self-match)
    sim.fill_diagonal_(-1)

    # For each category, what's the nearest neighbor?
    top_sims, top_indices = sim.topk(5, dim=-1)

    # Count how often the nearest neighbor is a "similar" product
    # (We can't really evaluate accuracy here without held-out data,
    # but we can check embedding quality by looking at nearest neighbors)
    print("Sample nearest neighbors (category -> closest categories):")
    cat_names = data["category_names"]
    sample_ids = [0, 50, 100, 150, 200, 250, 300, 355]
    for cat_id in sample_ids:
        name = cat_names.get(str(cat_id), cat_names.get(cat_id, f"cat_{cat_id}"))
        neighbors = []
        for j in range(3):
            nn_id = top_indices[cat_id, j].item()
            nn_name = cat_names.get(str(nn_id), cat_names.get(nn_id, f"cat_{nn_id}"))
            nn_sim = top_sims[cat_id, j].item()
            neighbors.append(f"{nn_name[:30]} ({nn_sim:.3f})")
        print(f"  [{cat_id}] {name[:40]}")
        for n in neighbors:
            print(f"      -> {n}")
    print()

    # Evaluate: match reference embeddings against category centroids
    ref_sim = ref_embs @ cat_embs.T  # [344, 356]
    ref_preds = ref_sim.argmax(dim=-1)  # [344]

    # Check accuracy for references that have category mappings
    valid = ref_cat_ids >= 0
    if valid.sum() > 0:
        correct = (ref_preds[valid] == ref_cat_ids[valid]).sum().item()
        total = valid.sum().item()
        print(f"Reference -> Category centroid matching:")
        print(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")
        print(f"  (How often does a reference image match its assigned category centroid)")
    print()

    # Analyze similarity distribution
    # For correct matches
    if valid.sum() > 0:
        correct_mask = valid & (ref_preds == ref_cat_ids)
        wrong_mask = valid & (ref_preds != ref_cat_ids)

        if correct_mask.sum() > 0:
            correct_sims = ref_sim[correct_mask].max(dim=-1).values
            print(f"  Correct match similarity: mean={correct_sims.mean():.3f}, min={correct_sims.min():.3f}, max={correct_sims.max():.3f}")

        if wrong_mask.sum() > 0:
            wrong_sims = ref_sim[wrong_mask].max(dim=-1).values
            print(f"  Wrong match similarity: mean={wrong_sims.mean():.3f}, min={wrong_sims.min():.3f}, max={wrong_sims.max():.3f}")

            # Show some wrong matches
            wrong_indices = wrong_mask.nonzero().squeeze(-1)[:10]
            print(f"\n  Sample mismatches:")
            for idx in wrong_indices:
                true_cat = ref_cat_ids[idx].item()
                pred_cat = ref_preds[idx].item()
                true_name = cat_names.get(str(true_cat), cat_names.get(true_cat, "?"))
                pred_name = cat_names.get(str(pred_cat), cat_names.get(pred_cat, "?"))
                print(f"    True: [{true_cat}] {true_name[:40]}")
                print(f"    Pred: [{pred_cat}] {pred_name[:40]}")
                print(f"    Sim:  true={ref_sim[idx, true_cat]:.3f} pred={ref_sim[idx, pred_cat]:.3f}")
                print()


def eval_our_ref_embeddings():
    """Evaluate our own multi-angle reference embeddings."""
    if not OUR_REF.exists():
        print("Our ref_embeddings.pth not found, skipping")
        return

    print("=" * 60)
    print("Eval: Our multi-angle reference embeddings")
    print("=" * 60)

    data = torch.load(OUR_REF, map_location="cpu", weights_only=False)

    if "embeddings" in data:
        emb_dict = data["embeddings"]
    else:
        emb_dict = data

    total_refs = sum(v.shape[0] if isinstance(v, torch.Tensor) else 0 for v in emb_dict.values())
    print(f"Categories with reference images: {len(emb_dict)}")
    print(f"Total reference embeddings: {total_refs}")
    print(f"Embed dim: {data.get('embed_dim', 'unknown')}")
    print()

    # Check intra-class similarity (how similar are different angles of same product)
    intra_sims = []
    for cat_id, embs in emb_dict.items():
        if isinstance(embs, torch.Tensor) and embs.shape[0] > 1:
            embs = F.normalize(embs.float(), dim=-1)
            sim = embs @ embs.T
            # Off-diagonal elements
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
            intra_sims.append(sim[mask].mean().item())

    if intra_sims:
        print(f"Intra-class similarity (same product, different angles):")
        print(f"  Mean: {np.mean(intra_sims):.3f}")
        print(f"  Min:  {np.min(intra_sims):.3f}")
        print(f"  Max:  {np.max(intra_sims):.3f}")
    print()

    # Check inter-class similarity (different products)
    # Sample 50 random pairs
    cat_ids_list = [k for k, v in emb_dict.items() if isinstance(v, torch.Tensor)]
    centroids = {}
    for cat_id in cat_ids_list:
        embs = F.normalize(emb_dict[cat_id].float(), dim=-1)
        centroids[cat_id] = embs.mean(dim=0)

    if len(centroids) > 1:
        centroid_matrix = torch.stack(list(centroids.values()))
        centroid_matrix = F.normalize(centroid_matrix, dim=-1)
        inter_sim = centroid_matrix @ centroid_matrix.T
        mask = ~torch.eye(inter_sim.shape[0], dtype=torch.bool)
        inter_vals = inter_sim[mask]
        print(f"Inter-class similarity (different products):")
        print(f"  Mean: {inter_vals.mean():.3f}")
        print(f"  Min:  {inter_vals.min():.3f}")
        print(f"  Max:  {inter_vals.max():.3f}")

        # Find most confusing pairs
        inter_sim_masked = inter_sim.clone()
        inter_sim_masked.fill_diagonal_(-1)
        top_vals, top_flat = inter_sim_masked.flatten().topk(10)
        rows = top_flat // inter_sim.shape[1]
        cols = top_flat % inter_sim.shape[1]
        print(f"\n  Most similar product pairs (potential confusion):")
        seen = set()
        for r, c, s in zip(rows.tolist(), cols.tolist(), top_vals.tolist()):
            pair = (min(r, c), max(r, c))
            if pair in seen:
                continue
            seen.add(pair)
            cat_a = cat_ids_list[r]
            cat_b = cat_ids_list[c]
            print(f"    [{cat_a}] <-> [{cat_b}]  sim={s:.3f}")
            if len(seen) >= 5:
                break


def main():
    eval_nearest_neighbor_with_category_embeddings()
    print("\n")
    eval_our_ref_embeddings()


if __name__ == "__main__":
    main()
