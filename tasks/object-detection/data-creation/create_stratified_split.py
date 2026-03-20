"""
Stratified train/val split for NM i AI object detection challenge.
~100 train / ~150 val, optimized for val representativeness.
Uses DINOv2 embeddings for visual clustering + category/density stratification.
"""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
COCO_DIR = BASE / "data" / "coco_dataset" / "train"
ANN_FILE = COCO_DIR / "annotations.json"
IMG_DIR = COCO_DIR / "images"
OUT_DIR = BASE / "data" / "stratified_split"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TARGET = 100
N_CLUSTERS = 15

# ── 1. Load annotations ───────────────────────────────────────────────
print("Loading annotations...")
with open(ANN_FILE) as f:
    coco = json.load(f)

images_by_id = {img["id"]: img for img in coco["images"]}
categories_by_id = {cat["id"]: cat["name"] for cat in coco["categories"]}
n_categories = len(categories_by_id)
print(f"  {len(images_by_id)} images, {len(coco['annotations'])} annotations, {n_categories} categories")

# ── 2. Per-image features ─────────────────────────────────────────────
print("Computing per-image features...")
img_cat_counts = defaultdict(lambda: Counter())  # img_id -> Counter of cat_ids
img_ann_count = Counter()  # img_id -> total annotations

for ann in coco["annotations"]:
    img_id = ann["image_id"]
    img_cat_counts[img_id][ann["category_id"]] += 1
    img_ann_count[img_id] += 1

# Build feature dict per image
img_features = {}
for img_id, img_info in images_by_id.items():
    w, h = img_info["width"], img_info["height"]
    aspect = w / h if h > 0 else 1.0
    density = img_ann_count[img_id]
    cats_present = set(img_cat_counts[img_id].keys())
    img_features[img_id] = {
        "filename": img_info["file_name"],
        "width": w,
        "height": h,
        "aspect_ratio": aspect,
        "n_annotations": density,
        "categories": cats_present,
        "cat_counts": dict(img_cat_counts[img_id]),
    }

img_ids = sorted(img_features.keys())
filenames = [img_features[i]["filename"] for i in img_ids]

# ── 3. DINOv2 embeddings ──────────────────────────────────────────────
print("Computing DINOv2 embeddings...")
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Using device: {device}")

model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
model = model.eval().to(device)

data_cfg = resolve_data_config(model.pretrained_cfg)
transform = create_transform(**data_cfg)

embeddings = []
batch_size = 16
for start in range(0, len(img_ids), batch_size):
    batch_ids = img_ids[start : start + batch_size]
    tensors = []
    for iid in batch_ids:
        img_path = IMG_DIR / img_features[iid]["filename"]
        img = Image.open(img_path).convert("RGB")
        tensors.append(transform(img))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = model(batch)
    embeddings.append(feats.cpu().numpy())
    if (start // batch_size) % 5 == 0:
        print(f"  Processed {min(start + batch_size, len(img_ids))}/{len(img_ids)}")

embeddings = np.concatenate(embeddings, axis=0)  # (248, D)
print(f"  Embedding shape: {embeddings.shape}")

# ── 4. Clustering ─────────────────────────────────────────────────────
print(f"Clustering with KMeans k={N_CLUSTERS}...")
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(emb_scaled)

cluster_counts = Counter(cluster_labels)
print("  Cluster sizes:", dict(sorted(cluster_counts.items())))

# Store cluster assignment
for idx, iid in enumerate(img_ids):
    img_features[iid]["cluster"] = int(cluster_labels[idx])

# ── 5. Stratified split ───────────────────────────────────────────────
print("Creating stratified split...")

# Group images by cluster
cluster_to_imgs = defaultdict(list)
for idx, iid in enumerate(img_ids):
    cluster_to_imgs[img_features[iid]["cluster"]].append(iid)

# For each cluster, allocate ~40% to train, ~60% to val
# But also ensure every category appears in val
np.random.seed(42)

train_ids = []
val_ids = []

for c in sorted(cluster_to_imgs.keys()):
    members = cluster_to_imgs[c]
    np.random.shuffle(members)
    n_train = max(1, round(len(members) * TRAIN_TARGET / len(img_ids)))
    # At least 1 in each set if cluster has >= 2
    if len(members) >= 2:
        n_train = max(1, min(n_train, len(members) - 1))
    else:
        # Single-member cluster -> put in val (val is priority)
        n_train = 0
    train_ids.extend(members[:n_train])
    val_ids.extend(members[n_train:])

print(f"  Initial split: {len(train_ids)} train / {len(val_ids)} val")

# Check category coverage in val
val_cats = set()
for iid in val_ids:
    val_cats.update(img_features[iid]["categories"])

train_cats = set()
for iid in train_ids:
    train_cats.update(img_features[iid]["categories"])

all_cats = set()
for iid in img_ids:
    all_cats.update(img_features[iid]["categories"])

missing_in_val = all_cats - val_cats
print(f"  Categories in val: {len(val_cats)}/{len(all_cats)}")
if missing_in_val:
    print(f"  Missing in val: {missing_in_val}")
    # Move images from train that contain missing categories
    for cat_id in list(missing_in_val):
        # Find a train image with this category
        for iid in list(train_ids):
            if cat_id in img_features[iid]["categories"]:
                train_ids.remove(iid)
                val_ids.append(iid)
                val_cats.update(img_features[iid]["categories"])
                print(f"    Moved image {iid} to val for category {cat_id}")
                break

# Also ensure every category is in train if possible
missing_in_train = all_cats - train_cats
if missing_in_train:
    print(f"  Categories missing in train: {len(missing_in_train)}")
    # Move val images to train for missing categories (only if val is large enough)
    for cat_id in list(missing_in_train):
        if len(val_ids) <= 145:
            break
        for iid in list(val_ids):
            if cat_id in img_features[iid]["categories"]:
                val_ids.remove(iid)
                train_ids.append(iid)
                train_cats.update(img_features[iid]["categories"])
                print(f"    Moved image {iid} to train for category {cat_id}")
                break

print(f"  Final split: {len(train_ids)} train / {len(val_ids)} val")

# Recompute coverage
val_cats = set()
for iid in val_ids:
    val_cats.update(img_features[iid]["categories"])
train_cats = set()
for iid in train_ids:
    train_cats.update(img_features[iid]["categories"])

val_only = val_cats - train_cats
train_only = train_cats - val_cats

print(f"  Val categories: {len(val_cats)}/{len(all_cats)}")
print(f"  Train categories: {len(train_cats)}/{len(all_cats)}")
print(f"  Val-only categories: {len(val_only)} -> {[categories_by_id.get(c, c) for c in val_only]}")
print(f"  Train-only categories: {len(train_only)} -> {[categories_by_id.get(c, c) for c in train_only]}")

# ── 6. Save text files ────────────────────────────────────────────────
print("Saving split files...")

train_filenames = sorted([img_features[i]["filename"] for i in train_ids])
val_filenames = sorted([img_features[i]["filename"] for i in val_ids])

(OUT_DIR / "train_images.txt").write_text("\n".join(train_filenames) + "\n")
(OUT_DIR / "val_images.txt").write_text("\n".join(val_filenames) + "\n")

# ── 7. Split analysis JSON ────────────────────────────────────────────
# Cluster distribution comparison
train_cluster_dist = Counter([img_features[i]["cluster"] for i in train_ids])
val_cluster_dist = Counter([img_features[i]["cluster"] for i in val_ids])

# Density stats
train_densities = [img_features[i]["n_annotations"] for i in train_ids]
val_densities = [img_features[i]["n_annotations"] for i in val_ids]

# Category frequency in each split
train_cat_freq = Counter()
val_cat_freq = Counter()
for iid in train_ids:
    for cat_id, cnt in img_features[iid]["cat_counts"].items():
        train_cat_freq[cat_id] += cnt
for iid in val_ids:
    for cat_id, cnt in img_features[iid]["cat_counts"].items():
        val_cat_freq[cat_id] += cnt

analysis = {
    "split_sizes": {"train": len(train_ids), "val": len(val_ids)},
    "category_coverage": {
        "total": len(all_cats),
        "in_train": len(train_cats),
        "in_val": len(val_cats),
        "val_only": [categories_by_id.get(c, str(c)) for c in sorted(val_only)],
        "train_only": [categories_by_id.get(c, str(c)) for c in sorted(train_only)],
    },
    "density_stats": {
        "train_mean": float(np.mean(train_densities)),
        "train_std": float(np.std(train_densities)),
        "val_mean": float(np.mean(val_densities)),
        "val_std": float(np.std(val_densities)),
    },
    "cluster_distribution": {
        "n_clusters": N_CLUSTERS,
        "train": {str(k): v for k, v in sorted(train_cluster_dist.items())},
        "val": {str(k): v for k, v in sorted(val_cluster_dist.items())},
    },
    "image_assignments": {
        img_features[iid]["filename"]: {
            "split": "train" if iid in train_ids else "val",
            "cluster": img_features[iid]["cluster"],
            "n_annotations": img_features[iid]["n_annotations"],
        }
        for iid in img_ids
    },
}

with open(OUT_DIR / "split_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

# ── 8. Visualization ──────────────────────────────────────────────────
print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 8a. Cluster distribution
clusters = sorted(set(cluster_labels))
train_counts = [train_cluster_dist.get(c, 0) for c in clusters]
val_counts = [val_cluster_dist.get(c, 0) for c in clusters]
x = np.arange(len(clusters))
w = 0.35
axes[0, 0].bar(x - w/2, train_counts, w, label="Train", color="#2196F3")
axes[0, 0].bar(x + w/2, val_counts, w, label="Val", color="#FF9800")
axes[0, 0].set_xlabel("Cluster")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Cluster Distribution")
axes[0, 0].set_xticks(x)
axes[0, 0].legend()

# 8b. Density distribution
axes[0, 1].hist(train_densities, bins=20, alpha=0.6, label="Train", color="#2196F3")
axes[0, 1].hist(val_densities, bins=20, alpha=0.6, label="Val", color="#FF9800")
axes[0, 1].set_xlabel("Annotations per image")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Annotation Density Distribution")
axes[0, 1].legend()

# 8c. Category frequency comparison (top 30)
all_cat_ids_sorted = sorted(all_cats, key=lambda c: train_cat_freq.get(c, 0) + val_cat_freq.get(c, 0), reverse=True)[:30]
cat_names = [categories_by_id.get(c, str(c))[:20] for c in all_cat_ids_sorted]
train_f = [train_cat_freq.get(c, 0) for c in all_cat_ids_sorted]
val_f = [val_cat_freq.get(c, 0) for c in all_cat_ids_sorted]
y = np.arange(len(cat_names))
axes[1, 0].barh(y - 0.2, train_f, 0.4, label="Train", color="#2196F3")
axes[1, 0].barh(y + 0.2, val_f, 0.4, label="Val", color="#FF9800")
axes[1, 0].set_yticks(y)
axes[1, 0].set_yticklabels(cat_names, fontsize=6)
axes[1, 0].set_xlabel("Annotation count")
axes[1, 0].set_title("Top 30 Category Frequencies")
axes[1, 0].legend()
axes[1, 0].invert_yaxis()

# 8d. 2D embedding visualization (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_scaled)

train_mask = np.array([iid in set(train_ids) for iid in img_ids])
val_mask = ~train_mask

axes[1, 1].scatter(emb_2d[train_mask, 0], emb_2d[train_mask, 1],
                    c=[img_features[img_ids[i]]["cluster"] for i in range(len(img_ids)) if train_mask[i]],
                    cmap="tab20", marker="o", s=40, alpha=0.7, edgecolors="blue", linewidths=1.5, label="Train")
axes[1, 1].scatter(emb_2d[val_mask, 0], emb_2d[val_mask, 1],
                    c=[img_features[img_ids[i]]["cluster"] for i in range(len(img_ids)) if val_mask[i]],
                    cmap="tab20", marker="s", s=40, alpha=0.7, edgecolors="orange", linewidths=1.5, label="Val")
axes[1, 1].set_title("DINOv2 Embeddings (PCA) - Color=Cluster, Edge=Split")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "split_visualization.png", dpi=150, bbox_inches="tight")
print(f"  Saved visualization to {OUT_DIR / 'split_visualization.png'}")

# ── 9. Create YOLO-format dataset ─────────────────────────────────────
print("Creating YOLO-format dataset directories...")

# Build category id mapping: COCO cat_id -> YOLO class index (0-based contiguous)
cat_ids_sorted = sorted(categories_by_id.keys())
coco_to_yolo = {cid: idx for idx, cid in enumerate(cat_ids_sorted)}

# Pre-index annotations by image_id
anns_by_image = defaultdict(list)
for ann in coco["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)

def create_yolo_labels(image_ids, split_name):
    """Create YOLO label files for a set of images."""
    img_out = OUT_DIR / split_name / "images"
    lbl_out = OUT_DIR / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for iid in image_ids:
        info = img_features[iid]
        fname = info["filename"]
        w, h = info["width"], info["height"]

        # Symlink image
        src = IMG_DIR / fname
        dst = img_out / fname
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

        # Create YOLO label
        label_fname = Path(fname).stem + ".txt"
        lines = []
        for ann in anns_by_image[iid]:
            cat_idx = coco_to_yolo[ann["category_id"]]
            bx, by, bw, bh = ann["bbox"]  # COCO: x,y,w,h (top-left)
            # Convert to YOLO: center_x, center_y, w, h (normalized)
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            lines.append(f"{cat_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (lbl_out / label_fname).write_text("\n".join(lines) + "\n" if lines else "")

    print(f"  {split_name}: {len(image_ids)} images, labels written")

create_yolo_labels(train_ids, "train")
create_yolo_labels(val_ids, "val")

# dataset.yaml
yolo_names = {coco_to_yolo[cid]: categories_by_id[cid] for cid in cat_ids_sorted}

dataset_cfg = {
    "path": str(OUT_DIR.resolve()),
    "train": "train/images",
    "val": "val/images",
    "nc": n_categories,
    "names": yolo_names,
}

with open(OUT_DIR / "dataset.yaml", "w") as f:
    yaml.dump(dataset_cfg, f, default_flow_style=False, allow_unicode=True)

print(f"\nDataset YAML written to {OUT_DIR / 'dataset.yaml'}")
print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)
print(f"Train: {len(train_ids)} images, Val: {len(val_ids)} images")
print(f"Total categories: {len(all_cats)}")
print(f"Categories in train: {len(train_cats)} ({len(train_cats)/len(all_cats)*100:.1f}%)")
print(f"Categories in val: {len(val_cats)} ({len(val_cats)/len(all_cats)*100:.1f}%)")
print(f"Val-only categories: {len(val_only)}")
print(f"Train-only categories: {len(train_only)}")
print(f"Train density: {np.mean(train_densities):.1f} +/- {np.std(train_densities):.1f}")
print(f"Val density: {np.mean(val_densities):.1f} +/- {np.std(val_densities):.1f}")
print(f"Train annotations: {sum(train_densities)}")
print(f"Val annotations: {sum(val_densities)}")
print("=" * 60)
