"""Retrain DINOv2 linear probe on 18,576 classifier crops with hard negative awareness.

Changes from v2:
- Uses new classifier_crops directory (image-based, not JSON)
- More crops across all 356 categories
- Hard negative weighting for confusable pairs
- Saves to multiple submission dirs

Usage: CUDA_VISIBLE_DEVICES=1 python retrain_dino_probe_v3.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
DINO_WEIGHTS = SCRIPT_DIR / "submission-single-model" / "dinov2_vits14.pth"
CROPS_DIR = SCRIPT_DIR / "data-creation" / "data" / "classifier_crops"
HARD_PAIRS = SCRIPT_DIR / "data-creation" / "data" / "hard_negative_pairs" / "pair_manifest.json"
NUM_CLASSES = 356
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.1  # Higher LR needed for linear probe on normalized features


class ImageFolderDataset(Dataset):
    """Load crops from classifier_crops/{category_id}/*.jpg"""
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, category_id)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cat_id = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, cat_id


def load_crops():
    """Load all crops from the classifier_crops directory structure."""
    samples = []
    if not CROPS_DIR.exists():
        raise FileNotFoundError(f"Crops directory not found: {CROPS_DIR}")

    for cat_dir in sorted(CROPS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        try:
            cat_id = int(cat_dir.name)
        except ValueError:
            continue
        for img_path in cat_dir.iterdir():
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                samples.append((str(img_path), cat_id))

    return samples


def load_confusable_weights():
    """Load confusable pair weights for loss weighting."""
    weights = {}  # cat_id -> weight multiplier
    if HARD_PAIRS.exists():
        with open(HARD_PAIRS) as f:
            pairs = json.load(f)
        # Categories in confusable pairs get higher loss weight
        confusable_cats = set()
        for pair in pairs if isinstance(pairs, list) else pairs.get("pairs", []):
            cat_a = pair.get("cat_a", pair.get("category_a"))
            cat_b = pair.get("cat_b", pair.get("category_b"))
            sim = pair.get("similarity", 0.8)
            if cat_a is not None:
                confusable_cats.add(cat_a)
            if cat_b is not None:
                confusable_cats.add(cat_b)

        for cat in confusable_cats:
            weights[cat] = 1.5  # 1.5x loss weight for confusable categories (was 2.0, too aggressive)
        print(f"Loaded {len(confusable_cats)} confusable categories with 2x loss weight")
    else:
        print("No hard negative pairs file found, using uniform weights")

    return weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load DINOv2
    print("Loading DINOv2 ViT-S/14...")
    if not DINO_WEIGHTS.exists():
        # Try downloading or using timm pretrained
        print(f"Weights not found at {DINO_WEIGHTS}, using timm pretrained")
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
    else:
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
        state_dict = torch.load(DINO_WEIGHTS, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)
    embed_dim = 384  # ViT-S

    # Load crops
    print("Loading classifier crops...")
    all_samples = load_crops()
    print(f"Total crops: {len(all_samples)}")

    # Category distribution
    cat_counts = Counter(s[1] for s in all_samples)
    print(f"Categories with crops: {len(cat_counts)}")
    print(f"Min crops/cat: {min(cat_counts.values())}, Max: {max(cat_counts.values())}, Median: {sorted(cat_counts.values())[len(cat_counts)//2]}")

    # Load confusable weights
    confusable_weights = load_confusable_weights()

    # Split into train/val (90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    val_size = len(all_samples) // 10
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Extract features
    print("Extracting features...")
    t0 = time.time()

    def extract_features(sample_list):
        ds = ImageFolderDataset(sample_list, transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        all_feats, all_labels = [], []
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for i, (imgs, labels) in enumerate(loader):
                feats = model(imgs.to(device))
                feats = F.normalize(feats.float(), dim=-1)
                all_feats.append(feats.cpu())
                all_labels.append(labels)
                if (i + 1) % 50 == 0:
                    print(f"  [{(i+1)*BATCH_SIZE}/{len(sample_list)}] {time.time()-t0:.0f}s")
        return torch.cat(all_feats), torch.cat(all_labels)

    train_feats, train_labels = extract_features(train_samples)
    print(f"Train features: {train_feats.shape} in {time.time()-t0:.0f}s")

    val_feats, val_labels = extract_features(val_samples)
    print(f"Val features: {val_feats.shape}")

    # Build per-sample loss weights
    sample_weights = torch.ones(len(train_labels))
    for i, label in enumerate(train_labels.tolist()):
        if label in confusable_weights:
            sample_weights[i] = confusable_weights[label]

    # Train linear probe
    print(f"Training linear probe ({EPOCHS} epochs, lr={LR})...")
    probe = nn.Linear(embed_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=LR, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_feats_gpu = train_feats.to(device)
    train_labels_gpu = train_labels.to(device)
    sample_weights_gpu = sample_weights.to(device)

    best_val_acc = 0
    best_state = None

    for epoch in range(EPOCHS):
        probe.train()
        perm = torch.randperm(len(train_feats_gpu))
        total_loss = 0
        correct = 0
        for start in range(0, len(perm), 4096):
            idx = perm[start:start + 4096]
            logits = probe(train_feats_gpu[idx])
            # Weighted cross-entropy
            loss_per_sample = F.cross_entropy(logits, train_labels_gpu[idx], reduction='none')
            loss = (loss_per_sample * sample_weights_gpu[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == train_labels_gpu[idx]).sum().item()
        scheduler.step()

        train_acc = correct / len(train_feats)

        # Val accuracy
        probe.eval()
        with torch.inference_mode():
            val_logits = probe(val_feats.to(device))
            val_acc = (val_logits.argmax(1) == val_labels.to(device)).float().mean().item()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {"weight": probe.weight.data.cpu().clone(), "bias": probe.bias.data.cpu().clone()}
            marker = " *BEST*"

        print(f"  Epoch {epoch+1}/{EPOCHS}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} loss={total_loss/len(train_feats):.4f}{marker}")

    # Save best probe
    if best_state is None:
        best_state = {"weight": probe.weight.data.cpu(), "bias": probe.bias.data.cpu()}

    # Save to multiple locations
    save_paths = [
        SCRIPT_DIR / "linear_probe_v3.pth",
        SCRIPT_DIR / "submission-single-model" / "linear_probe_v3.pth",
        SCRIPT_DIR / "submission-ensemble" / "linear_probe_v3.pth",
        SCRIPT_DIR / "submission-dual-classifier" / "linear_probe_v3.pth",
    ]
    for p in save_paths:
        if p.parent.exists():
            torch.save(best_state, p)
            print(f"Saved to {p}")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
