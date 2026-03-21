"""Retrain DINOv2 linear probe v4 - FIXED hyperparameters.

v3 issue: 1.88% accuracy because:
1. SGD lr=0.01 too low for 356-class CE loss (starts ~9.8, ln(356)=5.87)
2. Features may not be properly loaded/normalized
3. Need Adam optimizer and much higher LR for linear probe

Fix: Adam lr=0.001, more epochs, gradient clipping, proper diagnostics.

Usage: CUDA_VISIBLE_DEVICES=1 python retrain_dino_probe_v4.py
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
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5


class ImageFolderDataset(Dataset):
    def __init__(self, samples, transform, img_size=518):
        self.samples = samples
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cat_id = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception as e:
            # Correct fallback size
            tensor = torch.zeros(3, self.img_size, self.img_size)
        return tensor, cat_id


def load_crops():
    samples = []
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"Free GPU memory: {free_mem:.1f} GB")

    # Load DINOv2
    print("Loading DINOv2 ViT-S/14...")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    if DINO_WEIGHTS.exists():
        state_dict = torch.load(DINO_WEIGHTS, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights: {len(state_dict)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
    else:
        print(f"  WARNING: No weights at {DINO_WEIGHTS}, using random init!")

    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)
    print(f"  Input size: {config['input_size']}")

    # Load crops
    print("Loading classifier crops...")
    all_samples = load_crops()
    print(f"Total crops: {len(all_samples)}")

    cat_counts = Counter(s[1] for s in all_samples)
    print(f"Categories: {len(cat_counts)}, Min: {min(cat_counts.values())}, Max: {max(cat_counts.values())}")

    # Split 90/10
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    val_size = len(all_samples) // 10
    train_samples = [all_samples[i] for i in indices[val_size:]]
    val_samples = [all_samples[i] for i in indices[:val_size]]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Extract features
    print("Extracting features...")
    t0 = time.time()

    def extract_features(sample_list, desc=""):
        ds = ImageFolderDataset(sample_list, transform, img_size=config['input_size'][1])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        all_feats, all_labels = [], []
        with torch.inference_mode():
            for i, (imgs, labels) in enumerate(loader):
                imgs = imgs.to(device)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        feats = model(imgs)
                else:
                    feats = model(imgs)
                # DO NOT normalize features for linear probe - raw features work better
                all_feats.append(feats.float().cpu())
                all_labels.append(labels)
                if (i + 1) % 50 == 0:
                    print(f"  {desc}[{(i+1)*BATCH_SIZE}/{len(sample_list)}] {time.time()-t0:.0f}s")
        return torch.cat(all_feats), torch.cat(all_labels)

    train_feats, train_labels = extract_features(train_samples, "train ")
    val_feats, val_labels = extract_features(val_samples, "val ")
    print(f"Features extracted in {time.time()-t0:.0f}s")
    print(f"  Train: {train_feats.shape}, Val: {val_feats.shape}")

    # Diagnostic: check feature statistics
    print(f"\nFeature diagnostics:")
    print(f"  Train mean: {train_feats.mean():.4f}, std: {train_feats.std():.4f}")
    print(f"  Train min: {train_feats.min():.4f}, max: {train_feats.max():.4f}")
    print(f"  Train norm (mean): {train_feats.norm(dim=1).mean():.4f}")

    # Check if features are all zeros (broken model)
    zero_ratio = (train_feats.abs() < 1e-6).float().mean()
    print(f"  Zero ratio: {zero_ratio:.4f}")
    if zero_ratio > 0.5:
        print("  WARNING: Features are mostly zeros! Model weights may not have loaded correctly.")

    # Check feature variance per dimension
    dim_std = train_feats.std(dim=0)
    dead_dims = (dim_std < 1e-6).sum().item()
    print(f"  Dead dimensions: {dead_dims}/{train_feats.shape[1]}")

    # Train linear probe with Adam (much better for linear probes)
    print(f"\nTraining linear probe (Adam, lr={LR}, {EPOCHS} epochs)...")
    probe = nn.Linear(train_feats.shape[1], NUM_CLASSES).to(device)

    # Initialize with small weights
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)

    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    train_feats_gpu = train_feats.to(device)
    train_labels_gpu = train_labels.to(device)

    best_val_acc = 0
    best_state = None
    patience = 0
    max_patience = 20

    for epoch in range(EPOCHS):
        probe.train()
        perm = torch.randperm(len(train_feats_gpu))
        total_loss = 0
        correct = 0

        for start in range(0, len(perm), 2048):
            idx = perm[start:start + 2048]
            logits = probe(train_feats_gpu[idx])
            loss = F.cross_entropy(logits, train_labels_gpu[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == train_labels_gpu[idx]).sum().item()

        scheduler.step()
        train_acc = correct / len(train_feats)
        avg_loss = total_loss / len(train_feats)

        # Val
        probe.eval()
        with torch.inference_mode():
            val_logits = probe(val_feats.to(device))
            val_preds = val_logits.argmax(1)
            val_acc = (val_preds == val_labels.to(device)).float().mean().item()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {"weight": probe.weight.data.cpu().clone(), "bias": probe.bias.data.cpu().clone()}
            marker = " *BEST*"
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or marker:
            print(f"  Epoch {epoch+1}/{EPOCHS}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}{marker}")

        if patience >= max_patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
            break

    # Save
    if best_state is None:
        best_state = {"weight": probe.weight.data.cpu(), "bias": probe.bias.data.cpu()}

    save_paths = [
        SCRIPT_DIR / "linear_probe_v4.pth",
        SCRIPT_DIR / "submission-single-model" / "linear_probe_v4.pth",
        SCRIPT_DIR / "submission-ensemble" / "linear_probe_v4.pth",
        SCRIPT_DIR / "submission-dual-classifier" / "linear_probe_v4.pth",
    ]
    for p in save_paths:
        if p.parent.exists():
            torch.save(best_state, p)
            print(f"Saved to {p}")

    print(f"\nBest val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

    # Per-class accuracy analysis on val
    probe.load_state_dict(best_state)
    probe.eval().to(device)
    with torch.inference_mode():
        val_logits = probe(val_feats.to(device))
        val_preds = val_logits.argmax(1).cpu()

    per_class_correct = Counter()
    per_class_total = Counter()
    for pred, true in zip(val_preds.tolist(), val_labels.tolist()):
        per_class_total[true] += 1
        if pred == true:
            per_class_correct[true] += 1

    worst = [(cid, per_class_correct[cid] / max(1, per_class_total[cid]), per_class_total[cid])
             for cid in per_class_total]
    worst.sort(key=lambda x: x[1])

    print(f"\n10 worst categories:")
    import json as _json
    ann_path = SCRIPT_DIR / "data-creation" / "data" / "coco_dataset" / "train" / "annotations.json"
    cat_names = {}
    if ann_path.exists():
        with open(ann_path) as f:
            coco = _json.load(f)
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    for cid, acc, n in worst[:10]:
        name = cat_names.get(cid, f"cat_{cid}")
        print(f"  cat {cid} ({name}): {acc:.0%} ({per_class_correct[cid]}/{n})")


if __name__ == "__main__":
    main()
