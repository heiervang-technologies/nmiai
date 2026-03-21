"""DINOv2 probe v5 - trained WITH normalized features (matches submission pipeline).

v4 got 84.9% but used raw features. The submission run.py normalizes features.
This v5 trains with normalized features so the probe is a drop-in replacement.

Usage: CUDA_VISIBLE_DEVICES=1 python retrain_dino_probe_v5_normalized.py
"""
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


class CropDS(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cat_id = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), cat_id
        except Exception:
            return torch.zeros(3, 518, 518), cat_id


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load DINOv2
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    weights = torch.load(DINO_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)

    # Load crops
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

    print(f"Crops: {len(samples)}")
    np.random.seed(42)
    idx = np.random.permutation(len(samples))
    val_n = len(samples) // 10
    train_samples = [samples[i] for i in idx[val_n:]]
    val_samples = [samples[i] for i in idx[:val_n]]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Extract NORMALIZED features (matching submission pipeline)
    def extract(sample_list):
        ds = CropDS(sample_list, transform)
        loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
        all_feats, all_labels = [], []
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for imgs, labs in loader:
                feats = model(imgs.to(device))
                feats = F.normalize(feats.float(), dim=-1)  # NORMALIZED like submission
                all_feats.append(feats.cpu())
                all_labels.append(labs)
        return torch.cat(all_feats), torch.cat(all_labels)

    t0 = time.time()
    train_feats, train_labels = extract(train_samples)
    val_feats, val_labels = extract(val_samples)
    print(f"Features extracted in {time.time()-t0:.0f}s")
    print(f"Feature stats: norm={train_feats.norm(dim=1).mean():.3f}, std={train_feats.std():.4f}")

    # Train probe - Adam + Xavier (from v4 fix)
    probe = nn.Linear(384, 356).to(device)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    tf = train_feats.to(device)
    tl = train_labels.to(device)
    best_acc, best_state = 0, None

    for epoch in range(60):
        probe.train()
        perm = torch.randperm(len(tf))
        total_loss, correct = 0, 0
        for start in range(0, len(perm), 4096):
            i = perm[start:start + 4096]
            logits = probe(tf[i])
            loss = F.cross_entropy(logits, tl[i])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(i)
            correct += (logits.argmax(1) == tl[i]).sum().item()
        scheduler.step()

        probe.eval()
        with torch.inference_mode():
            val_logits = probe(val_feats.to(device))
            val_acc = (val_logits.argmax(1) == val_labels.to(device)).float().mean().item()

        train_acc = correct / len(tf)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {"weight": probe.weight.data.cpu().clone(), "bias": probe.bias.data.cpu().clone()}
            marker = " *BEST*"

        if (epoch + 1) % 5 == 0 or marker:
            print(f"  Epoch {epoch+1}/60: train={train_acc:.4f} val={val_acc:.4f} loss={total_loss/len(tf):.4f}{marker}")

    # Save to submission directories
    save_paths = [
        SCRIPT_DIR / "linear_probe_v5_normalized.pth",
        SCRIPT_DIR / "submission-single-model" / "linear_probe_v5.pth",
        SCRIPT_DIR / "submission-ensemble" / "linear_probe_v5.pth",
    ]
    for p in save_paths:
        if p.parent.exists():
            torch.save(best_state, p)
            print(f"Saved: {p}")

    print(f"\nBest val accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    print("This probe uses NORMALIZED features - compatible with submission run.py")


if __name__ == "__main__":
    main()
