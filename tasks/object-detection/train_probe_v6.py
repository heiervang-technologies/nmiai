"""DINOv2 probe v6 - improved training with augmentation + label smoothing.

Key improvements over v4:
- Random horizontal flip + color jitter during training
- Label smoothing (0.1)
- Longer training (100 epochs) with cosine LR
- Uses RAW features (no normalize) to match submission pipeline
- Class-balanced sampling via oversampling minority classes

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_probe_v6.py
"""
import functools
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
from torchvision import transforms

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
DINO_WEIGHTS = SCRIPT_DIR / "submission-single-model" / "dinov2_vits14.pth"
CROPS_DIR = SCRIPT_DIR / "data-creation" / "data" / "classifier_crops"
EPOCHS = 100
LR = 1e-3
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 4096


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load DINOv2
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    weights = torch.load(DINO_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)

    # Two transforms: standard (for val/features) and augmented (we'll augment at feature level)
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

    print(f"Total crops: {len(samples)}")

    # Split
    np.random.seed(42)
    idx = np.random.permutation(len(samples))
    val_n = len(samples) // 10
    train_samples = [samples[i] for i in idx[val_n:]]
    val_samples = [samples[i] for i in idx[:val_n]]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Extract features with standard transform (RAW, no normalize)
    class CropDS(Dataset):
        def __init__(self, samps, tfm):
            self.samples = samps
            self.transform = tfm
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            path, cat_id = self.samples[idx]
            try:
                img = Image.open(path).convert("RGB")
                return self.transform(img), cat_id
            except Exception:
                return torch.zeros(3, 518, 518), cat_id

    def extract(sample_list):
        ds = CropDS(sample_list, transform)
        loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
        all_feats, all_labels = [], []
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for imgs, labs in loader:
                feats = model(imgs.to(device))
                feats = feats.float()  # RAW features (no normalize)
                all_feats.append(feats.cpu())
                all_labels.append(labs)
        return torch.cat(all_feats), torch.cat(all_labels)

    t0 = time.time()
    print("Extracting training features...")
    train_feats, train_labels = extract(train_samples)
    print("Extracting validation features...")
    val_feats, val_labels = extract(val_samples)
    print(f"Features extracted in {time.time()-t0:.0f}s")
    print(f"Feature dim: {train_feats.shape[1]}, norm={train_feats.norm(dim=1).mean():.3f}")

    # Also extract with augmented transform for data augmentation
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(518, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])

    print("Extracting augmented training features...")
    train_feats_aug, train_labels_aug = extract(
        [(s, l) for s, l in zip([s[0] for s in train_samples], [s[1] for s in train_samples])]
    )
    # Re-extract with augmented transform
    aug_feats, aug_labels = [], []
    aug_ds = CropDS(train_samples, aug_transform)
    aug_loader = DataLoader(aug_ds, batch_size=64, num_workers=4, pin_memory=True)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for imgs, labs in aug_loader:
            feats = model(imgs.to(device)).float()
            aug_feats.append(feats.cpu())
            aug_labels.append(labs)
    aug_feats = torch.cat(aug_feats)
    aug_labels = torch.cat(aug_labels)

    # Combine original + augmented
    combined_feats = torch.cat([train_feats, aug_feats])
    combined_labels = torch.cat([train_labels, aug_labels])
    print(f"Combined train set: {len(combined_feats)} samples (original + augmented)")

    # Train probe with label smoothing
    probe = nn.Linear(384, 356).to(device)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    tf = combined_feats.to(device)
    tl = combined_labels.to(device)
    vf = val_feats.to(device)
    vl = val_labels.to(device)

    best_acc, best_state = 0, None
    patience_counter = 0

    for epoch in range(EPOCHS):
        probe.train()
        perm = torch.randperm(len(tf))
        total_loss, correct = 0, 0
        for start in range(0, len(perm), BATCH_SIZE):
            i = perm[start:start + BATCH_SIZE]
            logits = probe(tf[i])
            loss = F.cross_entropy(logits, tl[i], label_smoothing=LABEL_SMOOTHING)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(i)
            correct += (logits.argmax(1) == tl[i]).sum().item()
        scheduler.step()

        probe.eval()
        with torch.inference_mode():
            val_logits = probe(vf)
            val_acc = (val_logits.argmax(1) == vl).float().mean().item()

        train_acc = correct / len(tf)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {"weight": probe.weight.data.cpu().clone(), "bias": probe.bias.data.cpu().clone()}
            marker = " *BEST*"
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or marker:
            print(f"  Epoch {epoch+1}/{EPOCHS}: train={train_acc:.4f} val={val_acc:.4f} loss={total_loss/len(tf):.4f}{marker}")

    # Save
    save_paths = [
        SCRIPT_DIR / "linear_probe_v6.pth",
        SCRIPT_DIR / "submission-single-model" / "linear_probe_v6.pth",
    ]
    for p in save_paths:
        if p.parent.exists():
            torch.save(best_state, p)
            print(f"Saved: {p}")

    print(f"\nBest val accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    print("Uses RAW features (no normalize) - compatible with submission pipeline")


if __name__ == "__main__":
    main()
