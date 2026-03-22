"""DINOv2 probe v7 - multi-augmentation + MLP head exploration.

v6 got 85.6% val, 0.9219 mAP. This version tries:
- 3x augmentation rounds instead of 1x
- Optional 2-layer MLP head
- Higher weight decay
- Cosine warmup

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_probe_v7.py
"""
import functools
import time
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
EPOCHS = 120
LR = 1e-3
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 4096
N_AUG_ROUNDS = 3  # Multiple augmentation passes


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

    # Augmented transforms with varying strength
    aug_transforms = [
        transforms.Compose([
            transforms.RandomResizedCrop(518, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std']),
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(518, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std']),
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(518, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std']),
        ]),
    ]

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
    np.random.seed(42)
    idx = np.random.permutation(len(samples))
    val_n = len(samples) // 10
    train_samples = [samples[i] for i in idx[val_n:]]
    val_samples = [samples[i] for i in idx[:val_n]]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

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

    def extract(sample_list, tfm):
        ds = CropDS(sample_list, tfm)
        loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
        all_feats, all_labels = [], []
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for imgs, labs in loader:
                feats = model(imgs.to(device)).float()  # RAW features
                all_feats.append(feats.cpu())
                all_labels.append(labs)
        return torch.cat(all_feats), torch.cat(all_labels)

    t0 = time.time()
    print("Extracting original features...")
    train_feats, train_labels = extract(train_samples, transform)
    val_feats, val_labels = extract(val_samples, transform)

    all_feats_list = [train_feats]
    all_labels_list = [train_labels]

    for i, aug_tfm in enumerate(aug_transforms[:N_AUG_ROUNDS]):
        print(f"Extracting augmented features (round {i+1}/{N_AUG_ROUNDS})...")
        af, al = extract(train_samples, aug_tfm)
        all_feats_list.append(af)
        all_labels_list.append(al)

    combined_feats = torch.cat(all_feats_list)
    combined_labels = torch.cat(all_labels_list)
    print(f"Combined: {len(combined_feats)} samples in {time.time()-t0:.0f}s")

    # ===== Train LINEAR probe =====
    print("\n=== Training LINEAR probe (v7) ===")
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

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {"weight": probe.weight.data.cpu().clone(), "bias": probe.bias.data.cpu().clone()}
            marker = " *BEST*"

        if (epoch + 1) % 10 == 0 or marker:
            print(f"  Epoch {epoch+1}/{EPOCHS}: train={correct/len(tf):.4f} val={val_acc:.4f} loss={total_loss/len(tf):.4f}{marker}")

    # Save
    for p in [SCRIPT_DIR / "linear_probe_v7.pth", SCRIPT_DIR / "submission-single-model" / "linear_probe_v7.pth"]:
        if p.parent.exists():
            torch.save(best_state, p)
            print(f"Saved: {p}")
    print(f"LINEAR best val: {best_acc:.4f} ({best_acc*100:.1f}%)")

    # ===== Train MLP probe (2-layer) =====
    print("\n=== Training MLP probe (v7_mlp) ===")
    class MLPProbe(nn.Module):
        def __init__(self, in_dim=384, hidden=512, out_dim=356):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.bn = nn.BatchNorm1d(hidden)
            self.fc2 = nn.Linear(hidden, out_dim)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        def forward(self, x):
            x = F.gelu(self.bn(self.fc1(x)))
            return self.fc2(x)

    mlp = MLPProbe().to(device)
    optimizer2 = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)

    best_mlp_acc, best_mlp_state = 0, None
    for epoch in range(EPOCHS):
        mlp.train()
        perm = torch.randperm(len(tf))
        total_loss, correct = 0, 0
        for start in range(0, len(perm), BATCH_SIZE):
            i = perm[start:start + BATCH_SIZE]
            logits = mlp(tf[i])
            loss = F.cross_entropy(logits, tl[i], label_smoothing=LABEL_SMOOTHING)
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer2.step()
            total_loss += loss.item() * len(i)
            correct += (logits.argmax(1) == tl[i]).sum().item()
        scheduler2.step()

        mlp.eval()
        with torch.inference_mode():
            val_logits = mlp(vf)
            val_acc = (val_logits.argmax(1) == vl).float().mean().item()

        marker = ""
        if val_acc > best_mlp_acc:
            best_mlp_acc = val_acc
            best_mlp_state = mlp.state_dict()
            marker = " *BEST*"

        if (epoch + 1) % 10 == 0 or marker:
            print(f"  Epoch {epoch+1}/{EPOCHS}: train={correct/len(tf):.4f} val={val_acc:.4f} loss={total_loss/len(tf):.4f}{marker}")

    # Save MLP
    for p in [SCRIPT_DIR / "mlp_probe_v7.pth"]:
        torch.save(best_mlp_state, p)
        print(f"Saved MLP: {p}")
    print(f"MLP best val: {best_mlp_acc:.4f} ({best_mlp_acc*100:.1f}%)")

    print(f"\n=== SUMMARY ===")
    print(f"Linear v7: {best_acc:.4f}")
    print(f"MLP v7:    {best_mlp_acc:.4f}")


if __name__ == "__main__":
    main()
