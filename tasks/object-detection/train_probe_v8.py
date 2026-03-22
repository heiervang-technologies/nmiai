"""DINOv2 probe v8 - wider MLP + 5x augmentation + longer training.

v7 MLP got 87.5% val, 0.9435 mAP. This version tries:
- Wider MLP (768 hidden instead of 512)
- 5x augmentation rounds
- 150 epochs
- Dropout for regularization

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_probe_v8.py
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
EPOCHS = 150
LR = 1e-3
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 4096
N_AUG_ROUNDS = 5


class MLPProbe(nn.Module):
    def __init__(self, in_dim=384, hidden=768, out_dim=356, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.gelu(self.bn(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    weights = torch.load(DINO_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)

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
        transforms.Compose([
            transforms.RandomResizedCrop(518, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std']),
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(518, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.2),
            transforms.RandomAutocontrast(p=0.2),
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
                feats = model(imgs.to(device)).float()
                all_feats.append(feats.cpu())
                all_labels.append(labs)
        return torch.cat(all_feats), torch.cat(all_labels)

    t0 = time.time()
    print("Extracting original features...")
    train_feats, train_labels = extract(train_samples, transform)
    val_feats, val_labels = extract(val_samples, transform)

    all_feats_list = [train_feats]
    all_labels_list = [train_labels]

    for i in range(N_AUG_ROUNDS):
        print(f"Extracting augmented features (round {i+1}/{N_AUG_ROUNDS})...")
        af, al = extract(train_samples, aug_transforms[i % len(aug_transforms)])
        all_feats_list.append(af)
        all_labels_list.append(al)

    combined_feats = torch.cat(all_feats_list)
    combined_labels = torch.cat(all_labels_list)
    print(f"Combined: {len(combined_feats)} samples in {time.time()-t0:.0f}s")

    # Train MLP probes with different widths
    configs = [
        ("mlp_512", 512, 0.1),
        ("mlp_768", 768, 0.1),
        ("mlp_1024", 1024, 0.15),
    ]

    tf = combined_feats.to(device)
    tl = combined_labels.to(device)
    vf = val_feats.to(device)
    vl = val_labels.to(device)

    best_overall_acc = 0
    best_overall_state = None
    best_overall_name = ""

    for name, hidden, dropout in configs:
        print(f"\n=== Training {name} (hidden={hidden}, dropout={dropout}) ===")
        mlp = MLPProbe(hidden=hidden, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_acc, best_state = 0, None
        for epoch in range(EPOCHS):
            mlp.train()
            perm = torch.randperm(len(tf))
            total_loss, correct = 0, 0
            for start in range(0, len(perm), BATCH_SIZE):
                i = perm[start:start + BATCH_SIZE]
                logits = mlp(tf[i])
                loss = F.cross_entropy(logits, tl[i], label_smoothing=LABEL_SMOOTHING)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(i)
                correct += (logits.argmax(1) == tl[i]).sum().item()
            scheduler.step()

            mlp.eval()
            with torch.inference_mode():
                val_logits = mlp(vf)
                val_acc = (val_logits.argmax(1) == vl).float().mean().item()

            marker = ""
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                marker = " *BEST*"

            if (epoch + 1) % 15 == 0 or marker:
                print(f"  Epoch {epoch+1}/{EPOCHS}: train={correct/len(tf):.4f} val={val_acc:.4f}{marker}")

        print(f"{name} best val: {best_acc:.4f} ({best_acc*100:.1f}%)")

        # Save individual
        torch.save(best_state, SCRIPT_DIR / f"{name}_probe_v8.pth")

        if best_acc > best_overall_acc:
            best_overall_acc = best_acc
            best_overall_state = best_state
            best_overall_name = name

    # Deploy best
    print(f"\n=== BEST: {best_overall_name} at {best_overall_acc:.4f} ===")
    for p in [SCRIPT_DIR / "mlp_probe_v8.pth", SCRIPT_DIR / "submission-single-model" / "mlp_probe_v8.pth"]:
        if p.parent.exists():
            torch.save(best_overall_state, p)
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()
