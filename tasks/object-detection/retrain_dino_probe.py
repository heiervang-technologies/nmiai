"""Retrain DINOv2 linear probe on extended 196K crop dataset.

Extracts DINOv2 ViT-S/14 features from all crops, trains a linear probe,
and saves the result as linear_probe.pth.

Usage: python retrain_dino_probe.py
"""

import json
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

SCRIPT_DIR = Path(__file__).resolve().parent
DINO_WEIGHTS = SCRIPT_DIR / "submission-single-model" / "dinov2_vits14.pth"
CROPS_JSON = SCRIPT_DIR / "data-creation" / "data" / "extra_crops" / "clean_combined_samples.json"
OUTPUT_PROBE = SCRIPT_DIR / "submission-single-model" / "linear_probe_v2.pth"
NUM_CLASSES = 356
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.01


class CropDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s["crop_path"]).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, s["category_id"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load DINOv2
    print("Loading DINOv2 ViT-S/14...")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    state_dict = torch.load(DINO_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)
    embed_dim = 384  # ViT-S

    # Load samples
    print("Loading samples...")
    samples = json.load(open(CROPS_JSON))
    print(f"Total samples: {len(samples)}")

    # Split into train/val (95/5)
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    val_size = len(samples) // 20
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Extract features
    print("Extracting features...")
    t0 = time.time()

    def extract_features(sample_list):
        ds = CropDataset(sample_list, transform)
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

    # Train linear probe
    print(f"Training linear probe ({EPOCHS} epochs, lr={LR})...")
    probe = nn.Linear(embed_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_feats_gpu = train_feats.to(device)
    train_labels_gpu = train_labels.to(device)

    for epoch in range(EPOCHS):
        probe.train()
        # Mini-batch training
        perm = torch.randperm(len(train_feats_gpu))
        total_loss = 0
        correct = 0
        for start in range(0, len(perm), 4096):
            idx = perm[start:start + 4096]
            logits = probe(train_feats_gpu[idx])
            loss = F.cross_entropy(logits, train_labels_gpu[idx])
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

        print(f"  Epoch {epoch+1}/{EPOCHS}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} loss={total_loss/len(train_feats):.4f}")

    # Save probe
    probe_state = {"weight": probe.weight.data.cpu(), "bias": probe.bias.data.cpu()}
    torch.save(probe_state, OUTPUT_PROBE)
    print(f"\nSaved probe to {OUTPUT_PROBE}")
    print(f"Final val accuracy: {val_acc:.4f}")

    # Also save to the main submission dir for easy packaging
    alt_path = SCRIPT_DIR / "linear_probe_196k.pth"
    torch.save(probe_state, alt_path)
    print(f"Also saved to {alt_path}")


if __name__ == "__main__":
    main()
