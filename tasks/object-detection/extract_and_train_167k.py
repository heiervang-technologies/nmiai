"""Extract DINOv2 features from 167K classifier dataset and train probe.

Designed to run on TITAN GPU after V7 finishes.
Usage: CUDA_VISIBLE_DEVICES=0 python3 extract_and_train_167k.py
"""
import json, time, torch, torch.nn as nn, torch.nn.functional as F
import timm, numpy as np
from pathlib import Path
from timm.data import create_transform, resolve_data_config
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DINO_WEIGHTS = Path("/home/me/ht/nmiai/tasks/object-detection/submission-single-model/dinov2_vits14.pth")
DATASET_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/classifier_dataset")
OUTPUT_PROBE = Path("/home/me/ht/nmiai/tasks/object-detection/submission-single-model/linear_probe_167k.pth")
BATCH_SIZE = 256
NUM_CLASSES = 356


class FolderDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform
        root = Path(root)
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            cat_id = int(class_dir.name)
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.samples.append((str(img_path), cat_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, label


def extract_features(model, loader, device, desc=""):
    all_feats, all_labels = [], []
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type=="cuda"):
        for i, (imgs, labels) in enumerate(loader):
            feats = model(imgs.to(device))
            feats = F.normalize(feats.float(), dim=-1)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
            if (i + 1) % 100 == 0:
                print(f"  {desc} [{(i+1)*BATCH_SIZE}/{len(loader.dataset)}] {time.time()-t0:.0f}s")
    return torch.cat(all_feats), torch.cat(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load DINOv2
    print("Loading DINOv2 ViT-S/14...")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    state = torch.load(DINO_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)

    # Load datasets
    print("Loading datasets...")
    train_ds = FolderDataset(DATASET_DIR / "train", transform)
    val_ds = FolderDataset(DATASET_DIR / "val", transform)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=False)

    # Extract features
    print("Extracting train features...")
    train_feats, train_labels = extract_features(model, train_loader, device, "train")
    print(f"Train features: {train_feats.shape}")

    print("Extracting val features...")
    val_feats, val_labels = extract_features(model, val_loader, device, "val")
    print(f"Val features: {val_feats.shape}")

    # Save features for reuse
    torch.save({"embeddings": train_feats, "labels": train_labels},
               DATASET_DIR / "train_features_dino.pth")
    torch.save({"embeddings": val_feats, "labels": val_labels},
               DATASET_DIR / "val_features_dino.pth")

    # Train linear probe with SGD
    print("Training linear probe (30 epochs)...")
    embed_dim = train_feats.shape[1]
    probe = nn.Linear(embed_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    train_feats_d = train_feats.to(device)
    train_labels_d = train_labels.to(device)
    val_feats_d = val_feats.to(device)
    val_labels_d = val_labels.to(device)

    best_acc = 0
    best_state = None
    for epoch in range(30):
        probe.train()
        perm = torch.randperm(len(train_feats_d))
        total_loss, correct = 0, 0
        for start in range(0, len(perm), 4096):
            idx = perm[start:start+4096]
            logits = probe(train_feats_d[idx])
            loss = F.cross_entropy(logits, train_labels_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == train_labels_d[idx]).sum().item()
        scheduler.step()

        probe.eval()
        with torch.inference_mode():
            val_logits = probe(val_feats_d)
            val_acc = (val_logits.argmax(1) == val_labels_d).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {"weight": probe.weight.data.cpu().clone(), "bias": probe.bias.data.cpu().clone()}

        if (epoch+1) % 5 == 0:
            train_acc = correct / len(train_feats)
            print(f"  Epoch {epoch+1}: train={train_acc:.4f} val={val_acc:.4f} best={best_acc:.4f}")

    # Save best probe
    torch.save(best_state, OUTPUT_PROBE)
    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Saved to {OUTPUT_PROBE}")


if __name__ == "__main__":
    main()
