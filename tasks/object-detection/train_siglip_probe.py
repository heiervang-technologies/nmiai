"""Fine-tune SigLIP classification head on classifier crops.

Phase 1: Freeze vision encoder, train only the classification head.
Phase 2: Unfreeze vision encoder with lower LR.

Classification head weights initialized from pre-computed text embeddings.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).parent
TEXT_EMBED_PATH = SCRIPT_DIR / "text_embeddings.pth"
CROPS_DIR = SCRIPT_DIR / "data-creation/data/classifier_crops"

NUM_CLASSES = 356
ALIASES = {59: 61, 170: 260, 36: 201}
# Reverse aliases for training: map alias targets back
ALIAS_TARGETS = set(ALIASES.values())


class CropDataset(Dataset):
    def __init__(self, samples, transform, augment=False):
        self.samples = samples
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        # Apply aliases: map rare IDs to common IDs
        label = ALIASES.get(label, label)
        return img, label


class SigLIPClassifier(nn.Module):
    def __init__(self, vision_model, text_embeddings, temperature=1.0):
        super().__init__()
        self.vision = vision_model
        embed_dim = vision_model.num_features
        self.head = nn.Linear(embed_dim, NUM_CLASSES, bias=False)
        # Initialize head with text embeddings
        if text_embeddings is not None:
            assert text_embeddings.shape == (NUM_CLASSES, embed_dim)
            self.head.weight.data.copy_(text_embeddings)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(temperature)))

    def forward(self, x, normalize=True):
        features = self.vision(x)
        if normalize:
            features = F.normalize(features, dim=-1)
            # Normalize head weights too for cosine similarity
            weight = F.normalize(self.head.weight, dim=-1)
            logits = features @ weight.T * self.logit_scale.exp()
        else:
            logits = self.head(features)
        return logits


def load_dataset(crops_dir):
    samples = []
    for cat_dir in sorted(crops_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_id = int(cat_dir.name)
        for img_path in sorted(cat_dir.glob("*.jpg")):
            samples.append((img_path, cat_id))
    return samples


def split_dataset(samples, val_ratio=0.1, seed=42):
    """Stratified split by category."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)

    train, val = [], []
    for cat_id, cat_samples in by_class.items():
        rng.shuffle(cat_samples)
        n_val = max(1, int(len(cat_samples) * val_ratio))
        val.extend(cat_samples[:n_val])
        train.extend(cat_samples[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.inference_mode()
def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
        preds = logits.argmax(dim=-1)
        # Apply aliases to both
        for i in range(len(labels)):
            gt = ALIASES.get(labels[i].item(), labels[i].item())
            pred = ALIASES.get(preds[i].item(), preds[i].item())
            correct += (gt == pred)
            total += 1
    return correct / total if total > 0 else 0.0


def train(
    model_name="vit_so400m_patch14_siglip_384",
    epochs_frozen=5,
    epochs_unfrozen=3,
    lr_head=1e-3,
    lr_vision=1e-5,
    batch_size=32,
    val_ratio=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load text embeddings for head initialization
    text_data = torch.load(TEXT_EMBED_PATH, map_location="cpu", weights_only=False)
    text_embeds = text_data["ensemble_embeddings"].float()
    print(f"Text embeddings: {text_embeds.shape}")

    # Load vision model
    print(f"Loading {model_name}...")
    vision_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    embed_dim = vision_model.num_features
    print(f"Embed dim: {embed_dim}")

    # Build classifier
    classifier = SigLIPClassifier(vision_model, text_embeds, temperature=100.0)
    classifier = classifier.to(device)

    # Get transforms
    config = resolve_data_config(vision_model.pretrained_cfg)
    train_transform = create_transform(**config, is_training=True)
    val_transform = create_transform(**config, is_training=False)

    # Load and split dataset
    all_samples = load_dataset(CROPS_DIR)
    train_samples, val_samples = split_dataset(all_samples, val_ratio)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = CropDataset(train_samples, train_transform)
    val_ds = CropDataset(val_samples, val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Phase 1: Frozen vision encoder
    print(f"\n{'='*60}")
    print(f"Phase 1: Frozen vision, training head only ({epochs_frozen} epochs)")
    print(f"{'='*60}")

    for param in classifier.vision.parameters():
        param.requires_grad = False
    classifier.head.weight.requires_grad = True
    classifier.logit_scale.requires_grad = True

    optimizer = torch.optim.AdamW(
        [{"params": [classifier.head.weight], "lr": lr_head},
         {"params": [classifier.logit_scale], "lr": lr_head * 0.1}],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_frozen * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc = 0.0
    for epoch in range(epochs_frozen):
        classifier.train()
        total_loss = 0
        t0 = time.time()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = classifier(images)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate_model(classifier, val_loader, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs_frozen}: loss={avg_loss:.4f} val_acc={100*val_acc:.2f}% [{elapsed:.1f}s]")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "head_weight": classifier.head.weight.data.cpu(),
                "logit_scale": classifier.logit_scale.data.cpu(),
                "model_name": model_name,
                "phase": "frozen",
                "val_acc": val_acc,
            }, SCRIPT_DIR / "siglip_head_frozen.pth")
            print(f"  -> Saved best frozen head (acc={100*val_acc:.2f}%)")

    # Phase 2: Unfreeze vision encoder
    if epochs_unfrozen > 0:
        print(f"\n{'='*60}")
        print(f"Phase 2: Unfrozen vision + head ({epochs_unfrozen} epochs)")
        print(f"{'='*60}")

        for param in classifier.vision.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW([
            {"params": classifier.vision.parameters(), "lr": lr_vision},
            {"params": [classifier.head.weight], "lr": lr_head * 0.1},
            {"params": [classifier.logit_scale], "lr": lr_head * 0.01},
        ], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_unfrozen * len(train_loader))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        for epoch in range(epochs_unfrozen):
            classifier.train()
            total_loss = 0
            t0 = time.time()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = classifier(images)
                    loss = F.cross_entropy(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_acc = evaluate_model(classifier, val_loader, device)
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1}/{epochs_unfrozen}: loss={avg_loss:.4f} val_acc={100*val_acc:.2f}% [{elapsed:.1f}s]")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(classifier.state_dict(), SCRIPT_DIR / "siglip_classifier_best.pth")
                torch.save({
                    "head_weight": classifier.head.weight.data.cpu(),
                    "logit_scale": classifier.logit_scale.data.cpu(),
                    "model_name": model_name,
                    "phase": "unfrozen",
                    "val_acc": val_acc,
                }, SCRIPT_DIR / "siglip_head_unfrozen.pth")
                print(f"  -> Saved best unfrozen model (acc={100*val_acc:.2f}%)")

    print(f"\nBest validation accuracy: {100*best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_so400m_patch14_siglip_384")
    parser.add_argument("--epochs-frozen", type=int, default=5)
    parser.add_argument("--epochs-unfrozen", type=int, default=3)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-vision", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(
        model_name=args.model,
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
        lr_head=args.lr_head,
        lr_vision=args.lr_vision,
        batch_size=args.batch_size,
    )
