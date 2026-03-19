"""
Train a linear probe classifier on DINOv2 embeddings from COCO training crops.

This extracts DINOv2 embeddings for all 22.7k annotated product crops,
then trains a lightweight linear classifier (356 classes) on top.

The linear probe provides a supervised classification signal that complements
the nearest-neighbor reference matching.

Usage: python train_linear_probe.py
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from sklearn.model_selection import train_test_split


DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"
COCO_IMAGES = DATA_ROOT / "coco_dataset" / "train" / "images"
OUTPUT_DIR = Path(__file__).parent

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
EMBED_BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 356


def extract_crops_and_labels():
    """Extract crop bboxes and category labels from COCO annotations."""
    with open(COCO_ANNOTATIONS) as f:
        coco = json.load(f)

    # Build image_id -> filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    crops_info = []  # (image_path, bbox, category_id)
    for ann in coco["annotations"]:
        img_file = id_to_file[ann["image_id"]]
        img_path = COCO_IMAGES / img_file
        bbox = ann["bbox"]  # [x, y, w, h] COCO format
        cat_id = ann["category_id"]
        crops_info.append((img_path, bbox, cat_id))

    return crops_info


@torch.no_grad()
def compute_all_embeddings(crops_info, model, transform, device):
    """Extract DINOv2 embeddings for all crops."""
    embeddings = []
    labels = []
    failed = 0

    # Group by image for efficiency (avoid reloading same image)
    from collections import defaultdict
    image_crops = defaultdict(list)
    for img_path, bbox, cat_id in crops_info:
        image_crops[str(img_path)].append((bbox, cat_id))

    total = len(crops_info)
    done = 0

    for img_path_str, crop_list in image_crops.items():
        img_path = Path(img_path_str)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            failed += len(crop_list)
            continue

        batch_tensors = []
        batch_labels = []

        for bbox, cat_id in crop_list:
            x, y, w, h = bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img.width, int(x + w))
            y2 = min(img.height, int(y + h))

            if x2 <= x1 or y2 <= y1:
                failed += 1
                continue

            crop = img.crop((x1, y1, x2, y2))
            tensor = transform(crop)
            batch_tensors.append(tensor)
            batch_labels.append(cat_id)

        # Process in sub-batches
        for i in range(0, len(batch_tensors), EMBED_BATCH_SIZE):
            sub_batch = torch.stack(batch_tensors[i : i + EMBED_BATCH_SIZE]).to(device)
            embs = model(sub_batch)
            embs = F.normalize(embs, dim=-1)
            embeddings.append(embs.cpu())
            labels.extend(batch_labels[i : i + EMBED_BATCH_SIZE])

        done += len(crop_list)
        if done % 2000 < len(crop_list):
            print(f"  Embedded {done}/{total} crops...")

    all_embeddings = torch.cat(embeddings, dim=0)
    all_labels = torch.tensor(labels, dtype=torch.long)
    print(f"Total: {all_embeddings.shape[0]} embeddings, {failed} failed crops")

    return all_embeddings, all_labels


def train_probe(embeddings, labels, device):
    """Train a linear classifier on the embeddings."""
    embed_dim = embeddings.shape[1]

    # Train/val split
    indices = np.arange(len(embeddings))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, stratify=labels.numpy())

    train_emb = embeddings[train_idx].to(device)
    train_lab = labels[train_idx].to(device)
    val_emb = embeddings[val_idx].to(device)
    val_lab = labels[val_idx].to(device)

    # Class weights for imbalanced classes
    class_counts = Counter(labels.numpy().tolist())
    total = sum(class_counts.values())
    weights = torch.zeros(NUM_CLASSES, device=device)
    for c in range(NUM_CLASSES):
        count = class_counts.get(c, 0)
        if count > 0:
            weights[c] = total / (NUM_CLASSES * count)
        else:
            weights[c] = 1.0
    weights = weights.clamp(max=10.0)  # Cap extreme weights

    # Linear probe
    probe = nn.Linear(embed_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_dataset = TensorDataset(train_emb, train_lab)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    best_val_acc = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        # Train
        probe.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_emb, batch_lab in train_loader:
            logits = probe(batch_emb)
            loss = F.cross_entropy(logits, batch_lab, weight=weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_emb.shape[0]
            correct += (logits.argmax(-1) == batch_lab).sum().item()
            total += batch_emb.shape[0]

        scheduler.step()

        train_acc = correct / total
        train_loss = total_loss / total

        # Validate
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_emb)
            val_loss = F.cross_entropy(val_logits, val_lab, weight=weights).item()
            val_acc = (val_logits.argmax(-1) == val_lab).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{NUM_EPOCHS}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DINOv2
    print(f"Loading {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval().to(device)
    data_config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_config, is_training=False)

    # Extract crops
    print("Extracting crop info from COCO annotations...")
    crops_info = extract_crops_and_labels()
    print(f"Total annotations: {len(crops_info)}")

    # Compute embeddings
    print("Computing DINOv2 embeddings for all crops...")
    embeddings, labels = compute_all_embeddings(crops_info, model, transform, device)

    # Save embeddings for reuse
    emb_cache = OUTPUT_DIR / "training_embeddings.pth"
    torch.save({"embeddings": embeddings, "labels": labels}, emb_cache)
    print(f"Cached training embeddings to {emb_cache}")

    # Train linear probe
    print("\nTraining linear probe...")
    best_state = train_probe(embeddings, labels, device)

    # Save
    probe_path = OUTPUT_DIR / "linear_probe.pth"
    torch.save(best_state, probe_path)
    print(f"Saved linear probe to {probe_path} ({probe_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
