"""
Pre-train MarkusNet-860M (pruned Qwen3.5-0.8B, 12 layers) on COCO-minitrain.

COCO-minitrain is a 25k image subset of COCO train2017 with 80 categories
and ~180k annotations. We extract object crops and train a classification head
on these 80 categories. The backbone weights then transfer to our 356 grocery
categories (only the classification head needs replacement).

Steps:
  1. Download COCO-minitrain annotations + images (from HuggingFace or Google Drive)
  2. Extract object crops from bounding box annotations, cache to disk
  3. Train classification head on 80 COCO categories
  4. Save checkpoint for transfer learning

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_coco_minitrain.py
"""

import functools
import json
import math
import os
import random
import subprocess
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

import wandb

print = functools.partial(print, flush=True)

# === CONFIG ===
PRUNED_DIR = Path(__file__).parent / "pruned"
COCO_MINITRAIN_DIR = Path(__file__).parent / "external_datasets" / "coco_minitrain"
CROP_CACHE_DIR = COCO_MINITRAIN_DIR / "crops"
SAMPLES_JSON = COCO_MINITRAIN_DIR / "samples.json"
OUTPUT_DIR = Path(__file__).parent / "training_output_coco_pretrain"

NUM_COCO_CLASSES = 80
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 3
WARMUP_STEPS = 300
LOG_EVERY = 20
SAVE_EVERY = 1000
MIN_CROP_SIZE = 16  # minimum crop dimension in pixels


# === COCO category id mapping ===
# COCO uses non-contiguous category IDs (1-90 with gaps).
# We remap to contiguous 0-79 for the classification head.
COCO_80_CATEGORIES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
COCO_ID_TO_CONTIGUOUS = {cid: i for i, cid in enumerate(COCO_80_CATEGORIES)}


# === Download & Prepare ===

def download_coco_minitrain():
    """Download COCO-minitrain from HuggingFace and extract."""
    COCO_MINITRAIN_DIR.mkdir(parents=True, exist_ok=True)

    annotations_file = COCO_MINITRAIN_DIR / "instances_minitrain2017.json"
    images_dir = COCO_MINITRAIN_DIR / "images"

    if annotations_file.exists() and images_dir.exists() and any(images_dir.iterdir()):
        print(f"COCO-minitrain already downloaded at {COCO_MINITRAIN_DIR}")
        return annotations_file, images_dir

    zip_path = COCO_MINITRAIN_DIR / "coco_minitrain_25k.zip"

    if not zip_path.exists():
        print("Downloading COCO-minitrain from HuggingFace...")
        # Try huggingface-cli first, fall back to wget
        hf_url = "https://huggingface.co/datasets/bryanbocao/coco_minitrain/resolve/main/coco_minitrain_25k.zip"
        try:
            subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(zip_path), hf_url],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try curl as fallback
            subprocess.run(
                ["curl", "-L", "-o", str(zip_path), hf_url],
                check=True,
            )
        print(f"Downloaded to {zip_path}")

    # Extract
    print("Extracting COCO-minitrain...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(COCO_MINITRAIN_DIR)

    # The zip might have a nested directory structure — find the annotation file
    # Common structures: coco_minitrain_25k/annotations/instances_minitrain2017.json
    # or annotations directly at root
    if not annotations_file.exists():
        # Search for the annotation file
        found = list(COCO_MINITRAIN_DIR.rglob("instances_minitrain2017.json"))
        if not found:
            # Try alternate names
            found = list(COCO_MINITRAIN_DIR.rglob("*.json"))
            found = [f for f in found if "instance" in f.name.lower() or "train" in f.name.lower()]

        if not found:
            # List what we got
            print("Contents after extraction:")
            for p in sorted(COCO_MINITRAIN_DIR.rglob("*"))[:30]:
                print(f"  {p.relative_to(COCO_MINITRAIN_DIR)}")
            raise FileNotFoundError(
                "Could not find COCO-minitrain annotation JSON. "
                "Check the extracted contents above."
            )

        # Move/symlink to expected location
        src = found[0]
        if src != annotations_file:
            print(f"Found annotations at: {src}")
            annotations_file = src

    # Find images directory
    if not images_dir.exists() or not any(images_dir.iterdir()):
        # Search for a directory with images
        for candidate in COCO_MINITRAIN_DIR.rglob("*.jpg"):
            images_dir = candidate.parent
            break
        else:
            # Images might be from COCO train2017 — we need to download them
            print("Images not found in zip. Will download from COCO train2017.")
            images_dir = download_coco_images(annotations_file)

    print(f"Annotations: {annotations_file}")
    print(f"Images dir: {images_dir}")
    n_images = sum(1 for _ in images_dir.glob("*.jpg"))
    print(f"Found {n_images} images")

    return annotations_file, images_dir


def download_coco_images(annotations_file):
    """Download COCO train2017 images that appear in minitrain annotations."""
    images_dir = COCO_MINITRAIN_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    # Load annotations to get image file names
    with open(annotations_file) as f:
        coco = json.load(f)

    image_filenames = {img["file_name"] for img in coco["images"]}
    print(f"Need {len(image_filenames)} images from COCO train2017")

    # Download COCO train2017 images zip (18GB) — this is large
    # Better approach: download individual images via COCO URL
    base_url = "http://images.cocodataset.org/train2017/"

    already = set(f.name for f in images_dir.glob("*.jpg"))
    to_download = image_filenames - already
    print(f"Already have {len(already)}, need to download {len(to_download)}")

    if to_download:
        print("Downloading individual images from COCO...")
        # Use wget in parallel for speed
        url_file = COCO_MINITRAIN_DIR / "download_urls.txt"
        with open(url_file, "w") as f:
            for fname in sorted(to_download):
                f.write(f"{base_url}{fname}\n")

        try:
            subprocess.run(
                [
                    "wget", "-q", "-P", str(images_dir),
                    "-i", str(url_file),
                    "--no-clobber",
                    "-j", "8",  # 8 parallel downloads
                ],
                check=True,
                timeout=7200,  # 2 hour timeout
            )
        except subprocess.CalledProcessError:
            # wget returns non-zero if some files already exist, that's fine
            pass

        url_file.unlink(missing_ok=True)

    n_images = sum(1 for _ in images_dir.glob("*.jpg"))
    print(f"Have {n_images}/{len(image_filenames)} images")
    return images_dir


def extract_crops(annotations_file, images_dir):
    """Extract object crops from COCO annotations and cache to disk.

    Returns list of dicts: [{"crop_path": str, "category_name": str, "category_id": int}, ...]
    """
    if SAMPLES_JSON.exists():
        print(f"Loading cached samples from {SAMPLES_JSON}")
        with open(SAMPLES_JSON) as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} cached crops")
        return samples

    print("Extracting crops from COCO annotations...")
    CROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(annotations_file) as f:
        coco = json.load(f)

    # Build lookups
    id_to_image = {img["id"]: img for img in coco["images"]}
    id_to_category = {cat["id"]: cat["name"] for cat in coco["categories"]}

    samples = []
    skipped = 0
    crop_idx = 0

    # Group annotations by image for efficiency
    from collections import defaultdict
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    total_images = len(anns_by_image)
    for img_idx, (image_id, anns) in enumerate(anns_by_image.items()):
        img_info = id_to_image.get(image_id)
        if img_info is None:
            continue

        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            skipped += len(anns)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Error opening {img_path}: {e}")
            skipped += len(anns)
            continue

        w, h = img.size

        for ann in anns:
            # Skip crowd annotations
            if ann.get("iscrowd", 0):
                skipped += 1
                continue

            # COCO bbox format: [x, y, width, height]
            bx, by, bw, bh = ann["bbox"]
            x1 = max(0, int(bx))
            y1 = max(0, int(by))
            x2 = min(w, int(bx + bw))
            y2 = min(h, int(by + bh))

            # Skip tiny crops
            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                skipped += 1
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop_path = CROP_CACHE_DIR / f"{crop_idx}.jpg"
            crop.save(crop_path, "JPEG", quality=90)

            coco_cat_id = ann["category_id"]
            contiguous_id = COCO_ID_TO_CONTIGUOUS.get(coco_cat_id)
            if contiguous_id is None:
                skipped += 1
                continue

            samples.append({
                "crop_path": str(crop_path),
                "category_name": id_to_category.get(coco_cat_id, "unknown"),
                "category_id": contiguous_id,
            })
            crop_idx += 1

        if (img_idx + 1) % 1000 == 0:
            print(f"  Processed {img_idx + 1}/{total_images} images, {crop_idx} crops so far")

    print(f"Extracted {len(samples)} crops, skipped {skipped}")

    # Save samples index
    with open(SAMPLES_JSON, "w") as f:
        json.dump(samples, f)
    print(f"Saved samples index to {SAMPLES_JSON}")

    return samples


# === Classification Head (same as train_pruned_multitask.py) ===
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        return self.head(pooled)


# === Dataset ===
class CropClassificationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        return {"image": crop, "label": s["category_id"]}


def process_batch(images, processor, prompt_text, device):
    """Process a batch of PIL images through the Qwen processor."""
    texts = []
    for img in images:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt_text},
        ]}]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ))

    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def compute_class_weights(samples, num_classes, device):
    """Inverse frequency class weights, clamped."""
    counts = torch.zeros(num_classes)
    for s in samples:
        counts[s["category_id"]] += 1
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * num_classes
    weights = weights.clamp(min=0.1, max=10.0)
    return weights.to(device)


@torch.no_grad()
def validate(model, cls_head, processor, val_samples, device, max_batches=50):
    """Run classification validation on a held-out split."""
    model.eval()
    cls_head.eval()

    val_dataset = CropClassificationDataset(val_samples)
    loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=0,
        collate_fn=lambda batch: {
            "images": [b["image"] for b in batch],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        },
        drop_last=False,
    )

    correct = 0
    total = 0
    top5_correct = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = batch["images"]
        labels = batch["labels"].to(device)

        inputs = process_batch(images, processor, "classify", device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.model(**inputs, output_hidden_states=True)
            hidden = outputs.last_hidden_state
            logits = cls_head(hidden)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        # Top-5 accuracy
        _, top5 = logits.topk(5, dim=-1)
        top5_correct += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        total += labels.shape[0]

    model.train()
    cls_head.train()
    acc = correct / max(1, total)
    top5_acc = top5_correct / max(1, total)
    return acc, top5_acc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Download COCO-minitrain
    print("\n=== Step 1: Download COCO-minitrain ===")
    annotations_file, images_dir = download_coco_minitrain()

    # Step 2: Extract crops
    print("\n=== Step 2: Extract object crops ===")
    all_samples = extract_crops(annotations_file, images_dir)

    # Print class distribution
    from collections import Counter
    class_counts = Counter(s["category_id"] for s in all_samples)
    print(f"Classes with samples: {len(class_counts)}/{NUM_COCO_CLASSES}")
    most_common = class_counts.most_common(5)
    least_common = class_counts.most_common()[-5:]
    print(f"Most common: {[(all_samples[0]['category_name'] if s[0] == all_samples[0]['category_id'] else s[0], s[1]) for s in most_common]}")
    print(f"Least common classes: {least_common}")

    # Train/val split (95/5)
    random.seed(42)
    random.shuffle(all_samples)
    val_size = max(500, len(all_samples) // 20)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Step 3: Initialize model
    print("\n=== Step 3: Load model ===")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nmiai-objdet"),
        name="markusnet-coco-pretrain",
        config={
            "model": "Qwen3.5-0.8B-pruned-12layers",
            "task": "coco-minitrain-pretrain",
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "num_classes": NUM_COCO_CLASSES,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        },
    )

    print("Loading pruned Qwen3.5 (12 layers)...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(PRUNED_DIR),
        dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    model = model.to(device)
    hidden_size = model.config.text_config.hidden_size
    print(f"Hidden size: {hidden_size}")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # Classification head: 80 COCO classes
    cls_head = ClassificationHead(hidden_size, NUM_COCO_CLASSES).to(device).to(torch.bfloat16)

    backbone_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in cls_head.parameters())
    print(f"Backbone: {backbone_params / 1e6:.1f}M | Cls head: {head_params / 1e6:.1f}M")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # Dataset and loader
    train_dataset = CropClassificationDataset(train_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=lambda batch: {
            "images": [b["image"] for b in batch],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        },
        drop_last=True,
        pin_memory=True,
    )

    # Class weights
    class_weights = compute_class_weights(train_samples, NUM_COCO_CLASSES, device)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

    # Optimizer — lower LR for backbone, higher for head
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": LR * 0.1},  # backbone: 1e-5
            {"params": cls_head.parameters(), "lr": LR},       # head: 1e-4
        ],
        weight_decay=0.01,
    )

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda, lr_lambda])

    # Step 4: Train
    print(f"\n=== Step 4: Training {EPOCHS} epochs ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    cls_head.train()
    global_step = 0
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"]
            labels = batch["labels"].to(device)

            inputs = process_batch(images, processor, "classify", device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.model(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state
                logits = cls_head(hidden)
                loss = F.cross_entropy(logits, labels, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(cls_head.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.shape[0]
            epoch_loss += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                acc = epoch_correct / max(1, epoch_total)
                lr_backbone = scheduler.get_last_lr()[0]
                lr_head = scheduler.get_last_lr()[1]
                gpu_mb = torch.cuda.memory_allocated() / 1024**2
                print(
                    f"[E{epoch+1}] Step {global_step}/{total_steps} | "
                    f"loss={loss.item():.4f} avg={avg_loss:.4f} | "
                    f"acc={acc:.3f} | lr_bb={lr_backbone:.2e} lr_hd={lr_head:.2e} | "
                    f"gpu={gpu_mb:.0f}MB"
                )
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/accuracy": acc,
                    "train/lr_backbone": lr_backbone,
                    "train/lr_head": lr_head,
                    "train/gpu_mb": gpu_mb,
                    "train/epoch": epoch + batch_idx / steps_per_epoch,
                }, step=global_step)

            if global_step % SAVE_EVERY == 0:
                ckpt_path = OUTPUT_DIR / f"checkpoint-{global_step}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "cls_head_state": cls_head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }, ckpt_path / "checkpoint.pt")
                print(f"Saved checkpoint to {ckpt_path}")

        # End of epoch — validate
        train_acc = epoch_correct / max(1, epoch_total)
        avg_loss = epoch_loss / max(1, len(train_loader))

        print(f"\n--- Validating epoch {epoch + 1} ---")
        val_acc, val_top5 = validate(model, cls_head, processor, val_samples, device)

        print(
            f"\n=== Epoch {epoch + 1}/{EPOCHS}: "
            f"loss={avg_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} val_top5={val_top5:.3f} ===\n"
        )

        wandb.log({
            "epoch/loss": avg_loss,
            "epoch/train_acc": train_acc,
            "epoch/val_acc": val_acc,
            "epoch/val_top5_acc": val_top5,
            "epoch/number": epoch + 1,
        }, step=global_step)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = OUTPUT_DIR / "best"
            best_path.mkdir(exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "cls_head_state": cls_head.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_top5": val_top5,
                "train_acc": train_acc,
                "num_classes": NUM_COCO_CLASSES,
                "hidden_size": hidden_size,
            }, best_path / "best_coco_pretrain.pt")
            print(f"New best model saved (val_acc={val_acc:.3f}, top5={val_top5:.3f})")

    # Final save
    final_path = OUTPUT_DIR / "final"
    final_path.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cls_head_state": cls_head.state_dict(),
        "global_step": global_step,
        "num_classes": NUM_COCO_CLASSES,
        "hidden_size": hidden_size,
    }, final_path / "final_coco_pretrain.pt")
    print(f"Final model saved to {final_path}")

    # Save backbone-only checkpoint for easy transfer
    backbone_path = OUTPUT_DIR / "backbone_only.pt"
    torch.save({
        "model_state": model.state_dict(),
        "hidden_size": hidden_size,
        "pretrained_on": "coco-minitrain-80",
        "epochs": EPOCHS,
        "best_val_acc": best_val_acc,
    }, backbone_path)
    print(f"Backbone-only checkpoint saved to {backbone_path}")
    print("Use this to initialize grocery classification (replace cls head 80 -> 356)")

    wandb.finish()
    print("\nPRE-TRAINING COMPLETE")
    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Backbone checkpoint: {backbone_path}")


if __name__ == "__main__":
    train()
