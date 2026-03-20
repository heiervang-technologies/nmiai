"""
Train Qwen3-VL-2B vision encoder (only) with classification head.
No language model, no text embeddings. Pure vision backbone + cls head.

Uses transformers to load pre-trained vision weights, then trains
with a standard PyTorch loop.

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_qwen3vl_vision.py
"""

import json
import math
import functools
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import wandb

print = functools.partial(print, flush=True)

# === CONFIG ===
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"
CROP_CACHE = Path(__file__).parent / "cached_dataset" / "crops"
SAMPLES_CACHE = Path(__file__).parent / "cached_dataset" / "samples.json"
OUTPUT_DIR = Path(__file__).parent / "training_output_vl2b"

NUM_CLASSES = 356
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 5
WARMUP_STEPS = 100
LOG_EVERY = 10
SAVE_EVERY = 500
LABEL_SMOOTHING = 0.05


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class VisionClassifier(nn.Module):
    """Wraps vision encoder + merger + classification head."""
    def __init__(self, visual, out_hidden, num_classes):
        super().__init__()
        self.visual = visual
        self.cls_head = ClassificationHead(out_hidden, num_classes)

    def forward(self, pixel_values, image_grid_thw):
        # Run through vision encoder + merger
        vision_out = self.visual(pixel_values, grid_thw=image_grid_thw)
        # Pool: mean over all vision tokens
        pooled = vision_out.mean(dim=0) if vision_out.dim() == 2 else vision_out.mean(dim=1)
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        return self.cls_head(pooled)


class CropDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        return {"image": crop, "label": s["category_id"]}


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def train():
    device = torch.device("cuda")
    print(f"Device: {device}")

    wandb.init(
        project="nmiai-objdet",
        name="qwen3vl-2b-vision-only-classify",
        config={
            "model": "Qwen3-VL-2B vision encoder only",
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "num_classes": NUM_CLASSES,
        },
    )

    # Load full model just to extract vision encoder
    print(f"Loading {MODEL_NAME} (extracting vision encoder only)...")
    full_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Extract vision encoder
    visual = full_model.model.visual
    out_hidden = full_model.config.vision_config.out_hidden_size  # 2048
    del full_model  # Free memory
    torch.cuda.empty_cache()

    v_params = sum(p.numel() for p in visual.parameters())
    print(f"Vision encoder: {v_params/1e6:.1f}M params")
    print(f"Output hidden: {out_hidden}")

    # Build classifier
    model = VisionClassifier(visual, out_hidden, NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
    print(f"GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Load data
    with open(SAMPLES_CACHE) as f:
        samples = json.load(f)
    print(f"Dataset: {len(samples)} samples")

    # Class weights
    counts = Counter(s["category_id"] for s in samples)
    total_n = sum(counts.values())
    class_weights = torch.zeros(NUM_CLASSES, device=device, dtype=torch.bfloat16)
    for c in range(NUM_CLASSES):
        count = counts.get(c, 0)
        class_weights[c] = (total_n / (NUM_CLASSES * count)) if count > 0 else 1.0
    class_weights = class_weights.clamp(max=10.0)

    dataset = CropDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, drop_last=True)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    global_step = 0
    best_acc = 0

    print(f"\n=== Training: {EPOCHS} epochs, batch={BATCH_SIZE} ===\n")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, batch in enumerate(loader):
            images = batch["images"]
            labels = batch["labels"].to(device)

            # Process images through Qwen processor (for pixel_values + grid_thw)
            texts = ["classify"] * len(images)
            inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(device)
            image_grid_thw = inputs["image_grid_thw"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixel_values, image_grid_thw)
                # Handle batching: vision encoder may concatenate all patches
                if logits.shape[0] != labels.shape[0]:
                    # Pool per-image if needed
                    logits = logits[:labels.shape[0]]
                loss = F.cross_entropy(logits, labels, weight=class_weights,
                                       label_smoothing=LABEL_SMOOTHING)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            epoch_correct += correct
            epoch_total += labels.shape[0]
            epoch_loss += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                acc = epoch_correct / max(1, epoch_total)
                lr = scheduler.get_last_lr()[0]
                gpu_mb = torch.cuda.memory_allocated() / 1024**2
                print(f"Step {global_step}/{total_steps} | loss={loss.item():.4f} avg={avg_loss:.4f} | acc={acc:.3f} | lr={lr:.2e} | gpu={gpu_mb:.0f}MB")
                wandb.log({"train/loss": loss.item(), "train/avg_loss": avg_loss,
                           "train/accuracy": acc, "train/learning_rate": lr}, step=global_step)

            if global_step % SAVE_EVERY == 0:
                acc = epoch_correct / max(1, epoch_total)
                if acc > best_acc:
                    best_acc = acc
                    best_path = OUTPUT_DIR / "best"
                    best_path.mkdir(exist_ok=True)
                    torch.save({
                        "visual_state": model.visual.state_dict(),
                        "cls_head_state": model.cls_head.state_dict(),
                        "global_step": global_step,
                        "accuracy": acc,
                    }, best_path / "best.pt")
                    print(f"New best: acc={acc:.3f}")

        avg_loss = epoch_loss / steps_per_epoch
        acc = epoch_correct / max(1, epoch_total)
        print(f"\n=== Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} acc={acc:.3f} ===\n")
        wandb.log({"epoch/loss": avg_loss, "epoch/accuracy": acc}, step=global_step)

    # Final save
    final_path = OUTPUT_DIR / "final"
    final_path.mkdir(exist_ok=True)
    torch.save({
        "visual_state": model.visual.state_dict(),
        "cls_head_state": model.cls_head.state_dict(),
        "global_step": global_step,
        "accuracy": best_acc,
    }, final_path / "final.pt")

    wandb.finish()
    print(f"DONE. Best accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    train()
