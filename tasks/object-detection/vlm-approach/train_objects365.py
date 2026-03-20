"""
Pre-train MarkusNet-860M (pruned Qwen3.5-0.8B, 12 text layers) on COCO detection.

Uses detection-datasets/coco from HuggingFace (has embedded images, 80 categories).
Crops annotated objects and trains the classification head. After pre-training,
we swap the head to 356 classes and fine-tune on competition data.

The backbone learns general visual features from diverse object crops, which
transfers well to grocery product classification.

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_objects365.py
"""

import functools
import math
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
import wandb

print = functools.partial(print, flush=True)

# === CONFIG ===
PRUNED_DIR = Path(__file__).parent / "pruned"
OUTPUT_DIR = Path(__file__).parent / "training_output_objects365"

# COCO has 80 categories (0-79)
NUM_CLASSES = 80
BATCH_SIZE = 8
LR = 1e-4
NUM_EPOCHS = 3  # COCO is ~118K images, ~860K objects - multiple epochs useful
WARMUP_STEPS = 500
LOG_EVERY = 10
SAVE_EVERY = 5000
MIN_CROP_SIZE = 16  # Minimum crop dimension in pixels


# === Classification Head ===
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


# === Streaming Dataset ===
class COCOCropStreamDataset(IterableDataset):
    """Streams COCO detection dataset and yields individual object crops."""

    def __init__(self, hf_dataset_stream):
        self.stream = hf_dataset_stream

    def __iter__(self):
        for row in self.stream:
            image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            objects = row["objects"]
            w, h = image.size
            bboxes = objects["bbox"]
            categories = objects["category"]

            for j in range(len(categories)):
                bbox = bboxes[j]
                category = categories[j]

                # detection-datasets/coco uses [x_min, y_min, x_max, y_max]
                x1_f, y1_f, x2_f, y2_f = bbox
                x1 = max(0, int(x1_f))
                y1 = max(0, int(y1_f))
                x2 = min(w, int(x2_f))
                y2 = min(h, int(y2_f))

                if x2 - x1 < MIN_CROP_SIZE or y2 - y1 < MIN_CROP_SIZE:
                    continue  # Skip tiny crops

                crop = image.crop((x1, y1, x2, y2))
                yield {"image": crop, "label": category}


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


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === Load COCO Detection ===
    print("Loading COCO detection dataset from HuggingFace (streaming)...")
    from datasets import load_dataset

    hf_train = load_dataset(
        "detection-datasets/coco",
        split="train",
        streaming=True,
    )
    hf_val = load_dataset(
        "detection-datasets/coco",
        split="val",
        streaming=True,
    )

    train_dataset = COCOCropStreamDataset(hf_train)
    val_dataset = COCOCropStreamDataset(hf_val)

    # COCO has ~118K train images, ~5K val images
    # ~860K object annotations in train, ~36K in val
    estimated_crops_per_epoch = 860_000
    estimated_total_steps = (estimated_crops_per_epoch // BATCH_SIZE) * NUM_EPOCHS
    print(f"Estimated ~{estimated_crops_per_epoch} crops/epoch, ~{estimated_total_steps} total steps")

    # === Init wandb ===
    wandb.init(
        project="nmiai-objdet",
        name="markusnet-coco-pretrain",
        config={
            "model": "Qwen3.5-0.8B-pruned-12layers",
            "dataset": "COCO-detection",
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": NUM_EPOCHS,
            "num_classes": NUM_CLASSES,
            "estimated_crops_per_epoch": estimated_crops_per_epoch,
        },
    )

    # === Load model ===
    print("Loading pruned Qwen3.5 (12 layers)...")
    pruned_dir = str(PRUNED_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        pruned_dir,
        dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    model = model.to(device)
    hidden_size = model.config.text_config.hidden_size
    print(f"Hidden size: {hidden_size}")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # === Classification head ===
    cls_head = ClassificationHead(hidden_size, NUM_CLASSES).to(device).to(torch.bfloat16)

    backbone_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in cls_head.parameters())
    print(f"Backbone: {backbone_params/1e6:.1f}M | Cls head: {head_params/1e6:.1f}M")
    print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # === DataLoader ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    # === Optimizer ===
    # Lower LR for backbone, higher for head
    param_groups = [
        {"params": list(model.parameters()), "lr": LR * 0.1},  # backbone: 1e-5
        {"params": list(cls_head.parameters()), "lr": LR},      # head: 1e-4
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, estimated_total_steps - WARMUP_STEPS)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * min(1, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # === Training loop ===
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    cls_head.train()
    global_step = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    best_val_acc = 0.0

    print(f"\n=== Starting COCO pre-training ===")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    print(f"Backbone LR: {LR*0.1:.1e}, Head LR: {LR:.1e}")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        # Re-create streaming dataset each epoch
        if epoch > 0:
            hf_train = load_dataset(
                "detection-datasets/coco",
                split="train",
                streaming=True,
            )
            train_dataset = COCOCropStreamDataset(hf_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                num_workers=4,
                collate_fn=collate_fn,
                drop_last=True,
                pin_memory=True,
                prefetch_factor=2,
            )

        for batch in train_loader:
            images = batch["images"]
            labels = batch["labels"].to(device)

            # Clamp labels to valid range
            labels = labels.clamp(0, NUM_CLASSES - 1)

            try:
                inputs = process_batch(images, processor, "classify", device)
            except Exception as e:
                print(f"Skipping batch (process error): {e}")
                continue

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.model(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state
                logits = cls_head(hidden)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(cls_head.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()

            # Track metrics
            preds = logits.argmax(dim=-1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.shape[0]
            running_loss += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                acc = running_correct / max(1, running_total)
                lr = scheduler.get_last_lr()[0]
                gpu_gb = torch.cuda.memory_allocated() / 1024**3
                print(
                    f"Step {global_step}/{estimated_total_steps} | "
                    f"loss={avg_loss:.4f} | acc={acc:.3f} | "
                    f"lr={lr:.2e} | gpu={gpu_gb:.1f}GB"
                )
                wandb.log({
                    "train/loss": avg_loss,
                    "train/accuracy": acc,
                    "train/learning_rate": lr,
                    "train/gpu_gb": gpu_gb,
                    "train/epoch": epoch + global_step / max(1, estimated_total_steps / NUM_EPOCHS),
                }, step=global_step)
                running_loss = 0.0
                running_correct = 0
                running_total = 0

            if global_step % SAVE_EVERY == 0:
                # Save checkpoint
                ckpt_path = OUTPUT_DIR / f"checkpoint-{global_step}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "cls_head_state": cls_head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "num_classes": NUM_CLASSES,
                }, ckpt_path / "checkpoint.pt")
                print(f"Saved checkpoint to {ckpt_path}")

                # Run quick validation
                print("Running validation...")
                val_acc = validate(model, cls_head, processor, device)
                print(f"  Val accuracy: {val_acc:.3f} (best: {best_val_acc:.3f})")
                wandb.log({"val/accuracy": val_acc}, step=global_step)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_path = OUTPUT_DIR / "best"
                    best_path.mkdir(exist_ok=True)
                    torch.save({
                        "model_state": model.state_dict(),
                        "cls_head_state": cls_head.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "num_classes": NUM_CLASSES,
                        "hidden_size": hidden_size,
                        "val_acc": val_acc,
                    }, best_path / "best.pt")
                    print(f"  New best model! val_acc={val_acc:.3f}")

                model.train()
                cls_head.train()

                # Clean up old checkpoints (keep last 3)
                ckpts = sorted(
                    OUTPUT_DIR.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[1])
                )
                for old in ckpts[:-3]:
                    shutil.rmtree(old)

        print(f"\n=== Epoch {epoch+1} complete at step {global_step} ===")

    # Final save
    final_path = OUTPUT_DIR / "final"
    final_path.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cls_head_state": cls_head.state_dict(),
        "global_step": global_step,
        "num_classes": NUM_CLASSES,
        "hidden_size": hidden_size,
    }, final_path / "final.pt")
    print(f"Final model saved to {final_path}")

    wandb.finish()
    print("COCO PRE-TRAINING COMPLETE")


@torch.no_grad()
def validate(model, cls_head, processor, device, max_batches=100):
    """Quick validation on COCO val set."""
    from datasets import load_dataset

    model.eval()
    cls_head.eval()

    hf_val = load_dataset(
        "detection-datasets/coco",
        split="val",
        streaming=True,
    )
    val_dataset = COCOCropStreamDataset(hf_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True,
    )

    correct = 0
    total = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        images = batch["images"]
        labels = batch["labels"].to(device).clamp(0, NUM_CLASSES - 1)

        try:
            inputs = process_batch(images, processor, "classify", device)
        except Exception:
            continue

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.model(**inputs, output_hidden_states=True)
            hidden = outputs.last_hidden_state
            logits = cls_head(hidden)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]

    return correct / max(1, total)


if __name__ == "__main__":
    train()
