"""
Continue training pruned Qwen3.5 from best checkpoint.
Adds class-weighted loss + label smoothing to push past 90% plateau toward 95%.

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_continue.py
"""

import json
import math
import functools
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import wandb

print = functools.partial(print, flush=True)

# === CONFIG ===
CHECKPOINT = Path(__file__).parent / "training_output" / "best" / "best.pt"
PRUNED_DIR = Path(__file__).parent / "pruned"
DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"
CROP_CACHE = Path(__file__).parent / "cached_dataset" / "crops"
SAMPLES_CACHE = Path(__file__).parent / "cached_dataset" / "samples.json"
OUTPUT_DIR = Path(__file__).parent / "training_output"

NUM_CLASSES = 356
BATCH_SIZE = 8
LR = 3e-5          # Lower LR for fine-tuning from checkpoint
EPOCHS = 10         # More epochs to push accuracy
WARMUP_STEPS = 50
LOG_EVERY = 10
SAVE_EVERY = 500
LABEL_SMOOTHING = 0.1


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


class CropDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        return {"image": crop, "label": s["category_id"]}


def compute_class_weights(samples):
    """Compute inverse-frequency class weights."""
    counts = Counter(s["category_id"] for s in samples)
    total = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        count = counts.get(c, 0)
        if count > 0:
            weights[c] = total / (NUM_CLASSES * count)
        else:
            weights[c] = 1.0
    weights = weights.clamp(max=10.0)  # Cap extreme weights
    return weights


def train():
    device = torch.device("cuda")
    print(f"Device: {device}")

    wandb.init(
        project="nmiai-objdet",
        name="qwen35-pruned12-continue-v2",
        config={
            "model": "Qwen3.5-0.8B-pruned-12layers",
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "label_smoothing": LABEL_SMOOTHING,
            "resume_from": str(CHECKPOINT),
        },
    )

    # Load model
    print("Loading pruned model...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(PRUNED_DIR),
        dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    hidden_size = model.config.text_config.hidden_size

    cls_head = ClassificationHead(hidden_size, NUM_CLASSES).to(device).to(torch.bfloat16)

    # Load checkpoint
    print(f"Loading checkpoint from {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    cls_head.load_state_dict(ckpt["cls_head_state"])
    prev_acc = ckpt.get("accuracy", 0)
    prev_step = ckpt.get("global_step", 0)
    print(f"Resumed from step {prev_step}, accuracy {prev_acc:.3f}")

    model = model.to(device)

    # Load data
    with open(SAMPLES_CACHE) as f:
        samples = json.load(f)
    print(f"Dataset: {len(samples)} samples")

    # Class weights
    class_weights = compute_class_weights(samples).to(device).to(torch.bfloat16)
    print(f"Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}, mean={class_weights.mean():.2f}")

    dataset = CropDataset(samples)

    def collate(batch):
        return {
            "images": [b["image"] for b in batch],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        collate_fn=collate, drop_last=True)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total")

    all_params = list(model.parameters()) + list(cls_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    cls_head.train()
    global_step = 0
    best_acc = prev_acc

    print(f"\n=== Continuing training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR} ===")
    print(f"=== Label smoothing={LABEL_SMOOTHING}, class-weighted loss ===\n")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, batch in enumerate(loader):
            images = batch["images"]
            labels = batch["labels"].to(device)

            texts = []
            for img in images:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "classify"},
                ]}]
                texts.append(processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                ))

            inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.model(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state
                logits = cls_head(hidden)
                loss = F.cross_entropy(logits, labels, weight=class_weights,
                                       label_smoothing=LABEL_SMOOTHING)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
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
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/accuracy": acc,
                    "train/learning_rate": lr,
                }, step=prev_step + global_step)

            if global_step % SAVE_EVERY == 0 and acc > best_acc:
                best_acc = acc
                best_path = OUTPUT_DIR / "best"
                best_path.mkdir(exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "cls_head_state": cls_head.state_dict(),
                    "global_step": prev_step + global_step,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "accuracy": acc,
                }, best_path / "best.pt")
                print(f"New best: acc={acc:.3f}")

        avg_loss = epoch_loss / steps_per_epoch
        acc = epoch_correct / max(1, epoch_total)
        print(f"\n=== Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} acc={acc:.3f} ===\n")
        wandb.log({"epoch/loss": avg_loss, "epoch/accuracy": acc}, step=prev_step + global_step)

    # Final save
    torch.save({
        "model_state": model.state_dict(),
        "cls_head_state": cls_head.state_dict(),
        "global_step": prev_step + global_step,
        "accuracy": acc,
    }, OUTPUT_DIR / "final" / "final.pt")

    wandb.finish()
    print(f"DONE. Best accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    train()
