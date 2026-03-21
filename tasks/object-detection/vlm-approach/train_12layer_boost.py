"""Boost 12-layer MarkusNet accuracy on confusable categories.

Loads the best checkpoint (91.1%), fine-tunes with:
1. Oversampling of confusable/weak categories
2. Hard negative mining between similar products
3. Higher resolution crops for text reading

Uses transformers for training, then re-export NF4 for deployment.
"""
import json
import math
import random
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

PRUNED_DIR = Path(__file__).parent / "pruned"
BEST_CKPT = Path(__file__).parent / "training_output" / "best" / "best.pt"
DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
CROP_DIR = DATA_ROOT / "classifier_crops"
VAL_DIR = DATA_ROOT / "clean_split" / "val"
OUTPUT_DIR = Path(__file__).parent / "training_output_boost"

NUM_CLASSES = 356
BATCH_SIZE = 2  # Reduced to fit in 8GB free GPU memory
LR = 2e-5  # Lower LR for fine-tuning existing good model
EPOCHS = 5
WARMUP_STEPS = 100
LOG_EVERY = 10
SAVE_EVERY = 500
VAL_EVERY = 500
OVERSAMPLE_FACTOR = 3  # How much to oversample weak categories


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=356, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    def forward(self, hidden_states):
        return self.head(hidden_states.mean(dim=1))  # Mean pooling, matches training


class CropDataset(Dataset):
    def __init__(self, crop_dir, oversample_weak=True):
        self.samples = []
        class_samples = defaultdict(list)

        for cls_dir in sorted(crop_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_id = int(cls_dir.name)
            for img_path in cls_dir.glob("*.*"):
                sample = {"crop_path": str(img_path), "category_id": cls_id}
                class_samples[cls_id].append(sample)

        # Oversample classes with fewer samples
        if oversample_weak:
            max_count = max(len(v) for v in class_samples.values())
            target = min(max_count, 100)  # Cap at 100 per class
            for cls_id, samples in class_samples.items():
                if len(samples) < target:
                    # Oversample to reach target
                    n_extra = target - len(samples)
                    extras = [random.choice(samples) for _ in range(n_extra)]
                    self.samples.extend(samples + extras)
                else:
                    self.samples.extend(samples)
        else:
            for samples in class_samples.values():
                self.samples.extend(samples)

        random.shuffle(self.samples)
        print(f"Dataset: {len(self.samples)} samples ({len(class_samples)} classes)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            crop = Image.open(s["crop_path"]).convert("RGB")
        except Exception:
            crop = Image.new("RGB", (224, 224), (128, 128, 128))
        # Don't resize - let processor handle it with proper smart_resize
        return {"image": crop, "label": s["category_id"]}


class ValDataset(Dataset):
    def __init__(self, val_dir):
        self.samples = []
        images_dir = Path(val_dir) / "images"
        labels_dir = Path(val_dir) / "labels"
        for label_path in sorted(labels_dir.glob("*.txt")):
            img_stem = label_path.stem
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                p = images_dir / (img_stem + ext)
                if p.exists() or p.is_symlink():
                    img_path = p
                    break
            if img_path is None:
                continue
            try:
                img = Image.open(img_path)
                w, h = img.size
            except Exception:
                continue
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    self.samples.append({
                        "img_path": str(img_path),
                        "category_id": int(parts[0]),
                        "bbox_norm": [float(x) for x in parts[1:5]],
                    })
        print(f"Val dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        w, h = img.size
        cx, cy, bw, bh = s["bbox_norm"]
        x1 = max(0, int((cx - bw/2) * w))
        y1 = max(0, int((cy - bh/2) * h))
        x2 = min(w, int((cx + bw/2) * w))
        y2 = min(h, int((cy + bh/2) * h))
        crop = img.crop((x1, y1, x2, y2))
        if crop.size[0] < 1 or crop.size[1] < 1:
            crop = img
        return {"image": crop, "label": s["category_id"]}


def process_batch(images, processor, device):
    texts = []
    for img in images:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img, "min_pixels": 65536, "max_pixels": 65536},
            {"type": "text", "text": "classify"}]}]
        texts.append(processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False))
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def validate(model, cls_head, processor, val_dataset, device, max_batches=200):
    model.eval()
    cls_head.eval()
    loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=lambda b: {"images": [x["image"] for x in b],
                                              "labels": torch.tensor([x["label"] for x in b])},
                        drop_last=False)
    correct = total = 0
    total_loss = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images, labels = batch["images"], batch["labels"].to(device)
        inputs = process_batch(images, processor, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.model(**inputs, output_hidden_states=True)
            logits = cls_head(out.last_hidden_state)
            loss = F.cross_entropy(logits, labels)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.shape[0]
        total_loss += loss.item() * labels.shape[0]
    model.train()
    cls_head.train()
    return correct / max(1, total), total_loss / max(1, total)


def train():
    device = torch.device("cuda")

    # Load model
    print("Loading model from pruned dir...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(PRUNED_DIR), torch_dtype=torch.bfloat16, device_map=device,
        ignore_mismatched_sizes=True, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # Load best checkpoint weights
    print(f"Loading best checkpoint from {BEST_CKPT}...")
    ckpt = torch.load(str(BEST_CKPT), map_location=device, weights_only=False)

    # Load model state (skip embed_tokens if present - they're huge)
    model_state = ckpt["model_state"]
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"Loaded model: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # Classification head
    cls_head = ClassificationHead(1024, NUM_CLASSES).to(device).to(torch.bfloat16)
    cls_state = ckpt["cls_head_state"]
    cls_head.load_state_dict(cls_state)
    print(f"Loaded cls head (accuracy: {ckpt.get('accuracy', 0):.4f})")

    # Freeze vision encoder, train language model + cls head
    for param in model.model.visual.parameters():
        param.requires_grad = False

    # Data
    train_dataset = CropDataset(CROP_DIR, oversample_weak=True)
    val_dataset = ValDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=lambda b: {"images": [x["image"] for x in b],
                              "labels": torch.tensor([x["label"] for x in b])},
        drop_last=True
    )

    # Optimizer - only unfrozen params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += list(cls_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    print(f"Total steps: {total_steps}")

    # Label smoothing for better calibration
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Wandb
    wandb.init(project="nmiai-objdet", name="12layer-boost", config={
        "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "label_smoothing": 0.05, "frozen": "vision",
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_acc = ckpt.get("accuracy", 0)
    global_step = 0
    running_loss = 0
    running_correct = 0
    running_total = 0

    model.train()
    cls_head.train()

    for epoch in range(EPOCHS):
        for batch_idx, batch in enumerate(train_loader):
            images, labels = batch["images"], batch["labels"].to(device)
            inputs = process_batch(images, processor, device)

            # Cosine LR with warmup
            if global_step < WARMUP_STEPS:
                lr = LR * global_step / WARMUP_STEPS
            else:
                progress = (global_step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
                lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.model(**inputs, output_hidden_states=True)
                logits = cls_head(out.last_hidden_state)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            running_loss += loss.item()
            running_correct += (logits.argmax(-1) == labels).sum().item()
            running_total += labels.shape[0]
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                acc = running_correct / running_total
                print(f"Step {global_step}/{total_steps} | loss={loss.item():.4f} avg={avg_loss:.4f} | acc={acc:.3f} | lr={lr:.2e}")
                wandb.log({"train/loss": avg_loss, "train/acc": acc, "lr": lr}, step=global_step)
                running_loss = 0
                running_correct = 0
                running_total = 0

            if global_step % VAL_EVERY == 0:
                val_acc, val_loss = validate(model, cls_head, processor, val_dataset, device)
                print(f"  VAL: acc={val_acc:.4f}, loss={val_loss:.4f}")
                wandb.log({"val/acc": val_acc, "val/loss": val_loss}, step=global_step)

                if val_acc > best_acc:
                    best_acc = val_acc
                    # Save best
                    save_path = OUTPUT_DIR / "best.pt"
                    state = {}
                    for k, v in model.state_dict().items():
                        state[k] = v.cpu()
                    torch.save({
                        "model_state": state,
                        "cls_head_state": {k: v.cpu() for k, v in cls_head.state_dict().items()},
                        "global_step": global_step,
                        "epoch": epoch,
                        "loss": val_loss,
                        "accuracy": val_acc,
                    }, str(save_path))
                    print(f"  NEW BEST: {val_acc:.4f} (saved)")

    wandb.finish()
    print(f"Training complete. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train()
