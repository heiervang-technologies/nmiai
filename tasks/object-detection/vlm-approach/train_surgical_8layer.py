"""Retrain surgically pruned 8-layer MarkusNet on 196K clean crops.

Loads from pruned_8layer_surgical/ (layers 3,6,7,11 removed).
Trains classification head to recover accuracy lost from pruning.
"""
import json
import math
import random
import functools
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import wandb

print = functools.partial(print, flush=True)

PRUNED_DIR = Path(__file__).parent / "pruned_8layer_surgical"
CHECKPOINT = PRUNED_DIR / "checkpoint.pt"
DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
VAL_DIR = DATA_ROOT / "clean_split" / "val"
OUTPUT_DIR = Path(__file__).parent / "training_output_8layer"

NUM_CLASSES = 356
BATCH_SIZE = 6
LR = 5e-5
EPOCHS = 3
WARMUP_STEPS = 200
LOG_EVERY = 10
SAVE_EVERY = 500
VAL_EVERY = 500
MAX_SAMPLES = 50000


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
        return self.head(hidden_states.mean(dim=1))


class CropDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        max_dim = max(crop.size)
        if max_dim > 384:
            scale = 384 / max_dim
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        return {"image": crop, "label": s["category_id"]}


class ValDataset(Dataset):
    def __init__(self, val_dir):
        self.samples = []
        images_dir = Path(val_dir) / "images"
        labels_dir = Path(val_dir) / "labels"
        for label_path in sorted(labels_dir.glob("*.txt")):
            img_path = images_dir / (label_path.stem + ".jpg")
            if not img_path.exists():
                continue
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    self.samples.append({
                        "img_path": str(img_path),
                        "category_id": int(parts[0]),
                        "bbox_norm": [float(x) for x in parts[1:5]],
                    })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        w, h = img.size
        cx, cy, bw, bh = s["bbox_norm"]
        px, py = max(0, int((cx - bw/2) * w)), max(0, int((cy - bh/2) * h))
        pw, ph = int(bw * w), int(bh * h)
        crop = img.crop((px, py, min(w, px+pw), min(h, py+ph)))
        if crop.size[0] < 1 or crop.size[1] < 1:
            crop = img
        return {"image": crop, "label": s["category_id"]}


def process_batch(images, processor, device):
    texts = []
    for img in images:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
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
    print(f"Device: {device}")

    wandb.init(project="nmiai-objdet", name="surgical-8layer-retrain",
               config={"layers": 8, "removed": [3, 6, 7, 11], "lr": LR, "batch_size": BATCH_SIZE})

    # Monkey-patch Qwen3.5 cache to handle zero full_attention layers
    from transformers.models.qwen3_5 import modeling_qwen3_5
    _orig_cache_init = modeling_qwen3_5.Qwen3_5DynamicCache.__init__
    def _patched_cache_init(self, config, *args, **kwargs):
        _orig_cache_init(self, config, *args, **kwargs)
        if len(self.transformer_layers) == 0:
            self.transformer_layers = [0]
    modeling_qwen3_5.Qwen3_5DynamicCache.__init__ = _patched_cache_init
    print("Patched HybridCache for all-linear model")

    # Load surgically pruned model
    print("Loading 8-layer surgical model...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(PRUNED_DIR), dtype=torch.bfloat16,
        ignore_mismatched_sizes=True, trust_remote_code=True)

    # Load pruned weights
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    hidden_size = model.config.text_config.hidden_size

    cls_head = ClassificationHead(hidden_size, NUM_CLASSES).to(device).to(torch.bfloat16)
    cls_head.load_state_dict(ckpt["cls_head_state"])
    del ckpt
    torch.cuda.empty_cache()

    print(f"Backbone: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Load data
    expanded = DATA_ROOT / "extra_crops" / "combined_samples.json"
    cache = Path(__file__).parent / "cached_dataset" / "samples.json"
    data_file = expanded if expanded.exists() else cache
    print(f"Loading data from {data_file}")
    with open(data_file) as f:
        samples = json.load(f)

    if len(samples) > MAX_SAMPLES:
        print(f"Subsampling {MAX_SAMPLES} from {len(samples)}...")
        by_cat = defaultdict(list)
        for s in samples:
            by_cat[s["category_id"]].append(s)
        subsampled = []
        for cat_id, cat_samples in by_cat.items():
            n = max(1, int(len(cat_samples) / len(samples) * MAX_SAMPLES))
            subsampled.extend(random.sample(cat_samples, min(n, len(cat_samples))))
        remaining = MAX_SAMPLES - len(subsampled)
        if remaining > 0:
            used = set(id(s) for s in subsampled)
            pool = [s for s in samples if id(s) not in used]
            subsampled.extend(random.sample(pool, min(remaining, len(pool))))
        samples = subsampled

    dataset = CropDataset(samples)
    val_dataset = ValDataset(VAL_DIR)
    print(f"Train: {len(dataset)} | Val: {len(val_dataset)}")

    # Class weights
    from collections import Counter
    counts = Counter(s["category_id"] for s in samples)
    class_weights = torch.zeros(NUM_CLASSES, device=device, dtype=torch.bfloat16)
    total_n = sum(counts.values())
    for c in range(NUM_CLASSES):
        cnt = counts.get(c, 0)
        class_weights[c] = (total_n / (NUM_CLASSES * cnt)) if cnt > 0 else 1.0
    class_weights = class_weights.clamp(max=10.0)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        collate_fn=lambda b: {"images": [x["image"] for x in b],
                                              "labels": torch.tensor([x["label"] for x in b])},
                        drop_last=True)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS

    all_params = list(model.parameters()) + list(cls_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        min(s / max(1, WARMUP_STEPS), 0.5 * (1 + math.cos(math.pi * max(0, s - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)))))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    cls_head.train()
    global_step = 0
    best_val_acc = 0.0

    print(f"\nSteps/epoch: {steps_per_epoch}, Total: {total_steps}")
    print(f"=== Starting 8-layer retraining ({EPOCHS} epochs) ===\n")

    for epoch in range(EPOCHS):
        el = ec = et = 0
        for batch in loader:
            images, labels = batch["images"], batch["labels"].to(device)
            inputs = process_batch(images, processor, device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.model(**inputs, output_hidden_states=True)
                logits = cls_head(out.last_hidden_state)
                loss = F.cross_entropy(logits, labels, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            ec += (logits.argmax(-1) == labels).sum().item()
            et += labels.shape[0]
            el += loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                acc = ec / max(1, et)
                print(f"Step {global_step}/{total_steps} | loss={loss.item():.4f} avg={el/(global_step - epoch*steps_per_epoch):.4f} | acc={acc:.3f} | lr={scheduler.get_last_lr()[0]:.2e}")
                wandb.log({"train/loss": loss.item(), "train/accuracy": acc}, step=global_step)

            if global_step % SAVE_EVERY == 0:
                ckpt_path = OUTPUT_DIR / f"checkpoint-{global_step}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save({"model_state": model.state_dict(), "cls_head_state": cls_head.state_dict(),
                             "global_step": global_step, "epoch": epoch}, ckpt_path / "checkpoint.pt")
                print(f"Saved checkpoint-{global_step}")

            if global_step % VAL_EVERY == 0:
                val_acc, val_loss = validate(model, cls_head, processor, val_dataset, device)
                print(f"--- Val: acc={val_acc:.3f} loss={val_loss:.4f} ---")
                wandb.log({"val/accuracy": val_acc, "val/loss": val_loss}, step=global_step)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_path = OUTPUT_DIR / "best"
                    best_path.mkdir(exist_ok=True)
                    torch.save({"model_state": model.state_dict(), "cls_head_state": cls_head.state_dict(),
                                 "global_step": global_step, "val_acc": val_acc}, best_path / "best.pt")
                    print(f"New best: val_acc={val_acc:.3f}")

        acc = ec / max(1, et)
        val_acc, val_loss = validate(model, cls_head, processor, val_dataset, device)
        print(f"\n=== Epoch {epoch+1}/{EPOCHS}: train_acc={acc:.3f} val_acc={val_acc:.3f} ===\n")

    wandb.finish()
    print("RETRAINING COMPLETE")


if __name__ == "__main__":
    train()
