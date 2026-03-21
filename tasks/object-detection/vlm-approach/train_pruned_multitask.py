"""
Multi-task training of pruned Qwen3.5-0.8B (12 layers) for grocery product
detection and classification.

Two heads on shared backbone:
  1. Classification head: crop image → product category (356 classes)
  2. Detection head: shelf image → bounding boxes + class (anchor-free grid)

Interleaved training: alternating cls and det batches each step.

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_pruned_multitask.py
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

# === CONFIG ===
PRUNED_DIR = Path(__file__).parent / "pruned"
DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"
COCO_IMAGES = DATA_ROOT / "coco_dataset" / "train" / "images"
CROP_CACHE = Path(__file__).parent / "cached_dataset" / "crops"
VAL_DIR = DATA_ROOT / "stratified_split" / "val"
OUTPUT_DIR = Path(__file__).parent / "training_output_multitask"

NUM_CLASSES = 356
CLS_BATCH_SIZE = 6
DET_BATCH_SIZE = 1  # shelf images are large
LR = 3e-5  # Lower LR since we resume from 91% checkpoint
EPOCHS = 3
WARMUP_STEPS = 200
RESUME_CHECKPOINT = Path(__file__).parent / "training_output_multitask" / "best" / "best.pt"
CLS_ONLY = True  # Focus on classification - det handled by YOLO
LOG_EVERY = 10
SAVE_EVERY = 250
DET_GRID = 14  # 14x14 grid for detection head
DET_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 1.0
MAX_OBJECTS_PER_CELL = 1  # simplified: 1 object per grid cell


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


# === Detection Head ===
class DetectionHead(nn.Module):
    """Lightweight anchor-free grid-based detection head.

    Predicts per-cell: [conf, x_offset, y_offset, w, h] (5 values).
    Class prediction is done separately via the shared cls_head on
    per-cell feature vectors, keeping this head small.
    """
    def __init__(self, hidden_size, grid_size=14, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        # Only predict 5 values per cell (conf + bbox), NOT class logits
        self.per_cell = 5

        self.project = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.grid_predict = nn.Linear(512, grid_size * grid_size * self.per_cell)

    def forward(self, hidden_states):
        # hidden_states: [B, seq_len, hidden_size]
        pooled = hidden_states.mean(dim=1)  # [B, hidden_size]
        h = self.project(pooled)
        out = self.grid_predict(h)  # [B, G*G*5]
        B = out.shape[0]
        out = out.view(B, self.grid_size, self.grid_size, self.per_cell)
        return out


# === Datasets ===
class CropClassificationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        # Cap at 384px to prevent OOM with variable crop sizes
        max_dim = max(crop.size)
        if max_dim > 384:
            scale = 384 / max_dim
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        return {"image": crop, "label": s["category_id"]}


class ShelfDetectionDataset(Dataset):
    """Full shelf images with COCO bounding box annotations."""

    def __init__(self, coco_path, images_dir):
        with open(coco_path) as f:
            coco = json.load(f)

        self.images_dir = Path(images_dir)
        self.id_to_file = {img["id"]: img for img in coco["images"]}
        self.categories = {c["id"]: c["name"] for c in coco["categories"]}

        # Group annotations by image
        self.image_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.image_anns[ann["image_id"]].append(ann)

        self.image_ids = list(self.image_anns.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.id_to_file[img_id]
        img_path = self.images_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        # Resize large shelf images to cap at 1024px to save VRAM
        max_dim = max(img.size)
        if max_dim > 1024:
            scale = 1024 / max_dim
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        w, h = img.size

        # Convert COCO bbox [x,y,w,h] to normalized [cx, cy, w, h]
        boxes = []
        classes = []
        for ann in self.image_anns[img_id]:
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0.001, min(1, nw))
            nh = max(0.001, min(1, nh))
            boxes.append([cx, cy, nw, nh])
            classes.append(ann["category_id"])

        return {
            "image": img,
            "boxes": boxes,
            "classes": classes,
        }


class ValClassificationDataset(Dataset):
    """YOLO-format val set for classification evaluation."""

    def __init__(self, val_dir):
        self.val_dir = Path(val_dir)
        self.images_dir = self.val_dir / "images"
        self.labels_dir = self.val_dir / "labels"
        self.samples = []

        for label_path in sorted(self.labels_dir.glob("*.txt")):
            img_name = label_path.stem + ".jpg"
            img_path = self.images_dir / img_name
            if not img_path.exists():
                continue

            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cat_id = int(parts[0])
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        self.samples.append({
                            "img_path": str(img_path),
                            "category_id": cat_id,
                            "bbox_norm": [cx, cy, bw, bh],
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        w, h = img.size
        cx, cy, bw, bh = s["bbox_norm"]
        # Convert normalized coords to pixel coords
        px = int((cx - bw / 2) * w)
        py = int((cy - bh / 2) * h)
        pw = int(bw * w)
        ph = int(bh * h)
        px, py = max(0, px), max(0, py)
        crop = img.crop((px, py, min(w, px + pw), min(h, py + ph)))
        if crop.size[0] < 1 or crop.size[1] < 1:
            crop = img  # fallback
        return {"image": crop, "label": s["category_id"]}


def build_det_targets(boxes_list, classes_list, grid_size, num_classes, device):
    """Build grid-based detection targets from variable-length box lists.

    Returns:
        conf_target: [B, G, G] - 1 where object center falls, 0 elsewhere
        bbox_target: [B, G, G, 4] - [x_off, y_off, w, h] offsets within cell
        cls_target:  [B, G, G] - class id per cell (0 = background)
    """
    B = len(boxes_list)
    G = grid_size
    conf_target = torch.zeros(B, G, G, device=device)
    bbox_target = torch.zeros(B, G, G, 4, device=device)
    cls_target = torch.zeros(B, G, G, dtype=torch.long, device=device)

    for b in range(B):
        for box, cls_id in zip(boxes_list[b], classes_list[b]):
            cx, cy, w, h = box
            # Which grid cell?
            gi = min(int(cx * G), G - 1)
            gj = min(int(cy * G), G - 1)
            # Offset within cell
            x_off = cx * G - gi
            y_off = cy * G - gj

            conf_target[b, gj, gi] = 1.0
            bbox_target[b, gj, gi] = torch.tensor([x_off, y_off, w, h])
            cls_target[b, gj, gi] = cls_id

    return conf_target, bbox_target, cls_target


def compute_det_loss(pred, conf_target, bbox_target):
    """Compute detection loss: focal conf + smooth L1 bbox.

    pred: [B, G, G, 5] - conf + 4 bbox values per cell
    """
    pred_conf = pred[..., 0]
    pred_bbox = pred[..., 1:5]

    # Focal loss for confidence (handles sparse grids)
    alpha = 0.25
    gamma = 2.0
    bce = F.binary_cross_entropy_with_logits(pred_conf, conf_target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    conf_loss = focal.mean()

    # Bbox loss only on positive cells
    pos_mask = conf_target > 0.5
    if pos_mask.sum() > 0:
        pred_bbox_pos = pred_bbox[pos_mask]  # [N, 4]
        target_bbox_pos = bbox_target[pos_mask]  # [N, 4]
        # Sigmoid for offsets (0-1)
        pred_xy = torch.sigmoid(pred_bbox_pos[:, :2])
        pred_wh = pred_bbox_pos[:, 2:4]
        target_xy = target_bbox_pos[:, :2]
        target_wh = target_bbox_pos[:, 2:4]
        bbox_loss = F.smooth_l1_loss(pred_xy, target_xy) + F.smooth_l1_loss(pred_wh, target_wh)
    else:
        bbox_loss = torch.tensor(0.0, device=pred.device)

    return conf_loss + bbox_loss, conf_loss, bbox_loss


def compute_class_weights(samples, num_classes, device):
    """Inverse frequency class weights, clamped."""
    counts = torch.zeros(num_classes)
    for s in samples:
        counts[s["category_id"]] += 1
    # Inverse frequency, with smoothing
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * num_classes  # normalize to mean=1
    weights = weights.clamp(min=0.1, max=10.0)
    return weights.to(device)


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


VAL_EVERY = 500  # validate every N steps

@torch.no_grad()
def validate_cls(model, cls_head, processor, val_dataset, device, max_batches=200):
    """Run classification validation and return accuracy + loss."""
    model.eval()
    cls_head.eval()

    loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=lambda batch: {
            "images": [b["image"] for b in batch],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        },
        drop_last=False,
    )

    correct = 0
    total = 0
    total_loss = 0.0
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
            loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
        total_loss += loss.item() * labels.shape[0]

    model.train()
    cls_head.train()
    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return acc, avg_loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(
        project="nmiai-objdet",
        name="qwen35-pruned12-interleaved",
        config={
            "model": "Qwen3.5-0.8B-pruned-12layers",
            "cls_batch_size": CLS_BATCH_SIZE,
            "det_batch_size": DET_BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "num_classes": NUM_CLASSES,
            "det_grid": DET_GRID,
            "det_loss_weight": DET_LOSS_WEIGHT,
            "cls_loss_weight": CLS_LOSS_WEIGHT,
        },
    )

    # Load pruned model
    print("Loading pruned Qwen3.5 (12 layers)...")
    pruned_dir = str(PRUNED_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        pruned_dir,
        dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    model = model.to(device)
    # gradient_checkpointing slows training 4x - disabled since we cap images at 1024px
    hidden_size = model.config.text_config.hidden_size
    print(f"Hidden size: {hidden_size}")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    # Heads
    cls_head = ClassificationHead(hidden_size, NUM_CLASSES).to(device).to(torch.bfloat16)
    det_head = DetectionHead(hidden_size, DET_GRID).to(device).to(torch.bfloat16)

    # Resume from best checkpoint (91% accuracy)
    if RESUME_CHECKPOINT.exists():
        print(f"Loading checkpoint from {RESUME_CHECKPOINT}...")
        ckpt = torch.load(str(RESUME_CHECKPOINT), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        cls_head.load_state_dict(ckpt["cls_head_state"])
        acc = ckpt.get('accuracy', ckpt.get('val_acc', '?'))
        acc_str = f"{acc:.3f}" if isinstance(acc, float) else str(acc)
        print(f"Resumed from epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')}, acc={acc_str}")
        del ckpt
        torch.cuda.empty_cache()
    else:
        print("WARNING: No checkpoint found, training from scratch!")

    backbone_params = sum(p.numel() for p in model.parameters())
    cls_params = sum(p.numel() for p in cls_head.parameters())
    det_params = sum(p.numel() for p in det_head.parameters())
    print(f"Backbone: {backbone_params/1e6:.1f}M | Cls head: {cls_params/1e6:.1f}M | Det head: {det_params/1e6:.1f}M")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Datasets - use expanded 201K dataset if available
    print("Preparing datasets...")
    expanded_file = DATA_ROOT / "extra_crops" / "combined_samples.json"
    cache_file = Path(__file__).parent / "cached_dataset" / "samples.json"
    samples_file = expanded_file if expanded_file.exists() else cache_file
    print(f"Using samples from: {samples_file}")
    with open(samples_file) as f:
        crop_samples = json.load(f)
    # Stratified subsample if dataset is very large (>50K) to fit overnight training
    MAX_SAMPLES = 50000
    if len(crop_samples) > MAX_SAMPLES:
        print(f"Subsampling {MAX_SAMPLES} from {len(crop_samples)} (stratified)...")
        from collections import Counter
        by_cat = defaultdict(list)
        for s in crop_samples:
            by_cat[s["category_id"]].append(s)
        # Proportional sampling per category
        total = len(crop_samples)
        subsampled = []
        for cat_id, samples_in_cat in by_cat.items():
            n = max(1, int(len(samples_in_cat) / total * MAX_SAMPLES))
            n = min(n, len(samples_in_cat))
            subsampled.extend(random.sample(samples_in_cat, n))
        # Fill remaining quota randomly
        remaining = MAX_SAMPLES - len(subsampled)
        if remaining > 0:
            used = set(id(s) for s in subsampled)
            pool = [s for s in crop_samples if id(s) not in used]
            subsampled.extend(random.sample(pool, min(remaining, len(pool))))
        crop_samples = subsampled
        print(f"Using {len(crop_samples)} subsampled crops")

    cls_dataset = CropClassificationDataset(crop_samples)

    det_dataset = ShelfDetectionDataset(COCO_ANNOTATIONS, COCO_IMAGES)

    val_dataset = ValClassificationDataset(VAL_DIR)
    print(f"Cls train: {len(cls_dataset)} | Det train: {len(det_dataset)} images | Val: {len(val_dataset)} crops")

    # Class weights for imbalanced classification
    class_weights = compute_class_weights(crop_samples, NUM_CLASSES, device)

    cls_loader = DataLoader(
        cls_dataset, batch_size=CLS_BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=lambda batch: {
            "images": [b["image"] for b in batch],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        },
        drop_last=True,
    )

    det_loader = DataLoader(
        det_dataset, batch_size=DET_BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=lambda batch: {
            "images": [b["image"] for b in batch],
            "boxes": [b["boxes"] for b in batch],
            "classes": [b["classes"] for b in batch],
        },
        drop_last=True,
    )

    # Steps calculation
    cls_steps_per_epoch = len(cls_loader)
    det_steps_per_epoch = 0 if CLS_ONLY else len(det_loader)
    steps_per_epoch = cls_steps_per_epoch + det_steps_per_epoch
    total_steps = steps_per_epoch * EPOCHS
    print(f"CLS_ONLY={CLS_ONLY}")
    print(f"Steps/epoch: {cls_steps_per_epoch} cls + {det_steps_per_epoch} det = {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    # Optimizer
    if CLS_ONLY:
        all_params = list(model.parameters()) + list(cls_head.parameters())
    else:
        all_params = list(model.parameters()) + list(cls_head.parameters()) + list(det_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    cls_head.train()
    det_head.train()
    global_step = 0
    best_val_acc = 0.0

    print(f"\n=== Starting interleaved training: {EPOCHS} epochs ===")

    for epoch in range(EPOCHS):
        epoch_cls_loss = 0
        epoch_det_loss = 0
        epoch_cls_correct = 0
        epoch_cls_total = 0
        cls_steps = 0
        det_steps = 0

        cls_iter = iter(cls_loader)
        det_iter = None if CLS_ONLY else iter(det_loader)

        # Interleave: for each det batch, do ~N cls batches (ratio ~ cls_steps/det_steps)
        cls_per_det = cls_steps_per_epoch if CLS_ONLY else max(1, cls_steps_per_epoch // max(1, det_steps_per_epoch))

        step_in_epoch = 0
        while True:
            # === Classification batch ===
            for _ in range(cls_per_det):
                try:
                    cls_batch = next(cls_iter)
                except StopIteration:
                    cls_iter = None
                    break

                images = cls_batch["images"]
                labels = cls_batch["labels"].to(device)

                inputs = process_batch(images, processor, "classify", device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model.model(**inputs, output_hidden_states=True)
                    hidden = outputs.last_hidden_state
                    logits = cls_head(hidden)
                    loss = F.cross_entropy(logits, labels, weight=class_weights) * CLS_LOSS_WEIGHT

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                scheduler.step()

                preds = logits.argmax(dim=-1)
                epoch_cls_correct += (preds == labels).sum().item()
                epoch_cls_total += labels.shape[0]
                epoch_cls_loss += loss.item()
                cls_steps += 1
                global_step += 1
                step_in_epoch += 1

                if global_step % LOG_EVERY == 0:
                    avg_cls = epoch_cls_loss / max(1, cls_steps)
                    acc = epoch_cls_correct / max(1, epoch_cls_total)
                    lr = scheduler.get_last_lr()[0]
                    gpu_mb = torch.cuda.memory_allocated() / 1024**2
                    print(f"[CLS] Step {global_step} | loss={loss.item():.4f} avg={avg_cls:.4f} | acc={acc:.3f} | lr={lr:.2e} | gpu={gpu_mb:.0f}MB")
                    wandb.log({
                        "cls/loss": loss.item(),
                        "cls/avg_loss": avg_cls,
                        "cls/accuracy": acc,
                        "train/learning_rate": lr,
                        "train/gpu_mb": gpu_mb,
                        "train/epoch": epoch + step_in_epoch / steps_per_epoch,
                    }, step=global_step)

                if global_step % SAVE_EVERY == 0:
                    ckpt_path = OUTPUT_DIR / f"checkpoint-{global_step}"
                    ckpt_path.mkdir(exist_ok=True)
                    torch.save({
                        "model_state": model.state_dict(),
                        "cls_head_state": cls_head.state_dict(),
                        "det_head_state": det_head.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "accuracy": epoch_cls_correct / max(1, epoch_cls_total),
                    }, ckpt_path / "checkpoint.pt")
                    print(f"Saved checkpoint to {ckpt_path}")

                if global_step % VAL_EVERY == 0:
                    print(f"--- Validating at step {global_step} ---")
                    val_acc, val_loss = validate_cls(model, cls_head, processor, val_dataset, device)
                    print(f"Val acc={val_acc:.3f} val_loss={val_loss:.4f}")
                    wandb.log({
                        "val/accuracy": val_acc,
                        "val/loss": val_loss,
                    }, step=global_step)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_path = OUTPUT_DIR / "best"
                        best_path.mkdir(exist_ok=True)
                        torch.save({
                            "model_state": model.state_dict(),
                            "cls_head_state": cls_head.state_dict(),
                            "det_head_state": det_head.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "val_acc": val_acc,
                            "val_loss": val_loss,
                        }, best_path / "best.pt")
                        print(f"New best model saved (val_acc={val_acc:.3f})")

            if cls_iter is None:
                break

            # === Detection batch (skip if CLS_ONLY) ===
            if CLS_ONLY or det_iter is None:
                continue
            try:
                det_batch = next(det_iter)
            except StopIteration:
                # Det exhausted, keep doing cls
                continue

            images = det_batch["images"]
            boxes_list = det_batch["boxes"]
            classes_list = det_batch["classes"]

            inputs = process_batch(images, processor, "detect", device)

            conf_target, bbox_target, _ = build_det_targets(
                boxes_list, classes_list, DET_GRID, NUM_CLASSES, device
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.model(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state
                det_pred = det_head(hidden)
                det_loss, conf_loss, bbox_loss = compute_det_loss(
                    det_pred, conf_target, bbox_target
                )
                det_loss = det_loss * DET_LOSS_WEIGHT

            optimizer.zero_grad()
            det_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_det_loss += det_loss.item()
            det_steps += 1
            global_step += 1
            step_in_epoch += 1

            if global_step % LOG_EVERY == 0:
                avg_det = epoch_det_loss / max(1, det_steps)
                lr = scheduler.get_last_lr()[0]
                print(f"[DET] Step {global_step} | loss={det_loss.item():.4f} avg={avg_det:.4f} | conf={conf_loss.item():.4f} bbox={bbox_loss.item():.4f}")
                wandb.log({
                    "det/loss": det_loss.item(),
                    "det/avg_loss": avg_det,
                    "det/conf_loss": conf_loss.item(),
                    "det/bbox_loss": bbox_loss.item(),
                    "train/epoch": epoch + step_in_epoch / steps_per_epoch,
                }, step=global_step)

            if global_step % SAVE_EVERY == 0:
                ckpt_path = OUTPUT_DIR / f"checkpoint-{global_step}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "cls_head_state": cls_head.state_dict(),
                    "det_head_state": det_head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }, ckpt_path / "checkpoint.pt")
                print(f"Saved checkpoint to {ckpt_path}")

        # End of epoch - validate
        avg_cls = epoch_cls_loss / max(1, cls_steps)
        avg_det = epoch_det_loss / max(1, det_steps)
        train_acc = epoch_cls_correct / max(1, epoch_cls_total)

        print(f"\n--- Validating epoch {epoch+1} ---")
        val_acc, val_loss = validate_cls(model, cls_head, processor, val_dataset, device)

        print(f"\n=== Epoch {epoch+1}/{EPOCHS}: cls_loss={avg_cls:.4f} det_loss={avg_det:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_loss={val_loss:.4f} ===\n")

        wandb.log({
            "epoch/cls_loss": avg_cls,
            "epoch/det_loss": avg_det,
            "epoch/train_acc": train_acc,
            "epoch/val_acc": val_acc,
            "val/accuracy": val_acc,
            "val/loss": val_loss,
            "epoch/number": epoch + 1,
        }, step=global_step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = OUTPUT_DIR / "best"
            best_path.mkdir(exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "cls_head_state": cls_head.state_dict(),
                "det_head_state": det_head.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "val_acc": val_acc,
                "train_acc": train_acc,
            }, best_path / "best.pt")
            print(f"New best model saved (val_acc={val_acc:.3f})")

    # Final save
    final_path = OUTPUT_DIR / "final"
    final_path.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cls_head_state": cls_head.state_dict(),
        "det_head_state": det_head.state_dict(),
        "global_step": global_step,
    }, final_path / "final.pt")
    print(f"Final model saved to {final_path}")

    wandb.finish()
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    train()
