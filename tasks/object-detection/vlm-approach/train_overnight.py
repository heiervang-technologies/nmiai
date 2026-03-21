"""
Overnight curriculum training pipeline for MarkusNet.

3 stages over ~8 hours:
  Stage 1 (4h): Pre-train on COCO train2017 (118k images, 80 classes)
                High LR, high batch size. Broad object understanding.
  Stage 2 (2h): Narrow to grocery/retail (SKU-110K crops if available,
                else COCO food/drink subset). Medium LR.
  Stage 3 (2h): Fine-tune on competition data (22.7k crops, 356 classes)
                with interleaved classification + detection. Low LR.

Each stage saves a checkpoint. Later stages load from previous.
Supports multi-GPU via torchrun:
  torchrun --nproc_per_node=N train_overnight.py [--stage 1|2|3] [--resume PATH]

Usage:
  # Full pipeline (all 3 stages):
  torchrun --nproc_per_node=2 train_overnight.py

  # Single stage:
  torchrun --nproc_per_node=2 train_overnight.py --stage 2 --resume checkpoints/stage1/best.pt

  # Single GPU:
  CUDA_VISIBLE_DEVICES=0 python train_overnight.py
"""

import argparse
import json
import math
import functools
import time
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

print = functools.partial(print, flush=True)


def get_base_model(model):
    """Unwrap DDP to get the underlying model's .model attribute."""
    m = model.module if hasattr(model, 'module') else model
    return m.model


# === PATHS ===
PRUNED_DIR = Path(__file__).parent / "pruned"
COMP_DATA = Path(__file__).parent / "cached_dataset"
COMP_SAMPLES = COMP_DATA / "samples.json"
COCO_DIR = Path(__file__).parent / "external_datasets" / "coco_minitrain"
COCO_TRAIN2017 = Path("/data/coco/train2017")  # Override via --coco-dir
VAL_DIR = Path(__file__).parent.parent / "data-creation" / "data" / "clean_split" / "val"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints_overnight"

# === STAGE CONFIGS ===
STAGE_CONFIGS = {
    1: {
        "name": "pretrain_coco",
        "description": "Broad object understanding on COCO",
        "lr": 1e-4,  # Reduced for stability (was 4e-4)
        "batch_size": 16,  # Per GPU (reduced from 64 for stability)
        "epochs": 3,
        "warmup_ratio": 0.05,
        "num_classes": 80,
        "max_hours": 4.0,
        "label_smoothing": 0.1,
    },
    2: {
        "name": "narrow_retail",
        "description": "Narrow to grocery/retail domain",
        "lr": 7e-5,  # sqrt-scaled from 5e-5 for 2x batch
        "batch_size": 16,  # Per GPU (was 8, 32 OOMs with 356 classes)
        "epochs": 5,
        "warmup_ratio": 0.03,
        "num_classes": 356,  # Use competition classes
        "max_hours": 2.0,
        "label_smoothing": 0.05,
    },
    3: {
        "name": "finetune_competition",
        "description": "Multi-task on competition data (cls + det)",
        "lr": 4e-5,  # sqrt-scaled from 2e-5 for 4x batch
        "batch_size": 16,  # Per GPU (was 4, 32+ OOMs with interleaved det)
        "epochs": 10,
        "warmup_ratio": 0.02,
        "num_classes": 356,
        "max_hours": 2.0,
        "label_smoothing": 0.02,
    },
}


# === HEADS ===
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
        return self.head(hidden_states.mean(dim=1))


class DetectionHead(nn.Module):
    """Grid-based anchor-free detection head."""
    def __init__(self, hidden_size, grid_size=14, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.project = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.GELU(), nn.Dropout(dropout))
        self.grid_predict = nn.Linear(512, grid_size * grid_size * 5)

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        out = self.grid_predict(self.project(pooled))
        B = out.shape[0]
        return out.view(B, self.grid_size, self.grid_size, 5)


# === DATASETS ===
class CropDataset(Dataset):
    """Generic crop classification dataset from samples.json format."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        return {"image": crop, "label": s["category_id"]}


class COCOCropDataset(Dataset):
    """COCO object crops for pre-training. Downloads/caches crops lazily."""
    # COCO category ID -> contiguous index mapping
    COCO_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,
                27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,
                51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,
                76,77,78,79,80,81,82,84,85,86,87,88,89,90]

    def __init__(self, annotations_path, images_dir, crop_cache_dir, max_samples=None):
        self.images_dir = Path(images_dir)
        self.crop_cache = Path(crop_cache_dir)
        self.crop_cache.mkdir(parents=True, exist_ok=True)
        self.id_to_idx = {cid: i for i, cid in enumerate(self.COCO_IDS)}

        with open(annotations_path) as f:
            coco = json.load(f)

        self.id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        self.samples = []
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            bbox = ann["bbox"]
            if bbox[2] < 10 or bbox[3] < 10:
                continue
            cat_idx = self.id_to_idx.get(ann["category_id"])
            if cat_idx is None:
                continue
            self.samples.append({
                "image_id": ann["image_id"],
                "ann_id": ann["id"],
                "bbox": bbox,
                "category_id": cat_idx,
            })

        if max_samples and len(self.samples) > max_samples:
            import random
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop_path = self.crop_cache / f"{s['ann_id']}.jpg"

        if crop_path.exists():
            crop = Image.open(crop_path).convert("RGB")
        else:
            img_path = self.images_dir / self.id_to_file[s["image_id"]]
            img = Image.open(img_path).convert("RGB")
            x, y, w, h = s["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
            crop = img.crop((x1, y1, x2, y2))
            # Resize large crops
            mx = max(crop.size)
            if mx > 512:
                scale = 512 / mx
                crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
            crop.save(str(crop_path), quality=85)

        return {"image": crop, "label": s["category_id"]}


class ShelfDetectionDataset(Dataset):
    """Full shelf images with COCO bbox annotations for detection head."""
    def __init__(self, annotations_path, images_dir):
        with open(annotations_path) as f:
            coco = json.load(f)
        self.images_dir = Path(images_dir)
        self.id_to_file = {img["id"]: img for img in coco["images"]}
        self.image_anns = defaultdict(list)
        for ann in coco["annotations"]:
            self.image_anns[ann["image_id"]].append(ann)
        self.image_ids = list(self.image_anns.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.id_to_file[img_id]
        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        w, h = img.size
        boxes, classes = [], []
        for ann in self.image_anns[img_id]:
            bx, by, bw, bh = ann["bbox"]
            boxes.append([max(0, min(1, (bx+bw/2)/w)), max(0, min(1, (by+bh/2)/h)),
                          max(0.001, min(1, bw/w)), max(0.001, min(1, bh/h))])
            classes.append(ann["category_id"])
        return {"image": img, "boxes": boxes, "classes": classes}


# === TRAINING UTILS ===
def process_batch(images, processor, prompt, device):
    texts = []
    for img in images:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


def collate_cls(batch):
    return {"images": [b["image"] for b in batch],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long)}


def collate_det(batch):
    return {"images": [b["image"] for b in batch],
            "boxes": [b["boxes"] for b in batch],
            "classes": [b["classes"] for b in batch]}


def build_det_targets(boxes_list, classes_list, grid_size, device):
    B, G = len(boxes_list), grid_size
    conf = torch.zeros(B, G, G, device=device)
    bbox = torch.zeros(B, G, G, 4, device=device)
    for b in range(B):
        for box, cls_id in zip(boxes_list[b], classes_list[b]):
            cx, cy, w, h = box
            gi, gj = min(int(cx * G), G-1), min(int(cy * G), G-1)
            conf[b, gj, gi] = 1.0
            bbox[b, gj, gi] = torch.tensor([cx*G - gi, cy*G - gj, w, h])
    return conf, bbox


def compute_det_loss(pred, conf_target, bbox_target):
    pred_conf, pred_bbox = pred[..., 0], pred[..., 1:5]
    bce = F.binary_cross_entropy_with_logits(pred_conf, conf_target, reduction='none')
    pt = torch.exp(-bce)
    conf_loss = (0.25 * (1-pt)**2 * bce).mean()
    pos = conf_target > 0.5
    if pos.sum() > 0:
        bbox_loss = F.smooth_l1_loss(torch.sigmoid(pred_bbox[pos][:, :2]), bbox_target[pos][:, :2]) + \
                    F.smooth_l1_loss(pred_bbox[pos][:, 2:], bbox_target[pos][:, 2:])
    else:
        bbox_loss = torch.tensor(0.0, device=pred.device)
    return conf_loss + bbox_loss


@torch.no_grad()
def validate(model, cls_head, processor, val_dir, device, max_crops=500):
    model.eval(); cls_head.eval()
    imgs_dir = Path(val_dir) / "images"
    labs_dir = Path(val_dir) / "labels"
    correct = total = 0
    for lab_path in sorted(labs_dir.glob("*.txt")):
        img_path = imgs_dir / (lab_path.stem + ".jpg")
        if not img_path.exists(): continue
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        for line in lab_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5: continue
            cid = int(parts[0])
            cx, cy, bw, bh = float(parts[1])*w, float(parts[2])*h, float(parts[3])*w, float(parts[4])*h
            x1, y1 = max(0, int(cx-bw/2)), max(0, int(cy-bh/2))
            x2, y2 = min(w, int(cx+bw/2)), min(h, int(cy+bh/2))
            if x2 <= x1 or y2 <= y1: continue
            crop = img.crop((x1, y1, x2, y2))
            inputs = process_batch([crop], processor, "classify", device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = get_base_model(model)(**inputs, output_hidden_states=True)
                pred = cls_head(out.last_hidden_state).argmax(-1).item()
            if pred == cid: correct += 1
            total += 1
            if total >= max_crops: break
        if total >= max_crops: break
    model.train(); cls_head.train()
    return correct / max(1, total)


def run_stage(stage_num, model, processor, cls_head, det_head, device, rank, world_size, args):
    cfg = STAGE_CONFIGS[stage_num]
    is_main = rank == 0
    if is_main:
        print(f"\n{'='*60}")
        print(f"STAGE {stage_num}: {cfg['description']}")
        print(f"LR={cfg['lr']}, batch={cfg['batch_size']}, epochs={cfg['epochs']}, max_hours={cfg['max_hours']}")
        print(f"{'='*60}\n")

    stage_dir = CHECKPOINT_DIR / f"stage{stage_num}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset based on stage
    if stage_num == 1:
        coco_ann = args.coco_dir / "annotations" / "instances_train2017.json"
        coco_imgs = args.coco_dir / "train2017"
        if not coco_ann.exists():
            # Try minitrain
            coco_ann = COCO_DIR / "annotations" / "instances_minitrain2017.json"
            coco_imgs = COCO_DIR / "images"
        if not coco_ann.exists():
            if is_main: print(f"COCO not found at {coco_ann}, skipping stage 1")
            return
        crop_cache = COCO_DIR / "crops"
        dataset = COCOCropDataset(coco_ann, coco_imgs, crop_cache,
                                  max_samples=args.max_samples_stage1)
        # Replace cls head for 80 classes
        cls_head_stage = ClassificationHead(1024, 80).to(device).to(torch.bfloat16)
        if world_size > 1:
            cls_head_stage = DDP(cls_head_stage, device_ids=[rank])
    elif stage_num == 2:
        # Use competition data with full 356 classes
        with open(COMP_SAMPLES) as f:
            samples = json.load(f)
        dataset = CropDataset(samples)
        cls_head_stage = cls_head  # Use 356-class head
    else:  # stage 3
        with open(COMP_SAMPLES) as f:
            samples = json.load(f)
        dataset = CropDataset(samples)
        cls_head_stage = cls_head

    if is_main:
        print(f"Dataset: {len(dataset)} samples")

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=(sampler is None),
                        sampler=sampler, num_workers=0, collate_fn=collate_cls, drop_last=True)

    # Detection loader for stage 3
    det_loader = None
    if stage_num == 3:
        comp_ann = Path(__file__).parent.parent / "data-creation" / "data" / "coco_dataset" / "train" / "annotations.json"
        comp_imgs = Path(__file__).parent.parent / "data-creation" / "data" / "coco_dataset" / "train" / "images"
        if comp_ann.exists():
            det_dataset = ShelfDetectionDataset(comp_ann, comp_imgs)
            det_loader = DataLoader(det_dataset, batch_size=1, shuffle=True,
                                    num_workers=0, collate_fn=collate_det, drop_last=True)
            if is_main: print(f"Detection dataset: {len(det_dataset)} shelf images")

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    if is_main:
        print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")

    # Class weights
    if hasattr(dataset, 'samples'):
        counts = Counter(s.get("category_id", s.get("label", 0)) for s in dataset.samples)
    else:
        counts = Counter()
    nc = cfg["num_classes"]
    class_weights = torch.ones(nc, device=device, dtype=torch.bfloat16)
    total_n = sum(counts.values())
    for c in range(nc):
        cnt = counts.get(c, 0)
        if cnt > 0:
            class_weights[c] = total_n / (nc * cnt)
    class_weights = class_weights.clamp(max=10.0)

    # Optimizer for this stage
    m = model.module if isinstance(model, DDP) else model
    ch = cls_head_stage.module if isinstance(cls_head_stage, DDP) else cls_head_stage
    dh = det_head.module if isinstance(det_head, DDP) else det_head
    params = list(m.parameters()) + list(ch.parameters())
    if det_head is not None and stage_num == 3:
        dh = det_head.module if isinstance(det_head, DDP) else det_head
        params += list(dh.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_time = time.time()
    global_step = 0
    best_val_acc = 0

    for epoch in range(cfg["epochs"]):
        if sampler: sampler.set_epoch(epoch)
        epoch_loss = epoch_correct = epoch_total = 0
        det_iter = iter(det_loader) if det_loader else None

        for batch_idx, batch in enumerate(loader):
            # Check time limit
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= cfg["max_hours"]:
                if is_main: print(f"Time limit reached ({elapsed_hours:.1f}h)")
                break

            images = batch["images"]
            labels = batch["labels"].to(device)

            inputs = process_batch(images, processor, "classify", device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = get_base_model(model)(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state
                logits = cls_head_stage(hidden)
                cls_loss = F.cross_entropy(logits, labels, weight=class_weights[:nc],
                                           label_smoothing=cfg["label_smoothing"])
                total_loss = cls_loss

            # Detection step (stage 3 only, interleaved)
            if det_iter and batch_idx % 5 == 0:  # Every 5th step
                try:
                    det_batch = next(det_iter)
                except StopIteration:
                    det_iter = iter(det_loader)
                    det_batch = next(det_iter)

                det_inputs = process_batch(det_batch["images"], processor, "detect", device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    det_out = get_base_model(model)(**det_inputs, output_hidden_states=True)
                    det_pred = det_head(det_out.last_hidden_state)
                    conf_t, bbox_t = build_det_targets(det_batch["boxes"], det_batch["classes"], 14, device)
                    det_loss = compute_det_loss(det_pred, conf_t, bbox_t)
                    total_loss = total_loss + det_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.shape[0]
            epoch_loss += cls_loss.item()
            global_step += 1

            if is_main and global_step % 20 == 0:
                acc = epoch_correct / max(1, epoch_total)
                lr = scheduler.get_last_lr()[0]
                gpu_mb = torch.cuda.memory_allocated() / 1024**2
                print(f"S{stage_num} Step {global_step}/{total_steps} | loss={cls_loss.item():.4f} avg={epoch_loss/(batch_idx+1):.4f} | acc={acc:.3f} | lr={lr:.2e} | {elapsed_hours:.1f}h | gpu={gpu_mb:.0f}MB")
                if args.wandb:
                    import wandb
                    wandb.log({
                        f"stage{stage_num}/loss": cls_loss.item(),
                        f"stage{stage_num}/accuracy": acc,
                        f"stage{stage_num}/lr": lr,
                        f"stage{stage_num}/epoch": epoch + 1,
                        "train/hours": elapsed_hours,
                        "train/epoch": epoch + 1,
                    }, step=global_step)

            if is_main and global_step % 500 == 0:
                torch.save({
                    "model_state": m.state_dict(),
                    "cls_head_state": ch.state_dict(),
                    "det_head_state": dh.state_dict() if det_head else None,
                    "stage": stage_num,
                    "step": global_step,
                    "accuracy": epoch_correct / max(1, epoch_total),
                }, stage_dir / f"checkpoint_{global_step}.pt")

        # End of epoch
        if is_main:
            acc = epoch_correct / max(1, epoch_total)
            elapsed_hours = (time.time() - start_time) / 3600
            print(f"S{stage_num} Epoch {epoch+1}: loss={epoch_loss/max(1,batch_idx+1):.4f} acc={acc:.3f} ({elapsed_hours:.1f}h)")

            # Validate (stage 2+3 only, uses competition val set)
            if stage_num >= 2 and VAL_DIR.exists():
                val_acc = validate(m, ch, processor, VAL_DIR, device)
                print(f"  Val accuracy: {val_acc:.3f}")
                if args.wandb:
                    import wandb
                    wandb.log({f"stage{stage_num}/val_acc": val_acc}, step=global_step)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        "model_state": m.state_dict(),
                        "cls_head_state": ch.state_dict(),
                        "det_head_state": dh.state_dict() if det_head else None,
                        "stage": stage_num, "val_acc": val_acc,
                    }, stage_dir / "best.pt")
                    print(f"  New best! val_acc={val_acc:.3f}")

        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= cfg["max_hours"]:
            break

    # Final save
    if is_main:
        torch.save({
            "model_state": m.state_dict(),
            "cls_head_state": ch.state_dict(),
            "det_head_state": dh.state_dict() if det_head else None,
            "stage": stage_num,
        }, stage_dir / "final.pt")
        print(f"Stage {stage_num} complete. Saved to {stage_dir}")

    # Transfer cls head weights for next stage (80 -> 356)
    if stage_num == 1 and isinstance(cls_head_stage, (DDP,)):
        pass  # Backbone weights transfer automatically, head gets replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, help="Run single stage (1/2/3), 0=all")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--coco-dir", type=Path, default=COCO_TRAIN2017.parent, help="COCO dataset root")
    parser.add_argument("--max-samples-stage1", type=int, default=200000, help="Max COCO samples")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    args = parser.parse_args()

    # DDP setup
    distributed = dist.is_available() and dist.is_initialized()
    if not distributed:
        try:
            dist.init_process_group(backend="nccl")
            distributed = True
        except Exception:
            distributed = False

    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0
    if is_main:
        print(f"{'='*60}")
        print(f"MarkusNet Overnight Curriculum Training")
        print(f"GPUs: {world_size}, Device: {device}")
        print(f"{'='*60}")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        if args.wandb:
            import wandb
            wandb.init(project="nmiai-objdet", name=f"overnight-curriculum-{world_size}gpu",
                       config={"stages": STAGE_CONFIGS, "world_size": world_size})

    # Load model
    if is_main: print("Loading pruned Qwen3.5 (12 layers)...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(PRUNED_DIR), dtype=torch.bfloat16,
        ignore_mismatched_sizes=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    hidden_size = model.config.text_config.hidden_size

    cls_head = ClassificationHead(hidden_size, 356).to(device).to(torch.bfloat16)
    det_head = DetectionHead(hidden_size, 14).to(device).to(torch.bfloat16)

    # Resume from checkpoint
    if args.resume and args.resume.exists():
        if is_main: print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "cls_head_state" in ckpt:
            cls_head.load_state_dict(ckpt["cls_head_state"])
        if ckpt.get("det_head_state"):
            det_head.load_state_dict(ckpt["det_head_state"])
        if is_main: print(f"Resumed from stage {ckpt.get('stage', '?')}")

    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        cls_head = DDP(cls_head, device_ids=[rank])
        det_head = DDP(det_head, device_ids=[rank])

    if is_main:
        bp = sum(p.numel() for p in model.parameters())
        print(f"Backbone: {bp/1e6:.0f}M params")

    # Run stages
    stages = [args.stage] if args.stage > 0 else [1, 2, 3]
    for s in stages:
        run_stage(s, model, processor, cls_head, det_head, device, rank, world_size, args)

    if distributed:
        dist.destroy_process_group()

    if is_main:
        print("\n" + "="*60)
        print("OVERNIGHT TRAINING COMPLETE")
        print("="*60)
        if args.wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
