"""
DDP training of pruned Qwen3.5 (MarkusNet-860M) on 2x RTX 3090.

Resumes from best checkpoint, uses class-weighted loss + label smoothing
to push past 90% toward 95%.

Usage on titan:
  torchrun --nproc_per_node=2 train_ddp.py
"""

import json
import math
import functools
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

print = functools.partial(print, flush=True)

# === CONFIG ===
PRUNED_DIR = Path(__file__).parent / "pruned"
CHECKPOINT = Path(__file__).parent / "training_output" / "best" / "best.pt"
SAMPLES_CACHE = Path(__file__).parent / "cached_dataset" / "samples.json"
OUTPUT_DIR = Path(__file__).parent / "training_output"

NUM_CLASSES = 356
BATCH_SIZE = 8       # Per GPU, so effective = 16 with 2 GPUs
LR = 3e-5
EPOCHS = 10
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


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def main():
    # Init DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank  # Assuming single node
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = rank == 0
    if is_main:
        print(f"DDP training with {world_size} GPUs")
        import wandb
        wandb.init(
            project="nmiai-objdet",
            name=f"markusnet-860m-ddp-{world_size}gpu",
            config={
                "batch_size_per_gpu": BATCH_SIZE,
                "effective_batch_size": BATCH_SIZE * world_size,
                "lr": LR,
                "epochs": EPOCHS,
                "label_smoothing": LABEL_SMOOTHING,
                "world_size": world_size,
            },
        )

    # Load model
    if is_main:
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
    if CHECKPOINT.exists():
        if is_main:
            print(f"Loading checkpoint: {CHECKPOINT}")
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        cls_head.load_state_dict(ckpt["cls_head_state"])
        prev_acc = ckpt.get("accuracy", 0)
        prev_step = ckpt.get("global_step", 0)
        if is_main:
            print(f"Resumed: step {prev_step}, acc {prev_acc:.3f}")
    else:
        prev_acc = 0
        prev_step = 0
        if is_main:
            print("No checkpoint found, training from scratch")

    model = model.to(device)

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    cls_head = DDP(cls_head, device_ids=[local_rank])

    # Load data
    with open(SAMPLES_CACHE) as f:
        samples = json.load(f)
    if is_main:
        print(f"Dataset: {len(samples)} samples")

    # Class weights
    counts = Counter(s["category_id"] for s in samples)
    total = sum(counts.values())
    class_weights = torch.zeros(NUM_CLASSES, device=device, dtype=torch.bfloat16)
    for c in range(NUM_CLASSES):
        count = counts.get(c, 0)
        class_weights[c] = (total / (NUM_CLASSES * count)) if count > 0 else 1.0
    class_weights = class_weights.clamp(max=10.0)

    dataset = CropDataset(samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=0, collate_fn=collate_fn, drop_last=True)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS
    if is_main:
        print(f"{steps_per_epoch} steps/epoch, {total_steps} total, effective batch={BATCH_SIZE * world_size}")

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

    if is_main:
        print(f"\n=== DDP Training: {EPOCHS} epochs, {world_size} GPUs, batch={BATCH_SIZE}x{world_size} ===\n")

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
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
                outputs = model.module.model(**inputs, output_hidden_states=True)
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

            if is_main and global_step % LOG_EVERY == 0:
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

            if is_main and global_step % SAVE_EVERY == 0:
                acc = epoch_correct / max(1, epoch_total)
                if acc > best_acc:
                    best_acc = acc
                    best_path = OUTPUT_DIR / "best"
                    best_path.mkdir(exist_ok=True, parents=True)
                    torch.save({
                        "model_state": model.module.state_dict(),
                        "cls_head_state": cls_head.module.state_dict(),
                        "global_step": prev_step + global_step,
                        "accuracy": acc,
                    }, best_path / "best.pt")
                    print(f"New best: acc={acc:.3f}")

        if is_main:
            avg_loss = epoch_loss / steps_per_epoch
            acc = epoch_correct / max(1, epoch_total)
            print(f"\n=== Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} acc={acc:.3f} ===\n")
            wandb.log({"epoch/loss": avg_loss, "epoch/accuracy": acc}, step=prev_step + global_step)

    if is_main:
        final_path = OUTPUT_DIR / "final"
        final_path.mkdir(exist_ok=True, parents=True)
        torch.save({
            "model_state": model.module.state_dict(),
            "cls_head_state": cls_head.module.state_dict(),
            "global_step": prev_step + global_step,
            "accuracy": best_acc,
        }, final_path / "final.pt")
        wandb.finish()
        print(f"DONE. Best accuracy: {best_acc:.3f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
