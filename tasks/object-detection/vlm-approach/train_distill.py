"""
Knowledge distillation training for MarkusNet-860M.

Teacher: Full Qwen3.5-0.8B (24 text layers) with trained classification head
Student: MarkusNet-860M (pruned 12 text layers) with same classification head

The teacher generates soft probability distributions over 356 grocery classes.
The student trains to match these using:
  total_loss = alpha * hard_CE_loss + (1-alpha) * T^2 * KL_div(soft_student, soft_teacher)

Two phases:
  Phase 1: Generate soft labels from teacher (saves to soft_labels.pt)
  Phase 2: Train student using soft + hard labels

Usage:
  # Phase 1: generate soft labels (can run on CPU, slow but works)
  CUDA_VISIBLE_DEVICES=0 uv run python train_distill.py --phase generate

  # Phase 2: distillation training
  CUDA_VISIBLE_DEVICES=0 uv run python train_distill.py --phase train

  # Both phases sequentially
  CUDA_VISIBLE_DEVICES=0 uv run python train_distill.py --phase both
"""

import argparse
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

print = functools.partial(print, flush=True)

# === CONFIG ===
PRUNED_DIR = Path(__file__).parent / "pruned"
STUDENT_CHECKPOINT = Path(__file__).parent / "training_output" / "best" / "best.pt"
SAMPLES_CACHE = Path(__file__).parent / "cached_dataset" / "samples.json"
OUTPUT_DIR = Path(__file__).parent / "training_output"
SOFT_LABELS_PATH = Path(__file__).parent / "training_output" / "soft_labels.pt"

TEACHER_MODEL_ID = "Qwen/Qwen3.5-0.8B"  # Full 24-layer model from HuggingFace

NUM_CLASSES = 356
HIDDEN_SIZE = 1024

# Distillation hyperparameters
TEMPERATURE = 4.0       # Softens probability distributions
ALPHA = 0.5             # Balance: 0=all soft, 1=all hard
BATCH_SIZE = 8
LR = 1e-5               # Lower than normal training — student is already decent
EPOCHS = 5
WARMUP_STEPS = 100
LOG_EVERY = 10
SAVE_EVERY = 500
LABEL_SMOOTHING = 0.05  # Light smoothing on hard labels (less than normal since soft labels help)

# Teacher inference batch size (can be larger since no gradients)
TEACHER_BATCH_SIZE = 16


class ClassificationHead(nn.Module):
    """Classification head — identical architecture for teacher and student."""
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
    """Loads crop images lazily with integer class labels."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        return {"image": crop, "label": s["category_id"], "index": idx}


class DistillDataset(Dataset):
    """Wraps crop dataset with precomputed teacher soft labels."""
    def __init__(self, samples, soft_logits):
        self.samples = samples
        self.soft_logits = soft_logits  # (N, NUM_CLASSES) tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        return {
            "image": crop,
            "label": s["category_id"],
            "teacher_logits": self.soft_logits[idx],
        }


def compute_class_weights(samples):
    """Compute inverse-frequency class weights, capped at 10x."""
    counts = Counter(s["category_id"] for s in samples)
    total = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        count = counts.get(c, 0)
        if count > 0:
            weights[c] = total / (NUM_CLASSES * count)
        else:
            weights[c] = 1.0
    weights = weights.clamp(max=10.0)
    return weights


def build_processor_inputs(images, processor, device):
    """Build model inputs from a batch of PIL images."""
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
    return {k: v.to(device) for k, v in inputs.items()}


# ============================================================================
# Phase 1: Generate soft labels from teacher
# ============================================================================

def generate_soft_labels():
    """
    Load the full (unpruned) Qwen3.5-0.8B as teacher, attach the trained
    classification head, and generate soft logits for every training crop.

    The teacher has 24 text layers vs the student's 12. The classification head
    weights are loaded from the student checkpoint — the head was trained on
    features from 12 layers, but the teacher's deeper 24-layer features should
    produce even better soft targets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 1: Generate Soft Labels ===")
    print(f"Device: {device}")
    print(f"Teacher model: {TEACHER_MODEL_ID}")

    # Load teacher (full 24-layer model)
    print("Loading teacher model (full Qwen3.5-0.8B, 24 text layers)...")
    teacher = AutoModelForImageTextToText.from_pretrained(
        TEACHER_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    num_teacher_layers = teacher.config.text_config.num_hidden_layers
    print(f"Teacher text layers: {num_teacher_layers}")

    processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True)
    hidden_size = teacher.config.text_config.hidden_size
    assert hidden_size == HIDDEN_SIZE, f"Expected hidden_size={HIDDEN_SIZE}, got {hidden_size}"

    # Load classification head from student checkpoint
    print(f"Loading classification head from {STUDENT_CHECKPOINT}")
    cls_head = ClassificationHead(hidden_size, NUM_CLASSES, dropout=0.0)  # No dropout at inference
    ckpt = torch.load(STUDENT_CHECKPOINT, map_location=device, weights_only=False)
    cls_head.load_state_dict(ckpt["cls_head_state"])
    cls_head = cls_head.to(device).to(torch.bfloat16)
    prev_acc = ckpt.get("accuracy", 0)
    print(f"Classification head loaded (student accuracy was {prev_acc:.3f})")

    teacher.eval()
    cls_head.eval()

    # Load dataset
    with open(SAMPLES_CACHE) as f:
        samples = json.load(f)
    print(f"Dataset: {len(samples)} samples")

    dataset = CropDataset(samples)

    def collate(batch):
        return {
            "images": [b["image"] for b in batch],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        }

    loader = DataLoader(
        dataset, batch_size=TEACHER_BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate, drop_last=False,
    )

    # Pre-allocate output tensor
    all_logits = torch.zeros(len(samples), NUM_CLASSES, dtype=torch.float32)
    teacher_correct = 0
    teacher_total = 0

    print(f"\nGenerating soft labels ({len(loader)} batches)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["images"]
            labels = batch["labels"].to(device)
            indices = batch["indices"]

            inputs = build_processor_inputs(images, processor, device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = teacher.model(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state
                logits = cls_head(hidden)

            # Store logits in float32 for precision
            all_logits[indices] = logits.float().cpu()

            # Track teacher accuracy
            preds = logits.argmax(dim=-1)
            teacher_correct += (preds == labels).sum().item()
            teacher_total += labels.shape[0]

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                acc = teacher_correct / max(1, teacher_total)
                print(f"  Batch {batch_idx + 1}/{len(loader)} | teacher acc so far: {acc:.3f}")

    teacher_acc = teacher_correct / max(1, teacher_total)
    print(f"\nTeacher accuracy on training set: {teacher_acc:.3f}")

    # Sanity checks
    print(f"Soft logits shape: {all_logits.shape}")
    print(f"Logit stats: min={all_logits.min():.2f}, max={all_logits.max():.2f}, "
          f"mean={all_logits.mean():.2f}, std={all_logits.std():.2f}")

    # Check soft label entropy (higher = more informative soft labels)
    soft_probs = F.softmax(all_logits / TEMPERATURE, dim=-1)
    entropy = -(soft_probs * (soft_probs + 1e-10).log()).sum(dim=-1).mean()
    print(f"Mean entropy of soft labels (T={TEMPERATURE}): {entropy:.2f} "
          f"(max possible: {math.log(NUM_CLASSES):.2f})")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "logits": all_logits,
        "teacher_accuracy": teacher_acc,
        "temperature": TEMPERATURE,
        "num_samples": len(samples),
        "num_classes": NUM_CLASSES,
        "teacher_model": TEACHER_MODEL_ID,
        "teacher_layers": num_teacher_layers,
    }, SOFT_LABELS_PATH)
    size_mb = SOFT_LABELS_PATH.stat().st_size / 1024**2
    print(f"Saved soft labels to {SOFT_LABELS_PATH} ({size_mb:.1f} MB)")

    # Free teacher memory
    del teacher, cls_head, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Phase 1 complete.\n")
    return all_logits


# ============================================================================
# Phase 2: Distillation training
# ============================================================================

def distillation_loss(student_logits, teacher_logits, hard_labels,
                      class_weights, temperature, alpha, label_smoothing):
    """
    Combined distillation loss:
      L = alpha * CE(student, hard_labels) + (1 - alpha) * T^2 * KL(soft_student || soft_teacher)

    The T^2 factor compensates for the reduced gradient magnitude when using
    temperature scaling. This is standard Hinton et al. (2015) distillation.
    """
    # Hard loss: standard cross-entropy with class weights and label smoothing
    hard_loss = F.cross_entropy(
        student_logits, hard_labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    # Soft loss: KL divergence between temperature-scaled distributions
    # KL(P_student || P_teacher) where P = softmax(logits / T)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence: sum over classes, mean over batch
    soft_loss = F.kl_div(
        student_log_soft,
        teacher_soft,
        reduction="batchmean",
    )

    total = alpha * hard_loss + (1 - alpha) * (temperature ** 2) * soft_loss

    return total, hard_loss, soft_loss


def train_distill():
    """Train student using soft labels from teacher + hard ground truth labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 2: Distillation Training ===")
    print(f"Device: {device}")

    # Load soft labels
    if not SOFT_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Soft labels not found at {SOFT_LABELS_PATH}. "
            "Run with --phase generate first."
        )

    print(f"Loading soft labels from {SOFT_LABELS_PATH}")
    soft_data = torch.load(SOFT_LABELS_PATH, map_location="cpu", weights_only=False)
    soft_logits = soft_data["logits"]
    teacher_acc = soft_data["teacher_accuracy"]
    print(f"Soft labels: {soft_logits.shape[0]} samples, "
          f"teacher accuracy was {teacher_acc:.3f}")

    # Initialize wandb
    import wandb
    wandb.init(
        project="nmiai-objdet",
        name=f"markusnet-860m-distill-T{TEMPERATURE}-a{ALPHA}",
        config={
            "model": "MarkusNet-860M (pruned 12 layers)",
            "teacher": TEACHER_MODEL_ID,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "label_smoothing": LABEL_SMOOTHING,
            "teacher_accuracy": teacher_acc,
        },
    )

    # Load student model (pruned 12-layer)
    print("Loading student model (pruned MarkusNet)...")
    student = AutoModelForImageTextToText.from_pretrained(
        str(PRUNED_DIR),
        dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    hidden_size = student.config.text_config.hidden_size
    assert hidden_size == HIDDEN_SIZE

    cls_head = ClassificationHead(hidden_size, NUM_CLASSES).to(device).to(torch.bfloat16)

    # Load student checkpoint
    print(f"Loading student checkpoint from {STUDENT_CHECKPOINT}")
    ckpt = torch.load(STUDENT_CHECKPOINT, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state"])
    cls_head.load_state_dict(ckpt["cls_head_state"])
    prev_acc = ckpt.get("accuracy", 0)
    prev_step = ckpt.get("global_step", 0)
    print(f"Student resumed: step {prev_step}, accuracy {prev_acc:.3f}")
    del ckpt

    student = student.to(device)

    # Load dataset
    with open(SAMPLES_CACHE) as f:
        samples = json.load(f)
    print(f"Dataset: {len(samples)} samples")

    assert len(samples) == soft_logits.shape[0], \
        f"Mismatch: {len(samples)} samples vs {soft_logits.shape[0]} soft labels"

    # Class weights for imbalanced classes
    class_weights = compute_class_weights(samples).to(device).to(torch.bfloat16)
    print(f"Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}, "
          f"mean={class_weights.mean():.2f}")

    dataset = DistillDataset(samples, soft_logits)

    def collate(batch):
        return {
            "images": [b["image"] for b in batch],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "teacher_logits": torch.stack([b["teacher_logits"] for b in batch]),
        }

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate, drop_last=True,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total")

    # Optimizer
    all_params = list(student.parameters()) + list(cls_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    student.train()
    cls_head.train()
    global_step = 0
    best_acc = prev_acc

    print(f"\n=== Distillation Training ===")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Alpha (hard weight): {ALPHA}")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Starting accuracy: {prev_acc:.3f}")
    print()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_hard_loss = 0
        epoch_soft_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, batch in enumerate(loader):
            images = batch["images"]
            labels = batch["labels"].to(device)
            teacher_logits_batch = batch["teacher_logits"].to(device).to(torch.bfloat16)

            inputs = build_processor_inputs(images, processor, device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = student.model(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state
                student_logits = cls_head(hidden)

                total_loss, hard_loss, soft_loss = distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits_batch,
                    hard_labels=labels,
                    class_weights=class_weights,
                    temperature=TEMPERATURE,
                    alpha=ALPHA,
                    label_smoothing=LABEL_SMOOTHING,
                )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            preds = student_logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            epoch_correct += correct
            epoch_total += labels.shape[0]
            epoch_loss += total_loss.item()
            epoch_hard_loss += hard_loss.item()
            epoch_soft_loss += soft_loss.item()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_hard = epoch_hard_loss / (batch_idx + 1)
                avg_soft = epoch_soft_loss / (batch_idx + 1)
                acc = epoch_correct / max(1, epoch_total)
                lr = scheduler.get_last_lr()[0]

                if device.type == "cuda":
                    gpu_mb = torch.cuda.memory_allocated() / 1024**2
                    gpu_str = f" | gpu={gpu_mb:.0f}MB"
                else:
                    gpu_str = ""

                print(
                    f"Step {global_step}/{total_steps} | "
                    f"loss={total_loss.item():.4f} (hard={hard_loss.item():.4f} soft={soft_loss.item():.4f}) | "
                    f"avg={avg_loss:.4f} | acc={acc:.3f} | lr={lr:.2e}{gpu_str}"
                )
                wandb.log({
                    "distill/total_loss": total_loss.item(),
                    "distill/hard_loss": hard_loss.item(),
                    "distill/soft_loss": soft_loss.item(),
                    "distill/avg_total_loss": avg_loss,
                    "distill/avg_hard_loss": avg_hard,
                    "distill/avg_soft_loss": avg_soft,
                    "distill/accuracy": acc,
                    "distill/learning_rate": lr,
                }, step=prev_step + global_step)

            if global_step % SAVE_EVERY == 0:
                acc = epoch_correct / max(1, epoch_total)
                if acc > best_acc:
                    best_acc = acc
                    best_path = OUTPUT_DIR / "best_distill"
                    best_path.mkdir(exist_ok=True, parents=True)
                    torch.save({
                        "model_state": student.state_dict(),
                        "cls_head_state": cls_head.state_dict(),
                        "global_step": prev_step + global_step,
                        "epoch": epoch,
                        "accuracy": acc,
                        "distill_config": {
                            "temperature": TEMPERATURE,
                            "alpha": ALPHA,
                            "teacher": TEACHER_MODEL_ID,
                            "teacher_accuracy": teacher_acc,
                        },
                    }, best_path / "best.pt")
                    print(f"  >> New best distilled: acc={acc:.3f}")

        # Epoch summary
        avg_loss = epoch_loss / steps_per_epoch
        avg_hard = epoch_hard_loss / steps_per_epoch
        avg_soft = epoch_soft_loss / steps_per_epoch
        acc = epoch_correct / max(1, epoch_total)
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS}: "
              f"loss={avg_loss:.4f} (hard={avg_hard:.4f} soft={avg_soft:.4f}) "
              f"acc={acc:.3f} ===\n")
        wandb.log({
            "epoch/total_loss": avg_loss,
            "epoch/hard_loss": avg_hard,
            "epoch/soft_loss": avg_soft,
            "epoch/accuracy": acc,
        }, step=prev_step + global_step)

        # Save at end of each epoch if best
        if acc > best_acc:
            best_acc = acc
            best_path = OUTPUT_DIR / "best_distill"
            best_path.mkdir(exist_ok=True, parents=True)
            torch.save({
                "model_state": student.state_dict(),
                "cls_head_state": cls_head.state_dict(),
                "global_step": prev_step + global_step,
                "epoch": epoch,
                "accuracy": acc,
                "distill_config": {
                    "temperature": TEMPERATURE,
                    "alpha": ALPHA,
                    "teacher": TEACHER_MODEL_ID,
                    "teacher_accuracy": teacher_acc,
                },
            }, best_path / "best.pt")
            print(f"  >> New best distilled (epoch end): acc={acc:.3f}")

    # Final save
    final_path = OUTPUT_DIR / "final_distill"
    final_path.mkdir(exist_ok=True, parents=True)
    torch.save({
        "model_state": student.state_dict(),
        "cls_head_state": cls_head.state_dict(),
        "global_step": prev_step + global_step,
        "accuracy": acc,
        "distill_config": {
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "teacher": TEACHER_MODEL_ID,
            "teacher_accuracy": teacher_acc,
        },
    }, final_path / "final.pt")

    wandb.finish()
    print(f"DONE. Best distilled accuracy: {best_acc:.3f} (was {prev_acc:.3f})")
    print(f"Improvement: {best_acc - prev_acc:+.3f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation for MarkusNet-860M")
    parser.add_argument("--phase", choices=["generate", "train", "both"], default="both",
                        help="Phase to run: generate soft labels, train, or both")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Distillation temperature (default: {TEMPERATURE})")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"Hard label weight (default: {ALPHA})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    # Use CLI args (passed directly to functions, no global override needed)

    if args.phase in ("generate", "both"):
        generate_soft_labels()

    if args.phase in ("train", "both"):
        train_distill()


if __name__ == "__main__":
    main()
