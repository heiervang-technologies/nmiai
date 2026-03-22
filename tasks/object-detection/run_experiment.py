#!/usr/bin/env python3
"""
MarkusNet Experiment Runner — single entry point for all autoresearch experiments.

Manages training, evaluation, checkpointing, and HuggingFace Hub uploads.
Keeps only best + last checkpoints to save disk space.

Usage:
  # Train classification on external crops
  python run_experiment.py classify \
    --data external_crops \
    --lr 1e-4 --epochs 20 --batch 16 \
    --tag "exp001_cls_external"

  # Train detection head on external shelf data
  python run_experiment.py detect \
    --data polish_1k \
    --head vitdet \
    --lr 1e-4 --epochs 30 --batch 8 \
    --tag "exp002_det_vitdet"

  # Train with LoRA
  python run_experiment.py classify \
    --data external_crops \
    --lora --lora-rank 16 \
    --lr 3e-4 --epochs 10 \
    --tag "exp003_lora_r16"

  # Evaluate a checkpoint
  python run_experiment.py eval \
    --checkpoint runs/exp001/best.pt

  # Upload best checkpoint to HuggingFace
  python run_experiment.py upload \
    --checkpoint runs/exp001/best.pt \
    --tag "exp001_cls_external"

  # Model soup from multiple checkpoints
  python run_experiment.py soup \
    --checkpoints runs/exp001/best.pt runs/exp002/best.pt runs/exp003/best.pt \
    --tag "soup_v1"

  # List all experiments and their scores
  python run_experiment.py list
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VLM_DIR = SCRIPT_DIR / "vlm-approach"
RUNS_DIR = SCRIPT_DIR / "markusnet_runs"
RESULTS_TSV = SCRIPT_DIR / "markusnet_results.tsv"
HF_REPO = "heiertech/markusnet-experiments"

# Evaluation
COCO_DIR = SCRIPT_DIR / "data-creation/data/coco_dataset/train"

# Data sources
DATA_SOURCES = {
    "external_crops": {
        "path": SCRIPT_DIR / "data-creation/data/classifier_crops",
        "type": "crops",
        "description": "18,741 grocery product crops from competition annotations",
    },
    "polish_1k": {
        "path": SCRIPT_DIR / "data-creation/data/pretrain_subset/train",
        "type": "yolo",
        "description": "1,000 Polish shelf images (detection, single class)",
    },
    "polish_full": {
        "path": SCRIPT_DIR / "data-creation/data/external/skus_on_shelves_pl/extracted",
        "type": "coco",
        "annotations": SCRIPT_DIR / "data-creation/data/external/skus_on_shelves_pl/extracted/annotations.json",
        "description": "27,244 Polish shelf images (detection, 7,942 classes)",
    },
    "sku110k": {
        "path": SCRIPT_DIR / "data-creation/data/external/sku110k_extracted/images",
        "type": "coco",
        "annotations": SCRIPT_DIR / "data-creation/data/external/sku110k_extracted/annotations.json",
        "description": "1,000 SKU-110K shelf images (detection, single class)",
    },
    "store_photos": {
        "path": SCRIPT_DIR / "data-creation/data/store_photos",
        "type": "unlabeled",
        "description": "39 Norwegian store photos (batch 1)",
    },
    "store_photos_b2": {
        "path": SCRIPT_DIR / "data-creation/data/store_photos_batch2",
        "type": "unlabeled",
        "description": "54 Norwegian store photos (batch 2)",
    },
    "product_refs": {
        "path": SCRIPT_DIR / "data-creation/data/product_images",
        "type": "reference",
        "description": "345 products × 7 angles (reference images)",
    },
    "sam3_pseudo": {
        "path": SCRIPT_DIR / "data-creation/data/pretrain_subset/sam3_pseudo" if (SCRIPT_DIR / "data-creation/data/pretrain_subset/sam3_pseudo").exists() else None,
        "type": "yolo",
        "description": "100 SAM3 pseudo-labeled store photos (356 classes)",
    },
}


def cmd_list(args):
    """List all experiments and scores."""
    if not RESULTS_TSV.exists():
        print("No experiments yet.")
        return

    print(f"{'Tag':<35} {'Combined':>8} {'Det':>8} {'Cls':>8} {'Data':>15} {'Date':>12}")
    print("-" * 95)
    with open(RESULTS_TSV) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 7:
                ts, tag, combined, det, cls, data, desc = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
                date = ts[:10] if ts else ""
                print(f"{tag:<35} {combined:>8} {det:>8} {cls:>8} {data:>15} {date:>12}")


def cmd_sources(args):
    """List available data sources."""
    print("Available data sources:\n")
    for name, info in DATA_SOURCES.items():
        path = info.get("path")
        exists = "OK" if path and path.exists() else "MISSING"
        print(f"  [{exists}] {name}")
        print(f"    {info['description']}")
        if path:
            print(f"    {path}")
        print()


def cmd_classify(args):
    """Train MarkusNet classification on crop data."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
    from PIL import Image

    tag = args.tag or f"cls_{args.data}_{int(time.time())}"
    run_dir = RUNS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Experiment: {tag} ===")
    print(f"Task: classification")
    print(f"Data: {args.data}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch}")
    print(f"LoRA: {args.lora} (rank={args.lora_rank})")
    print(f"Output: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading MarkusNet...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_name = "Qwen/Qwen3.5-VL-3B" if args.model_size == "3b" else "Qwen/Qwen3.5-VL-0.8B"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

    # Setup LoRA if requested
    if args.lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            print("peft not installed. Run: uv pip install peft")
            return

    # Add classification head
    hidden_dim = 1024  # MarkusNet decoder output dim
    num_classes = args.num_classes
    cls_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, num_classes),
    ).to(device).to(torch.float16)

    # If text embeddings exist, initialize cls head with them
    text_emb_path = VLM_DIR / "exported/category_text_embeddings.pt"
    if text_emb_path.exists() and num_classes == 356:
        data = torch.load(text_emb_path, map_location=device, weights_only=True)
        text_embs = data["embeddings"]  # [356, 1024]
        cls_head[-1].weight.data = text_embs.to(torch.float16)
        print("Initialized classification head from text embeddings")

    # Dataset
    data_info = DATA_SOURCES.get(args.data)
    if not data_info or not data_info.get("path") or not data_info["path"].exists():
        print(f"Data source '{args.data}' not found")
        return

    class CropDataset(Dataset):
        def __init__(self, root, processor):
            self.samples = []
            self.processor = processor
            for cat_dir in sorted(root.iterdir()):
                if not cat_dir.is_dir():
                    continue
                try:
                    cat_id = int(cat_dir.name)
                except ValueError:
                    continue
                for img_path in cat_dir.glob("*.jpg"):
                    self.samples.append((img_path, cat_id))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            return img, label

    dataset = CropDataset(data_info["path"], processor)
    print(f"Dataset: {len(dataset)} samples")

    n_val = max(1, len(dataset) // 10)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val],
                                     generator=torch.Generator().manual_seed(42))

    # Simple collate - process images in batch
    def collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Optimizer
    params = list(cls_head.parameters())
    if args.lora:
        params += [p for p in model.parameters() if p.requires_grad]
    elif not args.freeze_backbone:
        params += list(model.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        cls_head.train()
        total_loss = 0
        n_batches = 0

        for images, labels in train_loader:
            labels = labels.to(device)
            # Get vision features from MarkusNet
            # Use processor to prepare inputs
            inputs = processor(
                text=["classify"] * len(images),
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs, output_hidden_states=True)
                # Pool last hidden state
                hidden = outputs.hidden_states[-1]
                pooled = hidden.mean(dim=1)
                logits = cls_head(pooled)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        cls_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.to(device)
                inputs = processor(
                    text=["classify"] * len(images),
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    pooled = hidden.mean(dim=1)
                    logits = cls_head(pooled)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        acc = correct / max(total, 1) * 100
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} val_acc={acc:.1f}% (best={best_acc:.1f}%)")

        # Save checkpoints (only best + last)
        torch.save({
            "model_state": model.state_dict() if not args.lora else None,
            "cls_head": cls_head.state_dict(),
            "lora_state": model.state_dict() if args.lora else None,
            "epoch": epoch + 1,
            "acc": acc,
            "config": vars(args),
        }, run_dir / "last.pt")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict() if not args.lora else None,
                "cls_head": cls_head.state_dict(),
                "lora_state": model.state_dict() if args.lora else None,
                "epoch": epoch + 1,
                "acc": acc,
                "config": vars(args),
            }, run_dir / "best.pt")

    print(f"\nBest accuracy: {best_acc:.1f}%")
    print(f"Saved: {run_dir / 'best.pt'}")

    # Log result
    log_result(tag, 0, 0, best_acc / 100, args.data,
               f"cls lr={args.lr} ep={args.epochs} lora={args.lora}")

    # Auto-upload if requested
    if args.upload:
        upload_to_hf(run_dir / "best.pt", tag)


def cmd_eval(args):
    """Evaluate a checkpoint on competition val set."""
    print(f"Evaluating: {args.checkpoint}")
    # TODO: implement full detection eval using eval_honest.py
    print("Use eval_honest.py for YOLO detection eval")
    print(f"  python eval_honest.py {args.checkpoint}")


def cmd_upload(args):
    """Upload checkpoint to HuggingFace Hub."""
    upload_to_hf(args.checkpoint, args.tag)


def cmd_soup(args):
    """Create model soup from multiple checkpoints."""
    import torch
    print(f"Creating model soup from {len(args.checkpoints)} checkpoints...")

    states = []
    for ckpt_path in args.checkpoints:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "cls_head" in state:
            states.append(state["cls_head"])
        elif "model_state" in state and state["model_state"]:
            states.append(state["model_state"])
        else:
            states.append(state)

    # Uniform average
    soup = {}
    for key in states[0]:
        soup[key] = sum(s[key].float() for s in states) / len(states)
        soup[key] = soup[key].to(states[0][key].dtype)

    tag = args.tag or f"soup_{int(time.time())}"
    out_dir = RUNS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"cls_head": soup, "soup_sources": args.checkpoints}, out_dir / "best.pt")
    print(f"Soup saved: {out_dir / 'best.pt'}")

    if args.upload:
        upload_to_hf(out_dir / "best.pt", tag)


def log_result(tag, det_map, cls_map, combined, data_source, description):
    """Append result to TSV."""
    header = "timestamp\ttag\tcombined\tdet_map50\tcls_map50\tdata\tdescription\n"
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write(header)
    with open(RESULTS_TSV, "a") as f:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        f.write(f"{ts}\t{tag}\t{combined:.6f}\t{det_map:.6f}\t{cls_map:.6f}\t{data_source}\t{description}\n")
    print(f"Result logged to {RESULTS_TSV}")


def upload_to_hf(checkpoint_path, tag):
    """Upload checkpoint to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=f"{tag}/{Path(checkpoint_path).name}",
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"Experiment: {tag}",
        )
        print(f"Uploaded to https://huggingface.co/{HF_REPO}/tree/main/{tag}/")
    except Exception as e:
        print(f"Upload failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="MarkusNet Experiment Runner")
    subparsers = parser.add_subparsers(dest="command")

    # list
    subparsers.add_parser("list", help="List all experiments")

    # sources
    subparsers.add_parser("sources", help="List available data sources")

    # classify
    p = subparsers.add_parser("classify", help="Train classification")
    p.add_argument("--data", default="external_crops", help="Data source name")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=356)
    p.add_argument("--tag", default=None)
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    p.add_argument("--model-size", default="0.8b", choices=["0.8b", "3b"])
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--upload", action="store_true", help="Upload best to HF Hub")

    # detect
    p = subparsers.add_parser("detect", help="Train detection head")
    p.add_argument("--data", default="polish_1k")
    p.add_argument("--head", default="vitdet", choices=["vitdet", "detr", "centernet"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--tag", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--upload", action="store_true")

    # eval
    p = subparsers.add_parser("eval", help="Evaluate checkpoint")
    p.add_argument("--checkpoint", required=True)

    # upload
    p = subparsers.add_parser("upload", help="Upload to HF Hub")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True)

    # soup
    p = subparsers.add_parser("soup", help="Model soup from checkpoints")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--upload", action="store_true")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "sources":
        cmd_sources(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "detect":
        print("Detection training not yet implemented. Use train_overnight.py for now.")
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "soup":
        cmd_soup(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
