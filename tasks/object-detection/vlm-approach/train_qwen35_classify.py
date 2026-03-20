"""
Fine-tune Qwen3.5-0.8B for grocery product classification using Unsloth.

Usage: CUDA_VISIBLE_DEVICES=0 uv run python train_qwen35_classify.py
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from datasets import Dataset

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# === CONFIG ===
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 512
LORA_RANK = 32
LORA_ALPHA = 32
EPOCHS = 3
BATCH_SIZE = 64
GRAD_ACCUM = 1
LR = 2e-4
OUTPUT_DIR = Path(__file__).parent / "training_output"
CACHE_DIR = Path(__file__).parent / "cached_dataset"

DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"
COCO_IMAGES = DATA_ROOT / "coco_dataset" / "train" / "images"


def prepare_dataset():
    """Load COCO crops and cache to disk."""
    cache_file = CACHE_DIR / "samples.json"
    crop_dir = CACHE_DIR / "crops"

    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    print("Preparing dataset (first run, will cache)...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(exist_ok=True)

    with open(COCO_ANNOTATIONS) as f:
        coco = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    image_anns = defaultdict(list)
    for ann in coco["annotations"]:
        image_anns[ann["image_id"]].append(ann)

    samples = []
    skipped = 0

    for img_id, anns in image_anns.items():
        img_path = COCO_IMAGES / id_to_file[img_id]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += len(anns)
            continue

        for ann in anns:
            bbox = ann["bbox"]
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(img.width, int(bbox[0] + bbox[2]))
            y2 = min(img.height, int(bbox[1] + bbox[3]))

            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            crop = img.crop((x1, y1, x2, y2))
            max_dim = max(crop.size)
            if max_dim > 512:
                scale = 512 / max_dim
                crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
            elif max_dim < 64:
                scale = 64 / max_dim
                crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))

            crop_path = crop_dir / f"{ann['id']}.jpg"
            crop.save(str(crop_path), quality=90)

            samples.append({
                "crop_path": str(crop_path),
                "category_name": id_to_name[ann["category_id"]],
                "category_id": ann["category_id"],
            })

        if len(samples) % 5000 < len(anns):
            print(f"  Processed {len(samples)} crops...")

    with open(cache_file, "w") as f:
        json.dump(samples, f)

    print(f"Cached {len(samples)} samples ({skipped} skipped)")
    return samples


class LazyVisionDataset(torch.utils.data.Dataset):
    """Lazy-loading dataset that opens images on-the-fly."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        crop = Image.open(s["crop_path"]).convert("RGB")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop},
                        {"type": "text", "text": "What grocery product is this? Reply with only the exact product name."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": s["category_name"]},
                    ],
                },
            ],
        }


def main():
    print("=== Qwen3.5-0.8B Grocery Classification Training ===")
    print(f"Config: batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, lr={LR}, epochs={EPOCHS}")
    print(f"LoRA: r={LORA_RANK}, alpha={LORA_ALPHA}")

    # Prepare dataset first (CPU only, caches to disk)
    samples = prepare_dataset()
    print(f"Total samples: {len(samples)}")

    # Now load model
    print(f"\nLoading {MODEL_NAME}...")
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=42,
        use_rslora=False,
        use_gradient_checkpointing=False,  # Skip for speed, plenty VRAM
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after model load: {gpu_mem:.1f} GB")

    # Build lazy dataset (images loaded on-the-fly, not all in memory)
    print("\nBuilding lazy dataset...")
    dataset = LazyVisionDataset(samples)
    print(f"Dataset ready: {len(dataset)} samples (lazy loading)")

    # Training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=4,
            seed=42,
            optim="adamw_8bit",
            report_to="wandb",
            run_name="qwen35-0.8b-grocery-classify",
        ),
    )

    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory before training: {gpu_mem:.1f} GB")
    print(f"\nStarting training... ({len(dataset)} samples, {EPOCHS} epochs, batch={BATCH_SIZE})")

    trainer.train()

    # Save LoRA adapter
    adapter_dir = OUTPUT_DIR / "lora_classify"
    print(f"\nSaving LoRA adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    adapter_size = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())
    print(f"LoRA adapter size: {adapter_size/1024**2:.1f} MB")
    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()
