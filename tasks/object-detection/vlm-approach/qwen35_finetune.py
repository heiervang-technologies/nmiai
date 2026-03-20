"""
Fine-tune Qwen3.5-4B (native VLM) for grocery product classification.

Uses Unsloth for memory-efficient LoRA training on RTX 3090 (24GB).
Trains the model to classify product crops by outputting category names.

This creates a training-time classifier that can:
1. Generate pseudo-labels for unlabeled images
2. Be distilled into a smaller model for submission
3. Re-classify ambiguous YOLO detections

Usage:
  On titan: CUDA_VISIBLE_DEVICES=0 uv run python qwen35_finetune.py
"""

import json
from pathlib import Path

import torch
from unsloth import FastVisionModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from PIL import Image


# === CONFIG ===
MODEL_NAME = "unsloth/Qwen3.5-4B"  # Native VLM, fits on 24GB with LoRA
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 16
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4

# Data paths (adjust for titan)
DATA_ROOT = Path("/home/me/nmiai-vlm/data")
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"
COCO_IMAGES = DATA_ROOT / "coco_dataset" / "train" / "images"
OUTPUT_DIR = Path("/home/me/nmiai-vlm/qwen-finetune/output")


def load_coco_classification_data():
    """Convert COCO annotations into VLM classification training data.

    Each sample: image crop + prompt -> category name.
    """
    with open(COCO_ANNOTATIONS) as f:
        coco = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    samples = []
    for ann in coco["annotations"]:
        img_path = COCO_IMAGES / id_to_file[ann["image_id"]]
        bbox = ann["bbox"]  # [x, y, w, h]
        cat_name = id_to_name[ann["category_id"]]
        cat_id = ann["category_id"]

        samples.append({
            "image_path": str(img_path),
            "bbox": bbox,
            "category_name": cat_name,
            "category_id": cat_id,
        })

    return samples


def create_training_dataset(samples, max_samples=None):
    """Create HuggingFace Dataset with image crops and classification prompts."""

    if max_samples:
        samples = samples[:max_samples]

    processed = []
    for s in samples:
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            x, y, w, h = s["bbox"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img.width, int(x + w))
            y2 = min(img.height, int(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))

            processed.append({
                "image": crop,
                "question": "What grocery product is this? Reply with only the product name.",
                "answer": s["category_name"],
                "category_id": s["category_id"],
            })
        except Exception as e:
            continue

    return processed


def format_for_training(sample):
    """Format a sample for SFT training with Qwen3.5 vision format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]},
                ],
            },
        ],
    }


def main():
    device = "cuda:0"
    print(f"Loading {MODEL_NAME}...")

    # Load model with LoRA
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # BF16 recommended over QLoRA for Qwen3.5
        dtype=torch.bfloat16,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
    )

    print("Model loaded with LoRA adapters")

    # Load data
    print("Loading COCO classification data...")
    samples = load_coco_classification_data()
    print(f"Total annotations: {len(samples)}")

    print("Creating training dataset (cropping images)...")
    processed = create_training_dataset(samples)
    print(f"Processed samples: {len(processed)}")

    # Format for training
    formatted = [format_for_training(s) for s in processed]
    dataset = Dataset.from_list(formatted)

    print(f"Training dataset size: {len(dataset)}")

    # Training config
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        dataloader_num_workers=4,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter
    adapter_path = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Saved LoRA adapter to {adapter_path}")

    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
