"""LoRA Soup: Train multiple specialized LoRA adapters and merge them.

Strategy: Train N LoRA adapters on different data splits (by product category
groups), then average the LoRA weights to get a single merged adapter that
captures diverse specializations.

Usage:
    # Train all splits sequentially (or dispatch to multiple GPUs)
    CUDA_VISIBLE_DEVICES=0 python train_lora_soup.py train --split 0
    CUDA_VISIBLE_DEVICES=1 python train_lora_soup.py train --split 1
    ...
    # Or train all sequentially on one GPU
    python train_lora_soup.py train --all

    # Merge trained LoRA adapters
    python train_lora_soup.py merge

    # Train + merge in one go
    python train_lora_soup.py run
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

import torch

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# === CONFIG ===
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 512
LORA_RANK = 16        # Lower rank per adapter (merged soup has effective higher rank)
LORA_ALPHA = 16
BATCH_SIZE = 4        # Very small batch for safety
GRAD_ACCUM = 16       # Effective batch = 64
LR = 2e-4
EPOCHS_PER_SPLIT = 1  # Quick training per split, diversity > depth
LOAD_IN_4BIT = True   # QLoRA to save VRAM (only ~8GB free)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "lora_soup_output"
DATA_FILE = BASE_DIR.parent / "data-creation" / "data" / "extra_crops" / "combined_samples.json"
CACHE_DIR = BASE_DIR / "cached_dataset"

# Fallback: if extra_crops doesn't exist, use the original cached dataset
CACHE_SAMPLES = CACHE_DIR / "samples.json"

# Number of splits for the soup
NUM_SPLITS = 4


# ─── Data splitting ──────────────────────────────────────────────────────────

def classify_category(name: str) -> int:
    """Assign a category name to one of NUM_SPLITS groups."""
    nl = name.lower()

    # Group 0: Knekkebrød & bakery
    if any(k in nl for k in ['knekke', 'wasa', 'leksand', 'brød', 'rundstykk',
                              'baguett', 'toast', 'polarbrød', 'skedvi']):
        return 0

    # Group 1: Coffee & hot drinks
    if any(k in nl for k in ['kaffe', 'filtermalt', 'espresso', 'nescafe',
                              'koffeinfri', 'evergood', 'friele', 'ali ',
                              'kakao', 'chai', 'te ', 'twinings']):
        return 1

    # Group 2: Dairy, spreads & cold items
    if any(k in nl for k in ['ost', 'smør', 'yoghurt', 'melk', 'fløte',
                              'rømme', 'brunost', 'majones', 'dressing',
                              'margarin', 'butter', 'cream']):
        return 2

    # Group 3: Everything else (snacks, sauces, condiments, etc.)
    return 3


SPLIT_NAMES = ["bakery_knekke", "coffee_hotdrinks", "dairy_spreads", "other_products"]


def load_and_split_data():
    """Load samples and split into groups."""
    # Try extra_crops first, fall back to cached dataset
    if DATA_FILE.exists():
        print(f"Loading data from {DATA_FILE}")
        with open(DATA_FILE) as f:
            samples = json.load(f)
    elif CACHE_SAMPLES.exists():
        print(f"Loading data from {CACHE_SAMPLES}")
        with open(CACHE_SAMPLES) as f:
            samples = json.load(f)
    else:
        raise FileNotFoundError(f"No data found at {DATA_FILE} or {CACHE_SAMPLES}")

    splits = defaultdict(list)
    for s in samples:
        group = classify_category(s.get("category_name", ""))
        splits[group].append(s)

    print(f"Total samples: {len(samples)}")
    for i in range(NUM_SPLITS):
        print(f"  Split {i} ({SPLIT_NAMES[i]}): {len(splits[i])} samples")

    return splits


# ─── LoRA Training ───────────────────────────────────────────────────────────

def train_split(split_idx: int, samples: list):
    """Train a LoRA adapter on one data split."""
    from PIL import Image
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    split_name = SPLIT_NAMES[split_idx]
    split_dir = OUTPUT_DIR / f"split_{split_idx}_{split_name}"
    adapter_dir = split_dir / "lora_adapter"

    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        print(f"Split {split_idx} ({split_name}) already trained, skipping.")
        return adapter_dir

    print(f"\n{'='*60}")
    print(f"Training split {split_idx}: {split_name} ({len(samples)} samples)")
    print(f"{'='*60}")

    # Load model fresh for each split
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_16bit=not LOAD_IN_4BIT,
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
        random_state=42 + split_idx,  # Different seed per split
        use_rslora=False,
        use_gradient_checkpointing="unsloth",  # Save VRAM
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")

    # Build dataset
    class SplitDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            s = self.items[idx]
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

    dataset = SplitDataset(samples)

    # Calculate steps
    steps_per_epoch = len(dataset) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS_PER_SPLIT
    save_steps = max(100, steps_per_epoch // 2)

    split_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(split_dir),
            num_train_epochs=EPOCHS_PER_SPLIT,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=1,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=4,
            seed=42 + split_idx,
            optim="adamw_8bit",
            report_to="none",
            run_name=f"lora-soup-{split_name}",
        ),
    )

    print(f"Training {total_steps} steps ({EPOCHS_PER_SPLIT} epochs)...")
    trainer.train()

    # Save LoRA adapter
    print(f"Saving adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    adapter_size = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())
    print(f"Adapter size: {adapter_size/1024**2:.1f} MB")

    # Free GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return adapter_dir


# ─── LoRA Weight Merging ─────────────────────────────────────────────────────

def merge_lora_adapters(adapter_dirs: list, weights: list = None,
                        output_dir: Path = None):
    """Average multiple LoRA adapter weights into one merged adapter.

    LoRA adapters are just pairs of low-rank matrices (A, B) per layer.
    We average: merged_A = avg(A_1, ..., A_N), merged_B = avg(B_1, ..., B_N)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "merged_adapter"
    if weights is None:
        weights = [1.0 / len(adapter_dirs)] * len(adapter_dirs)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    print(f"\nMerging {len(adapter_dirs)} LoRA adapters:")
    for d, w in zip(adapter_dirs, weights):
        print(f"  {d.name}: weight={w:.3f}")

    # Load all adapter state dicts
    from safetensors.torch import load_file, save_file

    all_states = []
    for d in adapter_dirs:
        # LoRA adapters can be in safetensors or bin format
        safetensor_file = d / "adapter_model.safetensors"
        bin_file = d / "adapter_model.bin"
        if safetensor_file.exists():
            state = load_file(str(safetensor_file))
        elif bin_file.exists():
            state = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No adapter weights found in {d}")
        all_states.append(state)
        print(f"  Loaded {d.name}: {len(state)} tensors")

    # Verify all adapters have the same keys
    keys = set(all_states[0].keys())
    for i, state in enumerate(all_states[1:], 1):
        if set(state.keys()) != keys:
            missing = keys - set(state.keys())
            extra = set(state.keys()) - keys
            print(f"  WARNING: adapter {i} key mismatch. Missing: {missing}, Extra: {extra}")

    # Weighted average
    merged_state = {}
    for key in all_states[0].keys():
        tensors = [s[key] for s in all_states if key in s]
        ws = weights[:len(tensors)]
        ws_sum = sum(ws)
        ws = [w / ws_sum for w in ws]
        merged_state[key] = sum(w * t.float() for w, t in zip(ws, tensors))
        # Preserve original dtype
        merged_state[key] = merged_state[key].to(all_states[0][key].dtype)

    # Save merged adapter
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config from first adapter
    config_src = adapter_dirs[0] / "adapter_config.json"
    if config_src.exists():
        shutil.copy2(str(config_src), str(output_dir / "adapter_config.json"))

    # Save merged weights
    save_file(merged_state, str(output_dir / "adapter_model.safetensors"))

    # Copy tokenizer from first adapter
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "chat_template.jinja", "merges.txt", "vocab.json"]:
        src = adapter_dirs[0] / tok_file
        if src.exists():
            shutil.copy2(str(src), str(output_dir / tok_file))

    merged_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"\nMerged adapter saved to {output_dir} ({merged_size/1024**2:.1f} MB)")
    print(f"Tensors merged: {len(merged_state)}")

    return output_dir


def merge_lora_ties(adapter_dirs: list, output_dir: Path = None, top_k: float = 0.2):
    """TIES-Merging for LoRA adapters.

    Instead of simple averaging, uses:
    1. Trim: Keep only top_k% highest-magnitude parameters per adapter
    2. Elect Sign: Majority vote on parameter sign
    3. Disjoint Merge: Average only agreeing parameters

    This reduces interference between specialized adapters.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "merged_ties_adapter"

    from safetensors.torch import load_file, save_file

    print(f"\nTIES-Merging {len(adapter_dirs)} adapters (top_k={top_k}):")

    # Load all adapters
    all_states = []
    for d in adapter_dirs:
        safetensor_file = d / "adapter_model.safetensors"
        bin_file = d / "adapter_model.bin"
        if safetensor_file.exists():
            state = load_file(str(safetensor_file))
        elif bin_file.exists():
            state = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No adapter weights in {d}")
        all_states.append(state)

    # Since these are LoRA adapters (not fine-tuned from a base), the "task vector"
    # IS the adapter weight itself (base is zero).

    merged_state = {}
    for key in all_states[0].keys():
        tensors = [s[key].float() for s in all_states if key in s]
        if len(tensors) == 0:
            continue

        # Stack into [N, *shape]
        stacked = torch.stack(tensors)  # [N, ...]

        # Step 1: Trim - keep only top_k% by magnitude per adapter
        flat = stacked.reshape(len(tensors), -1)  # [N, D]
        for i in range(len(tensors)):
            magnitudes = flat[i].abs()
            threshold = torch.quantile(magnitudes[magnitudes > 0], 1.0 - top_k) \
                if (magnitudes > 0).any() else 0.0
            flat[i][magnitudes < threshold] = 0.0

        # Step 2: Elect sign - majority vote
        signs = torch.sign(flat)  # [N, D]
        sign_sum = signs.sum(dim=0)  # [D]
        elected_sign = torch.sign(sign_sum)  # [D]
        # Where tie (sign_sum == 0), use positive
        elected_sign[elected_sign == 0] = 1.0

        # Step 3: Disjoint merge - average only agreeing parameters
        agrees = (signs == elected_sign.unsqueeze(0))  # [N, D]
        flat_masked = flat * agrees.float()  # zero out disagreeing
        counts = agrees.float().sum(dim=0).clamp(min=1)  # [D]
        merged_flat = flat_masked.sum(dim=0) / counts

        merged_state[key] = merged_flat.reshape(tensors[0].shape).to(all_states[0][key].dtype)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    config_src = adapter_dirs[0] / "adapter_config.json"
    if config_src.exists():
        shutil.copy2(str(config_src), str(output_dir / "adapter_config.json"))
    save_file(merged_state, str(output_dir / "adapter_model.safetensors"))
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "chat_template.jinja", "merges.txt", "vocab.json"]:
        src = adapter_dirs[0] / tok_file
        if src.exists():
            shutil.copy2(str(src), str(output_dir / tok_file))

    merged_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"TIES-merged adapter saved to {output_dir} ({merged_size/1024**2:.1f} MB)")
    return output_dir


# ─── CLI ─────────────────────────────────────────────────────────────────────

def cmd_train(args):
    splits = load_and_split_data()

    if args.all:
        for i in range(NUM_SPLITS):
            train_split(i, splits[i])
    elif args.split is not None:
        train_split(args.split, splits[args.split])
    else:
        print("Specify --split N or --all")


def cmd_merge(args):
    adapter_dirs = []
    for i in range(NUM_SPLITS):
        d = OUTPUT_DIR / f"split_{i}_{SPLIT_NAMES[i]}" / "lora_adapter"
        if d.exists() and (d / "adapter_config.json").exists():
            adapter_dirs.append(d)
            print(f"Found adapter: {d}")
        else:
            print(f"Missing adapter: {d}")

    if len(adapter_dirs) < 2:
        print("Need at least 2 trained adapters to merge")
        return

    # Simple average
    merge_lora_adapters(adapter_dirs)

    # TIES merge
    merge_lora_ties(adapter_dirs)

    print("\nBoth merge methods complete. Compare on validation set to pick the best.")


def cmd_run(args):
    """Train all splits, then merge."""
    splits = load_and_split_data()
    for i in range(NUM_SPLITS):
        train_split(i, splits[i])
    cmd_merge(args)


def main():
    parser = argparse.ArgumentParser(description="LoRA Soup: train + merge specialized adapters")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("train", help="Train LoRA adapter(s)")
    p.add_argument("--split", type=int, default=None, help="Which split to train (0-3)")
    p.add_argument("--all", action="store_true", help="Train all splits sequentially")

    subparsers.add_parser("merge", help="Merge trained adapters")
    subparsers.add_parser("run", help="Train all + merge")

    p = subparsers.add_parser("info", help="Show data split info")

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "info":
        load_and_split_data()


if __name__ == "__main__":
    main()
