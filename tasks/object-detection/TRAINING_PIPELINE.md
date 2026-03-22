# Optimal Training Pipeline

## Overview

Three-stage curriculum training, each stage 30 minutes on Blackwell 6000.

```
Stage 1: Pre-train (external data only, detection-only)
    ↓ checkpoint
Stage 2: Fine-tune (competition data, all 356 classes)
    ↓ checkpoint
Stage 3: Polish (competition + best external, full augmentation)
    ↓ export ONNX → sandbox validation → submit
```

## Stage 1: Pre-train on External Data

**Goal**: Learn WHERE products are on shelves (detection transfer)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Data | v1_polish_5k (5K Polish shelf images) | Best distribution match |
| Classes | 1 (single "product" class) | Detection-only transfer |
| Base model | yolov8x.pt (COCO pretrained) | Standard starting point |
| Image size | 1280 | Match competition inference |
| Batch | auto (likely 32-48 on 96GB) | Maximize GPU utilization |
| Epochs | 20-30 | Time-limited to 30 min |
| LR | 0.001 | Standard AdamW |
| Augmentation | OFF | Clean data first |
| Freeze | None | Full model training |
| Time limit | 30 minutes | Hard budget |

**Output**: `pretrain_best.pt` — a YOLO model that knows how to find products on shelves but doesn't know Norwegian product categories.

**Validation**: Run eval_honest.py on detection mAP only. Target: >0.6 det_mAP@0.5

## Stage 2: Fine-tune on Competition Data

**Goal**: Learn WHAT the 356 Norwegian products are

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Data | All 248 competition images (YOLO format, 356 classes) | Target domain |
| Base model | `pretrain_best.pt` from Stage 1 | Transfer learned features |
| Image size | 1280 | Same as inference |
| Batch | auto | |
| Epochs | 50-100 | Small dataset, needs more epochs |
| LR | 0.0005 | Lower LR for fine-tuning (don't destroy pretrained features) |
| LR schedule | Cosine decay to 0.00001 | Smooth convergence |
| Warmup | 5 epochs | Gradual unfreezing |
| Augmentation | Minimal (fliplr=0.0, no mosaic yet) | Phase 1 still |
| Freeze backbone | First 3 epochs | Protect pretrained detection features |
| Patience | 20 | Let it converge |
| Time limit | 30 minutes | |

**Output**: `finetune_best.pt` — a YOLO model that can detect AND classify Norwegian products.

**Validation**: This is where we use competition data for training, so we can't validate honestly anymore. But we can use leave-one-out or k-fold for sanity checks.

## Stage 3: Polish with Full Data + Augmentation

**Goal**: Maximum performance, prepare for submission

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Data | Competition 248 + best external subset | Everything we have |
| Base model | `finetune_best.pt` from Stage 2 | Already knows products |
| Image size | 1280 | |
| Batch | auto | |
| Epochs | 30 | Short polish stage |
| LR | 0.0001 | Very low, don't overfit |
| Augmentation | mosaic=0.5, mixup=0.1, fliplr=0.5, hsv | Now augmentation helps |
| Close mosaic | 10 epochs before end | Clean final epochs |
| Time limit | 30 minutes | |

**Output**: `final_best.pt` → export to ONNX → sandbox validation → submit

## Alternative: Skip Stage 3

If Stage 2 produces strong enough results, skip Stage 3 to avoid overfitting risk. The simpler the pipeline, the more robust.

## Classifier Integration

Independently, the SigLIP classifier is being built:
1. Pre-compute text embeddings for 356 product names
2. Use as classification head weights
3. Fine-tune on competition crops
4. Quantize for submission

The final submission combines:
- YOLO ONNX for detection (~132MB)
- SigLIP classifier for re-ranking classifications (~100-200MB)
- Total must fit in 420MB ZIP

## Experiments Before Final Training

Before committing to the final pipeline, we need answers from the autoresearch loop:

1. **Which external data helps most?** (v1 vs v2 vs v3 vs v4 experiments)
2. **What is the best LR for Stage 1?** (clerk already tested 0.001 vs 0.002)
3. **Does more data help or hurt?** (5K vs 10K Polish)
4. **Does SKU-110K add value on top of Polish?** (v2 vs v1)

Only after these questions are answered do we commit to the final 3-stage pipeline.

## Time Budget

| Stage | Time | GPU | Status |
|-------|------|-----|--------|
| Stage 1: Pre-train | 30 min | Blackwell 6000 | Running (clerk has done 2 experiments) |
| Stage 2: Fine-tune | 30 min | Blackwell 6000 | Waiting for Stage 1 results |
| Stage 3: Polish | 30 min | Blackwell 6000 | Optional |
| Eval + export | 10 min | Any | |
| Sandbox validation | 10 min | L4 or emulated | Must pass before submission |
| **Total** | **~2 hours** | | |
