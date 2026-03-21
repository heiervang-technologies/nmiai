#!/bin/bash
# GPU 1 Experiment Queue - runs sequentially, never idle
set -o pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export WANDB_API_KEY="${WANDB_API_KEY}"
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/nmiai/tasks/object-detection/vlm-approach

# Swap in 201K combined samples as the primary dataset
cp cached_dataset/samples.json cached_dataset/samples_original.json 2>/dev/null
cp /workspace/nmiai/tasks/object-detection/data-creation/data/extra_crops/combined_samples.json cached_dataset/samples.json
echo "Swapped in 201K combined_samples.json"

echo ""
echo "============================================================"
echo "EXP 1: Classification on 201K crops (batch=64, stage 3)"
echo "Started: $(date)"
echo "============================================================"
uv run python train_overnight.py --stage 3 --coco-dir external_datasets/coco 2>&1 | tee /tmp/exp1_gpu1.log || echo "EXP 1 FINISHED/FAILED at $(date), continuing..."

echo ""
echo "============================================================"
echo "EXP 2: Lightning LoRA progressive dropout"
echo "Started: $(date)"
echo "============================================================"
uv run python train_lightning.py 2>&1 | tee /tmp/exp2_gpu1.log || echo "EXP 2 FINISHED/FAILED at $(date), continuing..."

echo ""
echo "============================================================"
echo "EXP 3: Knowledge distillation"
echo "Started: $(date)"
echo "============================================================"
uv run python train_distill.py 2>&1 | tee /tmp/exp3_gpu1.log || echo "EXP 3 FINISHED/FAILED at $(date), continuing..."

# Restore original samples
mv cached_dataset/samples_original.json cached_dataset/samples.json 2>/dev/null

echo ""
echo "============================================================"
echo "ALL GPU 1 EXPERIMENTS COMPLETE: $(date)"
echo "============================================================"
