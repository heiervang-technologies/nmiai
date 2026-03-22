#!/bin/bash
# Autoresearch loop: data mixture optimization
# Runs experiments sequentially, evaluates each, logs results
set -e

VENV="/home/me/ht/nmiai/.venv/bin/python"
OD_DIR="/home/me/ht/nmiai/tasks/object-detection"
PRETRAIN="$OD_DIR/data-creation/data/pretrain_subset/dataset.yaml"

echo "=== AUTORESEARCH: Data Mixture Optimization ==="
echo "Start: $(date)"

# exp001: Pretrain subset (2045 imgs), detection-only, 30 min
echo ""
echo "=== EXP001: Pretrain subset detection-only ==="
$VENV $OD_DIR/train_mixture.py "$PRETRAIN" \
  --max-minutes 30 \
  --experiment-id exp001_pretrain_detonly \
  --description "Pretrain 2K (Polish+Grocery+SKU110K) det-only 30min" \
  --unique-images 2045 \
  --data-sources "polish_1k,grocery_45,sku110k_1k" \
  --device 0

echo ""
echo "=== AUTORESEARCH COMPLETE ==="
echo "End: $(date)"
echo "Results: $OD_DIR/data_experiment_results.tsv"
