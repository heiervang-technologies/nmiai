#!/bin/bash
# NM i AI 2026 - Sandbox Test Runner (EXACT COMPETITION REPLICA)
#
# Usage:
#   ./test_submission.sh <submission.zip> [test_images_dir]
#
# Modes:
#   With Docker (default):    runs inside the sandbox container
#   Without Docker (fallback): runs natively with sandbox_runner.py
#
# Examples:
#   ./test_submission.sh ~/submission.zip
#   ./test_submission.sh ~/submission.zip ~/test_images/
#   QUICK=1 ./test_submission.sh ~/submission.zip   # only 5 images
#   NO_DOCKER=1 ./test_submission.sh ~/submission.zip  # skip Docker
#
# Competition constraints enforced:
#   - Python 3.11
#   - 4 vCPU, 8 GB RAM, L4 24GB VRAM
#   - 300s timeout with hard kill
#   - No network access
#   - Exact package versions pinned

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ZIP_PATH="$(realpath "${1:?Usage: ./test_submission.sh <submission.zip> [test_images_dir]}")"
DEFAULT_IMAGES="$SCRIPT_DIR/../data-creation/data/coco_dataset/train/images"
DEFAULT_ANNOTATIONS="$SCRIPT_DIR/../data-creation/data/coco_dataset/train/annotations.json"
TEST_IMAGES="$(realpath "${2:-$DEFAULT_IMAGES}")"
ANNOTATIONS="$(realpath "${3:-$DEFAULT_ANNOTATIONS}")"
IMAGE_NAME="nmiai-sandbox"
TIMEOUT=360  # 300s inference + 60s model loading grace period

echo "============================================================"
echo "  NM i AI 2026 - Submission Tester (EXACT REPLICA)"
echo "============================================================"
echo "ZIP:    $ZIP_PATH ($(du -h "$ZIP_PATH" | cut -f1))"
echo "Images: $TEST_IMAGES ($(ls "$TEST_IMAGES"/*.jpg "$TEST_IMAGES"/*.jpeg 2>/dev/null | wc -l) images)"
echo ""

# Quick mode: copy only 5 images to a temp dir
QUICK_DIR=""
if [ "${QUICK:-}" = "1" ]; then
    QUICK_DIR=$(mktemp -d)
    count=0
    for f in "$TEST_IMAGES"/*.jpg "$TEST_IMAGES"/*.jpeg; do
        [ -f "$f" ] || continue
        cp "$f" "$QUICK_DIR/"
        count=$((count + 1))
        [ "$count" -ge 5 ] && break
    done
    TEST_IMAGES="$QUICK_DIR"
    echo "QUICK MODE: Using $(ls "$QUICK_DIR" | wc -l) images"
    echo ""
fi

cleanup() {
    if [ -n "$QUICK_DIR" ] && [ -d "$QUICK_DIR" ]; then
        rm -rf "$QUICK_DIR"
    fi
}
trap cleanup EXIT

# Skip Docker if requested
if [ "${NO_DOCKER:-}" = "1" ]; then
    echo "Running natively (NO_DOCKER=1)..."
    python3 "$SCRIPT_DIR/sandbox_runner.py" \
        --zip "$ZIP_PATH" \
        --input "$TEST_IMAGES" \
        --output /tmp/nmiai_predictions.json \
        --timeout "$TIMEOUT"
    exit $?
fi

# Check if Docker image exists or needs rebuild
NEEDS_BUILD=0
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    NEEDS_BUILD=1
    echo "Docker image '$IMAGE_NAME' not found, building..."
else
    # Check if Dockerfile is newer than image
    IMAGE_CREATED=$(docker inspect --format='{{.Created}}' "$IMAGE_NAME" 2>/dev/null)
    DOCKERFILE_MOD=$(stat -c %Y "$SCRIPT_DIR/Dockerfile" 2>/dev/null || stat -f %m "$SCRIPT_DIR/Dockerfile" 2>/dev/null)
    echo "Docker image '$IMAGE_NAME' found"
    echo "  To rebuild: docker build -t $IMAGE_NAME $SCRIPT_DIR"
fi

if [ "$NEEDS_BUILD" = "1" ] || [ "${REBUILD:-}" = "1" ]; then
    echo "Building Docker image '$IMAGE_NAME'..."
    echo "(This will take several minutes on first run)"
    echo ""
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
    echo ""
fi

# Determine GPU flag
GPU_FLAG=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_FLAG="--gpus all"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "GPU: NOT AVAILABLE (WARNING: competition uses L4 GPU!)"
fi

echo ""
echo "Competition constraints:"
echo "  CPU:     4 vCPU"
echo "  RAM:     8 GB (hard limit)"
echo "  GPU:     L4 24GB VRAM"
echo "  Timeout: ${TIMEOUT}s (300s inference + 60s loading grace)"
echo "  Network: DISABLED"
echo ""
echo "Running sandbox container..."
echo "------------------------------------------------------------"

# Run the container with EXACT competition constraints
ANNOTATIONS_FLAG=""
if [ -f "$ANNOTATIONS" ]; then
    ANNOTATIONS_FLAG="-v $ANNOTATIONS:/data/annotations.json:ro"
    echo "Annotations: $ANNOTATIONS"
fi

docker run --rm \
    $GPU_FLAG \
    -v "$ZIP_PATH":/submission.zip:ro \
    -v "$TEST_IMAGES":/data/images:ro \
    $ANNOTATIONS_FLAG \
    --memory=8g \
    --memory-swap=8g \
    --cpus=4 \
    --network=none \
    --pids-limit=256 \
    "$IMAGE_NAME" \
    --zip /submission.zip \
    --input /data/images \
    --output /predictions.json \
    --annotations /data/annotations.json \
    --timeout "$TIMEOUT"
