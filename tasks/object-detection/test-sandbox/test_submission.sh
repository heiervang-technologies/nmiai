#!/bin/bash
# NM i AI 2026 - Sandbox Test Runner
#
# Usage:
#   ./test_submission.sh <submission.zip> [test_images_dir]
#
# Modes:
#   With Docker (recommended):  runs inside the sandbox container
#   Without Docker (fallback):  runs natively with sandbox_runner.py
#
# Examples:
#   ./test_submission.sh ~/submission.zip
#   ./test_submission.sh ~/submission.zip ~/test_images/
#   QUICK=1 ./test_submission.sh ~/submission.zip   # only 5 images
#   NO_DOCKER=1 ./test_submission.sh ~/submission.zip  # skip Docker

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ZIP_PATH="$(realpath "${1:?Usage: ./test_submission.sh <submission.zip> [test_images_dir]}")"
DEFAULT_IMAGES="$SCRIPT_DIR/../data-creation/data/coco_dataset/train/images"
TEST_IMAGES="$(realpath "${2:-$DEFAULT_IMAGES}")"
IMAGE_NAME="nmiai-sandbox"
TIMEOUT=300

echo "============================================================"
echo "  NM i AI 2026 - Submission Tester"
echo "============================================================"
echo "ZIP:    $ZIP_PATH ($(du -h "$ZIP_PATH" | cut -f1))"
echo "Images: $TEST_IMAGES"
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

# Check if Docker image exists, build if not
if docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Docker image '$IMAGE_NAME' found"
else
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
    echo "GPU: available (using --gpus all)"
else
    echo "GPU: not available (CPU mode)"
fi

echo ""
echo "Running sandbox container..."
echo "------------------------------------------------------------"

# Run the container
# Mount: ZIP as /submission.zip, images as /data/images (read-only)
docker run --rm \
    $GPU_FLAG \
    -v "$ZIP_PATH":/submission.zip:ro \
    -v "$TEST_IMAGES":/data/images:ro \
    --memory=16g \
    --cpus=4 \
    "$IMAGE_NAME" \
    --zip /submission.zip \
    --input /data/images \
    --output /predictions.json \
    --timeout "$TIMEOUT"
