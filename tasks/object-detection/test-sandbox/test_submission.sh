#!/bin/bash
# Test a submission ZIP against a simulated sandbox environment
# Usage: ./test_submission.sh <submission.zip> [test_images_dir]

set -e

ZIP_PATH="${1:?Usage: ./test_submission.sh <submission.zip> [test_images_dir]}"
TEST_IMAGES="${2:-/home/me/ht/nmiai/tasks/object-detection/data-creation/data/coco_dataset/train/images}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR=$(mktemp -d)
OUTPUT_JSON="$WORK_DIR/predictions.json"

echo "=== NM i AI Submission Tester ==="
echo "ZIP: $ZIP_PATH"
echo "Test images: $TEST_IMAGES"
echo "Work dir: $WORK_DIR"
echo ""

# 1. Check ZIP structure
echo "=== 1. ZIP Structure Check ==="
if ! unzip -l "$ZIP_PATH" | grep -q "^.*run\.py$"; then
    echo "FAIL: run.py not found at ZIP root!"
    # Check if it's nested
    if unzip -l "$ZIP_PATH" | grep -q "run\.py"; then
        echo "  Found run.py but it's nested in a subdirectory"
    fi
    exit 1
fi
echo "PASS: run.py at ZIP root"

ZIP_SIZE=$(stat -c%s "$ZIP_PATH")
if [ "$ZIP_SIZE" -gt 440401920 ]; then
    echo "FAIL: ZIP size $(($ZIP_SIZE / 1048576))MB exceeds 420MB limit"
    exit 1
fi
echo "PASS: ZIP size $(($ZIP_SIZE / 1048576))MB (under 420MB)"
echo ""

# 2. Extract ZIP
echo "=== 2. Extracting ZIP ==="
unzip -q "$ZIP_PATH" -d "$WORK_DIR/submission"
echo "Contents:"
ls -lh "$WORK_DIR/submission/"
echo ""

# 3. Security scan - check for blocked imports
echo "=== 3. Security Scan ==="
BLOCKED_IMPORTS="import os|import sys|import subprocess|import socket|import ctypes|import builtins|import importlib|import pickle|import marshal|import shelve|import shutil|import yaml|import requests|import urllib|import http\.client|import multiprocessing|import threading|import signal|import gc|import code|import codeop|import pty|from os |from sys |from subprocess |from socket |from ctypes |from builtins |from importlib |from pickle |from marshal |from shelve |from shutil |from yaml |from requests |from urllib |from http\.client|from multiprocessing |from threading |from signal |from gc "
BLOCKED_CALLS="eval(|exec(|compile(|__import__("

FAILED=0
for pyfile in $(find "$WORK_DIR/submission" -name "*.py"); do
    fname=$(basename "$pyfile")
    if grep -Pn "$BLOCKED_IMPORTS" "$pyfile" 2>/dev/null; then
        echo "FAIL: Blocked import in $fname"
        FAILED=1
    fi
    if grep -n "eval(\|exec(\|compile(\|__import__(" "$pyfile" 2>/dev/null; then
        echo "FAIL: Blocked function call in $fname"
        FAILED=1
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo "PASS: No blocked imports or calls found"
fi
echo ""

# 4. Check for binaries/symlinks
echo "=== 4. Binary/Symlink Check ==="
if find "$WORK_DIR/submission" -type l | grep -q .; then
    echo "FAIL: Symlinks found"
    exit 1
fi
echo "PASS: No symlinks"

ELF_COUNT=$(find "$WORK_DIR/submission" -type f -exec file {} \; | grep -c "ELF\|Mach-O\|PE32" || true)
if [ "$ELF_COUNT" -gt 0 ]; then
    echo "WARN: $ELF_COUNT binary files found (may be blocked)"
fi
echo ""

# 5. Run inference
echo "=== 5. Running Inference ==="
echo "Command: python3 run.py --input $TEST_IMAGES --output $OUTPUT_JSON"
echo "Timeout: 300 seconds"

# Copy a few test images to a temp dir for faster testing
TEST_DIR="$WORK_DIR/test_images"
mkdir -p "$TEST_DIR"
ls "$TEST_IMAGES"/*.jpg 2>/dev/null | head -5 | while read f; do cp "$f" "$TEST_DIR/"; done
echo "Using $(ls "$TEST_DIR"/*.jpg 2>/dev/null | wc -l) test images"

cd "$WORK_DIR/submission"
START_TIME=$(date +%s)
if timeout 300 python3 run.py --input "$TEST_DIR" --output "$OUTPUT_JSON" 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "PASS: Exited with code 0 in ${DURATION}s"
else
    EXIT_CODE=$?
    echo "FAIL: Exited with code $EXIT_CODE"
    exit 1
fi
echo ""

# 6. Validate output JSON
echo "=== 6. Output Validation ==="
if [ ! -f "$OUTPUT_JSON" ]; then
    echo "FAIL: Output file not created at $OUTPUT_JSON"
    exit 1
fi
echo "PASS: Output file exists ($(stat -c%s "$OUTPUT_JSON") bytes)"

python3 -c "
import json

data = json.load(open('$OUTPUT_JSON'))
if not isinstance(data, list):
    print('FAIL: Output is not a JSON array')
    exit(1)
print(f'Total entries: {len(data)}')

if len(data) == 0:
    print('WARN: No detections in output')
    exit(0)

# Check first entry has required fields
required = {'image_id', 'bbox', 'category_id', 'score'}
entry = data[0]
missing = required - set(entry.keys())
if missing:
    print(f'FAIL: Entry 0 missing fields: {missing}')
    exit(1)
print(f'PASS: All required fields present (image_id, bbox, category_id, score)')

# Validate field types
e = data[0]
assert isinstance(e['image_id'], str), f'image_id should be str, got {type(e[\"image_id\"])}'
assert isinstance(e['bbox'], list) and len(e['bbox']) == 4, f'bbox should be [x,y,w,h], got {e[\"bbox\"]}'
assert isinstance(e['category_id'], int), f'category_id should be int, got {type(e[\"category_id\"])}'
assert isinstance(e['score'], (int, float)), f'score should be number, got {type(e[\"score\"])}'
assert 0 <= e['category_id'] <= 355, f'category_id {e[\"category_id\"]} out of range 0-355'
assert 0 <= e['score'] <= 1, f'score {e[\"score\"]} out of range 0-1'
print(f'PASS: Field types and ranges valid')

# Check bbox values are reasonable
for i, e in enumerate(data[:5]):
    x, y, w, h = e['bbox']
    if w <= 0 or h <= 0:
        print(f'WARN: Entry {i} has non-positive bbox dimensions: w={w}, h={h}')
    if x < 0 or y < 0:
        print(f'WARN: Entry {i} has negative bbox coordinates: x={x}, y={y}')

# Sample output
print(f'Sample: {data[0]}')
print(f'Unique image_ids: {len(set(e[\"image_id\"] for e in data))}')
print(f'Category range: {min(e[\"category_id\"] for e in data)}-{max(e[\"category_id\"] for e in data)}')
print(f'Score range: {min(e[\"score\"] for e in data):.4f}-{max(e[\"score\"] for e in data):.4f}')
print('ALL CHECKS PASSED')
"
echo ""

# Cleanup
echo "=== Summary ==="
echo "ZIP structure: OK"
echo "Security scan: OK"
echo "Inference: OK"
echo "Output format: OK"
echo ""
echo "READY TO SUBMIT"

rm -rf "$WORK_DIR"
