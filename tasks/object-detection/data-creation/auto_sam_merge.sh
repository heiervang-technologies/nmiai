#!/bin/bash
# Auto-runs v3 copy-paste with SAM cutouts and merges into mega_dataset
# Run this after SAM segmentation completes

set -e
cd "$(dirname "$0")"

echo "=== Checking SAM cutouts ==="
SAM_COUNT=$(ls data/product_cutouts_sam/*.png 2>/dev/null | wc -l)
echo "SAM cutouts available: $SAM_COUNT"

if [ "$SAM_COUNT" -lt 50 ]; then
    echo "Too few SAM cutouts ($SAM_COUNT). Wait for SAM to finish or run with at least 50."
    exit 1
fi

echo "=== Running v3 copy-paste with SAM cutouts ==="
python3 -u silver_copypaste_v3_sam.py

echo "=== Merging v3 into mega_dataset ==="
python3 -c "
import shutil
from pathlib import Path
from collections import Counter

DATA_DIR = Path('data')
src_imgs = DATA_DIR / 'silver_copypaste_v3' / 'images'
src_lbls = DATA_DIR / 'silver_copypaste_v3' / 'labels'
dst_imgs = DATA_DIR / 'mega_dataset' / 'train' / 'images'
dst_lbls = DATA_DIR / 'mega_dataset' / 'train' / 'labels'

added = 0
for img in list(src_imgs.glob('*.jpg')) + list(src_imgs.glob('*.png')):
    lbl = src_lbls / (img.stem + '.txt')
    if lbl.exists() and not (dst_imgs / img.name).exists():
        shutil.copy2(str(img), str(dst_imgs / img.name))
        shutil.copy2(str(lbl), str(dst_lbls / lbl.name))
        added += 1

total = len(list(dst_imgs.glob('*')))
print(f'Added {added} v3 SAM images. Mega dataset: {total} images')
"

echo "=== DONE ==="
