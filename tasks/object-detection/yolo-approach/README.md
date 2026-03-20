# YOLO Approach — NM i AI 2026 Object Detection

## Executive Summary

**Goal:** Win the NorgesGruppen object detection challenge (356 grocery product categories on shelves).

**Core Strategy:** Fine-tune YOLOv8x end-to-end on all 356 classes, then boost classification with DINOv2 embedding matching against the 327 product reference images. This targets both the 70% detection score and 30% classification score simultaneously.

---

## 1. Model Selection: YOLOv8x

### Why YOLOv8x (not YOLO11x or YOLO26x)

| Factor | YOLOv8x | YOLO11x/26x |
|--------|---------|-------------|
| Sandbox support | Native (ultralytics 8.1.0 pre-installed) | Requires ultralytics >=8.3/9.x — NOT pre-installed |
| Risk | Zero — guaranteed to work | Must bundle newer ultralytics, risk compatibility issues with PyTorch 2.6 |
| mAP50-95 COCO | 53.9 | 54.7 / 57.5 |
| mAP50 gap | ~1-3% lower | Marginal improvement not worth the risk |
| Weights size | ~131 MB | ~115 MB |

**ultralytics 8.1.0** (the pre-installed version) was released ~Jan 2024 and supports YOLOv8 natively. YOLO11 requires ultralytics >=8.3 (Oct 2024), and YOLO26 requires ultralytics >=9.x (Jan 2026). Bundling a newer ultralytics in the 420MB ZIP is risky:
- Adds ~50-80MB to ZIP
- May have dependency conflicts with pre-installed PyTorch 2.6
- Import restrictions (`os` blocked) could break newer ultralytics internals

**Decision: YOLOv8x is the safe, proven choice.** The 1-3% mAP difference to YOLO11x/26x is not worth the risk of a broken submission.

### Fallback: Try YOLO11x/26x

If time permits, we can test bundling a newer ultralytics. But the primary path must be YOLOv8x.

---

## 2. Training Recipe

### Dataset Preparation

Convert the COCO annotations to YOLO format (ultralytics expects this for training):
- One `.txt` file per image with `class_id cx cy w h` (normalized)
- 356 classes (0-355)
- 80/20 train/val split (198 train, 50 val images)

### Hyperparameters

```yaml
# YOLOv8x fine-tuning config
model: yolov8x.pt          # COCO-pretrained
epochs: 200                 # With early stopping patience=50
imgsz: 1280                 # High res for dense shelf scenes (default 640 is too low)
batch: 4                    # Fit in 24GB VRAM at 1280 (may need batch=2)
optimizer: AdamW
lr0: 0.001                  # Lower than default — fine-tuning, not training from scratch
lrf: 0.01                   # Final LR = lr0 * lrf = 0.00001
warmup_epochs: 5
weight_decay: 0.0005
patience: 50                # Early stopping

# Augmentation
mosaic: 1.0                 # Full mosaic augmentation
mixup: 0.15                 # Light mixup
copy_paste: 0.3             # Copy-paste augmentation (critical for few-shot classes)
degrees: 5.0                # Light rotation (shelves are mostly straight)
translate: 0.1
scale: 0.5                  # Scale variation
fliplr: 0.5                 # Horizontal flip (shelves are symmetric)
flipud: 0.0                 # No vertical flip (shelves don't flip vertically)
hsv_h: 0.015                # Color jitter for lighting variation
hsv_s: 0.7
hsv_v: 0.4
erasing: 0.3                # Random erasing (simulates occlusion)

# Loss
box: 7.5                    # Default
cls: 0.5                    # Default
dfl: 1.5                    # Default

# NMS
nms: True
iou: 0.5                    # NMS IoU threshold (tune on val)
conf: 0.001                 # Very low conf for mAP evaluation
max_det: 300                # Dense shelves can have many products per image
```

### Why imgsz=1280?

Shelf images are dense with many small products. At 640px, small products lose detail needed for both detection and classification. At 1280:
- Better detection of small/distant products
- Better feature extraction for fine-grained classification
- L4 GPU (24GB) handles 1280 with batch=4 (or batch=2 if needed)
- Still fast enough: ~20ms per image at 1280 on L4

### Training Steps

1. **Convert COCO to YOLO format** — script to generate `.txt` label files
2. **Create data.yaml** — paths and 356 class names
3. **Download YOLOv8x COCO pretrained weights** — `yolov8x.pt`
4. **Fine-tune** — `yolo detect train data=data.yaml model=yolov8x.pt epochs=200 imgsz=1280`
5. **Evaluate** — check mAP50 on validation split
6. **Export best.pt** — the fine-tuned weights for submission

---

## 3. Handling 356 Classes with 248 Images

### The Math

- 22,700 annotations / 356 classes = ~64 annotations per class on average
- But distribution is likely very uneven (some classes have hundreds, some have <10)
- 248 images means ~91 annotations per image (dense scenes)

### Mitigation Strategies

1. **COCO pretraining transfer**: YOLOv8x learned general object detection from 80 COCO classes and millions of images. Fine-tuning on 356 grocery classes transfers this knowledge.

2. **Heavy augmentation**: Mosaic (4x effective data), copy-paste, mixup, color jitter, random erasing — effectively multiplies dataset size by 10-20x.

3. **High resolution (1280)**: More pixel information per product helps the model distinguish similar-looking products.

4. **Class frequency analysis**: Before training, analyze annotation distribution. If any class has <5 examples, consider:
   - Oversampling those images
   - Copy-paste augmentation specifically for rare classes
   - Grouping rare classes into parent categories

5. **Focal loss**: Built into YOLOv8 by default — handles class imbalance.

---

## 4. Leveraging 327 Product Reference Images

### Strategy: Two-Phase Use

#### Phase A: Copy-Paste Augmentation (Training Time)

1. Segment products from studio reference images (white/clean backgrounds → easy to extract)
2. For each training image, randomly paste 1-3 reference product cutouts onto shelf backgrounds
3. Add corresponding bounding box annotations
4. This creates synthetic training data for underrepresented classes

**Implementation:**
```python
# Pseudo-code for copy-paste augmentation
# 1. Load reference image, remove background (threshold on white bg)
# 2. Resize product to match typical shelf product scale
# 3. Place at random valid position on shelf image
# 4. Add bbox annotation
```

#### Phase B: DINOv2 Embedding Matching (Inference Time — Classification Boost)

For the 30% classification score:
1. Pre-compute DINOv2 ViT-S/14 embeddings for all 327 reference products (all angles)
2. Store as a numpy array (~5MB)
3. At inference: after YOLO detects bounding boxes, crop each detection
4. Compute DINOv2 embedding of the crop
5. Find nearest neighbor in reference embeddings → assign category_id
6. Use this to override YOLO's classification for low-confidence predictions

**When to override YOLO's class:**
- If YOLO classification confidence < 0.3, use DINOv2 match instead
- If DINOv2 match distance < threshold AND differs from YOLO class, prefer DINOv2
- This hybrid approach keeps YOLO's strong classifications but rescues weak ones

**Size budget:**
- DINOv2 ViT-S/14: ~88 MB (via `timm`, pre-installed in sandbox)
- Reference embeddings: ~5 MB
- Total: ~93 MB additional

---

## 5. mAP@0.5 Optimization

### Understanding the Metric

mAP@0.5 is computed per-class as the area under the precision-recall curve, then averaged across all classes. IoU threshold of 0.5 means bounding boxes need only 50% overlap — this is lenient.

### Key Optimizations

#### Confidence Threshold: 0.001
- For mAP evaluation, lower confidence = more recall = higher mAP
- The P-R curve calculation handles the precision trade-off
- Set `conf=0.001` in inference

#### NMS IoU Threshold: 0.45
- Dense grocery shelves have adjacent (but not overlapping) products
- Too high NMS IoU (>0.6) → duplicate detections
- Too low (<0.3) → misses adjacent same-class products
- 0.45 is a good starting point; tune on validation

#### max_det: 300
- Shelf images can have 60-100+ products
- Default max_det=300 in ultralytics is sufficient
- Don't accidentally limit detections

#### Test-Time Augmentation (TTA)
- Multi-scale: inference at [1024, 1280, 1536]
- Horizontal flip
- Merge with NMS or Weighted Boxes Fusion
- Expected improvement: +1-2% mAP
- Time cost: ~3x inference time (still under 300s budget)

**TTA in ultralytics:**
```python
results = model.predict(source=img, augment=True)  # Built-in TTA
```

#### Post-Processing
- Remove tiny detections (below minimum reasonable product size, e.g., <10x10 pixels)
- Category-specific confidence calibration if validation shows per-class bias

---

## 6. Maximizing the 30% Classification Score

### The Challenge

356 fine-grained grocery categories. Many products look similar (different flavors of same brand). Average ~64 training examples per class.

### Multi-Pronged Classification Strategy

1. **Primary: YOLOv8x end-to-end classification** — the model learns both detection and classification jointly. This is the baseline.

2. **Boost: DINOv2 embedding re-ranking** — for detections where YOLO is uncertain (confidence < 0.3), use DINOv2 nearest-neighbor against reference images. DINOv2 excels at fine-grained visual similarity.

3. **Boost: Ensemble classification** — if time permits, train a separate classifier (EfficientNet-B3 via timm, ~48MB) on crops from the 22.7k annotations. Average its predictions with YOLO's classification head.

### Expected Impact

- YOLO alone: ~40-50% classification accuracy on rare classes
- DINOv2 re-ranking on low-confidence: +5-10% on those cases
- Ensemble: +2-5% overall

---

## 7. Inference Pipeline (Submission)

### Architecture

```
Input images → YOLOv8x (detect + classify) → Post-process → [Optional: DINOv2 re-rank] → JSON output
```

### Size Budget

| Component | Size |
|-----------|------|
| `run.py` + utils | ~5 KB |
| `best.pt` (YOLOv8x fine-tuned) | ~131 MB |
| DINOv2 ViT-S/14 (via timm) | ~88 MB (or 0 if using timm pre-installed) |
| Reference embeddings | ~5 MB |
| **Total** | **~224 MB** (well under 420 MB) |

**Note:** timm is pre-installed in sandbox, so DINOv2 weights can be loaded from the ZIP without the timm package itself.

### Time Budget (300s limit)

| Step | Estimated Time |
|------|---------------|
| Model loading | ~5s |
| YOLOv8x inference (1280, ~250 images) | ~50-80s |
| TTA (3x) | ~150-240s |
| DINOv2 crop re-ranking | ~20-30s |
| JSON output | ~1s |
| **Total without TTA** | **~80-120s** |
| **Total with TTA** | **~180-300s** |

TTA is tight at 300s. Options:
- Use TTA with 2 scales only (not 3) → ~120-180s
- Skip TTA if it doesn't improve val mAP enough
- Use FP16 inference to halve model inference time

### run.py Skeleton

```python
"""NM i AI 2026 — Object Detection Submission"""
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model_path = Path(__file__).parent / 'best.pt'
    model = YOLO(str(model_path))

    # Get all images
    data_dir = Path(args.data)
    images = sorted(data_dir.glob('*.jpg'))

    results_list = []

    for img_path in images:
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=0.001,
            iou=0.45,
            max_det=300,
            imgsz=1280,
            device=device,
            half=True,  # FP16
            verbose=False
        )

        predictions = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                predictions.append({
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'category_id': int(boxes.cls[i].item()),
                    'score': float(boxes.conf[i].item())
                })

        for pred in predictions:
            pred['image_id'] = img_path.name
            results_list.append(pred)

    # Write output
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results_list))

if __name__ == '__main__':
    main()
```

---

## 8. Timeline & Submission Strategy

### Time Available

- **Start:** March 19, 2026 18:00 CET (challenges released)
- **End:** March 22, 2026 15:00 CET
- **Total:** ~69 hours
- **Submissions:** 10/day × 3 days + partial day ≈ 30-35 submissions total

### Phase Timeline

| Phase | Hours | Tasks | Submissions |
|-------|-------|-------|-------------|
| **1: Setup** | 0-3 | Download data, convert COCO→YOLO format, verify pipeline | 0 |
| **2: Baseline** | 3-8 | Train YOLOv8x 100 epochs at imgsz=640, submit | 1-2 |
| **3: Full Training** | 8-20 | Train YOLOv8x 200 epochs at imgsz=1280, heavy augmentation | 2-3 |
| **4: Optimization** | 20-36 | Tune NMS/conf thresholds, try TTA, evaluate on val | 3-5 |
| **5: DINOv2 Boost** | 36-48 | Implement embedding matching, integrate into pipeline | 2-3 |
| **6: Polish** | 48-60 | Ensemble experiments, final tuning, best submission | 3-5 |
| **7: Final** | 60-69 | Last submissions with best configuration | 2-3 |

### Submission Strategy

- **Day 1 (10 submissions):** Baseline + first full training results
- **Day 2 (10 submissions):** Optimization iterations, DINOv2 integration
- **Day 3 (10 submissions):** Final tuning, best ensemble, last-minute improvements

### Key Decisions to Make Early

1. After Phase 2 baseline: Is the mAP reasonable? If <20%, something is fundamentally wrong.
2. After Phase 3: Does imgsz=1280 help vs 640? How much?
3. After Phase 4: Does TTA fit in 300s? Worth the complexity?
4. After Phase 5: Does DINOv2 re-ranking actually improve classification?

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| ultralytics 8.1.0 doesn't support some YOLOv8x features | High | Test all features locally before submission |
| Training data not downloaded yet | Blocker | Download ASAP from app.ainm.no |
| 356 classes overfit on 248 images | High | Heavy augmentation, early stopping, COCO pretraining |
| Inference exceeds 300s | Medium | Skip TTA, use FP16, reduce imgsz |
| `os` import blocked breaks something | Medium | Test run.py in restricted environment |
| DINOv2 model doesn't fit in ZIP | Low | DINOv2 ViT-S is 88MB, total ~224MB, well under 420MB |
| Class distribution extremely skewed | Medium | Analyze before training, use class weights |

---

## 10. Data Download Needed

**CRITICAL:** Training data has not been downloaded yet. Need to download from app.ainm.no:
- `NM_NGD_coco_dataset.zip` (864 MB) — training images + COCO annotations
- `NM_NGD_product_images.zip` (60 MB) — reference product photos

These must be downloaded and extracted before any training can begin.

---

## 11. Alternative Models to Consider (Lower Priority)

If YOLOv8x baseline is disappointing:

1. **RF-DETR Medium** — Higher mAP, DINOv2 backbone, but must bundle rfdetr package (~150MB total)
2. **YOLOv8x + newer ultralytics** — Try bundling ultralytics 8.3+ for YOLO11x support
3. **Ensemble YOLOv8l + YOLOv8m** — Two models with WBF, more diverse predictions
4. **Class-agnostic detection + separate classifier** — YOLO detects all products as one class, then EfficientNet classifies crops

The primary path remains YOLOv8x end-to-end. Only pivot if baseline results are poor.
