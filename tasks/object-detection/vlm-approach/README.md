# VLM Approach — Object Detection Challenge Plan

**Goal:** Beat pure YOLO approaches by leveraging Vision-Language Models and foundation model embeddings for the 30% classification score, while matching or exceeding YOLO detection performance for the 70% detection score.

**Status:** PLANNING ONLY — no training yet.

---

## 1. Constraint Reality Check

Before planning, we must be brutally honest about what fits in the sandbox:

| Constraint | Value |
|---|---|
| ZIP size | 420 MB max |
| Inference timeout | 300 seconds |
| GPU | NVIDIA L4, 24 GB VRAM |
| Network | None |
| Runtime pip install | Not possible |
| Blocked imports | `os`, `subprocess`, `socket`, `ctypes`, `eval`, `exec` |

### Pre-installed (FREE — no ZIP space needed)
- `ultralytics` 8.1.0 (YOLOv8/YOLO11) ✅
- `torch` 2.6.0 + `torchvision` 0.21.0 ✅
- **`timm`** (PyTorch Image Models — includes DINOv2!) ✅
- `supervision` (Roboflow) ✅
- `onnxruntime-gpu` 1.20.0 ✅
- `safetensors` ✅
- `albumentations`, `scikit-learn`, `scipy`, `numpy`, `Pillow`, `opencv-python-headless` ✅

### NOT Pre-installed (must bundle in ZIP)
- `transformers` ❌ (~150-200 MB installed)
- `rfdetr` ❌ (needs transformers + peft)
- `peft` ❌
- `groundingdino` ❌

---

## 2. Model Feasibility Analysis

### ❌ Florence-2 — ELIMINATED
- **Why:** ~230M params = ~460 MB weights alone. **Exceeds 420 MB ZIP limit.**
- Also needs `transformers` library (not pre-installed).
- Dead on arrival.

### ❌ Grounding DINO (original) — ELIMINATED
- **Why:** Swin-T + BERT backbone = ~700 MB weights. Needs `transformers`.
- Even the Edge model (1.5/1.6) is API-only, no open weights for offline use.
- **356 text prompts × dense shelves = timeout risk** even if it fit.
- Dead on arrival.

### ⚠️ RF-DETR — HIGH RISK, HIGH REWARD
- **Model size:** Large variant ~136 MB weights. Fits in ZIP.
- **Problem:** Needs `rfdetr` package which depends on `transformers` + `peft`. Together ~200+ MB.
- **Total:** ~336 MB minimum. Tight but technically possible.
- **Mitigation:** Vendor only the minimal inference code from rfdetr, bypass transformers entirely by loading the DINOv2 backbone via `timm` (pre-installed) and the decoder directly.
- **Risk level:** HIGH — requires reverse-engineering the rfdetr inference path.
- **DINOv2 backbone** gives superior feature extraction for fine-grained grocery products.
- **SOTA on COCO:** 75.1 AP50 (Large variant) vs YOLO11x 54.7 AP50-95.
- **Designed for fine-tuning on small datasets** — converges faster than YOLO.

### ✅ DINOv2 via timm — PERFECT FIT
- **Pre-installed!** No ZIP space for the library.
- **ViT-S/14:** 22M params, ~84 MB weights. Excellent.
- **ViT-B/14:** 86M params, ~330 MB weights. Fits but tight.
- **Use case:** Extract embeddings from detected crops → match against 327 reference product image embeddings → classify.
- **Strength:** DINOv2 excels at fine-grained visual discrimination and instance retrieval (+34% mAP on Oxford Hard benchmarks).
- **This directly targets the 30% classification score.**

### ✅ YOLO (any variant) — GUARANTEED FIT
- Pre-installed, proven, fast.
- YOLO11x: ~115 MB, YOLO11l: ~50 MB.
- Handles the 70% detection score well.

### ⚠️ DE-ViT (Few-Shot with DINOv2) — INTERESTING BUT RISKY
- Purpose-built for reference-image-based detection.
- 50 AP50 on novel classes — great for rare categories.
- **Problem:** Research code, needs custom dependencies, not production-ready.
- **Could be adapted** since DINOv2 backbone is available via timm.

---

## 3. Recommended Strategy: Tiered Approaches

### Tier 1: YOLO + DINOv2 Hybrid (SAFE, HIGH-VALUE)

**This is the primary recommendation. It uses only pre-installed libraries.**

```
Pipeline: YOLO detection → Crop → DINOv2 embed → Reference matching → Classify
```

**Architecture:**
1. **Detector:** YOLO11x fine-tuned on 248 images (multi-class, 356 categories)
   - Provides detection boxes (70% score) AND first-pass classification
   - ~115 MB weights
2. **Classifier:** DINOv2 ViT-S/14 via `timm` for embedding extraction
   - Pre-compute embeddings of all 327 reference product images (multi-angle)
   - At inference: crop each detection, embed with DINOv2, nearest-neighbor match
   - ~84 MB weights + ~5 MB pre-computed reference embeddings
3. **Fusion:** Combine YOLO class confidence with DINOv2 cosine similarity
   - If YOLO is confident (>0.7) and DINOv2 agrees → keep YOLO class
   - If YOLO is uncertain → defer to DINOv2 nearest-neighbor
   - For 29 categories without reference images → trust YOLO only

**Size budget:**
| Component | Size |
|---|---|
| YOLO11x weights | ~115 MB |
| DINOv2 ViT-S weights | ~84 MB |
| Reference embeddings (327 × multi-angle × 384-dim) | ~5 MB |
| Code | ~2 MB |
| **Total** | **~206 MB** ✅ |

**Time budget (300s):**
| Step | Est. Time |
|---|---|
| Load models | ~5s |
| YOLO inference (all images, batch) | ~30-60s |
| Crop extraction (~100 detections/image) | ~5s |
| DINOv2 embedding (batched) | ~30-60s |
| Nearest-neighbor matching | ~2s |
| JSON output | ~1s |
| **Total** | **~75-135s** ✅ |

**Why this beats pure YOLO:**
- YOLO with 356 classes and ~63 examples/class will confuse similar products
- DINOv2 embeddings are specifically designed for fine-grained visual discrimination
- Reference images provide "ground truth" visual anchors for each product
- The hybrid corrects YOLO's classification mistakes → improves the 30% classification score
- Detection score stays the same or better (YOLO still does the boxes)

---

### Tier 2: RF-DETR with Vendored Inference (HIGH RISK, HIGHER CEILING)

**Only attempt if Tier 1 is working and there's time to experiment.**

**Concept:** RF-DETR has a DINOv2 backbone + custom decoder. Since `timm` is pre-installed:
1. Load DINOv2 backbone features via `timm`
2. Vendor only the RF-DETR decoder code (strip out transformers dependency)
3. Load fine-tuned RF-DETR weights (backbone + decoder)

**Why bother:**
- RF-DETR Large: 75.1 AP50 on COCO vs YOLO11x: ~71 AP50
- DINOv2 backbone gives better features for dense, fine-grained grocery shelves
- Designed for small-dataset fine-tuning — converges faster
- NMS-free → better for overlapping products on shelves

**Implementation steps:**
1. Fine-tune RF-DETR Large on the 248 images (on training machine with full rfdetr)
2. Export the model weights (backbone + decoder separately)
3. Write minimal inference-only code that:
   - Loads DINOv2 backbone via `timm`
   - Loads decoder weights directly with `torch.load` + `safetensors`
   - Runs the decoder forward pass (this is just standard PyTorch nn.Module)
4. Bundle: weights (~136 MB) + vendored decoder code (~5 MB) + run.py

**Risk factors:**
- May need transformers for tokenizer/config that's hard to strip
- Decoder may have unexpected dependencies
- Testing requires careful sandbox simulation

**Size budget:**
| Component | Size |
|---|---|
| RF-DETR Large weights | ~136 MB |
| Vendored decoder code | ~5 MB |
| DINOv2 reference embeddings (optional) | ~5 MB |
| Code | ~2 MB |
| **Total** | **~148 MB** ✅ |

---

### Tier 3: YOLO + DINOv2 + Ensemble (MAXIMUM EFFORT)

**Combine Tier 1 and Tier 2 if both work.**

```
Pipeline:
  YOLO11x → boxes + class predictions (confidence A)
  RF-DETR  → boxes + class predictions (confidence B)
  DINOv2   → crop embeddings → reference matching (confidence C)
  WBF fusion → merged boxes + fused class predictions
```

**Size budget:**
| Component | Size |
|---|---|
| YOLO11x weights | ~115 MB |
| RF-DETR Large weights | ~136 MB |
| DINOv2 ViT-S weights | ~84 MB |
| Reference embeddings | ~5 MB |
| Vendored code | ~10 MB |
| **Total** | **~350 MB** ✅ (within 420 MB) |

**Time budget:** Would need careful optimization. Two detectors + embeddings may push close to 300s. Could use YOLO11l (smaller) or skip TTA.

---

## 4. Training Plan

### 4.1 YOLO Training (Tier 1 — detection backbone)

**Option A: Multi-class (356 categories)**
```python
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
model.train(
    data='shelf_data.yaml',
    epochs=200,
    imgsz=1024,       # high-res for dense shelves
    batch=4,           # L4 VRAM constraint
    augment=True,      # mosaic, mixup, copy-paste
    lr0=0.001,
    close_mosaic=20,
    conf=0.001,        # low threshold for mAP
    iou=0.4,           # tighter NMS for dense scenes
)
```

**Option B: Single-class detector (just "product")**
- Simpler, better recall, but caps detection-only score
- Classification handled entirely by DINOv2
- May actually yield HIGHER total score if DINOv2 classification is strong enough

**Recommendation:** Train BOTH options and compare on validation split.

### 4.2 DINOv2 Reference Embedding Index

```python
import timm
import torch

# Load DINOv2 ViT-S/14 via timm
model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
model.eval().cuda()

# Pre-compute embeddings for all 327 reference products (multi-angle)
# Store as: {category_id: [embedding_1, embedding_2, ...]}
# At inference: crop → resize 518×518 → embed → cosine similarity → top-1 match
```

**Enhancement: Train a linear probe on DINOv2 embeddings**
- Extract DINOv2 embeddings for all 22.7k training annotation crops
- Train a simple linear classifier (356 classes) on top
- This combines reference matching + supervised signal
- Only adds ~500 KB to weights

### 4.3 Copy-Paste Augmentation (Critical)

Leverage the 327 reference product images:
1. Segment products from studio photos (easy — clean backgrounds)
2. Paste onto shelf training images at various positions/scales
3. Auto-generate COCO annotations
4. This is the single highest-value data augmentation for this challenge

### 4.4 RF-DETR Training (Tier 2 — if time permits)

```python
from rfdetr import RFDETRLarge
model = RFDETRLarge()
model.train(
    dataset_dir="shelf_coco/",
    epochs=100,
    batch_size=4,
    lr=1e-4,
    grad_accum_steps=4,
    image_size=560,
)
# Export weights for vendored inference
```

---

## 5. Inference Pipeline (run.py)

```python
"""
run.py — VLM Hybrid Inference Pipeline
Two-stage: YOLO detect → DINOv2 classify
"""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import timm
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device('cuda')

    # Stage 1: Load YOLO detector
    yolo = YOLO('model.pt')

    # Stage 2: Load DINOv2 classifier
    dino = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False)
    dino.load_state_dict(torch.load('dinov2_vits14.pth', map_location=device))
    dino.eval().to(device)

    # Load pre-computed reference embeddings
    ref_embeddings = torch.load('ref_embeddings.pth', map_location=device)
    # ref_embeddings: dict {category_id: tensor of shape [N_angles, 384]}

    # Optional: load linear probe
    linear_probe = torch.load('linear_probe.pth', map_location=device)

    results = []
    images_dir = Path(args.data)

    for img_path in sorted(images_dir.glob('*.jpg')):
        # YOLO detection
        detections = yolo.predict(img_path, conf=0.001, iou=0.4, imgsz=1024, verbose=False)

        predictions = []
        img = Image.open(img_path).convert('RGB')

        for det in detections[0].boxes:
            bbox = det.xywh[0].cpu().numpy()  # convert to COCO format
            yolo_cls = int(det.cls[0])
            yolo_conf = float(det.conf[0])

            # Crop for DINOv2
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            crop = img.crop((x1, y1, x2, y2)).resize((518, 518))
            crop_tensor = preprocess(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = dino(crop_tensor)  # [1, 384]

            # Nearest-neighbor against reference embeddings
            best_cat, best_sim = match_reference(embedding, ref_embeddings)

            # Fusion logic
            if yolo_conf > 0.7 and best_cat == yolo_cls:
                final_cls = yolo_cls
            elif best_sim > 0.85:
                final_cls = best_cat
            else:
                final_cls = yolo_cls  # fallback to YOLO

            predictions.append({
                'bbox': [float(bbox[0] - bbox[2]/2), float(bbox[1] - bbox[3]/2),
                         float(bbox[2]), float(bbox[3])],
                'category_id': final_cls,
                'confidence': yolo_conf
            })

        results.append({
            'image_id': img_path.name,
            'predictions': predictions
        })

    with open(args.output, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
```

*Note: This is pseudocode for the plan. Actual implementation will handle preprocessing, batching, edge cases, and the 29 categories without reference images.*

---

## 6. Why This Beats Pure YOLO

| Factor | Pure YOLO | YOLO + DINOv2 Hybrid |
|---|---|---|
| Detection (70%) | Strong | Same or better (same YOLO) |
| Classification (30%) | Weak on rare/similar products (~63 examples/class average) | Strong — DINOv2 + reference images provide visual anchors |
| 29 categories without reference images | Relies on few training examples | Falls back to YOLO (no worse) |
| Similar-looking products | High confusion rate | DINOv2 embeddings discriminate fine-grained differences |
| Reference images (327 products) | Unused at inference | Directly leveraged for classification |
| Model size | ~115 MB | ~206 MB (still well within limit) |
| Inference time | ~40s | ~120s (still well within limit) |

**Expected improvement:** +5-15% on classification mAP, translating to +1.5-4.5% on overall score. On a competitive leaderboard, this is the difference between podium and also-ran.

---

## 7. Risk Mitigation

| Risk | Mitigation |
|---|---|
| DINOv2 weights download blocked in sandbox | Bundle weights in ZIP (~84 MB — fits) |
| `timm` version doesn't support DINOv2 | Verify locally; fallback to raw `torch.hub` or vendored model code |
| Reference embedding matching is slow | Batch DINOv2 forward passes; pre-compute and cache reference embeddings as tensors |
| 29 categories lack reference images | Fall back to YOLO classification or train a DINOv2 linear probe on training crops |
| RF-DETR vendoring fails | Abandon Tier 2, stick with Tier 1 (YOLO + DINOv2 already strong) |
| Timeout risk with two models | Profile on L4; use FP16; reduce DINOv2 to ViT-S (smallest); batch crops |
| `os` import blocked | Use `pathlib` exclusively; test in sandboxed environment |

---

## 8. Implementation Priority Order

1. **[FIRST]** Train YOLO11x multi-class (356 categories) on 248 images → submit baseline
2. **[SECOND]** Build DINOv2 reference embedding index from 327 product images
3. **[THIRD]** Implement hybrid inference pipeline (YOLO detect + DINOv2 classify)
4. **[FOURTH]** Train DINOv2 linear probe on 22.7k training crops
5. **[FIFTH]** Copy-paste augmentation with reference images → retrain YOLO
6. **[SIXTH]** TTA (multi-scale + flip) + tune thresholds
7. **[SEVENTH]** RF-DETR experiment (Tier 2) — only if time permits
8. **[EIGHTH]** Ensemble (Tier 3) — only if both YOLO and RF-DETR work

---

## 9. Key Insights & Edge Cases

### The 70/30 Split Is Deceptive
- 70% detection means **finding boxes is critical** — don't sacrifice detection recall for classification accuracy
- BUT the remaining 30% is still 30 points on the leaderboard. On a competitive field, that's enormous.
- Teams that only do YOLO will cap at ~70% + whatever YOLO's classification gives them
- Our hybrid gives us the same detection + significantly better classification

### Dense Shelf Scenes
- ~64 annotations per image (very dense)
- NMS tuning is critical — too aggressive = missed products, too lenient = duplicates
- YOLO's built-in NMS may need adjustment for grocery shelves
- RF-DETR's NMS-free architecture could be advantageous here

### The Reference Image Goldmine
- 327 products with multi-angle photos is an underutilized resource
- Most teams will ignore these or use them only for augmentation
- Using them as a classification index via DINOv2 embeddings is our key differentiator
- This is essentially a "cheat code" for classification that few teams will exploit

### Input Resolution Matters
- Shelf images are wide with small products
- YOLO default 640px may miss small items
- Recommend 1024px or 1280px input (L4 can handle it)
- Higher resolution directly improves both detection and crop quality for classification

---

## 10. Files to Create

```
~/ht/nmiai/tasks/object-detection/vlm-approach/
├── README.md                    # This plan
├── train_yolo.py                # YOLO training script
├── build_embeddings.py          # Pre-compute DINOv2 reference embeddings
├── train_linear_probe.py        # Train linear classifier on DINOv2 embeddings
├── augment_copypaste.py         # Copy-paste augmentation pipeline
├── run.py                       # Inference pipeline (for submission ZIP)
├── submission/                  # Final ZIP contents
│   ├── run.py
│   ├── model.pt                 # Fine-tuned YOLO weights
│   ├── dinov2_vits14.pth        # DINOv2 ViT-S weights
│   ├── ref_embeddings.pth       # Pre-computed reference embeddings
│   └── linear_probe.pth         # Optional linear probe weights
└── experiments/                 # RF-DETR and other experiments
    ├── rfdetr_train.py
    └── rfdetr_vendor/           # Vendored RF-DETR inference code
```

---

## 11. Decision: Which Combination Gives Best Shot at Beating Pure YOLO?

**Answer: Tier 1 — YOLO11x + DINOv2 ViT-S Hybrid**

**Rationale:**
1. **Zero additional dependencies** — everything is pre-installed
2. **Directly exploits reference images** — the most underutilized resource in this challenge
3. **Targets the classification gap** — where pure YOLO is weakest
4. **Low risk** — YOLO detection is proven; DINOv2 classification only adds value, never hurts
5. **Fits comfortably** — 206 MB out of 420 MB, 120s out of 300s
6. **Fallback is safe** — if DINOv2 doesn't help, we still have a competitive YOLO submission

**The key insight:** This competition's 70/30 split means classification is the tiebreaker. Every team will have decent YOLO detection. The winner will be whoever cracks classification on 356 fine-grained grocery categories with minimal training data. DINOv2 + reference images is our answer.
