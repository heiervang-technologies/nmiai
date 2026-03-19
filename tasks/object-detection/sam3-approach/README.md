# SAM3 Approach — NM i AI 2026 Object Detection

## Executive Summary

**VERDICT: SAM3 is NOT viable as the primary model for this competition.**

The `facebook/sam3` checkpoint is **3.3GB** — far exceeding the **420MB ZIP limit**. SAM3 cannot be shipped in the submission. However, SAM3 has strong value as an **offline tool** for training data augmentation and as a research baseline to inform our YOLO strategy.

---

## 1. SAM3 Architecture Analysis

### What SAM3 Is
SAM3 (Segment Anything Model 3) is Meta's latest foundation model for promptable segmentation. Key capabilities:
- **Text-prompted instance segmentation**: "segment all [product name]" returns masks + bounding boxes + scores
- **Visual prompts**: boxes, points, masks as prompts
- **Automatic mask generation**: segment everything without prompts (via `pipeline("mask-generation")`)
- **Open-vocabulary**: understands 270K+ concepts via CLIP text encoder
- **Returns bounding boxes**: `post_process_instance_segmentation()` outputs `boxes` in xyxy format directly

### Architecture Components
- **Vision backbone**: ViT with 32 layers, hidden_size=1024, patch_size=14, image_size=1008x1008
- **Text encoder**: CLIP text model (24 layers, hidden_size=1024, vocab=49408)
- **DETR detector**: encoder (6 layers) + decoder (6 layers, 200 queries) + geometry encoder (3 layers)
- **Mask decoder**: 2-layer transformer, 3 upsampling stages
- **Tracker**: memory attention (4 layers), memory fuser, prompt encoder — for video tracking

### Model Sizes
| File | Size |
|------|------|
| `model.safetensors` | 3.3 GB |
| `sam3.pt` | 3.3 GB |
| Tokenizer + config | ~5 MB |

### Software Requirements
- `transformers >= 5.0.0.dev0` (not pre-installed in sandbox — sandbox has standard packages only)
- PyTorch 2.6 compatible (yes)
- HuggingFace `accelerate` for device management

---

## 2. Feasibility Analysis for Competition

### BLOCKER: Model Size (420MB ZIP limit)

| Budget Item | Size | Fits? |
|------------|------|-------|
| SAM3 full model | 3,300 MB | NO (8x over limit) |
| SAM3 quantized INT8 | ~825 MB | NO (2x over limit) |
| SAM3 quantized INT4 | ~413 MB | BARELY (no room for code) |
| YOLO11x for comparison | ~115 MB | YES (304 MB remaining) |

**Even aggressive quantization cannot make SAM3 fit.** INT4 quantization would also severely degrade segmentation quality, defeating the purpose.

### BLOCKER: Inference Speed (300s timeout)

SAM3 at 1008x1008 input on L4 GPU (estimated):
- Image encoding (ViT-L/32): ~200-400ms per image
- Text encoding: ~50ms (one-time)
- DETR detection: ~100-200ms per image
- Mask generation: ~100-200ms per image
- **Per-image total: ~400-800ms**
- **For ~250 test images: 100-200s** — fits within 300s but tight
- With automatic mask generation (no text prompt): potentially slower due to grid sampling

For comparison, YOLO11x: ~11ms per image = ~3s total for 250 images.

### BLOCKER: Sandbox Dependencies

The sandbox pre-installs `transformers` (standard release), but SAM3 requires `transformers >= 5.0.0.dev0`. We would need to bundle a dev version of transformers in the ZIP, adding ~50-100MB and risking compatibility issues.

### Not a Blocker: Classification

SAM3 does NOT classify — it only segments. It can find "product" instances but cannot distinguish between 356 product categories from text alone (product names are too similar for grocery items). A separate classification step would still be needed.

---

## 3. SAM3 vs YOLO for mAP@0.5

### Detection Quality Comparison

| Aspect | SAM3 | YOLO11x |
|--------|------|---------|
| **mAP@0.5 potential** | High (precise masks -> tight boxes) | High (native box prediction) |
| **Dense scene handling** | Excellent (instance segmentation) | Good (NMS tuning needed) |
| **Mask-to-box conversion** | Trivial: min/max of mask coords | Native boxes |
| **IoU@0.5 implications** | Masks give pixel-perfect boxes, but mAP@0.5 is lenient — overkill | Box predictions at IoU 0.5 are sufficient |
| **Small object detection** | Strong (high-res 1008px input) | Good at 640-1280px |
| **Classification** | None (needs separate step) | Native 356-class output |
| **Fine-tuning on 248 images** | Possible but complex | Easy (mature pipeline) |

### Key Insight
At **IoU 0.5** (our metric), the precision advantage of SAM3's masks over YOLO's boxes is **negligible**. Both will achieve similar detection mAP. YOLO wins on:
- Native classification (covers the 30% classification score)
- 100x faster inference
- 30x smaller model
- Pre-installed in sandbox
- Mature fine-tuning pipeline

### Where SAM3 Wins
- Instance segmentation quality (not scored in this competition)
- Open-vocabulary detection (useful if we had unknown categories)
- Automatic mask generation for densely packed shelves

---

## 4. Viable SAM3 Uses (Offline / Training Time)

### 4A. Copy-Paste Augmentation via SAM3 Segmentation

**This is SAM3's highest-value use for this competition.**

Use SAM3 locally to segment products from reference images and training images, then use those segments for copy-paste augmentation when training YOLO.

**Pipeline:**
1. Run SAM3 automatic mask generation on all 248 training images
2. Run SAM3 on 327 reference product images (studio backgrounds = easy segmentation)
3. Extract high-quality product segments with alpha masks
4. Use segments for copy-paste augmentation onto shelf backgrounds
5. Train YOLO on augmented dataset (5-10x more effective training data)

**Why SAM3 is better than simple thresholding for this:**
- Reference images have varied backgrounds (not always pure white)
- Products have irregular shapes (bottles, boxes, bags)
- SAM3 produces pixel-perfect masks without manual tuning

**Estimated effort:** 2-4 hours to set up pipeline
**Expected impact:** +2-5% mAP from better augmentation data

### 4B. SAM3 Embeddings for Reference Image Matching

SAM3's ViT backbone produces powerful visual embeddings. We could:
1. Extract SAM3 ViT embeddings from all 327 reference product images
2. At inference time, use a smaller model (DINOv2 ViT-S, 88MB) for embedding extraction
3. Pre-compute a mapping: SAM3 embedding space -> product category

**Problem:** We can't use SAM3's encoder at inference time (too large). We'd need to distill or use a different encoder. DINOv2 ViT-S is a better choice for inference-time embeddings.

**Verdict:** Use DINOv2 directly instead. SAM3 embeddings don't add enough value over DINOv2 to justify the complexity.

### 4C. SAM3 as Annotation Quality Checker

Run SAM3 on training images and compare its masks with ground truth boxes to:
- Find annotation errors in the training data
- Identify poorly annotated images
- Generate tighter bounding boxes from SAM3 masks

**Estimated effort:** 1-2 hours
**Expected impact:** Marginal (COCO annotations are usually decent)

### 4D. SAM3 for Synthetic Data Generation

Use SAM3 to:
1. Segment every product from every training image
2. Re-compose products into new shelf arrangements
3. Generate novel training images with known annotations

This is more ambitious than 4A but could generate much more diverse training data.

**Estimated effort:** 4-8 hours
**Expected impact:** +3-7% mAP if done well

---

## 5. Recommended Strategy

### Primary: Fine-tuned YOLO (for submission)
- YOLO11x or YOLO26x as the competition model
- Fits in 420MB, runs in <30s, pre-installed
- Native 356-class detection + classification
- See main approach README for details

### Secondary: SAM3 as offline augmentation tool
- Use SAM3 locally to generate copy-paste augmentation data
- Segment reference product images for clean product cutouts
- Augment YOLO training with copy-pasted products

### NOT recommended:
- SAM3 as the submission model (size/speed blockers)
- SAM3 quantized to INT4 (quality loss, still barely fits)
- SAM3 for inference-time classification (no native classification)
- Distilling SAM3 into a smaller model (too complex for 3-day competition)

---

## 6. Size and Speed Summary

### Does SAM3 fit in 420MB?

**NO.** The model is 3.3GB. Even with:
- FP16: ~1.65GB (NO)
- INT8: ~825MB (NO)
- INT4: ~413MB (barely, no room for code, severe quality loss)
- Pruned/distilled: would need custom work, not feasible in competition timeframe

### Can SAM3 run under 300s?

**Probably yes** for ~250 images at ~400-800ms each = 100-200s. But this is tight and leaves no room for classification or TTA. YOLO at ~11ms/image leaves 270+ seconds for classification and TTA.

---

## 7. Data Augmentation with SAM3

### Copy-Paste Augmentation Pipeline (Recommended)

```
[Reference Images (327)] --SAM3--> [Product Segments with Alpha]
[Training Images (248)] --SAM3--> [Shelf Backgrounds + Product Masks]
                                         |
                                         v
                    [Random Paste Products onto Shelves]
                                         |
                                         v
                    [Augmented Training Set (2000+ images)]
                                         |
                                         v
                              [Train YOLO on Augmented Data]
```

**Steps:**
1. Install SAM3 locally (already downloaded)
2. Run automatic mask generation on reference images
3. Filter masks by size/quality, save as PNG with alpha
4. For each training epoch, randomly paste 5-20 products onto shelf backgrounds
5. Generate COCO annotations for pasted products automatically

### Implementation Notes
- Use `Sam3TrackerModel` with automatic mask generation pipeline for reference images
- Use `Sam3Model` with text prompt "product" for training image segmentation
- Save segments as RGBA PNGs for reuse
- Implement geometric augmentation (scale 0.5-1.5x, rotation +/-15 degrees)
- Color harmonization: match brightness/contrast of segment to target shelf region

---

## 8. Comparison Matrix: All Approaches

| Criterion | SAM3 (submission) | SAM3 (offline aug) | YOLO11x | RF-DETR | YOLO + DINOv2 |
|-----------|-------------------|---------------------|---------|---------|---------------|
| Fits 420MB | NO (3.3GB) | N/A | YES (115MB) | YES (135MB) | YES (~210MB) |
| Runs <300s | Tight | N/A | YES (<30s) | YES (<60s) | YES (~90s) |
| Pre-installed | NO | N/A | YES | NO | Partial |
| Detection mAP | High | N/A | High | Higher | High |
| Classification | NO | N/A | YES | YES | YES (embeddings) |
| Implementation effort | High | Medium | Low | Medium | Medium |
| **Recommendation** | **REJECT** | **USE** | **PRIMARY** | **BACKUP** | **HYBRID** |

---

## 9. Action Items

- [ ] **NOW:** Set up SAM3 locally for copy-paste augmentation pipeline
- [ ] **NOW:** Segment all 327 reference product images with SAM3
- [ ] **NOW:** Segment training images to extract shelf backgrounds
- [ ] **LATER:** Generate augmented training dataset
- [ ] **LATER:** Feed augmented data to YOLO training pipeline
- [ ] **SKIP:** Any attempt to submit SAM3 as the inference model

---

## 10. Local SAM3 Setup (for offline use)

### Checkpoint Location
```
~/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/
  - model.safetensors (3.3GB)
  - sam3.pt (3.3GB)
  - config.json, tokenizer files, etc.
```

### Quick Test
```python
from transformers import Sam3Model, Sam3Processor
import torch
from PIL import Image

model = Sam3Model.from_pretrained("facebook/sam3").to("cuda")
processor = Sam3Processor.from_pretrained("facebook/sam3")

image = Image.open("path/to/shelf_image.jpg").convert("RGB")
inputs = processor(images=image, text="product", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs, threshold=0.3, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

# results["masks"], results["boxes"], results["scores"]
```

### Automatic Mask Generation (for reference images)
```python
from transformers import pipeline
generator = pipeline("mask-generation", model="facebook/sam3", device=0)
outputs = generator("path/to/reference_image.jpg", points_per_batch=64)
# outputs["masks"] - list of all detected masks
```
