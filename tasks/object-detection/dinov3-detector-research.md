# DINOv3 as Detection Backbone — Research Findings

**Date:** 2026-03-20
**Verdict: NO-GO for DINOv3. CONDITIONAL GO for DINOv2 feature matching only.**

---

## 1. DINOv3 Availability in Sandbox

**BLOCKER: DINOv3 is NOT available in the sandbox.**

- Sandbox has **timm 0.9.12** (Nov 2023)
- DINOv3 was added in **timm 1.0.20** (2025+)
- DINOv3 requires a custom `RotaryEmbeddingDinoV3` implementation in timm's EVA model
- We cannot install packages in the sandbox (no network, blocked imports)
- Bundling timm 1.0.x in the ZIP would blow the 420MB limit and may hit import restrictions

**DINOv2 IS available in timm 0.9.12:**
- `vit_small_patch14_dinov2.lvd142m` (21M params, 384-dim, ~86MB)
- `vit_small_patch14_reg4_dinov2.lvd142m` (register variant)
- `vit_base_patch14_dinov2.lvd142m` (86M params, 768-dim, ~330MB)

---

## 2. DINOv2 as YOLO Backbone

### Can ultralytics use a ViT/DINOv2 backbone?

**Partially, but impractical for our case.**

- A project [Yolo-DinoV2](https://github.com/itsprakhar/Yolo-DinoV2) exists that freezes DINOv2 as a feature extractor inside ultralytics YOLO
- However: ultralytics 8.1.0 in the sandbox is old and unlikely to support custom backbone configs
- ViT produces single-scale features; YOLO needs multi-scale FPN inputs — requires a custom adapter
- The Yolo-DinoV2 project has no benchmarks showing improvement over standard YOLO backbones
- Training would require the custom ultralytics fork — deployment complexity

**Recent academic work:**
- "DINOv3 Meets YOLO26" (arxiv 2603.00160, Mar 2026) showed +5.4% mAP improvement using DINOv3-finetuned ViT-S backbone in YOLO26-L for weed detection. But this uses DINOv3 (unavailable) and YOLO26 (not in sandbox ultralytics).

**Verdict: Not feasible within constraints.**

---

## 3. DINOv2 + Simple Detection Head (torchvision)

### Can we use torchvision.models.detection with DINOv2?

**Technically possible but problematic.**

- torchvision 0.21.0 is pre-installed with FasterRCNN, FCOS, RetinaNet
- These expect spatial feature maps (B, C, H, W) from the backbone
- DINOv2 ViT outputs patch tokens as (B, N, D) where N = (H/14)*(W/14), D = 384
- Reshaping to (B, 384, H/14, W/14) is straightforward but only gives single-scale features
- FasterRCNN with a single-scale backbone works but performs poorly (see GitHub issue #350 on dinov2 repo)

### Known issues (from facebookresearch/dinov2#350):
- DINOv2 + Faster R-CNN on Cityscapes: only detected 3/8 classes, boxes placed on wrong regions
- Root cause: ViT features lack the multi-scale pyramid that detection heads expect
- Solution that worked: DINOv2 + ViTDet + DINO decoder = ~52 boxAP on COCO (but requires detectron2, not available)

### Simple Feature Pyramid (ViTDet approach):
- ViTDet paper showed you can build a simple feature pyramid from single-scale ViT output
- Uses convolutions to create 4 scales: stride-4 (deconv), stride-8 (deconv), stride-16 (identity), stride-32 (conv)
- This is ~5-10M extra parameters
- But: implementing this from scratch + training in 24h is risky

**Verdict: Too much engineering risk for marginal gain.**

---

## 4. RF-DETR via ONNX

### Current status:
- RF-DETR uses DINOv2 backbone + custom DETR-style decoder
- We already tried training RF-DETR but classification loss collapsed on 356 classes
- ONNX export IS supported: `model.export()` produces `inference_model.onnx`
- [rf-detr-onnx](https://github.com/PierreMarieCurie/rf-detr-onnx) provides standalone ONNX inference

### Feasibility:
- RF-DETR-Base ONNX: ~200-300MB (fits in 420MB ZIP)
- onnxruntime-gpu 1.20.0 is pre-installed
- Input must be divisible by 14
- Can run on L4 GPU

### Problem:
- **We need a successfully trained RF-DETR checkpoint first**
- Training collapsed on 356 classes — this is the fundamental issue
- Even if we export to ONNX, the model quality is the bottleneck
- RF-DETR-Medium achieves 54.7% mAP on COCO, but COCO only has 80 classes
- 356 fine-grained grocery classes with 248 training images is a much harder problem

**Verdict: Only viable if we solve the training collapse. Not a backbone question.**

---

## 5. DINOv2 + DETR-style Decoder (from scratch)

### Architecture:
- DINOv2 ViT-S backbone (21M params, frozen) → patch tokens (B, N, 384)
- Transformer decoder: 6 layers, 256-dim, 8 heads, 300 queries
- Prediction heads: class (356) + bbox (4)
- Total decoder params: ~10-15M

### Parameter budget:
| Component | Params | Size |
|-----------|--------|------|
| DINOv2 ViT-S (frozen) | 21M | ~86MB |
| Transformer decoder (6L) | ~12M | ~48MB |
| Class + bbox heads | ~1M | ~4MB |
| **Total** | **~34M** | **~138MB** |

### Training estimate:
- Dataset: 248 images, ~22.7k boxes, 356 classes
- Training from scratch with frozen backbone: 50-100 epochs
- On RTX 3090: ~2-5 min/epoch → 2-8 hours total
- BUT: DETR-style models notoriously need long training (300 epochs on COCO)
- With only 248 images, convergence is uncertain

### Sandbox compatibility:
- Pure PyTorch — works
- No blocked imports needed (no os, subprocess, yaml, etc.)
- HOWEVER: `pickle` is blocked — loading `.pt` weights via `torch.load()` uses pickle internally
  - **This is a critical issue** — need to verify if `torch.load()` works in sandbox
  - safetensors IS available as an alternative

### Key risks:
1. DETR decoders are hard to train from scratch without careful hyperparameter tuning
2. Hungarian matching for 300 queries × 356 classes is compute-intensive
3. 248 training images is far too few for DETR-style training (original DETR needed 118k COCO images)
4. No established codebase — pure custom code risk

**Verdict: High risk, unlikely to beat YOLO in 24h.**

---

## 6. Practical Feasibility Summary

| Approach | Build in 24h? | Fits 420MB? | Runs <300s? | Beat YOLO? | Verdict |
|----------|---------------|-------------|-------------|------------|---------|
| DINOv3 backbone | N/A — not available in sandbox | — | — | — | **BLOCKED** |
| DINOv2 + YOLO backbone | Maybe | Yes | Yes | Unlikely | **NO-GO** |
| DINOv2 + torchvision det | Risky | Yes | Maybe | Unlikely | **NO-GO** |
| RF-DETR ONNX | If trained | Yes | Yes | Maybe | **BLOCKED by training** |
| DINOv2 + DETR scratch | Very risky | Yes | Maybe | Unlikely | **NO-GO** |
| DINOv2 for classification only | Already done | Yes | Yes | Complementary | **ALREADY IN USE** |

---

## 7. Recommendation

### NO-GO on DINOv3/DINOv2 as detection backbone.

**Reasons:**
1. **DINOv3 is unavailable** in sandbox timm 0.9.12
2. **DINOv2 as detection backbone is unproven** for this scale (356 classes, 248 images)
3. **Engineering risk is too high** for 24h remaining
4. **YOLO is a stronger baseline** — YOLOv8x already achieves 0.743 mAP50 on detection
5. **Sandbox restrictions** (blocked pickle, yaml, threading) make custom frameworks fragile

### What to do instead:

1. **Stick with YOLO11x/YOLO26x ONNX** for detection — models already trained
2. **Use DINOv2 for classification** (already done — 80.4% accuracy linear probe)
3. **Focus remaining time on:**
   - Submitting the best YOLO model + DINOv2 classifier pipeline
   - Tuning confidence/NMS thresholds
   - Ensemble of YOLO11x + YOLO26x with WBF
   - Test-time augmentation (multi-scale, flip)

### DEIMv2 — Worth watching but not now:
- [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) achieves 50.9 AP on COCO with only 9.71M params using DINOv3 backbone
- First sub-10M model to break 50 AP — impressive
- But requires DINOv3 (timm 1.0.20+) and custom training pipeline
- Would be relevant if sandbox gets updated, or for future competitions

---

## Sources

- [timm DINOv3 collection](https://huggingface.co/collections/timm/timm-dinov3)
- [DINOv3 GitHub (facebookresearch)](https://github.com/facebookresearch/dinov3)
- [DINOv2 + Faster R-CNN issues](https://github.com/facebookresearch/dinov2/issues/350)
- [Yolo-DinoV2](https://github.com/itsprakhar/Yolo-DinoV2)
- [DEIMv2: Real Time Object Detection Meets DINOv3](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [DINOv3 Meets YOLO26 for Weed Detection](https://arxiv.org/html/2603.00160v1)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [RF-DETR ONNX export docs](https://rfdetr.roboflow.com/latest/learn/export/)
- [rf-detr-onnx standalone inference](https://github.com/PierreMarieCurie/rf-detr-onnx)
- [ViTDet paper](https://arxiv.org/abs/2203.16527)
- [timm 0.9.12 on PyPI](https://pypi.org/project/timm/0.9.12/)
