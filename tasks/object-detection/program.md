# autoresearch: MarkusNet Object Detection

Turn MarkusNet (pruned Qwen3.5-0.8B, 351M params) into the best embedded VLM for zero-shot object detection.

## Research Question

Can a pruned Qwen3.5 VLM do both object detection AND zero-shot classification, trained only on external data, evaluated on competition data?

## Architecture: MarkusNet 351M

```
Input Image (variable size)
  → Vision Encoder (12 ViT blocks, 768d, 12 heads)
  → Spatial Merger (2×2 pool, 768→1024d)
  → Hybrid Decoder (9 Mamba + 3 full attention, 1024d)
  → Detection Head (bbox regression + classification)
```

Current state: classification-only (crop-based). Goal: add detection.

## Key Challenge

MarkusNet is a VLM — it processes single images and outputs class labels. Object detection requires:
1. **Dense prediction**: output bounding boxes at multiple scales
2. **Multi-object**: detect all products in one forward pass
3. **Zero-shot classification**: use text/reference embeddings instead of fixed class head

## Approaches to Test

### A. ViTDet-style: Feature Pyramid from ViT tokens
- Reshape decoder token sequence back to 2D spatial grid
- Extract features from layers 4, 8, 12 → build FPN (P3, P4, P5)
- Attach lightweight YOLO-style detection head
- Classification: dot product with pre-computed text embeddings (356 product names)
- Train detection head only (freeze backbone) → then full fine-tune

### B. DETR-style: Object queries on decoder output
- Add learned object queries (100-300)
- Cross-attend with decoder features
- Predict boxes + class logits per query
- Hungarian matching loss
- More elegant but slower convergence

### C. Sliding window with NMS
- Run MarkusNet at multiple positions/scales
- Each position classifies the central region
- NMS to merge overlapping detections
- Simplest but slowest inference

### D. Hybrid: MarkusNet backbone + YOLO head
- Use MarkusNet vision encoder + merger as backbone
- Discard the language decoder
- Attach standard YOLOv8 neck (PANet) + head
- Fine-tune end-to-end
- Loses VLM zero-shot capability but gains detection speed

## Evaluation

**Validation set**: All 248 competition images (COCO format, 22,731 annotations, 356 categories)
**Metric**: `combined_map = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`
**Training data**: External only (Polish shelves 27K, SKU-110K 7.5K, Grocery Shelves 45, store photos 93)
**Sacred rule**: Competition 248 images are validation ONLY

## Experiment Loop

LOOP FOREVER:

1. Pick an approach (A/B/C/D) and a specific modification
2. Implement the change
3. Train on external data only (GPU: local 3090 or rent cloud)
4. Evaluate on all 248 competition images with `eval_honest.py`
5. Log to `data_experiment_results.tsv`
6. Update `plot_progress.py`
7. If improved → keep. If not → revert.

## Current Baselines

| Model | det_mAP@0.5 | cls_mAP@0.5 | Combined | Notes |
|-------|-------------|-------------|----------|-------|
| YOLOv8x external pretrain | 0.543 | 0.000 | 0.380 | Single-class, no classification |
| YOLO World L zero-shot | 0.182 | 0.000 | 0.127 | Norwegian text prompts |
| MarkusNet 351M (crop cls) | — | 0.812 top-1 | — | Classification only, no detection |
| Server baseline (v6_clean) | — | — | 0.908 | Trained on competition data |

## Key Files

- `vlm-approach/run_markusnet.py` — full architecture (PyTorch native)
- `vlm-approach/export_markusnet.py` — export + quantization
- `vlm-approach/train_overnight.py` — 3-stage curriculum training
- `eval_honest.py` — honest evaluation on 248 competition images
- `descriptive_labels.json` — English descriptions for 356 categories
- `reference_embeddings.npz` — DINOv2 embeddings for KNN

## Constraints

- Must be pruned Qwen3.5 backbone (351M params target)
- No training on competition 248 images (external only for honest evaluation)
- Zero-shot classification: use text embeddings or reference images, not fixed class head
- Target: beat YOLOv8x external pretrain (0.543 det) AND add classification

**NEVER STOP**: Autonomous loop. If stuck, try a different approach.
