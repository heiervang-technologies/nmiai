# Object Detection Task - Overview

## NorgesGruppen Object Detection Challenge

Detect and classify grocery store products from shelf images provided by NorgesGruppen.

## Dataset

- **248 images** from grocery store shelves
- **22,700 annotations** (bounding boxes with category labels)
- **356 categories** of products
- **4 store sections**:
  - Egg
  - Frokost (Breakfast)
  - Knekkebrod (Crispbread)
  - Varmedrikker (Hot beverages)

## Format

The dataset is provided in **COCO format** (JSON annotations with image references).

## Scoring

The score is a weighted combination of two mAP metrics, both evaluated at IoU threshold 0.5:

- **70% Detection mAP@0.5** - Measures ability to locate objects (category-agnostic)
- **30% Classification mAP@0.5** - Measures ability to correctly classify detected objects

```
score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5
```

## Compute Environment

- **GPU**: NVIDIA L4
- **Network**: No network access during inference

Submissions must run entirely offline. All model weights and dependencies must be included in the submission package.
