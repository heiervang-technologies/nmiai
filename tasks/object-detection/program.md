# autoresearch: object-detection

Data-mixture optimization for object detection. We optimize **what data to train on** and **how to train**, not inference tricks.

## Philosophy: Zen of Model Training

**Transfer learning pyramid** (bottom → top):
1. **Broad OOD data** — large external datasets (Polish shelves, COCO, Open Images)
2. **In-domain bridge** — store photos, product reference images, curated subsets
3. **Target eval** — competition 248 images (validation ONLY until final stage)
4. **Submit** — only when local val convincingly beats 0.92 mAP@0.5

**Sacred rules:**
- The 248 competition images are **validation only**. Never train on them until the final submission stage.
- Augmentation ≠ unique samples. Track them separately. Augmentation is fancy upsampling.
- Pseudo-labels are suspect. Verify quality before trusting them.
- Maximize **mAP@0.5** (0.7 × detection + 0.3 × classification).
- Only 6 test submissions remaining. Don't waste them.

## Setup

1. **Agree on a run tag** with the user (e.g. `mar22-data`). Branch: `autoresearch/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the key files**:
   - `tasks/object-detection/README.md` — task context, scoring, constraints.
   - `tasks/object-detection/data-creation/data/coco_dataset/train/annotations.json` — the 248-image ground truth (THIS IS YOUR VAL SET).
   - `tasks/object-detection/submission-markusnet/run.py` — current best submission.
4. **Verify validation set**: All 248 competition images with COCO annotations must be available for eval.
5. **Initialize results.tsv** with header. Baseline = current best model on full 248-image val.
6. **Go**.

## Data inventory

| Source | Images | Type | Quality | Location |
|--------|--------|------|---------|----------|
| Competition (VAL ONLY) | 248 | Shelf photos | Ground truth | `data/coco_dataset/train/` |
| External Polish shelves | 27,246 | Shelf photos | COCO annotations, different products | `data/external/` |
| Store photos (Markus) | 39 + video | Shelf photos | Unlabeled, in-domain | `data/store_photos/` |
| Product references | 327 | Product crops | Multi-angle, clean | `data/product_images/` |
| Silver augmented | 6,477 | Synthetic | Copy-paste augmentation | `data/silver_augmented_dataset/` |
| Pseudo-labeled | 2,014 | Auto-labeled | SUSPECT - needs QA | `data/pseudo_labels/` |

## What you optimize

The **data mixture** fed to training. Augmentation is OFF by default. Fix the data first.

**Phase 1: Clean data only (NO augmentation)**
1. **Source selection**: Which external datasets to include/exclude
2. **Category mapping**: How to map external categories → our 356 categories
3. **Data quality filtering**: Removing bad labels, low-confidence pseudo-labels
4. **Sample curation**: Which images actually help? Remove harmful ones.
5. **Training schedule**: Learning rate, epochs, warmup, freeze stages
6. **Architecture**: Model size, backbone, head configuration

**Phase 2: Augmentation (only after Phase 1 converges)**
7. **Augmentation policy**: What augmentation, applied to which sources
   - Only worth doing once the clean dataset is already strong
   - The only thing worse than bad data is bad data + augmented bad data

## The experiment loop

LOOP FOREVER:

1. **Hypothesize**: What data change should improve mAP@0.5? (add source, filter bad labels, adjust mixture ratio, change augmentation)
2. **Build dataset**: Create the training dataset with the proposed mixture. Record exact composition:
   - Unique images per source
   - Augmented copies per source (separate count!)
   - Total annotations, category coverage
3. **Train**: Launch training on GPU. Use YOLOv8x as default architecture unless testing alternatives.
4. **Evaluate**: Run trained model on ALL 248 competition images. Report:
   ```
   detection_map50:      X.XXXX
   classification_map50: X.XXXX
   combined_map:         X.XXXX
   unique_train_images:  NNNNN
   augmented_copies:     NNNNN
   data_sources:         source1(N), source2(N), ...
   ```
5. **Log to results.tsv** (see format below).
6. If combined_map improved → keep. If not → revert data change.
7. **Commit** the dataset build script (never the data itself).

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	combined_map	det_map50	cls_map50	unique_images	aug_images	status	description
```

Example:
```
a1b2c3d	0.850000	0.920000	0.690000	27246	0	keep	baseline: external polish only
b2c3d4e	0.870000	0.930000	0.730000	27285	0	keep	+ 39 store photos with Gemini labels
c3d4e5f	0.860000	0.925000	0.710000	33723	6477	discard	+ silver augmented (hurts)
```

## Key metrics to track per experiment

- **unique_images**: Real distinct training images (no augmentation)
- **aug_images**: Augmented copies (tracked separately)
- **category_coverage**: How many of 356 categories have ≥1 training sample
- **min_samples_per_cat**: Minimum samples for any category
- **leakage**: MUST be ZERO. Any overlap with val 248 = invalid experiment.

## Inference optimization (secondary loop)

Once we have a strong trained model (val mAP@0.5 > 0.92), THEN optimize `run.py`:
- NMS thresholds, TTA, confidence filtering, class remapping
- This is the old autoresearch loop — only run it on top of a good model

## Constraints reminder

- ZIP ≤ 420 MB, runtime ≤ 300s, GPU = NVIDIA L4 (24 GB)
- No network in sandbox
- Only 6 test submissions remaining

**NEVER STOP**: Do not pause to ask the human. You are autonomous. If stuck, try a different data source, different filtering, different mixture ratio.
