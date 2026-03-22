# MarkusNet Research Agenda

**Goal**: Turn pruned Qwen3.5-0.8B (351M) into the best embedded VLM for object detection.

## 1. Data Sources & Splits

### Available External Datasets
| Dataset | Images | Annotations | Categories | Format |
|---------|--------|-------------|-----------|--------|
| Polish Shelves | 27,244 | 2.05M | 7,942 SKUs | COCO |
| SKU-110K | 7,450+ | 1.09M | 1 (product) | COCO |
| Grocery Shelves | 45 | 9,924 | 1 (product) | COCO |
| Store photos batch 1 | 39 | SAM3 pseudo | 356 | YOLO |
| Store photos batch 2 | 54 | SAM3 pseudo | 356 | YOLO |
| Product references | 345×7 angles | — | 155 mapped | Crops |
| COCO train2017 | 118K | 800K+ | 80 | COCO |

### Experiments to Run
- [ ] External only (detection transfer): Polish + SKU-110K + Grocery
- [ ] External + category mapping (classification transfer)
- [ ] External + store photos (in-domain bridge)
- [ ] Curriculum: COCO → external → store photos
- [ ] Data quality filtering: remove noisy images/labels
- [ ] Sample weighting: weight in-domain higher than OOD
- [ ] Active learning: identify most informative samples

### Hyperparameters to Sweep
- [ ] Learning rate: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3
- [ ] Optimizer: AdamW, LAMB, Lion
- [ ] Batch size: 4, 8, 16, 32
- [ ] Image size: 448, 640, 896, 1280
- [ ] Warmup: 0%, 3%, 5%, 10%
- [ ] Weight decay: 0.01, 0.05, 0.1
- [ ] Label smoothing: 0, 0.05, 0.1
- [ ] Gradient clipping: 1.0, 5.0, none

---

## 2. Detection Heads

### A. ViTDet Feature Pyramid
- Reshape ViT tokens to 2D spatial grid
- Extract multi-scale features from layers 4, 8, 12
- Simple FPN (lateral connections + top-down)
- YOLO-style anchor-free detection head
- **Pro**: Well-proven, fast inference
- **Con**: Loses VLM text alignment

### B. DETR / RT-DETR
- Learned object queries (100-300)
- Cross-attention with encoder features
- Hungarian matching loss
- **Pro**: Elegant, no NMS needed, end-to-end
- **Con**: Slow convergence, needs more data

### C. Grounding Head (YOLO-World style)
- Region-text alignment via dot product
- Text embeddings as classification weights
- Dense prediction with open vocabulary
- **Pro**: True zero-shot classification
- **Con**: Needs good text-image alignment

### D. Sliding Window + Classification
- Dense sliding window at multiple scales
- MarkusNet classifies each position
- NMS to merge
- **Pro**: Simplest, uses existing classifier
- **Con**: Very slow inference

### E. CenterNet-style
- Predict center heatmap + offset + size
- Single-stage, no anchors
- **Pro**: Simple, fast
- **Con**: Struggles with overlapping objects

### F. Deformable DETR
- Deformable attention over multi-scale features
- Much faster convergence than vanilla DETR
- **Pro**: Best of DETR + FPN
- **Con**: Complex implementation

---

## 3. Multi-Task Objectives

### Detection Objectives
- [ ] Box regression: L1, GIoU, CIoU, DIoU
- [ ] Classification: Cross-entropy, Focal loss, Quality Focal Loss
- [ ] Objectness: Binary cross-entropy, Varifocal loss
- [ ] Distribution Focal Loss (DFL) for box discretization

### Self-Supervised Objectives
- [ ] Masked image modeling (MAE-style): mask 75% patches, reconstruct
- [ ] Contrastive learning (CLIP-style): align image-text pairs
- [ ] DINO self-distillation: student-teacher with EMA
- [ ] Region-text alignment: align spatial features with product descriptions
- [ ] Rotation prediction: predict image rotation angle
- [ ] Jigsaw puzzle: predict patch permutation

### Multi-Task Combinations
- [ ] Detection + classification jointly
- [ ] Detection + contrastive text alignment
- [ ] Detection + masked image modeling (pre-training)
- [ ] Classification + KNN retrieval (reference matching)
- [ ] Dense captioning (describe what's at each location)

---

## 4. Video Processing

### Temporal Extensions
- [ ] Track products across video frames (MarkusNet already has tracker architecture from Qwen)
- [ ] Temporal consistency: same product = same class across frames
- [ ] Video-based pseudo-labeling: track a product, label once, propagate
- [ ] Temporal augmentation: use consecutive frames as different views

### Video Datasets
- [ ] Store walkthrough videos (4 videos from batch 1)
- [ ] Extract frames at different angles/distances
- [ ] Temporal interpolation for data augmentation

---

## 5. Model Soup & Merging

### Weight Averaging
- [ ] Uniform soup: average N checkpoints from different epochs
- [ ] Greedy soup: iteratively add checkpoints if they improve val score
- [ ] Exponential moving average (EMA): maintain running average during training
- [ ] Stochastic weight averaging (SWA): high constant LR + average

### Weight Interpolation
- [ ] SLERP: spherical linear interpolation between two models
- [ ] Linear interpolation: α * model_A + (1-α) * model_B
- [ ] Task arithmetic: model_base + α * (model_finetuned - model_base)
- [ ] TIES merging: trim, elect signs, merge

### Multi-Model Merging
- [ ] Merge detection-focused + classification-focused checkpoints
- [ ] Merge models trained on different data splits
- [ ] Cross-architecture distillation then merge

---

## 6. LoRA & Parameter-Efficient Fine-Tuning

### LoRA Variants
- [ ] Standard LoRA: low-rank adapters on attention Q/K/V/O
- [ ] QLoRA: LoRA on 4-bit quantized base
- [ ] LoRA on MLP layers (gate_proj, up_proj, down_proj)
- [ ] Rank sweep: r=4, 8, 16, 32, 64, 128
- [ ] Alpha sweep: α=8, 16, 32, 64
- [ ] Target modules: attention only, MLP only, both

### LoRA MoE (Mixture of Experts)
- [ ] Train separate LoRA adapters for different product categories
- [ ] Router network selects which adapter(s) to activate per token
- [ ] Capacity factor: how many experts per token (1, 2, 4)
- [ ] Expert specialization: by product type (bread, coffee, eggs, etc.)
- [ ] Merge LoRA MoE back to single model after training

### Other PEFT
- [ ] Adapters (bottleneck layers between transformer blocks)
- [ ] Prefix tuning (learnable prefix tokens)
- [ ] Prompt tuning (learnable soft prompts)
- [ ] BitFit (bias-only fine-tuning)
- [ ] Side-tuning (parallel adapter network)

---

## 7. Knowledge Distillation

### Teacher Models
- [ ] YOLO-World XL (open-vocab detection teacher)
- [ ] Qwen3-VL-8B (full VLM teacher)
- [ ] EVA-CLIP ViT-E (vision feature teacher)
- [ ] Grounding DINO (grounded detection teacher)

### Distillation Strategies
- [ ] Logit distillation (KL divergence on outputs)
- [ ] Feature distillation (MSE on intermediate features)
- [ ] Attention distillation (match attention patterns)
- [ ] Region distillation (dense feature alignment)
- [ ] Progressive distillation (shrink teacher gradually)

---

## 8. Architecture Modifications

### Vision Encoder
- [ ] Add learnable [DET] tokens to ViT input
- [ ] Window attention for higher resolution
- [ ] Flash Attention 2 for memory efficiency
- [ ] Adaptive token merging (reduce tokens dynamically)

### Hybrid Decoder
- [ ] Experiment with Mamba-to-attention ratio (currently 9:3)
- [ ] Try all-attention (standard transformer)
- [ ] Try all-Mamba (pure SSM)
- [ ] Cross-attention between vision and text features

### Output Head
- [ ] Multi-scale output (like FPN)
- [ ] Shared vs separate classification + regression heads
- [ ] Learnable anchor points
- [ ] Cascade detection (coarse → fine)

---

## Autoresearch Priority Queue

### Phase 1: Baseline + Quick Wins (this week)
1. ViTDet head on MarkusNet vision encoder (Approach A)
2. Text embedding classification head (zero-shot)
3. Evaluate on competition val set
4. Model soup from existing checkpoints
5. LoRA fine-tuning on external detection data

### Phase 2: Systematic Exploration
6. DETR head comparison
7. Data mixture optimization
8. Multi-task (detection + contrastive)
9. Self-supervised pre-training on shelf images
10. LoRA MoE for category specialization

### Phase 3: Advanced
11. Knowledge distillation from YOLO-World XL
12. Video temporal consistency
13. Architecture modifications (window attention, adaptive merging)
14. Weight merging experiments
15. Publication-ready evaluation on standard benchmarks

---

## Evaluation Protocol

- **Detection**: mAP@0.5 on 248 competition images (class-agnostic)
- **Classification**: mAP@0.5 on 248 competition images (class-aware, 356 cats)
- **Combined**: 0.7 × det + 0.3 × cls
- **Zero-shot**: No training on competition data
- **Few-shot**: k-shot per category from competition data (k=1, 5, 10)
- **Standard benchmarks**: COCO, LVIS, Objects365 (for publication)
