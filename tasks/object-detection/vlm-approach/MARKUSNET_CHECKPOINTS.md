# MarkusNet Checkpoints Reference

## What is MarkusNet?

MarkusNet is a pruned Qwen3.5-0.8B vision-language model repurposed as a grocery product classifier. The full pipeline uses ONNX YOLO for detection (bounding boxes) and MarkusNet for classification (category prediction on detected crops).

**Architecture:** Pruned Qwen3.5-0.8B = 12 ViT vision blocks + spatial merger + 12 hybrid text decoder blocks (3 linear attention + 1 full attention, repeated 3x) + a 2-layer classification head.

The original Qwen3.5-0.8B has 24 text layers; MarkusNet keeps only the first 12. The `embed_tokens` and `lm_head` are used during training but stripped for the exported submission checkpoint to save space.

---

## Checkpoint Inventory

### Primary Checkpoints

| File | Size | Description | Accuracy |
|------|------|-------------|----------|
| `training_output/best/best.pt` | 1.7 GB | Best training checkpoint (full model + cls head) | 91.1% |
| `training_output/final/final.pt` | 1.7 GB | Final training checkpoint (step 14205) | N/A |
| `exported/markusnet_351m_fp16.pt` | 670 MB | Stripped (no embed_tokens/lm_head), FP16 | 89.7% |
| `exported/markusnet_351m_int8.pt` | 338 MB | Stripped + per-channel INT8 quantized | 89.7% |

### Pruned Base Weights (Pre-training Initialization)

These are pruned versions of the base Qwen3.5-0.8B before any fine-tuning. Used to initialize training.

| File | Size | Text Layers Kept |
|------|------|-----------------|
| `pruned/pruned_fp16_6blocks.pth` | 433 MB | 6 |
| `pruned/pruned_fp16_8blocks.pth` | 509 MB | 8 |
| `pruned/pruned_fp16_10blocks.pth` | 591 MB | 10 |
| `pruned/pruned_fp16_12blocks.pth` | 668 MB | 12 |

The 12-block variant was used for training. The `pruned/` directory also contains `config.json` and `model.safetensors` (1.6 GB) that allow loading the pruned model via `transformers.AutoModelForImageTextToText`.

### Periodic Training Checkpoints

`training_output/checkpoint-{500..14000}/checkpoint.pt` -- 28 checkpoints at 500-step intervals. Each contains `model_state`, `cls_head_state`, `optimizer_state`, `global_step`, `epoch`.

### Supporting Model Files

| File | Size | Purpose |
|------|------|---------|
| `dinov2_vits14.pth` | 84 MB | DINOv2 ViT-S/14 backbone (alternative classifier) |
| `dinov3_vits16.pth` | 82 MB | DINOv3 ViT-S/16 backbone |
| `linear_probe.pth` | 0.5 MB | Linear probe on DINOv2 embeddings |
| `dinov3_linear_probe.pth` | 0.5 MB | Linear probe on DINOv3 embeddings |
| `ref_embeddings.pth` | 2.4 MB | DINOv2 reference embeddings per category |
| `dinov3_ref_embeddings.pth` | 2.4 MB | DINOv3 reference embeddings |
| `training_embeddings.pth` | 33 MB | All training crop embeddings |
| `best.onnx` (symlink) | ~115 MB | YOLO detector (symlink to yolo-approach) |

---

## Architecture Details

### Vision Encoder (12 ViT blocks)
- Hidden size: 768
- Intermediate size: 3072
- Attention heads: 12
- Patch size: 16x16, temporal patch: 2
- Position embeddings: 2304 (48x48 grid)
- Spatial merge: 2x2 -> output hidden 1024

### Text Decoder (12 hybrid layers)
- Hidden size: 1024
- Intermediate size: 3584
- Full attention heads: 8 (with 2 KV heads, head dim 256)
- Linear attention (Gated DeltaNet): 16 K/V heads, 128 dim, conv kernel 4
- Layer pattern: `[linear, linear, linear, full] x 3`
- RoPE: theta=10M, partial_rotary=0.25, mRoPE sections=[11,11,10]

### Classification Head
- `Linear(1024, 1024) -> GELU -> Dropout -> Linear(1024, 356)`
- Pools hidden states via mean over sequence length
- 356 grocery product categories

### Parameter Counts
- Full model (with embed_tokens + lm_head): ~858M
- Stripped model (vision + merger + 12 text blocks + cls head): ~351M
- Classification head alone: ~1.4M

---

## Checkpoint Format

All `.pt` checkpoints are `torch.save()` dicts. The training checkpoints (`best.pt`, `final.pt`, `checkpoint-*/checkpoint.pt`) contain:

```python
{
    "model_state": OrderedDict,    # Full model state_dict (transformers format)
    "cls_head_state": OrderedDict, # Classification head weights
    "global_step": int,
    "epoch": int,
    "accuracy": float,             # Training accuracy (in best.pt)
    "loss": float,                 # Training loss (in best.pt)
    # Some also have:
    "optimizer_state": OrderedDict,
    "det_head_state": OrderedDict, # Only in multitask checkpoints
}
```

The exported checkpoints (`markusnet_351m_fp16.pt`) contain:

```python
{
    "model_state": OrderedDict,    # Stripped (no embed_tokens/lm_head)
    "cls_head_state": OrderedDict,
    "accuracy": float,
    "global_step": int,
    "architecture": dict,          # Metadata about the model
}
```

The INT8 checkpoint (`markusnet_351m_int8.pt`) additionally contains:

```python
{
    "scales": dict,                # Per-channel quantization scales
    "quantization": "per_channel_int8",
}
```

### State Dict Key Prefixes

Training checkpoints use transformers-style keys:
- Vision: `model.visual.patch_embed.proj.weight`, `model.visual.blocks.{i}.attn.qkv.weight`, `model.visual.merger.linear_fc1.weight`
- Text: `model.language_model.embed_tokens.weight`, `model.language_model.layers.{i}.self_attn.q_proj.weight`, `model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight`
- Classification head: `head.0.weight`, `head.0.bias`, `head.3.weight`, `head.3.bias`

---

## How to Run Inference

### Option 1: Pure PyTorch (No Dependencies Beyond torch/PIL/cv2/onnxruntime)

This is the submission inference path. Uses `run_markusnet.py` which contains a complete standalone reimplementation of the Qwen3.5-0.8B architecture in pure PyTorch -- no `transformers` library needed.

```bash
cd ~/ht/nmiai/tasks/object-detection/vlm-approach

# Full pipeline: YOLO detection + MarkusNet classification
python run_markusnet.py --input /path/to/images --output /path/to/predictions.json
```

The script:
1. Loads YOLO via ONNX Runtime for detection
2. Loads MarkusNet from `training_output/best/best.pt`
3. For each image: detect boxes -> crop -> classify each crop
4. Outputs COCO-format JSON predictions

**Classify crops only (code example):**

```python
import torch
from PIL import Image
from run_markusnet import MarkusNet

device = torch.device("cuda")
dtype = torch.bfloat16

model = MarkusNet()
model.load_checkpoint("training_output/best/best.pt", device)
model = model.to(dtype).to(device)
model.eval()

# Classify a list of PIL image crops
crops = [Image.open("crop1.jpg"), Image.open("crop2.jpg")]
with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
    category_ids, confidences = model.classify_crops(crops, device)

print(category_ids)   # numpy array of predicted category indices (0-355)
print(confidences)    # numpy array of softmax confidence scores
```

**Note:** `run_markusnet.py` processes crops one at a time through the full vision+language pipeline (not true batching), in sub-batches of 64 crops. Each crop is resized to 448x448, converted to patches, run through the vision encoder, then the language model with the chat template token sequence, and finally the classification head.

### Option 2: Using transformers Library (For Training/Continued Fine-tuning)

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

device = torch.device("cuda")
PRUNED_DIR = "pruned"  # Contains config.json + model.safetensors

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    PRUNED_DIR,
    dtype=torch.bfloat16,
    ignore_mismatched_sizes=True,
    trust_remote_code=True,
)
model = model.to(device)

# Load trained weights from checkpoint
ckpt = torch.load("training_output/best/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])

# Build classification head
import torch.nn as nn
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=356, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        return self.head(pooled)

cls_head = ClassificationHead().to(device).to(torch.bfloat16)
cls_head.load_state_dict(ckpt["cls_head_state"])

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

# Classify a crop
crop = Image.open("crop.jpg").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": crop},
    {"type": "text", "text": "classify"},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs = processor(images=[crop], text=[text], return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()
cls_head.eval()
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    outputs = model.model(**inputs, output_hidden_states=True)
    hidden = outputs.last_hidden_state
    logits = cls_head(hidden)
    pred_class = logits.argmax(dim=-1).item()
    confidence = torch.softmax(logits, dim=-1).max().item()

print(f"Predicted class: {pred_class}, confidence: {confidence:.3f}")
```

---

## Training Scripts

| Script | Purpose | Base Model |
|--------|---------|-----------|
| `prune_qwen35.py` | Prune Qwen3.5-0.8B to K text layers, save to `pruned/` | Qwen3.5-0.8B (HF) |
| `train_qwen35_classify.py` | LoRA fine-tune with Unsloth/SFTTrainer on crop images | unsloth/Qwen3.5-0.8B |
| `train_pruned_multitask.py` | Interleaved cls + detection training on pruned model | pruned/12-block |
| `train_continue.py` | Continue training from best checkpoint with class-weighted loss | best.pt checkpoint |
| `train_coco_minitrain.py` | Pre-train on COCO-minitrain 80 classes for transfer learning | pruned/12-block |
| `train_qwen3vl_vision.py` | Vision-encoder-only training (Qwen3-VL-2B, alternative) | Qwen3-VL-2B-Instruct |
| `train_ddp.py` | DDP multi-GPU training variant | pruned/12-block |
| `train_objects365.py` | Training on Objects365 dataset | pruned/12-block |
| `train_distill.py` | Knowledge distillation training | pruned/12-block |
| `export_markusnet.py` | Strip embed_tokens/lm_head, export FP16 and INT8 | best.pt checkpoint |

### Training Pipeline That Produced `best.pt`

1. **Pruning:** `prune_qwen35.py` downloaded Qwen3.5-0.8B from HuggingFace, kept 12 text layers, saved to `pruned/`
2. **Training:** `train_pruned_multitask.py` trained interleaved classification (22.7k grocery crops, 356 classes) + detection (248 shelf images) with shared backbone. 8 epochs, batch size 4 cls / 1 det, lr=5e-5, AdamW, cosine schedule
3. **Result:** Step 14205, epoch 5, accuracy 91.1%, loss 0.368

### Training Data

- **Classification:** 22,731 grocery product crop images across 356 categories, cached in `cached_dataset/crops/` with metadata in `cached_dataset/samples.json`
- **Detection:** COCO-format annotations from `../data-creation/data/coco_dataset/train/`
- Source: crops are extracted from annotated shelf images with bounding boxes

---

## Metrics

| Checkpoint | Training Accuracy | Notes |
|-----------|-------------------|-------|
| `training_output/best/best.pt` | 91.1% | Best during multitask training |
| `exported/markusnet_351m_fp16.pt` | 89.7% | Earlier best (step 11364), used for export |
| `exported/markusnet_351m_int8.pt` | 89.7% | Same as FP16 export, quantized |

The combined competition score formula is: `0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`

Evaluation script: `eval_stratified_map.py` -- runs the full submission pipeline against a YOLO-format validation split and computes detection and classification mAP@0.5 separately.

---

## Dependencies

### For inference with `run_markusnet.py` (pure PyTorch path):
```
torch >= 2.6.0
onnxruntime-gpu >= 1.24.4
opencv-python-headless
Pillow
numpy
```

No `transformers` library needed -- the architecture is reimplemented in pure PyTorch.

### For training (any `train_*.py` script):
```
torch >= 2.6.0
torchvision >= 0.21.0
transformers >= 5.3.0
Pillow
numpy
wandb
```

Install with:
```bash
cd ~/ht/nmiai/tasks/object-detection/vlm-approach
uv sync
```

The `pyproject.toml` also lists `unsloth`, `timm`, `scikit-learn`, `ultralytics`, and `datasets` as dependencies (needed for specific training scripts).

### For the Unsloth LoRA training path (`train_qwen35_classify.py`):
```
unsloth[cu124] >= 2026.3.8
trl
datasets
```

---

## Key Design Decisions

1. **Why Qwen3.5-0.8B?** Native multimodal VLM with hybrid linear+full attention. Small enough to fit in 400MB submission ZIP when pruned and quantized.

2. **Why 12 text layers?** The 12-block pruning variant (668 MB FP16) keeps the full vision encoder and enough text capacity for classification, while fitting the submission budget after INT8 quantization (338 MB + ~115 MB YOLO = ~453 MB before stripping).

3. **Why reimplement in pure PyTorch?** The competition sandbox does not have `transformers` pre-installed. Bundling it would consume ~200 MB of the 420 MB ZIP budget. The pure PyTorch implementation in `run_markusnet.py` has zero external dependencies beyond torch/cv2/PIL/onnxruntime.

4. **Why keep embed_tokens during training?** The chat template token sequence (with special tokens for image boundaries) is needed to correctly process vision+text inputs through the language model layers. At export time, embed_tokens and lm_head are stripped since inference constructs embeddings directly from the vision encoder output.
