# Submission Guide — Object Detection Task

## Critical Learnings

### 1. Ultralytics Version Mismatch (SUBMISSION KILLER)
- **Sandbox has ultralytics 8.1.0** (Jan 2024)
- Training locally with newer ultralytics (8.4.24+) produces `.pt` files that **CANNOT load in 8.1.0**
- **Exit code 2** = model failed to load, not an argparse error
- **Fix:** Export to ONNX and use `onnxruntime-gpu` (pre-installed). Zero ultralytics dependency at inference.
- **Alternative:** Train with exactly ultralytics 8.1.0 to produce compatible `.pt` files

### 2. Sandbox Contract
```bash
python run.py --input /data/images/ --output /predictions.json
```
- `run.py` MUST be at ZIP root (not in a subfolder)
- Only `--data` and `--output` args. No other args.
- Exit code MUST be 0. Any non-zero = "evaluation failed"
- Max 420MB ZIP, 300s timeout, NVIDIA L4 GPU (24GB VRAM)

### 3. Blocked Imports
- `os`, `subprocess`, `socket`, `ctypes` — BLOCKED
- `eval()`, `exec()`, `compile()`, `__import__()` — BLOCKED
- Use `pathlib` for all file operations
- Use `open()` directly for file I/O (not os.path)
- `cv2.imread(str(path))` works fine

### 4. Output JSON Format
```json
[
  {
    "image_id": "img_00042.jpg",
    "predictions": [
      {
        "bbox": [x, y, width, height],
        "category_id": 0,
        "confidence": 0.95
      }
    ]
  }
]
```
- `bbox`: COCO format [x_min, y_min, width, height] (NOT xyxy)
- `category_id`: integer 0-355
- `score: float 0.0-1.0
- `image_id`: filename string including extension

### 5. ONNX Submission Pattern (RECOMMENDED)
```
submission.zip
├── run.py       ← pure onnxruntime inference, no ultralytics
├── best.onnx    ← exported model weights (FP16)
└── (optional: embeddings, classifier weights)
```

**ONNX export command:**
```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=1280, half=True, simplify=True)
```

**ONNX inference providers:**
```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(str(model_path), providers=providers)
```

### 6. Post-Processing for YOLOv8 ONNX
- Output shape: `[1, 4+nc, num_boxes]` — needs transpose to `[num_boxes, 4+nc]`
- First 4 values are center_x, center_y, width, height (NOT x1,y1,x2,y2)
- Convert to COCO format: x1 = cx - w/2, y1 = cy - h/2
- Apply letterbox inverse: subtract padding, divide by ratio
- Per-class NMS with IoU threshold ~0.45
- max_det=300 (shelf images are dense)
- conf_thresh=0.001 for mAP evaluation (very low to maximize recall)

### 7. Scoring Details
- **70% detection mAP** (IoU >= 0.5) — did you find the products?
- **30% classification mAP** (IoU >= 0.5) — did you identify the right product?
- Detection-only (all category_id=0) caps at 70% max score
- IoU 0.5 is lenient — loose boxes are acceptable

### 8. Submission Limits
- **3 submissions per day** (resets midnight UTC = 01:00 CET)
- **Max 2 in-flight** (concurrent evaluations)
- Infrastructure errors (their fault) don't count against limit
- Each eval takes ~5 min (300s timeout)

### 9. Pre-installed Packages (FREE — no ZIP space)
- PyTorch 2.6.0+cu124
- torchvision 0.21.0+cu124
- ultralytics 8.1.0 (ONLY supports YOLOv8, NOT YOLO11/26)
- onnxruntime-gpu 1.20.0
- opencv-python-headless 4.9.0.80
- timm (includes DINOv2 AND DINOv3)
- supervision, albumentations, Pillow, numpy, scipy, scikit-learn
- pycocotools, safetensors

### 10. Common Pitfalls
- [ ] Don't assume ultralytics version — always test with 8.1.0 or use ONNX
- [ ] Don't use `os.path` — use `pathlib`
- [ ] Don't forget to handle empty prediction lists (images with no detections)
- [ ] Don't use `indent=2` in json.dumps for large outputs (wastes time/space)
- [ ] Don't forget letterbox padding inverse when scaling boxes back
- [ ] Test with `python run.py --input <path> --output <path>` locally before uploading
- [ ] Verify ZIP has run.py at root: `unzip -l submission.zip | head`

## Submission Checklist

Before every upload:
1. [ ] `run.py` at ZIP root
2. [ ] Model weights file referenced correctly (relative to `__file__`)
3. [ ] No blocked imports (os, subprocess, socket, ctypes)
4. [ ] Output format matches spec (image_id, predictions with bbox/category_id/confidence)
5. [ ] bbox is COCO format [x, y, w, h] not [x1, y1, x2, y2]
6. [ ] Tested locally: `python run.py --input <test_images> --output /tmp/test.json`
7. [ ] JSON output is valid: `python -c "import json; json.load(open('/tmp/test.json'))"`
8. [ ] ZIP size under 420MB
9. [ ] Exit code is 0
