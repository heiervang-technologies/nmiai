# NorgesGruppen Data: Submission Format

## Zip Structure

Your `.zip` must contain `run.py` at the root. You may include model weights and Python helper files.

    submission.zip
    ├── run.py          # Required: entry point
    ├── model.onnx      # Optional: model weights (.pt, .onnx, .safetensors, .npy)
    └── utils.py        # Optional: helper code

**Limits:**

<div class="table-scroll-wrapper">

| Limit | Value |
|----|----|
| Max zip size (uncompressed) | 420 MB |
| Max files | 1000 |
| Max Python files | 10 |
| Max weight files (.pt, .pth, .onnx, .safetensors, .npy) | 3 |
| Max weight size total | 420 MB |
| Allowed file types | .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy |

</div>

## run.py Contract

Your script is executed as:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>python run.py --input /data/images --output /output/predictions.json</code></pre>
</figure>

### Input

`/data/images/` contains JPEG shelf images. File names use the format `img_XXXXX.jpg` (e.g., `img_00042.jpg`).

### Output

Write a JSON array to the `--output` path:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>[
  {
    &quot;image_id&quot;: 42,
    &quot;category_id&quot;: 0,
    &quot;bbox&quot;: [120.5, 45.0, 80.0, 110.0],
    &quot;score&quot;: 0.923
  }
]</code></pre>
</figure>

<div class="table-scroll-wrapper">

| Field | Type | Description |
|----|----|----|
| `image_id` | int | Numeric ID from filename (`img_00042.jpg` → `42`) |
| `category_id` | int | Product category ID (0-355). See `categories` list in annotations.json |
| `bbox` | \[x, y, w, h\] | Bounding box in COCO format |
| `score` | float | Confidence score (0-1) |

</div>

## Scoring

Your score combines detection and classification:

- **70% detection mAP** — did you find the products? (bounding box IoU ≥ 0.5, category ignored)
- **30% classification mAP** — did you identify the right product? (IoU ≥ 0.5 AND correct category_id)

Detection-only submissions (`category_id: 0` for all predictions) score up to 70%. Product identification adds the remaining 30%.

## Product Categories

The training data `annotations.json` contains a `categories` list mapping integer IDs to product names:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>&quot;categories&quot;: [
  {&quot;id&quot;: 0, &quot;name&quot;: &quot;VESTLANDSLEFSA TØRRE 10STK 360G&quot;, &quot;supercategory&quot;: &quot;product&quot;},
  {&quot;id&quot;: 1, &quot;name&quot;: &quot;COFFEE MATE 180G NESTLE&quot;, &quot;supercategory&quot;: &quot;product&quot;},
  ...
  {&quot;id&quot;: 356, &quot;name&quot;: &quot;unknown_product&quot;, &quot;supercategory&quot;: &quot;product&quot;}
]</code></pre>
</figure>

Your predictions must use the same `category_id` values. When training YOLOv8 on this COCO data (with `nc=357`), the model learns the mapping and outputs the correct category_id during inference.

## Sandbox Environment

Your code runs in a Docker container with these constraints:

<div class="table-scroll-wrapper">

| Resource | Limit                  |
|----------|------------------------|
| Python   | 3.11                   |
| CPU      | 4 vCPU                 |
| Memory   | 8 GB                   |
| GPU      | NVIDIA L4 (24 GB VRAM) |
| CUDA     | 12.4                   |
| Network  | None (fully offline)   |
| Timeout  | 300 seconds            |

</div>

### GPU

An NVIDIA L4 GPU is always available in the sandbox. Your code auto-detects it:

- `torch.cuda.is_available()` returns `True`
- No opt-in flag needed — GPU is always on
- For ONNX models, use `["CUDAExecutionProvider", "CPUExecutionProvider"]` as the provider list

### Pre-installed Packages

PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0, opencv-python-headless 4.9.0.80, albumentations 1.3.1, Pillow 10.2.0, numpy 1.26.4, scipy 1.12.0, scikit-learn 1.4.0, pycocotools 2.0.7, ensemble-boxes 1.0.9, timm 0.9.12, supervision 0.18.0, safetensors 0.4.2.

You **cannot** `pip install` at runtime.

## Training Environment

You can use **any computer vision architecture** — the sandbox supports all models via ONNX, custom PyTorch code, or the pre-installed frameworks. You don't need all sandbox packages for training — only match the versions of packages you actually use.

### Models available in the sandbox

These frameworks are pre-installed. If you train with the **exact same version**, you can submit `.pt` weights directly:

<div class="table-scroll-wrapper">

| Framework | Models | Pin this version |
|----|----|----|
| ultralytics 8.1.0 | YOLOv8n/s/m/l/x, YOLOv5u, RT-DETR-l/x | `ultralytics==8.1.0` |
| torchvision 0.21.0 | Faster R-CNN, RetinaNet, SSD, FCOS, Mask R-CNN | `torchvision==0.21.0` |
| timm 0.9.12 | ResNet, EfficientNet, ViT, Swin, ConvNeXt, etc. (as backbones) | `timm==0.9.12` |

</div>

### Models not in the sandbox

YOLOv9, YOLOv10, YOLO11, RF-DETR, Detectron2, MMDetection, HuggingFace Transformers — these packages are not installed. Two options:

1.  **Export to ONNX**: Export from any framework, load with `onnxruntime` in your `run.py`. Use opset version ≤ 20. Use `CUDAExecutionProvider` for GPU acceleration.
2.  **Include model code**: Put your model class in your `.py` files + `.pt` state_dict weights. Works if the model only uses standard PyTorch ops.

**HuggingFace `.bin` files**: The `.bin` extension is not allowed, but the format is identical to `.pt` (PyTorch pickle). Rename `.bin` → `.pt`, or convert with `safetensors.torch.save_file(state_dict, "model.safetensors")`.

**Models larger than 420 MB**: Quantize to FP16 or INT8 to fit within the 420 MB weight limit. FP16 is the recommended precision for L4 GPU inference — it's both smaller and faster.

### Setting up your training environment

Only install the packages you need, with matching versions:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code># YOLOv8 training
pip install ultralytics==8.1.0
 
# torchvision detector
pip install torch==2.6.0 torchvision==0.21.0
 
# Custom model with timm backbone
pip install torch==2.6.0 timm==0.9.12
 
# For GPU training, add the CUDA index:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124</code></pre>
</figure>

The sandbox has a GPU (NVIDIA L4 with CUDA 12.4), so GPU-trained weights run natively — no `map_location="cpu"` needed. Your code should auto-detect with `torch.cuda.is_available()`.

**Train anywhere:** You can train on any hardware — your laptop CPU, a cloud GPU, Google Colab, GCP VMs, etc. Models trained on any platform will run on the sandbox GPU. Use `state_dict` saves (not full model saves) or ONNX export for maximum compatibility.

### Version compatibility

<div class="table-scroll-wrapper">

| Risk | What happens | Fix |
|----|----|----|
| ultralytics 8.2+ weights on 8.1.0 | Model class changed, load fails | Pin `ultralytics==8.1.0` or export to ONNX |
| torch 2.7+ full model save on 2.6.0 | May reference newer operators | Use `torch.save(model.state_dict())`, not `torch.save(model)` |
| timm 1.0+ weights on 0.9.12 | Layer names changed, load fails | Pin `timm==0.9.12` or export to ONNX |
| ONNX opset \> 20 | onnxruntime 1.20.0 can't load it | Export with `opset_version=17` |

</div>

### Recommended weight formats

<div class="table-scroll-wrapper">

| Approach | Format | When to use |
|----|----|----|
| ONNX export | `.onnx` | Universal — any framework, 2-3x faster on CPU |
| ultralytics .pt (pinned 8.1.0) | `.pt` | Simple YOLOv8/RT-DETR workflow |
| state_dict + model class | `.pt` | Custom architectures with standard PyTorch ops |
| safetensors | `.safetensors` | Safe loading, no pickle, fast |

</div>

## Security Restrictions

The following imports are blocked by the security scanner:

- `os`, `sys`, `subprocess`, `socket`, `ctypes`, `builtins`, `importlib`
- `pickle`, `marshal`, `shelve`, `shutil`
- `yaml` (use `json` for config files instead)
- `requests`, `urllib`, `http.client`
- `multiprocessing`, `threading`, `signal`, `gc`
- `code`, `codeop`, `pty`

The following calls are blocked:

- `eval()`, `exec()`, `compile()`, `__import__()`, `getattr()` with dangerous names

Also blocked: ELF/Mach-O/PE binaries, symlinks, path traversal.

Use `pathlib` instead of `os` for file operations. Use `json` instead of `yaml` for config files.

## Creating Your Zip

`run.py` must be at the **root** of the zip — not inside a subfolder. This is the most common submission error.

**Linux / macOS (Terminal):**

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>cd my_submission/
zip -r ../submission.zip . -x &quot;.*&quot; &quot;__MACOSX/*&quot;</code></pre>
</figure>

**Windows (PowerShell):**

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="powershell" data-theme="github-dark-default"><code>cd my_submission
Compress-Archive -Path .\* -DestinationPath ..\submission.zip</code></pre>
</figure>

Do **not** right-click a folder and use "Compress" (macOS) or "Send to → Compressed folder" (Windows) — both nest files inside a subfolder.

**Verify your zip:**

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>unzip -l submission.zip | head -10</code></pre>
</figure>

You should see `run.py` directly — not `my_submission/run.py`.
