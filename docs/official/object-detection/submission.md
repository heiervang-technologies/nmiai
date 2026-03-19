# Object Detection - Submission Specification

## Entry Point

Place `run.py` at the root of your submission archive.

```bash
python run.py --input /data/images --output /output/predictions.json
```

## Output Format

The output file `predictions.json` must contain a JSON array of detection objects:

```json
[
  {
    "image_id": 1,
    "category_id": 42,
    "bbox": [100.0, 200.0, 50.0, 75.0],
    "score": 0.95
  }
]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | int | Image identifier matching the dataset |
| `category_id` | int | Category ID in range 0-355 |
| `bbox` | array[4] | Bounding box as `[x, y, width, height]` |
| `score` | float | Confidence score |

## Submission Constraints

| Constraint | Limit |
|------------|-------|
| Max uncompressed size | 420 MB |
| Max total files | 1,000 |
| Max Python files (.py) | 10 |
| Max weight files | 3 |

## Runtime Environment

| Resource | Specification |
|----------|---------------|
| Python | 3.11 |
| CPU | 4 vCPU |
| RAM | 8 GB |
| GPU | NVIDIA L4 (24 GB VRAM) |
| CUDA | 12.4 |
| Timeout | 300 seconds |

## Blocked Imports

The following Python modules are **blocked** and cannot be imported:

- `os`
- `sys`
- `subprocess`
- `socket`
- `pickle`
- `yaml`
- `requests`
- `urllib`
- `multiprocessing`
- `threading`

Use `pathlib` instead of `os` for all file system operations.
