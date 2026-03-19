"""Generate a valid data.yaml using the yaml library for proper escaping."""
import json
import yaml
from pathlib import Path

COCO_ANN = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/coco_dataset/train/annotations.json")
DATASET_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/yolo-approach/dataset")
OUTPUT = Path("/home/me/ht/nmiai/tasks/object-detection/yolo-approach/data.yaml")

with open(COCO_ANN) as f:
    coco = json.load(f)

categories = sorted(coco['categories'], key=lambda c: c['id'])
names = {c['id']: c['name'] for c in categories}

data = {
    'path': str(DATASET_DIR),
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(categories),
    'names': names,
}

with open(OUTPUT, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

# Verify it loads back
with open(OUTPUT) as f:
    loaded = yaml.safe_load(f)
assert loaded['nc'] == 356
assert len(loaded['names']) == 356
print(f"Written valid data.yaml to {OUTPUT}")
print(f"nc={loaded['nc']}, names={len(loaded['names'])}")
