"""Fix all dataset.yaml files to use proper YAML escaping via yaml.dump()."""
import json
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"


def generate_proper_yaml(dataset_path: Path, title: str):
    """Generate a properly escaped dataset.yaml using yaml.dump()."""
    with open(COCO_ANN) as f:
        coco = json.load(f)
    categories = coco["categories"]

    names = {cat["id"]: cat["name"] for cat in categories}

    yaml_data = {
        "path": str(dataset_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(categories),
        "names": names,
    }

    yaml_path = dataset_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"# {title}\n")
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Fixed: {yaml_path}")


def main():
    datasets = [
        (DATA_DIR / "yolo_dataset", "NorgesGruppen Object Detection Dataset"),
        (DATA_DIR / "yolo_augmented", "NorgesGruppen Object Detection Dataset (Augmented v1)"),
        (DATA_DIR / "yolo_augmented_v2", "NorgesGruppen Object Detection Dataset (Augmented v2 - SAM3)"),
    ]

    for path, title in datasets:
        if path.exists():
            generate_proper_yaml(path, title)
        else:
            print(f"Skipped (not found): {path}")

    print("\nAll dataset.yaml files regenerated with proper YAML escaping.")


if __name__ == "__main__":
    main()
