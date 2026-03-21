"""
Build the DEFINITIVE "train-on-everything" dataset combining ALL data sources.

Sources:
1. Original COCO 248 images with gold labels
2. Albumentations augmented images (from yolo_augmented v1)
3. Copy-paste augmented images (from augmented/ dir)
4. PL mapped images (capped at 500/category)
5. Pseudo-labeled store photos (conf>0.5)
6. Pseudo-labeled video frames (conf>0.5)
7. Boosted weak category images

Two variants:
- "final_with_val": 50-image val held out for model selection
- "final_no_val": Everything in train for final submission retraining
"""
import json
import shutil
import yaml
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"


def count_dataset(d: Path) -> tuple[int, int]:
    train = len(list((d / "train/images").iterdir())) if (d / "train/images").exists() else 0
    val = len(list((d / "val/images").iterdir())) if (d / "val/images").exists() else 0
    return train, val


def link_from(src_dir: Path, dst_images: Path, dst_labels: Path, prefix: str = "") -> int:
    """Link images and copy labels from a YOLO dataset split."""
    count = 0
    img_dir = src_dir / "images" if (src_dir / "images").exists() else src_dir
    lbl_dir = src_dir / "labels" if (src_dir / "labels").exists() else None

    for img in img_dir.iterdir():
        if not img.name.endswith((".jpg", ".jpeg", ".png")):
            continue
        name = f"{prefix}{img.name}" if prefix else img.name
        dst = dst_images / name
        if dst.exists():
            continue
        dst.symlink_to(img.resolve())

        if lbl_dir:
            lsrc = lbl_dir / (img.stem + ".txt")
            ldst = dst_labels / (f"{prefix}{img.stem}.txt" if prefix else (img.stem + ".txt"))
            if lsrc.exists() and not ldst.exists():
                shutil.copy2(lsrc, ldst)
        count += 1
    return count


def write_yaml(out_dir: Path, comment: str):
    with open(COCO_ANN) as f:
        coco = json.load(f)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    with open(out_dir / "dataset.yaml", "w") as f:
        f.write(f"# {comment}\n")
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def build(out_dir: Path, include_val_in_train: bool = False):
    for split in ["train", "val"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    sources = {}

    # 1. V5 train (includes V4 = V3 + pseudo-labels, plus boosted weak)
    v5 = DATA_DIR / "yolo_augmented_v5"
    if v5.exists():
        n = link_from(v5 / "train", out_dir / "train" / "images", out_dir / "train" / "labels")
        sources["V5 train"] = n

    # 2. V5 val -> either val or train
    if v5.exists():
        val_target = "train" if include_val_in_train else "val"
        n = link_from(v5 / "val", out_dir / val_target / "images", out_dir / val_target / "labels")
        sources[f"V5 val -> {val_target}"] = n

    # 3. If merging val into train, create a tiny dummy val (YOLO requires it)
    if include_val_in_train:
        # Use first 5 train images as dummy val
        train_imgs = sorted((out_dir / "train" / "images").iterdir())[:5]
        for img in train_imgs:
            dst = out_dir / "val" / "images" / img.name
            if not dst.exists():
                dst.symlink_to(img.resolve())
            lsrc = out_dir / "train" / "labels" / (img.stem + ".txt")
            ldst = out_dir / "val" / "labels" / (img.stem + ".txt")
            if lsrc.exists() and not ldst.exists():
                shutil.copy2(lsrc, ldst)

    return sources


def main():
    print("=== Building Final Datasets ===\n")

    # Variant 1: With val held out
    out1 = DATA_DIR / "yolo_final_with_val"
    if out1.exists():
        shutil.rmtree(out1)
    sources1 = build(out1, include_val_in_train=False)
    write_yaml(out1, "FINAL dataset with val held out for model selection")
    t1, v1 = count_dataset(out1)
    print(f"final_with_val: {t1} train, {v1} val")
    for name, count in sources1.items():
        print(f"  {name}: {count}")

    # Variant 2: Everything in train
    out2 = DATA_DIR / "yolo_final_no_val"
    if out2.exists():
        shutil.rmtree(out2)
    sources2 = build(out2, include_val_in_train=True)
    write_yaml(out2, "FINAL dataset with ALL data in train (for submission retraining)")
    t2, v2 = count_dataset(out2)
    print(f"\nfinal_no_val: {t2} train, {v2} dummy val")
    for name, count in sources2.items():
        print(f"  {name}: {count}")

    # Gap analysis on final dataset
    print(f"\n=== GAP ANALYSIS (final_with_val train) ===")
    with open(COCO_ANN) as f:
        cat_names = {c["id"]: c["name"] for c in json.load(f)["categories"]}

    counts = Counter()
    for f in (out1 / "train" / "labels").glob("*.txt"):
        for line in open(f):
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1

    vals = sorted(counts.values()) if counts else [0]
    print(f"Categories covered: {len(counts)}/356")
    print(f"Per-category: min={vals[0]}, P25={vals[len(vals)//4]}, median={vals[len(vals)//2]}, P75={vals[3*len(vals)//4]}, max={vals[-1]}")
    print(f"Total annotations: {sum(vals)}")

    # Categories below 20
    weak = [(cid, counts.get(cid, 0)) for cid in range(356) if counts.get(cid, 0) < 20]
    weak.sort(key=lambda x: x[1])
    print(f"\nCategories with <20 annotations: {len(weak)}")
    for cid, cnt in weak:
        print(f"  {cid:3d} ({cnt:3d}): {cat_names.get(cid, '?')}")


if __name__ == "__main__":
    main()
