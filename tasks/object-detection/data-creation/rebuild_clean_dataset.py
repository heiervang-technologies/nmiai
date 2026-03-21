"""
Rebuild datasets with a CLEAN val split (no data leakage).

Usage:
  python rebuild_clean_dataset.py --val-images <path_to_clean_val_image_list.txt>

The val image list should be a text file with one image filename per line,
OR a directory containing the val images.

Builds:
1. yolo_clean_with_val: All training data MINUS clean val images
2. yolo_clean_no_val: Everything in train (for final submission)
"""
import argparse
import json
import shutil
import yaml
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
COCO_IMAGES = DATA_DIR / "coco_dataset" / "train" / "images"


def get_val_stems(val_source: str) -> set[str]:
    """Get val image stems from a file list or directory."""
    p = Path(val_source)
    if p.is_dir():
        return {f.stem for f in p.iterdir() if f.suffix in (".jpg", ".jpeg", ".png")}
    elif p.is_file():
        # Text file with one filename per line
        stems = set()
        for line in p.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                stems.add(Path(line).stem)
        return stems
    else:
        raise FileNotFoundError(f"Val source not found: {val_source}")


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = max(0, min(1, (x + w / 2) / img_w))
    cy = max(0, min(1, (y + h / 2) / img_h))
    nw = max(0, min(1, w / img_w))
    nh = max(0, min(1, h / img_h))
    return cx, cy, nw, nh


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


def build_clean_dataset(val_stems: set[str], out_dir: Path, include_val_in_train: bool = False):
    """Build dataset ensuring NO val images leak into train."""
    for split in ["train", "val"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    with open(COCO_ANN) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    from collections import defaultdict
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    train_count = 0
    val_count = 0

    # 1. Original COCO images -> split into clean train/val
    for img_id, img_info in img_map.items():
        filename = img_info["file_name"]
        stem = Path(filename).stem
        img_w, img_h = img_info["width"], img_info["height"]

        if stem in val_stems:
            split = "val" if not include_val_in_train else "train"
        else:
            split = "train"

        # Find source image
        src = COCO_IMAGES / filename
        if not src.exists():
            continue

        dst = out_dir / split / "images" / filename
        if not dst.exists():
            dst.symlink_to(src.resolve())

        # Write YOLO labels
        anns = img_anns.get(img_id, [])
        label_file = out_dir / split / "labels" / (stem + ".txt")
        with open(label_file, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]
                cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                f.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        if split == "val":
            val_count += 1
        else:
            train_count += 1

    # 2. Add augmented/external data to train ONLY
    # These are safe - they're augmented copies or external data, not original images
    aug_sources = [
        DATA_DIR / "yolo_augmented_v5" / "train",  # V5 has augmented + PL + pseudo + boosted
    ]

    for src_split in aug_sources:
        if not (src_split / "images").exists():
            continue
        for img_path in (src_split / "images").iterdir():
            stem = img_path.stem
            # CRITICAL: Skip if this is an original image that's in val
            if stem in val_stems:
                continue
            # Skip original images (they're already added above)
            if img_path.name.startswith("img_"):
                continue

            dst = out_dir / "train" / "images" / img_path.name
            if dst.exists():
                continue
            dst.symlink_to(img_path.resolve())

            label_src = src_split / "labels" / (stem + ".txt")
            label_dst = out_dir / "train" / "labels" / (stem + ".txt")
            if label_src.exists() and not label_dst.exists():
                shutil.copy2(label_src, label_dst)
            train_count += 1

    # 3. If no_val mode, create dummy val
    if include_val_in_train:
        dummy = sorted((out_dir / "train" / "images").iterdir())[:5]
        for img in dummy:
            d = out_dir / "val" / "images" / img.name
            if not d.exists():
                d.symlink_to(img.resolve())
            ls = out_dir / "train" / "labels" / (img.stem + ".txt")
            ld = out_dir / "val" / "labels" / (img.stem + ".txt")
            if ls.exists() and not ld.exists():
                shutil.copy2(ls, ld)

    return train_count, val_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-images", required=True, help="Path to val image list or directory")
    args = parser.parse_args()

    val_stems = get_val_stems(args.val_images)
    print(f"Clean val images: {len(val_stems)}")

    # Build with val
    out1 = DATA_DIR / "yolo_clean_with_val"
    if out1.exists():
        shutil.rmtree(out1)
    train1, val1 = build_clean_dataset(val_stems, out1, include_val_in_train=False)
    write_yaml(out1, "CLEAN dataset - no data leakage between train/val")
    print(f"\nyolo_clean_with_val: {train1} train, {val1} val")

    # Build without val (everything in train)
    out2 = DATA_DIR / "yolo_clean_no_val"
    if out2.exists():
        shutil.rmtree(out2)
    train2, val2 = build_clean_dataset(val_stems, out2, include_val_in_train=True)
    write_yaml(out2, "CLEAN dataset - all data in train for final submission")
    print(f"yolo_clean_no_val: {train2} train, {val2} dummy val")

    # Gap analysis
    cat_names = {c["id"]: c["name"] for c in json.loads(Path(COCO_ANN).read_text())["categories"]}
    counts = Counter()
    for f in (out1 / "train" / "labels").glob("*.txt"):
        for line in open(f):
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1

    vals = sorted(counts.values()) if counts else [0]
    print(f"\nTrain gap analysis:")
    print(f"  Categories: {len(counts)}/356")
    print(f"  Per-cat: min={vals[0]}, median={vals[len(vals)//2]}, max={vals[-1]}")
    print(f"  Total annotations: {sum(vals)}")

    weak = [(cid, counts.get(cid, 0)) for cid in range(356) if counts.get(cid, 0) < 20]
    if weak:
        print(f"  Weak (<20): {len(weak)} categories")
        for cid, cnt in sorted(weak, key=lambda x: x[1]):
            print(f"    {cid}: {cat_names.get(cid, '?')} ({cnt})")


if __name__ == "__main__":
    main()
