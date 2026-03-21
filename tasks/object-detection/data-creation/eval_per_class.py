"""
Run per-class evaluation on val set using best model.
Identifies which categories have low AP so we can target data improvements.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
BEST_MODEL = Path("/home/me/ht/nmiai/runs/detect/yolov8x_v4_12802/weights/best.pt")
VAL_DIR = DATA_DIR / "yolo_augmented_v4" / "val"


def main():
    with open(COCO_ANN) as f:
        coco = json.load(f)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    print("Loading model...")
    model = YOLO(str(BEST_MODEL))

    print("Running validation...")
    results = model.val(
        data=str(DATA_DIR / "yolo_augmented_v4" / "dataset.yaml"),
        imgsz=1280,
        batch=4,
        device=0,
        verbose=False,
    )

    # Per-class AP50
    ap50_per_class = results.box.ap50  # numpy array, shape (nc,)
    ap_per_class = results.box.ap  # mAP50-95

    print(f"\nOverall mAP50: {results.box.map50:.4f}")
    print(f"Overall mAP50-95: {results.box.map:.4f}")

    # Sort by AP50 (worst first)
    class_aps = []
    for i in range(len(ap50_per_class)):
        class_aps.append({
            "category_id": i,
            "name": cat_names.get(i, f"class_{i}"),
            "ap50": float(ap50_per_class[i]),
            "ap": float(ap_per_class[i]),
        })

    class_aps.sort(key=lambda x: x["ap50"])

    print(f"\n{'='*80}")
    print("WORST 30 CATEGORIES (lowest AP50)")
    print(f"{'='*80}")
    for c in class_aps[:30]:
        print(f"  cat {c['category_id']:3d} AP50={c['ap50']:.3f} AP={c['ap']:.3f} | {c['name']}")

    print(f"\n{'='*80}")
    print("BEST 20 CATEGORIES (highest AP50)")
    print(f"{'='*80}")
    for c in class_aps[-20:]:
        print(f"  cat {c['category_id']:3d} AP50={c['ap50']:.3f} AP={c['ap']:.3f} | {c['name']}")

    # Save full per-class results
    out_path = Path(__file__).parent / "per_class_ap.json"
    with open(out_path, "w") as f:
        json.dump(class_aps, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_path}")

    # Identify categories that need the most help
    zero_ap = [c for c in class_aps if c["ap50"] == 0]
    low_ap = [c for c in class_aps if 0 < c["ap50"] < 0.3]
    print(f"\nCategories with AP50=0: {len(zero_ap)}")
    print(f"Categories with 0<AP50<0.3: {len(low_ap)}")
    print(f"Categories with AP50>=0.5: {sum(1 for c in class_aps if c['ap50'] >= 0.5)}")
    print(f"Categories with AP50>=0.8: {sum(1 for c in class_aps if c['ap50'] >= 0.8)}")


if __name__ == "__main__":
    main()
