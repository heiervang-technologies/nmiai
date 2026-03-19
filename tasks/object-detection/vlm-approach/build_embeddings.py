"""
Build DINOv2 reference embeddings from product reference images.

Reads product images organized by barcode, maps to COCO category IDs,
and saves embeddings for use in the hybrid inference pipeline.

Usage: python build_embeddings.py
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image


DATA_ROOT = Path(__file__).parent.parent / "data-creation" / "data"
COCO_ANNOTATIONS = DATA_ROOT / "coco_dataset" / "train" / "annotations.json"
PRODUCT_IMAGES = DATA_ROOT / "product_images"
METADATA = PRODUCT_IMAGES / "metadata.json"
OUTPUT_PATH = Path(__file__).parent / "ref_embeddings.pth"

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
BATCH_SIZE = 32


def build_barcode_to_category_map():
    """Build mapping from product barcode to COCO category_id."""
    with open(COCO_ANNOTATIONS) as f:
        coco = json.load(f)
    with open(METADATA) as f:
        meta = json.load(f)

    # COCO category name -> ID
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}

    # Product barcode -> category_id (via matching names)
    barcode_to_cat = {}
    unmatched = []

    for product in meta["products"]:
        name = product["product_name"]
        barcode = product["product_code"]

        if name in name_to_id:
            barcode_to_cat[barcode] = name_to_id[name]
        else:
            unmatched.append((barcode, name))

    print(f"Matched {len(barcode_to_cat)} products to category IDs")
    if unmatched:
        print(f"Unmatched: {len(unmatched)}")
        for bc, name in unmatched[:5]:
            print(f"  {bc}: {name}")

    return barcode_to_cat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model.eval().to(device)

    data_config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_config, is_training=False)

    # Build barcode -> category mapping
    barcode_to_cat = build_barcode_to_category_map()

    # Embed all reference images
    embeddings = {}  # category_id -> list of embedding tensors
    total_images = 0

    product_dirs = sorted(PRODUCT_IMAGES.iterdir())

    for product_dir in product_dirs:
        if not product_dir.is_dir():
            continue

        barcode = product_dir.name
        if barcode not in barcode_to_cat:
            continue

        cat_id = barcode_to_cat[barcode]
        image_files = sorted(product_dir.glob("*.jpg")) + sorted(product_dir.glob("*.png"))

        if not image_files:
            continue

        # Load and transform all angles
        batch_tensors = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                batch_tensors.append(tensor)
                total_images += 1
            except Exception as e:
                print(f"  Warning: failed to load {img_path}: {e}")

        if not batch_tensors:
            continue

        # Embed batch
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            embs = model(batch)
            embs = F.normalize(embs, dim=-1)

        embeddings[cat_id] = embs.cpu()

    print(f"\nEmbedded {total_images} images across {len(embeddings)} categories")

    # Save
    output = {
        "embeddings": embeddings,
        "model_name": MODEL_NAME,
        "embed_dim": model.num_features,
        "num_categories": len(embeddings),
        "total_images": total_images,
    }

    torch.save(output, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    # Also save the DINOv2 weights for submission
    weights_path = Path(__file__).parent / "dinov2_vits14.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved DINOv2 weights to {weights_path} ({weights_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
