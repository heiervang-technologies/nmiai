"""
Build DINOv2 embeddings for:
1. All product reference images (by barcode folder)
2. Cropped training annotations (by category_id)

Then use nearest-neighbor matching to map barcodes -> category_ids.
Save the reference embedding index for inference-time classification boost.
"""
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
PRODUCT_DIR = DATA_DIR / "product_images"
ANNOTATIONS = COCO_DIR / "annotations.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use DINOv2 ViT-S/14 - same as what will be available at inference (via timm)
MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
BATCH_SIZE = 32


def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(DEVICE).eval()
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config)
    return model, transform


@torch.no_grad()
def embed_images(model, transform, image_paths: list[Path]) -> np.ndarray:
    """Embed a list of images, return (N, D) numpy array."""
    embeddings = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch = []
        valid_indices = []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                batch.append(transform(img))
                valid_indices.append(j)
            except Exception as e:
                print(f"  Warning: failed to load {p}: {e}")
        if not batch:
            continue
        batch_tensor = torch.stack(batch).to(DEVICE)
        emb = model(batch_tensor)
        emb = F.normalize(emb, dim=-1)
        embeddings.append(emb.cpu().numpy())
    if not embeddings:
        return np.zeros((0, 384))  # ViT-S dim
    return np.concatenate(embeddings, axis=0)


def embed_reference_products(model, transform):
    """Embed all reference product images, averaged per product."""
    print("=== Embedding reference products ===")
    product_dirs = sorted([d for d in PRODUCT_DIR.iterdir() if d.is_dir()])

    barcodes = []
    product_embeddings = []

    for pdir in tqdm(product_dirs, desc="Products"):
        barcode = pdir.name
        image_paths = sorted(pdir.glob("*.jpg")) + sorted(pdir.glob("*.png"))
        if not image_paths:
            continue
        embs = embed_images(model, transform, image_paths)
        if len(embs) == 0:
            continue
        # Average embedding for this product
        avg_emb = embs.mean(axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        barcodes.append(barcode)
        product_embeddings.append(avg_emb)

    product_embeddings = np.stack(product_embeddings)
    print(f"Embedded {len(barcodes)} products, shape: {product_embeddings.shape}")
    return barcodes, product_embeddings


def embed_training_crops(model, transform):
    """Crop annotated boxes from training images and embed per category."""
    print("\n=== Embedding training crops (per category) ===")
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    # Group annotations by category
    from collections import defaultdict
    cat_anns = defaultdict(list)
    for ann in coco["annotations"]:
        cat_anns[ann["category_id"]].append(ann)

    # Group annotations by image for efficient loading
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    category_embeddings = {}
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Process each category
    for cat_id in tqdm(sorted(cat_anns.keys()), desc="Categories"):
        anns = cat_anns[cat_id]
        # Sample up to 20 crops per category to keep it fast
        sample = anns[:20] if len(anns) > 20 else anns

        crops = []
        for ann in sample:
            img_file = img_id_to_file.get(ann["image_id"])
            if not img_file:
                continue
            img_path = COCO_DIR / "images" / img_file
            try:
                img = Image.open(img_path).convert("RGB")
                x, y, w, h = ann["bbox"]
                # Add small padding
                pad = 5
                x1 = max(0, int(x) - pad)
                y1 = max(0, int(y) - pad)
                x2 = min(img.width, int(x + w) + pad)
                y2 = min(img.height, int(y + h) + pad)
                crop = img.crop((x1, y1, x2, y2))
                if crop.width < 10 or crop.height < 10:
                    continue
                crops.append(transform(crop))
            except Exception as e:
                continue

        if not crops:
            continue

        batch = torch.stack(crops).to(DEVICE)
        with torch.no_grad():
            embs = model(batch)
            embs = F.normalize(embs, dim=-1)

        avg_emb = embs.mean(dim=0).cpu().numpy()
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        category_embeddings[cat_id] = avg_emb

    print(f"Embedded {len(category_embeddings)} categories")
    return category_embeddings, cat_names


def build_mapping(barcodes, product_embeddings, category_embeddings, cat_names):
    """Match reference products to categories using cosine similarity."""
    print("\n=== Building barcode -> category mapping ===")

    cat_ids = sorted(category_embeddings.keys())
    cat_emb_matrix = np.stack([category_embeddings[cid] for cid in cat_ids])

    # Compute similarity matrix: (num_products, num_categories)
    sim_matrix = product_embeddings @ cat_emb_matrix.T

    mapping = {}
    confidences = {}
    for i, barcode in enumerate(barcodes):
        best_idx = np.argmax(sim_matrix[i])
        best_cat_id = cat_ids[best_idx]
        confidence = float(sim_matrix[i, best_idx])
        mapping[barcode] = {
            "category_id": best_cat_id,
            "category_name": cat_names.get(best_cat_id, "unknown"),
            "confidence": confidence,
        }
        confidences[barcode] = confidence

    # Stats
    confs = list(confidences.values())
    print(f"Mapped {len(mapping)} products")
    print(f"Confidence: min={min(confs):.3f}, max={max(confs):.3f}, "
          f"mean={np.mean(confs):.3f}, median={np.median(confs):.3f}")

    # High confidence (>0.5)
    high_conf = sum(1 for c in confs if c > 0.5)
    print(f"High confidence (>0.5): {high_conf}/{len(confs)}")

    return mapping


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, transform = load_model()
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    # 1. Embed reference products
    barcodes, product_embeddings = embed_reference_products(model, transform)

    # 2. Embed training crops
    category_embeddings, cat_names = embed_training_crops(model, transform)

    # 3. Build mapping
    mapping = build_mapping(barcodes, product_embeddings, category_embeddings, cat_names)

    # 4. Save everything
    # Reference embeddings (for inference)
    np.savez_compressed(
        OUTPUT_DIR / "reference_embeddings.npz",
        embeddings=product_embeddings,
        barcodes=np.array(barcodes),
    )

    # Category embeddings (for inference)
    cat_ids_arr = np.array(sorted(category_embeddings.keys()))
    cat_emb_arr = np.stack([category_embeddings[cid] for cid in cat_ids_arr])
    np.savez_compressed(
        OUTPUT_DIR / "category_embeddings.npz",
        embeddings=cat_emb_arr,
        category_ids=cat_ids_arr,
    )

    # Barcode-to-category mapping
    with open(OUTPUT_DIR / "barcode_category_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    # Category name index (for inference)
    with open(OUTPUT_DIR / "category_index.json", "w") as f:
        json.dump(cat_names, f, indent=2, ensure_ascii=False)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print(f"  - reference_embeddings.npz ({product_embeddings.nbytes / 1024:.0f} KB)")
    print(f"  - category_embeddings.npz ({cat_emb_arr.nbytes / 1024:.0f} KB)")
    print(f"  - barcode_category_mapping.json")
    print(f"  - category_index.json")


if __name__ == "__main__":
    main()
