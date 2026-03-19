"""
Convert npz embeddings to PyTorch .pth format for inference.
Run this after build_embeddings.py completes.

Output: data/ref_embeddings.pth containing:
- reference_embeddings: (N_products, 384) tensor
- reference_barcodes: list of barcode strings
- category_embeddings: (N_categories, 384) tensor
- category_ids: list of int category IDs
- category_names: dict mapping category_id -> name
- barcode_to_category: dict mapping barcode -> category_id
- model_name: str
"""
import json
from pathlib import Path

import numpy as np
import torch

OUTPUT_DIR = Path(__file__).parent / "outputs"
DATA_DIR = Path(__file__).parent / "data"


def main():
    # Load npz files
    ref_data = np.load(OUTPUT_DIR / "reference_embeddings.npz", allow_pickle=True)
    cat_data = np.load(OUTPUT_DIR / "category_embeddings.npz", allow_pickle=True)

    # Load mapping
    with open(OUTPUT_DIR / "barcode_category_mapping.json") as f:
        mapping = json.load(f)
    with open(OUTPUT_DIR / "category_index.json") as f:
        cat_names = json.load(f)

    # Build barcode -> category_id simple mapping
    barcode_to_cat = {bc: info["category_id"] for bc, info in mapping.items()}

    # Convert to torch
    ref_embeddings = torch.from_numpy(ref_data["embeddings"]).float()
    ref_barcodes = ref_data["barcodes"].tolist()
    cat_embeddings = torch.from_numpy(cat_data["embeddings"]).float()
    cat_ids = cat_data["category_ids"].tolist()

    pth_data = {
        "reference_embeddings": ref_embeddings,
        "reference_barcodes": ref_barcodes,
        "category_embeddings": cat_embeddings,
        "category_ids": cat_ids,
        "category_names": cat_names,
        "barcode_to_category": barcode_to_cat,
        "model_name": "vit_small_patch14_dinov2.lvd142m",
        "embedding_dim": 384,
    }

    out_path = DATA_DIR / "ref_embeddings.pth"
    torch.save(pth_data, out_path)

    print(f"Saved to {out_path}")
    print(f"  Reference embeddings: {ref_embeddings.shape}")
    print(f"  Category embeddings: {cat_embeddings.shape}")
    print(f"  Barcodes: {len(ref_barcodes)}")
    print(f"  Categories: {len(cat_ids)}")
    print(f"  File size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
