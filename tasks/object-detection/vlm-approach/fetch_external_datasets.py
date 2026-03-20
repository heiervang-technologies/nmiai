"""
Download and prepare external grocery datasets for pre-training.

Datasets (by priority):
1. SKU-110K — 11.7k dense shelf images, 1.73M bboxes (HuggingFace)
2. Grocery Store Dataset — 5,125 images, 81 fine-grained classes (GitHub)
3. Freiburg Groceries — 5,000 images, 25 classes (GitHub)
4. RP2K — 500k images, 2000 products (need to find download)

Usage: uv run python fetch_external_datasets.py
"""

import functools
from pathlib import Path

print = functools.partial(print, flush=True)

DATA_DIR = Path(__file__).parent / "external_datasets"


def download_sku110k():
    """Download SKU-110K from HuggingFace."""
    out_dir = DATA_DIR / "sku110k"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"SKU-110K already downloaded at {out_dir}")
        return

    print("Downloading SKU-110K from HuggingFace...")
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("PrashantDixit0/SKU-110K", split="train")
    print(f"SKU-110K: {len(ds)} samples")
    print(f"Features: {ds.features}")

    # Save as arrow format for fast loading
    ds.save_to_disk(str(out_dir))
    print(f"Saved to {out_dir}")


def download_grocery_store():
    """Download Grocery Store Dataset from GitHub."""
    import subprocess
    out_dir = DATA_DIR / "grocery_store"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Grocery Store already downloaded at {out_dir}")
        return

    print("Cloning Grocery Store Dataset...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repo
    import shutil
    repo_dir = DATA_DIR / "GroceryStoreDataset"
    if not repo_dir.exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/marcusklasson/GroceryStoreDataset.git",
            str(repo_dir)
        ], check=True)

    print(f"Grocery Store Dataset cloned to {repo_dir}")
    # Check contents
    dataset_dir = repo_dir / "dataset"
    if dataset_dir.exists():
        n_images = sum(1 for _ in dataset_dir.rglob("*.jpg")) + sum(1 for _ in dataset_dir.rglob("*.png"))
        print(f"Found {n_images} images")


def download_freiburg():
    """Download Freiburg Groceries Dataset."""
    import subprocess
    out_dir = DATA_DIR / "freiburg"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Freiburg already downloaded at {out_dir}")
        return

    print("Cloning Freiburg Groceries...")
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = DATA_DIR / "freiburg_groceries"
    if not repo_dir.exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/PhilJd/freiburg_groceries_dataset.git",
            str(repo_dir)
        ], check=True)

    print(f"Freiburg cloned to {repo_dir}")
    # Run their download script if it exists
    dl_script = repo_dir / "download_dataset.py"
    if dl_script.exists():
        print("Running Freiburg download script...")
        subprocess.run(["python3", str(dl_script)], cwd=str(repo_dir))


def download_rpc():
    """Download RPC (Retail Product Checkout) dataset info."""
    print("RPC dataset: Check https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset")
    print("(Large dataset, manual download may be needed)")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"External datasets directory: {DATA_DIR}")
    print()

    # Priority order: most useful first
    print("=== 1. SKU-110K (dense shelf detection) ===")
    download_sku110k()
    print()

    print("=== 2. Grocery Store Dataset (fine-grained classification) ===")
    download_grocery_store()
    print()

    print("=== 3. Freiburg Groceries (basic classification) ===")
    download_freiburg()
    print()

    print("=== 4. RPC (retail checkout) ===")
    download_rpc()

    print("\nDone!")


if __name__ == "__main__":
    main()
