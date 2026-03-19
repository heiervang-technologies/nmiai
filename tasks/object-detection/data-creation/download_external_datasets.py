"""
Download external datasets for supplementary training.
Priority: SKU-110K (via ultralytics auto-download) and SKUs on Shelves PL (HuggingFace).
"""
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "external"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_sku110k():
    """
    SKU-110K is automatically downloaded by ultralytics when you train with it.
    We just need to reference the ultralytics config.
    But we can also manually download for analysis.
    """
    print("=== SKU-110K ===")
    print("SKU-110K can be auto-downloaded via ultralytics:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolov8n.pt')")
    print("  model.train(data='SKU-110K.yaml', ...)")
    print()
    print("For manual download, use Kaggle:")
    print("  kaggle datasets download -d thedatasith/sku110k-annotations")
    print()

    # Try ultralytics auto-download config
    sku_yaml = DATA_DIR / "sku110k.yaml"
    if not sku_yaml.exists():
        sku_yaml.write_text("""# SKU-110K Dataset (auto-downloaded by ultralytics)
# https://docs.ultralytics.com/datasets/detect/sku-110k/
path: ../datasets/SKU-110K
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: 'object'
""")
        print(f"Wrote {sku_yaml}")


def download_skus_on_shelves():
    """Download SKUs on Shelves PL from HuggingFace."""
    print("\n=== SKUs on Shelves PL ===")
    out_dir = DATA_DIR / "skus_on_shelves_pl"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Already exists at {out_dir}")
        return

    try:
        # Try using huggingface_hub
        subprocess.run([
            sys.executable, "-c",
            f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='shelfwise-by-form/SKUs_on_shelves_PL',
    repo_type='dataset',
    local_dir='{out_dir}',
    max_workers=4,
)
print('Download complete')
"""
        ], check=True)
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Install huggingface_hub: uv add huggingface_hub")


def download_grocery_store_dataset():
    """Download Swedish Grocery Store Dataset from GitHub."""
    print("\n=== Swedish Grocery Store Dataset ===")
    out_dir = DATA_DIR / "grocery_store_dataset"
    if out_dir.exists():
        print(f"Already exists at {out_dir}")
        return

    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/marcusklasson/GroceryStoreDataset.git",
        str(out_dir),
    ])


if __name__ == "__main__":
    download_sku110k()
    download_skus_on_shelves()
    # download_grocery_store_dataset()  # Classification only, lower priority
