"""
Fetch product images from Kassalapp web search results.

Discovery: Product images are hosted on bilder.ngdata.no (NorgesGruppen CDN).
URL pattern: https://bilder.ngdata.no/{EAN}/{store}/large.jpg
Stores: kmh, meny, spar, joker, etc.

Strategy:
1. Search each category name on kassal.app/varer?sok=QUERY
2. Parse the page to find bilder.ngdata.no image URLs
3. Download the images

This uses httpx to fetch the Kassalapp search pages (server-rendered HTML).
"""
import json
import re
import time
from collections import Counter
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = DATA_DIR / "scraped_products"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

KASSAL_SEARCH = "https://kassal.app/varer"
NGDATA_PATTERN = re.compile(r'https://bilder\.ngdata\.no/[^"\'>\s]+\.jpg')
RATE_LIMIT = 2.0  # Be respectful: 2 seconds between requests


def load_categories():
    with open(COCO_ANN) as f:
        coco = json.load(f)
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    ann_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    return categories, ann_counts


def search_kassal_web(client: httpx.Client, query: str) -> list[str]:
    """Search Kassalapp web and extract product image URLs."""
    try:
        resp = client.get(KASSAL_SEARCH, params={"sok": query}, timeout=20)
        if resp.status_code != 200:
            return []
        # Extract all bilder.ngdata.no URLs
        urls = list(set(NGDATA_PATTERN.findall(resp.text)))
        return urls
    except Exception as e:
        print(f"    Error: {e}")
        return []


def download_image(client: httpx.Client, url: str, save_path: Path) -> bool:
    if save_path.exists():
        return True
    try:
        resp = client.get(url, timeout=15)
        if resp.status_code != 200 or len(resp.content) < 1000:
            return False
        save_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-categories", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    categories, ann_counts = load_categories()
    # Sort: rarest first
    sorted_cats = sorted(categories.items(), key=lambda x: ann_counts.get(x[0], 0))

    print(f"Categories: {len(categories)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if METADATA_FILE.exists():
        metadata = json.loads(METADATA_FILE.read_text())

    client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
        follow_redirects=True,
    )

    total_found = 0
    total_searched = 0
    total_images = 0

    cats_to_process = sorted_cats
    if args.max_categories > 0:
        cats_to_process = cats_to_process[:args.max_categories]

    for cat_id, cat_name in cats_to_process:
        if args.skip_existing and str(cat_id) in metadata and metadata[str(cat_id)].get("images"):
            continue

        if cat_name.strip() == "" or cat_name == "unknown_product":
            continue

        cat_dir = OUTPUT_DIR / str(cat_id)
        cat_dir.mkdir(parents=True, exist_ok=True)

        ann_count = ann_counts.get(cat_id, 0)
        print(f"[{cat_id}] {cat_name} (anns: {ann_count})")

        # Search Kassalapp web
        image_urls = search_kassal_web(client, cat_name)
        time.sleep(RATE_LIMIT)

        images_found = []
        for j, url in enumerate(image_urls[:5]):  # Max 5 images per category
            img_name = f"ng_{j}.jpg"
            img_path = cat_dir / img_name
            if download_image(client, url, img_path):
                # Extract EAN from URL
                ean_match = re.search(r'/(\d+)/', url)
                ean = ean_match.group(1) if ean_match else ""
                images_found.append({
                    "source": "ngdata",
                    "url": url,
                    "filename": img_name,
                    "ean": ean,
                })
                total_images += 1
            time.sleep(0.3)

        if images_found:
            total_found += 1
            print(f"  -> {len(images_found)} images")
        else:
            print(f"  -> no images found")

        metadata[str(cat_id)] = {
            "category_name": cat_name,
            "annotation_count": ann_count,
            "images": images_found,
        }

        total_searched += 1

        if total_searched % 20 == 0:
            METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
            print(f"\n--- {total_searched} searched, {total_found} with images, {total_images} total images ---\n")

    client.close()
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\n=== COMPLETE ===")
    print(f"Searched: {total_searched}")
    print(f"With images: {total_found}")
    print(f"Total images: {total_images}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
