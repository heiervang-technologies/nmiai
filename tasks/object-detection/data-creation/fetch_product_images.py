"""
Fetch product images from free APIs for our 356 grocery categories.

Sources:
1. Open Food Facts (free, no auth, has Norwegian products)
2. Kassalapp API (free tier, needs Bearer token, 60 req/min)

Priority: 84 categories with 1-5 annotations first.
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

OFF_BASE = "https://world.openfoodfacts.org/cgi/search.pl"
KASSAL_BASE = "https://kassal.app/api/v1/products"

RATE_LIMIT = 1.5  # seconds between requests
HTTP_TIMEOUT = 30.0


def load_categories():
    with open(COCO_ANN) as f:
        coco = json.load(f)
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    ann_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    return categories, ann_counts


def extract_search_terms(name: str) -> list[str]:
    """Extract search terms from Norwegian product names."""
    # Remove weight/quantity
    clean = re.sub(r'\d+\s*(g|kg|ml|l|cl|dl|stk|pos|kapsler|pcs)\b', '', name, flags=re.IGNORECASE)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Try brand-first if last word looks like a brand
    words = clean.split()
    if len(words) >= 2:
        # Try reversed (brand first, then product)
        return [clean, " ".join(reversed(words))]
    return [clean]


def search_off(client: httpx.Client, query: str, max_results: int = 5) -> list[dict]:
    """Search Open Food Facts."""
    try:
        resp = client.get(OFF_BASE, params={
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": max_results,
            "fields": "product_name,image_front_url,image_url,code,brands",
        }, timeout=HTTP_TIMEOUT)
        data = resp.json()
        return data.get("products", [])
    except Exception as e:
        print(f"    OFF error: {e}")
        return []


def search_kassal(client: httpx.Client, query: str, api_key: str) -> list[dict]:
    """Search Kassalapp API."""
    try:
        resp = client.get(KASSAL_BASE, params={
            "search": query,
            "size": 5,
        }, headers={
            "Authorization": f"Bearer {api_key}",
        }, timeout=HTTP_TIMEOUT)
        if resp.status_code == 401:
            return []
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        print(f"    Kassal error: {e}")
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
    parser.add_argument("--kassalapp-key", default="", help="Kassalapp API key")
    parser.add_argument("--max-categories", type=int, default=0, help="Limit categories to process (0=all)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    categories, ann_counts = load_categories()
    sorted_cats = sorted(categories.items(), key=lambda x: ann_counts.get(x[0], 0))

    print(f"Categories: {len(categories)}")
    rare = sum(1 for cid in categories if ann_counts.get(cid, 0) <= 5)
    print(f"Rare (<=5 annotations): {rare}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if METADATA_FILE.exists():
        metadata = json.loads(METADATA_FILE.read_text())

    client = httpx.Client(headers={"User-Agent": "NMiAI2026-Research/1.0"}, follow_redirects=True)

    total_found = 0
    total_searched = 0
    kassalapp_key = args.kassalapp_key

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

        terms = extract_search_terms(cat_name)
        images_found = []

        ann_count = ann_counts.get(cat_id, 0)
        print(f"\n[{cat_id}] {cat_name} (anns: {ann_count})")

        # 1. Open Food Facts
        for term in terms[:1]:
            products = search_off(client, term)
            time.sleep(RATE_LIMIT)

            for j, p in enumerate(products):
                img_url = p.get("image_front_url") or p.get("image_url")
                if not img_url:
                    continue
                img_name = f"off_{j}.jpg"
                if download_image(client, img_url, cat_dir / img_name):
                    images_found.append({
                        "source": "openfoodfacts",
                        "url": img_url,
                        "filename": img_name,
                        "product_name": p.get("product_name", ""),
                        "barcode": p.get("code", ""),
                        "brands": p.get("brands", ""),
                    })
                    print(f"    OFF: {p.get('product_name', '?')[:50]}")
                time.sleep(0.3)

            if products:
                break  # Got results, skip alternative terms

        # 2. Kassalapp (if key provided)
        if kassalapp_key and len(images_found) == 0:
            for term in terms[:1]:
                products = search_kassal(client, term, kassalapp_key)
                time.sleep(RATE_LIMIT)

                for j, p in enumerate(products):
                    img_url = p.get("image")
                    if not img_url:
                        continue
                    img_name = f"kassal_{j}.jpg"
                    if download_image(client, img_url, cat_dir / img_name):
                        images_found.append({
                            "source": "kassalapp",
                            "url": img_url,
                            "filename": img_name,
                            "product_name": p.get("name", ""),
                            "ean": p.get("ean", ""),
                        })
                        print(f"    Kassal: {p.get('name', '?')[:50]}")
                    time.sleep(0.3)

        metadata[str(cat_id)] = {
            "category_name": cat_name,
            "annotation_count": ann_count,
            "images": images_found,
        }

        if images_found:
            total_found += 1
        total_searched += 1

        if total_searched % 10 == 0:
            METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
            print(f"\n--- {total_searched} searched, {total_found} with images ---\n")

    client.close()
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\n=== COMPLETE ===")
    print(f"Searched: {total_searched}")
    print(f"With images: {total_found}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
