"""
Scrape product images from oda.com for categories with few reference images.
Uses httpx to download images from URLs extracted via browser.
Run after extracting URLs with Playwright.
"""
import json
import re
import time
from pathlib import Path
import httpx

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = DATA_DIR / "scraped_products"

# Map of category_id -> list of oda image URLs (populated by Playwright scraping)
ODA_URLS_FILE = Path(__file__).parent / "oda_image_urls.json"


def download_oda_images():
    if not ODA_URLS_FILE.exists():
        print(f"No {ODA_URLS_FILE} found. Run Playwright extraction first.")
        return

    urls_data = json.loads(ODA_URLS_FILE.read_text())
    client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
        follow_redirects=True, timeout=15
    )

    total = 0
    for cat_id_str, urls in urls_data.items():
        d = OUTPUT_DIR / cat_id_str
        d.mkdir(parents=True, exist_ok=True)
        existing = len(list(d.glob("*.jpg")))

        for j, url in enumerate(urls):
            fname = f"oda_{j}.jpg"
            fpath = d / fname
            if fpath.exists():
                continue
            try:
                resp = client.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.content) > 2000:
                    fpath.write_bytes(resp.content)
                    total += 1
            except Exception:
                pass
            time.sleep(0.2)

    client.close()
    print(f"Downloaded {total} new images from oda.com")


if __name__ == "__main__":
    download_oda_images()
