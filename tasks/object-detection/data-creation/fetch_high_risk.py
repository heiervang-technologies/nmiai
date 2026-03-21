"""
Targeted scraper for high-risk categories.
Tries multiple angles:
1. ngdata.no store variants (kmh, meny, spar, joker, coop, etc.)
2. Kassalapp web search with refined queries
3. Open Food Facts
4. colonigrossisten.no / handlehjelpen
"""
import json
import re
import time
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = DATA_DIR / "scraped_products"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

NGDATA_STORES = ["meny", "kmh", "spar", "joker", "coop", "extra"]
NGDATA_SIZES = ["large", "medium"]
KASSAL_SEARCH = "https://kassal.app/varer"
NGDATA_PATTERN = re.compile(r'https://bilder\.ngdata\.no/[^"\'>\s]+\.jpg')
OFF_BASE = "https://world.openfoodfacts.org/cgi/search.pl"

# High-risk categories: (cat_id, product_name)
HIGH_RISK = [
    (114, "CLEAN MATCHA GREEN TE ØKOL 20POS PUKKA"),
    (183, "KNEKKEBRØD GODT FOR DEG OST 190G SIGDAL"),
    (156, "GRANOLA CRAZELNUT 500G START!"),
    (91, "SANDWICH PESTO 37G WASA"),
    (119, "SOFT FLORA STEKE&BAKE 500G"),
    (263, "GRANOLA RASPBERRY 500G START!"),
    (48, "SMØREMYK 600G ELDORADO"),
    (161, "HAVRERINGER 250G SYNNØVE FINDEN"),
    (51, "Tørresvik Gårdsegg 6stk"),
    (17, "KNEKKEBRØD GL.FRI 150G BRISK"),
    (180, "MÜSLI PAPAYA GLUTENFRI 350G AXA"),
    (350, "BLÅ JAVA HELE BØNNER 340G COTW"),
    (190, "EARL GREY TEA ØKOLOGISK 15POS JACOBS"),
    (77, "BAKEKAKAO 250G REGIA"),
    (28, "LADY GREY TE 200G TWININGS"),
    (104, "SMØR USALTET 250G TINE"),
    (125, "SJOKOLADEDRIKK 512G RETT I KOPPEN"),
    (133, "ZOEGAS KAFFE SKÅNEROST 450G"),
    (313, "FLOTT MATFETT 500G"),
    (353, "EXCELSO COLOMBIA FILTERMALT 200G JACOBS"),
    (165, "COTW COLOMBIA EXCELSO KAFFEKAPSEL 10STK"),
    (16, "CAPPUCCINO 8KAPSLER DOLCE GUSTO"),
    (179, "SJOKOLADEDRIKK 10X32G RETT I KOPPEN"),
    (74, "NESCAFE BRASERO REFILL 180G"),
    (112, "ESPRESSO ITALIAN HELE BØNNER 500G JACOBS"),
    (124, "O'BOY MINDRE SUKKER 500G POSE FREIA"),
    (252, "SIDAMO ETIOPIA HELE BØNNER 340G COTW"),
    (128, "SMØREMYK LETT 400G ELDORADO"),
    (101, "SANDWICH CHEESE HERBS 30G WASA"),
    (226, "LEKSANDS TREKANT HAVRE 200G"),
    (127, "FRIELE FROKOST HEL 500G"),
    (312, "Tørresvik Gård Kvalitetsegg 10stk"),
    (113, "ERTERFLATBRØD CRISP 210G"),
    (273, "VITA HJERTEGO LETTMARGARIN 370G"),
    (303, "KNEKKEBRØD HAVRE 220G SIGDAL"),
    (110, "BREMYKT 250G"),
    (247, "MOUNT KENYA FILTERMALT 200G JACOBS"),
    (224, "SOFT FLORA LETT 540G"),
    (168, "BRELETT 540G"),
    (13, "GRISSINI 125G GRANFORNO"),
    (118, "BLÅ JAVA HELE BØNNER 500G JACOBS UTVALGT"),
    (249, "COTW BREAKFAST BLEND KAFFEKAPSEL 10STK"),
]

# Known EANs from previous scraping
KNOWN_EANS = {
    114: ["5060229014436"],
    183: ["7071848108169"],
    156: ["7310130013416"],
    119: ["7036110009704"],
    263: ["7310130013423"],
    48: ["7035620044830"],
    161: ["7071319034003"],
    17: ["7090022850229", "7090022850304"],
    180: ["7310130010507"],
    350: ["7040913336486"],
    190: ["7035620008542"],
    28: ["70177087807"],
    104: ["7038010011702"],
    125: ["7039010563864", "7039010563833"],
    133: ["7310731103011", "7310731101734"],
    353: ["7040913336905"],
    165: ["7040913336523"],
    16: ["7613036305648"],
    179: ["7039010563857"],
    112: ["7040913337001"],
    124: ["7622201150686"],
    252: ["7040913336165"],
    128: ["7035620029509"],
    101: ["7300400127394"],
    226: ["731208004193"],
    113: ["7035380000602"],
    273: ["7036110010595"],
    303: ["7071848005666"],
    247: ["7040913336912"],
    224: ["7036110011943"],
    168: ["7036110011479"],  # brelett common EAN
    118: ["7040913336950"],
    249: ["7040913336561"],
}


def try_ngdata_variants(client: httpx.Client, ean: str, cat_dir: Path, existing_count: int) -> list[dict]:
    """Try different store/size combinations on ngdata.no."""
    found = []
    seen_sizes = set()  # deduplicate by content length
    idx = existing_count
    for store in NGDATA_STORES:
        for size in NGDATA_SIZES:
            url = f"https://bilder.ngdata.no/{ean}/{store}/{size}.jpg"
            try:
                resp = client.get(url, timeout=10)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    content_size = len(resp.content)
                    if content_size in seen_sizes:
                        continue  # Skip duplicate images
                    seen_sizes.add(content_size)
                    fname = f"ng_{store}_{size}_{idx}.jpg"
                    fpath = cat_dir / fname
                    if not fpath.exists():
                        fpath.write_bytes(resp.content)
                        found.append({
                            "source": "ngdata",
                            "url": url,
                            "filename": fname,
                            "ean": ean,
                            "store": store,
                        })
                        idx += 1
                        print(f"    ngdata: {store}/{size} OK ({content_size} bytes)")
                time.sleep(0.3)
            except Exception:
                continue
    return found


def search_kassal_web(client: httpx.Client, query: str) -> list[str]:
    """Search Kassalapp web and extract product image URLs."""
    try:
        resp = client.get(KASSAL_SEARCH, params={"sok": query}, timeout=20)
        if resp.status_code != 200:
            return []
        urls = list(set(NGDATA_PATTERN.findall(resp.text)))
        return urls
    except Exception as e:
        print(f"    Kassal error: {e}")
        return []


def search_off(client: httpx.Client, query: str) -> list[dict]:
    """Search Open Food Facts."""
    try:
        resp = client.get(OFF_BASE, params={
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 5,
            "fields": "product_name,image_front_url,image_url,code,brands",
        }, timeout=30)
        return resp.json().get("products", [])
    except Exception as e:
        print(f"    OFF error: {e}")
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


def generate_search_variants(name: str) -> list[str]:
    """Generate multiple search queries from a product name."""
    variants = [name]
    # Remove weight/quantity
    clean = re.sub(r'\d+\s*(g|kg|ml|l|cl|dl|stk|pos|kapsler|pcs)\b', '', name, flags=re.IGNORECASE)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if clean != name:
        variants.append(clean)
    # Just brand + product type
    words = clean.split()
    if len(words) > 2:
        variants.append(" ".join(words[:3]))
        variants.append(" ".join(words[-2:]) + " " + words[0])
    return variants


def main():
    metadata = {}
    if METADATA_FILE.exists():
        metadata = json.loads(METADATA_FILE.read_text())

    client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
        follow_redirects=True,
    )

    total_new = 0

    for cat_id, cat_name in HIGH_RISK:
        cat_dir = OUTPUT_DIR / str(cat_id)
        cat_dir.mkdir(parents=True, exist_ok=True)

        existing_files = list(cat_dir.glob("*.jpg"))
        existing_count = len(existing_files)
        print(f"\n[{cat_id}] {cat_name} (existing: {existing_count})")

        new_images = []
        existing_urls = set()
        if str(cat_id) in metadata:
            for img in metadata[str(cat_id)].get("images", []):
                existing_urls.add(img.get("url", ""))

        # Strategy 1: ngdata store variants for known EANs
        eans = KNOWN_EANS.get(cat_id, [])
        for ean in eans:
            imgs = try_ngdata_variants(client, ean, cat_dir, existing_count + len(new_images))
            for img in imgs:
                if img["url"] not in existing_urls:
                    new_images.append(img)
                    existing_urls.add(img["url"])

        # Strategy 2: Search variants for OFF
        search_variants = generate_search_variants(cat_name)

        # Strategy 3: Open Food Facts (if still few images)
        if existing_count + len(new_images) < 3:
            for query in search_variants[:2]:
                products = search_off(client, query)
                time.sleep(1.5)
                for j, p in enumerate(products):
                    img_url = p.get("image_front_url") or p.get("image_url")
                    if not img_url or img_url in existing_urls:
                        continue
                    idx = existing_count + len(new_images)
                    fname = f"off_{idx}.jpg"
                    fpath = cat_dir / fname
                    if download_image(client, img_url, fpath):
                        new_images.append({
                            "source": "openfoodfacts",
                            "url": img_url,
                            "filename": fname,
                            "product_name": p.get("product_name", ""),
                            "barcode": p.get("code", ""),
                        })
                        existing_urls.add(img_url)
                        print(f"    OFF: {p.get('product_name', '?')[:50]}")
                    time.sleep(0.3)
                if products:
                    break

        # Strategy 4: Try EAN lookup on Open Food Facts directly
        if existing_count + len(new_images) < 3:
            for ean in eans:
                try:
                    resp = client.get(f"https://world.openfoodfacts.org/api/v2/product/{ean}.json",
                                      params={"fields": "image_front_url,image_url,images"},
                                      timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        product = data.get("product", {})
                        # Get all available image URLs
                        images_dict = product.get("images", {})
                        for img_key in ["front", "front_no", "1", "2", "3", "ingredients", "nutrition"]:
                            if img_key in images_dict:
                                img_info = images_dict[img_key]
                                rev = img_info.get("rev", "")
                                img_url = f"https://images.openfoodfacts.org/images/products/{'/'.join([ean[i:i+3] for i in range(0, len(ean), 3)])}/{img_key}.{rev}.400.jpg"
                                # Simpler: use known URL pattern
                                pass
                        # Try direct image URLs
                        for field in ["image_front_url", "image_url"]:
                            img_url = product.get(field)
                            if img_url and img_url not in existing_urls:
                                idx = existing_count + len(new_images)
                                fname = f"off_ean_{idx}.jpg"
                                fpath = cat_dir / fname
                                if download_image(client, img_url, fpath):
                                    new_images.append({
                                        "source": "off_ean",
                                        "url": img_url,
                                        "filename": fname,
                                        "ean": ean,
                                    })
                                    existing_urls.add(img_url)
                                    print(f"    OFF EAN: {ean}")
                    time.sleep(1.0)
                except Exception:
                    pass

        # Update metadata
        if str(cat_id) not in metadata:
            metadata[str(cat_id)] = {"category_name": cat_name, "images": []}
        metadata[str(cat_id)]["images"].extend(new_images)

        if new_images:
            total_new += len(new_images)
            print(f"  => +{len(new_images)} new (total: {existing_count + len(new_images)})")
        else:
            print(f"  => no new images found")

    client.close()
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\n=== COMPLETE: {total_new} new images across {len(HIGH_RISK)} categories ===")


if __name__ == "__main__":
    main()
