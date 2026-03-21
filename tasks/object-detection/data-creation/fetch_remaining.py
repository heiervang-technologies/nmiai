"""Fetch product images for the 45 remaining categories from Kassalapp."""
import json
import re
import time
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent / "data"
SCRAPED_DIR = DATA_DIR / "scraped_products"
NGDATA_RE = re.compile(r'https://bilder\.ngdata\.no/[^"\x27>\s]+\.jpg')

# Also try cdcimg.coop.no and api.vetduat.no patterns
ALT_RE = re.compile(r'https://(?:cdcimg\.coop\.no|api\.vetduat\.no)/[^"\x27>\s]+\.(?:jpg|png)')

with open(DATA_DIR / "coco_dataset/train/annotations.json") as f:
    cat_names = {c["id"]: c["name"] for c in json.load(f)["categories"]}

# Missing categories
missing = []
for cid in range(356):
    d = SCRAPED_DIR / str(cid)
    if not d.exists() or not any(d.glob("*.jpg")):
        if cat_names.get(cid, "").strip() and cat_names.get(cid) != "unknown_product":
            missing.append(cid)

print(f"Missing: {len(missing)} categories")

client = httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True, timeout=20)

found = 0
for cid in missing:
    name = cat_names[cid]
    out_dir = SCRAPED_DIR / str(cid)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple search variations
    search_terms = [name]
    # Simplify: remove weight, try just brand + product type
    import re as _re
    simple = _re.sub(r'\d+\s*(g|kg|ml|l|cl|stk|pos|pk)\b', '', name, flags=_re.IGNORECASE).strip()
    if simple != name:
        search_terms.append(simple)

    got_image = False
    for term in search_terms:
        resp = client.get("https://kassal.app/varer", params={"sok": term})
        urls = list(set(NGDATA_RE.findall(resp.text)))
        if not urls:
            urls = list(set(ALT_RE.findall(resp.text)))

        for j, url in enumerate(urls[:3]):
            ext = ".jpg" if url.endswith(".jpg") else ".png"
            img_path = out_dir / f"ng_{j}{ext}"
            try:
                r = client.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 500:
                    img_path.write_bytes(r.content)
                    got_image = True
            except Exception:
                pass

        time.sleep(1.5)
        if got_image:
            break

    if got_image:
        found += 1
        print(f"  [{cid}] {name[:50]} -> FOUND")
    else:
        print(f"  [{cid}] {name[:50]} -> MISS")

client.close()
print(f"\nFound images for {found}/{len(missing)} remaining categories")

# Final coverage
total_covered = sum(1 for cid in range(356) if (SCRAPED_DIR / str(cid)).exists() and any((SCRAPED_DIR / str(cid)).glob("*.*")))
print(f"Total coverage: {total_covered}/356 ({total_covered/356*100:.0f}%)")
