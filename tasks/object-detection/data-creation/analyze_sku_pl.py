"""
Analyze SKUs on Shelves PL dataset and attempt to map categories to our 356 NorgesGruppen categories.

Strategy for mapping:
1. Parse PL category names (likely Polish product names with brand/weight info)
2. Extract brand names, product types, weights from both datasets
3. Use fuzzy string matching on normalized names
4. For unmatched categories, try mapping by product type (e.g., all "mleko" -> milk products)
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).parent / "data"
PL_DIR = DATA_DIR / "external" / "skus_on_shelves_pl" / "extracted"
OUR_ANNOTATIONS = DATA_DIR / "coco_dataset" / "train" / "annotations.json"

# Common Norwegian-Polish product type mappings
NO_PL_TYPE_MAP = {
    # Norwegian -> Polish equivalents
    "melk": "mleko",
    "brød": "chleb",
    "ost": "ser",
    "smør": "masło",
    "yoghurt": "jogurt",
    "juice": "sok",
    "kaffe": "kawa",
    "te": "herbata",
    "sukker": "cukier",
    "mel": "mąka",
    "ris": "ryż",
    "pasta": "makaron",
    "sjokolade": "czekolada",
    "kjeks": "ciastka",
    "chips": "chipsy",
    "øl": "piwo",
    "vann": "woda",
    "brus": "napój",
    "egg": "jajka",
    "fisk": "ryba",
    "kylling": "kurczak",
    "svinekjøtt": "wieprzowina",
    "storfe": "wołowina",
    "pølse": "kiełbasa",
    "skinke": "szynka",
    "leverpostei": "pasztet",
    "smøreost": "serek",
    "fløte": "śmietana",
    "is": "lody",
    "knekkebrød": "pieczywo chrupkie",
    "müsli": "musli",
    "havregryn": "płatki owsiane",
    "cornflakes": "płatki kukurydziane",
}

# Common international brands found in both Norwegian and Polish stores
SHARED_BRANDS = {
    "coca-cola", "pepsi", "fanta", "sprite", "nestlé", "nestle", "nescafé", "nescafe",
    "lipton", "twinings", "dolce gusto", "jacobs", "douwe egberts",
    "kellogg", "nesquik", "nutella", "ferrero", "haribo", "mars", "snickers",
    "kitkat", "milka", "oreo", "pringles", "lay's", "lays", "doritos",
    "heinz", "knorr", "maggi", "barilla", "findus", "dr. oetker", "oetker",
    "arla", "danone", "activia", "président", "president", "philadelphia",
    "hellmann's", "hellmanns", "colgate", "dove", "nivea", "pampers",
    "wasa", "ryvita", "carr's", "tuc", "ritz",
    "evergood", "friele",  # Norwegian but might appear
    "first price", "eldorado",  # Norwegian store brands
}


def normalize_name(name: str) -> str:
    """Normalize product name for comparison."""
    name = name.lower().strip()
    # Remove weight/quantity patterns
    name = re.sub(r'\d+\s*(g|kg|ml|l|cl|dl|oz|stk|szt|pos|kapsler|sztuk)\b', '', name)
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def extract_brand(name: str) -> str | None:
    """Try to extract brand name from product name."""
    name_lower = name.lower()
    for brand in SHARED_BRANDS:
        if brand in name_lower:
            return brand
    return None


def analyze_pl_dataset():
    """Analyze the Polish dataset structure."""
    ann_path = PL_DIR / "annotations.json"
    if not ann_path.exists():
        print(f"Annotations not found at {ann_path}")
        print("Extraction may still be in progress.")
        return None

    print("Loading PL annotations (this may take a moment for 223MB file)...")
    with open(ann_path) as f:
        pl_coco = json.load(f)

    print(f"\n=== SKUs on Shelves PL - OVERVIEW ===")
    print(f"Images: {len(pl_coco['images'])}")
    print(f"Annotations: {len(pl_coco['annotations'])}")
    print(f"Categories: {len(pl_coco['categories'])}")

    # Category analysis
    pl_cats = pl_coco["categories"]
    pl_cat_names = {c["id"]: c["name"] for c in pl_cats}
    pl_supercats = {c["id"]: c.get("supercategory", "N/A") for c in pl_cats}

    # Supercategory distribution
    supercat_counts = Counter(c.get("supercategory", "N/A") for c in pl_cats)
    print(f"\n=== SUPERCATEGORIES ===")
    for sc, count in supercat_counts.most_common(30):
        print(f"  {sc}: {count} categories")

    # Annotation counts per category
    pl_ann_counts = Counter(ann["category_id"] for ann in pl_coco["annotations"])

    # Sample category names
    print(f"\n=== SAMPLE CATEGORY NAMES (first 20) ===")
    for cat in pl_cats[:20]:
        count = pl_ann_counts.get(cat["id"], 0)
        print(f"  id={cat['id']}, name='{cat['name']}', super='{cat.get('supercategory', 'N/A')}', count={count}")

    # Check for shared brands
    print(f"\n=== SHARED BRAND ANALYSIS ===")
    brand_matches = defaultdict(list)
    for cat in pl_cats:
        brand = extract_brand(cat["name"])
        if brand:
            brand_matches[brand].append(cat)

    print(f"Brands found in PL dataset: {len(brand_matches)}")
    for brand, cats in sorted(brand_matches.items()):
        total_anns = sum(pl_ann_counts.get(c["id"], 0) for c in cats)
        print(f"  {brand}: {len(cats)} categories, {total_anns} annotations")

    return pl_coco


def build_category_mapping():
    """Map PL categories to our NorgesGruppen categories."""
    # Load our categories
    with open(OUR_ANNOTATIONS) as f:
        our_coco = json.load(f)
    our_cats = {c["id"]: c["name"] for c in our_coco["categories"]}

    pl_ann_path = PL_DIR / "annotations.json"
    if not pl_ann_path.exists():
        print("PL annotations not yet available")
        return

    print("Loading PL annotations...")
    with open(pl_ann_path) as f:
        pl_coco = json.load(f)
    pl_cats = {c["id"]: c for c in pl_coco["categories"]}

    # Strategy 1: Exact brand + product type matching
    mapping = {}  # pl_cat_id -> our_cat_id
    match_reasons = {}

    # Extract brands from our categories
    our_brand_cats = defaultdict(list)
    for cat_id, name in our_cats.items():
        brand = extract_brand(name)
        if brand:
            our_brand_cats[brand].append((cat_id, name))

    # Try to match PL categories to ours
    for pl_cat_id, pl_cat in pl_cats.items():
        pl_name = pl_cat["name"]
        pl_brand = extract_brand(pl_name)

        # Strategy 1: Same brand, best fuzzy match on product name
        if pl_brand and pl_brand in our_brand_cats:
            best_score = 0
            best_our_id = None
            pl_norm = normalize_name(pl_name)
            for our_id, our_name in our_brand_cats[pl_brand]:
                our_norm = normalize_name(our_name)
                score = SequenceMatcher(None, pl_norm, our_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_our_id = our_id

            if best_score > 0.4 and best_our_id is not None:
                mapping[pl_cat_id] = {
                    "our_category_id": best_our_id,
                    "our_category_name": our_cats[best_our_id],
                    "pl_category_name": pl_name,
                    "confidence": best_score,
                    "match_type": "brand_fuzzy",
                }

        # Strategy 2: Direct fuzzy match (no brand filter)
        if pl_cat_id not in mapping:
            pl_norm = normalize_name(pl_name)
            best_score = 0
            best_our_id = None
            for our_id, our_name in our_cats.items():
                our_norm = normalize_name(our_name)
                score = SequenceMatcher(None, pl_norm, our_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_our_id = our_id

            if best_score > 0.6 and best_our_id is not None:
                mapping[pl_cat_id] = {
                    "our_category_id": best_our_id,
                    "our_category_name": our_cats[best_our_id],
                    "pl_category_name": pl_name,
                    "confidence": best_score,
                    "match_type": "fuzzy",
                }

    print(f"\n=== CATEGORY MAPPING RESULTS ===")
    print(f"PL categories: {len(pl_cats)}")
    print(f"Our categories: {len(our_cats)}")
    print(f"Mapped: {len(mapping)}")

    # Stats by match type
    type_counts = Counter(m["match_type"] for m in mapping.values())
    for mt, count in type_counts.most_common():
        print(f"  {mt}: {count}")

    # Confidence distribution
    confs = [m["confidence"] for m in mapping.values()]
    if confs:
        import numpy as np
        print(f"\nConfidence: min={min(confs):.3f}, max={max(confs):.3f}, mean={np.mean(confs):.3f}")

    # Show some high-confidence matches
    print(f"\n=== TOP 20 MATCHES (highest confidence) ===")
    sorted_matches = sorted(mapping.items(), key=lambda x: x[1]["confidence"], reverse=True)
    for pl_id, info in sorted_matches[:20]:
        print(f"  PL: '{info['pl_category_name']}' -> OUR: '{info['our_category_name']}' "
              f"(conf={info['confidence']:.3f}, type={info['match_type']})")

    # Save mapping
    out_path = Path(__file__).parent / "outputs" / "pl_to_ng_category_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nMapping saved to {out_path}")

    # Also save as a simple lookup for the conversion script
    simple_mapping = {str(pl_id): info["our_category_id"] for pl_id, info in mapping.items()}
    simple_path = Path(__file__).parent / "outputs" / "pl_to_ng_simple_mapping.json"
    with open(simple_path, "w") as f:
        json.dump(simple_mapping, f, indent=2)
    print(f"Simple mapping saved to {simple_path}")

    return mapping


def main():
    print("=" * 60)
    print("SKUs on Shelves PL - Analysis & Category Mapping")
    print("=" * 60)

    pl_coco = analyze_pl_dataset()
    if pl_coco is not None:
        build_category_mapping()


if __name__ == "__main__":
    main()
