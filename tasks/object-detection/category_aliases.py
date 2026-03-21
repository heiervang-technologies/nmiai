"""
Category Aliasing for Inference Post-Processing.

Some categories in the dataset are the SAME product with different annotation spellings.
At inference time, if the classifier is uncertain between aliased categories, we should
pick the one with more training data (higher prior probability).

Also provides a confusion-aware confidence boost: if the model's top-2 predictions
are a known confusable pair, boost confidence in the prediction rather than suppressing it.

Usage in submission run.py:
    from category_aliases import apply_aliases, get_confusable_boost

    # After classification
    category_id = apply_aliases(predicted_category_id)

    # Optional: boost confidence for confusable pairs
    score = get_confusable_boost(top1_cat, top2_cat, top1_score)
"""

# Categories that are the SAME product (annotation spelling errors)
# Map: less-common spelling -> more-common spelling
IDENTICAL_ALIASES = {
    59: 61,    # MÜSLI BLÅBÆR 630G AXA (8 ann) -> MUSLI BLÅBÆR 630G AXA (144 ann)
    170: 260,  # MÜSLI ENERGI 650G AXA (8 ann) -> MUSLI ENERGI 675G AXA (108 ann)
    36: 201,   # MÜSLI FRUKT MÜSLI 700G AXA (14 ann) -> MUSLI FRUKT 700G AXA (45 ann)
}

# Known confusable pairs (NOT identical, but very similar visually)
# (cat_a, cat_b): distinguishing_feature
CONFUSABLE_PAIRS = {
    (200, 225): "pack_size_16vs10",      # Evergood Classic Kaffekapsel
    (103, 344): "pack_size_10vs16",      # Evergood Dark Roast Kaffekapsel
    (27, 189): "pack_size_50vs25",       # Yellow Label Tea Lipton
    (246, 338): "weight_260vs520",       # Husman Knekkebrød Wasa
    (102, 139): "pack_size_12vs6",       # Egg Frittgående Solegg Prior
    (159, 163): "pack_size_10vs6",       # Galåvolden Gårdsegg
    (4, 138): "pack_size_6vs10",         # Økologiske Egg
    (3, 38): "weight_260vs270",          # Knekkebrød Din Stund
    (175, 302): "weight_500vs250",       # Melange Margarin
    (67, 326): "weight_230vs540",        # Soft Flora Original
    (224, 253): "weight_540vs235",       # Soft Flora Lett
    (125, 169): "packaging",            # Sjokoladedrikk Rett i Koppen
    (125, 179): "packaging",            # Sjokoladedrikk Rett i Koppen
    (340, 341): "roast_dark_vs_classic", # Evergood Hele Bønner
    (40, 144): "grind_filter_vs_kok",   # Friele Frokost
    (49, 347): "grind_filter_vs_press", # Evergood Dark Roast
    (100, 304): "grind_filter_vs_kok",  # Evergood Classic
    (141, 304): "grind_press_vs_kok",   # Evergood Classic
    (196, 332): "variant_regular_dark", # COTW Lungo Kaffekapsel
    (213, 237): "size_and_count",       # Egg Frittgående Prior
}

# Build reverse lookup
_CONFUSABLE_SET = set()
for a, b in CONFUSABLE_PAIRS:
    _CONFUSABLE_SET.add((a, b))
    _CONFUSABLE_SET.add((b, a))


def apply_aliases(category_id: int) -> int:
    """Map spelling-variant categories to their canonical form."""
    return IDENTICAL_ALIASES.get(category_id, category_id)


def are_confusable(cat_a: int, cat_b: int) -> bool:
    """Check if two categories are a known confusable pair."""
    return (cat_a, cat_b) in _CONFUSABLE_SET


def get_confusable_boost(top1_cat: int, top2_cat: int, top1_score: float) -> float:
    """
    If top-1 and top-2 predictions are a known confusable pair,
    boost confidence slightly since the model is at least in the right
    product family.
    """
    if are_confusable(top1_cat, top2_cat):
        # The model correctly identified the product family
        # Small boost since it's a legitimate confusion
        return min(top1_score * 1.05, 0.99)
    return top1_score


def apply_aliases_to_predictions(predictions: list) -> list:
    """
    Apply category aliases to a list of COCO-format predictions.
    Each prediction: {"image_id": ..., "bbox": ..., "category_id": ..., "score": ...}
    """
    for pred in predictions:
        pred["category_id"] = apply_aliases(pred["category_id"])
    return predictions


if __name__ == "__main__":
    # Self-test
    assert apply_aliases(59) == 61
    assert apply_aliases(170) == 260
    assert apply_aliases(36) == 201
    assert apply_aliases(100) == 100  # No alias
    assert are_confusable(200, 225) == True
    assert are_confusable(225, 200) == True
    assert are_confusable(0, 1) == False
    print("All tests passed.")
    print(f"Identical aliases: {len(IDENTICAL_ALIASES)}")
    print(f"Confusable pairs: {len(CONFUSABLE_PAIRS)}")
