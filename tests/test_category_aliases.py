"""Tests for tasks/object-detection/category_aliases.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "object-detection"))

from category_aliases import (
    CONFUSABLE_PAIRS,
    IDENTICAL_ALIASES,
    apply_aliases,
    apply_aliases_to_predictions,
    are_confusable,
    get_confusable_boost,
)


# ---------------------------------------------------------------------------
# apply_aliases
# ---------------------------------------------------------------------------

class TestApplyAliases:
    def test_known_alias_59_to_61(self):
        assert apply_aliases(59) == 61

    def test_known_alias_170_to_260(self):
        assert apply_aliases(170) == 260

    def test_known_alias_36_to_201(self):
        assert apply_aliases(36) == 201

    def test_unknown_id_returns_unchanged(self):
        assert apply_aliases(100) == 100

    def test_zero_returns_zero(self):
        assert apply_aliases(0) == 0

    def test_all_known_aliases_map_correctly(self):
        for src, dst in IDENTICAL_ALIASES.items():
            assert apply_aliases(src) == dst

    def test_canonical_id_returns_itself(self):
        # Canonical IDs (values in IDENTICAL_ALIASES) should not be remapped
        for canonical in IDENTICAL_ALIASES.values():
            assert apply_aliases(canonical) == canonical


# ---------------------------------------------------------------------------
# are_confusable
# ---------------------------------------------------------------------------

class TestAreConfusable:
    def test_known_pair_200_225(self):
        assert are_confusable(200, 225) is True

    def test_pair_is_symmetric(self):
        assert are_confusable(225, 200) is True

    def test_unknown_pair_returns_false(self):
        assert are_confusable(0, 1) is False

    def test_same_id_returns_false(self):
        assert are_confusable(200, 200) is False

    def test_all_pairs_are_symmetric(self):
        for a, b in CONFUSABLE_PAIRS:
            assert are_confusable(a, b) is True
            assert are_confusable(b, a) is True


# ---------------------------------------------------------------------------
# get_confusable_boost
# ---------------------------------------------------------------------------

class TestGetConfusableBoost:
    def test_confusable_pair_boosts_score(self):
        result = get_confusable_boost(200, 225, 0.8)
        assert result > 0.8

    def test_non_confusable_pair_unchanged(self):
        result = get_confusable_boost(0, 1, 0.8)
        assert result == 0.8

    def test_boost_capped_at_0_99(self):
        result = get_confusable_boost(200, 225, 0.98)
        assert result <= 0.99

    def test_zero_score_stays_zero_for_non_confusable(self):
        result = get_confusable_boost(0, 1, 0.0)
        assert result == 0.0

    def test_high_score_confusable_does_not_exceed_limit(self):
        result = get_confusable_boost(200, 225, 1.0)
        assert result <= 0.99


# ---------------------------------------------------------------------------
# apply_aliases_to_predictions
# ---------------------------------------------------------------------------

class TestApplyAliasesToPredictions:
    def test_empty_list_returns_empty(self):
        assert apply_aliases_to_predictions([]) == []

    def test_known_alias_is_remapped(self):
        preds = [{"image_id": 1, "bbox": [0, 0, 10, 10], "category_id": 59, "score": 0.9}]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["category_id"] == 61

    def test_unknown_category_unchanged(self):
        preds = [{"image_id": 1, "bbox": [0, 0, 10, 10], "category_id": 100, "score": 0.9}]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["category_id"] == 100

    def test_modifies_list_in_place(self):
        preds = [{"category_id": 59}]
        result = apply_aliases_to_predictions(preds)
        assert result is preds

    def test_multiple_predictions(self):
        preds = [
            {"category_id": 59},
            {"category_id": 170},
            {"category_id": 999},
        ]
        apply_aliases_to_predictions(preds)
        assert preds[0]["category_id"] == 61
        assert preds[1]["category_id"] == 260
        assert preds[2]["category_id"] == 999

    def test_other_fields_preserved(self):
        preds = [{"image_id": 42, "score": 0.75, "category_id": 59, "bbox": [1, 2, 3, 4]}]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["image_id"] == 42
        assert result[0]["score"] == 0.75
        assert result[0]["bbox"] == [1, 2, 3, 4]
