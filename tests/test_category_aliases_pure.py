"""Pure-function tests for object-detection/category_aliases.py.

Covers: apply_aliases, are_confusable, get_confusable_boost,
        apply_aliases_to_predictions, IDENTICAL_ALIASES, CONFUSABLE_PAIRS

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_OD_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "object-detection")
sys.path.insert(0, _OD_DIR)

from category_aliases import (
    apply_aliases,
    are_confusable,
    get_confusable_boost,
    apply_aliases_to_predictions,
    IDENTICAL_ALIASES,
    CONFUSABLE_PAIRS,
    _CONFUSABLE_SET,
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

    def test_no_alias_returns_original(self):
        assert apply_aliases(100) == 100

    def test_zero_returns_zero(self):
        assert apply_aliases(0) == 0

    def test_all_aliases_map_to_different_id(self):
        for src, dst in IDENTICAL_ALIASES.items():
            assert src != dst

    def test_idempotent_after_alias(self):
        # Applying alias to the target is a no-op
        for src, dst in IDENTICAL_ALIASES.items():
            assert apply_aliases(dst) == dst


# ---------------------------------------------------------------------------
# are_confusable
# ---------------------------------------------------------------------------

class TestAreConfusable:
    def test_known_pair_forward(self):
        assert are_confusable(200, 225) is True

    def test_known_pair_reverse(self):
        # Confusability is symmetric
        assert are_confusable(225, 200) is True

    def test_unknown_pair_false(self):
        assert are_confusable(1, 2) is False

    def test_same_category_not_confusable(self):
        # Self-confusion is not in the set
        assert are_confusable(200, 200) is False

    def test_confusable_set_symmetric(self):
        # All pairs in _CONFUSABLE_SET should appear both ways
        for a, b in list(_CONFUSABLE_SET):
            assert (b, a) in _CONFUSABLE_SET

    def test_all_defined_pairs_are_confusable(self):
        for (a, b) in CONFUSABLE_PAIRS:
            assert are_confusable(a, b)
            assert are_confusable(b, a)


# ---------------------------------------------------------------------------
# get_confusable_boost
# ---------------------------------------------------------------------------

class TestGetConfusableBoost:
    def test_confusable_pair_boosts_score(self):
        # Confusable pair → score increases
        original = 0.7
        boosted = get_confusable_boost(200, 225, original)
        assert boosted > original

    def test_non_confusable_no_boost(self):
        original = 0.7
        result = get_confusable_boost(1, 2, original)
        assert result == original

    def test_boost_capped_at_0_99(self):
        # Very high score should be capped
        result = get_confusable_boost(200, 225, 0.99)
        assert result <= 0.99

    def test_boost_factor_is_1_05(self):
        # 0.80 * 1.05 = 0.84
        result = get_confusable_boost(200, 225, 0.80)
        assert result == pytest.approx(0.84, rel=1e-5)

    def test_zero_score_stays_zero(self):
        result = get_confusable_boost(200, 225, 0.0)
        assert result == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(get_confusable_boost(200, 225, 0.6), float)


# ---------------------------------------------------------------------------
# apply_aliases_to_predictions
# ---------------------------------------------------------------------------

class TestApplyAliasesToPredictions:
    def test_maps_aliased_category(self):
        preds = [{"image_id": 1, "bbox": [0, 0, 10, 10], "category_id": 59, "score": 0.9}]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["category_id"] == 61

    def test_no_alias_unchanged(self):
        preds = [{"image_id": 1, "bbox": [0, 0, 10, 10], "category_id": 100, "score": 0.5}]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["category_id"] == 100

    def test_empty_list(self):
        assert apply_aliases_to_predictions([]) == []

    def test_multiple_predictions(self):
        preds = [
            {"image_id": 1, "bbox": [], "category_id": 59, "score": 0.9},
            {"image_id": 2, "bbox": [], "category_id": 170, "score": 0.8},
            {"image_id": 3, "bbox": [], "category_id": 100, "score": 0.7},
        ]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["category_id"] == 61
        assert result[1]["category_id"] == 260
        assert result[2]["category_id"] == 100

    def test_modifies_in_place_and_returns(self):
        preds = [{"image_id": 1, "bbox": [], "category_id": 59, "score": 0.9}]
        result = apply_aliases_to_predictions(preds)
        assert result is preds  # same list returned

    def test_other_fields_untouched(self):
        preds = [{"image_id": 42, "bbox": [1, 2, 3, 4], "category_id": 36, "score": 0.75}]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["image_id"] == 42
        assert result[0]["bbox"] == [1, 2, 3, 4]
        assert result[0]["score"] == pytest.approx(0.75)
