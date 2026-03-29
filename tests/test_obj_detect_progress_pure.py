"""Tests for tasks/object-detection/project_progress.py — pure helpers.

Covers: to_float, extract_metrics.
Also covers tasks/object-detection/category_aliases.py pure helpers:
apply_aliases, are_confusable, get_confusable_boost.
All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_OD_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "object-detection")
sys.path.insert(0, _OD_DIR)

# Import from object-detection project_progress (not accounting version)
import importlib, importlib.util

spec = importlib.util.spec_from_file_location(
    "od_project_progress",
    Path(_OD_DIR) / "project_progress.py",
)
od_pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(od_pp)

to_float = od_pp.to_float
extract_metrics = od_pp.extract_metrics

from category_aliases import (
    apply_aliases,
    apply_aliases_to_predictions,
    are_confusable,
    get_confusable_boost,
)


# ---------------------------------------------------------------------------
# project_progress.to_float (object-detection version)
# ---------------------------------------------------------------------------

class TestToFloatOD:
    def test_valid_int_string(self):
        assert to_float("42") == 42.0

    def test_valid_float_string(self):
        result = to_float("3.14")
        assert abs(result - 3.14) < 1e-9

    def test_none_gives_none(self):
        assert to_float(None) is None

    def test_empty_gives_none(self):
        assert to_float("") is None

    def test_timeout_gives_none(self):
        assert to_float("TIMEOUT") is None

    def test_na_gives_none(self):
        assert to_float("N/A") is None

    def test_tilde_stripped(self):
        result = to_float("~0.85")
        assert result is not None
        assert abs(result - 0.85) < 1e-9

    def test_whitespace_stripped(self):
        result = to_float("  2.5  ")
        assert result is not None
        assert abs(result - 2.5) < 1e-9

    def test_non_numeric_gives_none(self):
        assert to_float("abc") is None

    def test_negative_value(self):
        result = to_float("-1.5")
        assert result is not None
        assert abs(result - (-1.5)) < 1e-9


# ---------------------------------------------------------------------------
# project_progress.extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_single_metric(self):
        row = {"metric": "mAP50", "value": "0.72"}
        result = extract_metrics(row)
        assert "mAP50" in result
        assert abs(result["mAP50"] - 0.72) < 1e-9

    def test_two_metrics(self):
        row = {"metric": "mAP50", "value": "0.72", "metric2": "mAP75", "value2": "0.55"}
        result = extract_metrics(row)
        assert "mAP50" in result
        assert "mAP75" in result

    def test_empty_metric_name_skipped(self):
        row = {"metric": "", "value": "0.5"}
        result = extract_metrics(row)
        assert result == {}

    def test_none_value_skipped(self):
        row = {"metric": "mAP50", "value": "N/A"}
        result = extract_metrics(row)
        assert result == {}

    def test_returns_dict(self):
        row = {"metric": "loss", "value": "0.1"}
        assert isinstance(extract_metrics(row), dict)

    def test_missing_keys_gives_empty(self):
        row = {}
        assert extract_metrics(row) == {}


# ---------------------------------------------------------------------------
# category_aliases.apply_aliases
# ---------------------------------------------------------------------------

class TestApplyAliases:
    def test_known_alias_59_to_61(self):
        assert apply_aliases(59) == 61

    def test_known_alias_170_to_260(self):
        assert apply_aliases(170) == 260

    def test_known_alias_36_to_201(self):
        assert apply_aliases(36) == 201

    def test_no_alias_unchanged(self):
        assert apply_aliases(100) == 100

    def test_unknown_category_unchanged(self):
        assert apply_aliases(9999) == 9999


# ---------------------------------------------------------------------------
# category_aliases.are_confusable
# ---------------------------------------------------------------------------

class TestAreConfusable:
    def test_known_pair(self):
        assert are_confusable(200, 225)

    def test_reversed_pair(self):
        assert are_confusable(225, 200)

    def test_non_confusable(self):
        assert not are_confusable(1, 2)

    def test_same_category_not_confusable(self):
        assert not are_confusable(100, 100)


# ---------------------------------------------------------------------------
# category_aliases.get_confusable_boost
# ---------------------------------------------------------------------------

class TestGetConfusableBoost:
    def test_confusable_pair_boosts_score(self):
        score = 0.8
        result = get_confusable_boost(200, 225, score)
        assert result > score

    def test_boost_capped_at_0_99(self):
        result = get_confusable_boost(200, 225, 0.99)
        assert result <= 0.99

    def test_non_confusable_unchanged(self):
        score = 0.8
        result = get_confusable_boost(1, 2, score)
        assert result == score

    def test_returns_float(self):
        assert isinstance(get_confusable_boost(200, 225, 0.7), float)


# ---------------------------------------------------------------------------
# category_aliases.apply_aliases_to_predictions
# ---------------------------------------------------------------------------

class TestApplyAliasesToPredictions:
    def test_applies_alias(self):
        preds = [{"image_id": 1, "category_id": 59, "score": 0.9, "bbox": []}]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["category_id"] == 61

    def test_no_alias_unchanged(self):
        preds = [{"image_id": 1, "category_id": 100, "score": 0.9, "bbox": []}]
        result = apply_aliases_to_predictions(preds)
        assert result[0]["category_id"] == 100

    def test_returns_list(self):
        preds = [{"image_id": 1, "category_id": 1, "score": 0.5, "bbox": []}]
        assert isinstance(apply_aliases_to_predictions(preds), list)

    def test_empty_input(self):
        assert apply_aliases_to_predictions([]) == []
