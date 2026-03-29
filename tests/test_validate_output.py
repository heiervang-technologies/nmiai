"""Tests for tasks/object-detection/test-sandbox/validate_output.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
        / "tasks"
        / "object-detection"
        / "test-sandbox"
    ),
)

from validate_output import validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, data) -> Path:
    p = tmp_path / "preds.json"
    p.write_text(json.dumps(data))
    return p


def _valid_entry(**overrides) -> dict:
    base = {
        "image_id": 1,
        "bbox": [10.0, 20.0, 50.0, 30.0],
        "category_id": 5,
        "score": 0.9,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Root format
# ---------------------------------------------------------------------------

class TestRootFormat:
    def test_empty_array_returns_ok(self, tmp_path):
        p = _write(tmp_path, [])
        ok, issues = validate(p)
        assert ok is True

    def test_empty_array_has_warning(self, tmp_path):
        p = _write(tmp_path, [])
        _, issues = validate(p)
        assert any("WARNING" in i for i in issues)

    def test_non_array_returns_false(self, tmp_path):
        p = _write(tmp_path, {"predictions": []})
        ok, _ = validate(p)
        assert ok is False

    def test_nested_format_rejected(self, tmp_path):
        p = _write(tmp_path, [{"predictions": [], "image_id": 1}])
        ok, issues = validate(p)
        assert ok is False
        assert any("WRONG FORMAT" in i for i in issues)

    def test_confidence_field_rejected(self, tmp_path):
        p = _write(tmp_path, [{"image_id": 1, "bbox": [0, 0, 1, 1], "category_id": 0, "confidence": 0.9}])
        ok, issues = validate(p)
        assert ok is False
        assert any("confidence" in i for i in issues)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_valid_entry_passes(self, tmp_path):
        p = _write(tmp_path, [_valid_entry()])
        ok, _ = validate(p)
        assert ok is True

    def test_missing_score_field(self, tmp_path):
        entry = _valid_entry()
        del entry["score"]
        p = _write(tmp_path, [entry])
        ok, _ = validate(p)
        assert ok is False

    def test_missing_bbox_field(self, tmp_path):
        entry = _valid_entry()
        del entry["bbox"]
        p = _write(tmp_path, [entry])
        ok, _ = validate(p)
        assert ok is False

    def test_missing_category_id_field(self, tmp_path):
        entry = _valid_entry()
        del entry["category_id"]
        p = _write(tmp_path, [entry])
        ok, issues = validate(p)
        # Missing field should produce an issue
        assert len(issues) > 0


# ---------------------------------------------------------------------------
# Category ID validation
# ---------------------------------------------------------------------------

class TestCategoryId:
    def test_zero_is_valid(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(category_id=0)])
        ok, _ = validate(p)
        assert ok is True

    def test_356_is_valid(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(category_id=356)])
        ok, _ = validate(p)
        assert ok is True

    def test_357_is_invalid(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(category_id=357)])
        ok, issues = validate(p)
        assert ok is False or any("357" in i for i in issues)

    def test_negative_category_is_invalid(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(category_id=-1)])
        ok, issues = validate(p)
        assert ok is False or any("-1" in i for i in issues)

    def test_string_category_is_invalid(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(category_id="5")])
        ok, issues = validate(p)
        assert ok is False or any("category_id" in i for i in issues)


# ---------------------------------------------------------------------------
# Score validation
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_zero_is_valid(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(score=0.0)])
        ok, _ = validate(p)
        assert ok is True

    def test_score_one_is_valid(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(score=1.0)])
        ok, _ = validate(p)
        assert ok is True

    def test_score_above_one_flagged(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(score=1.5)])
        _, issues = validate(p)
        assert any("1.5" in i for i in issues)

    def test_score_negative_flagged(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(score=-0.1)])
        _, issues = validate(p)
        assert any("-0.1" in i for i in issues)


# ---------------------------------------------------------------------------
# BBox validation
# ---------------------------------------------------------------------------

class TestBbox:
    def test_negative_width_flagged(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(bbox=[10.0, 10.0, -5.0, 20.0])])
        _, issues = validate(p)
        assert any("negative" in i.lower() for i in issues)

    def test_three_element_bbox_flagged(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(bbox=[10.0, 10.0, 50.0])])
        _, issues = validate(p)
        assert any("bbox" in i for i in issues)

    def test_negative_x_gives_warning(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(bbox=[-5.0, 10.0, 50.0, 30.0])])
        ok, issues = validate(p)
        # Negative x/y is a WARNING not a failure
        assert any("WARNING" in i and "negative" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# image_id handling
# ---------------------------------------------------------------------------

class TestImageId:
    def test_int_image_id_passes(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(image_id=42)])
        ok, _ = validate(p)
        assert ok is True

    def test_string_image_id_passes_with_warning(self, tmp_path):
        p = _write(tmp_path, [_valid_entry(image_id="img_00042")])
        ok, issues = validate(p)
        assert ok is True
        assert any("WARNING" in i and "image_id" in i for i in issues)
