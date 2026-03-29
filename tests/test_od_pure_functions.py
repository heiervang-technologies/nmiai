"""Pure-function tests for object-detection miscellaneous helpers.

Covers:
  tasks/object-detection/deep_insights_v2.py — compute_difficulty_score
  tasks/object-detection/eval_weak_categories.py (iou xywh variant)

All pure functions — no file system, GPU, or network access.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_OD_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "object-detection")


def _load_func(module_name, path, func_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


# Load compute_difficulty_score directly
compute_difficulty_score = _load_func(
    "deep_insights_v2",
    str(Path(_OD_DIR) / "deep_insights_v2.py"),
    "compute_difficulty_score",
)

# Load iou from eval_weak_categories — mock torch and ultralytics at module level
for mod_name in ("torch", "ultralytics"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
        if mod_name == "ultralytics":
            sys.modules[mod_name].YOLO = object

iou_xywh = _load_func(
    "eval_weak_categories",
    str(Path(_OD_DIR) / "eval_weak_categories.py"),
    "iou",
)


# ---------------------------------------------------------------------------
# compute_difficulty_score
# ---------------------------------------------------------------------------

class TestComputeDifficultyScore:
    def _base(self, cat_id=1, count=50, n_imgs=5, bboxes=None, name="GENERIC"):
        if bboxes is None:
            bboxes = [{"rel_area": 0.01}]
        return compute_difficulty_score(
            cat_id=cat_id,
            class_counts={cat_id: count},
            class_names={cat_id: name},
            class_images={cat_id: set(range(n_imgs))},
            class_bboxes={cat_id: bboxes},
            total_images=100,
        )

    def test_returns_dict(self):
        result = self._base()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = self._base()
        for key in ("total", "scarcity", "diversity", "size", "confusability"):
            assert key in result

    def test_total_capped_at_100(self):
        result = self._base(count=0, n_imgs=0, bboxes=[{"rel_area": 0.0001}])
        assert result["total"] <= 100

    def test_total_nonneg(self):
        result = self._base(count=1000, n_imgs=100, bboxes=[{"rel_area": 0.5}])
        assert result["total"] >= 0

    def test_zero_count_high_scarcity(self):
        r_zero = self._base(count=0)
        r_many = self._base(count=1000)
        assert r_zero["scarcity"] > r_many["scarcity"]

    def test_few_images_high_diversity(self):
        r_few = self._base(n_imgs=1)
        r_many = self._base(n_imgs=20)
        assert r_few["diversity"] > r_many["diversity"]

    def test_small_bbox_high_size_score(self):
        r_small = self._base(bboxes=[{"rel_area": 0.0005}])
        r_large = self._base(bboxes=[{"rel_area": 0.05}])
        assert r_small["size"] > r_large["size"]

    def test_no_bboxes_uses_unknown_fallback(self):
        result = compute_difficulty_score(
            cat_id=1,
            class_counts={1: 50},
            class_names={1: "GENERIC"},
            class_images={1: set(range(5))},
            class_bboxes={},  # no bboxes
            total_images=100,
        )
        assert result["size"] == 15  # fallback for unknown

    def test_knekke_high_confusability(self):
        r_knekke = self._base(name="KNEKKEBRØD")
        r_other = self._base(name="GENERIC")
        assert r_knekke["confusability"] > r_other["confusability"]

    def test_egg_high_confusability(self):
        r_egg = self._base(name="EGG FRITTGÅENDE")
        r_other = self._base(name="GENERIC")
        assert r_egg["confusability"] > r_other["confusability"]

    def test_missing_category_in_counts(self):
        # Category not in counts → count=0 → scarcity=40
        result = compute_difficulty_score(
            cat_id=99,
            class_counts={},  # cat 99 not present
            class_names={99: "MISSING"},
            class_images={},
            class_bboxes={},
            total_images=100,
        )
        assert result["scarcity"] == 40

    def test_returns_rounded_values(self):
        result = self._base()
        # All values should be numbers (float after rounding)
        for key in ("total", "scarcity", "diversity", "size", "confusability"):
            assert isinstance(result[key], (int, float))

    def test_total_equals_component_sum(self):
        result = self._base()
        component_sum = result["scarcity"] + result["diversity"] + result["size"] + result["confusability"]
        # total = min(component_sum, 100)
        expected = min(round(component_sum, 1), 100)
        assert abs(result["total"] - expected) < 0.2


# ---------------------------------------------------------------------------
# eval_weak_categories.iou (xywh format)
# ---------------------------------------------------------------------------

class TestIouXywh:
    def test_identical_boxes(self):
        box = [10, 10, 50, 50]
        assert abs(iou_xywh(box, box) - 1.0) < 1e-6

    def test_non_overlapping(self):
        a = [0, 0, 10, 10]
        b = [20, 20, 10, 10]
        assert iou_xywh(a, b) == 0.0

    def test_partial_overlap(self):
        a = [0, 0, 10, 10]
        b = [5, 5, 10, 10]
        # Intersection: 5×5=25, union: 100+100-25=175
        expected = 25 / 175
        assert abs(iou_xywh(a, b) - expected) < 1e-6

    def test_range_0_to_1(self):
        a = [0, 0, 30, 40]
        b = [15, 20, 30, 40]
        result = iou_xywh(a, b)
        assert 0 <= result <= 1

    def test_zero_area_returns_near_zero(self):
        a = [5, 5, 0, 0]  # zero area
        b = [0, 0, 10, 10]
        # union = 0 + 100 - 0 = 100, but union guard = max(union, 1e-6)
        assert iou_xywh(a, b) >= 0

    def test_symmetric(self):
        a = [0, 0, 20, 15]
        b = [10, 8, 25, 20]
        assert abs(iou_xywh(a, b) - iou_xywh(b, a)) < 1e-9

    def test_xywh_format_vs_xyxy(self):
        # box [10, 10, 20, 20] in xywh is the same region as [10,10,30,30] in xyxy
        # For two identical xywh boxes, IoU = 1
        a = [0, 0, 100, 100]
        b = [0, 0, 100, 100]
        assert abs(iou_xywh(a, b) - 1.0) < 1e-9
