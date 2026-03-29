"""Pure-function tests for OD data-creation conversion helpers.

Covers:
  build_barcode_mapping.py : normalize
  convert_to_yolo.py       : coco_to_yolo_bbox

Both are pure string/math functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_OD_DATA_DIR = str(
    Path(__file__).resolve().parent.parent
    / "tasks" / "object-detection" / "data-creation"
)
sys.path.insert(0, _OD_DATA_DIR)

from build_barcode_mapping import normalize
from convert_to_yolo import coco_to_yolo_bbox


# ---------------------------------------------------------------------------
# normalize  (build_barcode_mapping)
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert normalize("COCA-COLA") == "coca-cola"

    def test_strips_whitespace(self):
        assert normalize("  melk  ") == "melk"

    def test_removes_grams(self):
        result = normalize("Sjokolade 200g")
        assert "200g" not in result

    def test_removes_kg(self):
        result = normalize("Poteter 2kg")
        assert "2kg" not in result

    def test_removes_ml(self):
        result = normalize("Brus 500ml")
        assert "500ml" not in result

    def test_removes_l(self):
        result = normalize("Juice 1l")
        assert "1l" not in result

    def test_removes_cl(self):
        result = normalize("Drink 33cl")
        assert "33cl" not in result

    def test_removes_stk(self):
        result = normalize("Kapsler 12stk")
        assert "12stk" not in result

    def test_removes_kapsler_with_digit(self):
        result = normalize("Ibuprofen 12kapsler")
        assert "kapsler" not in result

    def test_collapses_whitespace(self):
        result = normalize("Melk   hvit   fullmelk")
        assert "  " not in result

    def test_returns_string(self):
        result = normalize("Juice 1L")
        assert isinstance(result, str)

    def test_empty_string(self):
        result = normalize("")
        assert result == ""

    def test_unit_at_start(self):
        result = normalize("500g havregryn")
        assert "500g" not in result


# ---------------------------------------------------------------------------
# coco_to_yolo_bbox  (convert_to_yolo)
# ---------------------------------------------------------------------------

class TestCocoToYoloBbox:
    def test_basic_center(self):
        # Box at top-left (0,0), size (100,100) in 200x200 image
        # cx = (0 + 50) / 200 = 0.25, cy = 0.25, nw = 0.5, nh = 0.5
        cx, cy, nw, nh = coco_to_yolo_bbox([0, 0, 100, 100], 200, 200)
        assert cx == pytest.approx(0.25)
        assert cy == pytest.approx(0.25)
        assert nw == pytest.approx(0.5)
        assert nh == pytest.approx(0.5)

    def test_full_image_box(self):
        # Box covering full 640x480 image
        cx, cy, nw, nh = coco_to_yolo_bbox([0, 0, 640, 480], 640, 480)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert nw == pytest.approx(1.0)
        assert nh == pytest.approx(1.0)

    def test_values_clamped_at_one(self):
        # Box slightly larger than image → clamp to 1.0
        cx, cy, nw, nh = coco_to_yolo_bbox([0, 0, 1000, 1000], 100, 100)
        assert cx <= 1.0
        assert cy <= 1.0
        assert nw <= 1.0
        assert nh <= 1.0

    def test_values_non_negative(self):
        # Negative x/y → clamped at 0
        cx, cy, nw, nh = coco_to_yolo_bbox([-10, -10, 50, 50], 100, 100)
        assert cx >= 0.0
        assert cy >= 0.0
        assert nw >= 0.0
        assert nh >= 0.0

    def test_returns_four_floats(self):
        result = coco_to_yolo_bbox([10, 20, 30, 40], 640, 480)
        assert len(result) == 4
        for v in result:
            assert isinstance(float(v), float)

    def test_symmetry_center(self):
        # Box centered in image → cx, cy = 0.5
        cx, cy, nw, nh = coco_to_yolo_bbox([100, 100, 200, 200], 400, 400)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)

    def test_small_box_in_corner(self):
        # Tiny 10x10 box at (0,0) in 1000x1000 image
        cx, cy, nw, nh = coco_to_yolo_bbox([0, 0, 10, 10], 1000, 1000)
        assert cx == pytest.approx(0.005)
        assert cy == pytest.approx(0.005)
        assert nw == pytest.approx(0.01)
        assert nh == pytest.approx(0.01)

    def test_aspect_ratio_preserved(self):
        # 200 wide, 100 tall in 400x400 image
        cx, cy, nw, nh = coco_to_yolo_bbox([0, 0, 200, 100], 400, 400)
        assert nw == pytest.approx(0.5)
        assert nh == pytest.approx(0.25)

    def test_non_square_image(self):
        cx, cy, nw, nh = coco_to_yolo_bbox([0, 0, 640, 480], 1280, 960)
        assert cx == pytest.approx(0.25)
        assert cy == pytest.approx(0.25)
        assert nw == pytest.approx(0.5)
        assert nh == pytest.approx(0.5)
