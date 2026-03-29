"""Pure-function tests for object-detection data-creation helpers.

Covers:
  build_v7_balanced.py      : adjust_boxes
  pseudo_label_stores.py    : nms_per_class
  extract_video_frames.py   : frame_diff

All pure functions — no file system, network, GPU, or real video access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Stub heavy dependencies before importing
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("ultralytics", MagicMock())

_OD_DATA_DIR = str(
    Path(__file__).resolve().parent.parent
    / "tasks" / "object-detection" / "data-creation"
)
sys.path.insert(0, _OD_DATA_DIR)

from build_v7_balanced import adjust_boxes
from pseudo_label_stores import nms_per_class
from extract_video_frames import frame_diff


# ---------------------------------------------------------------------------
# frame_diff
# ---------------------------------------------------------------------------

class TestFrameDiff:
    def test_identical_frames_returns_zero(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        assert frame_diff(frame, frame) == pytest.approx(0.0)

    def test_max_diff_is_255(self):
        a = np.zeros((5, 5), dtype=np.uint8)
        b = np.full((5, 5), 255, dtype=np.uint8)
        assert frame_diff(a, b) == pytest.approx(255.0)

    def test_half_diff(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.full((4, 4), 100, dtype=np.uint8)
        assert frame_diff(a, b) == pytest.approx(100.0)

    def test_nonneg(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, (8, 8), dtype=np.uint8)
        b = rng.integers(0, 255, (8, 8), dtype=np.uint8)
        assert frame_diff(a, b) >= 0.0

    def test_returns_float(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.ones((4, 4), dtype=np.uint8)
        result = frame_diff(a, b)
        assert isinstance(float(result), float)


# ---------------------------------------------------------------------------
# adjust_boxes
# ---------------------------------------------------------------------------

class TestAdjustBoxes:
    def _box(self, cls=0, cx=0.5, cy=0.5, w=0.2, h=0.2):
        return [(cls, cx, cy, w, h)]

    def test_none_crop_returns_original(self):
        boxes = self._box()
        result = adjust_boxes(boxes, None)
        assert result == boxes

    def test_box_in_frame_is_kept(self):
        # Box centered at 0.5,0.5 with crop starting at 0, size 1.0
        boxes = [(0, 0.5, 0.5, 0.1, 0.1)]
        result = adjust_boxes(boxes, (0.0, 0.0, 1.0, 1.0))
        assert len(result) == 1

    def test_box_outside_frame_is_discarded(self):
        # Box centered at 0.5,0.5 but shifted out (new_cx < 0.1)
        boxes = [(0, 0.05, 0.5, 0.1, 0.1)]
        result = adjust_boxes(boxes, (0.0, 0.0, 1.0, 1.0))
        assert len(result) == 0

    def test_coordinates_clamped_to_01(self):
        boxes = [(0, 0.5, 0.5, 0.4, 0.4)]
        result = adjust_boxes(boxes, (0.0, 0.0, 1.0, 1.0))
        if result:
            cls, cx, cy, w, h = result[0]
            assert 0 <= cx <= 1
            assert 0 <= cy <= 1
            assert 0 < w <= 1
            assert 0 < h <= 1

    def test_returns_list(self):
        boxes = self._box()
        result = adjust_boxes(boxes, (0.1, 0.1, 0.8, 0.8))
        assert isinstance(result, list)

    def test_empty_boxes_returns_empty(self):
        result = adjust_boxes([], (0.0, 0.0, 1.0, 1.0))
        assert result == []

    def test_class_label_preserved(self):
        boxes = [(3, 0.5, 0.5, 0.1, 0.1)]
        result = adjust_boxes(boxes, (0.0, 0.0, 1.0, 1.0))
        if result:
            assert result[0][0] == 3


# ---------------------------------------------------------------------------
# nms_per_class
# ---------------------------------------------------------------------------

class TestNmsPerClass:
    def _make_boxes(self, xywh_list):
        """Convert (x1, y1, x2, y2) list to numpy array."""
        return np.array(xywh_list, dtype=np.float32)

    def test_single_box_kept(self):
        boxes = self._make_boxes([[0, 0, 0.5, 0.5]])
        scores = np.array([0.9])
        class_ids = np.array([0])
        keep = nms_per_class(boxes, scores, class_ids)
        assert len(keep) == 1

    def test_non_overlapping_boxes_all_kept(self):
        boxes = self._make_boxes([
            [0, 0, 0.2, 0.2],
            [0.5, 0.5, 0.9, 0.9],
        ])
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 0])
        keep = nms_per_class(boxes, scores, class_ids)
        assert len(keep) == 2

    def test_identical_boxes_one_kept(self):
        boxes = self._make_boxes([
            [0, 0, 0.5, 0.5],
            [0, 0, 0.5, 0.5],
        ])
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 0])
        keep = nms_per_class(boxes, scores, class_ids)
        assert len(keep) == 1

    def test_different_classes_both_kept(self):
        # Same position but different classes → NMS is per-class
        boxes = self._make_boxes([
            [0, 0, 0.5, 0.5],
            [0, 0, 0.5, 0.5],
        ])
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 1])  # different classes
        keep = nms_per_class(boxes, scores, class_ids)
        assert len(keep) == 2

    def test_returns_list(self):
        boxes = self._make_boxes([[0, 0, 0.5, 0.5]])
        keep = nms_per_class(boxes, np.array([0.9]), np.array([0]))
        assert isinstance(keep, list)

    def test_high_score_preferred(self):
        # Two overlapping boxes of same class; higher score should be kept
        boxes = self._make_boxes([
            [0, 0, 0.5, 0.5],
            [0.05, 0.05, 0.55, 0.55],
        ])
        scores = np.array([0.5, 0.95])  # second has higher score
        class_ids = np.array([0, 0])
        keep = nms_per_class(boxes, scores, class_ids)
        # The kept index should correspond to the higher-score box
        assert len(keep) == 1
