"""Pure-function tests for object-detection/project_progress.py.

Covers: to_float, extract_metrics

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# mock matplotlib before importing
sys.modules.setdefault("matplotlib", MagicMock())
sys.modules.setdefault("matplotlib.pyplot", MagicMock())

_OD_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "object-detection")
sys.path.insert(0, _OD_DIR)

from project_progress import to_float, extract_metrics


# ---------------------------------------------------------------------------
# to_float
# ---------------------------------------------------------------------------

class TestToFloat:
    def test_plain_number(self):
        assert to_float("3.14") == pytest.approx(3.14)

    def test_tilde_prefix(self):
        assert to_float("~0.85") == pytest.approx(0.85)

    def test_integer_string(self):
        assert to_float("42") == pytest.approx(42.0)

    def test_none_returns_none(self):
        assert to_float(None) is None

    def test_empty_string_returns_none(self):
        assert to_float("") is None

    def test_whitespace_only_returns_none(self):
        assert to_float("   ") is None

    def test_timeout_returns_none(self):
        assert to_float("TIMEOUT") is None

    def test_na_returns_none(self):
        assert to_float("N/A") is None

    def test_non_numeric_returns_none(self):
        assert to_float("abc") is None

    def test_negative_number(self):
        assert to_float("-1.5") == pytest.approx(-1.5)

    def test_whitespace_stripped(self):
        assert to_float("  0.9  ") == pytest.approx(0.9)

    def test_tilde_with_whitespace(self):
        assert to_float("~  0.7") == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_single_metric(self):
        row = {"metric": "mAP50", "value": "0.85", "metric2": "", "value2": ""}
        result = extract_metrics(row)
        assert result == {"mAP50": pytest.approx(0.85)}

    def test_two_metrics(self):
        row = {"metric": "mAP50", "value": "0.85", "metric2": "combined", "value2": "0.72"}
        result = extract_metrics(row)
        assert result["mAP50"] == pytest.approx(0.85)
        assert result["combined"] == pytest.approx(0.72)

    def test_invalid_value_skipped(self):
        row = {"metric": "mAP50", "value": "TIMEOUT", "metric2": "combined", "value2": "0.5"}
        result = extract_metrics(row)
        assert "mAP50" not in result
        assert result["combined"] == pytest.approx(0.5)

    def test_empty_metric_skipped(self):
        row = {"metric": "", "value": "0.5", "metric2": "score", "value2": "0.9"}
        result = extract_metrics(row)
        assert "" not in result
        assert result["score"] == pytest.approx(0.9)

    def test_none_values_skipped(self):
        row = {"metric": None, "value": None}
        result = extract_metrics(row)
        assert result == {}

    def test_missing_keys_gives_empty(self):
        row = {}
        result = extract_metrics(row)
        assert result == {}

    def test_tilde_value_parsed(self):
        row = {"metric": "mAP", "value": "~0.42", "metric2": "", "value2": ""}
        result = extract_metrics(row)
        assert result["mAP"] == pytest.approx(0.42)

    def test_returns_dict(self):
        row = {"metric": "x", "value": "1.0"}
        assert isinstance(extract_metrics(row), dict)
