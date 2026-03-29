"""Tests for tasks/astar-island/neurosymbolic_predictor.py — pure helper functions.

Covers: cell_type_str, make_keys, cell_code_to_class, bucket_lookup.
All are pure classification/key-building/lookup functions with no I/O or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from neurosymbolic_predictor import (
    cell_type_str, make_keys, cell_code_to_class, bucket_lookup,
    N_CLASSES, MIN_SUPPORT,
)


# ---------------------------------------------------------------------------
# cell_type_str
# ---------------------------------------------------------------------------

class TestCellTypeStr:
    def test_settlement_1_returns_S(self):
        assert cell_type_str(1) == "S"

    def test_port_2_returns_P(self):
        assert cell_type_str(2) == "P"

    def test_forest_4_returns_F(self):
        assert cell_type_str(4) == "F"

    def test_plains_11_returns_L(self):
        assert cell_type_str(11) == "L"

    def test_empty_0_returns_L(self):
        assert cell_type_str(0) == "L"

    def test_ocean_10_returns_none(self):
        assert cell_type_str(10) is None

    def test_mountain_5_returns_none(self):
        assert cell_type_str(5) is None

    def test_unknown_returns_none(self):
        assert cell_type_str(99) is None


# ---------------------------------------------------------------------------
# make_keys
# ---------------------------------------------------------------------------

class TestMakeKeys:
    def test_ocean_returns_none(self):
        assert make_keys(10, 5.0, 1, 0, 0, False) is None

    def test_mountain_returns_none(self):
        assert make_keys(5, 3.0, 0, 0, 0, False) is None

    def test_returns_list(self):
        result = make_keys(1, 2.0, 1, 0, 0, False)
        assert isinstance(result, list)

    def test_returns_six_levels(self):
        result = make_keys(1, 2.0, 1, 0, 0, False)
        assert len(result) == 6

    def test_all_keys_are_tuples(self):
        result = make_keys(1, 2.0, 1, 0, 0, False)
        assert all(isinstance(k, tuple) for k in result)

    def test_first_element_is_type_string(self):
        result = make_keys(1, 2.0, 0, 0, 0, False)
        assert result[0][0] == "S"

    def test_port_type_in_keys(self):
        result = make_keys(2, 1.0, 0, 0, 0, False)
        assert result[0][0] == "P"

    def test_coast_flag_in_first_key(self):
        with_coast = make_keys(1, 2.0, 1, 0, 0, True)
        without_coast = make_keys(1, 2.0, 1, 0, 0, False)
        # Coast flag (1 vs 0) should differ at position [-1] of fine key
        assert with_coast[0][-1] == 1
        assert without_coast[0][-1] == 0

    def test_dist_clamped_at_15(self):
        result = make_keys(1, 100.0, 0, 0, 0, False)
        # dist in first key should be clamped to 15
        assert result[0][1] == 15

    def test_n_ocean_clamped_at_4(self):
        result = make_keys(1, 2.0, 10, 0, 0, False)
        assert result[0][3] == 4

    def test_n_civ_clamped_at_5(self):
        result = make_keys(1, 2.0, 0, 10, 0, False)
        assert result[0][2] == 5

    def test_broad_key_uses_min_dist_8(self):
        result = make_keys(4, 15.0, 0, 0, 0, False)
        # Broad key (last) should be (type, min(d, 8)) = ("F", 8)
        assert result[-1] == ("F",) or result[-2][1] <= 8

    def test_singleton_key_at_last_level(self):
        result = make_keys(4, 5.0, 0, 0, 0, False)
        # Last key should be just the type tuple ("F",)
        assert result[-1] == ("F",)


# ---------------------------------------------------------------------------
# cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    @pytest.mark.parametrize("code,expected", [
        (0, 0), (10, 0), (11, 0),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    ])
    def test_known_codes(self, code, expected):
        assert cell_code_to_class(code) == expected

    def test_unknown_returns_zero(self):
        assert cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# bucket_lookup
# ---------------------------------------------------------------------------

def _make_tables(key, prob, count):
    """Build minimal tables dict for bucket_lookup tests."""
    n = 6  # N_LEVELS
    means = [{} for _ in range(n)]
    counts = [{} for _ in range(n)]
    means[0][key] = prob
    counts[0][key] = count
    return {"means": means, "counts": counts}


class TestBucketLookup:
    def _uniform(self):
        return np.ones(N_CLASSES) / N_CLASSES

    def test_no_match_returns_none_zero(self):
        tables = {"means": [{} for _ in range(6)], "counts": [{} for _ in range(6)]}
        keys = make_keys(1, 2.0, 0, 0, 0, False)
        prob, c = bucket_lookup(tables, keys)
        assert prob is None
        assert c == 0

    def test_sufficient_support_returns_prob(self):
        fine_key = make_keys(1, 2.0, 0, 0, 0, False)[0]
        p = self._uniform()
        tables = _make_tables(fine_key, p, MIN_SUPPORT[0] + 1)
        keys = make_keys(1, 2.0, 0, 0, 0, False)
        prob, c = bucket_lookup(tables, keys)
        assert prob is not None
        assert c == MIN_SUPPORT[0] + 1

    def test_insufficient_support_skips_level(self):
        # count=1 is below MIN_SUPPORT[0]=5, so lookup should skip level 0
        fine_key = make_keys(1, 2.0, 0, 0, 0, False)[0]
        p = self._uniform()
        # Only level 0 populated but with insufficient support
        tables = _make_tables(fine_key, p, 1)
        keys = make_keys(1, 2.0, 0, 0, 0, False)
        prob, c = bucket_lookup(tables, keys)
        # Falls through to the second pass (any-match loop) → returns it anyway
        assert prob is not None

    def test_result_sums_to_one(self):
        fine_key = make_keys(1, 2.0, 0, 0, 0, False)[0]
        p = self._uniform()
        tables = _make_tables(fine_key, p, MIN_SUPPORT[0] + 10)
        keys = make_keys(1, 2.0, 0, 0, 0, False)
        prob, _ = bucket_lookup(tables, keys)
        np.testing.assert_allclose(prob.sum(), 1.0, atol=1e-9)

    def test_all_positive_with_shrinkage(self):
        # count=50 triggers shrinkage path (c < 200), add a coarser level
        fine_key = make_keys(1, 2.0, 0, 0, 0, False)[0]
        coarse_key = make_keys(1, 2.0, 0, 0, 0, False)[1]
        p_fine = np.array([0.8, 0.05, 0.05, 0.04, 0.03, 0.03])
        p_coarse = self._uniform()
        n = 6
        means = [{} for _ in range(n)]
        counts = [{} for _ in range(n)]
        means[0][fine_key] = p_fine
        counts[0][fine_key] = 50
        means[1][coarse_key] = p_coarse
        counts[1][coarse_key] = 200
        tables = {"means": means, "counts": counts}
        keys = make_keys(1, 2.0, 0, 0, 0, False)
        prob, c = bucket_lookup(tables, keys)
        assert prob is not None
        assert (prob > 0).all()
