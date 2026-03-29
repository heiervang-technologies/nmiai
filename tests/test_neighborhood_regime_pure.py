"""Pure-function tests for neighborhood_predictor.py and regime_predictor.py.

Covers:
  neighborhood_predictor: cell_to_type, dist_bin, extract_features
  regime_predictor: cell_bucket, cell_code_to_class, classify_round, lookup

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# neighborhood_predictor imports requests at module level
sys.modules.setdefault("requests", MagicMock())

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

import neighborhood_predictor as nbr
import regime_predictor as rp


GRID_SIZE = 40
N_CLASSES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_grid(size: int = GRID_SIZE, code: int = 11) -> np.ndarray:
    g = np.full((size, size), code, dtype=np.int32)
    g[5, 5] = 1    # settlement
    g[5, 6] = 2    # port
    g[10, 10] = 4  # forest
    g[0, 0] = 10   # ocean
    g[-1, -1] = 5  # mountain
    return g


# ---------------------------------------------------------------------------
# neighborhood_predictor.cell_to_type
# ---------------------------------------------------------------------------

class TestCellToType:
    @pytest.mark.parametrize("code,expected", [
        (1, "settlement"),
        (2, "port"),
        (4, "forest"),
        (11, "plains"),
        (0, "empty"),
    ])
    def test_known_codes(self, code, expected):
        assert nbr.cell_to_type(code) == expected

    def test_ocean_returns_none(self):
        assert nbr.cell_to_type(10) is None

    def test_mountain_returns_none(self):
        assert nbr.cell_to_type(5) is None

    def test_unknown_returns_none(self):
        assert nbr.cell_to_type(99) is None


# ---------------------------------------------------------------------------
# neighborhood_predictor.dist_bin
# ---------------------------------------------------------------------------

class TestDistBin:
    @pytest.mark.parametrize("d,expected", [
        (0, 0), (1, 0),
        (2, 1), (3, 1),
        (4, 2), (5, 2), (6, 2),
        (7, 3), (100, 3),
    ])
    def test_binning(self, d, expected):
        assert nbr.dist_bin(d) == expected


# ---------------------------------------------------------------------------
# neighborhood_predictor.extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_5_values(self):
        g = _uniform_grid()
        result = nbr.extract_features(g)
        assert len(result) == 5

    def test_shapes(self):
        g = _uniform_grid()
        types, n_settle, n_forest, n_ocean, dist_bins = nbr.extract_features(g)
        for arr in (n_settle, n_forest, n_ocean, dist_bins):
            assert arr.shape == (GRID_SIZE, GRID_SIZE)
        assert types.shape == (GRID_SIZE, GRID_SIZE)

    def test_ocean_type_is_none(self):
        g = _uniform_grid()
        types, *_ = nbr.extract_features(g)
        assert types[0, 0] is None  # ocean cell

    def test_settlement_type_correct(self):
        g = _uniform_grid()
        types, *_ = nbr.extract_features(g)
        assert types[5, 5] == "settlement"

    def test_dist_bins_zero_at_settlement(self):
        g = _uniform_grid()
        _, _, _, _, dist_bins = nbr.extract_features(g)
        assert dist_bins[5, 5] == 0

    def test_n_settle_positive_near_settlement(self):
        g = _uniform_grid()
        _, n_settle, _, _, _ = nbr.extract_features(g)
        # At least one neighbor of (5,5) has n_settle > 0
        assert n_settle[5, 4] > 0 or n_settle[4, 5] > 0

    def test_n_ocean_positive_near_ocean(self):
        g = _uniform_grid()
        _, _, _, n_ocean, _ = nbr.extract_features(g)
        # Cell adjacent to (0,0) should count it
        assert n_ocean[0, 1] > 0 or n_ocean[1, 0] > 0

    def test_no_civ_grid_fills_dist_99(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)  # no civ
        _, _, _, _, dist_bins = nbr.extract_features(g)
        # dist_bin(99) = 3
        assert (dist_bins == 3).all()


# ---------------------------------------------------------------------------
# regime_predictor.cell_bucket
# ---------------------------------------------------------------------------

class TestRpCellBucket:
    def test_ocean_returns_none(self):
        assert rp.cell_bucket(rp.OCEAN, 5.0, 2, 1, True) is None

    def test_mountain_returns_none(self):
        assert rp.cell_bucket(rp.MOUNTAIN, 3.0, 0, 0, False) is None

    def test_returns_4_keys_for_land(self):
        keys = rp.cell_bucket(rp.SETTLEMENT, 3.0, 1, 2, True)
        assert keys is not None and len(keys) == 4

    def test_token_S_for_settlement(self):
        assert rp.cell_bucket(rp.SETTLEMENT, 2.0, 0, 0, False)[0][0] == "S"

    def test_token_P_for_port(self):
        assert rp.cell_bucket(rp.PORT, 2.0, 1, 0, True)[0][0] == "P"

    def test_token_F_for_forest(self):
        assert rp.cell_bucket(4, 5.0, 0, 0, False)[0][0] == "F"

    def test_token_L_for_plains(self):
        assert rp.cell_bucket(11, 5.0, 0, 0, False)[0][0] == "L"

    def test_broad_key_caps_dist_at_8(self):
        keys_8 = rp.cell_bucket(rp.SETTLEMENT, 8.0, 0, 0, False)
        keys_15 = rp.cell_bucket(rp.SETTLEMENT, 15.0, 0, 0, False)
        assert keys_8[3] == keys_15[3]


# ---------------------------------------------------------------------------
# regime_predictor.cell_code_to_class
# ---------------------------------------------------------------------------

class TestRpCellCodeToClass:
    @pytest.mark.parametrize("code,expected", [
        (0, 0), (10, 0), (11, 0),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    ])
    def test_known_codes(self, code, expected):
        assert rp.cell_code_to_class(code) == expected

    def test_unknown_returns_zero(self):
        assert rp.cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# regime_predictor.classify_round
# ---------------------------------------------------------------------------

class TestRpClassifyRound:
    def _make_round_data(self, settle_prob: float, size: int = 10) -> dict:
        ig = np.full((size, size), 11, dtype=np.int32)
        ig[5, 5] = 1
        gt = np.zeros((size, size, N_CLASSES), dtype=np.float64)
        gt[:, :, 0] = 1.0 - settle_prob
        gt[:, :, 1] = settle_prob
        return {0: {"initial_grid": ig, "ground_truth": gt}}

    def test_harsh(self):
        assert rp.classify_round(self._make_round_data(0.01)) == "harsh"

    def test_prosperous(self):
        assert rp.classify_round(self._make_round_data(0.5)) == "prosperous"

    def test_moderate(self):
        assert rp.classify_round(self._make_round_data(0.12)) == "moderate"


# ---------------------------------------------------------------------------
# regime_predictor.lookup
# ---------------------------------------------------------------------------

class TestRpLookup:
    def _tables_counts(self):
        p = np.ones(N_CLASSES) / N_CLASSES
        tables = [
            {("S", 2, 0, 1, 1): p},    # fine
            {("S", 2, 0, 1): p},        # mid
            {("S", 2, 1): p},            # coarse
            {("S", 2): p},               # broad
        ]
        counts = [
            {("S", 2, 0, 1, 1): 10},
            {("S", 2, 0, 1): 5},
            {("S", 2, 1): 2},
            {("S", 2): 1},
        ]
        return tables, counts

    def test_returns_fine_when_sufficient(self):
        tables, counts = self._tables_counts()
        keys = (("S", 2, 0, 1, 1), ("S", 2, 0, 1), ("S", 2, 1), ("S", 2))
        _, _, level = rp.lookup(tables, counts, keys)
        assert level == 0

    def test_skips_insufficient_support(self):
        tables, counts = self._tables_counts()
        counts[0][("S", 2, 0, 1, 1)] = 2  # below fine min_count=5
        counts[1][("S", 2, 0, 1)] = 10     # meets mid min_count=8
        keys = (("S", 2, 0, 1, 1), ("S", 2, 0, 1), ("S", 2, 1), ("S", 2))
        _, _, level = rp.lookup(tables, counts, keys)
        assert level == 1  # fine skipped, mid passes

    def test_no_match_returns_none(self):
        tables = [{}, {}, {}, {}]
        counts = [{}, {}, {}, {}]
        keys = (("X",), ("X",), ("X",), ("X",))
        prob, _, _ = rp.lookup(tables, counts, keys)
        assert prob is None
