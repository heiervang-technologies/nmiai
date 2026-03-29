"""Tests for tasks/astar-island/bayesian_template_predictor.py — pure helpers.

Covers: cell_code_to_class, cell_bucket, posterior_weights.
All pure functions requiring no file I/O or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from bayesian_template_predictor import (
    cell_bucket,
    cell_code_to_class,
    posterior_weights,
)


# ---------------------------------------------------------------------------
# cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    def test_empty_0_maps_to_0(self):
        assert cell_code_to_class(0) == 0

    def test_ocean_10_maps_to_0(self):
        assert cell_code_to_class(10) == 0

    def test_plains_11_maps_to_0(self):
        assert cell_code_to_class(11) == 0

    def test_settlement_1_maps_to_1(self):
        assert cell_code_to_class(1) == 1

    def test_port_2_maps_to_2(self):
        assert cell_code_to_class(2) == 2

    def test_code_3_maps_to_3(self):
        assert cell_code_to_class(3) == 3

    def test_forest_4_maps_to_4(self):
        assert cell_code_to_class(4) == 4

    def test_mountain_5_maps_to_5(self):
        assert cell_code_to_class(5) == 5

    def test_unknown_code_maps_to_0(self):
        assert cell_code_to_class(99) == 0

    def test_returns_int(self):
        assert isinstance(cell_code_to_class(1), int)


# ---------------------------------------------------------------------------
# cell_bucket
# ---------------------------------------------------------------------------

class TestCellBucket:
    def test_ocean_returns_none(self):
        assert cell_bucket(10, 5.0, 2, 1, False) is None

    def test_mountain_returns_none(self):
        assert cell_bucket(5, 3.0, 0, 0, False) is None

    def test_returns_tuple_of_four(self):
        result = cell_bucket(1, 2.0, 1, 0, False)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_settlement_type_is_S(self):
        fine, mid, coarse, broad = cell_bucket(1, 2.0, 0, 0, False)
        assert fine[0] == "S"

    def test_port_type_is_P(self):
        fine, mid, coarse, broad = cell_bucket(2, 2.0, 1, 0, True)
        assert fine[0] == "P"

    def test_forest_type_is_F(self):
        fine, mid, coarse, broad = cell_bucket(4, 5.0, 0, 0, False)
        assert fine[0] == "F"

    def test_plains_type_is_L(self):
        fine, mid, coarse, broad = cell_bucket(0, 5.0, 0, 0, False)
        assert fine[0] == "L"

    def test_empty_type_is_L(self):
        fine, _, _, _ = cell_bucket(11, 3.0, 0, 0, False)
        assert fine[0] == "L"

    def test_fine_bucket_has_five_elements(self):
        fine, mid, coarse, broad = cell_bucket(1, 2.0, 1, 2, True)
        assert len(fine) == 5

    def test_mid_bucket_has_four_elements(self):
        fine, mid, coarse, broad = cell_bucket(1, 2.0, 1, 2, True)
        assert len(mid) == 4

    def test_coarse_bucket_has_three_elements(self):
        fine, mid, coarse, broad = cell_bucket(1, 2.0, 1, 2, True)
        assert len(coarse) == 3

    def test_broad_bucket_has_two_elements(self):
        fine, mid, coarse, broad = cell_bucket(1, 2.0, 1, 2, True)
        assert len(broad) == 2

    def test_coast_flag_reflected_in_fine_bucket(self):
        fine_coast, _, _, _ = cell_bucket(4, 3.0, 2, 0, True)
        fine_inland, _, _, _ = cell_bucket(4, 3.0, 2, 0, False)
        # Coast flag is the last element of fine (index 4)
        assert fine_coast[4] == 1
        assert fine_inland[4] == 0

    def test_high_n_ocean_clamped_to_4(self):
        fine, _, _, _ = cell_bucket(1, 2.0, 10, 0, False)
        assert fine[2] == 4

    def test_high_n_civ_clamped_to_4(self):
        fine, _, _, _ = cell_bucket(1, 2.0, 0, 10, False)
        assert fine[3] == 4

    def test_high_dist_clamped_to_15(self):
        fine, _, _, _ = cell_bucket(1, 100.0, 0, 0, False)
        assert fine[1] == 15


# ---------------------------------------------------------------------------
# posterior_weights
# ---------------------------------------------------------------------------

class TestPosteriorWeights:
    def test_returns_dict(self):
        log_liks = {1: -10.0, 2: -8.0, 3: -12.0}
        result = posterior_weights(log_liks, [1, 2, 3])
        assert isinstance(result, dict)

    def test_keys_match_train_rounds(self):
        log_liks = {1: -5.0, 2: -3.0}
        result = posterior_weights(log_liks, [1, 2])
        assert set(result.keys()) == {1, 2}

    def test_weights_sum_to_one(self):
        log_liks = {1: -10.0, 2: -8.0, 3: -9.0}
        result = posterior_weights(log_liks, [1, 2, 3])
        assert sum(result.values()) == pytest.approx(1.0)

    def test_higher_log_lik_gets_higher_weight(self):
        log_liks = {1: -20.0, 2: -5.0}  # round 2 is much more likely
        result = posterior_weights(log_liks, [1, 2])
        assert result[2] > result[1]

    def test_equal_log_liks_uniform_weights(self):
        log_liks = {1: -5.0, 2: -5.0, 3: -5.0}
        result = posterior_weights(log_liks, [1, 2, 3])
        for w in result.values():
            assert w == pytest.approx(1 / 3)

    def test_weights_are_floats(self):
        log_liks = {1: -3.0, 2: -7.0}
        result = posterior_weights(log_liks, [1, 2])
        for w in result.values():
            assert isinstance(w, float)

    def test_single_round_weight_is_one(self):
        log_liks = {5: -42.0}
        result = posterior_weights(log_liks, [5])
        assert result[5] == pytest.approx(1.0)
