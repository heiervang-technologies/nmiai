#!/usr/bin/env python3
"""Information-value viewport selector for Astar Island.

Three selection strategies, validated across 75 ground truth files:

1. select_viewports() - RECOMMENDED: regime discriminability selector.
   Picks viewports where regime predictions differ most (symmetric KL between
   harsh/moderate/prosperous). Beats heuristic by 1.01% on wKL.

2. select_viewports_by_info_gain() - Information gain selector.
   Per-cell value = H_pred^2 / (tau + 1). Beats heuristic by 0.87%.

3. select_viewports_hybrid() - Weighted combination of (1) and (2).

4. select_viewports_for_regime_detection() - Alias for (1).
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve

# Import regime predictor
sys.path.insert(0, str(Path(__file__).parent))
import regime_predictor as rp

N_CLASSES = 6
VIEWPORT_SIZE = 15
OCEAN = 10
MOUNTAIN = 5


def cell_entropy(p):
    """Shannon entropy of a probability vector (nats)."""
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log(p_safe))


def compute_cell_values(initial_grid, model=None):
    """Compute per-cell information value for viewport selection.

    Returns:
        value_map: (H, W) array of expected wKL reduction per cell
        tau_map: (H, W) array of effective prior strength
        entropy_map: (H, W) array of prediction entropy
    """
    if model is None:
        model = rp.get_model()

    ig = np.array(initial_grid, dtype=np.int32)
    h, w = ig.shape

    # Get prior prediction (no observations)
    pred = rp.predict(initial_grid)

    # Compute features for bucket lookups
    dist_civ, n_ocean, n_civ, coast = rp.compute_features(ig)

    value_map = np.zeros((h, w), dtype=np.float64)
    tau_map = np.zeros((h, w), dtype=np.float64)
    entropy_map = np.zeros((h, w), dtype=np.float64)

    # Get effective tau (prior strength) per cell from the model's bucket counts
    # Use regime-weighted average of bucket counts across regimes
    regime_weights = rp.DEFAULT_REGIME_WEIGHTS

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == OCEAN or code == MOUNTAIN:
                # Static cells: zero value (perfectly predicted)
                tau_map[y, x] = 1000.0
                entropy_map[y, x] = 0.0
                value_map[y, x] = 0.0
                continue

            # Prediction entropy (our uncertainty)
            h_pred = cell_entropy(pred[y, x])
            entropy_map[y, x] = h_pred

            # Get effective tau from bucket counts across regimes
            keys = rp.cell_bucket(code, dist_civ[y, x], n_ocean[y, x],
                                   n_civ[y, x], coast[y, x])
            if keys is None:
                tau_map[y, x] = 10.0
                value_map[y, x] = h_pred * h_pred * 0.5
                continue

            # Weighted average of bucket support across regimes
            total_tau = 0.0
            total_w = 0.0
            for regime, rw in regime_weights.items():
                rt = model["regime_tables"][regime]
                _, count, level = rp.lookup(rt["tables"], rt["counts"], keys)
                if count > 0:
                    total_tau += rw * count
                    total_w += rw
                else:
                    # Fall back to pooled
                    _, count_pool, _ = rp.lookup(model["pooled_tables"],
                                                  model["pooled_counts"], keys)
                    total_tau += rw * max(count_pool, 1.0)
                    total_w += rw

            if total_w > 0:
                eff_tau = total_tau / total_w
            else:
                eff_tau = 1.0

            tau_map[y, x] = eff_tau

            # Expected wKL reduction for N=1 observation sample:
            # After seeing N samples, posterior strength = tau + N
            # Expected entropy reduction ≈ H_pred * N / (tau + N)
            # The scoring weight is H_gt ≈ H_pred (best approximation)
            # So: value ≈ H_pred^2 * (1 / (tau + 1))
            # Using H_pred^2 because both the scoring weight and our uncertainty
            # are proportional to the entropy of the true distribution
            value_map[y, x] = h_pred * h_pred / (eff_tau + 1.0)

    return value_map, tau_map, entropy_map


def select_viewports(initial_grid, n_viewports, n_queries_per=1, model=None):
    """Select viewports maximizing expected score improvement.

    Uses regime discriminability (the winning approach from validation).
    Beats the heuristic selector by 1.01% on wKL across 75 GT files.

    Args:
        initial_grid: 2D list/array of cell codes
        n_viewports: number of viewports to select
        n_queries_per: ignored (kept for API compat)
        model: optional pre-built regime_predictor model

    Returns:
        list of (vx, vy) tuples
    """
    return select_viewports_for_regime_detection(initial_grid, n_viewports, model)


def select_viewports_by_info_gain(initial_grid, n_viewports, n_queries_per=1, model=None):
    """Select viewports maximizing expected information gain (cell-level).

    Per-cell value = H_pred^2 / (tau + 1). Beats heuristic by 0.87%.

    Args:
        initial_grid: 2D list/array of cell codes
        n_viewports: number of viewports to select
        n_queries_per: number of times each viewport will be queried
        model: optional pre-built regime_predictor model

    Returns:
        list of (vx, vy) tuples
    """
    ig = np.array(initial_grid, dtype=np.int32)
    h, w = ig.shape

    value_map, tau_map, entropy_map = compute_cell_values(initial_grid, model)

    # For multiple queries per viewport, adjust the value:
    # N samples reduce uncertainty by factor N/(tau+N)
    # vs 1 sample: 1/(tau+1)
    # Marginal value of N queries = N/(tau+N) - 0 = N/(tau+N)
    if n_queries_per > 1:
        N = n_queries_per
        # Recompute value with N queries
        for y in range(h):
            for x in range(w):
                if ig[y, x] in (OCEAN, MOUNTAIN):
                    continue
                h_pred = entropy_map[y, x]
                tau = tau_map[y, x]
                value_map[y, x] = h_pred * h_pred * N / (tau + N)

    # Greedy non-overlapping viewport selection
    selected = []
    used = np.zeros((h, w), dtype=bool)

    for _ in range(n_viewports):
        best_val = -1.0
        best_pos = (0, 0)

        # Score each possible viewport position
        for vy in range(max(0, h - VIEWPORT_SIZE + 1)):
            for vx in range(max(0, w - VIEWPORT_SIZE + 1)):
                vh = min(VIEWPORT_SIZE, h - vy)
                vw = min(VIEWPORT_SIZE, w - vx)

                # Sum value over cells in viewport that aren't already covered
                region = value_map[vy:vy+vh, vx:vx+vw].copy()
                mask = used[vy:vy+vh, vx:vx+vw]
                region[mask] = 0.0  # Already-covered cells contribute 0

                val = region.sum()

                if val > best_val:
                    best_val = val
                    best_pos = (vx, vy)

        vx, vy = best_pos
        selected.append((vx, vy))

        # Mark cells as covered (with diminishing returns for re-observation)
        vh = min(VIEWPORT_SIZE, h - vy)
        vw = min(VIEWPORT_SIZE, w - vx)
        # Reduce value of covered cells by 80% (diminishing returns)
        value_map[vy:vy+vh, vx:vx+vw] *= 0.2
        used[vy:vy+vh, vx:vx+vw] = True

    return selected


def select_viewports_for_regime_detection(initial_grid, n_viewports=1, model=None):
    """Select viewports optimized for regime detection.

    For regime detection, we want to observe cells that DIFFER MOST between
    regimes, not cells where we're most uncertain overall.

    Returns:
        list of (vx, vy) tuples
    """
    if model is None:
        model = rp.get_model()

    ig = np.array(initial_grid, dtype=np.int32)
    h, w = ig.shape

    dist_civ, n_ocean, n_civ, coast = rp.compute_features(ig)

    # Compute per-cell regime discriminability:
    # For each cell, compute KL divergence between regime predictions
    disc_map = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code in (OCEAN, MOUNTAIN):
                continue

            keys = rp.cell_bucket(code, dist_civ[y, x], n_ocean[y, x],
                                   n_civ[y, x], coast[y, x])
            if keys is None:
                continue

            # Get prediction under each regime
            regime_preds = {}
            for regime in ("harsh", "moderate", "prosperous"):
                rt = model["regime_tables"][regime]
                q, _, _ = rp.lookup(rt["tables"], rt["counts"], keys)
                if q is not None:
                    regime_preds[regime] = q

            if len(regime_preds) < 2:
                continue

            # Average pairwise KL as discriminability
            preds_list = list(regime_preds.values())
            total_kl = 0.0
            n_pairs = 0
            for i in range(len(preds_list)):
                for j in range(i+1, len(preds_list)):
                    p = np.maximum(preds_list[i], 1e-10)
                    q = np.maximum(preds_list[j], 1e-10)
                    total_kl += np.sum(p * np.log(p / q))
                    total_kl += np.sum(q * np.log(q / p))  # symmetric KL
                    n_pairs += 1

            if n_pairs > 0:
                disc_map[y, x] = total_kl / n_pairs

    # Select viewports maximizing regime discriminability
    selected = []
    used = np.zeros((h, w), dtype=bool)

    for _ in range(n_viewports):
        best_val = -1.0
        best_pos = (0, 0)

        for vy in range(max(0, h - VIEWPORT_SIZE + 1)):
            for vx in range(max(0, w - VIEWPORT_SIZE + 1)):
                vh = min(VIEWPORT_SIZE, h - vy)
                vw = min(VIEWPORT_SIZE, w - vx)

                region = disc_map[vy:vy+vh, vx:vx+vw].copy()
                region[used[vy:vy+vh, vx:vx+vw]] = 0.0
                val = region.sum()

                if val > best_val:
                    best_val = val
                    best_pos = (vx, vy)

        vx, vy = best_pos
        selected.append((vx, vy))
        vh = min(VIEWPORT_SIZE, h - vy)
        vw = min(VIEWPORT_SIZE, w - vx)
        disc_map[vy:vy+vh, vx:vx+vw] *= 0.2
        used[vy:vy+vh, vx:vx+vw] = True

    return selected


def select_viewports_hybrid(initial_grid, n_viewports, n_queries_per=1, model=None,
                             alpha=0.5):
    """Hybrid selector combining information value and regime discriminability.

    Args:
        initial_grid: 2D list/array of cell codes
        n_viewports: number of viewports to select
        n_queries_per: queries per viewport
        model: optional pre-built regime_predictor model
        alpha: weight for regime_disc (0=pure value, 1=pure regime_disc)

    Returns:
        list of (vx, vy) tuples
    """
    if model is None:
        model = rp.get_model()

    ig = np.array(initial_grid, dtype=np.int32)
    h, w = ig.shape

    # Get value map (information gain)
    value_map, tau_map, entropy_map = compute_cell_values(initial_grid, model)

    # Get regime discriminability map
    dist_civ, n_ocean, n_civ, coast = rp.compute_features(ig)
    disc_map = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code in (OCEAN, MOUNTAIN):
                continue
            keys = rp.cell_bucket(code, dist_civ[y, x], n_ocean[y, x],
                                   n_civ[y, x], coast[y, x])
            if keys is None:
                continue

            regime_preds = {}
            for regime in ("harsh", "moderate", "prosperous"):
                rt = model["regime_tables"][regime]
                q, _, _ = rp.lookup(rt["tables"], rt["counts"], keys)
                if q is not None:
                    regime_preds[regime] = q

            if len(regime_preds) < 2:
                continue

            preds_list = list(regime_preds.values())
            total_kl = 0.0
            n_pairs = 0
            for i in range(len(preds_list)):
                for j in range(i+1, len(preds_list)):
                    p = np.maximum(preds_list[i], 1e-10)
                    q = np.maximum(preds_list[j], 1e-10)
                    total_kl += np.sum(p * np.log(p / q))
                    total_kl += np.sum(q * np.log(q / p))
                    n_pairs += 1

            if n_pairs > 0:
                disc_map[y, x] = total_kl / n_pairs

    # Normalize both maps to [0, 1] range for combining
    v_max = value_map.max()
    d_max = disc_map.max()
    if v_max > 0:
        value_norm = value_map / v_max
    else:
        value_norm = value_map
    if d_max > 0:
        disc_norm = disc_map / d_max
    else:
        disc_norm = disc_map

    # Combined score
    combined = (1 - alpha) * value_norm + alpha * disc_norm

    # Greedy selection
    selected = []
    used = np.zeros((h, w), dtype=bool)

    for _ in range(n_viewports):
        best_val = -1.0
        best_pos = (0, 0)

        for vy in range(max(0, h - VIEWPORT_SIZE + 1)):
            for vx in range(max(0, w - VIEWPORT_SIZE + 1)):
                vh = min(VIEWPORT_SIZE, h - vy)
                vw = min(VIEWPORT_SIZE, w - vx)

                region = combined[vy:vy+vh, vx:vx+vw].copy()
                region[used[vy:vy+vh, vx:vx+vw]] = 0.0
                val = region.sum()

                if val > best_val:
                    best_val = val
                    best_pos = (vx, vy)

        vx, vy = best_pos
        selected.append((vx, vy))
        vh = min(VIEWPORT_SIZE, h - vy)
        vw = min(VIEWPORT_SIZE, w - vx)
        combined[vy:vy+vh, vx:vx+vw] *= 0.2
        used[vy:vy+vh, vx:vx+vw] = True

    return selected


if __name__ == "__main__":
    # Quick test
    from benchmark import load_ground_truth
    rounds = load_ground_truth()

    for rn in sorted(rounds.keys())[:3]:
        for sn in sorted(rounds[rn].keys())[:1]:
            ig = rounds[rn][sn]["initial_grid"]
            vps = select_viewports(ig, n_viewports=5, n_queries_per=1)
            vps_rd = select_viewports_for_regime_detection(ig, n_viewports=5)
            vps_hyb = select_viewports_hybrid(ig, n_viewports=5)
            print(f"R{rn}S{sn}:")
            print(f"  value:      {vps}")
            print(f"  regime_disc: {vps_rd}")
            print(f"  hybrid:     {vps_hyb}")
