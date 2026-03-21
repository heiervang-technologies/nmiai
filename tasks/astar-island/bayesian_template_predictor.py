#!/usr/bin/env python3
"""Bayesian per-round template predictor for Astar Island.

Instead of 3 regime buckets (harsh/moderate/prosperous), uses each historical
round as its own template. Weights templates by observation likelihood under
each round's bucket priors.

Honest leave-one-round-out CV: each fold trains WITHOUT the test round.

Features per cell (same as regime_predictor):
  initial_type, dist_to_civ, ocean_adj, n_ocean, n_civ, coast
"""

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
N_CLASSES = 6
FLOOR = 1e-6

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11
EMPTY = 0

KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
VIEWPORT_SIZE = 15

OCEAN_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
MOUNTAIN_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature computation (mirrors regime_predictor)
# ---------------------------------------------------------------------------

def compute_features(ig):
    """Compute per-cell features: dist_to_civ, n_ocean, n_civ, coast."""
    h, w = ig.shape
    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN

    dist_civ = distance_transform_edt(~civ) if civ.any() else np.full((h, w), 99.0)
    n_ocean = convolve(ocean.astype(np.int32), KERNEL, mode="constant")
    n_civ = convolve(civ.astype(np.int32), KERNEL, mode="constant")
    coast = (n_ocean > 0) & ~ocean

    return dist_civ, n_ocean, n_civ, coast


def cell_bucket(code, dist, n_ocean, n_civ, coast):
    """Hierarchical bucket keys: fine -> mid -> coarse -> broad."""
    if code in (OCEAN, MOUNTAIN):
        return None

    d = int(min(max(math.floor(dist), 0), 15))
    no = min(int(n_ocean), 4)
    nc = min(int(n_civ), 4)
    c = 1 if coast else 0

    if code == SETTLEMENT:
        t = "S"
    elif code == PORT:
        t = "P"
    elif code == FOREST:
        t = "F"
    else:
        t = "L"

    fine = (t, d, no, nc, c)
    mid = (t, d, no, c)
    coarse = (t, d, c)
    broad = (t, min(d, 8))
    return fine, mid, coarse, broad


def cell_code_to_class(cell):
    if cell in (0, 10, 11):
        return 0
    if cell == 1:
        return 1
    if cell == 2:
        return 2
    if cell == 3:
        return 3
    if cell == 4:
        return 4
    if cell == 5:
        return 5
    return 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth():
    """Load all GT, grouped by round number."""
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(f.stem.split("_")[0].replace("round", ""))
        sn = int(f.stem.split("_")[1].replace("seed", ""))
        data = json.loads(f.read_text())
        if "ground_truth" not in data or "initial_grid" not in data:
            continue
        rounds.setdefault(rn, {})[sn] = {
            "initial_grid": np.array(data["initial_grid"], dtype=np.int32),
            "ground_truth": np.array(data["ground_truth"], dtype=np.float64),
        }
    return rounds


# ---------------------------------------------------------------------------
# Per-round bucket tables
# ---------------------------------------------------------------------------

def build_round_bucket_tables(rounds, round_nums):
    """Build per-round and pooled bucket tables from specified rounds.

    Returns:
        per_round_tables: {rn: {level: {key: avg_prob_vec}}}
        per_round_counts: {rn: {level: {key: count}}}
        pooled_tables: {level: {key: avg_prob_vec}}
        pooled_counts: {level: {key: count}}
    """
    per_round_sums = {}
    per_round_counts = {}
    pooled_sums = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(4)]
    pooled_counts = [defaultdict(float) for _ in range(4)]

    for rn in round_nums:
        rn_sums = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(4)]
        rn_counts = [defaultdict(float) for _ in range(4)]

        for sn, sd in rounds[rn].items():
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]
            dist_civ, n_ocean, n_civ, coast = compute_features(ig)
            h, w = ig.shape

            for y in range(h):
                for x in range(w):
                    code = int(ig[y, x])
                    keys = cell_bucket(code, dist_civ[y, x], n_ocean[y, x],
                                       n_civ[y, x], coast[y, x])
                    if keys is None:
                        continue
                    prob = gt[y, x]
                    for level, key in enumerate(keys):
                        rn_sums[level][key] = rn_sums[level][key] + prob
                        rn_counts[level][key] += 1.0
                        pooled_sums[level][key] = pooled_sums[level][key] + prob
                        pooled_counts[level][key] += 1.0

        # Finalize per-round
        per_round_sums[rn] = rn_sums
        per_round_counts[rn] = rn_counts

    # Convert sums to averages
    per_round_tables = {}
    per_round_counts_final = {}
    for rn in round_nums:
        tables = [{} for _ in range(4)]
        counts = [{} for _ in range(4)]
        for level in range(4):
            for key, vec in per_round_sums[rn][level].items():
                c = per_round_counts[rn][level][key]
                if c > 0:
                    tables[level][key] = vec / c
                counts[level][key] = c
        per_round_tables[rn] = tables
        per_round_counts_final[rn] = counts

    pooled_tables_final = [{} for _ in range(4)]
    pooled_counts_final = [{} for _ in range(4)]
    for level in range(4):
        for key, vec in pooled_sums[level].items():
            c = pooled_counts[level][key]
            if c > 0:
                pooled_tables_final[level][key] = vec / c
            pooled_counts_final[level][key] = c

    return per_round_tables, per_round_counts_final, pooled_tables_final, pooled_counts_final


# ---------------------------------------------------------------------------
# Hierarchical lookup with interpolation
# ---------------------------------------------------------------------------

def lookup(tables, counts, keys, min_counts=(5, 8, 10, 1)):
    """Hierarchical lookup with support thresholds."""
    for level, key in enumerate(keys):
        if key in tables[level]:
            count = counts[level].get(key, 0)
            if count >= min_counts[level]:
                return tables[level][key], count, level
    # Fallback: any match
    for level, key in enumerate(keys):
        if key in tables[level]:
            return tables[level][key], counts[level].get(key, 0), level
    return None, 0, 3


def interpolate_lookup(tables, counts, ig, dist_civ, n_ocean, n_civ, coast, y, x):
    """Linear interpolation between integer distances."""
    code = int(ig[y, x])
    d = float(dist_civ[y, x])
    lo = int(min(max(math.floor(d), 0), 15))
    hi = min(lo + 1, 15)
    frac = float(np.clip(d - lo, 0, 1))

    keys_lo = cell_bucket(code, lo, n_ocean[y, x], n_civ[y, x], coast[y, x])
    if keys_lo is None:
        return None, 0

    q_lo, s_lo, _ = lookup(tables, counts, keys_lo)

    if hi == lo or frac < 0.01:
        return q_lo, s_lo

    keys_hi = cell_bucket(code, hi, n_ocean[y, x], n_civ[y, x], coast[y, x])
    q_hi, s_hi, _ = lookup(tables, counts, keys_hi)

    if q_lo is None:
        return q_hi, s_hi
    if q_hi is None:
        return q_lo, s_lo

    q = (1 - frac) * q_lo + frac * q_hi
    q = np.maximum(q, 1e-12)
    q /= q.sum()
    return q, (1 - frac) * s_lo + frac * s_hi


# ---------------------------------------------------------------------------
# Per-round template prediction
# ---------------------------------------------------------------------------

def predict_single_template(ig, dist_civ, n_ocean, n_civ, coast,
                            round_tables, round_counts,
                            pooled_tables, pooled_counts):
    """Generate prediction grid using a single round's template, falling back to pooled."""
    h, w = ig.shape
    pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == OCEAN:
                pred[y, x] = OCEAN_DIST
                continue
            if code == MOUNTAIN:
                pred[y, x] = MOUNTAIN_DIST
                continue

            # Try round-specific lookup
            q_round, s_round = interpolate_lookup(
                round_tables, round_counts, ig, dist_civ, n_ocean, n_civ, coast, y, x
            )
            # Pooled fallback
            q_pooled, s_pooled = interpolate_lookup(
                pooled_tables, pooled_counts, ig, dist_civ, n_ocean, n_civ, coast, y, x
            )

            if q_round is not None and q_pooled is not None:
                # Shrink toward pooled based on support
                alpha = s_round / (s_round + 3.0)  # shrinkage parameter
                pred[y, x] = alpha * q_round + (1 - alpha) * q_pooled
            elif q_round is not None:
                pred[y, x] = q_round
            elif q_pooled is not None:
                pred[y, x] = q_pooled
            else:
                pred[y, x] = np.ones(N_CLASSES) / N_CLASSES

    return pred


# ---------------------------------------------------------------------------
# Observation simulation (same as benchmark_experiments)
# ---------------------------------------------------------------------------

def simulate_observations(gt, ig, n_viewports, rng):
    """Simulate observations by picking high-entropy viewports and sampling from GT."""
    h, w, _ = gt.shape
    observations = []

    from scipy.ndimage import distance_transform_cdt
    civ = (ig == SETTLEMENT) | (ig == PORT)
    dist = distance_transform_cdt(~civ, metric="taxicab") if civ.any() else np.full((h, w), 20)

    # Score each viewport position by entropy of dynamic cells
    scores = np.zeros((max(h - VIEWPORT_SIZE + 1, 1), max(w - VIEWPORT_SIZE + 1, 1)))
    for vy in range(h - VIEWPORT_SIZE + 1):
        for vx in range(w - VIEWPORT_SIZE + 1):
            patch = gt[vy:vy + VIEWPORT_SIZE, vx:vx + VIEWPORT_SIZE]
            ig_patch = ig[vy:vy + VIEWPORT_SIZE, vx:vx + VIEWPORT_SIZE]
            dynamic = (ig_patch != OCEAN) & (ig_patch != MOUNTAIN)
            if dynamic.any():
                ent = -np.sum(np.maximum(patch, 1e-10) * np.log2(np.maximum(patch, 1e-10)), axis=2)
                scores[vy, vx] = ent[dynamic].sum()

    used = set()
    for _ in range(n_viewports):
        flat = scores.ravel()
        sorted_idx = np.argsort(-flat)
        best_pos = (0, 0)
        for idx in sorted_idx:
            vy = idx // scores.shape[1]
            vx = idx % scores.shape[1]
            too_close = any(abs(vy - uvy) < 8 and abs(vx - uvx) < 8 for uvy, uvx in used)
            if not too_close:
                best_pos = (vy, vx)
                break
        vy, vx = best_pos
        used.add((vy, vx))

        grid_sample = []
        for dy in range(VIEWPORT_SIZE):
            row = []
            for dx in range(VIEWPORT_SIZE):
                y2, x2 = vy + dy, vx + dx
                if 0 <= y2 < h and 0 <= x2 < w:
                    probs = gt[y2, x2]
                    cls = rng.choice(6, p=probs)
                    row.append(cls)
                else:
                    row.append(0)
            grid_sample.append(row)

        observations.append({
            "grid": grid_sample,
            "viewport_x": int(vx),
            "viewport_y": int(vy),
        })

    return observations


# ---------------------------------------------------------------------------
# Bayesian template weighting
# ---------------------------------------------------------------------------

def compute_template_log_likelihoods(ig, observations, per_round_tables,
                                     per_round_counts, train_rounds):
    """Compute log P(observations | template_round) for each template round."""
    dist_civ, n_ocean, n_civ, coast = compute_features(ig)
    h, w = ig.shape
    LOG_FLOOR = -6.0

    log_liks = {}
    for rn in train_rounds:
        log_lik = 0.0
        for obs in observations:
            grid = obs.get("grid", [])
            vx = int(obs.get("viewport_x", 0))
            vy = int(obs.get("viewport_y", 0))
            for dy, row in enumerate(grid):
                for dx, cell_val in enumerate(row):
                    y = vy + dy
                    x = vx + dx
                    if not (0 <= y < h and 0 <= x < w):
                        continue
                    code = int(ig[y, x])
                    if code in (OCEAN, MOUNTAIN):
                        continue

                    q, _ = interpolate_lookup(
                        per_round_tables[rn], per_round_counts[rn],
                        ig, dist_civ, n_ocean, n_civ, coast, y, x
                    )
                    if q is None:
                        continue

                    cls = cell_code_to_class(int(cell_val))
                    prob = max(float(q[cls]), 1e-6)
                    log_lik += max(np.log(prob), LOG_FLOOR)

        log_liks[rn] = log_lik

    return log_liks


def posterior_weights(log_liks, train_rounds):
    """Compute posterior weights from log-likelihoods with uniform prior."""
    log_vals = np.array([log_liks[rn] for rn in train_rounds])
    # Uniform prior
    log_vals -= log_vals.max()
    weights = np.exp(log_vals)
    total = weights.sum()
    if total > 0 and np.isfinite(total):
        weights /= total
    else:
        weights = np.ones(len(train_rounds)) / len(train_rounds)
    return {rn: float(w) for rn, w in zip(train_rounds, weights)}


# ---------------------------------------------------------------------------
# Prediction with Bayesian template mixture
# ---------------------------------------------------------------------------

def predict_bayesian_mixture(ig, per_round_tables, per_round_counts,
                             pooled_tables, pooled_counts,
                             train_rounds, observations=None):
    """Predict using Bayesian weighted mixture of per-round templates.

    Without observations: uniform mixture over all template rounds.
    With observations: posterior-weighted mixture.
    """
    h, w = ig.shape
    dist_civ, n_ocean, n_civ, coast = compute_features(ig)

    # Build per-round predictions
    template_preds = {}
    for rn in train_rounds:
        template_preds[rn] = predict_single_template(
            ig, dist_civ, n_ocean, n_civ, coast,
            per_round_tables[rn], per_round_counts[rn],
            pooled_tables, pooled_counts
        )

    # Compute weights
    if observations:
        log_liks = compute_template_log_likelihoods(
            ig, observations, per_round_tables, per_round_counts, train_rounds
        )
        weights = posterior_weights(log_liks, train_rounds)
    else:
        weights = {rn: 1.0 / len(train_rounds) for rn in train_rounds}

    # Weighted mixture prediction
    pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)
    for rn in train_rounds:
        pred += weights[rn] * template_preds[rn]

    # Apply tau=20 overlay on cells with 2+ observations
    if observations:
        obs_counts = np.zeros((h, w, N_CLASSES), dtype=np.float64)
        obs_total = np.zeros((h, w), dtype=np.float64)
        for obs in observations:
            grid = obs.get("grid", [])
            vx = int(obs.get("viewport_x", 0))
            vy = int(obs.get("viewport_y", 0))
            for dy, row in enumerate(grid):
                for dx, cell_val in enumerate(row):
                    y = vy + dy
                    x = vx + dx
                    if 0 <= y < h and 0 <= x < w:
                        cls = cell_code_to_class(int(cell_val))
                        obs_counts[y, x, cls] += 1.0
                        obs_total[y, x] += 1.0

        # Bayesian update with tau=20 pseudo-count on cells with 2+ observations
        tau = 20.0
        mask = obs_total >= 2.0
        if mask.any():
            posterior_num = obs_counts[mask] + tau * pred[mask]
            posterior_denom = obs_total[mask, None] + tau
            pred[mask] = posterior_num / posterior_denom

    # Structural constraints
    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code in (OCEAN, MOUNTAIN):
                continue
            if not coast[y, x]:
                pred[y, x, 2] = 0.0  # No port if not coastal
            pred[y, x, 5] = 0.0  # No mountain on non-mountain cells
            if dist_civ[y, x] > 10:
                pred[y, x, 3] = 0.0  # No ruin far from settlements

    # Floor and renormalize
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred, weights


# ---------------------------------------------------------------------------
# KL divergence and entropy (matching benchmark.py)
# ---------------------------------------------------------------------------

def kl_divergence(p, q):
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def entropy(p):
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


# ---------------------------------------------------------------------------
# Honest leave-one-round-out CV
# ---------------------------------------------------------------------------

def run_cv(n_obs_viewports=3, n_obs_queries=3):
    """Run honest LOOCV.

    For each held-out round:
    1. Build bucket tables from 14 OTHER rounds
    2. For each seed: predict with and without observations
    3. Report per-round wKL
    """
    rng = np.random.RandomState(42)
    rounds = load_ground_truth()
    all_round_nums = sorted(rounds.keys())

    print(f"Rounds available: {all_round_nums} ({len(all_round_nums)} total)")
    print(f"Observation config: {n_obs_viewports} viewports x {n_obs_queries} queries = {n_obs_viewports * n_obs_queries} total obs cells")
    print()

    results_no_obs = []
    results_with_obs = []

    for held_out in all_round_nums:
        train_rounds = [rn for rn in all_round_nums if rn != held_out]

        t0 = time.time()
        per_round_tables, per_round_counts, pooled_tables, pooled_counts = \
            build_round_bucket_tables(rounds, train_rounds)
        build_time = time.time() - t0

        wkls_no_obs = []
        wkls_with_obs = []

        for sn in sorted(rounds[held_out].keys()):
            sd = rounds[held_out][sn]
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]

            # No observations
            pred_no, _ = predict_bayesian_mixture(
                ig, per_round_tables, per_round_counts,
                pooled_tables, pooled_counts, train_rounds,
                observations=None
            )
            kl = kl_divergence(gt, pred_no)
            H = entropy(gt)
            wkl = H * kl
            dynamic = H > 0.01
            if dynamic.any():
                wkls_no_obs.append(float(wkl[dynamic].mean()))

            # With observations: simulate n_obs_viewports viewports
            obs = simulate_observations(gt, ig, n_obs_viewports, rng)
            pred_obs, weights = predict_bayesian_mixture(
                ig, per_round_tables, per_round_counts,
                pooled_tables, pooled_counts, train_rounds,
                observations=obs
            )
            kl_obs = kl_divergence(gt, pred_obs)
            wkl_obs = H * kl_obs
            if dynamic.any():
                wkls_with_obs.append(float(wkl_obs[dynamic].mean()))

        no_obs_mean = np.mean(wkls_no_obs)
        obs_mean = np.mean(wkls_with_obs)
        delta = obs_mean - no_obs_mean
        marker = "WORSE" if delta > 0 else "BETTER"
        results_no_obs.append(no_obs_mean)
        results_with_obs.append(obs_mean)

        # Show top-3 template weights for last seed
        top3 = sorted(weights.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"R{rn}:{w:.2f}" for rn, w in top3)

        print(f"  R{held_out:2d}: no_obs={no_obs_mean:.6f}  obs={obs_mean:.6f}  "
              f"delta={delta:+.6f} [{marker}]  top_weights=[{top3_str}]  "
              f"build={build_time:.1f}s")

    mean_no_obs = np.mean(results_no_obs)
    mean_obs = np.mean(results_with_obs)

    print(f"\n{'='*70}")
    print(f"MEAN no_obs:  {mean_no_obs:.6f}")
    print(f"MEAN with_obs: {mean_obs:.6f}")
    print(f"MEAN delta:   {mean_obs - mean_no_obs:+.6f}")
    improvement = (mean_no_obs - mean_obs) / mean_no_obs * 100
    print(f"Improvement:  {improvement:+.1f}%")

    return {
        "per_round_no_obs": {rn: v for rn, v in zip(all_round_nums, results_no_obs)},
        "per_round_obs": {rn: v for rn, v in zip(all_round_nums, results_with_obs)},
        "mean_no_obs": mean_no_obs,
        "mean_obs": mean_obs,
    }


# ---------------------------------------------------------------------------
# Production predict function
# ---------------------------------------------------------------------------

_CACHED_MODEL = None


def get_or_build_model():
    """Build model from all available GT data."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        rounds = load_ground_truth()
        all_rounds = sorted(rounds.keys())
        per_round_tables, per_round_counts, pooled_tables, pooled_counts = \
            build_round_bucket_tables(rounds, all_rounds)
        _CACHED_MODEL = {
            "per_round_tables": per_round_tables,
            "per_round_counts": per_round_counts,
            "pooled_tables": pooled_tables,
            "pooled_counts": pooled_counts,
            "train_rounds": all_rounds,
        }
    return _CACHED_MODEL


def predict(initial_grid, observations=None):
    """Public API: predict using Bayesian template mixture."""
    model = get_or_build_model()
    ig = np.array(initial_grid, dtype=np.int32)
    pred, _ = predict_bayesian_mixture(
        ig,
        model["per_round_tables"],
        model["per_round_counts"],
        model["pooled_tables"],
        model["pooled_counts"],
        model["train_rounds"],
        observations=observations,
    )
    return pred


if __name__ == "__main__":
    results = run_cv(n_obs_viewports=3, n_obs_queries=3)
