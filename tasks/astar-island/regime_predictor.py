#!/usr/bin/env python3
"""Regime-aware predictor for Astar Island.

Builds 3 regime-specific priors (harsh/moderate/prosperous) from historical
data. For new rounds:
- Without observations: uses a weighted mixture of all 3 regime priors
- With observations: classifies regime from settlement counts in observed
  cells, then uses the matching regime prior

The regime is determined by settlement growth rate on frontier cells:
  harsh:      < 0.05 (rounds 3, 8, 10)
  moderate:   0.05 - 0.20 (rounds 1, 4, 5, 7, 9, 12, 13)
  prosperous: > 0.20 (rounds 2, 6, 11)
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve

GT_DIR = Path(__file__).parent / "ground_truth"
N_CLASSES = 6
FLOOR = 0.005

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4

KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

# Regime thresholds on frontier settlement rate
HARSH_THRESHOLD = 0.05
PROSPEROUS_THRESHOLD = 0.20

# Default regime prior weights when no observations available
# Weighted toward moderate since it's most common (7/13 rounds)
DEFAULT_REGIME_WEIGHTS = {"harsh": 0.23, "moderate": 0.54, "prosperous": 0.23}

OCEAN_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
MOUNTAIN_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

_MODEL = None


def compute_features(ig):
    """Compute per-cell features for bucketing."""
    h, w = ig.shape
    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN

    if civ.any():
        dist_civ = distance_transform_edt(~civ)
    else:
        dist_civ = np.full((h, w), 99.0)

    n_ocean = convolve(ocean.astype(np.int32), KERNEL, mode="constant")
    n_civ = convolve(civ.astype(np.int32), KERNEL, mode="constant")
    coast = (n_ocean > 0) & ~ocean

    return dist_civ, n_ocean, n_civ, coast


def cell_bucket(code, dist, n_ocean, n_civ, coast):
    """Fine-grained bucket for a cell."""
    if code == OCEAN:
        return None
    if code == MOUNTAIN:
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


def classify_round(round_data):
    """Classify a round's regime from its ground truth."""
    rates = []
    for sn, sd in round_data.items():
        ig = sd["initial_grid"]
        gt = sd["ground_truth"]
        dist_civ, _, _, _ = compute_features(ig)
        frontier = (dist_civ >= 1) & (dist_civ <= 5) & (ig != OCEAN) & (ig != MOUNTAIN)
        if frontier.any():
            rates.append(gt[frontier, 1].mean())
    rate = np.mean(rates) if rates else 0.1
    if rate < HARSH_THRESHOLD:
        return "harsh"
    elif rate > PROSPEROUS_THRESHOLD:
        return "prosperous"
    return "moderate"


def build_model():
    """Build regime-specific lookup tables."""
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(f.stem.split("_")[0].replace("round", ""))
        sn = int(f.stem.split("_")[1].replace("seed", ""))
        with open(f) as fh:
            data = json.load(fh)
        if "ground_truth" not in data or "initial_grid" not in data:
            continue
        if rn not in rounds:
            rounds[rn] = {}
        rounds[rn][sn] = {
            "initial_grid": np.array(data["initial_grid"], dtype=np.int32),
            "ground_truth": np.array(data["ground_truth"], dtype=np.float64),
        }

    round_regimes = {rn: classify_round(rd) for rn, rd in rounds.items()}

    regime_tables = {}
    for regime in ("harsh", "moderate", "prosperous"):
        sums = [defaultdict(lambda: np.zeros(N_CLASSES)) for _ in range(4)]
        counts = [defaultdict(float) for _ in range(4)]

        for rn, rd in rounds.items():
            if round_regimes[rn] != regime:
                continue
            for sn, sd in rd.items():
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
                            sums[level][key] += prob
                            counts[level][key] += 1.0

        tables = []
        for level in range(4):
            t = {}
            for key, vec in sums[level].items():
                c = counts[level][key]
                if c > 0:
                    t[key] = vec / c
            tables.append(t)

        regime_tables[regime] = {"tables": tables, "counts": [dict(c) for c in counts]}

    # Pooled table
    pooled_sums = [defaultdict(lambda: np.zeros(N_CLASSES)) for _ in range(4)]
    pooled_counts = [defaultdict(float) for _ in range(4)]
    for rn, rd in rounds.items():
        for sn, sd in rd.items():
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
                        pooled_sums[level][key] += prob
                        pooled_counts[level][key] += 1.0

    pooled_tables = []
    for level in range(4):
        t = {}
        for key, vec in pooled_sums[level].items():
            c = pooled_counts[level][key]
            if c > 0:
                t[key] = vec / c
        pooled_tables.append(t)

    return {
        "regime_tables": regime_tables,
        "pooled_tables": pooled_tables,
        "pooled_counts": [dict(c) for c in pooled_counts],
        "round_regimes": round_regimes,
    }


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build_model()
    return _MODEL


def lookup(tables, counts_list, keys, min_counts=(5, 8, 10, 1)):
    """Hierarchical lookup with support thresholds."""
    for level, key in enumerate(keys):
        if key in tables[level]:
            count = counts_list[level].get(key, 0)
            if count >= min_counts[level]:
                return tables[level][key], count, level
    for level, key in enumerate(keys):
        if key in tables[level]:
            return tables[level][key], counts_list[level].get(key, 0), level
    return None, 0, 3


def interpolate_lookup(tables, counts_list, ig, dist_civ, n_ocean, n_civ, coast, y, x):
    """Linear interpolation between integer distances."""
    code = int(ig[y, x])
    d = float(dist_civ[y, x])
    lo = int(min(max(math.floor(d), 0), 15))
    hi = min(lo + 1, 15)
    frac = float(np.clip(d - lo, 0, 1))

    no = n_ocean[y, x]
    nc = n_civ[y, x]
    c = coast[y, x]

    keys_lo = cell_bucket(code, lo, no, nc, c)
    if keys_lo is None:
        return None, 0

    q_lo, s_lo, _ = lookup(tables, counts_list, keys_lo)

    if hi == lo or frac < 0.01:
        return q_lo, s_lo

    keys_hi = cell_bucket(code, hi, no, nc, c)
    q_hi, s_hi, _ = lookup(tables, counts_list, keys_hi)

    if q_lo is None:
        return q_hi, s_hi
    if q_hi is None:
        return q_lo, s_lo

    q = (1 - frac) * q_lo + frac * q_hi
    q = np.maximum(q, 1e-12)
    q /= q.sum()
    return q, (1 - frac) * s_lo + frac * s_hi


def cell_code_to_class(cell):
    """Map cell codes to prediction class indices."""
    if cell in (0, 10, 11): return 0
    if cell == 1: return 1
    if cell == 2: return 2
    if cell == 3: return 3
    if cell == 4: return 4
    if cell == 5: return 5
    return 0


def detect_regime_from_observations(ig, observations):
    """Bayesian regime posterior from full observation likelihoods.

    For each regime, compute log P(observations | regime) using the regime's
    bucket priors, then derive posterior weights via Bayes' rule.
    """
    if not observations:
        return DEFAULT_REGIME_WEIGHTS.copy()

    model = get_model()
    dist_civ, n_ocean, n_civ, coast = compute_features(ig)
    h, w = ig.shape

    LOG_FLOOR = -6.0  # floor for log-likelihood per cell

    log_likelihoods = {}
    for regime in ("harsh", "moderate", "prosperous"):
        rt = model["regime_tables"][regime]
        log_lik = 0.0
        n_cells = 0

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

                    # Look up this cell's prior under this regime
                    q, _ = interpolate_lookup(
                        rt["tables"], rt["counts"],
                        ig, dist_civ, n_ocean, n_civ, coast, y, x
                    )
                    if q is None:
                        continue

                    cls = cell_code_to_class(int(cell_val))
                    prob = max(float(q[cls]), 1e-6)
                    log_lik += max(np.log(prob), LOG_FLOOR)
                    n_cells += 1

        log_likelihoods[regime] = log_lik

    if not log_likelihoods:
        return DEFAULT_REGIME_WEIGHTS.copy()

    # Bayesian posterior: P(regime|obs) ∝ P(obs|regime) * P(regime)
    log_prior = {
        "harsh": np.log(DEFAULT_REGIME_WEIGHTS["harsh"]),
        "moderate": np.log(DEFAULT_REGIME_WEIGHTS["moderate"]),
        "prosperous": np.log(DEFAULT_REGIME_WEIGHTS["prosperous"]),
    }

    log_posterior = {r: log_likelihoods[r] + log_prior[r] for r in log_likelihoods}

    # Numerical stability: subtract max
    max_lp = max(log_posterior.values())
    weights = {r: np.exp(log_posterior[r] - max_lp) for r in log_posterior}
    total = sum(weights.values())
    if total > 0:
        weights = {r: w / total for r, w in weights.items()}
    else:
        return DEFAULT_REGIME_WEIGHTS.copy()

    # Floor weights to avoid zero
    for r in weights:
        weights[r] = max(weights[r], 0.01)
    total = sum(weights.values())
    weights = {r: w / total for r, w in weights.items()}

    return weights


def predict(initial_grid, observations=None):
    """Predict using regime-aware priors."""
    model = get_model()
    ig = np.array(initial_grid, dtype=np.int32)
    h, w = ig.shape
    dist_civ, n_ocean, n_civ, coast = compute_features(ig)

    regime_weights = detect_regime_from_observations(ig, observations or [])

    pred = np.zeros((h, w, N_CLASSES))

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == OCEAN:
                pred[y, x] = OCEAN_DIST
                continue
            if code == MOUNTAIN:
                pred[y, x] = MOUNTAIN_DIST
                continue

            cell_pred = np.zeros(N_CLASSES)
            total_w = 0.0

            for regime, rw in regime_weights.items():
                if rw < 0.01:
                    continue
                rt = model["regime_tables"][regime]
                q, support = interpolate_lookup(
                    rt["tables"], rt["counts"], ig, dist_civ, n_ocean, n_civ, coast, y, x
                )
                if q is not None:
                    cell_pred += rw * q
                    total_w += rw
                else:
                    q_pool, _ = interpolate_lookup(
                        model["pooled_tables"], model["pooled_counts"],
                        ig, dist_civ, n_ocean, n_civ, coast, y, x
                    )
                    if q_pool is not None:
                        cell_pred += rw * q_pool
                        total_w += rw

            if total_w > 0:
                pred[y, x] = cell_pred / total_w
            else:
                pred[y, x] = np.ones(N_CLASSES) / N_CLASSES

    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark import load_ground_truth, evaluate_predictor
    model = get_model()
    rounds = load_ground_truth()
    result = evaluate_predictor(predict, rounds)
    print(f"Mean wKL: {result['mean_weighted_kl']:.6f}")
    print(f"Mean KL:  {result['mean_kl']:.6f}")
