#!/usr/bin/env python3
"""Surrogate Monte Carlo predictor with SPATIAL CORRELATIONS for Astar Island.

Key innovation over independent bucket priors: captures the clustering of
settlement growth.  If one cell becomes a settlement, its neighbor is more
likely to also become one.

Approach:
1. Build hierarchical transition tables from ground truth (like bayesian_template)
2. Build CONDITIONAL transition tables: P(final | initial, features, n_civ_neighbors)
3. Gibbs-like sampling:
   a) Start from base prior predictions
   b) Sample a full 40x40 grid
   c) Re-estimate each cell conditioned on its sampled neighbors
   d) Resample from updated probabilities
   e) Repeat for several Gibbs sweeps
   f) Average 200 runs -> probability tensor

Public API:
    predict(initial_grid, observations=None) -> (40, 40, 6) tensor
    cross_validate(n_sims=200) -> dict with CV results
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

np.random.seed(42)

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
N_CLASSES = 6
FLOOR = 1e-6
N_SIMS = 200
N_SWEEPS = 2
MC_BLEND = 0.65  # blend ratio: 0=pure MC, 1=pure base prior
SPATIAL_WEIGHT = 0.5  # how much weight to spatial conditional vs base prior in Gibbs

# Cell type codes
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5
OCEAN = 10
PLAINS = 11

KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

OCEAN_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
MOUNTAIN_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
UNIFORM = np.ones(N_CLASSES, dtype=np.float64) / N_CLASSES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(prob):
    prob = np.maximum(prob, FLOOR)
    s = prob.sum()
    if not np.isfinite(s) or s <= 0:
        return UNIFORM.copy()
    return prob / s


def cell_code_to_class(cell):
    """Map cell codes to class indices 0-5."""
    mapping = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    return mapping.get(int(cell), 0)


def cell_type_token(code):
    if code == SETTLEMENT:
        return "S"
    if code == PORT:
        return "P"
    if code == FOREST:
        return "F"
    return "L"  # Empty, Plains, Ruin


def quantize_dist(dist):
    return int(min(max(math.floor(float(dist)), 0), 15))


def quantize_neighbor_civ(n):
    """Quantize number of civ neighbors (0-8) into bins."""
    n = int(n)
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n <= 3:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(ig):
    """Compute per-cell features from initial grid."""
    h, w = ig.shape
    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN

    dist_civ = distance_transform_edt(~civ) if civ.any() else np.full((h, w), 99.0)
    n_ocean = convolve(ocean.astype(np.int32), KERNEL, mode="constant")
    coast = (n_ocean > 0) & ~ocean
    n_civ = convolve(civ.astype(np.int32), KERNEL, mode="constant")

    return {
        "dist_civ": dist_civ,
        "n_ocean": n_ocean,
        "coast": coast,
        "n_civ": n_civ,
    }


# ---------------------------------------------------------------------------
# Bucket key generation
# ---------------------------------------------------------------------------

def base_bucket_keys(code, dist, n_ocean, coast):
    """Generate hierarchical bucket keys (fine -> coarse)."""
    if code in (OCEAN, MOUNTAIN):
        return None
    t = cell_type_token(code)
    d = quantize_dist(dist)
    no = min(int(n_ocean), 4)
    c = 1 if coast else 0
    return (
        (t, d, no, c),       # fine
        (t, d, c),            # mid
        (t, min(d, 8), c),    # coarse
        (t, min(d, 4)),       # broadest
    )


def spatial_bucket_keys(code, dist, n_ocean, coast, n_civ_neighbors):
    """Bucket keys conditioned on number of civ neighbors in final state."""
    if code in (OCEAN, MOUNTAIN):
        return None
    t = cell_type_token(code)
    d = quantize_dist(dist)
    no = min(int(n_ocean), 4)
    c = 1 if coast else 0
    nc = quantize_neighbor_civ(n_civ_neighbors)
    return (
        (t, d, no, c, nc),    # fine + spatial
        (t, d, c, nc),         # mid + spatial
        (t, min(d, 8), c, nc), # coarse + spatial
        (t, min(d, 4), nc),    # broadest + spatial
    )


# ---------------------------------------------------------------------------
# Lookup with support thresholds
# ---------------------------------------------------------------------------

def lookup_tables(tables, counts, keys, min_counts=(10, 15, 20, 1)):
    """Hierarchical lookup: return most specific table entry with enough support."""
    for level, key in enumerate(keys):
        if key in tables[level] and counts[level].get(key, 0) >= min_counts[level]:
            return tables[level][key], float(counts[level][key]), level
    # Fallback: any match
    for level, key in enumerate(keys):
        if key in tables[level]:
            return tables[level][key], float(counts[level].get(key, 0)), level
    return None, 0.0, len(keys) - 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth():
    """Load all GT files, grouped by round number."""
    rounds = {}
    for path in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(path.stem.split("_")[0].replace("round", ""))
        sn = int(path.stem.split("_")[1].replace("seed", ""))
        with open(path) as fh:
            data = json.load(fh)
        if "initial_grid" not in data or "ground_truth" not in data:
            continue
        rounds.setdefault(rn, {})[sn] = {
            "initial_grid": np.asarray(data["initial_grid"], dtype=np.int32),
            "ground_truth": np.asarray(data["ground_truth"], dtype=np.float64),
        }
    return rounds


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(rounds):
    """Build base and spatial conditional tables from ground truth rounds.

    Base tables: P(final_class | initial_type, dist, n_ocean, coast)
    Spatial tables: P(final_class | initial_type, dist, n_ocean, coast, n_civ_neighbors_in_GT)

    The spatial tables are built using the GT itself to compute neighbor counts,
    so at prediction time we use the sampled neighbors instead.
    """
    n_levels = 4
    base_sums = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(n_levels)]
    base_counts = [defaultdict(float) for _ in range(n_levels)]
    spatial_sums = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(n_levels)]
    spatial_counts = [defaultdict(float) for _ in range(n_levels)]

    # Per-regime tables for better conditioning
    regime_base_sums = defaultdict(lambda: [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(n_levels)])
    regime_base_counts = defaultdict(lambda: [defaultdict(float) for _ in range(n_levels)])
    regime_spatial_sums = defaultdict(lambda: [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(n_levels)])
    regime_spatial_counts = defaultdict(lambda: [defaultdict(float) for _ in range(n_levels)])

    round_regimes = {}

    for rn, seeds in rounds.items():
        regime = classify_round(seeds)
        round_regimes[rn] = regime

        for sn, sd in seeds.items():
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]
            feats = compute_features(ig)
            h, w = ig.shape

            # Compute neighbor civ counts from GT argmax (most likely class)
            gt_argmax = gt.argmax(axis=2)
            gt_civ = ((gt_argmax == 1) | (gt_argmax == 2)).astype(np.int32)
            n_civ_gt = convolve(gt_civ, KERNEL, mode="constant")

            for y in range(h):
                for x in range(w):
                    code = int(ig[y, x])
                    bkeys = base_bucket_keys(code, feats["dist_civ"][y, x],
                                             feats["n_ocean"][y, x], feats["coast"][y, x])
                    if bkeys is None:
                        continue

                    vec = gt[y, x]
                    skeys = spatial_bucket_keys(code, feats["dist_civ"][y, x],
                                                feats["n_ocean"][y, x], feats["coast"][y, x],
                                                n_civ_gt[y, x])

                    for level, key in enumerate(bkeys):
                        base_sums[level][key] += vec
                        base_counts[level][key] += 1.0
                        regime_base_sums[regime][level][key] += vec
                        regime_base_counts[regime][level][key] += 1.0

                    if skeys is not None:
                        for level, key in enumerate(skeys):
                            spatial_sums[level][key] += vec
                            spatial_counts[level][key] += 1.0
                            regime_spatial_sums[regime][level][key] += vec
                            regime_spatial_counts[regime][level][key] += 1.0

    # Finalize tables
    def finalize(sums_list, counts_list):
        tables = [{} for _ in range(n_levels)]
        out_counts = [{} for _ in range(n_levels)]
        for level in range(n_levels):
            for key, vec in sums_list[level].items():
                c = counts_list[level][key]
                if c > 0:
                    tables[level][key] = normalize(vec / c)
                    out_counts[level][key] = float(c)
        return tables, out_counts

    base_tables, base_count_tables = finalize(base_sums, base_counts)
    spatial_tables, spatial_count_tables = finalize(spatial_sums, spatial_counts)

    regime_models = {}
    for regime in ["harsh", "moderate", "prosperous"]:
        if regime in regime_base_sums:
            rb, rbc = finalize(regime_base_sums[regime], regime_base_counts[regime])
            rs, rsc = finalize(regime_spatial_sums[regime], regime_spatial_counts[regime])
            regime_models[regime] = {
                "base_tables": rb, "base_counts": rbc,
                "spatial_tables": rs, "spatial_counts": rsc,
            }

    return {
        "base_tables": base_tables,
        "base_counts": base_count_tables,
        "spatial_tables": spatial_tables,
        "spatial_counts": spatial_count_tables,
        "regime_models": regime_models,
        "round_regimes": round_regimes,
    }


def classify_round(seeds):
    """Classify a round as harsh/moderate/prosperous based on frontier settlement rate."""
    frontier_rates = []
    for sn, sd in seeds.items():
        ig = sd["initial_grid"]
        gt = sd["ground_truth"]
        feats = compute_features(ig)
        frontier = (
            (feats["dist_civ"] >= 1.0) & (feats["dist_civ"] <= 5.0) &
            (ig != OCEAN) & (ig != MOUNTAIN)
        )
        if frontier.any():
            frontier_rates.append(float(gt[frontier, 1].mean()))  # Settlement class
    fr = float(np.mean(frontier_rates)) if frontier_rates else 0.10
    if fr < 0.05:
        return "harsh"
    if fr > 0.20:
        return "prosperous"
    return "moderate"


# ---------------------------------------------------------------------------
# Structural constraints
# ---------------------------------------------------------------------------

def apply_constraints(prob, code, dist, coast):
    """Apply hard structural constraints."""
    q = np.maximum(prob, FLOOR).copy()
    if code == OCEAN:
        return OCEAN_DIST.copy()
    if code == MOUNTAIN:
        return MOUNTAIN_DIST.copy()
    q[5] = 0.0  # No mountain on non-mountain cells
    if not coast:
        q[2] = 0.0  # No port if not coastal
    if dist > 10.0:
        q[3] = 0.0  # No ruin far from civ
    if code == FOREST and dist >= 12.0:
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    if code == PLAINS and dist >= 12.0:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return normalize(q)


def structured_fallback(code, dist, coast):
    """Fallback priors when no table match."""
    if code == FOREST and dist >= 12.0:
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    if code == PLAINS and dist >= 12.0:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if code == FOREST:
        return normalize(np.array([0.10, 0.08, 0.01 if coast else 0.0, 0.01, 0.80, 0.0]))
    if code == SETTLEMENT:
        return normalize(np.array([0.42, 0.32, 0.01 if coast else 0.0, 0.03, 0.22, 0.0]))
    if code == PORT:
        return normalize(np.array([0.18, 0.18, 0.34 if coast else 0.0, 0.01, 0.29, 0.0]))
    return normalize(np.array([0.88, 0.08, 0.02 if coast else 0.0, 0.01, 0.01, 0.0]))


# ---------------------------------------------------------------------------
# Prediction: base prior (independent, no spatial)
# ---------------------------------------------------------------------------

def base_prior(ig, feats, model, regime=None):
    """Compute per-cell base prior without spatial conditioning."""
    h, w = ig.shape
    pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)

    # Choose tables: blend regime-specific with pooled
    bt = model["base_tables"]
    bc = model["base_counts"]
    has_regime = regime and regime in model.get("regime_models", {})

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == OCEAN:
                pred[y, x] = OCEAN_DIST
                continue
            if code == MOUNTAIN:
                pred[y, x] = MOUNTAIN_DIST
                continue

            dist = feats["dist_civ"][y, x]
            coast = feats["coast"][y, x]
            bkeys = base_bucket_keys(code, dist, feats["n_ocean"][y, x], coast)
            if bkeys is None:
                pred[y, x] = UNIFORM.copy()
                continue

            q_pool, s_pool, _ = lookup_tables(bt, bc, bkeys)

            if has_regime:
                rm = model["regime_models"][regime]
                q_reg, s_reg, _ = lookup_tables(rm["base_tables"], rm["base_counts"], bkeys, (5, 8, 10, 1))
                if q_reg is not None and q_pool is not None:
                    # Blend: more weight to regime-specific when support is high
                    alpha = min(s_reg / (s_reg + 30.0), 0.7)
                    q = (1.0 - alpha) * q_pool + alpha * q_reg
                    q = normalize(q)
                elif q_reg is not None:
                    q = q_reg
                elif q_pool is not None:
                    q = q_pool
                else:
                    q = structured_fallback(code, dist, coast)
            elif q_pool is not None:
                q = q_pool
            else:
                q = structured_fallback(code, dist, coast)

            pred[y, x] = apply_constraints(q, code, dist, coast)

    return pred


# ---------------------------------------------------------------------------
# Spatial conditional lookup
# ---------------------------------------------------------------------------

def get_spatial_prob(code, dist, n_ocean, coast, n_civ_neighbors, model, regime=None):
    """Get probability conditioned on neighbor civ count."""
    skeys = spatial_bucket_keys(code, dist, n_ocean, coast, n_civ_neighbors)
    if skeys is None:
        return None

    st = model["spatial_tables"]
    sc = model["spatial_counts"]
    q_pool, s_pool, _ = lookup_tables(st, sc, skeys, (8, 12, 15, 1))

    if regime and regime in model.get("regime_models", {}):
        rm = model["regime_models"][regime]
        q_reg, s_reg, _ = lookup_tables(rm["spatial_tables"], rm["spatial_counts"], skeys, (4, 6, 8, 1))
        if q_reg is not None and q_pool is not None:
            alpha = min(s_reg / (s_reg + 20.0), 0.65)
            return normalize((1.0 - alpha) * q_pool + alpha * q_reg)
        if q_reg is not None:
            return q_reg
    return q_pool


# ---------------------------------------------------------------------------
# Gibbs sampling with spatial correlations
# ---------------------------------------------------------------------------

def run_gibbs_mc(ig, feats, model, base_pred, n_sims=N_SIMS, n_sweeps=N_SWEEPS, regime=None):
    """Run surrogate Monte Carlo with Gibbs sweeps for spatial correlations.

    1. Sample initial grid from base_pred
    2. For each Gibbs sweep:
       - Compute neighbor civ counts from current sample
       - Blend spatial conditional with base prior
       - Resample each dynamic cell
    3. Accumulate class counts across simulations
    """
    h, w = ig.shape
    counts = np.zeros((h, w, N_CLASSES), dtype=np.float64)

    # Pre-identify dynamic cells
    dynamic_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if int(ig[y, x]) not in (OCEAN, MOUNTAIN):
                dynamic_mask[y, x] = True
    dynamic_coords = list(zip(*np.where(dynamic_mask)))

    # Pre-compute static features for each dynamic cell
    cell_info = {}
    for y, x in dynamic_coords:
        code = int(ig[y, x])
        cell_info[(y, x)] = {
            "code": code,
            "dist": float(feats["dist_civ"][y, x]),
            "n_ocean": int(feats["n_ocean"][y, x]),
            "coast": bool(feats["coast"][y, x]),
        }

    # Precompute cumulative probabilities for initial sampling
    base_cdf = np.cumsum(base_pred, axis=2)

    for sim in range(n_sims):
        rng = np.random.RandomState(42 + sim * 7919)

        # Step 1: Sample initial state from base prior
        states = np.zeros((h, w), dtype=np.int8)
        states[ig == MOUNTAIN] = 5
        # Ocean stays 0 which maps to class 0

        # Vectorized sampling from base_pred
        u = rng.random((h, w))
        for y, x in dynamic_coords:
            idx = np.searchsorted(base_cdf[y, x], u[y, x])
            states[y, x] = min(idx, N_CLASSES - 1)

        # Step 2: Gibbs sweeps - resample each cell conditioned on neighbors
        for sweep in range(n_sweeps):
            # Compute neighbor civ counts from current states
            civ_map = ((states == 1) | (states == 2)).astype(np.int32)
            n_civ_current = convolve(civ_map, KERNEL, mode="constant")

            # Shuffle order for each sweep
            order = list(dynamic_coords)
            rng.shuffle(order)

            # Ramp spatial weight with sweeps
            spatial_weight = SPATIAL_WEIGHT + 0.1 * min(sweep, 3)

            for y, x in order:
                info = cell_info[(y, x)]
                code = info["code"]
                dist = info["dist"]
                coast = info["coast"]
                n_ocean = info["n_ocean"]

                # Get spatially-conditioned probability
                n_civ_nb = int(n_civ_current[y, x])
                q_spatial = get_spatial_prob(code, dist, n_ocean, coast, n_civ_nb, model, regime)

                if q_spatial is not None:
                    # Blend spatial with base
                    q = (1.0 - spatial_weight) * base_pred[y, x] + spatial_weight * q_spatial
                    q = apply_constraints(normalize(q), code, dist, coast)
                else:
                    q = base_pred[y, x]

                # Sample new state
                old_state = int(states[y, x])
                cdf = np.cumsum(q)
                new_state = min(int(np.searchsorted(cdf, rng.random())), N_CLASSES - 1)

                if new_state != old_state:
                    # Update neighbor counts incrementally
                    old_civ = 1 if old_state in (1, 2) else 0
                    new_civ = 1 if new_state in (1, 2) else 0
                    delta = new_civ - old_civ
                    if delta != 0:
                        for dy, dx in NEIGHBOR_OFFSETS:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                n_civ_current[ny, nx] += delta
                    states[y, x] = new_state

        # Step 3: Accumulate counts
        for cls in range(N_CLASSES):
            counts[:, :, cls] += (states == cls)

    # Normalize to probabilities
    pred = counts / float(n_sims)
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def kl_divergence(p, q):
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def entropy(p):
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


def score_prediction(gt, pred):
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    kl = kl_divergence(gt, pred)
    H = entropy(gt)
    wkl = H * kl
    dynamic = H > 0.01
    if not dynamic.any():
        return {"wkl": 0.0, "kl": 0.0, "dynamic_cells": 0}
    return {
        "wkl": float(wkl[dynamic].mean()),
        "kl": float(kl[dynamic].mean()),
        "dynamic_cells": int(dynamic.sum()),
    }


def score_from_wkl(wkl):
    return max(0.0, 100.0 * (1.0 - wkl))


# ---------------------------------------------------------------------------
# Regime estimation for new rounds
# ---------------------------------------------------------------------------

def estimate_regime_from_model(model):
    """When no GT available for current round, use most common regime."""
    regimes = list(model.get("round_regimes", {}).values())
    if not regimes:
        return "moderate"
    from collections import Counter
    return Counter(regimes).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(initial_grid, observations=None, round_num=None):
    """Predict final state probabilities for a 40x40 initial grid.

    Args:
        initial_grid: 40x40 grid of cell codes
        observations: optional list of observation dicts (not used in base MC)
        round_num: optional round number for regime estimation

    Returns:
        (40, 40, 6) probability tensor
    """
    ig = np.asarray(initial_grid, dtype=np.int32)
    rounds = load_ground_truth()
    model = build_model(rounds)
    return predict_with_model(ig, model, round_num=round_num)


def predict_with_model(ig, model, round_num=None, n_sims=N_SIMS, n_sweeps=N_SWEEPS,
                       mc_blend=MC_BLEND):
    """Predict using a pre-built model.

    Args:
        mc_blend: how much of the base prior to blend with MC result (0=pure MC, 1=pure base)
    """
    ig = np.asarray(ig, dtype=np.int32)
    feats = compute_features(ig)

    # Estimate regime
    regime = estimate_regime_from_model(model)

    # Base prior (independent)
    base_pred = base_prior(ig, feats, model, regime=regime)

    if n_sims <= 0:
        return base_pred

    # Gibbs MC with spatial correlations
    mc_pred = run_gibbs_mc(ig, feats, model, base_pred,
                           n_sims=n_sims, n_sweeps=n_sweeps, regime=regime)

    # Blend MC with base to reduce sampling noise while keeping spatial signal
    pred = (1.0 - mc_blend) * mc_pred + mc_blend * base_pred
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(n_sims=N_SIMS, n_sweeps=N_SWEEPS, mc_blend=MC_BLEND):
    """Leave-one-round-out cross-validation."""
    rounds = load_ground_truth()
    round_nums = sorted(rounds.keys())
    per_round = []
    all_seed_scores = []

    # Also compute base (no spatial) for comparison
    all_base_scores = []

    print(f"Spatial MC CV: {len(round_nums)} rounds, {n_sims} sims, {n_sweeps} sweeps, blend={mc_blend}")
    print("=" * 70)

    for held_out in round_nums:
        train = {rn: data for rn, data in rounds.items() if rn != held_out}
        model = build_model(train)

        # Estimate regime from training data regimes
        regime = estimate_regime_from_model(model)

        seed_scores = []
        base_scores = []

        for sn, sd in sorted(rounds[held_out].items()):
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]
            feats = compute_features(ig)

            # Base prior (no spatial)
            bp = base_prior(ig, feats, model, regime=regime)
            base_metrics = score_prediction(gt, bp)
            base_scores.append(base_metrics["wkl"])
            all_base_scores.append(base_metrics["wkl"])

            # Spatial MC with blending
            mc_pred = run_gibbs_mc(ig, feats, model, bp,
                                   n_sims=n_sims, n_sweeps=n_sweeps, regime=regime)
            blended = (1.0 - mc_blend) * mc_pred + mc_blend * bp
            blended = np.maximum(blended, FLOOR)
            blended /= blended.sum(axis=2, keepdims=True)
            metrics = score_prediction(gt, blended)
            seed_scores.append(metrics["wkl"])
            all_seed_scores.append(metrics["wkl"])

        mean_wkl = float(np.mean(seed_scores))
        mean_base = float(np.mean(base_scores))
        delta = mean_wkl - mean_base
        marker = "BETTER" if delta < 0 else "WORSE"
        per_round.append({
            "held_out_round": int(held_out),
            "mean_wkl": mean_wkl,
            "base_wkl": mean_base,
            "score": score_from_wkl(mean_wkl),
            "base_score": score_from_wkl(mean_base),
        })
        print(f"R{held_out:02d}: spatial wKL={mean_wkl:.6f} (score={score_from_wkl(mean_wkl):.2f}) "
              f"| base wKL={mean_base:.6f} (score={score_from_wkl(mean_base):.2f}) "
              f"| delta={delta:+.6f} {marker}")

    overall = float(np.mean(all_seed_scores))
    overall_base = float(np.mean(all_base_scores))
    print(f"\n{'=' * 70}")
    print(f"Overall CV:  spatial wKL={overall:.6f} (score={score_from_wkl(overall):.2f})")
    print(f"             base    wKL={overall_base:.6f} (score={score_from_wkl(overall_base):.2f})")
    print(f"             delta     ={overall - overall_base:+.6f}")

    return {
        "cv_mean_wkl": overall,
        "cv_base_wkl": overall_base,
        "cv_score": score_from_wkl(overall),
        "per_round": per_round,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if mode == "cv":
        n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        n_sweeps = int(sys.argv[3]) if len(sys.argv) > 3 else N_SWEEPS
        results = cross_validate(n_sims=n_sims, n_sweeps=n_sweeps)
    elif mode == "base":
        # Just base prior, no MC
        rounds = load_ground_truth()
        round_nums = sorted(rounds.keys())
        all_scores = []
        for held_out in round_nums:
            train = {rn: d for rn, d in rounds.items() if rn != held_out}
            model = build_model(train)
            regime = estimate_regime_from_model(model)
            for sn, sd in sorted(rounds[held_out].items()):
                ig = sd["initial_grid"]
                gt = sd["ground_truth"]
                feats = compute_features(ig)
                bp = base_prior(ig, feats, model, regime=regime)
                m = score_prediction(gt, bp)
                all_scores.append(m["wkl"])
                print(f"R{held_out:02d} s{sn}: base wKL={m['wkl']:.6f}")
        print(f"\nOverall base wKL: {np.mean(all_scores):.6f}")
    else:
        raise SystemExit(f"Unknown mode: {mode}")
