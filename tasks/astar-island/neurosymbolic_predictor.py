#!/usr/bin/env python3
"""Neurosymbolic cellular automaton predictor for Astar Island.

Combines:
  1. REGIME-SPECIFIC TRANSITION MATRICES learned from replay data
  2. VECTORIZED MONTE CARLO simulation (numpy)
  3. BUCKET-BASED GT AVERAGES with hierarchical Bayesian shrinkage
  4. HARD CONSTRAINTS enforced throughout
  5. REGIME DETECTION from observations (Bayesian posterior)
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve

GT_DIR = Path(__file__).parent / "ground_truth"
REPLAY_DIR = Path(__file__).parent / "replays"
N_CLASSES = 6
FLOOR = 1e-6

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
PLAINS = 11
EMPTY_CODE = 0

CODE_TO_CLASS = {0: 0, 11: 0, 10: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
OCEAN_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
MOUNTAIN_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

HARSH_THRESHOLD = 0.05
PROSPEROUS_THRESHOLD = 0.20
DEFAULT_REGIME_WEIGHTS = {"harsh": 0.23, "moderate": 0.54, "prosperous": 0.23}

_MODEL = None

N_SIMS = 500
N_STEPS = 50
MC_WEIGHT = 0.8
MC_DECAY = 4.0

MAX_FROM = 6
MAX_NCIV = 6
MAX_MOD4 = 4
MAX_COAST = 2


# ──────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────

def compute_features(ig):
    h, w = ig.shape
    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN
    forest = ig == FOREST
    if civ.any():
        dist_civ = distance_transform_edt(~civ)
    else:
        dist_civ = np.full((h, w), 99.0)
    n_ocean = convolve(ocean.astype(np.int32), KERNEL, mode="constant")
    n_civ = convolve(civ.astype(np.int32), KERNEL, mode="constant")
    n_forest = convolve(forest.astype(np.int32), KERNEL, mode="constant")
    coast = (n_ocean > 0) & ~ocean
    return dist_civ, n_ocean, n_civ, n_forest, coast


# ──────────────────────────────────────────────────────────────────────
# Transition matrix learning
# ──────────────────────────────────────────────────────────────────────

def learn_transitions(replay_files):
    """Learn transition probabilities. Returns (6,6,4,2,6) array."""
    counts = np.zeros((MAX_FROM, MAX_NCIV, MAX_MOD4, MAX_COAST, N_CLASSES))
    counts_nc = np.zeros((MAX_FROM, MAX_NCIV, MAX_MOD4, N_CLASSES))
    counts_nm = np.zeros((MAX_FROM, MAX_NCIV, N_CLASSES))

    for fpath in replay_files:
        with open(fpath) as fh:
            data = json.load(fh)
        frames = data["frames"]
        for si in range(len(frames) - 1):
            g0 = np.array(frames[si]["grid"], dtype=np.int32)
            g1 = np.array(frames[si + 1]["grid"], dtype=np.int32)
            step = int(frames[si]["step"])
            mod4 = step % 4
            h, w = g0.shape

            civ0 = ((g0 == SETTLEMENT) | (g0 == PORT)).astype(np.int32)
            nc_map = convolve(civ0, KERNEL, mode="constant")
            oc_map = convolve((g0 == OCEAN).astype(np.int32), KERNEL, mode="constant")

            mask = (g0 != MOUNTAIN) & (g0 != OCEAN)
            ys, xs = np.where(mask)

            fc = np.array([CODE_TO_CLASS.get(int(g0[y, x]), 0) for y, x in zip(ys, xs)])
            tc = np.array([CODE_TO_CLASS.get(int(g1[y, x]), 0) for y, x in zip(ys, xs)])
            nc = np.minimum(nc_map[ys, xs], 5)
            cv = (oc_map[ys, xs] > 0).astype(np.int32)

            for i in range(len(ys)):
                f, n, t, c = int(fc[i]), int(nc[i]), int(tc[i]), int(cv[i])
                counts[f, n, mod4, c, t] += 1
                counts_nc[f, n, mod4, t] += 1
                counts_nm[f, n, t] += 1

    trans = np.zeros((MAX_FROM, MAX_NCIV, MAX_MOD4, MAX_COAST, N_CLASSES))
    for f in range(MAX_FROM):
        for n in range(MAX_NCIV):
            for m in range(MAX_MOD4):
                for c in range(MAX_COAST):
                    tot = counts[f, n, m, c].sum()
                    if tot >= 20:
                        trans[f, n, m, c] = counts[f, n, m, c] / tot
                    elif counts_nc[f, n, m].sum() >= 20:
                        trans[f, n, m, c] = counts_nc[f, n, m] / counts_nc[f, n, m].sum()
                    elif counts_nm[f, n].sum() >= 5:
                        trans[f, n, m, c] = counts_nm[f, n] / counts_nm[f, n].sum()
                    else:
                        trans[f, n, m, c, f] = 1.0
    return trans


# ──────────────────────────────────────────────────────────────────────
# Monte Carlo simulation
# ──────────────────────────────────────────────────────────────────────

def simulate_forward(initial_grid, trans_array, n_sims=N_SIMS, n_steps=N_STEPS):
    ig = np.asarray(initial_grid, dtype=np.int32)
    h, w = ig.shape
    n_cells = h * w

    is_ocean = ig == OCEAN
    is_mountain = ig == MOUNTAIN
    is_static = (is_ocean | is_mountain).ravel()
    dynamic_idx = np.where(~is_static)[0]

    init_flat = np.zeros(n_cells, dtype=np.int32)
    for i in range(n_cells):
        init_flat[i] = CODE_TO_CLASS.get(int(ig.ravel()[i]), 0)

    ocean_mask = is_ocean.astype(np.int32)
    coast_flat = ((convolve(ocean_mask, KERNEL, mode="constant") > 0) & ~is_ocean).astype(np.int32).ravel()

    trans_cdf = np.cumsum(trans_array, axis=-1)
    outcome_counts = np.zeros((n_cells, N_CLASSES), dtype=np.int32)
    rng = np.random.RandomState(42)

    for sim in range(n_sims):
        state = init_flat.copy()
        for step in range(n_steps):
            mod4 = step % 4
            state_2d = state.reshape(h, w)
            civ_2d = ((state_2d == 1) | (state_2d == 2)).astype(np.int32)
            n_civ_flat = np.minimum(convolve(civ_2d, KERNEL, mode="constant").ravel(), 5)

            from_cls = state[dynamic_idx]
            nciv = n_civ_flat[dynamic_idx]
            coast_v = coast_flat[dynamic_idx]

            cdfs = trans_cdf[from_cls, nciv, mod4, coast_v]
            u = rng.random(len(dynamic_idx))
            new_cls = np.zeros(len(dynamic_idx), dtype=np.int32)
            for c in range(N_CLASSES - 1):
                new_cls += (u > cdfs[:, c]).astype(np.int32)
            state[dynamic_idx] = new_cls

        for i in range(n_cells):
            outcome_counts[i, state[i]] += 1

    dist = outcome_counts.astype(np.float64) / n_sims
    dist = dist.reshape(h, w, N_CLASSES)
    dist[is_ocean] = OCEAN_DIST
    dist[is_mountain] = MOUNTAIN_DIST
    return dist


# ──────────────────────────────────────────────────────────────────────
# Bucket model
# ──────────────────────────────────────────────────────────────────────

def cell_type_str(code):
    if code == SETTLEMENT: return "S"
    if code == PORT: return "P"
    if code == FOREST: return "F"
    if code in (PLAINS, EMPTY_CODE): return "L"
    return None


def make_keys(code, dist, n_ocean, n_civ, n_forest, coast):
    t = cell_type_str(code)
    if t is None: return None
    d = int(min(max(math.floor(dist), 0), 15))
    no = min(int(n_ocean), 4)
    nc = min(int(n_civ), 5)
    nf = min(int(n_forest), 4)
    c = 1 if coast else 0
    return [
        (t, d, nc, no, nf, c),
        (t, d, nc, no, c),
        (t, d, nc, c),
        (t, d, c),
        (t, min(d, 8)),
        (t,),
    ]


N_LEVELS = 6
MIN_SUPPORT = [5, 5, 8, 10, 1, 1]


def build_bucket_tables(rounds):
    sums = [defaultdict(lambda: np.zeros(N_CLASSES)) for _ in range(N_LEVELS)]
    counts = [defaultdict(float) for _ in range(N_LEVELS)]
    for rn, seeds in rounds.items():
        for sn, sd in seeds.items():
            ig = np.asarray(sd["initial_grid"], dtype=np.int32)
            gt = np.asarray(sd["ground_truth"], dtype=np.float64)
            h, w = ig.shape
            dist_civ, n_ocean, n_civ, n_forest, coast = compute_features(ig)
            for y in range(h):
                for x in range(w):
                    code = int(ig[y, x])
                    if code in (OCEAN, MOUNTAIN): continue
                    keys = make_keys(code, dist_civ[y, x], n_ocean[y, x],
                                    n_civ[y, x], n_forest[y, x], coast[y, x])
                    if keys is None: continue
                    prob = gt[y, x]
                    for level, key in enumerate(keys):
                        sums[level][key] += prob
                        counts[level][key] += 1.0
    means = []
    for level in range(N_LEVELS):
        m = {}
        for key in sums[level]:
            c = counts[level][key]
            if c > 0: m[key] = sums[level][key] / c
        means.append(m)
    return {"means": means, "counts": [dict(c) for c in counts]}


def bucket_lookup(tables, keys):
    means = tables["means"]
    cnts = tables["counts"]
    for level, key in enumerate(keys):
        if key in means[level]:
            c = cnts[level].get(key, 0)
            if c >= MIN_SUPPORT[level]:
                prob = means[level][key].copy()
                if level < N_LEVELS - 1 and c < 200:
                    for cl in range(level + 1, N_LEVELS):
                        ck = keys[cl]
                        if ck in means[cl]:
                            shrink = 20.0 / (level + 1)
                            alpha = c / (c + shrink)
                            prob = alpha * prob + (1 - alpha) * means[cl][ck]
                            prob = np.maximum(prob, 1e-12)
                            prob /= prob.sum()
                            break
                return prob, c
    for level, key in enumerate(keys):
        if key in means[level]:
            return means[level][key].copy(), cnts[level].get(key, 0)
    return None, 0


def interp_bucket(tables, code, dist, n_ocean, n_civ, n_forest, coast):
    d = float(dist)
    lo = int(min(max(math.floor(d), 0), 15))
    hi = min(lo + 1, 15)
    frac = float(np.clip(d - lo, 0, 1))
    keys_lo = make_keys(code, lo, n_ocean, n_civ, n_forest, coast)
    if keys_lo is None: return None, 0
    q_lo, s_lo = bucket_lookup(tables, keys_lo)
    if hi == lo or frac < 0.01: return q_lo, s_lo
    keys_hi = make_keys(code, hi, n_ocean, n_civ, n_forest, coast)
    q_hi, s_hi = bucket_lookup(tables, keys_hi)
    if q_lo is None: return q_hi, s_hi
    if q_hi is None: return q_lo, s_lo
    q = (1 - frac) * q_lo + frac * q_hi
    q = np.maximum(q, 1e-12)
    q /= q.sum()
    return q, (1 - frac) * s_lo + frac * s_hi


# ──────────────────────────────────────────────────────────────────────
# Regime detection
# ──────────────────────────────────────────────────────────────────────

def classify_round(round_data):
    frontier_rates = []
    for sn, sd in round_data.items():
        ig = np.asarray(sd["initial_grid"], dtype=np.int32)
        gt = np.asarray(sd["ground_truth"], dtype=np.float64)
        dist_civ = compute_features(ig)[0]
        frontier = (dist_civ >= 1) & (dist_civ <= 5) & (ig != OCEAN) & (ig != MOUNTAIN)
        if frontier.any():
            frontier_rates.append(gt[frontier, 1].mean())
    f_rate = np.mean(frontier_rates) if frontier_rates else 0.1
    if f_rate < HARSH_THRESHOLD: return "harsh"
    if f_rate > PROSPEROUS_THRESHOLD: return "prosperous"
    return "moderate"


def cell_code_to_class(cell):
    if cell in (0, 10, 11): return 0
    if cell == 1: return 1
    if cell == 2: return 2
    if cell == 3: return 3
    if cell == 4: return 4
    if cell == 5: return 5
    return 0


def detect_regime(ig, observations, model):
    if not observations:
        return DEFAULT_REGIME_WEIGHTS.copy()
    dist_civ, n_ocean, n_civ, n_forest, coast = compute_features(ig)
    h, w = ig.shape
    log_liks = {}
    for regime in ("harsh", "moderate", "prosperous"):
        rt = model["regime_tables"].get(regime)
        if rt is None:
            log_liks[regime] = -1000.0
            continue
        ll = 0.0
        for obs in observations:
            grid = obs.get("grid", [])
            vx, vy = int(obs.get("viewport_x", 0)), int(obs.get("viewport_y", 0))
            for dy, row in enumerate(grid):
                for dx, cv in enumerate(row):
                    y, x = vy + dy, vx + dx
                    if not (0 <= y < h and 0 <= x < w): continue
                    code = int(ig[y, x])
                    if code in (OCEAN, MOUNTAIN): continue
                    q, _ = interp_bucket(rt, code, dist_civ[y, x], n_ocean[y, x],
                                        n_civ[y, x], n_forest[y, x], coast[y, x])
                    if q is None: continue
                    cls = cell_code_to_class(int(cv))
                    ll += max(np.log(max(float(q[cls]), 1e-6)), -6.0)
        log_liks[regime] = ll

    log_prior = {r: np.log(DEFAULT_REGIME_WEIGHTS[r]) for r in DEFAULT_REGIME_WEIGHTS}
    log_post = {r: log_liks[r] + log_prior[r] for r in log_liks}
    mx = max(log_post.values())
    wt = {r: max(np.exp(log_post[r] - mx), 0.01) for r in log_post}
    t = sum(wt.values())
    return {r: v / t for r, v in wt.items()}


# ──────────────────────────────────────────────────────────────────────
# Model building
# ──────────────────────────────────────────────────────────────────────

def get_replay_files(round_nums):
    """Get replay files for given round numbers."""
    files = []
    for f in sorted(REPLAY_DIR.glob("round*_seed*.json")):
        if "extra" in f.stem: continue
        rn = int(f.stem.split("_")[0].replace("round", ""))
        if rn in round_nums:
            files.append(f)
    dense_dir = REPLAY_DIR / "dense_training"
    if dense_dir.exists():
        for f in sorted(dense_dir.glob("*.json")):
            rn = int(f.stem.split("_")[0].replace("round", ""))
            if rn in round_nums:
                files.append(f)
    return files


def build_model_from_data(rounds):
    for rn, seeds in rounds.items():
        for sn, sd in seeds.items():
            if not isinstance(sd["initial_grid"], np.ndarray):
                sd["initial_grid"] = np.array(sd["initial_grid"], dtype=np.int32)
            if not isinstance(sd["ground_truth"], np.ndarray):
                sd["ground_truth"] = np.array(sd["ground_truth"], dtype=np.float64)

    round_regimes = {rn: classify_round(rd) for rn, rd in rounds.items()}

    # Regime-specific bucket tables
    regime_tables = {}
    for regime in ("harsh", "moderate", "prosperous"):
        regime_rounds = {rn: rd for rn, rd in rounds.items() if round_regimes[rn] == regime}
        regime_tables[regime] = build_bucket_tables(regime_rounds) if regime_rounds else None

    pooled_tables = build_bucket_tables(rounds)

    # Learn transition matrices
    train_rounds = set(rounds.keys())
    all_replay_files = get_replay_files(train_rounds)
    pooled_trans = learn_transitions(all_replay_files) if all_replay_files else None

    # Regime-specific transitions
    regime_trans = {}
    for regime in ("harsh", "moderate", "prosperous"):
        regime_round_nums = {rn for rn, r in round_regimes.items() if r == regime}
        regime_files = get_replay_files(regime_round_nums)
        if regime_files:
            regime_trans[regime] = learn_transitions(regime_files)
        else:
            regime_trans[regime] = None

    return {
        "regime_tables": regime_tables,
        "pooled_tables": pooled_tables,
        "round_regimes": round_regimes,
        "pooled_trans": pooled_trans,
        "regime_trans": regime_trans,
    }


def build_model():
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(f.stem.split("_")[0].replace("round", ""))
        sn = int(f.stem.split("_")[1].replace("seed", ""))
        with open(f) as fh:
            data = json.load(fh)
        if "ground_truth" not in data or "initial_grid" not in data:
            continue
        rounds.setdefault(rn, {})[sn] = {
            "initial_grid": np.array(data["initial_grid"], dtype=np.int32),
            "ground_truth": np.array(data["ground_truth"], dtype=np.float64),
        }
    return build_model_from_data(rounds)


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build_model()
    return _MODEL


# ──────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────

def predict_with_model(initial_grid, model, observations=None):
    return _predict_impl(initial_grid, model, observations)


def predict(initial_grid, observations=None):
    return _predict_impl(initial_grid, get_model(), observations)


def _predict_impl(initial_grid, model, observations=None):
    ig = np.asarray(initial_grid, dtype=np.int32)
    h, w = ig.shape
    dist_civ, n_ocean, n_civ, n_forest, coast = compute_features(ig)

    regime_w = detect_regime(ig, observations or [], model)

    # MC simulation with pooled transitions (most stable)
    mc_dist = None
    if model["pooled_trans"] is not None:
        mc_dist = simulate_forward(ig, model["pooled_trans"], n_sims=N_SIMS, n_steps=N_STEPS)

    # Bucket prediction
    bucket_pred = np.zeros((h, w, N_CLASSES))
    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == OCEAN:
                bucket_pred[y, x] = OCEAN_DIST
                continue
            if code == MOUNTAIN:
                bucket_pred[y, x] = MOUNTAIN_DIST
                continue

            d = dist_civ[y, x]
            no = n_ocean[y, x]
            nc = n_civ[y, x]
            nf = n_forest[y, x]
            is_coast = coast[y, x]

            cell_pred = np.zeros(N_CLASSES)
            tw = 0.0
            for regime, rw in regime_w.items():
                if rw < 0.01: continue
                rt = model["regime_tables"].get(regime)
                if rt is None: continue
                q, _ = interp_bucket(rt, code, d, no, nc, nf, is_coast)
                if q is not None:
                    cell_pred += rw * q
                    tw += rw
            if tw < 0.5:
                q_pool, _ = interp_bucket(model["pooled_tables"], code, d, no, nc, nf, is_coast)
                if q_pool is not None:
                    cell_pred += (1.0 - tw) * q_pool
                    tw = 1.0
            if tw > 0:
                bucket_pred[y, x] = cell_pred / tw
            else:
                bucket_pred[y, x] = np.ones(N_CLASSES) / N_CLASSES

    # Blend MC and bucket
    if mc_dist is not None:
        pred = np.zeros((h, w, N_CLASSES))
        for y in range(h):
            for x in range(w):
                code = int(ig[y, x])
                if code in (OCEAN, MOUNTAIN):
                    pred[y, x] = bucket_pred[y, x]
                    continue
                d = dist_civ[y, x]
                mc_w = MC_WEIGHT * min(1.0, MC_DECAY / max(d, 0.5))
                pred[y, x] = mc_w * mc_dist[y, x] + (1 - mc_w) * bucket_pred[y, x]
    else:
        pred = bucket_pred.copy()

    # Hard constraints
    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code in (OCEAN, MOUNTAIN): continue
            if not coast[y, x]:
                pred[y, x, 2] = 0.0
            elif n_ocean[y, x] >= 2 and 1 <= dist_civ[y, x] <= 3:
                pred[y, x, 2] *= 2.0
            pred[y, x, 5] = 0.0
            if dist_civ[y, x] > 10:
                pred[y, x, 3] = 0.0

    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, str(Path(__file__).parent))
    from eval import load_ground_truth, score_prediction, eval_cv, report

    rounds = load_ground_truth()

    print("=== Neurosymbolic Predictor ===")
    t0 = time.time()
    model = build_model_from_data(rounds)
    print(f"Model built in {time.time() - t0:.1f}s")
    print(f"Regimes: {model['round_regimes']}")

    results = []
    for rn in sorted(rounds.keys()):
        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            p = predict_with_model(sd["initial_grid"], model)
            s = score_prediction(sd["ground_truth"], p)
            s["round"] = rn
            s["seed"] = sn
            results.append(s)
    report(results, "IN-SAMPLE (no obs)")
    print(f"  Time: {time.time() - t0:.1f}s")

    print("\n=== Leave-one-round-out CV ===")
    t0 = time.time()
    cv_results = eval_cv("neurosymbolic_predictor", rounds, 0, 0)
    report(cv_results, "CV (no obs)")
    print(f"  Time: {time.time() - t0:.1f}s")
