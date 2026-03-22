#!/usr/bin/env python3
"""Learned cellular automaton simulator from replay data.

Learns per-step transition rules from 90 replay files and runs fast fully-vectorized
Monte Carlo simulations. No per-cell Python loops during simulation.

Architecture:
1. Build dense transition probability array indexed by
   (step, curr_class, n_civ_bin, n_forest_bin, dist_bin) -> P[6 classes]
2. At each step: compute features as numpy arrays, index into table, sample in parallel.
3. Repeat for N simulations, aggregate outcomes.

Cell codes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain, 10=Ocean, 11=Plains
Output channels: 0=Empty/Plains/Ocean, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_cdt

BASE_DIR = Path(__file__).parent
REPLAY_DIR = BASE_DIR / "replays"
GT_DIR = BASE_DIR / "ground_truth"

N_CLASSES = 6
OCEAN = 10
MOUNTAIN = 5

OCEAN_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
MOUNTAIN_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

# Bin sizes for features
N_CIV_BINS = 5    # 0, 1, 2, 3, 4+
N_FOREST_BINS = 5 # 0, 1, 2, 3, 4+
N_OCEAN_BINS = 4  # 0, 1, 2, 3+
N_DIST_BINS = 4   # 0 (<=1), 1 (<=3), 2 (<=6), 3 (>6)
N_STEPS = 50

# Map cell codes to class indices
_C2C = np.zeros(12, dtype=np.int8)
_C2C[0] = 0; _C2C[1] = 1; _C2C[2] = 2; _C2C[3] = 3
_C2C[4] = 4; _C2C[5] = 5; _C2C[10] = 0; _C2C[11] = 0

_CLASS_TO_CODE = np.array([11, 1, 2, 3, 4, 5], dtype=np.int16)


def grid_to_class(grid):
    return _C2C[grid]


def count_neighbors_of_class(cls_grid, cls_val, H, W):
    """Count 8-neighbors equal to cls_val for all cells. Returns (H,W) int array."""
    mask = (cls_grid == cls_val).astype(np.int32)
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    result = np.zeros((H, W), dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            result += padded[1+dr:H+1+dr, 1+dc:W+1+dc]
    return result


def compute_features_vectorized(cls_grid, code_grid, H, W):
    """Compute binned features for all cells at once.

    Returns:
        n_civ_bin: (H,W) int, 0-4
        n_forest_bin: (H,W) int, 0-4
        n_ocean_bin: (H,W) int, 0-3
        dist_bin: (H,W) int, 0-3
    """
    n_civ = count_neighbors_of_class(cls_grid, 1, H, W) + \
            count_neighbors_of_class(cls_grid, 2, H, W)
    n_forest = count_neighbors_of_class(cls_grid, 4, H, W)
    n_ocean_raw = count_neighbors_of_class(cls_grid, 0, H, W)
    # Note: class 0 includes ocean AND plains AND empty - we need code-level ocean
    # Let's count ocean from code_grid instead
    ocean_mask = (code_grid == OCEAN).astype(np.int32)
    padded = np.pad(ocean_mask, 1, mode='constant', constant_values=0)
    n_ocean = np.zeros((H, W), dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            n_ocean += padded[1+dr:H+1+dr, 1+dc:W+1+dc]

    n_civ_bin = np.minimum(n_civ, N_CIV_BINS - 1).astype(np.int8)
    n_forest_bin = np.minimum(n_forest, N_FOREST_BINS - 1).astype(np.int8)
    n_ocean_bin = np.minimum(n_ocean, N_OCEAN_BINS - 1).astype(np.int8)

    # Distance to civ
    civ_mask = (code_grid == 1) | (code_grid == 2)
    if civ_mask.any():
        dist = distance_transform_cdt(~civ_mask, metric='taxicab')
    else:
        dist = np.full((H, W), 99, dtype=np.int32)

    dist_bin = np.zeros((H, W), dtype=np.int8)
    dist_bin[dist <= 1] = 0
    dist_bin[(dist > 1) & (dist <= 3)] = 1
    dist_bin[(dist > 3) & (dist <= 6)] = 2
    dist_bin[dist > 6] = 3

    return n_civ_bin, n_forest_bin, n_ocean_bin, dist_bin


def load_replays(exclude_rounds=None):
    replays = []
    for f in sorted(REPLAY_DIR.glob("round*_seed*.json")):
        rn = int(f.stem.split("_")[0].replace("round", ""))
        if exclude_rounds and rn in exclude_rounds:
            continue
        replays.append(json.loads(f.read_text()))
    return replays


def build_transition_array(replays):
    """Build dense transition probability array from replays.

    Shape: (N_STEPS, N_CLASSES, N_CIV_BINS, N_FOREST_BINS, N_OCEAN_BINS, N_DIST_BINS, N_CLASSES)

    For each (step, current_class, neighborhood_features), stores the probability
    distribution over next-step classes.
    """
    shape = (N_STEPS, N_CLASSES, N_CIV_BINS, N_FOREST_BINS, N_OCEAN_BINS, N_DIST_BINS, N_CLASSES)
    counts = np.zeros(shape, dtype=np.int32)

    for replay in replays:
        frames = replay['frames']
        for t in range(len(frames) - 1):
            grid_t = np.array(frames[t]['grid'], dtype=np.int16)
            grid_t1 = np.array(frames[t + 1]['grid'], dtype=np.int16)
            step = frames[t]['step']
            H, W = grid_t.shape

            cls_t = grid_to_class(grid_t)
            cls_t1 = grid_to_class(grid_t1)

            mutable = (grid_t != OCEAN) & (grid_t != MOUNTAIN)

            n_civ_bin, n_forest_bin, n_ocean_bin, dist_bin = \
                compute_features_vectorized(cls_t, grid_t, H, W)

            rows, cols = np.where(mutable)
            currs = cls_t[rows, cols]
            nexts = cls_t1[rows, cols]
            civs = n_civ_bin[rows, cols]
            forests = n_forest_bin[rows, cols]
            oceans = n_ocean_bin[rows, cols]
            dists = dist_bin[rows, cols]

            # Vectorized increment using np.add.at
            idx = (np.full(len(rows), step, dtype=np.int32),
                   currs, civs, forests, oceans, dists, nexts)
            np.add.at(counts, idx, 1)

    # Compute probabilities with hierarchical fallback
    # 1. Fine-grained (per step, per feature combo)
    totals = counts.sum(axis=-1, keepdims=True)

    # 2. Coarser: collapse forest dimension
    coarse1_counts = counts.sum(axis=3, keepdims=False)  # sum over forest
    # shape: (50, 6, 5, 4, 4, 6) - step, cls, civ, ocean, dist, next
    coarse1_totals = coarse1_counts.sum(axis=-1, keepdims=True)

    # 3. Even coarser: collapse ocean too
    coarse2_counts = coarse1_counts.sum(axis=3, keepdims=False)  # sum over ocean
    # shape: (50, 6, 5, 4, 6) - step, cls, civ, dist, next
    coarse2_totals = coarse2_counts.sum(axis=-1, keepdims=True)

    # 4. Global: just step + class
    global_counts = coarse2_counts.sum(axis=(2, 3))  # sum over civ, dist
    # shape: (50, 6, 6)
    global_totals = global_counts.sum(axis=-1, keepdims=True)

    # Build probability table with fallback
    MIN_FINE = 5
    MIN_COARSE1 = 3
    MIN_COARSE2 = 2

    probs = np.zeros(shape, dtype=np.float32)

    for s in range(N_STEPS):
        for c in range(N_CLASSES):
            for ci in range(N_CIV_BINS):
                for fo in range(N_FOREST_BINS):
                    for oc in range(N_OCEAN_BINS):
                        for db in range(N_DIST_BINS):
                            t = totals[s, c, ci, fo, oc, db, 0]
                            if t >= MIN_FINE:
                                probs[s, c, ci, fo, oc, db] = counts[s, c, ci, fo, oc, db] / t
                            elif coarse1_totals[s, c, ci, oc, db, 0] >= MIN_COARSE1:
                                probs[s, c, ci, fo, oc, db] = coarse1_counts[s, c, ci, oc, db] / coarse1_totals[s, c, ci, oc, db, 0]
                            elif coarse2_totals[s, c, ci, db, 0] >= MIN_COARSE2:
                                probs[s, c, ci, fo, oc, db] = coarse2_counts[s, c, ci, db] / coarse2_totals[s, c, ci, db, 0]
                            elif global_totals[s, c, 0] > 0:
                                probs[s, c, ci, fo, oc, db] = global_counts[s, c] / global_totals[s, c, 0]
                            else:
                                probs[s, c, ci, fo, oc, db, c] = 1.0

    return probs


def simulate_fast(initial_grid, probs_table, n_sims=200, rng=None):
    """Run fully vectorized Monte Carlo simulation.

    At each step:
    1. Compute features for all cells (vectorized)
    2. Index into probs_table to get per-cell transition distributions
    3. Sample from distributions (vectorized)

    Returns: (H, W, N_CLASSES) probability tensor
    """
    if rng is None:
        rng = np.random.RandomState(42)

    grid = np.array(initial_grid, dtype=np.int16)
    H, W = grid.shape

    ocean_mask = grid == OCEAN
    mountain_mask = grid == MOUNTAIN
    mutable = ~(ocean_mask | mountain_mask)

    result_counts = np.zeros((H, W, N_CLASSES), dtype=np.int32)
    result_counts[ocean_mask, 0] = n_sims
    result_counts[mountain_mask, 5] = n_sims

    cls_init = grid_to_class(grid)

    for sim in range(n_sims):
        cls_grid = cls_init.copy()
        code_grid = grid.copy()

        for step in range(N_STEPS):
            n_civ_bin, n_forest_bin, n_ocean_bin, dist_bin = \
                compute_features_vectorized(cls_grid, code_grid, H, W)

            # Index into probability table for all cells at once
            # probs_table shape: (50, 6, 5, 5, 4, 4, 6)
            per_cell_probs = probs_table[step, cls_grid, n_civ_bin, n_forest_bin, n_ocean_bin, dist_bin]
            # per_cell_probs shape: (H, W, 6)

            # Vectorized multinomial sampling
            cumprobs = np.cumsum(per_cell_probs, axis=-1)
            u = rng.random((H, W, 1))
            new_cls = (u >= cumprobs).sum(axis=-1).astype(np.int8)
            new_cls = np.minimum(new_cls, N_CLASSES - 1)

            # Only update mutable cells
            cls_grid = np.where(mutable, new_cls, cls_grid)

            # Update code_grid for dist_to_civ computation
            code_grid = _CLASS_TO_CODE[cls_grid]
            code_grid[ocean_mask] = OCEAN
            code_grid[mountain_mask] = MOUNTAIN

        # Record final state
        for c in range(N_CLASSES):
            result_counts[:, :, c] += (cls_grid == c).astype(np.int32)

    # Normalize
    result = result_counts.astype(np.float64) / n_sims
    result = np.maximum(result, 1e-6)
    result /= result.sum(axis=2, keepdims=True)
    return result


def build_gt_lookup(exclude_rounds=None):
    """Build lookup tables from ground truth data (like neighborhood_predictor)."""
    from collections import defaultdict

    def cell_to_type(code):
        if code == 1: return "settlement"
        elif code == 2: return "port"
        elif code == 4: return "forest"
        elif code in (11, 0): return "plains"
        else: return None

    def dist_bin(d):
        if d <= 1: return 0
        elif d <= 3: return 1
        elif d <= 6: return 2
        else: return 3

    fine_data = defaultdict(list)
    mid_data = defaultdict(list)
    coarse_data = defaultdict(list)
    type_data = defaultdict(list)

    for path in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(path.stem.split("_")[0].replace("round", ""))
        if exclude_rounds and rn in exclude_rounds:
            continue
        data = json.loads(path.read_text())
        if "ground_truth" not in data or "initial_grid" not in data:
            continue

        ig = np.array(data["initial_grid"])
        gt = np.array(data["ground_truth"])
        H, W = ig.shape

        settle_mask = ((ig == 1) | (ig == 2)).astype(np.int32)
        forest_mask = (ig == 4).astype(np.int32)
        ocean_mask_arr = (ig == OCEAN).astype(np.int32)

        def count_neighbors(mask):
            padded = np.pad(mask, 1, mode='constant', constant_values=0)
            result = np.zeros_like(mask, dtype=np.int32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    result += padded[1+dy:H+1+dy, 1+dx:W+1+dx]
            return result

        n_settle = count_neighbors(settle_mask)
        n_forest = count_neighbors(forest_mask)
        n_ocean = count_neighbors(ocean_mask_arr)

        civ_mask = (ig == 1) | (ig == 2)
        if civ_mask.any():
            dist_map = distance_transform_cdt(~civ_mask, metric='taxicab')
        else:
            dist_map = np.full((H, W), 99, dtype=np.int32)
        dist_bins = np.vectorize(dist_bin)(dist_map)

        for y in range(H):
            for x in range(W):
                t = cell_to_type(ig[y, x])
                if t is None:
                    continue
                prob = gt[y, x]
                ns = min(int(n_settle[y, x]), 4)
                nf = min(int(n_forest[y, x]), 4)
                no = min(int(n_ocean[y, x]), 3)
                db = int(dist_bins[y, x])

                fine_key = (t, ns, nf, no, db)
                fine_data[fine_key].append(prob)
                ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
                mid_key = (t, ns_bin, 1 if no > 0 else 0, db)
                mid_data[mid_key].append(prob)
                coarse_key = (t, db)
                coarse_data[coarse_key].append(prob)
                type_key = (t,)
                type_data[type_key].append(prob)

    fine = {k: np.mean(v, axis=0) for k, v in fine_data.items() if len(v) >= 10}
    mid = {k: np.mean(v, axis=0) for k, v in mid_data.items() if len(v) >= 10}
    coarse = {k: np.mean(v, axis=0) for k, v in coarse_data.items()}
    type_table = {k: np.mean(v, axis=0) for k, v in type_data.items()}

    return fine, mid, coarse, type_table


def predict_gt_lookup(initial_grid, fine, mid, coarse, type_table):
    """Predict using GT lookup tables."""
    def cell_to_type(code):
        if code == 1: return "settlement"
        elif code == 2: return "port"
        elif code == 4: return "forest"
        elif code in (11, 0): return "plains"
        else: return None

    def dist_bin(d):
        if d <= 1: return 0
        elif d <= 3: return 1
        elif d <= 6: return 2
        else: return 3

    ig = np.array(initial_grid)
    H, W = ig.shape

    settle_mask = ((ig == 1) | (ig == 2)).astype(np.int32)
    forest_mask = (ig == 4).astype(np.int32)
    ocean_mask_arr = (ig == OCEAN).astype(np.int32)

    def count_neighbors(mask):
        padded = np.pad(mask, 1, mode='constant', constant_values=0)
        result = np.zeros_like(mask, dtype=np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                result += padded[1+dy:H+1+dy, 1+dx:W+1+dx]
        return result

    n_settle = count_neighbors(settle_mask)
    n_forest = count_neighbors(forest_mask)
    n_ocean = count_neighbors(ocean_mask_arr)

    civ_mask = (ig == 1) | (ig == 2)
    if civ_mask.any():
        dist_map = distance_transform_cdt(~civ_mask, metric='taxicab')
    else:
        dist_map = np.full((H, W), 99, dtype=np.int32)
    dist_bins = np.vectorize(dist_bin)(dist_map)

    pred = np.zeros((H, W, N_CLASSES), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            code = ig[y, x]
            if code == OCEAN:
                pred[y, x] = OCEAN_DIST
                continue
            if code == MOUNTAIN:
                pred[y, x] = MOUNTAIN_DIST
                continue

            t = cell_to_type(code)
            ns = min(int(n_settle[y, x]), 4)
            nf = min(int(n_forest[y, x]), 4)
            no = min(int(n_ocean[y, x]), 3)
            db = int(dist_bins[y, x])

            fine_key = (t, ns, nf, no, db)
            if fine_key in fine:
                pred[y, x] = fine[fine_key]
                continue
            ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
            mid_key = (t, ns_bin, 1 if no > 0 else 0, db)
            if mid_key in mid:
                pred[y, x] = mid[mid_key]
                continue
            coarse_key = (t, db)
            if coarse_key in coarse:
                pred[y, x] = coarse[coarse_key]
                continue
            type_key = (t,)
            if type_key in type_table:
                pred[y, x] = type_table[type_key]
            else:
                pred[y, x] = np.ones(N_CLASSES) / N_CLASSES

    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def predict(initial_grid, n_sims=200, probs_table=None, exclude_rounds=None,
            gt_tables=None, blend_weight=0.5):
    """Main predict function. Blends MC simulation with GT lookup."""
    if probs_table is None:
        replays = load_replays(exclude_rounds=exclude_rounds)
        probs_table = build_transition_array(replays)

    pred_mc = simulate_fast(initial_grid, probs_table, n_sims=n_sims)

    if gt_tables is None:
        gt_tables = build_gt_lookup(exclude_rounds=exclude_rounds)

    fine, mid, coarse, type_table = gt_tables
    pred_gt = predict_gt_lookup(initial_grid, fine, mid, coarse, type_table)

    # Blend
    pred = blend_weight * pred_mc + (1 - blend_weight) * pred_gt
    pred = np.maximum(pred, 1e-6)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


# ── Validation ───────────────────────────────────────────────────────────────

def kl_divergence(p, q):
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def entropy(p):
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


def validate(n_sims=200, floor=1e-6, blend_weight=0.5, mc_only=False, gt_only=False):
    """Leave-one-round-out cross-validation."""
    gt_rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(f.stem.split("_")[0].replace("round", ""))
        sn = int(f.stem.split("_")[1].replace("seed", ""))
        data = json.loads(f.read_text())
        if "ground_truth" not in data or "initial_grid" not in data:
            continue
        if rn not in gt_rounds:
            gt_rounds[rn] = {}
        gt_rounds[rn][sn] = {
            "initial_grid": data["initial_grid"],
            "ground_truth": np.array(data["ground_truth"]),
        }

    round_nums = sorted(gt_rounds.keys())
    results = []

    for held_out in round_nums:
        exclude = {held_out}

        if not gt_only:
            replays = load_replays(exclude_rounds=exclude)
            probs_table = build_transition_array(replays)

        if not mc_only:
            gt_tables = build_gt_lookup(exclude_rounds=exclude)
            fine, mid, coarse, type_table = gt_tables

        round_kls = []
        round_wkls = []

        for sn, sd in gt_rounds[held_out].items():
            if mc_only:
                pred = simulate_fast(sd['initial_grid'], probs_table, n_sims=n_sims)
            elif gt_only:
                pred = predict_gt_lookup(sd['initial_grid'], fine, mid, coarse, type_table)
            else:
                pred_mc = simulate_fast(sd['initial_grid'], probs_table, n_sims=n_sims)
                pred_gt = predict_gt_lookup(sd['initial_grid'], fine, mid, coarse, type_table)
                pred = blend_weight * pred_mc + (1 - blend_weight) * pred_gt

            pred = np.maximum(pred, floor)
            pred /= pred.sum(axis=2, keepdims=True)

            gt = sd['ground_truth']
            kl = kl_divergence(gt, pred)
            H = entropy(gt)
            wkl = H * kl
            dynamic = H > 0.01

            if dynamic.any():
                round_kls.append(float(kl[dynamic].mean()))
                round_wkls.append(float(wkl[dynamic].mean()))

        mean_kl = np.mean(round_kls) if round_kls else 0
        mean_wkl = np.mean(round_wkls) if round_wkls else 0
        results.append({
            'held_out': held_out,
            'mean_kl': float(mean_kl),
            'mean_wkl': float(mean_wkl),
        })
        print(f"  R{held_out}: KL={mean_kl:.6f} wKL={mean_wkl:.6f}")

    overall_kl = np.mean([r['mean_kl'] for r in results])
    overall_wkl = np.mean([r['mean_wkl'] for r in results])
    print(f"\n  CV Mean KL:  {overall_kl:.6f}")
    print(f"  CV Mean wKL: {overall_wkl:.6f}")
    return {'cv_kl': float(overall_kl), 'cv_wkl': float(overall_wkl), 'per_round': results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--n-sims", type=int, default=200)
    parser.add_argument("--floor", type=float, default=1e-6)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mc-only", action="store_true")
    parser.add_argument("--gt-only", action="store_true")
    parser.add_argument("--blend", type=float, default=0.5)
    args = parser.parse_args()

    if args.profile:
        replays = load_replays()
        t0 = time.time()
        probs_table = build_transition_array(replays)
        print(f"Built table in {time.time()-t0:.1f}s, shape={probs_table.shape}")

        with open(GT_DIR / "round1_seed0.json") as f:
            data = json.load(f)
        grid = np.array(data['initial_grid'], dtype=np.int16)

        t0 = time.time()
        pred = simulate_fast(grid, probs_table, n_sims=200)
        print(f"200 sims in {time.time()-t0:.1f}s")
        print(f"Pred shape: {pred.shape}, sum check: {pred[5,5].sum():.4f}")

    elif args.validate:
        mode = "mc_only" if args.mc_only else ("gt_only" if args.gt_only else f"blend={args.blend}")
        print(f"Validating n_sims={args.n_sims}, floor={args.floor}, mode={mode}")
        t0 = time.time()
        result = validate(n_sims=args.n_sims, floor=args.floor,
                         blend_weight=args.blend, mc_only=args.mc_only,
                         gt_only=args.gt_only)
        print(f"Took {time.time()-t0:.1f}s")
