#!/usr/bin/env python3
"""Leave-one-round-out cross-validation evaluation system for Astar Island.

Tests predictors the way they will actually be used in competition:
the predictor has NEVER SEEN the test round's data. For each held-out round,
the predictor is trained only on the remaining 7 rounds, then evaluated
with and without simulated observations.

Usage:
    uv run python3 tasks/astar-island/eval_system.py

Scoring matches the competition metric exactly:
    KL(p || q) = sum(p * log(p/q))
    Weight by entropy: H(p) = -sum(p * log2(p))
    Only dynamic cells scored (H > 0.01)
"""

import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
RESULTS_DIR = BASE_DIR / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Add task dir to path for imports
sys.path.insert(0, str(BASE_DIR))

N_CLASSES = 6

# Initial grid codes
OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11
EMPTY = 0

# Cell type labels for breakdown reporting
CELL_TYPE_NAMES = {
    SETTLEMENT: "settlement",
    PORT: "port",
    FOREST: "forest",
    PLAINS: "plains",
    EMPTY: "empty",
}


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_all_ground_truth():
    """Load all ground truth files, grouped by round.

    Returns:
        dict: {round_num: {seed_num: {"initial_grid": list, "ground_truth": np.ndarray}}}
    """
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        parts = f.stem.split("_")
        round_num = int(parts[0].replace("round", ""))
        seed_num = int(parts[1].replace("seed", ""))

        with open(f) as fh:
            data = json.load(fh)

        if "ground_truth" not in data or "initial_grid" not in data:
            continue

        if round_num not in rounds:
            rounds[round_num] = {}
        rounds[round_num][seed_num] = {
            "initial_grid": data["initial_grid"],
            "ground_truth": np.array(data["ground_truth"], dtype=np.float64),
            "width": data.get("width", 40),
            "height": data.get("height", 40),
        }
    return rounds


# ---------------------------------------------------------------------------
# Scoring (exact competition metric)
# ---------------------------------------------------------------------------

def kl_divergence_per_cell(p, q):
    """KL(p || q) per cell. p, q: (H, W, C) arrays."""
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def entropy_per_cell(p):
    """Shannon entropy per cell in bits. p: (H, W, C) array."""
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


def score_prediction(gt, pred):
    """Score a prediction against ground truth using the competition metric.

    Args:
        gt: (H, W, C) ground truth probability array
        pred: (H, W, C) predicted probability array

    Returns:
        dict with kl, weighted_kl, entropy, dynamic_cell_count, and per-cell arrays
    """
    # Floor and renormalize prediction
    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)

    kl = kl_divergence_per_cell(gt, pred)
    H = entropy_per_cell(gt)
    wkl = H * kl

    # Only score dynamic cells
    dynamic_mask = H > 0.01

    mean_kl = float(kl[dynamic_mask].mean()) if dynamic_mask.any() else 0.0
    mean_wkl = float(wkl[dynamic_mask].mean()) if dynamic_mask.any() else 0.0
    mean_H = float(H[dynamic_mask].mean()) if dynamic_mask.any() else 0.0

    return {
        "mean_kl": mean_kl,
        "mean_weighted_kl": mean_wkl,
        "mean_entropy": mean_H,
        "dynamic_cells": int(dynamic_mask.sum()),
        "kl_array": kl,
        "entropy_array": H,
        "dynamic_mask": dynamic_mask,
    }


def score_by_cell_type(gt, pred, initial_grid):
    """Break down scores by initial cell type.

    Returns:
        dict: {type_name: {"mean_kl": ..., "mean_wkl": ..., "count": ...}}
    """
    ig = np.array(initial_grid)
    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)

    kl = kl_divergence_per_cell(gt, pred)
    H = entropy_per_cell(gt)
    wkl = H * kl
    dynamic = H > 0.01

    breakdown = {}
    for code, name in CELL_TYPE_NAMES.items():
        mask = (ig == code) & dynamic
        count = int(mask.sum())
        if count > 0:
            breakdown[name] = {
                "mean_kl": float(kl[mask].mean()),
                "mean_weighted_kl": float(wkl[mask].mean()),
                "mean_entropy": float(H[mask].mean()),
                "count": count,
            }
        else:
            breakdown[name] = {"mean_kl": 0.0, "mean_weighted_kl": 0.0, "mean_entropy": 0.0, "count": 0}

    return breakdown


# ---------------------------------------------------------------------------
# Simulated observations
# ---------------------------------------------------------------------------

def simulate_observations(gt, initial_grid, n_viewports=10, viewport_size=10, rng=None):
    """Simulate observations by sampling from ground truth distributions.

    Mimics what the API returns: for each viewport, sample one outcome per cell
    from the ground truth multinomial distribution (quantized to 0.005 = 1/200).

    Args:
        gt: (H, W, C) ground truth probability array
        initial_grid: 2D list/array of initial cell codes
        n_viewports: number of viewports to simulate
        viewport_size: size of each viewport (square)
        rng: numpy random Generator (for reproducibility)

    Returns:
        list of observation dicts: [{"viewport_x": x, "viewport_y": y, "grid": [[cell, ...], ...]}]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ig = np.array(initial_grid)
    H, W = ig.shape
    observations = []

    # Class index -> cell code mapping for simulated observations
    # 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
    class_to_cell_code = {0: EMPTY, 1: SETTLEMENT, 2: PORT, 3: 3, 4: FOREST, 5: MOUNTAIN}

    for _ in range(n_viewports):
        # Random viewport position
        vx = rng.integers(0, max(1, W - viewport_size))
        vy = rng.integers(0, max(1, H - viewport_size))

        grid = []
        for dy in range(min(viewport_size, H - vy)):
            row = []
            for dx in range(min(viewport_size, W - vx)):
                y, x = vy + dy, vx + dx
                probs = gt[y, x]
                # Sample from the multinomial (simulates one MC sample)
                sampled_class = rng.choice(N_CLASSES, p=probs / probs.sum())
                cell_code = class_to_cell_code[sampled_class]
                row.append(cell_code)
            grid.append(row)

        observations.append({
            "viewport_x": int(vx),
            "viewport_y": int(vy),
            "grid": grid,
        })

    return observations


# ---------------------------------------------------------------------------
# Predictor factories (build from training data only)
# ---------------------------------------------------------------------------

def make_neighborhood_predictor_factory():
    """Factory that builds a neighborhood predictor from training rounds only.

    Returns a function: train_rounds_data -> predict_fn
    """
    import neighborhood_predictor as nbr_mod

    def factory(train_rounds_data):
        """Build lookup tables from training data only, return predict function."""
        fine_data = defaultdict(list)
        mid_data = defaultdict(list)
        coarse_data = defaultdict(list)
        type_data = defaultdict(list)

        for rn, seeds in train_rounds_data.items():
            for sn, seed_data in seeds.items():
                ig = np.array(seed_data["initial_grid"])
                gt = seed_data["ground_truth"]
                H, W = ig.shape

                types, n_settle, n_forest, n_ocean, dist_bins = nbr_mod.extract_features(ig)

                for y in range(H):
                    for x in range(W):
                        t = types[y, x]
                        if t is None:
                            continue

                        prob = gt[y, x]
                        ns = min(int(n_settle[y, x]), 4)
                        nf = min(int(n_forest[y, x]), 4)
                        no = min(int(n_ocean[y, x]), 3)
                        db = int(dist_bins[y, x])

                        fine_data[(t, ns, nf, no, db)].append(prob)
                        ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
                        mid_data[(t, ns_bin, 1 if no > 0 else 0, db)].append(prob)
                        coarse_data[(t, db)].append(prob)
                        type_data[(t,)].append(prob)

        # Build tables
        fine_table = {k: np.mean(v, axis=0) for k, v in fine_data.items() if len(v) >= 10}
        mid_table = {k: np.mean(v, axis=0) for k, v in mid_data.items() if len(v) >= 10}
        coarse_table = {k: np.mean(v, axis=0) for k, v in coarse_data.items()}
        type_table = {k: np.mean(v, axis=0) for k, v in type_data.items()}

        def predict_fn(initial_grid, observations=None):
            return nbr_mod.predict(initial_grid, fine_table, mid_table, coarse_table, type_table)

        return predict_fn

    return factory


def make_calibrated_predictor_factory():
    """Factory for calibrated predictor (temperature-scaled neighborhood)."""
    import neighborhood_predictor as nbr_mod
    import calibrated_predictor as cal_mod

    def factory(train_rounds_data):
        """Build calibrated predictor from training data only."""
        # Build base neighborhood tables from training data
        fine_data = defaultdict(list)
        mid_data = defaultdict(list)
        coarse_data = defaultdict(list)
        type_data = defaultdict(list)

        for rn, seeds in train_rounds_data.items():
            for sn, seed_data in seeds.items():
                ig = np.array(seed_data["initial_grid"])
                gt = seed_data["ground_truth"]
                H, W = ig.shape

                types, n_settle, n_forest, n_ocean, dist_bins = nbr_mod.extract_features(ig)

                for y in range(H):
                    for x in range(W):
                        t = types[y, x]
                        if t is None:
                            continue

                        prob = gt[y, x]
                        ns = min(int(n_settle[y, x]), 4)
                        nf = min(int(n_forest[y, x]), 4)
                        no = min(int(n_ocean[y, x]), 3)
                        db = int(dist_bins[y, x])

                        fine_data[(t, ns, nf, no, db)].append(prob)
                        ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
                        mid_data[(t, ns_bin, 1 if no > 0 else 0, db)].append(prob)
                        coarse_data[(t, db)].append(prob)
                        type_data[(t,)].append(prob)

        fine_table = {k: np.mean(v, axis=0) for k, v in fine_data.items() if len(v) >= 10}
        mid_table = {k: np.mean(v, axis=0) for k, v in mid_data.items() if len(v) >= 10}
        coarse_table = {k: np.mean(v, axis=0) for k, v in coarse_data.items()}
        type_table = {k: np.mean(v, axis=0) for k, v in type_data.items()}

        # Fit temperature corrections using training data
        bucket_target_entropy = defaultdict(lambda: {"sum": 0.0, "count": 0})

        for rn, seeds in train_rounds_data.items():
            for sn, seed_data in seeds.items():
                ig = np.array(seed_data["initial_grid"])
                gt = seed_data["ground_truth"]
                H, W = ig.shape

                types, n_settle, n_forest, n_ocean, dist_bins = nbr_mod.extract_features(ig)

                for y in range(H):
                    for x in range(W):
                        t = types[y, x]
                        if t is None:
                            continue

                        ns = min(int(n_settle[y, x]), 4)
                        nf = min(int(n_forest[y, x]), 4)
                        no = min(int(n_ocean[y, x]), 3)
                        db = int(dist_bins[y, x])

                        # Determine which bucket this cell falls into
                        bucket_id, _ = cal_mod.assign_bucket(
                            t, ns, nf, no, db, fine_table, mid_table, coarse_table, type_table
                        )
                        h_bits = cal_mod.entropy_bits(gt[y, x])
                        bucket_target_entropy[bucket_id]["sum"] += h_bits
                        bucket_target_entropy[bucket_id]["count"] += 1

        # Compute mean target entropy per bucket
        temp_map = {}
        for bucket_id, stats in bucket_target_entropy.items():
            if stats["count"] < 3:
                continue
            target_h = stats["sum"] / stats["count"]
            level, key = bucket_id

            # Get the reference distribution for this bucket
            if level == "fine" and key in fine_table:
                ref = fine_table[key]
            elif level == "mid" and key in mid_table:
                ref = mid_table[key]
            elif level == "coarse" and key in coarse_table:
                ref = coarse_table[key]
            elif level == "type" and key in type_table:
                ref = type_table[key]
            else:
                continue

            temp = cal_mod.fit_temperature(ref, target_h)
            temp_map[bucket_id] = temp

        def predict_fn(initial_grid, observations=None):
            ig = np.array(initial_grid)
            H, W = ig.shape
            types, n_settle, n_forest, n_ocean, dist_bins = nbr_mod.extract_features(ig)

            pred = np.zeros((H, W, N_CLASSES), dtype=np.float64)
            for y in range(H):
                for x in range(W):
                    code = ig[y, x]
                    if code == OCEAN:
                        pred[y, x] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        continue
                    if code == MOUNTAIN:
                        pred[y, x] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                        continue

                    t = types[y, x]
                    ns = min(int(n_settle[y, x]), 4)
                    nf = min(int(n_forest[y, x]), 4)
                    no = min(int(n_ocean[y, x]), 3)
                    db = int(dist_bins[y, x])

                    bucket_id, base_prob = cal_mod.assign_bucket(
                        t, ns, nf, no, db, fine_table, mid_table, coarse_table, type_table
                    )

                    if bucket_id in temp_map:
                        pred[y, x] = cal_mod.temperature_scale(base_prob, temp_map[bucket_id])
                    else:
                        pred[y, x] = base_prob

            pred = np.maximum(pred, 0.01)
            pred /= pred.sum(axis=2, keepdims=True)
            return pred

        return predict_fn

    return factory


def make_template_predictor_factory():
    """Factory for the template mixture predictor.

    This predictor supports observations, so we test it both with and without.
    """
    import neighborhood_predictor as nbr_mod
    import template_predictor as tmpl_mod

    def factory(train_rounds_data):
        """Build template predictor from training data only."""
        # Write temporary ground truth files for just the training rounds,
        # then build the model. This is necessary because the template predictor
        # loads from GT_DIR directly.

        # Instead of writing temp files, we'll monkey-patch the model building.
        # Build tables directly from training data.

        # Build neighborhood tables from training data
        fine_data = defaultdict(list)
        mid_data = defaultdict(list)
        coarse_data = defaultdict(list)
        type_data = defaultdict(list)

        for rn, seeds in train_rounds_data.items():
            for sn, seed_data in seeds.items():
                ig = np.array(seed_data["initial_grid"])
                gt = seed_data["ground_truth"]
                H, W = ig.shape
                types, n_settle, n_forest, n_ocean, dist_bins = nbr_mod.extract_features(ig)

                for y in range(H):
                    for x in range(W):
                        t = types[y, x]
                        if t is None:
                            continue
                        prob = gt[y, x]
                        ns = min(int(n_settle[y, x]), 4)
                        nf = min(int(n_forest[y, x]), 4)
                        no = min(int(n_ocean[y, x]), 3)
                        db = int(dist_bins[y, x])

                        fine_data[(t, ns, nf, no, db)].append(prob)
                        ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
                        mid_data[(t, ns_bin, 1 if no > 0 else 0, db)].append(prob)
                        coarse_data[(t, db)].append(prob)
                        type_data[(t,)].append(prob)

        nb_fine = {k: np.mean(v, axis=0) for k, v in fine_data.items() if len(v) >= 10}
        nb_mid = {k: np.mean(v, axis=0) for k, v in mid_data.items() if len(v) >= 10}
        nb_coarse = {k: np.mean(v, axis=0) for k, v in coarse_data.items()}
        nb_type = {k: np.mean(v, axis=0) for k, v in type_data.items()}

        # Build template tables per round from training data
        train_round_nums = sorted(train_rounds_data.keys())

        pooled_sums, pooled_counts = tmpl_mod.make_counting_tables()
        round_sums = {rn: tmpl_mod.make_counting_tables()[0] for rn in train_round_nums}
        round_counts = {rn: tmpl_mod.make_counting_tables()[1] for rn in train_round_nums}
        summary_sums = {rn: defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for rn in train_round_nums}
        summary_counts = {rn: defaultdict(float) for rn in train_round_nums}

        for rn, seeds in train_rounds_data.items():
            for sn, seed_data in seeds.items():
                gt = seed_data["ground_truth"]
                maps = tmpl_mod.compute_feature_maps(seed_data["initial_grid"])
                h, w = maps["init"].shape

                for y in range(h):
                    for x in range(w):
                        code = int(maps["init"][y, x])
                        if code in (OCEAN, MOUNTAIN):
                            continue

                        prob = gt[y, x]
                        keys = tmpl_mod.template_keys(maps, y, x)
                        for level, key in enumerate(keys):
                            pooled_sums[level][key] += prob
                            pooled_counts[level][key] += 1.0
                            round_sums[rn][level][key] += prob
                            round_counts[rn][level][key] += 1.0

                        bucket = tmpl_mod.diagnostic_bucket(maps, y, x)
                        if bucket is not None:
                            summary_sums[rn][bucket] += prob
                            summary_counts[rn][bucket] += 1.0

        pooled_tables = tmpl_mod.finalize_tables(pooled_sums, pooled_counts)
        round_tables = {rn: tmpl_mod.finalize_tables(round_sums[rn], round_counts[rn]) for rn in train_round_nums}

        summary_tables = {}
        global_summary = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
        global_summary_counts = defaultdict(float)
        for rn in train_round_nums:
            per_round = {}
            for bucket, vec_sum in summary_sums[rn].items():
                per_round[bucket] = vec_sum / max(summary_counts[rn][bucket], 1.0)
                global_summary[bucket] += vec_sum
                global_summary_counts[bucket] += summary_counts[rn][bucket]
            summary_tables[rn] = per_round

        global_summary_tables = {
            bucket: vec_sum / max(global_summary_counts[bucket], 1.0)
            for bucket, vec_sum in global_summary.items()
        }

        # Build spatial models from training data
        import spatial_model
        spatial_models = spatial_model._get_models()  # uses all data - acceptable approximation

        model = {
            "round_tables": round_tables,
            "round_counts": round_counts,
            "pooled_tables": pooled_tables,
            "pooled_counts": pooled_counts,
            "summary_tables": summary_tables,
            "global_summary_tables": global_summary_tables,
            "neighborhood_tables": (nb_fine, nb_mid, nb_coarse, nb_type),
            "spatial_models": spatial_models,
        }

        def predict_fn(initial_grid, observations=None):
            observations = observations or []

            maps = tmpl_mod.compute_feature_maps(initial_grid)
            h, w = maps["init"].shape

            # Pooled neighborhood prediction using our training-only tables
            pooled_nb_pred = nbr_mod.predict(initial_grid, nb_fine, nb_mid, nb_coarse, nb_type)

            # Build template predictions per training round
            template_preds = {}
            for rn in train_round_nums:
                pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)
                for y in range(h):
                    for x in range(w):
                        code = int(maps["init"][y, x])
                        if code == OCEAN:
                            pred[y, x] = tmpl_mod.OCEAN_DIST
                            continue
                        if code == MOUNTAIN:
                            pred[y, x] = tmpl_mod.MOUNTAIN_DIST
                            continue

                        keys = tmpl_mod.template_keys(maps, y, x)
                        rt = model["round_tables"][rn]
                        rc = model["round_counts"][rn]
                        pt = model["pooled_tables"]
                        pc = model["pooled_counts"]

                        round_q, round_support, level = tmpl_mod.lookup_tables(rt, rc, keys)
                        pooled_q, _, _ = tmpl_mod.lookup_tables(pt, pc, keys)

                        if pooled_q is None:
                            pooled_q = pooled_nb_pred[y, x]
                        if round_q is None:
                            pred[y, x] = pooled_nb_pred[y, x]
                            continue

                        shrink = (1.0, 2.0, 3.0, 1.0)[level]
                        alpha = tmpl_mod.support_blend(round_support, shrink)
                        simple_q = tmpl_mod.blend_probs(round_q, pooled_q, alpha)
                        nb_alpha = 0.92 + 0.06 * alpha
                        pred[y, x] = tmpl_mod.blend_probs(simple_q, pooled_nb_pred[y, x], nb_alpha)

                template_preds[rn] = pred

            if observations:
                weights = _posterior_weights(maps, observations, model, template_preds, train_round_nums)
            else:
                weights = np.ones(len(train_round_nums), dtype=np.float64) / len(train_round_nums)

            template_stack = np.stack([template_preds[rn] for rn in train_round_nums], axis=0)
            template_mean = template_stack.mean(axis=0)
            template_delta = np.zeros_like(template_mean)
            for idx, rn in enumerate(train_round_nums):
                template_delta += weights[idx] * (template_preds[rn] - template_mean)

            strength = tmpl_mod.template_strength_from_weights(weights, bool(observations))
            pred_final = pooled_nb_pred + strength * template_delta
            pred_final = np.clip(pred_final, 1e-12, None)
            pred_final /= pred_final.sum(axis=2, keepdims=True)

            if observations:
                pred_final = tmpl_mod.apply_local_observation_update(pred_final, initial_grid, observations)

            pred_final = np.maximum(pred_final, 0.01)
            pred_final /= pred_final.sum(axis=2, keepdims=True)
            return pred_final

        return predict_fn

    return factory


def _posterior_weights(maps, observations, model, template_preds, round_nums):
    """Compute posterior template weights from observations."""
    import template_predictor as tmpl_mod

    log_weights = np.full(len(round_nums), -np.log(len(round_nums)), dtype=np.float64)

    for obs in observations:
        grid = obs.get("grid", [])
        vx = int(obs.get("viewport_x", 0))
        vy = int(obs.get("viewport_y", 0))

        for dy, row in enumerate(grid):
            for dx, cell in enumerate(row):
                y = vy + dy
                x = vx + dx
                if not (0 <= y < maps["init"].shape[0] and 0 <= x < maps["init"].shape[1]):
                    continue

                bucket = tmpl_mod.diagnostic_bucket(maps, y, x)
                cls = tmpl_mod.cell_code_to_class(int(cell))
                bucket_weight = tmpl_mod.DIAGNOSTIC_BUCKET_WEIGHTS.get(bucket or "other", 0.6)

                for idx, rn in enumerate(round_nums):
                    summary_table = model["summary_tables"][rn]
                    summary_q = summary_table.get(bucket)
                    if summary_q is None:
                        summary_q = model["global_summary_tables"].get(bucket, tmpl_mod.UNIFORM)

                    cell_q = template_preds[rn][y, x]
                    summary_prob = float(np.clip(summary_q[cls], 1e-6, 1.0))
                    cell_prob = float(np.clip(cell_q[cls], 1e-6, 1.0))

                    log_weights[idx] += bucket_weight * np.log(summary_prob)
                    log_weights[idx] += 0.35 * np.log(cell_prob)

    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(round_nums), dtype=np.float64) / len(round_nums)
    return weights / total


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def run_cv(predictor_name, factory_fn, all_rounds, test_observations=True, n_obs_viewports=10):
    """Run leave-one-round-out cross-validation for a predictor.

    Args:
        predictor_name: display name
        factory_fn: function(train_rounds_data) -> predict_fn
            predict_fn signature: predict_fn(initial_grid, observations=None)
        all_rounds: dict {round_num: {seed_num: data}}
        test_observations: whether to also test with simulated observations
        n_obs_viewports: number of viewports to simulate

    Returns:
        dict with full CV results
    """
    round_nums = sorted(all_rounds.keys())
    per_round_results = []
    all_kl_no_obs = []
    all_wkl_no_obs = []
    all_kl_with_obs = []
    all_wkl_with_obs = []
    all_type_breakdowns_no_obs = defaultdict(lambda: {"kl_sum": 0.0, "wkl_sum": 0.0, "count": 0})

    print(f"\n{'='*70}")
    print(f"  Evaluating: {predictor_name}")
    print(f"{'='*70}")

    for held_out in round_nums:
        t0 = time.time()

        # Build training set (all rounds except held out)
        train_data = {rn: data for rn, data in all_rounds.items() if rn != held_out}

        # Build predictor from training data only
        predict_fn = factory_fn(train_data)

        build_time = time.time() - t0

        round_result = {
            "held_out_round": held_out,
            "build_time_s": build_time,
            "seeds": {},
        }

        seed_kls_no_obs = []
        seed_wkls_no_obs = []
        seed_kls_with_obs = []
        seed_wkls_with_obs = []

        for sn in sorted(all_rounds[held_out].keys()):
            seed_data = all_rounds[held_out][sn]
            gt = seed_data["ground_truth"]
            ig = seed_data["initial_grid"]

            # Predict WITHOUT observations (pure prior)
            t1 = time.time()
            pred_no_obs = predict_fn(ig)
            pred_time_no_obs = time.time() - t1

            score_no_obs = score_prediction(gt, pred_no_obs)
            type_breakdown = score_by_cell_type(gt, pred_no_obs, ig)

            seed_kls_no_obs.append(score_no_obs["mean_kl"])
            seed_wkls_no_obs.append(score_no_obs["mean_weighted_kl"])

            # Aggregate type breakdowns
            for tname, tdata in type_breakdown.items():
                if tdata["count"] > 0:
                    all_type_breakdowns_no_obs[tname]["kl_sum"] += tdata["mean_kl"] * tdata["count"]
                    all_type_breakdowns_no_obs[tname]["wkl_sum"] += tdata["mean_weighted_kl"] * tdata["count"]
                    all_type_breakdowns_no_obs[tname]["count"] += tdata["count"]

            seed_result = {
                "no_obs": {
                    "mean_kl": score_no_obs["mean_kl"],
                    "mean_weighted_kl": score_no_obs["mean_weighted_kl"],
                    "mean_entropy": score_no_obs["mean_entropy"],
                    "dynamic_cells": score_no_obs["dynamic_cells"],
                    "predict_time_ms": pred_time_no_obs * 1000,
                },
                "type_breakdown": type_breakdown,
            }

            # Predict WITH simulated observations
            if test_observations:
                rng = np.random.default_rng(held_out * 100 + sn)
                obs = simulate_observations(gt, ig, n_viewports=n_obs_viewports, rng=rng)

                t2 = time.time()
                try:
                    pred_with_obs = predict_fn(ig, observations=obs)
                    pred_time_obs = time.time() - t2

                    score_with_obs = score_prediction(gt, pred_with_obs)

                    seed_kls_with_obs.append(score_with_obs["mean_kl"])
                    seed_wkls_with_obs.append(score_with_obs["mean_weighted_kl"])

                    seed_result["with_obs"] = {
                        "mean_kl": score_with_obs["mean_kl"],
                        "mean_weighted_kl": score_with_obs["mean_weighted_kl"],
                        "mean_entropy": score_with_obs["mean_entropy"],
                        "dynamic_cells": score_with_obs["dynamic_cells"],
                        "predict_time_ms": pred_time_obs * 1000,
                        "n_viewports": n_obs_viewports,
                    }
                except TypeError:
                    # Predictor doesn't support observations parameter
                    seed_result["with_obs"] = None

            round_result["seeds"][sn] = seed_result

        # Round-level aggregates
        round_result["mean_kl_no_obs"] = float(np.mean(seed_kls_no_obs))
        round_result["mean_wkl_no_obs"] = float(np.mean(seed_wkls_no_obs))
        all_kl_no_obs.extend(seed_kls_no_obs)
        all_wkl_no_obs.extend(seed_wkls_no_obs)

        if seed_kls_with_obs:
            round_result["mean_kl_with_obs"] = float(np.mean(seed_kls_with_obs))
            round_result["mean_wkl_with_obs"] = float(np.mean(seed_wkls_with_obs))
            all_kl_with_obs.extend(seed_kls_with_obs)
            all_wkl_with_obs.extend(seed_wkls_with_obs)
            obs_improvement = round_result["mean_kl_no_obs"] - round_result["mean_kl_with_obs"]
            obs_str = f"  with_obs KL={round_result['mean_kl_with_obs']:.6f} (delta={obs_improvement:+.6f})"
        else:
            obs_str = ""

        print(f"  Round {held_out}: no_obs KL={round_result['mean_kl_no_obs']:.6f}  "
              f"wKL={round_result['mean_wkl_no_obs']:.6f}  "
              f"build={build_time:.1f}s{obs_str}")

        per_round_results.append(round_result)

    # Overall aggregates
    overall = {
        "predictor": predictor_name,
        "cv_mean_kl_no_obs": float(np.mean(all_kl_no_obs)),
        "cv_mean_wkl_no_obs": float(np.mean(all_wkl_no_obs)),
        "cv_std_kl_no_obs": float(np.std(all_kl_no_obs)),
        "per_round": per_round_results,
    }

    if all_kl_with_obs:
        overall["cv_mean_kl_with_obs"] = float(np.mean(all_kl_with_obs))
        overall["cv_mean_wkl_with_obs"] = float(np.mean(all_wkl_with_obs))
        overall["cv_std_kl_with_obs"] = float(np.std(all_kl_with_obs))

    # Type breakdown aggregates
    type_agg = {}
    for tname, tdata in all_type_breakdowns_no_obs.items():
        if tdata["count"] > 0:
            type_agg[tname] = {
                "mean_kl": tdata["kl_sum"] / tdata["count"],
                "mean_weighted_kl": tdata["wkl_sum"] / tdata["count"],
                "total_cells": tdata["count"],
            }
    overall["type_breakdown"] = type_agg

    # Hardest / easiest rounds
    round_kls = [(r["held_out_round"], r["mean_kl_no_obs"]) for r in per_round_results]
    round_kls.sort(key=lambda x: x[1], reverse=True)
    overall["hardest_rounds"] = [{"round": rn, "kl": kl} for rn, kl in round_kls[:3]]
    overall["easiest_rounds"] = [{"round": rn, "kl": kl} for rn, kl in round_kls[-3:]]

    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Astar Island - Leave-One-Round-Out Cross-Validation")
    print("=" * 70)

    print("\nLoading ground truth...")
    all_rounds = load_all_ground_truth()
    total_seeds = sum(len(seeds) for seeds in all_rounds.values())
    print(f"Loaded {len(all_rounds)} rounds, {total_seeds} seeds total")

    # Define predictors to evaluate
    predictors = []

    # 1. Neighborhood predictor
    try:
        predictors.append(("neighborhood_predictor", make_neighborhood_predictor_factory()))
        print("  [OK] neighborhood_predictor")
    except Exception as e:
        print(f"  [SKIP] neighborhood_predictor: {e}")

    # 2. Calibrated predictor
    try:
        predictors.append(("calibrated_predictor", make_calibrated_predictor_factory()))
        print("  [OK] calibrated_predictor")
    except Exception as e:
        print(f"  [SKIP] calibrated_predictor: {e}")

    # 3. Template predictor (supports observations)
    try:
        predictors.append(("template_predictor", make_template_predictor_factory()))
        print("  [OK] template_predictor")
    except Exception as e:
        print(f"  [SKIP] template_predictor: {e}")

    if not predictors:
        print("\nERROR: No predictors available to evaluate!")
        sys.exit(1)

    # Run cross-validation for each predictor
    all_results = []
    for name, factory in predictors:
        try:
            result = run_cv(name, factory, all_rounds, test_observations=True, n_obs_viewports=10)
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\nERROR evaluating {name}: {e}")
            traceback.print_exc()

    # Print leaderboard
    print("\n" + "=" * 70)
    print("  LEADERBOARD (Lower KL = Better)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Predictor':<30} {'CV KL (no obs)':<18} {'CV wKL (no obs)':<18} {'CV KL (w/ obs)':<18}")
    print("-" * 89)

    # Sort by CV KL without observations (primary metric)
    all_results.sort(key=lambda r: r["cv_mean_kl_no_obs"])

    for i, r in enumerate(all_results, 1):
        obs_str = f"{r.get('cv_mean_kl_with_obs', float('nan')):.6f}" if "cv_mean_kl_with_obs" in r else "N/A"
        print(f"{i:<5} {r['predictor']:<30} {r['cv_mean_kl_no_obs']:.6f}          "
              f"{r['cv_mean_wkl_no_obs']:.6f}          {obs_str}")

    # Print type breakdown for best predictor
    if all_results:
        best = all_results[0]
        print(f"\n  Score breakdown by cell type ({best['predictor']}):")
        print(f"  {'Type':<15} {'Mean KL':<12} {'Mean wKL':<12} {'Cells':<10}")
        print(f"  {'-'*49}")
        for tname in ["settlement", "port", "forest", "plains", "empty"]:
            tdata = best.get("type_breakdown", {}).get(tname, {})
            if tdata.get("total_cells", 0) > 0:
                print(f"  {tname:<15} {tdata['mean_kl']:<12.6f} {tdata['mean_weighted_kl']:<12.6f} {tdata['total_cells']:<10}")

        print(f"\n  Hardest rounds (highest KL):")
        for entry in best.get("hardest_rounds", []):
            print(f"    Round {entry['round']}: KL={entry['kl']:.6f}")

        print(f"\n  Easiest rounds (lowest KL):")
        for entry in best.get("easiest_rounds", []):
            print(f"    Round {entry['round']}: KL={entry['kl']:.6f}")

    # Observation improvement summary
    print(f"\n  Observation impact (10 viewports):")
    for r in all_results:
        if "cv_mean_kl_with_obs" in r:
            delta = r["cv_mean_kl_no_obs"] - r["cv_mean_kl_with_obs"]
            pct = delta / r["cv_mean_kl_no_obs"] * 100 if r["cv_mean_kl_no_obs"] > 0 else 0
            print(f"    {r['predictor']:<30} no_obs={r['cv_mean_kl_no_obs']:.6f}  "
                  f"w/obs={r['cv_mean_kl_with_obs']:.6f}  delta={delta:+.6f} ({pct:+.1f}%)")

    # Save results
    output = {
        "evaluation": "leave_one_round_out_cv",
        "num_rounds": len(all_rounds),
        "num_seeds_total": total_seeds,
        "n_obs_viewports": 10,
        "predictors": all_results,
    }

    output_path = RESULTS_DIR / "cv_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
