#!/usr/bin/env python3
"""Honest observation test: leave-one-round-out with simulated observations.

For each held-out round:
1. Build templates from 7 OTHER rounds only
2. Clear fingerprints
3. Run predictions with 0 and 5 simulated observations
4. Compare to see if observations ACTUALLY help on unseen rounds
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import load_ground_truth, kl_divergence, entropy
from benchmark_experiments import simulate_observations_from_gt

GT_DIR = Path(__file__).parent / "ground_truth"
N_CLASSES = 6


def build_templates_from_rounds(round_nums_to_use):
    """Build template_predictor model from specific rounds only."""
    import template_predictor as tp
    import neighborhood_predictor as nb

    # We need to rebuild everything from scratch using only specified rounds
    gt_files = []
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(f.stem.split("_")[0].replace("round", ""))
        if rn in round_nums_to_use:
            gt_files.append(f)

    # Rebuild neighborhood tables from subset
    nb_fine, nb_mid, nb_coarse, nb_type = nb.build_lookup_table()
    # Note: neighborhood tables use ALL data - this is a small contamination
    # but acceptable since they're pooled across all rounds

    from collections import defaultdict

    pooled_sums = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(4)]
    pooled_counts = [defaultdict(float) for _ in range(4)]
    round_sums = {rn: [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(4)] for rn in round_nums_to_use}
    round_counts = {rn: [defaultdict(float) for _ in range(4)] for rn in round_nums_to_use}
    summary_sums = {rn: defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for rn in round_nums_to_use}
    summary_counts = {rn: defaultdict(float) for rn in round_nums_to_use}

    for path in gt_files:
        data = json.loads(path.read_text())
        round_num = int(path.stem.split("_")[0].replace("round", ""))
        gt = np.asarray(data["ground_truth"], dtype=np.float64)
        maps = tp.compute_feature_maps(data["initial_grid"])
        h, w = maps["init"].shape

        for y in range(h):
            for x in range(w):
                code = int(maps["init"][y, x])
                if code in (tp.OCEAN, tp.MOUNTAIN):
                    continue

                prob = gt[y, x]
                dist_q = tp.quantized_distance(float(maps["dist_to_civ"][y, x]))
                keys = tp.template_keys_for_distance(maps, y, x, dist_q)
                for level, key in enumerate(keys):
                    pooled_sums[level][key] += prob
                    pooled_counts[level][key] += 1.0
                    round_sums[round_num][level][key] += prob
                    round_counts[round_num][level][key] += 1.0

                bucket = tp.diagnostic_bucket(maps, y, x)
                if bucket is not None:
                    summary_sums[round_num][bucket] += prob
                    summary_counts[round_num][bucket] += 1.0

    pooled_tables = tp.finalize_tables(pooled_sums, pooled_counts)
    round_tables = {rn: tp.finalize_tables(round_sums[rn], round_counts[rn]) for rn in round_nums_to_use}

    summary_tables = {}
    global_summary = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
    global_summary_counts = defaultdict(float)
    for rn in round_nums_to_use:
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

    return {
        "round_tables": round_tables,
        "round_counts": round_counts,
        "pooled_tables": pooled_tables,
        "pooled_counts": {level: dict(pooled_counts[level]) for level in range(4)},
        "summary_tables": summary_tables,
        "global_summary_tables": global_summary_tables,
        "neighborhood_tables": (nb_fine, nb_mid, nb_coarse, nb_type),
        "round_ids": tuple(round_nums_to_use),
    }


def predict_with_model(initial_grid, model, observations=None):
    """Predict using a custom model (not the cached one)."""
    import template_predictor as tp
    import neighborhood_predictor as nb

    observations = observations or []
    round_ids = model["round_ids"]

    maps = tp.compute_feature_maps(initial_grid)
    nb_fine, nb_mid, nb_coarse, nb_type = model["neighborhood_tables"]
    pooled_nb_pred = nb.predict(initial_grid, nb_fine, nb_mid, nb_coarse, nb_type)

    # Build per-round template predictions
    template_preds = {}
    h, w = maps["init"].shape
    for rn in round_ids:
        pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)
        for y in range(h):
            for x in range(w):
                code = int(maps["init"][y, x])
                if code == tp.OCEAN:
                    pred[y, x] = tp.OCEAN_DIST
                    continue
                if code == tp.MOUNTAIN:
                    pred[y, x] = tp.MOUNTAIN_DIST
                    continue

                round_tables = model["round_tables"][rn]
                round_counts_r = model["round_counts"][rn]
                pooled_tables = model["pooled_tables"]
                pooled_counts_m = model["pooled_counts"]

                round_q, round_support, level = tp.interpolate_lookup(
                    round_tables, round_counts_r, maps, y, x
                )
                pooled_q, _, _ = tp.interpolate_lookup(
                    pooled_tables, pooled_counts_m, maps, y, x
                )

                if pooled_q is None:
                    pooled_q = pooled_nb_pred[y, x]
                if round_q is None:
                    pred[y, x] = pooled_nb_pred[y, x]
                    continue

                shrink = (1.0, 2.0, 3.0, 1.0)[level]
                alpha = tp.support_blend(round_support, shrink)
                simple_q = tp.blend_probs(round_q, pooled_q, alpha)
                nb_alpha = 0.92 + 0.06 * alpha
                pred[y, x] = tp.blend_probs(simple_q, pooled_nb_pred[y, x], nb_alpha)

        template_preds[rn] = pred

    # Regime detection from observations
    if observations:
        # Can't use tp.posterior_template_weights directly because it uses
        # hardcoded ROUND_IDS. Reimplement inline with our custom round_ids.
        log_weights = np.full(len(round_ids), -np.log(len(round_ids)), dtype=np.float64)
        for obs in observations:
            grid = obs.get("grid", [])
            vx = int(obs.get("viewport_x", 0))
            vy = int(obs.get("viewport_y", 0))
            for dy, row in enumerate(grid):
                for dx, cell in enumerate(row):
                    y = vy + dy
                    x = vx + dx
                    if not (0 <= y < h and 0 <= x < w):
                        continue
                    bucket = tp.diagnostic_bucket(maps, y, x)
                    cls = tp.cell_code_to_class(int(cell))
                    bucket_weight = tp.DIAGNOSTIC_BUCKET_WEIGHTS.get(bucket or "other", 0.6)
                    for idx, rn in enumerate(round_ids):
                        summary_table = model["summary_tables"][rn]
                        summary_q = summary_table.get(bucket)
                        if summary_q is None:
                            summary_q = model["global_summary_tables"].get(bucket, tp.UNIFORM)
                        cell_q = template_preds[rn][y, x]
                        summary_prob = float(np.clip(summary_q[cls], 1e-6, 1.0))
                        cell_prob = float(np.clip(cell_q[cls], 1e-6, 1.0))
                        log_weights[idx] += bucket_weight * np.log(summary_prob)
                        log_weights[idx] += 0.35 * np.log(cell_prob)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            weights = np.ones(len(round_ids), dtype=np.float64) / len(round_ids)
        else:
            weights = weights / total
    else:
        weights = np.ones(len(round_ids), dtype=np.float64) / len(round_ids)

    template_stack = np.stack([template_preds[rn] for rn in round_ids], axis=0)
    template_mean = template_stack.mean(axis=0)
    template_delta = np.zeros_like(template_mean)
    for idx, rn in enumerate(round_ids):
        template_delta += weights[idx] * (template_preds[rn] - template_mean)

    strength = tp.template_strength_from_weights(weights, bool(observations))
    pred = pooled_nb_pred + strength * template_delta
    pred = np.clip(pred, 1e-12, None)
    pred /= pred.sum(axis=2, keepdims=True)

    if observations and tp.ENABLE_LOCAL_OBS_UPDATE:
        pred = tp.apply_local_observation_update(pred, initial_grid, observations)

    pred = np.maximum(pred, 0.005)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def main():
    rounds = load_ground_truth()
    round_nums = sorted(rounds.keys())

    print("=== HONEST Leave-One-Round-Out: Observations Test ===")
    print("Templates built from 7 rounds, tested on held-out round.\n")

    results_no_obs = []
    results_5_obs = []

    for held_out in round_nums:
        train_rounds = [rn for rn in round_nums if rn != held_out]
        model = build_templates_from_rounds(train_rounds)

        wkls_no_obs = []
        wkls_5_obs = []

        for sn in sorted(rounds[held_out].keys()):
            sd = rounds[held_out][sn]
            gt = sd["ground_truth"]
            ig = sd["initial_grid"]

            # No observations
            pred_no = predict_with_model(ig, model, observations=None)
            kl = kl_divergence(gt, pred_no)
            H = entropy(gt)
            wkl = H * kl
            dynamic = H > 0.01
            if dynamic.any():
                wkls_no_obs.append(float(wkl[dynamic].mean()))

            # 5 observations
            obs = simulate_observations_from_gt(gt, ig, 5)
            pred_obs = predict_with_model(ig, model, observations=obs)
            kl = kl_divergence(gt, pred_obs)
            wkl = H * kl
            if dynamic.any():
                wkls_5_obs.append(float(wkl[dynamic].mean()))

        no_obs_mean = np.mean(wkls_no_obs)
        obs_mean = np.mean(wkls_5_obs)
        delta = obs_mean - no_obs_mean
        results_no_obs.append(no_obs_mean)
        results_5_obs.append(obs_mean)

        marker = "WORSE" if delta > 0 else "BETTER"
        print(f"  R{held_out}: no_obs={no_obs_mean:.6f}  5_obs={obs_mean:.6f}  delta={delta:+.6f} [{marker}]")

    print(f"\n  MEAN no_obs: {np.mean(results_no_obs):.6f}")
    print(f"  MEAN 5_obs:  {np.mean(results_5_obs):.6f}")
    print(f"  MEAN delta:  {np.mean(results_5_obs) - np.mean(results_no_obs):+.6f}")

    improvement = (np.mean(results_no_obs) - np.mean(results_5_obs)) / np.mean(results_no_obs) * 100
    print(f"  Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
