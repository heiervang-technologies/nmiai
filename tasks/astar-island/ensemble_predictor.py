#!/usr/bin/env python3
"""Ensemble predictor for Astar Island.

Blends neighborhood_predictor and predictor (HistGBT hybrid) outputs
with a uniform prior. Uses leave-one-round-out CV to find optimal weights,
plus per-cell adaptive blending based on estimated entropy.

Key insight: probability mixtures are SAFER than model selection because
cross-entropy is convex in q — a blend can never be worse than the worst
component (and is usually better than both).
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
N_CLASSES = 6

# ── Import predictors ─────────────────────────────────────────────────────

import neighborhood_predictor as nbr_mod
import predictor as hgbt_mod


# ── Helpers ────────────────────────────────────────────────────────────────

def floor_renorm(pred, floor=0.01):
    """Floor probabilities and renormalize."""
    pred = np.maximum(pred, floor)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def weighted_kl_divergence(p, q):
    """Entropy-weighted KL divergence (the actual scoring metric)."""
    p_safe = np.maximum(p, 1e-12)
    q_safe = np.maximum(q, 1e-12)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=2)
    weights = 0.25 + entropy
    kl = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=2)
    return float(np.sum(weights * kl) / np.sum(weights))


def cell_entropy(pred):
    """Per-cell Shannon entropy from a probability tensor (H, W, C)."""
    p = np.maximum(pred, 1e-12)
    return -np.sum(p * np.log(p), axis=2)


def round_number_from_path(path):
    match = re.search(r"round(\d+)_seed\d+\.json$", path.name)
    return int(match.group(1))


# ── Predictor wrappers ─────────────────────────────────────────────────────

def _make_nbr_predictor(train_paths):
    """Build neighborhood predictor trained on given paths."""
    from collections import defaultdict
    fine_data = defaultdict(list)
    mid_data = defaultdict(list)
    coarse_data = defaultdict(list)
    type_data = defaultdict(list)

    for path in train_paths:
        with open(path) as f:
            data = json.load(f)
        ig = np.array(data["initial_grid"])
        gt = np.array(data["ground_truth"])
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

    MIN_FINE = 10
    MIN_MID = 10
    fine_table = {k: np.mean(v, axis=0) for k, v in fine_data.items() if len(v) >= MIN_FINE}
    mid_table = {k: np.mean(v, axis=0) for k, v in mid_data.items() if len(v) >= MIN_MID}
    coarse_table = {k: np.mean(v, axis=0) for k, v in coarse_data.items()}
    type_table = {k: np.mean(v, axis=0) for k, v in type_data.items()}

    def pred_fn(initial_grid):
        return nbr_mod.predict(initial_grid, fine_table, mid_table, coarse_table, type_table)
    return pred_fn


def _make_hgbt_predictor(train_paths):
    """Build HistGBT hybrid predictor trained on given paths."""
    X_basic, X_phase, X_dsl, Y, _, rule_names = hgbt_mod.load_training_data(train_paths)
    model = hgbt_mod.HybridPriorModel().fit(X_basic, X_phase, X_dsl, Y, rule_names)

    def pred_fn(initial_grid):
        return hgbt_mod.build_prediction(initial_grid, model, observations=[])
    return pred_fn


# ── Ensemble weight optimization ──────────────────────────────────────────

def _collect_predictions(train_paths, test_paths):
    """Get predictions from both models and ground truth for test set."""
    nbr_fn = _make_nbr_predictor(train_paths)
    hgbt_fn = _make_hgbt_predictor(train_paths)

    results = []
    for path in test_paths:
        data = json.loads(path.read_text())
        ig = data["initial_grid"]
        gt = np.array(data["ground_truth"], dtype=np.float64)

        pred_nbr = nbr_fn(ig)
        pred_hgbt = hgbt_fn(ig)
        uniform = np.full_like(gt, 1.0 / N_CLASSES)

        results.append({
            "gt": gt,
            "nbr": pred_nbr,
            "hgbt": pred_hgbt,
            "uniform": uniform,
            "ig": np.array(ig),
        })
    return results


def _blend_and_score(w, predictions_list):
    """Score a global blend across all test examples."""
    w_nbr, w_hgbt, w_uni = w[0], w[1], w[2]
    total_kl = 0.0
    count = 0
    for preds in predictions_list:
        blended = w_nbr * preds["nbr"] + w_hgbt * preds["hgbt"] + w_uni * preds["uniform"]
        blended = floor_renorm(blended)
        total_kl += weighted_kl_divergence(preds["gt"], blended)
        count += 1
    return total_kl / count if count > 0 else 1e6


def _adaptive_blend_and_score(params, predictions_list):
    """Score adaptive blend with entropy-dependent weight interpolation.

    params: [w_nbr_low, w_hgbt_low, w_uni_low, w_nbr_high, w_hgbt_high, w_uni_high]
    For low-entropy cells use (low weights), for high-entropy cells use (high weights),
    interpolate between based on cell entropy.
    """
    w_low = np.array(params[:3])
    w_high = np.array(params[3:6])

    total_kl = 0.0
    count = 0
    for preds in predictions_list:
        # Estimate per-cell entropy from the average of predictors
        avg_pred = 0.5 * preds["nbr"] + 0.5 * preds["hgbt"]
        ent = cell_entropy(avg_pred)  # (H, W)

        # Normalize entropy to [0, 1]: max entropy = log(6) ~ 1.79
        max_ent = np.log(N_CLASSES)
        alpha = np.clip(ent / max_ent, 0.0, 1.0)  # (H, W)

        # Interpolate weights per cell
        alpha_3d = alpha[:, :, np.newaxis]  # (H, W, 1)
        w_cell = (1.0 - alpha_3d) * w_low + alpha_3d * w_high  # (H, W, 3)

        blended = (w_cell[:, :, 0:1] * preds["nbr"]
                   + w_cell[:, :, 1:2] * preds["hgbt"]
                   + w_cell[:, :, 2:3] * preds["uniform"])
        blended = floor_renorm(blended)
        total_kl += weighted_kl_divergence(preds["gt"], blended)
        count += 1
    return total_kl / count if count > 0 else 1e6


def optimize_weights():
    """Leave-one-round-out CV to find optimal blend weights."""
    gt_files = sorted(GT_DIR.glob("round*_seed*.json"))
    rounds = sorted({round_number_from_path(p) for p in gt_files})

    if len(rounds) < 2:
        log.warning("Not enough rounds for CV, using defaults")
        return np.array([0.4, 0.4, 0.2]), np.array([0.4, 0.4, 0.2]), np.array([0.3, 0.5, 0.2])

    all_predictions = []
    for holdout in rounds:
        train = [p for p in gt_files if round_number_from_path(p) != holdout]
        test = [p for p in gt_files if round_number_from_path(p) == holdout]
        preds = _collect_predictions(train, test)
        all_predictions.extend(preds)

    log.info(f"Collected {len(all_predictions)} test examples across {len(rounds)} CV folds")

    # Optimize global weights
    best_global = None
    best_global_score = 1e9
    # Grid search + refine
    for w1 in np.arange(0.1, 0.9, 0.1):
        for w2 in np.arange(0.05, 0.9 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0.01:
                continue
            w = np.array([w1, w2, w3])
            score = _blend_and_score(w, all_predictions)
            if score < best_global_score:
                best_global_score = score
                best_global = w.copy()

    log.info(f"Best global weights: nbr={best_global[0]:.2f} hgbt={best_global[1]:.2f} "
             f"uni={best_global[2]:.2f} score={best_global_score:.6f}")

    # Optimize adaptive weights using scipy
    x0 = np.array([
        best_global[0], best_global[1], best_global[2],  # low-entropy
        best_global[0], best_global[1], best_global[2],  # high-entropy
    ])

    def objective(params):
        # Normalize each set of weights to sum to 1
        w_low = np.abs(params[:3])
        w_low /= w_low.sum()
        w_high = np.abs(params[3:6])
        w_high /= w_high.sum()
        return _adaptive_blend_and_score(
            np.concatenate([w_low, w_high]), all_predictions
        )

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    w_low_opt = np.abs(result.x[:3])
    w_low_opt /= w_low_opt.sum()
    w_high_opt = np.abs(result.x[3:6])
    w_high_opt /= w_high_opt.sum()

    adaptive_score = _adaptive_blend_and_score(
        np.concatenate([w_low_opt, w_high_opt]), all_predictions
    )

    log.info(f"Adaptive weights (low entropy): nbr={w_low_opt[0]:.3f} hgbt={w_low_opt[1]:.3f} "
             f"uni={w_low_opt[2]:.3f}")
    log.info(f"Adaptive weights (high entropy): nbr={w_high_opt[0]:.3f} hgbt={w_high_opt[1]:.3f} "
             f"uni={w_high_opt[2]:.3f}")
    log.info(f"Adaptive score: {adaptive_score:.6f} (global: {best_global_score:.6f})")

    # Use adaptive if it's better
    if adaptive_score < best_global_score:
        log.info("Using adaptive blending (better than global)")
        return best_global, w_low_opt, w_high_opt
    else:
        log.info("Using global blending (adaptive not better)")
        return best_global, best_global, best_global


# ── Cached state ───────────────────────────────────────────────────────────

_cached = {}


def _ensure_models():
    """Build models and optimize weights once, cache results."""
    if _cached:
        return

    log.info("Building ensemble: training both predictors on all ground truth...")
    gt_files = sorted(GT_DIR.glob("round*_seed*.json"))

    # Optimize weights via CV
    log.info("Optimizing blend weights via leave-one-round-out CV...")
    global_w, w_low, w_high = optimize_weights()
    _cached["global_w"] = global_w
    _cached["w_low"] = w_low
    _cached["w_high"] = w_high

    # Build final models on ALL data
    _cached["nbr_fn"] = _make_nbr_predictor(gt_files)
    _cached["hgbt_fn"] = _make_hgbt_predictor(gt_files)

    log.info("Ensemble ready.")


# ── Public API ─────────────────────────────────────────────────────────────

def predict(initial_grid):
    """Predict 40x40x6 probability tensor from initial grid.

    Blends neighborhood predictor and HistGBT predictor with uniform prior,
    using per-cell adaptive weights based on estimated entropy.
    """
    _ensure_models()

    pred_nbr = _cached["nbr_fn"](initial_grid)
    pred_hgbt = _cached["hgbt_fn"](initial_grid)
    uniform = np.full_like(pred_nbr, 1.0 / N_CLASSES)

    w_low = _cached["w_low"]
    w_high = _cached["w_high"]

    # Estimate per-cell entropy from average of predictors
    avg_pred = 0.5 * pred_nbr + 0.5 * pred_hgbt
    ent = cell_entropy(avg_pred)  # (H, W)
    max_ent = np.log(N_CLASSES)
    alpha = np.clip(ent / max_ent, 0.0, 1.0)
    alpha_3d = alpha[:, :, np.newaxis]  # (H, W, 1)

    # Interpolate blend weights based on cell entropy
    w_cell = (1.0 - alpha_3d) * w_low + alpha_3d * w_high  # (H, W, 3)

    blended = (w_cell[:, :, 0:1] * pred_nbr
               + w_cell[:, :, 1:2] * pred_hgbt
               + w_cell[:, :, 2:3] * uniform)

    # Floor and renormalize — mixtures of valid dists are valid,
    # but floor ensures no class goes to zero
    blended = floor_renorm(blended)

    return blended


if __name__ == "__main__":
    # Quick self-test
    _ensure_models()
    gt_files = sorted(GT_DIR.glob("round*_seed*.json"))
    data = json.loads(gt_files[0].read_text())
    pred = predict(data["initial_grid"])
    gt = np.array(data["ground_truth"], dtype=np.float64)
    score = weighted_kl_divergence(gt, pred)
    log.info(f"Self-test score on {gt_files[0].name}: {score:.6f}")
    log.info(f"Prediction shape: {pred.shape}, sum range: [{pred.sum(axis=2).min():.4f}, {pred.sum(axis=2).max():.4f}]")
