#!/usr/bin/env python3
"""Eval V2: End-to-end pipeline evaluation for Astar Island.

Fixes 6 gaps in eval.py:
1. Tests BTP natively (no build_model_from_data needed)
2. Full end-to-end pipeline: predict + external tau + sigma smoothing
3. Real blitz viewport selection (matches auto_watcher heuristic)
4. Paired A/B comparison with per-seed deltas
5. Competition score (100*exp(-3*wKL)) alongside wKL
6. Per-cell category breakdown integrated

Usage:
    uv run python3 eval_v2.py                          # BTP full pipeline, LOOCV
    uv run python3 eval_v2.py --no-pipeline            # BTP raw (no tau/smoothing overlay)
    uv run python3 eval_v2.py --predictor regime_predictor  # Different predictor
    uv run python3 eval_v2.py --compare btp regime     # A/B comparison
    uv run python3 eval_v2.py --obs 5 --tau 20         # Custom obs/tau (no pipeline)
    uv run python3 eval_v2.py --breakdown              # Per-cell category breakdown
"""

import argparse
import importlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    convolve,
    distance_transform_cdt,
    distance_transform_edt,
    gaussian_filter,
)

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
N_CLASSES = 6
FLOOR = 1e-6
VIEWPORT = 15
KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4


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


def wkl_to_score(wkl):
    """Convert wKL to competition score (0-100)."""
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))


def score_prediction(gt, pred):
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    kl = kl_divergence(gt, pred)
    H = entropy(gt)
    wkl = H * kl
    dynamic = H > 0.01

    if not dynamic.any():
        return {"wkl": 0.0, "score": 100.0, "dynamic_cells": 0}

    mean_wkl = float(wkl[dynamic].mean())
    return {
        "wkl": mean_wkl,
        "score": wkl_to_score(mean_wkl),
        "kl": float(kl[dynamic].mean()),
        "entropy": float(H[dynamic].mean()),
        "dynamic_cells": int(dynamic.sum()),
    }


# ---------------------------------------------------------------------------
# Cell category classification (from score_diagnosis.py)
# ---------------------------------------------------------------------------

def classify_cells(ig):
    """Classify cells into diagnostic categories."""
    h, w = ig.shape
    categories = np.empty((h, w), dtype=object)

    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN
    dist = distance_transform_edt(~civ) if civ.any() else np.full((h, w), 99.0)
    n_ocean = convolve(ocean.astype(np.int32), KERNEL, mode="constant")

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            d = float(dist[y, x])
            no = int(n_ocean[y, x])

            if code == OCEAN:
                categories[y, x] = "ocean"
            elif code == MOUNTAIN:
                categories[y, x] = "mountain"
            elif code in (SETTLEMENT, PORT):
                categories[y, x] = "settlement"
            elif no >= 2 and d <= 4:
                categories[y, x] = "coastal_frontier"
            elif d <= 1.5:
                categories[y, x] = "near_civ"
            elif d <= 5:
                categories[y, x] = "mid_range"
            elif code == FOREST:
                categories[y, x] = "deep_forest"
            else:
                categories[y, x] = "wilderness"
    return categories


def score_by_category(gt, pred, ig):
    """Breakdown wKL contribution by cell category."""
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    kl = kl_divergence(gt, pred)
    H = entropy(gt)
    wkl = H * kl
    dynamic = H > 0.01
    categories = classify_cells(ig)

    total_wkl = float(wkl[dynamic].sum()) if dynamic.any() else 0.0
    breakdown = {}
    for cat in np.unique(categories):
        mask = (categories == cat) & dynamic
        if mask.any():
            cat_wkl = float(wkl[mask].sum())
            breakdown[cat] = {
                "wkl_sum": cat_wkl,
                "wkl_mean": float(wkl[mask].mean()),
                "pct_of_total": cat_wkl / total_wkl * 100 if total_wkl > 0 else 0,
                "n_cells": int(mask.sum()),
            }
    return breakdown


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth():
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
# Blitz viewport selection (matches auto_watcher exactly)
# ---------------------------------------------------------------------------

def select_blitz_viewports(ig, n_viewports=3):
    """Select viewports using same heuristic as auto_watcher blitz.

    Scores: 3*(settlement count) + (cells at taxicab distance 1-4 from civ)
    Grid: step=3, viewport=15x15, overlap < 30%
    """
    h, w = ig.shape
    civ = (ig == SETTLEMENT) | (ig == PORT)
    cd = distance_transform_cdt(~civ, metric="taxicab") if civ.any() else np.full((h, w), 99)

    ranked = []
    for vy in range(0, max(h - VIEWPORT + 1, 1), 3):
        for vx in range(0, max(w - VIEWPORT + 1, 1), 3):
            patch_ig = ig[vy:vy + VIEWPORT, vx:vx + VIEWPORT]
            patch_cd = cd[vy:vy + VIEWPORT, vx:vx + VIEWPORT]
            sc = 3 * (patch_ig == SETTLEMENT).sum() + ((patch_cd >= 1) & (patch_cd <= 4)).sum()
            ranked.append((sc, vx, vy))
    ranked.sort(reverse=True)

    vps = []
    used = set()
    for sc, vx, vy in ranked:
        cells = set(
            (vy + dy, vx + dx)
            for dy in range(min(VIEWPORT, h - vy))
            for dx in range(min(VIEWPORT, w - vx))
        )
        if len(cells & used) / max(len(cells), 1) < 0.3:
            vps.append((vx, vy))
            used |= cells
        if len(vps) >= n_viewports:
            break
    return vps


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
# Observation simulation
# ---------------------------------------------------------------------------

def simulate_blitz_observations(gt, ig, n_viewports=3, n_queries=3, rng_seed=42):
    """Simulate blitz observations matching auto_watcher pipeline.

    Uses blitz viewport selection (not entropy-based), samples from GT distribution.
    """
    rng = np.random.RandomState(rng_seed)
    h, w, _ = gt.shape
    vps = select_blitz_viewports(ig, n_viewports)

    observations = []
    for vx, vy in vps:
        for _ in range(n_queries):
            grid = []
            for dy in range(VIEWPORT):
                row = []
                for dx in range(VIEWPORT):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < h and 0 <= x < w:
                        row.append(int(rng.choice(N_CLASSES, p=gt[y, x])))
                    else:
                        row.append(0)
                grid.append(row)
            observations.append({
                "grid": grid,
                "viewport_x": int(vx),
                "viewport_y": int(vy),
            })
    return observations


def simulate_entropy_observations(gt, ig, n_viewports=5, rng_seed=42):
    """Simulate observations using entropy-based selection (eval.py style)."""
    rng = np.random.RandomState(rng_seed)
    h, w, _ = gt.shape
    H = entropy(gt)

    H_pad = np.zeros((h + 1, w + 1))
    H_pad[1:, 1:] = np.cumsum(np.cumsum(H, axis=0), axis=1)

    def window_sum(vy, vx):
        return (H_pad[vy + VIEWPORT, vx + VIEWPORT]
                - H_pad[vy, vx + VIEWPORT]
                - H_pad[vy + VIEWPORT, vx]
                + H_pad[vy, vx])

    observations = []
    used = set()
    max_vy = h - VIEWPORT
    max_vx = w - VIEWPORT

    for _ in range(n_viewports):
        best_score = -1.0
        best_pos = (0, 0)
        for vy in range(max_vy + 1):
            for vx in range(max_vx + 1):
                if any(abs(vy - uvy) < 8 and abs(vx - uvx) < 8 for uvy, uvx in used):
                    continue
                s = window_sum(vy, vx)
                if s > best_score:
                    best_score = s
                    best_pos = (vy, vx)
        vy, vx = best_pos
        used.add((vy, vx))

        grid = []
        for dy in range(VIEWPORT):
            row = []
            for dx in range(VIEWPORT):
                y, x = vy + dy, vx + dx
                if 0 <= y < h and 0 <= x < w:
                    row.append(int(rng.choice(N_CLASSES, p=gt[y, x])))
                else:
                    row.append(0)
            grid.append(row)
        observations.append({
            "grid": grid,
            "viewport_x": int(vx),
            "viewport_y": int(vy),
        })
    return observations


# ---------------------------------------------------------------------------
# Post-processing pipeline (matches auto_watcher exactly)
# ---------------------------------------------------------------------------

def apply_external_tau_overlay(pred, ig, observations, tau=100.0):
    """External Dirichlet tau overlay (same as auto_watcher applies on top of BTP)."""
    h, w, nc = pred.shape
    counts = np.zeros_like(pred)
    obs_count = np.zeros((h, w), dtype=int)

    for obs in observations:
        for dy, row in enumerate(obs["grid"]):
            for dx, cell in enumerate(row):
                y = obs["viewport_y"] + dy
                x = obs["viewport_x"] + dx
                if 0 <= y < h and 0 <= x < w:
                    counts[y, x, cell_code_to_class(cell)] += 1
                    obs_count[y, x] += 1

    for y in range(h):
        for x in range(w):
            if obs_count[y, x] >= 2 and int(ig[y, x]) not in (OCEAN, MOUNTAIN):
                alpha = tau * pred[y, x]
                post = counts[y, x] + alpha
                pred[y, x] = post / post.sum()

    return pred


def apply_sigma_smoothing(pred, ig, sigma=0.3):
    """Gaussian spatial smoothing (same as auto_watcher)."""
    ocean = ig == OCEAN
    mount = ig == MOUNTAIN
    for c in range(N_CLASSES):
        ch = pred[:, :, c].copy()
        ch[ocean | mount] = 0
        ch = gaussian_filter(ch, sigma=sigma)
        pred[:, :, c] = ch
    pred[ocean] = np.array([1, 0, 0, 0, 0, 0], dtype=np.float64)
    pred[mount] = np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def full_pipeline(pred, ig, observations, tau=100.0, sigma=0.3):
    """Apply the full auto_watcher post-processing pipeline."""
    if observations:
        pred = apply_external_tau_overlay(pred, ig, observations, tau=tau)
    pred = apply_sigma_smoothing(pred, ig, sigma=sigma)
    return pred


# ---------------------------------------------------------------------------
# Predictor wrappers
# ---------------------------------------------------------------------------

def make_btp_predictor(train_rounds_data):
    """Build a BTP predictor trained on specific rounds."""
    import bayesian_template_predictor as btp
    all_round_nums = sorted(train_rounds_data.keys())
    per_round_tables, per_round_counts, pooled_tables, pooled_counts = \
        btp.build_round_bucket_tables(train_rounds_data, all_round_nums)

    model = {
        "per_round_tables": per_round_tables,
        "per_round_counts": per_round_counts,
        "pooled_tables": pooled_tables,
        "pooled_counts": pooled_counts,
        "train_rounds": all_round_nums,
    }

    def predict_fn(initial_grid, observations=None):
        ig = np.array(initial_grid, dtype=np.int32)
        pred, weights = btp.predict_bayesian_mixture(
            ig,
            model["per_round_tables"],
            model["per_round_counts"],
            model["pooled_tables"],
            model["pooled_counts"],
            model["train_rounds"],
            observations=observations,
        )
        return pred

    return predict_fn


def make_module_predictor(module_name, train_rounds_data):
    """Build a predictor from a module that supports build_model_from_data."""
    mod = importlib.import_module(module_name)
    if hasattr(mod, "build_model_from_data"):
        model = mod.build_model_from_data(train_rounds_data)
        def predict_fn(initial_grid, observations=None):
            return mod.predict_with_model(initial_grid, model, observations=observations)
        return predict_fn
    else:
        # Fallback: use predict() directly (in-sample only)
        def predict_fn(initial_grid, observations=None):
            if observations:
                return mod.predict(initial_grid, observations=observations)
            return mod.predict(initial_grid)
        return predict_fn


# ---------------------------------------------------------------------------
# LOOCV engine
# ---------------------------------------------------------------------------

def run_loocv(rounds, predictor_name="btp", obs_mode="blitz", n_vp=3, n_queries=3,
              pipeline=True, tau=100.0, sigma=0.3, breakdown=False,
              store_preds=False):
    """Leave-one-round-out cross-validation.

    Args:
        predictor_name: "btp" or module name
        obs_mode: "blitz" (auto_watcher style), "entropy" (eval.py style), or "none"
        n_vp: viewports per seed
        n_queries: queries per viewport (blitz mode only)
        pipeline: apply full post-processing (external tau + sigma)
        breakdown: include per-cell category breakdown
        store_preds: store predictions for calibration/tail analysis
    """
    round_nums = sorted(rounds.keys())
    results = []

    for held_out in round_nums:
        train_data = {rn: seeds for rn, seeds in rounds.items() if rn != held_out}

        if predictor_name == "btp":
            predict_fn = make_btp_predictor(train_data)
        else:
            predict_fn = make_module_predictor(predictor_name, train_data)

        seed_results = []
        for sn in sorted(rounds[held_out].keys()):
            sd = rounds[held_out][sn]
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]

            # Simulate observations
            if obs_mode == "blitz":
                obs = simulate_blitz_observations(gt, ig, n_vp, n_queries, rng_seed=42 + sn)
            elif obs_mode == "entropy":
                obs = simulate_entropy_observations(gt, ig, n_vp, rng_seed=42 + sn)
            else:
                obs = None

            # Predict
            pred = predict_fn(ig, observations=obs)

            # Post-processing pipeline
            if pipeline and obs:
                pred = full_pipeline(pred, ig, obs, tau=tau, sigma=sigma)
            elif pipeline:
                pred = apply_sigma_smoothing(pred, ig, sigma=sigma)

            # Score
            s = score_prediction(gt, pred)
            s["round"] = held_out
            s["seed"] = sn

            if breakdown:
                s["breakdown"] = score_by_category(gt, pred, ig)
            if store_preds:
                s["_pred"] = pred.copy()

            seed_results.append(s)

        round_wkl = np.mean([s["wkl"] for s in seed_results])
        round_score = wkl_to_score(round_wkl)

        results.append({
            "round": held_out,
            "wkl": round_wkl,
            "score": round_score,
            "seeds": seed_results,
        })

    return results


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------

def run_comparison(rounds, config_a, config_b):
    """Paired A/B comparison between two configs."""
    round_nums = sorted(rounds.keys())
    paired_deltas = []

    print(f"\n{'='*70}")
    print(f"  A/B COMPARISON")
    print(f"  A: {config_a['label']}")
    print(f"  B: {config_b['label']}")
    print(f"{'='*70}")

    for held_out in round_nums:
        train_data = {rn: seeds for rn, seeds in rounds.items() if rn != held_out}

        for cfg_label, cfg in [("A", config_a), ("B", config_b)]:
            if cfg["predictor"] == "btp":
                cfg[f"_predict_fn_{held_out}"] = make_btp_predictor(train_data)
            else:
                cfg[f"_predict_fn_{held_out}"] = make_module_predictor(cfg["predictor"], train_data)

        a_wkls = []
        b_wkls = []

        for sn in sorted(rounds[held_out].keys()):
            sd = rounds[held_out][sn]
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]
            rng_seed = 42 + sn

            for cfg in [config_a, config_b]:
                predict_fn = cfg[f"_predict_fn_{held_out}"]
                obs_mode = cfg.get("obs_mode", "blitz")
                if obs_mode == "blitz":
                    obs = simulate_blitz_observations(gt, ig, cfg.get("n_vp", 3),
                                                      cfg.get("n_queries", 3), rng_seed)
                elif obs_mode == "entropy":
                    obs = simulate_entropy_observations(gt, ig, cfg.get("n_vp", 5), rng_seed)
                else:
                    obs = None

                pred = predict_fn(ig, observations=obs)
                if cfg.get("pipeline", True) and obs:
                    pred = full_pipeline(pred, ig, obs,
                                         tau=cfg.get("tau", 100.0),
                                         sigma=cfg.get("sigma", 0.3))
                elif cfg.get("pipeline", True):
                    pred = apply_sigma_smoothing(pred, ig, sigma=cfg.get("sigma", 0.3))

                s = score_prediction(gt, pred)
                if cfg is config_a:
                    a_wkls.append(s["wkl"])
                else:
                    b_wkls.append(s["wkl"])

        a_mean = np.mean(a_wkls)
        b_mean = np.mean(b_wkls)
        delta = b_mean - a_mean
        paired_deltas.append(delta)
        winner = "B" if delta < 0 else "A"
        pct = abs(delta) / a_mean * 100

        print(f"  R{held_out:2d}: A={a_mean:.4f}  B={b_mean:.4f}  "
              f"delta={delta:+.4f} ({pct:.1f}%)  [{winner} wins]")

    deltas = np.array(paired_deltas)
    mean_delta = deltas.mean()
    se = deltas.std() / np.sqrt(len(deltas))
    t_stat = mean_delta / se if se > 0 else 0
    a_wins = (deltas > 0).sum()
    b_wins = (deltas < 0).sum()

    print(f"\n  {'─'*50}")
    print(f"  Mean delta (B-A): {mean_delta:+.6f}")
    print(f"  Std error:        {se:.6f}")
    print(f"  t-statistic:      {t_stat:.2f} (|t|>2 ≈ p<0.05)")
    print(f"  Round wins:       A={a_wins}, B={b_wins}, tie={len(deltas)-a_wins-b_wins}")
    overall = "B is better" if mean_delta < 0 else "A is better"
    sig = "SIGNIFICANT" if abs(t_stat) > 2 else "not significant"
    print(f"  Verdict:          {overall} ({sig})")
    print(f"{'='*70}")

    return {"mean_delta": float(mean_delta), "se": float(se), "t_stat": float(t_stat)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results, label):
    """Print results with both wKL and competition score."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Round':>6s}  {'wKL':>8s}  {'Score':>6s}  {'seeds':>5s}  {'leaderboard':>11s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*5}  {'─'*11}")

    overall_wkls = []
    weighted_sum = 0.0
    weight_total = 0.0

    for r in sorted(results, key=lambda x: x["round"]):
        rn = r["round"]
        n = len(r["seeds"])
        lb_weight = 1.05 ** rn
        weighted_sum += r["score"] * lb_weight
        weight_total += lb_weight
        overall_wkls.append(r["wkl"])
        print(f"  R{rn:<5d} {r['wkl']:8.4f}  {r['score']:6.1f}  {n:5d}  {r['score']*lb_weight:11.1f}")

    mean_wkl = np.mean(overall_wkls)
    mean_score = wkl_to_score(mean_wkl)
    weighted_avg = weighted_sum / weight_total if weight_total > 0 else 0

    print(f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*5}  {'─'*11}")
    print(f"  {'MEAN':>6s} {mean_wkl:8.4f}  {mean_score:6.1f}")
    print(f"\n  Weighted leaderboard score: {weighted_avg:.1f}")

    best = min(results, key=lambda x: x["wkl"])
    worst = max(results, key=lambda x: x["wkl"])
    print(f"  Best:  R{best['round']} (wKL={best['wkl']:.4f}, score={best['score']:.1f})")
    print(f"  Worst: R{worst['round']} (wKL={worst['wkl']:.4f}, score={worst['score']:.1f})")
    print(f"{'='*70}")

    return {"mean_wkl": float(mean_wkl), "mean_score": mean_score, "weighted_lb": weighted_avg}


def report_calibration(results, rounds):
    """Calibration analysis: predicted confidence vs GT probability."""
    print(f"\n  Calibration analysis (overconfidence check):")
    print(f"  {'Round':<8s} {'Conf bin':<12s} {'Pred mean':<10s} {'GT mean':<10s} {'Bias':<10s} {'N cells':<8s}")
    print(f"  {'-'*60}")

    for r in sorted(results, key=lambda x: x["round"]):
        rn = r["round"]
        all_pred_max = []
        all_gt_match = []

        for s in r["seeds"]:
            sn = s["seed"]
            sd = rounds[rn][sn]
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]

            # Reconstruct prediction (we don't store it, recompute from score)
            # Use seed-level wkl as proxy — can't reconstruct without re-predicting
            # Instead, store pred in seed_results during run_loocv
            if "_pred" not in s:
                continue
            pred = s["_pred"]
            H = entropy(gt)
            dynamic = H > 0.01

            for y in range(ig.shape[0]):
                for x in range(ig.shape[1]):
                    if not dynamic[y, x] or int(ig[y, x]) in (OCEAN, MOUNTAIN):
                        continue
                    pred_argmax = pred[y, x].argmax()
                    all_pred_max.append(pred[y, x, pred_argmax])
                    all_gt_match.append(gt[y, x, pred_argmax])

        if not all_pred_max:
            continue
        pred_arr = np.array(all_pred_max)
        gt_arr = np.array(all_gt_match)

        for lo, hi, label in [(0.3, 0.5, "0.3-0.5"), (0.5, 0.7, "0.5-0.7"),
                               (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, "0.9-1.0")]:
            mask = (pred_arr >= lo) & (pred_arr < hi)
            if mask.sum() > 0:
                bias = pred_arr[mask].mean() - gt_arr[mask].mean()
                flag = "  OVERCONF" if bias > 0.1 else ""
                print(f"  R{rn:<6d} {label:<12s} {pred_arr[mask].mean():<10.3f} "
                      f"{gt_arr[mask].mean():<10.3f} {bias:<+10.3f} {mask.sum():<8d}{flag}")
        print()


def report_tail_risk(results, rounds):
    """Analyze tail risk: how concentrated is loss in worst cells?"""
    print(f"\n  Tail risk analysis (loss concentration in worst cells):")
    print(f"  {'Round':<8s} {'top-10 %':<10s} {'top-50 %':<10s} {'max/mean':<10s} {'max wKL':<10s}")
    print(f"  {'-'*50}")

    for r in sorted(results, key=lambda x: x["round"]):
        rn = r["round"]
        # Use first seed for quick analysis
        s = r["seeds"][0]
        if "_pred" not in s:
            continue
        sd = rounds[rn][s["seed"]]
        gt = sd["ground_truth"]
        pred = s["_pred"]

        pred_s = np.maximum(pred, FLOOR)
        pred_s /= pred_s.sum(axis=2, keepdims=True)
        kl = kl_divergence(gt, pred_s)
        H = entropy(gt)
        wkl = H * kl
        dynamic = H > 0.01

        if not dynamic.any():
            continue
        flat = wkl[dynamic].ravel()
        sorted_wkl = np.sort(flat)[::-1]
        total = flat.sum()

        top10_pct = sorted_wkl[:10].sum() / total * 100
        top50_pct = sorted_wkl[:50].sum() / total * 100
        ratio = sorted_wkl[0] / flat.mean()

        print(f"  R{rn:<6d} {top10_pct:<10.1f} {top50_pct:<10.1f} {ratio:<10.1f} {sorted_wkl[0]:<10.4f}")


def report_breakdown(results):
    """Print per-cell category breakdown aggregated across rounds."""
    cat_totals = defaultdict(lambda: {"wkl_sum": 0.0, "n_cells": 0})

    for r in results:
        for s in r["seeds"]:
            if "breakdown" not in s:
                continue
            for cat, info in s["breakdown"].items():
                cat_totals[cat]["wkl_sum"] += info["wkl_sum"]
                cat_totals[cat]["n_cells"] += info["n_cells"]

    total_wkl = sum(v["wkl_sum"] for v in cat_totals.values())
    print(f"\n  Per-cell category breakdown (% of total wKL loss):")
    print(f"  {'Category':<20s}  {'wKL sum':>10s}  {'% total':>8s}  {'cells':>8s}  {'wKL/cell':>10s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")

    for cat, info in sorted(cat_totals.items(), key=lambda x: -x[1]["wkl_sum"]):
        if cat in ("ocean", "mountain"):
            continue
        pct = info["wkl_sum"] / total_wkl * 100 if total_wkl > 0 else 0
        per_cell = info["wkl_sum"] / info["n_cells"] if info["n_cells"] > 0 else 0
        print(f"  {cat:<20s}  {info['wkl_sum']:10.4f}  {pct:7.1f}%  {info['n_cells']:8d}  {per_cell:10.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eval V2: End-to-end pipeline evaluation")
    parser.add_argument("--predictor", "-p", default="btp", help="Predictor: 'btp' or module name")
    parser.add_argument("--obs", default="blitz", choices=["blitz", "entropy", "none"],
                        help="Observation mode")
    parser.add_argument("--n-vp", type=int, default=3, help="Viewports per seed")
    parser.add_argument("--n-queries", type=int, default=3, help="Queries per viewport (blitz)")
    parser.add_argument("--no-pipeline", action="store_true", help="Skip post-processing pipeline")
    parser.add_argument("--tau", type=float, default=100.0, help="External tau overlay")
    parser.add_argument("--sigma", type=float, default=0.3, help="Gaussian smoothing sigma")
    parser.add_argument("--breakdown", action="store_true", help="Per-cell category breakdown")
    parser.add_argument("--calibration", action="store_true", help="Calibration analysis")
    parser.add_argument("--tail-risk", action="store_true", help="Tail risk (loss concentration)")
    parser.add_argument("--diagnostics", action="store_true", help="All diagnostics (breakdown+calibration+tail)")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two predictors (names or 'btp')")
    parser.add_argument("--insample", action="store_true", help="In-sample (no CV)")
    args = parser.parse_args()

    sys.path.insert(0, str(BASE_DIR))
    rounds = load_ground_truth()
    n_rounds = len(rounds)
    n_seeds = sum(len(s) for s in rounds.values())
    print(f"Data: {n_rounds} rounds, {n_seeds} seeds")

    t0 = time.time()

    if args.compare:
        config_a = {"predictor": args.compare[0], "label": args.compare[0],
                     "obs_mode": args.obs, "n_vp": args.n_vp, "n_queries": args.n_queries,
                     "pipeline": not args.no_pipeline, "tau": args.tau, "sigma": args.sigma}
        config_b = {"predictor": args.compare[1], "label": args.compare[1],
                     "obs_mode": args.obs, "n_vp": args.n_vp, "n_queries": args.n_queries,
                     "pipeline": not args.no_pipeline, "tau": args.tau, "sigma": args.sigma}
        run_comparison(rounds, config_a, config_b)
    else:
        pipeline = not args.no_pipeline
        obs_label = f"{args.obs} ({args.n_vp}vp x {args.n_queries}q)" if args.obs != "none" else "none"
        pipe_label = f"tau={args.tau}, σ={args.sigma}" if pipeline else "raw"
        label = f"LOOCV | {args.predictor} | obs={obs_label} | {pipe_label}"

        do_breakdown = args.breakdown or args.diagnostics
        do_calibration = args.calibration or args.diagnostics
        do_tail = args.tail_risk or args.diagnostics
        store = do_calibration or do_tail

        results = run_loocv(
            rounds, args.predictor, args.obs, args.n_vp, args.n_queries,
            pipeline=pipeline, tau=args.tau, sigma=args.sigma,
            breakdown=do_breakdown, store_preds=store,
        )
        summary = report(results, label)

        if do_breakdown:
            report_breakdown(results)
        if do_calibration:
            report_calibration(results, rounds)
        if do_tail:
            report_tail_risk(results, rounds)

        # Save
        out = {
            "predictor": args.predictor,
            "obs_mode": args.obs,
            "n_vp": args.n_vp,
            "n_queries": args.n_queries,
            "pipeline": pipeline,
            "tau": args.tau,
            "sigma": args.sigma,
            **summary,
        }
        out_path = BASE_DIR / "benchmark_results" / f"eval_v2_{args.predictor}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
