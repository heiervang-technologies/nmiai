#!/usr/bin/env python3
"""Definitive comprehensive evaluation of Astar Island predictor.

Simulates full live pipeline across all 75 ground truth files:
1. Simulate blitz observations (sample from GT on hottest viewport)
2. Run regime_predictor.predict() with those observations
3. Apply tau overlay on cells with 2+ samples
4. Compute weighted KL against ground truth

Reports: overall, per-round, per-regime, per-distance-band, per-class residuals,
worst cells, and tau/viewport variations.
"""

import json
import sys
import time
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt, distance_transform_cdt

sys.path.insert(0, str(Path(__file__).parent))

import regime_predictor as rp
from benchmark import load_ground_truth, kl_divergence, entropy

np.random.seed(42)
RNG = np.random.RandomState(42)

GT_DIR = Path(__file__).parent / "ground_truth"
VIEWPORT_SIZE = 15
N_CLASSES = 6
FLOOR = 1e-6

CLASS_NAMES = ["Ocean/Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def simulate_observations_from_gt(gt, ig, n_viewports, rng=None, strategy="hotspot",
                                   min_dist=8):
    """Simulate observations by sampling from GT distribution on hottest viewports."""
    if rng is None:
        rng = RNG
    gt = np.array(gt)
    ig = np.array(ig)
    h, w, _ = gt.shape
    observations = []

    if strategy == "hotspot":
        civ = (ig == 1) | (ig == 2)
        if civ.any():
            dist = distance_transform_cdt(~civ, metric="taxicab")
        else:
            dist = np.full((h, w), 20)

        # Score each viewport position by entropy of dynamic cells
        max_vy = h - VIEWPORT_SIZE + 1
        max_vx = w - VIEWPORT_SIZE + 1
        scores = np.zeros((max_vy, max_vx))
        for vy in range(max_vy):
            for vx in range(max_vx):
                patch = gt[vy:vy+VIEWPORT_SIZE, vx:vx+VIEWPORT_SIZE]
                ig_patch = ig[vy:vy+VIEWPORT_SIZE, vx:vx+VIEWPORT_SIZE]
                dynamic = (ig_patch != 10) & (ig_patch != 5)
                if dynamic.any():
                    ent = -np.sum(np.maximum(patch, 1e-10) * np.log2(np.maximum(patch, 1e-10)), axis=2)
                    scores[vy, vx] = ent[dynamic].sum()

        used_positions = []
        for _ in range(n_viewports):
            flat = scores.ravel()
            sorted_idx = np.argsort(-flat)
            best_pos = (0, 0)
            for idx in sorted_idx:
                vy = idx // scores.shape[1]
                vx = idx % scores.shape[1]
                too_close = False
                for uvy, uvx in used_positions:
                    if abs(vy - uvy) < min_dist and abs(vx - uvx) < min_dist:
                        too_close = True
                        break
                if not too_close:
                    best_pos = (vy, vx)
                    break
            vy, vx = best_pos
            used_positions.append((vy, vx))

            grid_sample = []
            for dy in range(VIEWPORT_SIZE):
                row = []
                for dx in range(VIEWPORT_SIZE):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < h and 0 <= x < w:
                        probs = gt[y, x]
                        cls = rng.choice(6, p=probs)
                        code = [0, 1, 2, 3, 4, 5][cls]
                        row.append(code)
                    else:
                        row.append(0)
                grid_sample.append(row)

            observations.append({
                "grid": grid_sample,
                "viewport_x": int(vx),
                "viewport_y": int(vy),
            })

    return observations


def apply_tau_overlay(pred, ig, observations, tau):
    """Apply Bayesian tau overlay: for cells with 2+ observations, blend toward observed frequency."""
    h, w, c = pred.shape
    obs_counts = np.zeros((h, w, N_CLASSES))
    obs_total = np.zeros((h, w))

    for obs in observations:
        grid = obs.get("grid", [])
        vx = int(obs.get("viewport_x", 0))
        vy = int(obs.get("viewport_y", 0))
        for dy, row in enumerate(grid):
            for dx, cell_val in enumerate(row):
                y = vy + dy
                x = vx + dx
                if 0 <= y < h and 0 <= x < w:
                    cls = rp.cell_code_to_class(int(cell_val))
                    obs_counts[y, x, cls] += 1
                    obs_total[y, x] += 1

    result = pred.copy()
    mask = obs_total >= 2
    if mask.any():
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            n = obs_total[y, x]
            freq = obs_counts[y, x] / n
            alpha = n / (n + tau)
            blended = (1 - alpha) * pred[y, x] + alpha * freq
            blended = np.maximum(blended, FLOOR)
            blended /= blended.sum()
            result[y, x] = blended

    return result


def compute_wkl(gt, pred):
    """Compute per-cell weighted KL and return (mean_wkl, mean_kl, per_cell_wkl, per_cell_kl, H, dynamic_mask)."""
    pred_safe = np.maximum(pred, 0.01)
    pred_safe /= pred_safe.sum(axis=2, keepdims=True)

    kl = kl_divergence(gt, pred_safe)
    H = entropy(gt)
    wkl = H * kl
    dynamic = H > 0.01

    mean_kl = kl[dynamic].mean() if dynamic.any() else 0
    mean_wkl = wkl[dynamic].mean() if dynamic.any() else 0
    return float(mean_wkl), float(mean_kl), wkl, kl, H, dynamic


def run_pipeline(gt, ig, n_viewports, tau, n_queries_per_viewport=1,
                 viewport_strategy="hotspot", min_dist=8, rng=None,
                 mod5_prior=None, round_num=None):
    """Run full pipeline: observations -> predict -> tau overlay -> score."""
    if rng is None:
        rng = RNG

    total_queries = n_viewports * n_queries_per_viewport

    if n_queries_per_viewport > 1:
        # Multiple queries per viewport: same viewport positions, multiple samples
        base_obs = simulate_observations_from_gt(gt, ig, n_viewports, rng=rng,
                                                  strategy=viewport_strategy, min_dist=min_dist)
        observations = list(base_obs)
        for obs in base_obs:
            for _ in range(n_queries_per_viewport - 1):
                grid_sample = []
                vy = obs["viewport_y"]
                vx = obs["viewport_x"]
                h, w, _ = np.array(gt).shape
                for dy in range(VIEWPORT_SIZE):
                    row = []
                    for dx in range(VIEWPORT_SIZE):
                        y, x = vy + dy, vx + dx
                        if 0 <= y < h and 0 <= x < w:
                            probs = np.array(gt)[y, x]
                            cls = rng.choice(6, p=probs)
                            row.append([0, 1, 2, 3, 4, 5][cls])
                        else:
                            row.append(0)
                    grid_sample.append(row)
                observations.append({
                    "grid": grid_sample,
                    "viewport_x": vx,
                    "viewport_y": vy,
                })
    else:
        observations = simulate_observations_from_gt(gt, ig, n_viewports, rng=rng,
                                                      strategy=viewport_strategy, min_dist=min_dist)

    # If mod5 prior, adjust regime weights
    if mod5_prior and round_num is not None:
        pos = (round_num - 1) % 5
        # position 0 -> prosperous weight boost, position 2 -> harsh weight boost
        pass  # handled below after predict

    pred = rp.predict(ig, observations=observations)

    if tau > 0 and observations:
        pred = apply_tau_overlay(pred, np.array(ig), observations, tau)

    return pred, observations


def main():
    print("=" * 80)
    print("DEFINITIVE COMPREHENSIVE EVALUATION - Astar Island Predictor")
    print("=" * 80)

    # Force rebuild model
    rp._MODEL = None
    model = rp.get_model()
    round_regimes = model["round_regimes"]
    print(f"\nModel round regimes: {round_regimes}")

    rounds = load_ground_truth()
    all_round_nums = sorted(rounds.keys())
    print(f"Loaded {len(all_round_nums)} rounds, {sum(len(s) for s in rounds.values())} seeds total")
    print(f"Rounds: {all_round_nums}")

    # =========================================================================
    # BASELINE: regime_predictor with 10 blitz obs, tau=30
    # =========================================================================
    print("\n" + "=" * 80)
    print("BASELINE: 10 blitz observations (hotspot), tau=30")
    print("=" * 80)

    all_results = []
    per_round_results = defaultdict(list)
    per_regime_results = defaultdict(list)
    per_cell_data = []  # for worst-cell analysis

    rng = np.random.RandomState(42)

    for rn in all_round_nums:
        regime = round_regimes.get(rn, "unknown")
        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            gt = np.array(sd["ground_truth"])
            ig = sd["initial_grid"]
            ig_np = np.array(ig, dtype=np.int32)

            pred, obs = run_pipeline(gt, ig, n_viewports=2, tau=30,
                                      n_queries_per_viewport=5, min_dist=8, rng=rng)

            mean_wkl, mean_kl, wkl_map, kl_map, H_map, dynamic = compute_wkl(gt, pred)

            result = {
                "round": rn, "seed": sn, "regime": regime,
                "mean_wkl": mean_wkl, "mean_kl": mean_kl,
            }
            all_results.append(result)
            per_round_results[rn].append(mean_wkl)
            per_regime_results[regime].append(mean_wkl)

            # Collect per-cell data for worst-cell and distance-band analysis
            dist_civ, _, _, _ = rp.compute_features(ig_np)

            h, w = ig_np.shape
            for y in range(h):
                for x in range(w):
                    if not dynamic[y, x]:
                        continue
                    code = int(ig_np[y, x])
                    if code in (10, 5):  # ocean, mountain
                        continue

                    cell_info = {
                        "round": rn, "seed": sn, "y": y, "x": x,
                        "wkl": float(wkl_map[y, x]),
                        "kl": float(kl_map[y, x]),
                        "entropy": float(H_map[y, x]),
                        "dist_civ": float(dist_civ[y, x]),
                        "code": code,
                        "regime": regime,
                        "gt": gt[y, x].tolist(),
                        "pred": pred[y, x].tolist(),
                    }
                    per_cell_data.append(cell_info)

    # Overall stats
    overall_wkl = np.mean([r["mean_wkl"] for r in all_results])
    overall_kl = np.mean([r["mean_kl"] for r in all_results])
    print(f"\n  OVERALL Mean wKL: {overall_wkl:.6f}")
    print(f"  OVERALL Mean KL:  {overall_kl:.6f}")
    print(f"  Total seeds evaluated: {len(all_results)}")

    # Per-round
    print("\n  Per-round mean wKL:")
    round_means = {}
    for rn in all_round_nums:
        m = np.mean(per_round_results[rn])
        round_means[rn] = m
        regime = round_regimes.get(rn, "?")
        print(f"    R{rn:2d} ({regime:10s}): {m:.6f}")

    # Per-regime
    print("\n  Per-regime mean wKL:")
    regime_means = {}
    for regime in sorted(per_regime_results.keys()):
        m = np.mean(per_regime_results[regime])
        regime_means[regime] = m
        n = len(per_regime_results[regime])
        print(f"    {regime:12s}: {m:.6f} (n={n})")

    # Per-distance band
    print("\n  Per-distance band wKL:")
    bands = [(0, 2, "d0-2"), (2, 5, "d2-5"), (5, 8, "d5-8"), (8, 999, "d8+")]
    band_results = defaultdict(list)
    for cell in per_cell_data:
        d = cell["dist_civ"]
        for lo, hi, label in bands:
            if lo <= d < hi:
                band_results[label].append(cell["wkl"])
                break
    for lo, hi, label in bands:
        vals = band_results[label]
        if vals:
            print(f"    {label}: mean={np.mean(vals):.6f}  median={np.median(vals):.6f}  n={len(vals)}")

    # Per-class residuals
    print("\n  Per-class residuals (pred - gt, averaged over dynamic cells):")
    class_residuals = defaultdict(list)
    for cell in per_cell_data:
        for c in range(N_CLASSES):
            class_residuals[c].append(cell["pred"][c] - cell["gt"][c])
    for c in range(N_CLASSES):
        vals = class_residuals[c]
        m = np.mean(vals)
        std = np.std(vals)
        direction = "OVER" if m > 0 else "UNDER"
        print(f"    Class {c} ({CLASS_NAMES[c]:12s}): mean_residual={m:+.6f} std={std:.6f} [{direction}-predict]")

    # TOP 50 worst cells
    print("\n  TOP 50 worst cells by wKL:")
    per_cell_data.sort(key=lambda c: -c["wkl"])
    worst_50 = per_cell_data[:50]

    # Analyze patterns
    worst_codes = defaultdict(int)
    worst_regimes = defaultdict(int)
    worst_bands = defaultdict(int)
    worst_rounds = defaultdict(int)
    for cell in worst_50:
        worst_codes[cell["code"]] += 1
        worst_regimes[cell["regime"]] += 1
        d = cell["dist_civ"]
        for lo, hi, label in bands:
            if lo <= d < hi:
                worst_bands[label] += 1
                break
        worst_rounds[cell["round"]] += 1

    for i, cell in enumerate(worst_50[:20]):
        gt_str = " ".join(f"{v:.2f}" for v in cell["gt"])
        pred_str = " ".join(f"{v:.2f}" for v in cell["pred"])
        print(f"    #{i+1}: R{cell['round']}S{cell['seed']} ({cell['y']},{cell['x']}) "
              f"wKL={cell['wkl']:.4f} d={cell['dist_civ']:.1f} code={cell['code']} "
              f"regime={cell['regime']}")
        print(f"         GT:   [{gt_str}]")
        print(f"         Pred: [{pred_str}]")

    print(f"\n  Worst 50 cells - code distribution: {dict(worst_codes)}")
    print(f"  Worst 50 cells - regime distribution: {dict(worst_regimes)}")
    print(f"  Worst 50 cells - distance band distribution: {dict(worst_bands)}")
    print(f"  Worst 50 cells - round distribution: {dict(worst_rounds)}")

    # =========================================================================
    # VARIATION TESTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("VARIATION TESTS")
    print("=" * 80)

    variations = {}

    # --- Tau variations ---
    print("\n  --- Tau Variations (2 viewports x 5 queries) ---")
    for tau in [0, 20, 30, 40, 50]:
        rng_v = np.random.RandomState(42)
        wkls = []
        for rn in all_round_nums:
            for sn in sorted(rounds[rn].keys()):
                sd = rounds[rn][sn]
                gt = np.array(sd["ground_truth"])
                ig = sd["initial_grid"]
                pred, _ = run_pipeline(gt, ig, n_viewports=2, tau=tau,
                                        n_queries_per_viewport=5, min_dist=8, rng=rng_v)
                mean_wkl, _, _, _, _, _ = compute_wkl(gt, pred)
                wkls.append(mean_wkl)
        m = np.mean(wkls)
        variations[f"tau={tau}"] = m
        print(f"    tau={tau:3d}: mean_wKL = {m:.6f}")

    # --- Viewport strategy variations ---
    print("\n  --- Viewport Strategy Variations (tau=30) ---")

    # 2 viewports x 5 queries (baseline)
    rng_v = np.random.RandomState(42)
    wkls_2x5 = []
    for rn in all_round_nums:
        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            gt = np.array(sd["ground_truth"])
            ig = sd["initial_grid"]
            pred, _ = run_pipeline(gt, ig, n_viewports=2, tau=30,
                                    n_queries_per_viewport=5, min_dist=8, rng=rng_v)
            mean_wkl, _, _, _, _, _ = compute_wkl(gt, pred)
            wkls_2x5.append(mean_wkl)
    m_2x5 = np.mean(wkls_2x5)
    variations["2vp x 5q"] = m_2x5
    print(f"    2 viewports x 5 queries: mean_wKL = {m_2x5:.6f}")

    # 1 viewport x 10 queries
    rng_v = np.random.RandomState(42)
    wkls_1x10 = []
    for rn in all_round_nums:
        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            gt = np.array(sd["ground_truth"])
            ig = sd["initial_grid"]
            pred, _ = run_pipeline(gt, ig, n_viewports=1, tau=30,
                                    n_queries_per_viewport=10, min_dist=0, rng=rng_v)
            mean_wkl, _, _, _, _, _ = compute_wkl(gt, pred)
            wkls_1x10.append(mean_wkl)
    m_1x10 = np.mean(wkls_1x10)
    variations["1vp x 10q"] = m_1x10
    print(f"    1 viewport x 10 queries: mean_wKL = {m_1x10:.6f}")

    # --- Mod-5 cycle prior ---
    print("\n  --- Mod-5 Cycle Prior Test ---")
    # Test: adjust regime weights based on round position in mod-5 cycle
    # position 0 -> boost prosperous, position 2 -> boost harsh
    MOD5_WEIGHTS = {
        0: {"harsh": 0.10, "moderate": 0.30, "prosperous": 0.60},
        1: {"harsh": 0.20, "moderate": 0.60, "prosperous": 0.20},
        2: {"harsh": 0.60, "moderate": 0.30, "prosperous": 0.10},
        3: {"harsh": 0.20, "moderate": 0.60, "prosperous": 0.20},
        4: {"harsh": 0.20, "moderate": 0.60, "prosperous": 0.20},
    }

    # Save original defaults
    orig_weights = rp.DEFAULT_REGIME_WEIGHTS.copy()

    rng_v = np.random.RandomState(42)
    wkls_mod5 = []
    for rn in all_round_nums:
        pos = (rn - 1) % 5
        rp.DEFAULT_REGIME_WEIGHTS.update(MOD5_WEIGHTS[pos])
        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            gt = np.array(sd["ground_truth"])
            ig = sd["initial_grid"]
            pred, _ = run_pipeline(gt, ig, n_viewports=2, tau=30,
                                    n_queries_per_viewport=5, min_dist=8, rng=rng_v)
            mean_wkl, _, _, _, _, _ = compute_wkl(gt, pred)
            wkls_mod5.append(mean_wkl)

    # Restore original
    rp.DEFAULT_REGIME_WEIGHTS.update(orig_weights)

    m_mod5 = np.mean(wkls_mod5)
    variations["mod5_prior"] = m_mod5
    print(f"    Mod-5 cycle prior: mean_wKL = {m_mod5:.6f}")

    # --- No observations baseline ---
    print("\n  --- No Observations Baseline ---")
    rng_v = np.random.RandomState(42)
    wkls_noobs = []
    for rn in all_round_nums:
        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            gt = np.array(sd["ground_truth"])
            ig = sd["initial_grid"]
            pred = rp.predict(ig, observations=None)
            mean_wkl, _, _, _, _, _ = compute_wkl(gt, pred)
            wkls_noobs.append(mean_wkl)
    m_noobs = np.mean(wkls_noobs)
    variations["no_obs"] = m_noobs
    print(f"    No observations: mean_wKL = {m_noobs:.6f}")

    # =========================================================================
    # WRITE DEFINITIVE REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("WRITING DEFINITIVE REPORT")
    print("=" * 80)

    report_lines = []
    report_lines.append("# Definitive Evaluation: Astar Island Predictor")
    report_lines.append("")
    report_lines.append(f"**Date**: 2026-03-21")
    report_lines.append(f"**Rounds evaluated**: {len(all_round_nums)} ({min(all_round_nums)}-{max(all_round_nums)})")
    report_lines.append(f"**Seeds per round**: 5 (seeds 0-4)")
    report_lines.append(f"**Total evaluations**: {len(all_results)}")
    report_lines.append(f"**Random seed**: 42 (reproducible)")
    report_lines.append("")

    report_lines.append("## Leaderboard: Configuration Variants")
    report_lines.append("")
    report_lines.append("| Rank | Configuration | Mean wKL | vs Baseline |")
    report_lines.append("|------|--------------|----------|-------------|")
    sorted_vars = sorted(variations.items(), key=lambda x: x[1])
    baseline_val = variations.get("tau=30", overall_wkl)
    for i, (name, val) in enumerate(sorted_vars):
        delta = val - baseline_val
        marker = "" if name == "tau=30" else f"{delta:+.6f}"
        report_lines.append(f"| {i+1} | {name} | {val:.6f} | {marker} |")

    report_lines.append("")
    report_lines.append("## Overall Baseline (2vp x 5q, tau=30)")
    report_lines.append("")
    report_lines.append(f"- **Mean wKL**: {overall_wkl:.6f}")
    report_lines.append(f"- **Mean KL**: {overall_kl:.6f}")
    report_lines.append("")

    report_lines.append("## Per-Round Breakdown")
    report_lines.append("")
    report_lines.append("| Round | Regime | Mean wKL |")
    report_lines.append("|-------|--------|----------|")
    for rn in all_round_nums:
        regime = round_regimes.get(rn, "?")
        m = round_means[rn]
        report_lines.append(f"| R{rn} | {regime} | {m:.6f} |")

    report_lines.append("")
    report_lines.append("## Per-Regime Breakdown")
    report_lines.append("")
    report_lines.append("| Regime | Mean wKL | N seeds |")
    report_lines.append("|--------|----------|---------|")
    for regime in sorted(regime_means.keys()):
        m = regime_means[regime]
        n = len(per_regime_results[regime])
        report_lines.append(f"| {regime} | {m:.6f} | {n} |")

    report_lines.append("")
    report_lines.append("## Per-Distance Band")
    report_lines.append("")
    report_lines.append("| Band | Mean wKL | Median wKL | N cells |")
    report_lines.append("|------|----------|------------|---------|")
    for lo, hi, label in bands:
        vals = band_results[label]
        if vals:
            report_lines.append(f"| {label} | {np.mean(vals):.6f} | {np.median(vals):.6f} | {len(vals)} |")

    report_lines.append("")
    report_lines.append("## Per-Class Residuals (pred - gt)")
    report_lines.append("")
    report_lines.append("| Class | Name | Mean Residual | Std | Direction |")
    report_lines.append("|-------|------|---------------|-----|-----------|")
    for c in range(N_CLASSES):
        vals = class_residuals[c]
        m = np.mean(vals)
        std = np.std(vals)
        direction = "OVER" if m > 0 else "UNDER"
        report_lines.append(f"| {c} | {CLASS_NAMES[c]} | {m:+.6f} | {std:.6f} | {direction} |")

    report_lines.append("")
    report_lines.append("## TOP 50 Worst Cells Analysis")
    report_lines.append("")
    report_lines.append("### Pattern Summary")
    report_lines.append("")
    report_lines.append(f"- **Cell code distribution**: {dict(worst_codes)}")
    code_map = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest"}
    worst_code_str = ", ".join(f"{code_map.get(k, k)}: {v}" for k, v in sorted(worst_codes.items(), key=lambda x: -x[1]))
    report_lines.append(f"  - Decoded: {worst_code_str}")
    report_lines.append(f"- **Regime distribution**: {dict(worst_regimes)}")
    report_lines.append(f"- **Distance band distribution**: {dict(worst_bands)}")
    report_lines.append(f"- **Round distribution**: {dict(worst_rounds)}")
    report_lines.append("")

    report_lines.append("### Worst 20 Individual Cells")
    report_lines.append("")
    report_lines.append("| # | Round | Seed | Position | wKL | Dist | Code | Regime | GT | Pred |")
    report_lines.append("|---|-------|------|----------|-----|------|------|--------|----|------|")
    for i, cell in enumerate(worst_50[:20]):
        gt_str = " ".join(f"{v:.2f}" for v in cell["gt"])
        pred_str = " ".join(f"{v:.2f}" for v in cell["pred"])
        report_lines.append(f"| {i+1} | R{cell['round']} | S{cell['seed']} | ({cell['y']},{cell['x']}) | "
                          f"{cell['wkl']:.4f} | {cell['dist_civ']:.1f} | {cell['code']} | "
                          f"{cell['regime']} | [{gt_str}] | [{pred_str}] |")

    report_lines.append("")
    report_lines.append("## Tau Sensitivity")
    report_lines.append("")
    report_lines.append("| Tau | Mean wKL |")
    report_lines.append("|-----|----------|")
    for tau in [0, 20, 30, 40, 50]:
        key = f"tau={tau}"
        if key in variations:
            report_lines.append(f"| {tau} | {variations[key]:.6f} |")

    report_lines.append("")
    report_lines.append("## Viewport Strategy")
    report_lines.append("")
    report_lines.append("| Strategy | Mean wKL |")
    report_lines.append("|----------|----------|")
    report_lines.append(f"| 2 viewports x 5 queries | {m_2x5:.6f} |")
    report_lines.append(f"| 1 viewport x 10 queries | {m_1x10:.6f} |")
    report_lines.append(f"| No observations | {m_noobs:.6f} |")

    report_lines.append("")
    report_lines.append("## Key Findings")
    report_lines.append("")

    best_config = sorted_vars[0]
    worst_config = sorted_vars[-1]
    obs_improvement = ((m_noobs - baseline_val) / m_noobs * 100) if m_noobs > 0 else 0

    report_lines.append(f"1. **Best configuration**: {best_config[0]} (wKL={best_config[1]:.6f})")
    report_lines.append(f"2. **Worst configuration**: {worst_config[0]} (wKL={worst_config[1]:.6f})")
    report_lines.append(f"3. **Observation improvement**: {obs_improvement:.1f}% reduction from no-obs baseline")

    # Identify hardest regime
    hardest_regime = max(regime_means.items(), key=lambda x: x[1])
    easiest_regime = min(regime_means.items(), key=lambda x: x[1])
    report_lines.append(f"4. **Hardest regime**: {hardest_regime[0]} (wKL={hardest_regime[1]:.6f})")
    report_lines.append(f"5. **Easiest regime**: {easiest_regime[0]} (wKL={easiest_regime[1]:.6f})")

    # Distance insight
    worst_band = max(band_results.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
    report_lines.append(f"6. **Worst distance band**: {worst_band[0]} (mean wKL={np.mean(worst_band[1]):.6f})")

    # Class insight
    worst_class_residual = max(range(N_CLASSES), key=lambda c: abs(np.mean(class_residuals[c])))
    report_lines.append(f"7. **Largest class bias**: {CLASS_NAMES[worst_class_residual]} "
                       f"(residual={np.mean(class_residuals[worst_class_residual]):+.6f})")

    report_lines.append("")

    report_path = Path(__file__).parent / "definitive_eval.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport written to {report_path}")

    # Also print per-seed detail for completeness
    print("\n  Per-seed detail:")
    for r in all_results:
        print(f"    R{r['round']:2d}S{r['seed']}: wKL={r['mean_wkl']:.6f} ({r['regime']})")


if __name__ == "__main__":
    main()
