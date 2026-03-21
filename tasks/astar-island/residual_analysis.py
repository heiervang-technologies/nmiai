#!/usr/bin/env python3
"""Comprehensive residual analysis of regime_predictor vs ground truth.

Finds systematic error patterns to close the 88.6 -> 94.5 gap.
"""

import json
import sys
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve, label

sys.path.insert(0, str(Path(__file__).parent))
import regime_predictor as rp
from benchmark import load_ground_truth, kl_divergence, entropy

# Class names
CLASS_NAMES = ["Ocean", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

def run_analysis():
    print("Loading ground truth...")
    rounds = load_ground_truth()
    total_seeds = sum(len(s) for s in rounds.values())
    print(f"Loaded {len(rounds)} rounds, {total_seeds} seeds")

    # Build model (in-sample, since we want to see what patterns remain)
    print("Building model...")
    model = rp.build_model()

    # Collect per-cell residuals with rich features
    all_residuals = []  # list of dicts per dynamic cell

    KERNEL = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)

    for rn in sorted(rounds.keys()):
        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            ig = np.array(sd["initial_grid"], dtype=np.int32)
            gt = np.array(sd["ground_truth"], dtype=np.float64)

            pred = rp.predict(ig)
            # Match benchmark clamping
            pred = np.maximum(pred, 0.01)
            pred /= pred.sum(axis=2, keepdims=True)

            h, w = ig.shape

            # Compute features
            civ = (ig == rp.SETTLEMENT) | (ig == rp.PORT)
            ocean = ig == rp.OCEAN
            mountain = ig == rp.MOUNTAIN
            forest = ig == rp.FOREST

            dist_civ = distance_transform_edt(~civ) if civ.any() else np.full((h,w), 99.0)
            dist_ocean = distance_transform_edt(~ocean) if ocean.any() else np.full((h,w), 99.0)
            dist_mountain = distance_transform_edt(~mountain) if mountain.any() else np.full((h,w), 99.0)

            n_ocean = convolve(ocean.astype(np.int32), KERNEL, mode="constant")
            n_civ = convolve(civ.astype(np.int32), KERNEL, mode="constant")
            n_forest = convolve(forest.astype(np.int32), KERNEL, mode="constant")
            n_mountain = convolve(mountain.astype(np.int32), KERNEL, mode="constant")
            coast = (n_ocean > 0) & ~ocean

            # Connected component sizes for forest
            forest_labels, n_components = label(forest)
            forest_cc_size = np.zeros((h, w), dtype=np.int32)
            if n_components > 0:
                for lbl in range(1, n_components + 1):
                    mask = forest_labels == lbl
                    forest_cc_size[mask] = mask.sum()

            # Local heterogeneity: number of distinct terrain types in 3x3
            heterogeneity = np.zeros((h, w), dtype=np.int32)
            for y in range(h):
                for x in range(w):
                    neighborhood = set()
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                neighborhood.add(int(ig[ny, nx]))
                    heterogeneity[y, x] = len(neighborhood)

            # Per-cell KL
            kl_per_cell = kl_divergence(gt, pred)
            H_per_cell = entropy(gt)
            wkl_per_cell = H_per_cell * kl_per_cell

            for y in range(h):
                for x in range(w):
                    code = int(ig[y, x])
                    if code in (rp.OCEAN, rp.MOUNTAIN):
                        continue
                    if H_per_cell[y, x] < 0.01:
                        continue  # static cell

                    # Border cell?
                    is_border = (y == 0 or y == h-1 or x == 0 or x == w-1)
                    is_near_border = (y <= 1 or y >= h-2 or x <= 1 or x >= w-2)

                    residual = gt[y, x] - pred[y, x]  # per-class residual

                    all_residuals.append({
                        "round": rn, "seed": sn, "y": y, "x": x,
                        "code": code,
                        "type": {rp.SETTLEMENT: "S", rp.PORT: "P", rp.FOREST: "F"}.get(code, "L"),
                        "dist_civ": float(dist_civ[y, x]),
                        "dist_ocean": float(dist_ocean[y, x]),
                        "dist_mountain": float(dist_mountain[y, x]),
                        "n_ocean": int(n_ocean[y, x]),
                        "n_civ": int(n_civ[y, x]),
                        "n_forest": int(n_forest[y, x]),
                        "n_mountain": int(n_mountain[y, x]),
                        "coast": bool(coast[y, x]),
                        "forest_cc_size": int(forest_cc_size[y, x]),
                        "heterogeneity": int(heterogeneity[y, x]),
                        "is_border": is_border,
                        "is_near_border": is_near_border,
                        "kl": float(kl_per_cell[y, x]),
                        "wkl": float(wkl_per_cell[y, x]),
                        "entropy": float(H_per_cell[y, x]),
                        "gt": gt[y, x].copy(),
                        "pred": pred[y, x].copy(),
                        "residual": residual.copy(),
                        "regime": model["round_regimes"].get(rn, "unknown"),
                    })

    print(f"Collected {len(all_residuals)} dynamic cell records")

    # Convert to structured arrays for fast analysis
    N = len(all_residuals)
    gt_arr = np.array([r["gt"] for r in all_residuals])  # N x 6
    pred_arr = np.array([r["pred"] for r in all_residuals])  # N x 6
    res_arr = gt_arr - pred_arr  # N x 6
    kl_arr = np.array([r["kl"] for r in all_residuals])
    wkl_arr = np.array([r["wkl"] for r in all_residuals])

    features = {k: np.array([r[k] for r in all_residuals])
                for k in ["dist_civ", "dist_ocean", "dist_mountain",
                          "n_ocean", "n_civ", "n_forest", "n_mountain",
                          "forest_cc_size", "heterogeneity"]}
    types = np.array([r["type"] for r in all_residuals])
    coast = np.array([r["coast"] for r in all_residuals])
    border = np.array([r["is_border"] for r in all_residuals])
    near_border = np.array([r["is_near_border"] for r in all_residuals])
    regimes = np.array([r["regime"] for r in all_residuals])
    rounds_arr = np.array([r["round"] for r in all_residuals])

    total_wkl = wkl_arr.sum()
    mean_wkl = wkl_arr.mean()
    print(f"Total wKL: {total_wkl:.2f}, Mean wKL per cell: {mean_wkl:.6f}")

    report = []
    report.append("# Residual Audit: Regime Predictor Systematic Error Patterns\n")
    report.append(f"Analyzed {N} dynamic cells across {total_seeds} seeds, {len(rounds)} rounds.\n")
    report.append(f"Overall mean wKL per dynamic cell: {mean_wkl:.6f}\n")
    report.append(f"Overall mean KL per dynamic cell: {kl_arr.mean():.6f}\n")

    # =========================================================================
    # 1. PER-CLASS RESIDUAL BIAS BY CELL TYPE
    # =========================================================================
    report.append("\n## 1. Per-Class Residual Bias by Cell Type\n")
    report.append("Positive residual = we UNDER-predict (GT > pred). Negative = we OVER-predict.\n")

    for ctype in ["S", "P", "F", "L"]:
        mask = types == ctype
        if mask.sum() == 0:
            continue
        n_cells = mask.sum()
        mean_res = res_arr[mask].mean(axis=0)
        mean_abs_res = np.abs(res_arr[mask]).mean(axis=0)
        type_wkl = wkl_arr[mask].mean()

        report.append(f"\n### Cell type: {ctype} ({n_cells} cells, mean wKL={type_wkl:.6f})\n")
        report.append("| Class | Mean Residual | Mean |Residual| | Direction |\n")
        report.append("|-------|--------------|----------------|----------|\n")
        for c in range(6):
            direction = "UNDER" if mean_res[c] > 0.001 else ("OVER" if mean_res[c] < -0.001 else "~OK")
            report.append(f"| {CLASS_NAMES[c]} | {mean_res[c]:+.5f} | {mean_abs_res[c]:.5f} | {direction} |\n")

    # =========================================================================
    # 2. FEATURES WE DON'T USE: Distance to ocean, distance to mountain,
    #    forest CC size, heterogeneity, n_forest, n_mountain
    # =========================================================================
    report.append("\n## 2. Unused Feature Correlations\n")
    report.append("Features NOT in bucket key: dist_ocean, dist_mountain, forest_cc_size, heterogeneity, n_forest, n_mountain\n")

    for feat_name in ["dist_ocean", "dist_mountain", "forest_cc_size", "heterogeneity", "n_forest", "n_mountain"]:
        feat_vals = features[feat_name].astype(float)

        # Bin the feature
        if feat_name in ("n_forest", "n_mountain"):
            bins = sorted(set(feat_vals))
        elif feat_name == "heterogeneity":
            bins = sorted(set(feat_vals))
        elif feat_name == "forest_cc_size":
            bins = [0, 1, 5, 15, 50, 150, 500, 9999]
        else:
            bins = [0, 1, 2, 3, 5, 8, 12, 20, 40]

        report.append(f"\n### {feat_name}\n")
        report.append("| Bin | N cells | Mean wKL | Mean KL | Settlement Resid | Forest Resid |\n")
        report.append("|-----|---------|----------|---------|-----------------|-------------|\n")

        if feat_name in ("n_forest", "n_mountain", "heterogeneity"):
            for bval in bins:
                mask = feat_vals == bval
                if mask.sum() < 10:
                    continue
                mwkl = wkl_arr[mask].mean()
                mkl = kl_arr[mask].mean()
                sr = res_arr[mask, 1].mean()
                fr = res_arr[mask, 4].mean()
                report.append(f"| {bval} | {mask.sum()} | {mwkl:.6f} | {mkl:.6f} | {sr:+.5f} | {fr:+.5f} |\n")
        else:
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i+1]
                mask = (feat_vals >= lo) & (feat_vals < hi)
                if mask.sum() < 10:
                    continue
                mwkl = wkl_arr[mask].mean()
                mkl = kl_arr[mask].mean()
                sr = res_arr[mask, 1].mean()
                fr = res_arr[mask, 4].mean()
                report.append(f"| [{lo},{hi}) | {mask.sum()} | {mwkl:.6f} | {mkl:.6f} | {sr:+.5f} | {fr:+.5f} |\n")

    # =========================================================================
    # 3. INTERACTION EFFECTS
    # =========================================================================
    report.append("\n## 3. Interaction Effects\n")

    # Forest cells: coast x near_settlement interactions
    report.append("\n### Forest cells: Coast x Distance-to-Civ interaction\n")
    forest_mask = types == "F"
    report.append("| Coast | Dist Civ Bin | N | Mean wKL | Settle Resid | Port Resid | Forest Resid |\n")
    report.append("|-------|-------------|---|----------|-------------|-----------|-------------|\n")

    for is_coast in [True, False]:
        for dlo, dhi in [(0, 2), (2, 4), (4, 7), (7, 12), (12, 40)]:
            mask = forest_mask & (coast == is_coast) & \
                   (features["dist_civ"] >= dlo) & (features["dist_civ"] < dhi)
            if mask.sum() < 10:
                continue
            mwkl = wkl_arr[mask].mean()
            sr = res_arr[mask, 1].mean()
            pr = res_arr[mask, 2].mean()
            fr = res_arr[mask, 4].mean()
            report.append(f"| {is_coast} | [{dlo},{dhi}) | {mask.sum()} | {mwkl:.6f} | {sr:+.5f} | {pr:+.5f} | {fr:+.5f} |\n")

    # Forest near coast AND near settlement vs only near settlement
    report.append("\n### Forest: Coast + Near-Civ vs Only Near-Civ\n")
    near_civ = features["dist_civ"] < 4
    f_coast_civ = forest_mask & coast & near_civ
    f_only_civ = forest_mask & ~coast & near_civ
    f_coast_only = forest_mask & coast & ~near_civ
    f_neither = forest_mask & ~coast & ~near_civ

    for label_str, mask in [("Coast+NearCiv", f_coast_civ), ("OnlyNearCiv", f_only_civ),
                       ("CoastOnly", f_coast_only), ("Neither", f_neither)]:
        if mask.sum() < 5:
            continue
        mwkl = wkl_arr[mask].mean()
        mkl = kl_arr[mask].mean()
        mean_res = res_arr[mask].mean(axis=0)
        report.append(f"\n**{label_str}** ({mask.sum()} cells, mean wKL={mwkl:.6f}, mean KL={mkl:.6f})\n")
        report.append(f"  Residuals: Ocean={mean_res[0]:+.5f} Settle={mean_res[1]:+.5f} Port={mean_res[2]:+.5f} Ruin={mean_res[3]:+.5f} Forest={mean_res[4]:+.5f} Mtn={mean_res[5]:+.5f}\n")

    # =========================================================================
    # 4. BORDER vs INTERIOR CELLS
    # =========================================================================
    report.append("\n## 4. Border vs Interior Cell Errors\n")

    for label_str, mask in [("Border (edge row/col)", border),
                       ("Near-border (2 rows/cols)", near_border),
                       ("Interior", ~near_border)]:
        n = mask.sum()
        if n < 10:
            continue
        mwkl = wkl_arr[mask].mean()
        mkl = kl_arr[mask].mean()
        mean_res = res_arr[mask].mean(axis=0)
        report.append(f"\n### {label_str} ({n} cells)\n")
        report.append(f"Mean wKL: {mwkl:.6f}, Mean KL: {mkl:.6f}\n")
        report.append(f"Residuals: Ocean={mean_res[0]:+.5f} Settle={mean_res[1]:+.5f} Port={mean_res[2]:+.5f} Ruin={mean_res[3]:+.5f} Forest={mean_res[4]:+.5f} Mtn={mean_res[5]:+.5f}\n")

    # =========================================================================
    # 5. NONLINEAR DISTANCE EFFECT
    # =========================================================================
    report.append("\n## 5. Nonlinear Distance Effects\n")
    report.append("Checking if fractional distances matter (our model floors to integers).\n")

    report.append("\n### Settlement probability residual by fine-grained distance (forest cells)\n")
    report.append("| Dist Range | N | Mean wKL | Settle Resid | Actual Settle Prob | Predicted Settle |\n")
    report.append("|-----------|---|----------|-------------|-------------------|------------------|\n")

    for dlo in np.arange(0, 16, 0.5):
        dhi = dlo + 0.5
        mask = forest_mask & (features["dist_civ"] >= dlo) & (features["dist_civ"] < dhi)
        if mask.sum() < 20:
            continue
        mwkl = wkl_arr[mask].mean()
        sr = res_arr[mask, 1].mean()
        actual_s = gt_arr[mask, 1].mean()
        pred_s = pred_arr[mask, 1].mean()
        report.append(f"| [{dlo:.1f},{dhi:.1f}) | {mask.sum()} | {mwkl:.6f} | {sr:+.5f} | {actual_s:.5f} | {pred_s:.5f} |\n")

    # Check if distance interpolation is working — compare exact vs floor distance
    report.append("\n### Interpolation residual: cells at half-integer vs integer distances\n")
    dist_civ_arr = features["dist_civ"]
    frac_part = dist_civ_arr - np.floor(dist_civ_arr)
    near_int = (frac_part < 0.1) | (frac_part > 0.9)
    mid_frac = (frac_part > 0.3) & (frac_part < 0.7)

    for label_str, mask_extra in [("Near-integer dist", near_int), ("Mid-fractional dist", mid_frac)]:
        mask = forest_mask & mask_extra & (dist_civ_arr > 0.5) & (dist_civ_arr < 10)
        if mask.sum() < 20:
            continue
        mwkl = wkl_arr[mask].mean()
        mkl = kl_arr[mask].mean()
        report.append(f"\n**{label_str}** ({mask.sum()} cells): mean wKL={mwkl:.6f}, mean KL={mkl:.6f}\n")

    # =========================================================================
    # 6. PER-REGIME ANALYSIS
    # =========================================================================
    report.append("\n## 6. Per-Regime Error Breakdown\n")

    for regime in ["harsh", "moderate", "prosperous"]:
        mask = regimes == regime
        if mask.sum() == 0:
            continue
        n = mask.sum()
        mwkl = wkl_arr[mask].mean()
        mkl = kl_arr[mask].mean()
        mean_res = res_arr[mask].mean(axis=0)

        report.append(f"\n### {regime} ({n} cells, mean wKL={mwkl:.6f})\n")
        report.append(f"Residuals: Ocean={mean_res[0]:+.5f} Settle={mean_res[1]:+.5f} Port={mean_res[2]:+.5f} Ruin={mean_res[3]:+.5f} Forest={mean_res[4]:+.5f} Mtn={mean_res[5]:+.5f}\n")

        # High-error cells in this regime
        regime_wkl = wkl_arr[mask]
        p90 = np.percentile(regime_wkl, 90)
        high_err = mask & (wkl_arr > p90)
        if high_err.sum() > 0:
            he_res = res_arr[high_err].mean(axis=0)
            report.append(f"Top 10% error cells ({high_err.sum()} cells, wKL>{p90:.4f}):\n")
            report.append(f"  Residuals: Ocean={he_res[0]:+.5f} Settle={he_res[1]:+.5f} Port={he_res[2]:+.5f} Ruin={he_res[3]:+.5f} Forest={he_res[4]:+.5f} Mtn={he_res[5]:+.5f}\n")

    # =========================================================================
    # 7. WORST CELLS: Top wKL contributors
    # =========================================================================
    report.append("\n## 7. Worst Cells Analysis\n")

    # Top 5% cells by wKL
    p95 = np.percentile(wkl_arr, 95)
    worst_mask = wkl_arr > p95
    worst_n = worst_mask.sum()
    worst_total_wkl = wkl_arr[worst_mask].sum()

    report.append(f"\nTop 5% cells ({worst_n} cells) contribute {worst_total_wkl:.2f} wKL ({100*worst_total_wkl/total_wkl:.1f}% of total)\n")

    # Characterize these cells
    report.append("\n### Worst cells by type:\n")
    for ctype in ["S", "P", "F", "L"]:
        type_worst = worst_mask & (types == ctype)
        if type_worst.sum() == 0:
            continue
        n = type_worst.sum()
        pct = 100 * n / worst_n
        mean_res = res_arr[type_worst].mean(axis=0)
        report.append(f"- {ctype}: {n} ({pct:.1f}%), residuals: Settle={mean_res[1]:+.5f} Port={mean_res[2]:+.5f} Forest={mean_res[4]:+.5f}\n")

    report.append("\n### Worst cells feature distribution:\n")
    for feat_name in ["dist_civ", "dist_ocean", "heterogeneity", "forest_cc_size"]:
        vals = features[feat_name][worst_mask]
        all_vals = features[feat_name]
        report.append(f"- {feat_name}: worst mean={vals.mean():.2f} (all mean={all_vals.mean():.2f}), worst median={np.median(vals):.2f}\n")

    report.append(f"- coast: worst {100*coast[worst_mask].mean():.1f}% vs all {100*coast.mean():.1f}%\n")
    report.append(f"- border: worst {100*near_border[worst_mask].mean():.1f}% vs all {100*near_border.mean():.1f}%\n")

    # =========================================================================
    # 8. PER-ROUND ANALYSIS - which rounds contribute most error?
    # =========================================================================
    report.append("\n## 8. Per-Round Error Contribution\n")
    report.append("| Round | Regime | N cells | Mean wKL | Mean KL | Top Residual Class |\n")
    report.append("|-------|--------|---------|----------|---------|-------------------|\n")

    for rn in sorted(set(rounds_arr)):
        mask = rounds_arr == rn
        n = mask.sum()
        mwkl = wkl_arr[mask].mean()
        mkl = kl_arr[mask].mean()
        mean_res = res_arr[mask].mean(axis=0)
        worst_class = CLASS_NAMES[np.argmax(np.abs(mean_res))]
        regime = model["round_regimes"].get(rn, "?")
        report.append(f"| R{rn} | {regime} | {n} | {mwkl:.6f} | {mkl:.6f} | {worst_class} ({mean_res[np.argmax(np.abs(mean_res))]:+.5f}) |\n")

    # =========================================================================
    # 9. FOREST CC SIZE INTERACTION
    # =========================================================================
    report.append("\n## 9. Forest Connected Component Size Effect\n")
    report.append("Do large forest patches behave differently from small ones?\n")

    report.append("| CC Size Bin | N | Mean wKL | Settle Resid | Forest Resid | Actual Forest Prob |\n")
    report.append("|------------|---|----------|-------------|-------------|-------------------|\n")

    for lo, hi in [(0, 1), (1, 5), (5, 15), (15, 50), (50, 150), (150, 500), (500, 9999)]:
        mask = forest_mask & (features["forest_cc_size"] >= lo) & (features["forest_cc_size"] < hi)
        if mask.sum() < 10:
            continue
        mwkl = wkl_arr[mask].mean()
        sr = res_arr[mask, 1].mean()
        fr = res_arr[mask, 4].mean()
        actual_f = gt_arr[mask, 4].mean()
        report.append(f"| [{lo},{hi}) | {mask.sum()} | {mwkl:.6f} | {sr:+.5f} | {fr:+.5f} | {actual_f:.5f} |\n")

    # =========================================================================
    # 10. POSITION ON MAP (quadrant analysis)
    # =========================================================================
    report.append("\n## 10. Map Position Effects\n")

    y_arr = np.array([r["y"] for r in all_residuals])
    x_arr = np.array([r["x"] for r in all_residuals])

    report.append("| Quadrant | N | Mean wKL | Settle Resid | Forest Resid |\n")
    report.append("|----------|---|----------|-------------|-------------|\n")

    for qlabel, ymask, xmask in [
        ("Top-Left", y_arr < 20, x_arr < 20),
        ("Top-Right", y_arr < 20, x_arr >= 20),
        ("Bottom-Left", y_arr >= 20, x_arr < 20),
        ("Bottom-Right", y_arr >= 20, x_arr >= 20),
    ]:
        mask = ymask & xmask
        if mask.sum() < 10:
            continue
        mwkl = wkl_arr[mask].mean()
        sr = res_arr[mask, 1].mean()
        fr = res_arr[mask, 4].mean()
        report.append(f"| {qlabel} | {mask.sum()} | {mwkl:.6f} | {sr:+.5f} | {fr:+.5f} |\n")

    # Radial distance from center
    report.append("\n### Radial distance from map center\n")
    center_y, center_x = 20, 20
    radial_dist = np.sqrt((y_arr - center_y)**2 + (x_arr - center_x)**2)

    report.append("| Radial Dist | N | Mean wKL | Settle Resid | Forest Resid |\n")
    report.append("|------------|---|----------|-------------|-------------|\n")
    for dlo, dhi in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]:
        mask = (radial_dist >= dlo) & (radial_dist < dhi)
        if mask.sum() < 10:
            continue
        mwkl = wkl_arr[mask].mean()
        sr = res_arr[mask, 1].mean()
        fr = res_arr[mask, 4].mean()
        report.append(f"| [{dlo},{dhi}) | {mask.sum()} | {mwkl:.6f} | {sr:+.5f} | {fr:+.5f} |\n")

    # =========================================================================
    # 11. SPECIFIC HIGH-IMPACT PATTERNS
    # =========================================================================
    report.append("\n## 11. Actionable Patterns Summary\n")
    report.append("Patterns ranked by total wKL impact (cells * mean_wkl).\n\n")

    patterns = []

    # Pattern: Forest cells with high heterogeneity
    for het_thresh in [3, 4, 5]:
        mask = forest_mask & (features["heterogeneity"] >= het_thresh)
        if mask.sum() > 20:
            impact = wkl_arr[mask].sum()
            patterns.append((f"Forest + heterogeneity>={het_thresh}", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Pattern: Forest large CC
    mask = forest_mask & (features["forest_cc_size"] >= 50)
    if mask.sum() > 20:
        impact = wkl_arr[mask].sum()
        patterns.append(("Forest large CC (>=50)", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Pattern: Coastal forest near civ
    mask = forest_mask & coast & (features["dist_civ"] < 3)
    if mask.sum() > 10:
        impact = wkl_arr[mask].sum()
        patterns.append(("Coastal forest near civ (<3)", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Pattern: Forest near mountain
    mask = forest_mask & (features["dist_mountain"] < 3)
    if mask.sum() > 10:
        impact = wkl_arr[mask].sum()
        patterns.append(("Forest near mountain (<3)", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Pattern: Settlement cells
    mask = types == "S"
    if mask.sum() > 10:
        impact = wkl_arr[mask].sum()
        patterns.append(("Settlement cells", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Pattern: Port cells
    mask = types == "P"
    if mask.sum() > 10:
        impact = wkl_arr[mask].sum()
        patterns.append(("Port cells", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Pattern: Near-border dynamic cells
    mask = near_border
    if mask.sum() > 20:
        impact = wkl_arr[mask].sum()
        patterns.append(("Near-border cells", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Pattern: dist_ocean < 2 (very close to ocean but not coastal)
    mask = (features["dist_ocean"] < 2) & ~coast & forest_mask
    if mask.sum() > 10:
        impact = wkl_arr[mask].sum()
        patterns.append(("Forest very near ocean (non-coast)", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Pattern: n_mountain > 0 on non-mountain cells
    mask = features["n_mountain"] > 0
    if mask.sum() > 10:
        impact = wkl_arr[mask].sum()
        patterns.append(("Adjacent to mountain", mask.sum(), wkl_arr[mask].mean(), impact, res_arr[mask].mean(axis=0)))

    # Sort by total wKL impact
    patterns.sort(key=lambda x: -x[3])

    for name, n, mean_wkl_p, total_impact, mean_res in patterns:
        report.append(f"### {name}\n")
        report.append(f"- Cells: {n}, Mean wKL: {mean_wkl_p:.6f}, Total wKL impact: {total_impact:.2f} ({100*total_impact/total_wkl:.1f}% of total)\n")
        report.append(f"- Residuals: Ocean={mean_res[0]:+.5f} Settle={mean_res[1]:+.5f} Port={mean_res[2]:+.5f} Ruin={mean_res[3]:+.5f} Forest={mean_res[4]:+.5f} Mtn={mean_res[5]:+.5f}\n")
        # What should we predict?
        report.append(f"- Correction needed: ")
        corrections = []
        for c in range(6):
            if abs(mean_res[c]) > 0.005:
                direction = "increase" if mean_res[c] > 0 else "decrease"
                corrections.append(f"{direction} {CLASS_NAMES[c]} by ~{abs(mean_res[c]):.4f}")
        report.append(", ".join(corrections) if corrections else "residuals are small")
        report.append("\n\n")

    # =========================================================================
    # 12. CORRELATION MATRIX: features vs per-class residuals
    # =========================================================================
    report.append("\n## 12. Feature-Residual Correlations\n")
    report.append("Pearson correlation between features and per-class residuals.\n")
    report.append("Strong correlations indicate features we should incorporate.\n\n")

    report.append("| Feature | Ocean | Settlement | Port | Ruin | Forest | Mountain |\n")
    report.append("|---------|-------|-----------|------|------|--------|----------|\n")

    for feat_name in ["dist_ocean", "dist_mountain", "forest_cc_size", "heterogeneity",
                      "n_forest", "n_mountain", "dist_civ", "n_ocean", "n_civ"]:
        fvals = features[feat_name].astype(float)
        corrs = []
        for c in range(6):
            r = np.corrcoef(fvals, res_arr[:, c])[0, 1]
            corrs.append(r)
        strong = [i for i, r in enumerate(corrs) if abs(r) > 0.05]
        markers = ["**" if abs(c) > 0.05 else "" for c in corrs]
        report.append(f"| {feat_name} | {markers[0]}{corrs[0]:+.4f}{markers[0]} | {markers[1]}{corrs[1]:+.4f}{markers[1]} | {markers[2]}{corrs[2]:+.4f}{markers[2]} | {markers[3]}{corrs[3]:+.4f}{markers[3]} | {markers[4]}{corrs[4]:+.4f}{markers[4]} | {markers[5]}{corrs[5]:+.4f}{markers[5]} |\n")

    # Write report
    output_path = Path(__file__).parent / "residual_audit.md"
    with open(output_path, "w") as f:
        f.writelines(report)

    print(f"\nReport written to {output_path}")
    print(f"Total wKL: {total_wkl:.2f}")
    print(f"Top 5% cells contribute: {100*worst_total_wkl/total_wkl:.1f}%")

    return report


if __name__ == "__main__":
    run_analysis()
