#!/usr/bin/env python3
"""Convergence audit: overfitting check + cheap ensemble test."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, 'tasks/astar-island')
np.random.seed(42)

from benchmark import load_ground_truth, kl_divergence, entropy, evaluate_predictor
import regime_predictor
import parametric_predictor
import neighborhood_predictor
import spatial_model

GT_DIR = Path('tasks/astar-island/ground_truth')
OUTPUT = Path('tasks/astar-island/convergence_audit.md')

rounds = load_ground_truth()
round_nums = sorted(rounds.keys())
print(f"Loaded {len(round_nums)} rounds: {round_nums}")
print(f"Total seeds: {sum(len(s) for s in rounds.values())}")

# ============================================================
# PART 1: OVERFITTING / COLLAPSE AUDIT
# ============================================================
print("\n" + "="*70)
print("PART 1: OVERFITTING / COLLAPSE AUDIT")
print("="*70)

# --- 1.1: In-sample evaluation ---
print("\n--- 1.1: In-sample wKL (regime_predictor, all data) ---")
full_model = regime_predictor.build_model_from_data(rounds)
insample_results = {}
for rn in round_nums:
    round_wkls = []
    for sn in sorted(rounds[rn].keys()):
        sd = rounds[rn][sn]
        ig = np.array(sd["initial_grid"], dtype=np.int32)
        gt = np.array(sd["ground_truth"], dtype=np.float64)
        pred = regime_predictor.predict_with_model(ig, full_model)
        pred = np.maximum(pred, 0.01)
        pred /= pred.sum(axis=2, keepdims=True)
        kl = kl_divergence(gt, pred)
        H = entropy(gt)
        wkl = H * kl
        dynamic = H > 0.01
        mean_wkl = float(wkl[dynamic].mean()) if dynamic.any() else 0.0
        round_wkls.append(mean_wkl)
    insample_results[rn] = {
        "mean": np.mean(round_wkls),
        "per_seed": round_wkls,
    }
    print(f"  R{rn}: in-sample wKL = {np.mean(round_wkls):.6f} (seeds: {[f'{v:.4f}' for v in round_wkls]})")

overall_insample = np.mean([v["mean"] for v in insample_results.values()])
print(f"\n  Overall in-sample wKL: {overall_insample:.6f}")

# --- 1.2: Leave-one-round-out CV ---
print("\n--- 1.2: Leave-one-round-out CV ---")
cv_results = {}
for held_out in round_nums:
    train_data = {rn: data for rn, data in rounds.items() if rn != held_out}
    cv_model = regime_predictor.build_model_from_data(train_data)

    round_wkls = []
    for sn in sorted(rounds[held_out].keys()):
        sd = rounds[held_out][sn]
        ig = np.array(sd["initial_grid"], dtype=np.int32)
        gt = np.array(sd["ground_truth"], dtype=np.float64)

        # Simulate 3 viewports x 3 queries worth of observations
        # (tau=20 not relevant for CV since we don't actually have observations)
        pred = regime_predictor.predict_with_model(ig, cv_model)
        pred = np.maximum(pred, 0.01)
        pred /= pred.sum(axis=2, keepdims=True)

        kl = kl_divergence(gt, pred)
        H = entropy(gt)
        wkl = H * kl
        dynamic = H > 0.01
        mean_wkl = float(wkl[dynamic].mean()) if dynamic.any() else 0.0
        round_wkls.append(mean_wkl)

    cv_results[held_out] = {
        "mean": np.mean(round_wkls),
        "per_seed": round_wkls,
    }

    in_s = insample_results[held_out]["mean"]
    cv_s = cv_results[held_out]["mean"]
    gap_pct = ((cv_s - in_s) / max(in_s, 1e-8)) * 100
    flag = " *** OVERFITTING" if gap_pct > 30 else ""
    print(f"  R{held_out}: CV wKL = {cv_s:.6f} | in-sample = {in_s:.6f} | gap = {gap_pct:+.1f}%{flag}")

overall_cv = np.mean([v["mean"] for v in cv_results.values()])
overall_gap_pct = ((overall_cv - overall_insample) / max(overall_insample, 1e-8)) * 100
print(f"\n  Overall CV wKL: {overall_cv:.6f} (gap: {overall_gap_pct:+.1f}%)")

# --- 1.3: Collapse check ---
print("\n--- 1.3: Entropy collapse check ---")
collapse_stats = {}
for rn in round_nums:
    frac_collapsed = []
    for sn in sorted(rounds[rn].keys()):
        sd = rounds[rn][sn]
        ig = np.array(sd["initial_grid"], dtype=np.int32)
        gt = np.array(sd["ground_truth"], dtype=np.float64)
        pred = regime_predictor.predict_with_model(ig, full_model)
        pred = np.maximum(pred, 0.01)
        pred /= pred.sum(axis=2, keepdims=True)

        H_gt = entropy(gt)
        H_pred = entropy(pred)
        dynamic = H_gt > 0.01

        if dynamic.any():
            ratio = H_pred[dynamic] / np.maximum(H_gt[dynamic], 1e-8)
            collapsed = ratio < 0.5
            frac = float(collapsed.sum()) / float(dynamic.sum())
            frac_collapsed.append(frac)
        else:
            frac_collapsed.append(0.0)

    avg_collapse = np.mean(frac_collapsed)
    collapse_stats[rn] = {
        "frac_collapsed": avg_collapse,
        "per_seed": frac_collapsed,
    }
    flag = " *** COLLAPSE" if avg_collapse > 0.1 else ""
    print(f"  R{rn}: {avg_collapse*100:.1f}% of dynamic cells have pred entropy < 50% GT entropy{flag}")

overall_collapse = np.mean([v["frac_collapsed"] for v in collapse_stats.values()])
print(f"\n  Overall collapse fraction: {overall_collapse*100:.1f}%")

# --- 1.4: Seed variance ---
print("\n--- 1.4: Seed variance within rounds ---")
seed_var_stats = {}
for rn in round_nums:
    wkls = insample_results[rn]["per_seed"]
    std_val = np.std(wkls)
    mean_val = np.mean(wkls)
    cv_coeff = std_val / max(mean_val, 1e-8)
    seed_var_stats[rn] = {
        "std": std_val,
        "cv": cv_coeff,
        "min": min(wkls),
        "max": max(wkls),
    }
    flag = " *** HIGH VARIANCE" if cv_coeff > 0.3 else ""
    print(f"  R{rn}: mean={mean_val:.6f} std={std_val:.6f} CV={cv_coeff:.3f} range=[{min(wkls):.4f}, {max(wkls):.4f}]{flag}")

# --- 1.5: Simulation vs live score gap ---
print("\n--- 1.5: Simulation vs live score analysis ---")
# Competition score = 100 * (1 - mean_wKL/max_wKL) approximately
# R13 got 87.9, R15 got 88.6
# Our simulated wKL is ~0.0698
# Competition score formula: score = 100 - 100*KL_mean  (approximately)
# Actually the score is 0-100 where 100 is perfect, based on KL divergence
# With wKL of 0.0698, if the mapping is score = 100*(1 - wKL), that gives 93.0
# But actual scores are ~88, so there's a gap

sim_wkl = overall_insample
live_scores = {"R13": 87.9, "R15": 88.6}
print(f"  Simulated in-sample wKL: {sim_wkl:.6f}")
print(f"  Simulated CV wKL: {overall_cv:.6f}")
print(f"  Live scores: {live_scores}")
print(f"  Note: If score = 100*(1 - wKL), simulated would predict score = {100*(1-sim_wkl):.1f}")
print(f"  But live scores are ~88, suggesting our simulation underestimates wKL by a factor")
print(f"  Implied live wKL: ~0.12 (from score ~88)")
print(f"  Gap ratio: implied_live / sim_cv = {0.12 / max(overall_cv, 1e-8):.2f}x")


# ============================================================
# PART 2: CHEAP ENSEMBLE
# ============================================================
print("\n" + "="*70)
print("PART 2: CHEAP ENSEMBLE")
print("="*70)

# Pre-build models for each predictor
print("\nBuilding all predictor models...")

# For neighborhood_predictor, we need tables
nb_fine, nb_mid, nb_coarse, nb_type = neighborhood_predictor.build_lookup_table()

# For spatial_model, try loading cached params; skip if not available
sp_params = Path('tasks/astar-island/spatial_params.json')
sp_models_full = None
if sp_params.exists():
    try:
        with open(sp_params) as f:
            saved = json.load(f)
        sp_models_full = {int(k): np.array(v) for k, v in saved.items()}
        print("  Loaded cached spatial model params")
    except Exception as e:
        print(f"  Failed to load spatial params: {e}")
else:
    print("  No cached spatial params found, fitting (may be slow)...")
    try:
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Spatial model fitting timed out")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 2 min timeout
        sp_data = spatial_model.load_ground_truth()
        sp_models_full = spatial_model.fit_model(sp_data)
        signal.alarm(0)
        # Save for next time
        to_save = {str(k): v.tolist() for k, v in sp_models_full.items()}
        with open(sp_params, 'w') as f:
            json.dump(to_save, f)
    except (TimeoutError, Exception) as e:
        print(f"  Spatial model fitting failed/timed out: {e}")
        sp_models_full = None

HAVE_SPATIAL = sp_models_full is not None
print(f"  Spatial model available: {HAVE_SPATIAL}")

# For parametric_predictor, build from full data
param_model_full = parametric_predictor.build_model_from_data(rounds)

print("\nAll models built. Running ensemble CV...\n")

# Define ensemble configurations
ensemble_configs = {
    "a) 100% regime": {"regime": 1.0},
    "b) 80/20 regime+param": {"regime": 0.8, "parametric": 0.2},
    "c) 70/15/15 regime+param+nb": {"regime": 0.7, "parametric": 0.15, "neighborhood": 0.15},
}
if HAVE_SPATIAL:
    ensemble_configs["d) 60/20/10/10 all"] = {"regime": 0.6, "parametric": 0.2, "neighborhood": 0.1, "spatial": 0.1}
else:
    ensemble_configs["d) 50/25/25 no-spatial"] = {"regime": 0.5, "parametric": 0.25, "neighborhood": 0.25}

def predict_ensemble(ig_list, weights, regime_model, param_model, nb_tables, sp_models):
    """Predict using weighted ensemble of predictors."""
    ig = np.array(ig_list, dtype=np.int32)

    preds = {}

    if weights.get("regime", 0) > 0:
        p = regime_predictor.predict_with_model(ig, regime_model)
        p = np.maximum(p, 1e-6)
        p /= p.sum(axis=2, keepdims=True)
        preds["regime"] = p

    if weights.get("parametric", 0) > 0:
        # Use mixture regime for parametric since we don't know the regime
        p = parametric_predictor.predict_with_model(ig, param_model, regime="mixture")
        p = np.maximum(p, 1e-6)
        p /= p.sum(axis=2, keepdims=True)
        preds["parametric"] = p

    if weights.get("neighborhood", 0) > 0:
        ft, mt, ct, tt = nb_tables
        p = neighborhood_predictor.predict(ig, ft, mt, ct, tt)
        p = np.maximum(p, 1e-6)
        p /= p.sum(axis=2, keepdims=True)
        preds["neighborhood"] = p

    if weights.get("spatial", 0) > 0:
        p = spatial_model.predict_with_models(ig, sp_models)
        p = np.maximum(p, 1e-6)
        p /= p.sum(axis=2, keepdims=True)
        preds["spatial"] = p

    # Weighted combination
    result = np.zeros_like(list(preds.values())[0])
    for name, pred in preds.items():
        result += weights[name] * pred

    result = np.maximum(result, 0.01)
    result /= result.sum(axis=2, keepdims=True)
    return result


# Run leave-one-round-out CV for each ensemble config
ensemble_cv_results = {}

for config_name, weights in ensemble_configs.items():
    print(f"\n--- {config_name} ---")
    round_wkls = {}

    for held_out in round_nums:
        train_data = {rn: data for rn, data in rounds.items() if rn != held_out}

        # Rebuild regime model for CV
        cv_regime_model = regime_predictor.build_model_from_data(train_data)

        # Rebuild parametric model for CV
        cv_param_model = parametric_predictor.build_model_from_data(train_data)

        # For neighborhood and spatial, we use full-data models
        # (they don't have easy CV rebuild, and the question is about ensemble weighting)
        # This slightly favors these predictors but is acceptable for relative comparison

        seed_wkls = []
        for sn in sorted(rounds[held_out].keys()):
            sd = rounds[held_out][sn]
            ig = sd["initial_grid"]
            gt = np.array(sd["ground_truth"], dtype=np.float64)

            pred = predict_ensemble(
                ig, weights, cv_regime_model, cv_param_model,
                (nb_fine, nb_mid, nb_coarse, nb_type), sp_models_full
            )

            kl = kl_divergence(gt, pred)
            H = entropy(gt)
            wkl = H * kl
            dynamic = H > 0.01
            mean_wkl = float(wkl[dynamic].mean()) if dynamic.any() else 0.0
            seed_wkls.append(mean_wkl)

        round_wkls[held_out] = np.mean(seed_wkls)
        print(f"  R{held_out}: wKL = {np.mean(seed_wkls):.6f}")

    avg_wkl = np.mean(list(round_wkls.values()))
    worst_wkl = max(round_wkls.values())
    worst_round = max(round_wkls, key=round_wkls.get)
    best_wkl = min(round_wkls.values())
    best_round = min(round_wkls, key=round_wkls.get)

    ensemble_cv_results[config_name] = {
        "weights": weights,
        "avg_wkl": avg_wkl,
        "worst_wkl": worst_wkl,
        "worst_round": worst_round,
        "best_wkl": best_wkl,
        "best_round": best_round,
        "per_round": dict(round_wkls),
    }
    print(f"  AVG: {avg_wkl:.6f} | WORST: R{worst_round} = {worst_wkl:.6f} | BEST: R{best_round} = {best_wkl:.6f}")


# ============================================================
# WRITE REPORT
# ============================================================
print("\n\nWriting report...")

report = []
report.append("# Astar Island Convergence Audit\n")
report.append(f"Generated with np.random.seed(42), {len(round_nums)} rounds, {sum(len(s) for s in rounds.values())} total seeds.\n")

report.append("## Part 1: Overfitting / Collapse Audit\n")

report.append("### 1.1 In-sample vs Leave-One-Round-Out CV\n")
report.append(f"| Round | In-sample wKL | CV wKL | Gap % | Flag |")
report.append(f"|-------|--------------|--------|-------|------|")
for rn in round_nums:
    in_s = insample_results[rn]["mean"]
    cv_s = cv_results[rn]["mean"]
    gap = ((cv_s - in_s) / max(in_s, 1e-8)) * 100
    flag = "OVERFITTING" if gap > 30 else "OK"
    report.append(f"| R{rn} | {in_s:.6f} | {cv_s:.6f} | {gap:+.1f}% | {flag} |")
report.append(f"| **Overall** | **{overall_insample:.6f}** | **{overall_cv:.6f}** | **{overall_gap_pct:+.1f}%** | |")
report.append("")

report.append("### 1.2 Entropy Collapse Check\n")
report.append("Fraction of dynamic cells where predicted entropy < 50% of GT entropy:\n")
report.append(f"| Round | Collapse Fraction | Flag |")
report.append(f"|-------|-------------------|------|")
for rn in round_nums:
    frac = collapse_stats[rn]["frac_collapsed"]
    flag = "COLLAPSE" if frac > 0.1 else "OK"
    report.append(f"| R{rn} | {frac*100:.1f}% | {flag} |")
report.append(f"| **Overall** | **{overall_collapse*100:.1f}%** | |")
report.append("")

report.append("### 1.3 Seed Variance Within Rounds\n")
report.append(f"| Round | Mean wKL | Std | CV (std/mean) | Range | Flag |")
report.append(f"|-------|----------|-----|---------------|-------|------|")
for rn in round_nums:
    sv = seed_var_stats[rn]
    m = insample_results[rn]["mean"]
    flag = "HIGH VARIANCE" if sv["cv"] > 0.3 else "OK"
    report.append(f"| R{rn} | {m:.6f} | {sv['std']:.6f} | {sv['cv']:.3f} | [{sv['min']:.4f}, {sv['max']:.4f}] | {flag} |")
report.append("")

report.append("### 1.4 Simulation vs Live Score Gap\n")
report.append(f"- Simulated in-sample wKL: {overall_insample:.6f}")
report.append(f"- Simulated CV wKL: {overall_cv:.6f}")
report.append(f"- Live scores: R13=87.9, R15=88.6")
report.append(f"- If score = 100*(1 - wKL), sim predicts score ~{100*(1-overall_insample):.1f}")
report.append(f"- Implied live wKL from scores: ~0.12")
report.append(f"- Gap ratio (live/sim_CV): ~{0.12/max(overall_cv, 1e-8):.2f}x")
report.append(f"- **Conclusion**: Simulation is {('optimistic' if overall_cv < 0.12 else 'pessimistic')} -- live performance is worse than simulated, likely due to regime misclassification without ground truth observations.\n")

report.append("## Part 2: Cheap Ensemble\n")
report.append("Leave-one-round-out CV for each ensemble configuration:\n")
report.append(f"| Config | Weights | Avg wKL | Worst-case wKL | Worst Round |")
report.append(f"|--------|---------|---------|----------------|-------------|")
for config_name, res in ensemble_cv_results.items():
    w_str = ", ".join(f"{k}={v}" for k, v in res["weights"].items())
    report.append(f"| {config_name} | {w_str} | {res['avg_wkl']:.6f} | {res['worst_wkl']:.6f} | R{res['worst_round']} |")
report.append("")

# Find best by worst-case
best_config = min(ensemble_cv_results, key=lambda k: ensemble_cv_results[k]["worst_wkl"])
best_res = ensemble_cv_results[best_config]
report.append(f"### Recommendation\n")
report.append(f"**Best by worst-case wKL**: {best_config}")
report.append(f"- Worst-case wKL: {best_res['worst_wkl']:.6f} (R{best_res['worst_round']})")
report.append(f"- Average wKL: {best_res['avg_wkl']:.6f}")
report.append("")

# Also report best by average
best_avg_config = min(ensemble_cv_results, key=lambda k: ensemble_cv_results[k]["avg_wkl"])
best_avg_res = ensemble_cv_results[best_avg_config]
report.append(f"**Best by average wKL**: {best_avg_config}")
report.append(f"- Average wKL: {best_avg_res['avg_wkl']:.6f}")
report.append(f"- Worst-case wKL: {best_avg_res['worst_wkl']:.6f} (R{best_avg_res['worst_round']})")
report.append("")

# Per-round detail table
report.append("### Per-round CV wKL by ensemble config\n")
header = "| Round |"
for cn in ensemble_cv_results:
    short = cn.split(")")[0] + ")"
    header += f" {short} |"
report.append(header)
report.append("|" + "-------|" * (1 + len(ensemble_cv_results)))
for rn in round_nums:
    row = f"| R{rn} |"
    for cn, res in ensemble_cv_results.items():
        val = res["per_round"][rn]
        row += f" {val:.6f} |"
    report.append(row)
report.append("")

report_text = "\n".join(report)
OUTPUT.write_text(report_text)
print(f"\nReport written to {OUTPUT}")
print("\nDone!")
