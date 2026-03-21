#!/usr/bin/env python3
"""Score diagnosis: where are we losing points?

For each round, compute:
1. wKL breakdown by cell category (settlement, frontier, coastal, forest, etc.)
2. In-sample ceiling (if we had the right template)
3. Out-of-sample score (pooled prior from other rounds)
4. Contribution of each category to total score
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve

sys.path.insert(0, str(Path(__file__).parent))

GT_DIR = Path(__file__).parent / "ground_truth"
N_CLASSES = 6

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11
EMPTY = 0

KERNEL = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)


def kl_per_cell(p, q):
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def entropy_per_cell(p):
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


def classify_cells(ig):
    """Classify each cell into diagnostic categories."""
    h, w = ig.shape
    categories = np.empty((h, w), dtype=object)

    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN

    if civ.any():
        dist = distance_transform_edt(~civ)
    else:
        dist = np.full((h, w), 99.0)

    n_ocean = convolve((ig == OCEAN).astype(np.int32), KERNEL, mode="constant")
    n_civ = convolve(civ.astype(np.int32), KERNEL, mode="constant")

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            d = float(dist[y, x])
            no = int(n_ocean[y, x])
            nc = int(n_civ[y, x])

            if code == OCEAN:
                categories[y, x] = "ocean"
            elif code == MOUNTAIN:
                categories[y, x] = "mountain"
            elif code == SETTLEMENT:
                categories[y, x] = "init_settlement"
            elif code == PORT:
                categories[y, x] = "init_port"
            elif no >= 2 and d <= 4:
                categories[y, x] = "coastal_frontier"
            elif code == FOREST and d <= 3:
                categories[y, x] = "forest_near_civ"
            elif d <= 2:
                categories[y, x] = "near_civ"
            elif d <= 5:
                categories[y, x] = "mid_range"
            elif d <= 10:
                categories[y, x] = "far_range"
            else:
                categories[y, x] = "remote"

    return categories


def load_all_gt():
    """Load all ground truth grouped by round."""
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        parts = f.stem.split("_")
        rn = int(parts[0].replace("round", ""))
        sn = int(parts[1].replace("seed", ""))
        with open(f) as fh:
            data = json.load(fh)
        if "ground_truth" not in data or "initial_grid" not in data:
            continue
        if rn not in rounds:
            rounds[rn] = {}
        rounds[rn][sn] = {
            "initial_grid": np.array(data["initial_grid"], dtype=np.int32),
            "ground_truth": np.array(data["ground_truth"], dtype=np.float64),
        }
    return rounds


def build_pooled_prior(rounds_data, exclude_round=None):
    """Build simple pooled average prior by cell type + distance band."""
    buckets = defaultdict(lambda: {"sum": np.zeros(N_CLASSES), "count": 0})

    for rn, seeds in rounds_data.items():
        if rn == exclude_round:
            continue
        for sn, sd in seeds.items():
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]
            cats = classify_cells(ig)
            h, w = ig.shape
            for y in range(h):
                for x in range(w):
                    cat = cats[y, x]
                    if cat in ("ocean", "mountain"):
                        continue
                    buckets[cat]["sum"] += gt[y, x]
                    buckets[cat]["count"] += 1

    priors = {}
    for cat, data in buckets.items():
        if data["count"] > 0:
            p = data["sum"] / data["count"]
            p = np.maximum(p, 0.005)
            p /= p.sum()
            priors[cat] = p
        else:
            priors[cat] = np.ones(N_CLASSES) / N_CLASSES
    return priors


def predict_with_prior(ig, priors):
    """Simple prediction using category priors."""
    h, w = ig.shape
    pred = np.zeros((h, w, N_CLASSES))
    cats = classify_cells(ig)

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == OCEAN:
                pred[y, x, 0] = 1.0
            elif code == MOUNTAIN:
                pred[y, x, 5] = 1.0
            else:
                cat = cats[y, x]
                pred[y, x] = priors.get(cat, np.ones(N_CLASSES) / N_CLASSES)

    pred = np.maximum(pred, 0.005)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def score_breakdown(ig, gt, pred):
    """Compute wKL breakdown by cell category."""
    cats = classify_cells(ig)
    kl = kl_per_cell(gt, pred)
    H = entropy_per_cell(gt)
    wkl = H * kl

    results = {}
    all_cats = np.unique(cats)
    total_weighted_kl = 0.0
    total_weight = 0.0

    for cat in all_cats:
        mask = cats == cat
        if not mask.any():
            continue

        dynamic = (H > 0.01) & mask
        if not dynamic.any():
            results[cat] = {"mean_wkl": 0.0, "mean_kl": 0.0, "count": int(mask.sum()),
                           "dynamic_count": 0, "total_wkl": 0.0, "mean_entropy": 0.0}
            continue

        cat_wkl = wkl[dynamic]
        cat_kl = kl[dynamic]
        cat_H = H[dynamic]

        results[cat] = {
            "mean_wkl": float(cat_wkl.mean()),
            "mean_kl": float(cat_kl.mean()),
            "count": int(mask.sum()),
            "dynamic_count": int(dynamic.sum()),
            "total_wkl": float(cat_wkl.sum()),
            "mean_entropy": float(cat_H.mean()),
        }
        total_weighted_kl += cat_wkl.sum()
        total_weight += cat_H.sum()

    # Compute fraction of total loss per category
    for cat in results:
        if total_weighted_kl > 0:
            results[cat]["pct_of_total_loss"] = results[cat]["total_wkl"] / total_weighted_kl * 100
        else:
            results[cat]["pct_of_total_loss"] = 0.0

    return results


def main():
    rounds = load_all_gt()
    print(f"Loaded {len(rounds)} rounds: {sorted(rounds.keys())}")

    # Import our predictors
    import template_predictor as tp
    tp.get_model()

    all_results = {}

    for rn in sorted(rounds.keys()):
        print(f"\n{'='*60}")
        print(f"ROUND {rn}")
        print(f"{'='*60}")

        # Build out-of-sample prior (exclude this round)
        oos_priors = build_pooled_prior(rounds, exclude_round=rn)

        # Build in-sample prior (include this round)
        insample_priors = build_pooled_prior(rounds, exclude_round=None)

        round_oos_wkl = []
        round_insample_wkl = []
        round_ceiling_wkl = []
        round_breakdowns = []

        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]

            # Out-of-sample prediction (what we'd submit for a new round)
            oos_pred = predict_with_prior(ig, oos_priors)

            # In-sample prediction (all data including this round)
            insample_pred = predict_with_prior(ig, insample_priors)

            # Ceiling: template predictor with fingerprinting (known round)
            tp._GRID_FINGERPRINTS[tp._grid_fingerprint(ig)] = rn
            ceiling_pred = tp.predict(ig)
            ceiling_pred = np.maximum(ceiling_pred, 0.005)
            ceiling_pred /= ceiling_pred.sum(axis=2, keepdims=True)

            # Compute scores
            kl_oos = kl_per_cell(gt, oos_pred)
            kl_insample = kl_per_cell(gt, insample_pred)
            kl_ceiling = kl_per_cell(gt, ceiling_pred)
            H = entropy_per_cell(gt)
            dynamic = H > 0.01

            if dynamic.any():
                wkl_oos = float((H * kl_oos)[dynamic].mean())
                wkl_insample = float((H * kl_insample)[dynamic].mean())
                wkl_ceiling = float((H * kl_ceiling)[dynamic].mean())
            else:
                wkl_oos = wkl_insample = wkl_ceiling = 0.0

            round_oos_wkl.append(wkl_oos)
            round_insample_wkl.append(wkl_insample)
            round_ceiling_wkl.append(wkl_ceiling)

            # Breakdown for this seed using OOS prediction
            breakdown = score_breakdown(ig, gt, oos_pred)
            round_breakdowns.append(breakdown)

        mean_oos = np.mean(round_oos_wkl)
        mean_insample = np.mean(round_insample_wkl)
        mean_ceiling = np.mean(round_ceiling_wkl)

        # Approximate score (0-100 scale, rough mapping)
        # Score = 100 * (1 - wKL / max_wKL) approximately
        # Based on R9=83.5 at wKL~0.10, leader ~92 at wKL~0.05

        print(f"  Out-of-sample wKL:  {mean_oos:.6f}")
        print(f"  In-sample wKL:      {mean_insample:.6f}")
        print(f"  Ceiling (template): {mean_ceiling:.6f}")
        print(f"  Gap (oos - ceiling): {mean_oos - mean_ceiling:.6f}")

        # Aggregate breakdown across seeds
        agg_breakdown = defaultdict(lambda: {"total_wkl": 0, "count": 0, "dynamic_count": 0})
        for bd in round_breakdowns:
            for cat, vals in bd.items():
                agg_breakdown[cat]["total_wkl"] += vals["total_wkl"]
                agg_breakdown[cat]["count"] += vals["count"]
                agg_breakdown[cat]["dynamic_count"] += vals["dynamic_count"]

        total_loss = sum(v["total_wkl"] for v in agg_breakdown.values())
        print(f"\n  Loss breakdown (% of total wKL loss):")
        sorted_cats = sorted(agg_breakdown.items(), key=lambda x: -x[1]["total_wkl"])
        for cat, vals in sorted_cats:
            if vals["total_wkl"] < 0.001:
                continue
            pct = vals["total_wkl"] / total_loss * 100 if total_loss > 0 else 0
            avg_wkl = vals["total_wkl"] / max(vals["dynamic_count"], 1)
            print(f"    {cat:20s}: {pct:5.1f}%  (total={vals['total_wkl']:.3f}, cells={vals['dynamic_count']}, avg_wkl={avg_wkl:.4f})")

        all_results[rn] = {
            "oos_wkl": float(mean_oos),
            "insample_wkl": float(mean_insample),
            "ceiling_wkl": float(mean_ceiling),
            "gap": float(mean_oos - mean_ceiling),
            "breakdown": {cat: {"pct": v["total_wkl"] / total_loss * 100 if total_loss > 0 else 0,
                               "total_wkl": v["total_wkl"],
                               "dynamic_count": v["dynamic_count"]}
                         for cat, v in sorted_cats if v["total_wkl"] > 0.001},
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS ALL ROUNDS")
    print(f"{'='*60}")
    print(f"{'Round':>6s} {'OOS wKL':>10s} {'Ceiling':>10s} {'Gap':>10s} {'Top loss category':>25s}")
    for rn in sorted(all_results.keys()):
        r = all_results[rn]
        top_cat = max(r["breakdown"].items(), key=lambda x: x[1]["pct"])[0] if r["breakdown"] else "n/a"
        print(f"  R{rn:<4d} {r['oos_wkl']:10.6f} {r['ceiling_wkl']:10.6f} {r['gap']:10.6f} {top_cat:>25s}")

    overall_oos = np.mean([r["oos_wkl"] for r in all_results.values()])
    overall_ceiling = np.mean([r["ceiling_wkl"] for r in all_results.values()])
    print(f"\n  Overall OOS mean:     {overall_oos:.6f}")
    print(f"  Overall ceiling mean: {overall_ceiling:.6f}")
    print(f"  Overall gap:          {overall_oos - overall_ceiling:.6f}")

    # Aggregate category breakdown across ALL rounds
    print(f"\n  AGGREGATE LOSS BY CATEGORY (all rounds):")
    total_by_cat = defaultdict(float)
    for rn, r in all_results.items():
        for cat, vals in r["breakdown"].items():
            total_by_cat[cat] += vals["total_wkl"]
    grand_total = sum(total_by_cat.values())
    for cat, total in sorted(total_by_cat.items(), key=lambda x: -x[1]):
        pct = total / grand_total * 100 if grand_total > 0 else 0
        print(f"    {cat:20s}: {pct:5.1f}%  (total={total:.3f})")

    # Save
    out_path = Path(__file__).parent / "benchmark_results" / "score_diagnosis.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
