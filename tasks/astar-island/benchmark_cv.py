#!/usr/bin/env python3
"""Proper leave-one-round-out cross-validation.

Retrains models from scratch on 7 rounds, evaluates on held-out round.
This gives the true out-of-sample score we'd expect on competition rounds.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import load_ground_truth, kl_divergence, entropy


def cv_spatial_model():
    """Leave-one-round-out CV for spatial_model."""
    import spatial_model as sm

    rounds = load_ground_truth()
    round_nums = sorted(rounds.keys())
    results = []

    for held_out in round_nums:
        # Train on everything except held_out
        train_data = []
        for rn in round_nums:
            if rn == held_out:
                continue
            for sn, sd in rounds[rn].items():
                train_data.append({
                    'initial_grid': sd['initial_grid'],
                    'ground_truth': sd['ground_truth'],
                })

        # Fit model on training data only
        models = sm.fit_model(train_data)

        # Evaluate on held-out round
        round_kls = []
        round_wkls = []
        for sn, sd in rounds[held_out].items():
            pred = sm.predict_with_models(sd['initial_grid'], models)
            gt = sd['ground_truth']

            kl = kl_divergence(gt, pred)
            H = entropy(gt)
            wkl = H * kl
            dynamic = H > 0.01

            if dynamic.any():
                round_kls.append(float(kl[dynamic].mean()))
                round_wkls.append(float(wkl[dynamic].mean()))

        mean_kl = np.mean(round_kls)
        mean_wkl = np.mean(round_wkls)
        results.append({
            'held_out': held_out,
            'mean_kl': float(mean_kl),
            'mean_wkl': float(mean_wkl),
            'n_seeds': len(round_kls),
        })
        print(f"  R{held_out} held out: KL={mean_kl:.6f} wKL={mean_wkl:.6f}")

    overall_kl = np.mean([r['mean_kl'] for r in results])
    overall_wkl = np.mean([r['mean_wkl'] for r in results])
    print(f"\n  CV Mean KL:  {overall_kl:.6f}")
    print(f"  CV Mean wKL: {overall_wkl:.6f}")
    return {'cv_kl': float(overall_kl), 'cv_wkl': float(overall_wkl), 'per_round': results}


def cv_neighborhood():
    """Leave-one-round-out CV for neighborhood_predictor.

    Rebuilds lookup tables from 7 rounds only.
    """
    import neighborhood_predictor as nb

    rounds = load_ground_truth()
    round_nums = sorted(rounds.keys())
    results = []

    for held_out in round_nums:
        # Build lookup tables from training rounds only
        train_files = []
        gt_dir = Path(__file__).parent / "ground_truth"
        for f in sorted(gt_dir.glob("round*_seed*.json")):
            rn = int(f.stem.split("_")[0].replace("round", ""))
            if rn != held_out:
                train_files.append(f)

        # Rebuild tables using only training files
        fine, mid, coarse, type_t = nb._build_lookup_from_files(train_files) if hasattr(nb, '_build_lookup_from_files') else (None, None, None, None)

        if fine is None:
            # Fallback: use the module's internal build but monkey-patch GT_DIR
            # This is imperfect but better than nothing
            print(f"  R{held_out}: no _build_lookup_from_files, using full tables (CONTAMINATED)")
            fine_t, mid_t, coarse_t, type_t = nb.build_lookup_table()
            for sn, sd in rounds[held_out].items():
                pred = nb.predict(sd['initial_grid'], fine_t, mid_t, coarse_t, type_t)
                gt = sd['ground_truth']
                kl = kl_divergence(gt, pred)
                H = entropy(gt)
                wkl = H * kl
                dynamic = H > 0.01
                if dynamic.any():
                    results.append({
                        'held_out': held_out, 'seed': sn,
                        'kl': float(kl[dynamic].mean()),
                        'wkl': float(wkl[dynamic].mean()),
                    })
            continue

        # Evaluate on held-out
        round_kls = []
        round_wkls = []
        for sn, sd in rounds[held_out].items():
            pred = nb.predict(sd['initial_grid'], fine, mid, coarse, type_t)
            gt = sd['ground_truth']
            kl = kl_divergence(gt, pred)
            H = entropy(gt)
            wkl = H * kl
            dynamic = H > 0.01
            if dynamic.any():
                round_kls.append(float(kl[dynamic].mean()))
                round_wkls.append(float(wkl[dynamic].mean()))

        mean_kl = np.mean(round_kls)
        mean_wkl = np.mean(round_wkls)
        results.append({
            'held_out': held_out,
            'mean_kl': float(mean_kl),
            'mean_wkl': float(mean_wkl),
        })
        print(f"  R{held_out} held out: KL={mean_kl:.6f} wKL={mean_wkl:.6f}")

    if results and 'mean_kl' in results[0]:
        overall_kl = np.mean([r['mean_kl'] for r in results])
        overall_wkl = np.mean([r['mean_wkl'] for r in results])
        print(f"\n  CV Mean KL:  {overall_kl:.6f}")
        print(f"  CV Mean wKL: {overall_wkl:.6f}")
        return {'cv_kl': float(overall_kl), 'cv_wkl': float(overall_wkl), 'per_round': results}
    return {'error': 'no _build_lookup_from_files'}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("spatial", "all"):
        print("=== SPATIAL MODEL - Leave-One-Round-Out CV ===")
        t0 = time.time()
        spatial_results = cv_spatial_model()
        print(f"  Time: {time.time() - t0:.1f}s")

    if target in ("neighborhood", "all"):
        print("\n=== NEIGHBORHOOD - Leave-One-Round-Out CV ===")
        t0 = time.time()
        nb_results = cv_neighborhood()
        print(f"  Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
