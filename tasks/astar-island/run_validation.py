import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path("tasks/astar-island").absolute()))
import benchmark
import regime_predictor

with open("tasks/astar-island/ground_truth/round14_seed0.json") as f:
    data = json.load(f)
ig = np.array(data["initial_grid"])
gt = np.array(data["ground_truth"])

def mock_detect(*args, **kwargs):
    return {"harsh": 0.0, "moderate": 0.0, "prosperous": 1.0}
regime_predictor.detect_regime_from_observations = mock_detect

def run_predict(floor_val, use_struct_zeros):
    model = regime_predictor.get_model()
    h, w = ig.shape
    dist_civ, n_ocean, n_civ, coast = regime_predictor.compute_features(ig)
    regime_weights = regime_predictor.detect_regime_from_observations(ig, [])
    
    pred = np.zeros((h, w, 6))
    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == 10:
                pred[y, x] = regime_predictor.OCEAN_DIST
                continue
            if code == 5:
                pred[y, x] = regime_predictor.MOUNTAIN_DIST
                continue
            
            cell_pred = np.zeros(6)
            total_w = 0.0
            for regime, rw in regime_weights.items():
                if rw < 0.01: continue
                rt = model["regime_tables"][regime]
                q, _ = regime_predictor.interpolate_lookup(
                    rt["tables"], rt["counts"], ig, dist_civ, n_ocean, n_civ, coast, y, x)
                if q is not None:
                    cell_pred += rw * q
                    total_w += rw
                else:
                    q_pool, _ = regime_predictor.interpolate_lookup(
                        model["pooled_tables"], model["pooled_counts"], ig, dist_civ, n_ocean, n_civ, coast, y, x)
                    if q_pool is not None:
                        cell_pred += rw * q_pool
                        total_w += rw
            
            if total_w > 0:
                pred[y, x] = cell_pred / total_w
            else:
                pred[y, x] = np.ones(6) / 6.0

    if use_struct_zeros:
        dist_civ_struct, n_ocean_struct, _, coast_struct = regime_predictor.compute_features(ig)
        for y in range(h):
            for x in range(w):
                code = int(ig[y, x])
                if code in (10, 5): continue
                if not coast_struct[y, x]: pred[y, x, 2] = 0.0
                pred[y, x, 5] = 0.0
                if dist_civ_struct[y, x] > 10: pred[y, x, 3] = 0.0

    pred = np.maximum(pred, floor_val)
    pred_sum = pred.sum(axis=2, keepdims=True)
    pred_sum = np.where(pred_sum == 0, 1.0, pred_sum)
    pred /= pred_sum
    
    kl = benchmark.kl_divergence(gt, pred)
    H = benchmark.entropy(gt)
    wkl = H * kl
    dynamic = H > 0.01
    return wkl[dynamic].mean() if dynamic.any() else 0

score1 = run_predict(floor_val=1e-6, use_struct_zeros=False)
score2 = run_predict(floor_val=1e-6, use_struct_zeros=True)
score3 = run_predict(floor_val=0.0, use_struct_zeros=True)

md = f"""# R14 Seed 0 Validation (Prosperous)

- Baseline (no structural zeros, FLOOR=1e-6): **{score1:.6f} wKL**
- Fix applied (structural zeros, FLOOR=1e-6): **{score2:.6f} wKL**
- Zero floor (structural zeros, FLOOR=0.0):   **{score3:.6f} wKL**
"""

with open("tasks/astar-island/gemini_validation.md", "w") as f:
    f.write(md)

print("Validation complete. Check gemini_validation.md.")
