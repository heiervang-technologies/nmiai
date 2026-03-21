#!/usr/bin/env python3
"""Parametric decay predictor for Astar Island.

Fits smooth continuous exponential decay functions per (regime, cell_type, coast):
    P(class | d) = A * exp(-B * d) + C

where d is Euclidean distance to nearest initial civ. This eliminates
bucket boundary artifacts from the regime_predictor's integer-distance bucketing.

Key groups:
- 3 regimes x 4 cell types x 2 coast states x 6 classes = up to 144 curves
- Each curve is A*exp(-B*d)+C, 3 parameters
- Plus zero-distance (initial settlement/port) priors per group
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve
from scipy.optimize import curve_fit

GT_DIR = Path(__file__).parent / "ground_truth"
N_CLASSES = 6
FLOOR = 1e-6

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4

KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

HARSH_THRESHOLD = 0.05
PROSPEROUS_THRESHOLD = 0.20

DEFAULT_REGIME_WEIGHTS = {"harsh": 0.23, "moderate": 0.54, "prosperous": 0.23}

OCEAN_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
MOUNTAIN_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

_MODEL = None


def exp_decay(d, a, b):
    return a * np.exp(-b * d)


def compute_features(ig):
    h, w = ig.shape
    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN
    if civ.any():
        dist_civ = distance_transform_edt(~civ)
    else:
        dist_civ = np.full((h, w), 99.0)
    n_ocean = convolve(ocean.astype(np.int32), KERNEL, mode="constant")
    coast = (n_ocean > 0) & ~ocean
    return dist_civ, n_ocean, coast


def classify_round(round_data):
    frontier_rates = []
    for sn, sd in round_data.items():
        ig = sd["initial_grid"]
        gt = sd["ground_truth"]
        dist_civ, _, _ = compute_features(ig)
        frontier = (dist_civ >= 1) & (dist_civ <= 5) & (ig != OCEAN) & (ig != MOUNTAIN)
        if frontier.any():
            frontier_rates.append(gt[frontier, 1].mean())
    f_rate = np.mean(frontier_rates) if frontier_rates else 0.1
    if f_rate < HARSH_THRESHOLD:
        return "harsh"
    elif f_rate > PROSPEROUS_THRESHOLD:
        return "prosperous"
    return "moderate"


def _cell_type(code):
    if code == SETTLEMENT: return "S"
    if code == PORT: return "P"
    if code == FOREST: return "F"
    return "L"


def _fit_class_curve(bd, y_data, weights):
    """Fit A*exp(-B*d)+C. Returns (A, B, C)."""
    if y_data.max() < 1e-6:
        return (0.0, 1.0, float(y_data.mean()))

    far = bd >= max(8, bd.max() - 2)
    C_est = float(y_data[far].mean()) if far.sum() > 0 else float(y_data[-1])
    y_s = y_data - C_est

    if np.abs(y_s).max() < 1e-6:
        return (0.0, 1.0, C_est)

    try:
        if y_s[0] > 0:
            bounds = ([0, 0.01], [2.0, 10.0])
        else:
            bounds = ([-2.0, 0.01], [0, 10.0])
        popt, _ = curve_fit(
            exp_decay, bd, y_s,
            p0=[float(y_s[0]), 0.3],
            bounds=bounds,
            sigma=1.0 / weights,
            maxfev=5000,
        )
        return (float(popt[0]), float(popt[1]), C_est)
    except Exception:
        return (0.0, 1.0, float(y_data.mean()))


def _fit_group(items, min_points=10):
    """Fit curves from a list of (distance, prob_vec) tuples.

    Uses 0.5-unit bins for finer resolution than integer bins.
    Returns {cls: (A, B, C)} or None.
    """
    if len(items) < min_points:
        return None

    distances = np.array([p[0] for p in items])
    probs = np.array([p[1] for p in items])

    max_d = min(distances.max(), 20)
    # Use 0.5-unit bins for finer resolution
    bd_list, bp_list, bn_list = [], [], []
    d_val = 0.75
    while d_val <= max_d:
        mask = (distances >= d_val - 0.25) & (distances < d_val + 0.25)
        if mask.sum() >= 2:
            bd_list.append(d_val)
            bp_list.append(probs[mask].mean(axis=0))
            bn_list.append(mask.sum())
        d_val += 0.5

    # Also try integer bins if too few half-bins
    if len(bd_list) < 4:
        bd_list, bp_list, bn_list = [], [], []
        for di in range(1, int(max_d) + 1):
            mask = (distances >= di - 0.5) & (distances < di + 0.5)
            if mask.sum() >= 3:
                bd_list.append(di)
                bp_list.append(probs[mask].mean(axis=0))
                bn_list.append(mask.sum())

    if len(bd_list) < 3:
        return None

    bd = np.array(bd_list, dtype=np.float64)
    bp = np.array(bp_list)
    bn = np.sqrt(np.array(bn_list, dtype=np.float64))

    cls_params = {}
    for cls in range(N_CLASSES):
        cls_params[cls] = _fit_class_curve(bd, bp[:, cls], bn)
    return cls_params


def build_model_from_data(rounds):
    """Build parametric model."""
    for rn, seeds in rounds.items():
        for sn, sd in seeds.items():
            if not isinstance(sd["initial_grid"], np.ndarray):
                sd["initial_grid"] = np.array(sd["initial_grid"], dtype=np.int32)
            if not isinstance(sd["ground_truth"], np.ndarray):
                sd["ground_truth"] = np.array(sd["ground_truth"], dtype=np.float64)

    round_regimes = {rn: classify_round(rd) for rn, rd in rounds.items()}

    # Collect data at multiple granularities:
    # Fine:   (regime, cell_type, coast, n_ocean_clamped)
    # Medium: (regime, cell_type, coast)
    # Coarse: (regime, cell_type)
    group_fine = defaultdict(list)
    group_medium = defaultdict(list)
    group_coarse = defaultdict(list)
    zero_fine = defaultdict(list)
    zero_medium = defaultdict(list)
    zero_coarse = defaultdict(list)

    for rn, rd in rounds.items():
        regime = round_regimes[rn]
        for sn, sd in rd.items():
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]
            dist_civ, n_ocean, coast_map = compute_features(ig)
            h, w = ig.shape

            for y in range(h):
                for x in range(w):
                    code = int(ig[y, x])
                    if code == OCEAN or code == MOUNTAIN:
                        continue
                    d = float(dist_civ[y, x])
                    ct = _cell_type(code)
                    c = bool(coast_map[y, x])
                    no = min(int(n_ocean[y, x]), 3)
                    prob = gt[y, x]

                    if d < 0.5:
                        zero_fine[(regime, ct, c, no)].append(prob)
                        zero_medium[(regime, ct, c)].append(prob)
                        zero_coarse[(regime, ct)].append(prob)
                    else:
                        entry = (d, prob)
                        group_fine[(regime, ct, c, no)].append(entry)
                        group_medium[(regime, ct, c)].append(entry)
                        group_coarse[(regime, ct)].append(entry)

    # Fit curves at each level
    curves_fine = {}
    for key, items in group_fine.items():
        curves_fine[key] = _fit_group(items, min_points=20)

    curves_medium = {}
    for key, items in group_medium.items():
        curves_medium[key] = _fit_group(items)

    curves_coarse = {}
    for key, items in group_coarse.items():
        curves_coarse[key] = _fit_group(items)

    # Zero priors
    zp_fine = {k: np.mean(v, axis=0) for k, v in zero_fine.items() if v}
    zp_medium = {k: np.mean(v, axis=0) for k, v in zero_medium.items() if v}
    zp_coarse = {k: np.mean(v, axis=0) for k, v in zero_coarse.items() if v}

    return {
        "curves_fine": curves_fine,
        "curves_medium": curves_medium,
        "curves_coarse": curves_coarse,
        "zp_fine": zp_fine,
        "zp_medium": zp_medium,
        "zp_coarse": zp_coarse,
        "round_regimes": round_regimes,
    }


def _eval_curve(params, d):
    """Evaluate A*exp(-B*d)+C."""
    A, B, C = params
    return A * math.exp(-B * d) + C


def _predict_cell(d, regime, ct, coast, n_ocean, model):
    """Get prediction for a single cell using hierarchical parametric curves."""
    cf = model["curves_fine"]
    cm = model["curves_medium"]
    cc = model["curves_coarse"]
    no = min(int(n_ocean), 3)

    if d < 0.5:
        # Zero-distance cell - hierarchical prior lookup
        zf = model["zp_fine"]
        zm = model["zp_medium"]
        zc = model["zp_coarse"]
        key_f = (regime, ct, coast, no)
        key_m = (regime, ct, coast)
        key_c = (regime, ct)
        if key_f in zf:
            return zf[key_f].copy()
        if key_m in zm:
            return zm[key_m].copy()
        if key_c in zc:
            return zc[key_c].copy()
        return np.ones(N_CLASSES) / N_CLASSES

    # Hierarchical curve lookup: fine -> medium -> coarse
    # Fine: (regime, ct, coast, n_ocean) - best if enough data
    # Medium: (regime, ct, coast)
    # Coarse: (regime, ct)
    key_fine = (regime, ct, coast, no)
    key_med = (regime, ct, coast)
    key_coarse = (regime, ct)

    cp = cf.get(key_fine) or cm.get(key_med) or cc.get(key_coarse)
    if cp is None:
        return np.ones(N_CLASSES) / N_CLASSES

    pred = np.zeros(N_CLASSES)
    for cls in range(N_CLASSES):
        if cls in cp:
            pred[cls] = _eval_curve(cp[cls], d)

    # Structural constraints
    pred[5] = 0.0  # Mountain impossible
    if not coast:
        pred[2] = 0.0  # Port impossible if not coastal
    if d > 10:
        pred[3] = 0.0  # Ruin negligible

    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum()
    return pred


def build_model():
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(f.stem.split("_")[0].replace("round", ""))
        sn = int(f.stem.split("_")[1].replace("seed", ""))
        with open(f) as fh:
            data = json.load(fh)
        if "ground_truth" not in data or "initial_grid" not in data:
            continue
        rounds.setdefault(rn, {})[sn] = {
            "initial_grid": np.array(data["initial_grid"], dtype=np.int32),
            "ground_truth": np.array(data["ground_truth"], dtype=np.float64),
        }
    return build_model_from_data(rounds)


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build_model()
    return _MODEL


def predict_with_model(initial_grid, model, regime="moderate"):
    return _predict_impl(initial_grid, model, regime=regime)


def predict(initial_grid, regime="moderate"):
    model = get_model()
    return _predict_impl(initial_grid, model, regime=regime)


def _predict_impl(initial_grid, model, regime="moderate"):
    ig = np.array(initial_grid, dtype=np.int32)
    h, w = ig.shape
    dist_civ, n_ocean, coast_map = compute_features(ig)

    if regime == "mixture":
        regime_weights = DEFAULT_REGIME_WEIGHTS
    else:
        regime_weights = {regime: 1.0}

    pred = np.zeros((h, w, N_CLASSES))

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == OCEAN:
                pred[y, x] = OCEAN_DIST
                continue
            if code == MOUNTAIN:
                pred[y, x] = MOUNTAIN_DIST
                continue

            d = float(dist_civ[y, x])
            ct = _cell_type(code)
            coast = bool(coast_map[y, x])

            no = int(n_ocean[y, x])

            cell_pred = np.zeros(N_CLASSES)
            total_w = 0.0
            for reg, rw in regime_weights.items():
                if rw < 0.01:
                    continue
                p = _predict_cell(d, reg, ct, coast, no, model)
                cell_pred += rw * p
                total_w += rw

            if total_w > 0:
                pred[y, x] = cell_pred / total_w
            else:
                pred[y, x] = np.ones(N_CLASSES) / N_CLASSES

    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark import load_ground_truth, evaluate_predictor, cross_validate

    print("=== PARAMETRIC DECAY PREDICTOR ===\n")
    rounds = load_ground_truth()

    # In-sample
    for regime in ("harsh", "moderate", "prosperous", "mixture"):
        result = evaluate_predictor(lambda ig, r=regime: predict(ig, regime=r), rounds)
        print(f"[{regime:11s}] wKL={result['mean_weighted_kl']:.6f}  KL={result['mean_kl']:.6f}")

    # CV
    print("\n=== LEAVE-ONE-ROUND-OUT CV ===\n")
    def make_predictor(train_rounds):
        model = build_model_from_data(train_rounds)
        def pred_fn(ig):
            return predict_with_model(ig, model, regime="mixture")
        return pred_fn

    cv = cross_validate(make_predictor, rounds)
    print(f"CV mean wKL: {cv['cv_mean_weighted_kl']:.6f}")
    print(f"CV mean KL:  {cv['cv_mean_kl']:.6f}")
    print()
    for r in cv["per_round"]:
        rn = r["held_out_round"]
        print(f"  Hold out R{rn}: wKL={r['mean_weighted_kl']:.6f}  KL={r['mean_kl']:.6f}")
