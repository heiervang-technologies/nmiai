#!/usr/bin/env python3
"""Distance-decay spatial influence model for Astar Island.

Uses continuous Euclidean distance features with smooth parametric decay
functions, fitted per initial cell type using scipy.optimize.

Architecture per dynamic cell:
- Compute features: dist_settle, dist_ocean, dist_forest, coast, neighbor counts
- For each initial type, predict log-odds for each class using
  linear combinations of distance-based features
- Softmax to get probabilities
- Fit all parameters by minimizing entropy-weighted KL divergence

Classes: 0=ocean, 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain
Initial codes: 1=settlement, 2=port, 4=forest, 5=mountain, 10=ocean, 11=land
Static: ocean(10) -> [1,0,0,0,0,0], mountain(5) -> [0,0,0,0,0,1]
"""

import json
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_dilation, uniform_filter
from scipy.optimize import minimize

GT_DIR = Path(__file__).parent / "ground_truth"
PARAMS_FILE = Path(__file__).parent / "spatial_params.json"
N_CLASSES = 6
FLOOR = 0.01


def extract_features(initial_grid):
    """Extract per-cell features.

    Returns (features_dict, ig) where features_dict contains 40x40 arrays.
    """
    ig = np.array(initial_grid, dtype=int)
    H, W = ig.shape

    has_settle = (ig == 1).any()
    has_ocean = (ig == 10).any()
    has_forest = (ig == 4).any()
    has_civ = ((ig == 1) | (ig == 2)).any()

    dist_settle = distance_transform_edt(ig != 1) if has_settle else np.full((H, W), 40.0)
    dist_ocean = distance_transform_edt(ig != 10) if has_ocean else np.full((H, W), 40.0)
    dist_forest = distance_transform_edt(ig != 4) if has_forest else np.full((H, W), 40.0)
    dist_civ = distance_transform_edt(~((ig == 1) | (ig == 2))) if has_civ else np.full((H, W), 40.0)

    # Coast flag
    if has_ocean:
        coast = (binary_dilation(ig == 10) & (ig != 10)).astype(float)
    else:
        coast = np.zeros((H, W))

    # Neighbor counts
    def count_neighbors(mask):
        m = mask.astype(np.int32)
        p = np.pad(m, 1, mode='constant')
        r = np.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                r += p[1+dy:H+1+dy, 1+dx:W+1+dx]
        return r.astype(float)

    n_settle = count_neighbors((ig == 1) | (ig == 2))
    n_forest = count_neighbors(ig == 4)
    n_ocean = count_neighbors(ig == 10)

    # Distance to mountain
    has_mtn = (ig == 5).any()
    dist_mtn = distance_transform_edt(ig != 5) if has_mtn else np.full((H, W), 40.0)

    # Densities at multiple scales
    settle_mask = ((ig == 1) | (ig == 2)).astype(float)
    forest_mask_f = (ig == 4).astype(float)
    ocean_mask_f = (ig == 10).astype(float)

    settle_density_5 = uniform_filter(settle_mask, size=5, mode='constant')
    settle_density_9 = uniform_filter(settle_mask, size=9, mode='constant')
    settle_density_15 = uniform_filter(settle_mask, size=15, mode='constant')
    forest_density_5 = uniform_filter(forest_mask_f, size=5, mode='constant')
    forest_density_9 = uniform_filter(forest_mask_f, size=9, mode='constant')
    ocean_density_5 = uniform_filter(ocean_mask_f, size=5, mode='constant')

    # Mountain neighbors
    n_mtn = count_neighbors(ig == 5)

    # Distance to edge of map
    rows_arr = np.arange(H)[:, np.newaxis] * np.ones((1, W))
    cols_arr = np.ones((H, 1)) * np.arange(W)[np.newaxis, :]
    dist_edge = np.minimum(np.minimum(rows_arr, H-1-rows_arr),
                           np.minimum(cols_arr, W-1-cols_arr))

    return {
        'ds': dist_settle,
        'do': dist_ocean,
        'df': dist_forest,
        'dc': dist_civ,
        'dm': dist_mtn,
        'coast': coast,
        'ns': n_settle,
        'nf': n_forest,
        'no': n_ocean,
        'nm': n_mtn,
        'sd5': settle_density_5,
        'sd9': settle_density_9,
        'sd15': settle_density_15,
        'fd5': forest_density_5,
        'fd9': forest_density_9,
        'od5': ocean_density_5,
        'de': dist_edge,
    }, ig


def make_feature_matrix(feat_dict, ig, type_code):
    """Create feature matrix for cells of given type.

    Returns (X, rows, cols) where X is (N, n_features).
    """
    mask = ig == type_code
    if not mask.any():
        return None, None, None

    rows, cols = np.where(mask)
    N = len(rows)

    ds = feat_dict['ds'][rows, cols]
    do_ = feat_dict['do'][rows, cols]
    df = feat_dict['df'][rows, cols]
    dc = feat_dict['dc'][rows, cols]
    dm = feat_dict['dm'][rows, cols]
    coast = feat_dict['coast'][rows, cols]
    ns = feat_dict['ns'][rows, cols]
    nf = feat_dict['nf'][rows, cols]
    no_ = feat_dict['no'][rows, cols]
    nm = feat_dict['nm'][rows, cols]
    sd5 = feat_dict['sd5'][rows, cols]
    sd9 = feat_dict['sd9'][rows, cols]
    sd15 = feat_dict['sd15'][rows, cols]
    fd5 = feat_dict['fd5'][rows, cols]
    fd9 = feat_dict['fd9'][rows, cols]
    od5 = feat_dict['od5'][rows, cols]
    de = feat_dict['de'][rows, cols]

    # Precompute transforms
    exp_ds_02 = np.exp(-ds * 0.2)
    exp_ds_05 = np.exp(-ds * 0.5)
    exp_ds_10 = np.exp(-ds * 1.0)
    exp_do_02 = np.exp(-do_ * 0.2)
    exp_do_05 = np.exp(-do_ * 0.5)
    exp_df_03 = np.exp(-df * 0.3)
    exp_df_08 = np.exp(-df * 0.8)
    exp_dm_02 = np.exp(-dm * 0.2)
    log_ds = np.log1p(ds)
    log_do = np.log1p(do_)
    log_df = np.log1p(df)
    log_dm = np.log1p(dm)
    inv_ds = 1.0 / (1.0 + ds)
    inv_do = 1.0 / (1.0 + do_)
    inv_df = 1.0 / (1.0 + df)

    X = np.column_stack([
        np.ones(N),                      # 0: bias
        # Distance decays for settlements
        exp_ds_02, exp_ds_05, exp_ds_10, # 1-3
        log_ds, inv_ds,                  # 4-5
        # Distance to ocean
        exp_do_02, exp_do_05,            # 6-7
        log_do, inv_do,                  # 8-9
        # Distance to forest
        exp_df_03, exp_df_08,            # 10-11
        log_df, inv_df,                  # 12-13
        # Distance to mountain
        exp_dm_02, log_dm,               # 14-15
        # Distance to edge
        de / 20.0,                       # 16: normalized
        # Local features
        coast,                           # 17
        ns, nf, no_, nm,                 # 18-21
        # Multi-scale densities
        sd5, sd9, sd15,                  # 22-24
        fd5, fd9,                        # 25-26
        od5,                             # 27
        # Interactions: coast
        coast * exp_ds_02,               # 28
        coast * inv_ds,                  # 29
        coast * ns,                      # 30
        # Interactions: density * distance
        sd9 * exp_ds_05,                 # 31
        fd9 * exp_df_03,                 # 32
        # Interactions: neighbors * distance
        ns * exp_ds_05,                  # 33
        nf * exp_df_03,                  # 34
        no_ * exp_do_05,                 # 35
        # Quadratic terms
        log_ds ** 2,                     # 36
        log_do ** 2,                     # 37
        log_df ** 2,                     # 38
        # Distance cross-terms
        log_ds * log_do,                 # 39
        log_ds * log_df,                 # 40
        log_do * log_df,                 # 41
        # Higher order
        inv_ds * inv_do,                 # 42
        exp_ds_02 * exp_df_03,           # 43: settle-forest proximity interaction
        sd9 * fd9,                       # 44: settlement-forest density interaction
    ])

    return X, rows, cols


def predict_logodds(X, weights):
    """Compute log-odds for each class.

    weights: (n_features, n_classes-1) - predict 5 classes, 6th is reference
    """
    logits = X @ weights  # (N, 5)
    # Add reference class (ocean = class 0)
    logits_full = np.column_stack([np.zeros(logits.shape[0]), logits])
    # Softmax
    logits_full -= logits_full.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_full)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs


def fit_model(data_list):
    """Fit per-type multinomial logistic regression models.

    Returns dict of {type_code: weights_matrix}.
    """
    type_codes = [1, 2, 4, 11]
    n_feat = 45
    n_classes_minus1 = 5  # predict relative to class 0

    models = {}

    for type_code in type_codes:
        # Collect all training data for this type
        all_X = []
        all_Y = []

        for d in data_list:
            ig = np.array(d['initial_grid'], dtype=int)
            gt = np.array(d['ground_truth'])
            feat_dict, ig_ = extract_features(ig)

            X, rows, cols = make_feature_matrix(feat_dict, ig_, type_code)
            if X is None:
                continue

            Y = gt[rows, cols]  # (N, 6) probabilities
            all_X.append(X)
            all_Y.append(Y)

        if not all_X:
            continue

        X_all = np.concatenate(all_X, axis=0)
        Y_all = np.concatenate(all_Y, axis=0)

        print(f"Type {type_code}: {X_all.shape[0]} samples, {X_all.shape[1]} features")

        # Weight by ground truth entropy in bits (matches scoring metric: H * KL)
        gt_entropy_bits = -np.sum(np.maximum(Y_all, 1e-10) * np.log2(np.maximum(Y_all, 1e-10)), axis=1)
        # Use entropy as weight, with small floor for near-deterministic cells
        sample_weights = np.maximum(gt_entropy_bits, 0.01)

        # Minimal regularization - overfitting is manageable with our sample sizes
        n_samples = X_all.shape[0]
        reg = 0.0001 if n_samples < 100 else 0.0
        print(f"  Using reg={reg:.4f} for {n_samples} samples")

        w0 = np.zeros((n_feat, n_classes_minus1))

        def loss(w_flat):
            W = w_flat.reshape(n_feat, n_classes_minus1)
            probs = predict_logodds(X_all, W)
            probs = np.maximum(probs, 1e-10)

            # Cross-entropy: -sum(Y * log(pred)), weighted
            ce = -np.sum(Y_all * np.log(probs), axis=1)
            return np.mean(ce * sample_weights)

        def grad(w_flat):
            W = w_flat.reshape(n_feat, n_classes_minus1)
            probs = predict_logodds(X_all, W)
            probs = np.maximum(probs, 1e-10)

            # Gradient of cross-entropy with softmax
            diff = probs[:, 1:] - Y_all[:, 1:]
            weighted_diff = diff * sample_weights[:, np.newaxis]

            grad_W = X_all.T @ weighted_diff / len(X_all)
            grad_W += reg * W

            return grad_W.ravel()

        def loss_with_reg(w_flat):
            W = w_flat.reshape(n_feat, n_classes_minus1)
            base = loss(w_flat)
            return base + 0.5 * reg * np.sum(W**2)

        result = minimize(
            loss_with_reg, w0.ravel(),
            jac=grad,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12}
        )

        W_fitted = result.x.reshape(n_feat, n_classes_minus1)
        models[type_code] = W_fitted

        # Evaluate
        probs = predict_logodds(X_all, W_fitted)
        probs = np.maximum(probs, FLOOR)
        probs /= probs.sum(axis=1, keepdims=True)
        ce = -np.sum(Y_all * np.log(np.maximum(probs, 1e-10)), axis=1)
        print(f"  Final CE: {np.mean(ce):.6f}, weighted CE: {np.mean(ce * sample_weights):.6f}")

    return models


def predict_with_models(initial_grid, models):
    """Predict using fitted models."""
    feat_dict, ig = extract_features(initial_grid)
    H, W = ig.shape
    pred = np.zeros((H, W, N_CLASSES))

    # Static
    pred[ig == 10, 0] = 1.0
    pred[ig == 5, 5] = 1.0

    for type_code, W_mat in models.items():
        X, rows, cols = make_feature_matrix(feat_dict, ig, type_code)
        if X is None:
            continue

        probs = predict_logodds(X, W_mat)
        pred[rows, cols] = probs

    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


# ── Data loading ────────────────────────────────────────────────────────────

def load_ground_truth():
    data = []
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        with open(f) as fh:
            d = json.load(fh)
        if 'ground_truth' in d and 'initial_grid' in d:
            data.append({
                'initial_grid': d['initial_grid'],
                'ground_truth': np.array(d['ground_truth']),
            })
    return data


# ── Cached model ────────────────────────────────────────────────────────────

_models = None

def _get_models():
    global _models
    if _models is None:
        # Try loading cached
        if PARAMS_FILE.exists():
            with open(PARAMS_FILE) as f:
                saved = json.load(f)
            _models = {int(k): np.array(v) for k, v in saved.items()}
        else:
            data = load_ground_truth()
            _models = fit_model(data)
            # Save
            to_save = {str(k): v.tolist() for k, v in _models.items()}
            with open(PARAMS_FILE, 'w') as f:
                json.dump(to_save, f)
    return _models


def predict(initial_grid):
    """Predict P(class | initial_grid) for 40x40 grid."""
    models = _get_models()
    return predict_with_models(initial_grid, models)


if __name__ == "__main__":
    import sys

    # Delete cached params to force refit
    if PARAMS_FILE.exists():
        PARAMS_FILE.unlink()

    data = load_ground_truth()
    models = fit_model(data)

    # Save
    to_save = {str(k): v.tolist() for k, v in models.items()}
    with open(PARAMS_FILE, 'w') as f:
        json.dump(to_save, f)

    # In-sample evaluation
    from benchmark import kl_divergence, entropy
    all_kl = []
    all_wkl = []
    for d in data:
        pred = predict_with_models(d['initial_grid'], models)
        gt = d['ground_truth']
        kl_vals = kl_divergence(gt, pred)
        H = entropy(gt)
        dynamic = H > 0.01
        wkl = H * kl_vals
        if dynamic.any():
            all_kl.append(kl_vals[dynamic].mean())
            all_wkl.append(wkl[dynamic].mean())
    print(f"\nIn-sample KL: {np.mean(all_kl):.6f}")
    print(f"In-sample wKL: {np.mean(all_wkl):.6f}")
