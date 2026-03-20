#!/usr/bin/env python3
"""Train a fast prior model from accumulated ground truth and submit predictions.

Strategy:
- Load all completed-round ground truth files.
- Train two simple feature-based models:
  1. Entropy-weighted empirical Bayes bucket model.
  2. Small HistGradientBoosting regressors (one per class).
- Blend them into a calibrated prior for the active round.
- If current-round observations exist on disk, update the prior with a Dirichlet posterior.
- Submit predictions for all seeds in the active round.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import binary_dilation, binary_erosion, convolve, distance_transform_cdt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

BASE = "https://api.ainm.no"
TASK_DIR = Path(__file__).parent
LOG_DIR = TASK_DIR / "logs"
GT_DIR = TASK_DIR / "ground_truth"
MODEL_DIR = TASK_DIR / "model_cache"
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TOKEN = os.environ.get("AINM_TOKEN", "")
if not TOKEN:
    token_file = TASK_DIR / ".token"
    if token_file.exists():
        TOKEN = token_file.read_text().strip()

session = requests.Session()
if TOKEN:
    session.cookies.set("access_token", TOKEN)
    session.headers["Authorization"] = f"Bearer {TOKEN}"

CROSS_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
MANHATTAN_R2_KERNEL = np.array(
    [[0, 0, 1, 0, 0],
     [0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 0, 0]],
    dtype=np.int32,
)
MANHATTAN_R3_KERNEL = np.array(
    [[0, 0, 0, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1],
     [0, 1, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 1, 0, 0],
     [0, 0, 0, 1, 0, 0, 0]],
    dtype=np.int32,
)


def log_api_call(endpoint, method, request_data, response_data, status_code, elapsed_ms):
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "timestamp": timestamp,
        "method": method,
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
        "status_code": status_code,
        "elapsed_ms": elapsed_ms,
    }
    with open(LOG_DIR / "api_calls.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def api_get(endpoint):
    url = f"{BASE}{endpoint}"
    t0 = time.time()
    resp = session.get(url)
    elapsed = (time.time() - t0) * 1000
    data = resp.json() if resp.ok else {"error": resp.text}
    log_api_call(endpoint, "GET", None, data, resp.status_code, elapsed)
    return data


def api_post(endpoint, payload):
    url = f"{BASE}{endpoint}"
    t0 = time.time()
    resp = session.post(url, json=payload)
    elapsed = (time.time() - t0) * 1000
    data = resp.json() if resp.ok else {"error": resp.text, "status": resp.status_code}
    log_payload = payload.copy()
    if "prediction" in log_payload:
        log_payload["prediction"] = "<tensor>"
    log_api_call(endpoint, "POST", log_payload, data, resp.status_code, elapsed)
    return data, resp.status_code


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


def encode_initial_type(init):
    out = np.zeros(init.shape, dtype=np.int32)
    out[np.isin(init, [0, 11])] = 0
    out[init == 1] = 1
    out[init == 2] = 2
    out[init == 3] = 3
    out[init == 4] = 4
    out[init == 5] = 5
    out[init == 10] = 6
    return out


def extract_feature_maps(initial_grid):
    init = np.asarray(initial_grid, dtype=np.int32)
    settlement = init == 1
    port = init == 2
    civ = (init == 1) | (init == 2)
    ocean = init == 10
    forest = init == 4
    mountain = init == 5

    if civ.any():
        dist_to_civ = distance_transform_cdt(~civ, metric="taxicab").astype(np.int32)
    else:
        dist_to_civ = np.full(init.shape, 99, dtype=np.int32)

    coast = binary_dilation(ocean, np.ones((3, 3), dtype=bool)) & ~ocean
    forest_interior = binary_erosion(forest, structure=CROSS_KERNEL)
    settlement_neighbors = convolve(settlement.astype(np.int32), NEIGHBOR_KERNEL, mode="constant", cval=0)
    port_neighbors = convolve(port.astype(np.int32), NEIGHBOR_KERNEL, mode="constant", cval=0)
    civ_neighbors = convolve(civ.astype(np.int32), NEIGHBOR_KERNEL, mode="constant", cval=0)
    forest_neighbors = convolve(forest.astype(np.int32), NEIGHBOR_KERNEL, mode="constant", cval=0)
    ocean_neighbors = convolve(ocean.astype(np.int32), NEIGHBOR_KERNEL, mode="constant", cval=0)
    mountain_neighbors = convolve(mountain.astype(np.int32), NEIGHBOR_KERNEL, mode="constant", cval=0)
    local_civ_density_r2 = convolve(civ.astype(np.int32), MANHATTAN_R2_KERNEL, mode="constant", cval=0)
    local_civ_density = convolve(civ.astype(np.int32), MANHATTAN_R3_KERNEL, mode="constant", cval=0)
    mountain_adjacent = convolve(mountain.astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant", cval=0)

    if ocean.any():
        dist_to_ocean = distance_transform_cdt(~ocean, metric="taxicab").astype(np.int32)
    else:
        dist_to_ocean = np.full(init.shape, 99, dtype=np.int32)

    init_type = encode_initial_type(init)
    return {
        "init": init,
        "init_type": init_type,
        "dist_to_civ": dist_to_civ,
        "coast": coast.astype(np.int32),
        "forest_interior": forest_interior.astype(np.int32),
        "settlement_neighbors": np.clip(settlement_neighbors, 0, 8).astype(np.int32),
        "port_neighbors": np.clip(port_neighbors, 0, 8).astype(np.int32),
        "civ_neighbors": np.clip(civ_neighbors, 0, 8).astype(np.int32),
        "forest_neighbors": np.clip(forest_neighbors, 0, 8).astype(np.int32),
        "ocean_neighbors": np.clip(ocean_neighbors, 0, 8).astype(np.int32),
        "mountain_neighbors": np.clip(mountain_neighbors, 0, 8).astype(np.int32),
        "local_civ_density_r2": np.clip(local_civ_density_r2, 0, 13).astype(np.int32),
        "local_civ_density": np.clip(local_civ_density, 0, 8).astype(np.int32),
        "mountain_adjacent": np.clip(mountain_adjacent, 0, 8).astype(np.int32),
        "dist_to_ocean": np.clip(dist_to_ocean, 0, 15).astype(np.int32),
    }


def feature_matrix_from_maps(maps):
    return np.stack(
        [
            maps["init_type"],
            np.clip(maps["dist_to_civ"], 0, 15),
            maps["coast"],
            maps["forest_interior"],
            maps["local_civ_density"],
            maps["mountain_adjacent"],
            maps["dist_to_ocean"],
        ],
        axis=-1,
    )


def phase_feature_matrix_from_maps(maps):
    init_type = maps["init_type"].reshape(-1)
    init_one_hot = np.eye(7, dtype=np.float64)[init_type]

    dist_to_civ = np.clip(maps["dist_to_civ"], 0, 8).astype(np.float64)
    dist_to_ocean = np.clip(maps["dist_to_ocean"], 0, 8).astype(np.float64)
    coast = maps["coast"].astype(np.float64)
    forest_interior = maps["forest_interior"].astype(np.float64)
    settlement_neighbors = maps["settlement_neighbors"].astype(np.float64)
    port_neighbors = maps["port_neighbors"].astype(np.float64)
    civ_neighbors = maps["civ_neighbors"].astype(np.float64)
    forest_neighbors = maps["forest_neighbors"].astype(np.float64)
    ocean_neighbors = maps["ocean_neighbors"].astype(np.float64)
    mountain_neighbors = maps["mountain_neighbors"].astype(np.float64)
    local_civ_density_r2 = maps["local_civ_density_r2"].astype(np.float64)
    local_civ_density_r3 = maps["local_civ_density"].astype(np.float64)

    civ_closeness = np.maximum(0.0, 4.0 - dist_to_civ)
    ocean_closeness = np.maximum(0.0, 4.0 - dist_to_ocean)

    # Interpretable phase-inspired summary signals.
    growth_signal = 1.4 * settlement_neighbors + 0.8 * port_neighbors + 0.35 * local_civ_density_r2 - 0.4 * mountain_neighbors
    trade_signal = coast * (1.2 * settlement_neighbors + 1.0 * port_neighbors + civ_closeness)
    conflict_signal = np.maximum(0.0, settlement_neighbors - 1.0) * (1.0 + 0.15 * local_civ_density_r2)
    winter_risk = np.maximum(0.0, dist_to_civ - 1.0) + np.maximum(0.0, 2.0 - settlement_neighbors) + 0.3 * np.maximum(0.0, mountain_neighbors - 1.0)
    reclamation_signal = 1.1 * forest_neighbors + 0.8 * forest_interior + 0.35 * np.maximum(0.0, dist_to_civ - 1.0)

    raw_features = np.stack(
        [
            dist_to_civ,
            dist_to_ocean,
            coast,
            forest_interior,
            settlement_neighbors,
            port_neighbors,
            civ_neighbors,
            forest_neighbors,
            ocean_neighbors,
            mountain_neighbors,
            local_civ_density_r2,
            local_civ_density_r3,
            civ_closeness,
            ocean_closeness,
            growth_signal,
            trade_signal,
            conflict_signal,
            winter_risk,
            reclamation_signal,
        ],
        axis=-1,
    ).reshape(-1, 19)

    return np.concatenate([init_one_hot, raw_features], axis=1)


def entropy_weight(y):
    return 0.25 + (-np.sum(y * np.log(np.clip(y, 1e-12, 1.0)), axis=1))


class BucketModel:
    def __init__(self):
        self.sum_by_level = [dict(), dict(), dict()]
        self.weight_by_level = [dict(), dict(), dict()]
        self.global_mean = None

    @staticmethod
    def keys(x):
        init_type, dist_to_civ, coast, forest_interior, civ_density, _, _ = [int(v) for v in x]
        dist_bin = min(dist_to_civ, 6)
        density_bin = min(civ_density, 4)
        return [
            (init_type, dist_bin, coast, forest_interior, density_bin),
            (init_type, dist_bin, coast),
            (init_type,),
        ]

    def fit(self, X, Y, sample_weight):
        self.global_mean = np.average(Y, axis=0, weights=sample_weight)
        for x, y, w in zip(X, Y, sample_weight):
            for level, key in enumerate(self.keys(x)):
                if key not in self.sum_by_level[level]:
                    self.sum_by_level[level][key] = np.zeros(6, dtype=np.float64)
                    self.weight_by_level[level][key] = 0.0
                self.sum_by_level[level][key] += w * y
                self.weight_by_level[level][key] += w
        return self

    def predict_with_support(self, X):
        preds = np.zeros((len(X), 6), dtype=np.float64)
        supports = np.zeros(len(X), dtype=np.float64)
        for i, x in enumerate(X):
            pred = None
            support = 0.0
            thresholds = (30.0, 10.0, 1.0)
            for level, key in enumerate(self.keys(x)):
                weight = self.weight_by_level[level].get(key, 0.0)
                if weight >= thresholds[level]:
                    pred = self.sum_by_level[level][key] / weight
                    support = weight
                    break
            if pred is None:
                pred = self.global_mean
            preds[i] = pred
            supports[i] = support
        return preds, supports


class PhaseRuleModel:
    """Interpretable linear rule model over phase-inspired spatial features."""

    def __init__(self, alpha=1.5):
        self.alpha = alpha
        self.regressors = []

    def fit(self, X, Y):
        weights = entropy_weight(Y)
        self.regressors = []
        for cls in range(6):
            reg = Ridge(alpha=self.alpha)
            reg.fit(X, Y[:, cls], sample_weight=weights)
            self.regressors.append(reg)
        return self

    def predict_prior(self, X):
        pred = np.zeros((len(X), 6), dtype=np.float64)
        for cls, reg in enumerate(self.regressors):
            pred[:, cls] = reg.predict(X)
        pred = np.clip(pred, 0.0, None)
        pred_sum = pred.sum(axis=1, keepdims=True)
        return np.divide(
            pred,
            pred_sum,
            out=np.full_like(pred, 1.0 / 6.0),
            where=pred_sum > 0,
        )


def dsl_rule_feature_matrix_from_maps(maps):
    init_type = maps["init_type"]

    init_masks = [
        ("init=plains", init_type == 0),
        ("init=settlement", init_type == 1),
        ("init=port", init_type == 2),
        ("init=forest", init_type == 4),
    ]
    structural_masks = [
        ("coast", maps["coast"] == 1),
        ("forest_interior", maps["forest_interior"] == 1),
        ("dist_civ<=1", maps["dist_to_civ"] <= 1),
        ("dist_civ<=2", maps["dist_to_civ"] <= 2),
        ("dist_civ<=4", maps["dist_to_civ"] <= 4),
        ("dist_ocean<=1", maps["dist_to_ocean"] <= 1),
        ("dist_ocean<=2", maps["dist_to_ocean"] <= 2),
        ("settle_n>=1", maps["settlement_neighbors"] >= 1),
        ("settle_n>=2", maps["settlement_neighbors"] >= 2),
        ("port_n>=1", maps["port_neighbors"] >= 1),
        ("civ_n>=1", maps["civ_neighbors"] >= 1),
        ("civ_n>=2", maps["civ_neighbors"] >= 2),
        ("forest_n>=4", maps["forest_neighbors"] >= 4),
        ("ocean_n>=1", maps["ocean_neighbors"] >= 1),
        ("ocean_n>=3", maps["ocean_neighbors"] >= 3),
        ("mountain_n>=1", maps["mountain_neighbors"] >= 1),
        ("density_r2>=2", maps["local_civ_density_r2"] >= 2),
        ("density_r2>=4", maps["local_civ_density_r2"] >= 4),
    ]

    names = []
    cols = []

    def add_rule(name, mask):
        names.append(name)
        cols.append(mask.reshape(-1).astype(bool))

    for name, mask in init_masks:
        add_rule(name, mask)
    for name, mask in structural_masks:
        add_rule(name, mask)
    for init_name, init_mask in init_masks:
        for struct_name, struct_mask in structural_masks:
            add_rule(f"{init_name}&{struct_name}", init_mask & struct_mask)

    matrix = np.stack(cols, axis=1)
    return matrix, names


class DSLRuleModel:
    """Greedy ARC-style rule refinement over a compact local predicate DSL."""

    def __init__(self, max_rules=12, min_support=180.0, min_gain=5e-5):
        self.max_rules = max_rules
        self.min_support = min_support
        self.min_gain = min_gain
        self.rules = []
        self.rule_names = []

    def fit(self, X_rules, Y, base_pred, rule_names):
        self.rule_names = list(rule_names)
        self.rules = []

        weights = entropy_weight(Y)
        total_weight = float(weights.sum())
        current_pred = np.clip(base_pred.copy(), 1e-9, 1.0)
        log_y = np.log(np.clip(Y, 1e-12, 1.0))
        available = np.ones(X_rules.shape[1], dtype=bool)

        for _ in range(self.max_rules):
            log_current = np.log(np.clip(current_pred, 1e-12, 1.0))
            best = None

            for rule_idx in np.flatnonzero(available):
                mask = X_rules[:, rule_idx]
                if not np.any(mask):
                    continue

                rule_weights = weights[mask]
                support = float(rule_weights.sum())
                if support < self.min_support:
                    continue

                delta = np.average(log_y[mask] - log_current[mask], axis=0, weights=rule_weights)
                strength = min(0.85, support / (support + 250.0))

                corrected = current_pred[mask] * np.exp(strength * delta)
                corrected /= corrected.sum(axis=1, keepdims=True)

                base_loss = np.sum(
                    rule_weights
                    * np.sum(Y[mask] * (log_y[mask] - log_current[mask]), axis=1)
                )
                corrected_loss = np.sum(
                    rule_weights
                    * np.sum(Y[mask] * (log_y[mask] - np.log(np.clip(corrected, 1e-12, 1.0))), axis=1)
                )
                gain = float((base_loss - corrected_loss) / total_weight)

                if best is None or gain > best["gain"]:
                    best = {
                        "index": int(rule_idx),
                        "mask": mask,
                        "delta": delta,
                        "strength": float(strength),
                        "support": support,
                        "gain": gain,
                        "corrected": corrected,
                    }

            if best is None or best["gain"] <= self.min_gain:
                break

            available[best["index"]] = False
            current_pred[best["mask"]] = best["corrected"]
            self.rules.append(
                {
                    "index": best["index"],
                    "name": self.rule_names[best["index"]],
                    "delta": best["delta"],
                    "strength": best["strength"],
                    "support": best["support"],
                    "gain": best["gain"],
                }
            )

        return self

    def predict_prior(self, X_rules, base_pred):
        pred = np.clip(base_pred.copy(), 1e-9, 1.0)
        for rule in self.rules:
            mask = X_rules[:, rule["index"]]
            if not np.any(mask):
                continue
            corrected = pred[mask] * np.exp(rule["strength"] * rule["delta"])
            corrected /= corrected.sum(axis=1, keepdims=True)
            pred[mask] = corrected
        return pred

    def describe_rules(self, limit=None):
        rows = []
        for rule in self.rules[:limit]:
            rows.append(
                f"{rule['name']} support={rule['support']:.1f} gain={rule['gain']:.6f} strength={rule['strength']:.3f}"
            )
        return rows


class HybridPriorModel:
    def __init__(self, use_phase_model=True, use_dsl_model=True):
        self.use_phase_model = use_phase_model
        self.use_dsl_model = use_dsl_model
        self.bucket = BucketModel()
        self.regressors = []
        self.phase_model = PhaseRuleModel() if use_phase_model else None
        self.dsl_model = DSLRuleModel() if use_dsl_model else None

    def fit(self, X_basic, X_phase, X_dsl, Y, rule_names):
        weights = entropy_weight(Y)
        self.bucket.fit(X_basic, Y, weights)
        self.regressors = []
        for cls in range(6):
            reg = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.08,
                max_iter=120,
                min_samples_leaf=40,
                random_state=cls,
            )
            reg.fit(X_basic, Y[:, cls], sample_weight=weights)
            self.regressors.append(reg)
        if self.use_phase_model:
            self.phase_model.fit(X_phase, Y)
        if self.use_dsl_model:
            base_pred = self._predict_without_dsl(X_basic, X_phase)
            self.dsl_model.fit(X_dsl, Y, base_pred, rule_names)
        return self

    def _predict_without_dsl(self, X_basic, X_phase):
        bucket_pred, bucket_support = self.bucket.predict_with_support(X_basic)
        gbr_pred = np.zeros((len(X_basic), 6), dtype=np.float64)
        for cls, reg in enumerate(self.regressors):
            gbr_pred[:, cls] = reg.predict(X_basic)
        gbr_pred = np.clip(gbr_pred, 0.0, None)
        gbr_sum = gbr_pred.sum(axis=1, keepdims=True)
        gbr_pred = np.divide(
            gbr_pred,
            gbr_sum,
            out=np.full_like(gbr_pred, 1.0 / 6.0),
            where=gbr_sum > 0,
        )

        # Keep the tree model as a light correction only.
        # In practice the bucket prior is more stable on tiny round counts.
        blend = np.full((len(X_basic), 1), 0.15, dtype=np.float64)
        blend[bucket_support < 30.0] = 0.0
        pred = blend * gbr_pred + (1.0 - blend) * bucket_pred

        if self.use_phase_model:
            phase_pred = self.phase_model.predict_prior(X_phase)
            phase_blend = np.full((len(X_basic), 1), 0.25, dtype=np.float64)
            phase_blend[bucket_support < 30.0] = 0.45
            static_mask = np.isin(X_basic[:, 0], [5, 6])
            phase_blend[static_mask] = 0.0
            pred = (1.0 - phase_blend) * pred + phase_blend * phase_pred
        pred = np.clip(pred, 0.0, None)
        pred_sum = pred.sum(axis=1, keepdims=True)
        return np.divide(
            pred,
            pred_sum,
            out=np.full_like(pred, 1.0 / 6.0),
            where=pred_sum > 0,
        )

    def predict_prior(self, X_basic, X_phase, X_dsl):
        pred = self._predict_without_dsl(X_basic, X_phase)
        if self.use_dsl_model:
            pred = self.dsl_model.predict_prior(X_dsl, pred)
        pred = np.clip(pred, 0.0, None)
        pred_sum = pred.sum(axis=1, keepdims=True)
        return np.divide(
            pred,
            pred_sum,
            out=np.full_like(pred, 1.0 / 6.0),
            where=pred_sum > 0,
        )


MODEL_CONFIGS = {
    "base": {"use_phase_model": False, "use_dsl_model": False},
    "phase": {"use_phase_model": True, "use_dsl_model": False},
    "dsl": {"use_phase_model": True, "use_dsl_model": True},
}
MODEL_COMPLEXITY_ORDER = ["base", "phase", "dsl"]


def build_model(config_name):
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model config: {config_name}")
    return HybridPriorModel(**MODEL_CONFIGS[config_name])


def load_training_data(paths=None):
    gt_files = sorted(paths if paths is not None else GT_DIR.glob("round*_seed*.json"))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found in {GT_DIR}")

    X_basic_parts = []
    X_phase_parts = []
    X_dsl_parts = []
    Y_parts = []
    rule_names = None
    for path in gt_files:
        data = json.loads(path.read_text())
        feature_maps = extract_feature_maps(data["initial_grid"])
        X_basic_parts.append(feature_matrix_from_maps(feature_maps).reshape(-1, 7))
        X_phase_parts.append(phase_feature_matrix_from_maps(feature_maps))
        X_dsl, current_rule_names = dsl_rule_feature_matrix_from_maps(feature_maps)
        X_dsl_parts.append(X_dsl)
        if rule_names is None:
            rule_names = current_rule_names
        Y_parts.append(np.asarray(data["ground_truth"], dtype=np.float64).reshape(-1, 6))
    return (
        np.concatenate(X_basic_parts, axis=0),
        np.concatenate(X_phase_parts, axis=0),
        np.concatenate(X_dsl_parts, axis=0),
        np.concatenate(Y_parts, axis=0),
        gt_files,
        rule_names,
    )


def tau_from_prior(prior, init_type):
    if init_type == 6 or init_type == 5:
        return 100.0
    h = -np.sum(prior * np.log(np.clip(prior, 1e-12, 1.0)))
    return 1.0 + 4.0 * h / np.log(6.0)


def load_round_observations(round_number, seed_idx):
    path = LOG_DIR / f"round{round_number}" / f"observations_seed{seed_idx}.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def build_prediction(initial_grid, model, observations):
    maps = extract_feature_maps(initial_grid)
    X_basic = feature_matrix_from_maps(maps).reshape(-1, 7)
    X_phase = phase_feature_matrix_from_maps(maps)
    X_dsl, _ = dsl_rule_feature_matrix_from_maps(maps)
    prior = model.predict_prior(X_basic, X_phase, X_dsl).reshape(maps["init"].shape + (6,))

    # Static per-seed overrides.
    ocean = maps["init"] == 10
    mountain = maps["init"] == 5
    prior[ocean] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    prior[mountain] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    counts = np.zeros_like(prior)
    obs_count = np.zeros(prior.shape[:2], dtype=np.int32)
    for obs in observations:
        grid = obs["grid"]
        vx = obs["viewport_x"]
        vy = obs["viewport_y"]
        for dy, row in enumerate(grid):
            for dx, cell in enumerate(row):
                y = vy + dy
                x = vx + dx
                if 0 <= y < counts.shape[0] and 0 <= x < counts.shape[1]:
                    counts[y, x, cell_code_to_class(cell)] += 1.0
                    obs_count[y, x] += 1

    init_type = maps["init_type"]
    tau = np.zeros(init_type.shape, dtype=np.float64)
    flat_prior = prior.reshape(-1, 6)
    flat_types = init_type.reshape(-1)
    tau[:] = np.array([tau_from_prior(p, t) for p, t in zip(flat_prior, flat_types)]).reshape(init_type.shape)

    posterior = counts + tau[:, :, None] * prior
    posterior_pred = posterior / posterior.sum(axis=2, keepdims=True)

    # Treat observations as soft evidence. Even one sample is informative, but
    # repeated samples should dominate quickly on high-variance frontier cells.
    obs_strength = 1.0 - np.exp(-obs_count.astype(np.float64) / 1.8)
    pred = (1.0 - obs_strength[:, :, None]) * prior + obs_strength[:, :, None] * posterior_pred

    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def find_active_round():
    rounds = api_get("/astar-island/rounds")
    for r in rounds:
        if r.get("status") == "active":
            return r
    return None


def submit_prediction(round_id, seed_index, prediction):
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction.tolist(),
    }
    return api_post("/astar-island/submit", payload)


def round_number_from_path(path):
    match = re.search(r"round(\d+)_seed\d+\.json$", path.name)
    if not match:
        raise ValueError(f"Could not parse round number from {path}")
    return int(match.group(1))


def weighted_kl_divergence(p, q):
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=2)
    weights = 0.25 + entropy
    kl = np.sum(p * np.log(np.clip(p, 1e-12, 1.0) / np.clip(q, 1e-12, 1.0)), axis=2)
    return float(np.sum(weights * kl) / np.sum(weights))


def evaluate_leave_one_round_out(model_configs=("base", "phase", "dsl"), verbose=True):
    gt_files = sorted(GT_DIR.glob("round*_seed*.json"))
    rounds = sorted({round_number_from_path(path) for path in gt_files})
    if len(rounds) < 2:
        raise ValueError("Need ground truth from at least 2 rounds for leave-one-round-out evaluation")

    summary = {name: [] for name in model_configs}

    for holdout_round in rounds:
        train_paths = [path for path in gt_files if round_number_from_path(path) != holdout_round]
        test_paths = [path for path in gt_files if round_number_from_path(path) == holdout_round]

        X_train_basic, X_train_phase, X_train_dsl, Y_train, _, rule_names = load_training_data(train_paths)
        trained_models = {
            name: build_model(name).fit(X_train_basic, X_train_phase, X_train_dsl, Y_train, rule_names)
            for name in model_configs
        }
        round_scores = {name: [] for name in model_configs}
        for path in test_paths:
            data = json.loads(path.read_text())
            gt = np.asarray(data["ground_truth"], dtype=np.float64)
            for name, model in trained_models.items():
                pred = build_prediction(data["initial_grid"], model, observations=[])
                round_scores[name].append(weighted_kl_divergence(gt, pred))

        round_means = {name: float(np.mean(scores)) for name, scores in round_scores.items()}
        for name, value in round_means.items():
            summary[name].append(value)

        if verbose:
            base_mean = round_means.get("base")
            parts = [f"Round {holdout_round}:"]
            for name in model_configs:
                part = f"{name}={round_means[name]:.6f}"
                if base_mean is not None and name != "base":
                    part += f" delta={round_means[name] - base_mean:+.6f}"
                parts.append(part)
            print(" ".join(parts))

    overall = {name: float(np.mean(values)) for name, values in summary.items()}
    if verbose:
        parts = ["Overall:"]
        base_mean = overall.get("base")
        for name in model_configs:
            part = f"{name}={overall[name]:.6f}"
            if base_mean is not None and name != "base":
                part += f" delta={overall[name] - base_mean:+.6f}"
            parts.append(part)
        print(" ".join(parts))
    return summary, overall


def select_best_model_config():
    _, overall = evaluate_leave_one_round_out(verbose=False)
    best_name = min(
        overall,
        key=lambda name: (overall[name], MODEL_COMPLEXITY_ORDER.index(name) if name in MODEL_COMPLEXITY_ORDER else 99),
    )
    return best_name, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-submit", action="store_true", help="Build predictions but do not submit.")
    parser.add_argument("--eval-loro", action="store_true", help="Run leave-one-round-out offline evaluation and exit.")
    parser.add_argument(
        "--model-config",
        choices=["auto", *MODEL_CONFIGS.keys()],
        default="base",
        help="Model family to train. Default is the validated base model; 'auto' reruns leave-one-round-out selection.",
    )
    args = parser.parse_args()

    if args.eval_loro:
        evaluate_leave_one_round_out()
        return

    if not TOKEN:
        print("ERROR: No auth token. Set AINM_TOKEN or create tasks/astar-island/.token")
        sys.exit(1)

    active = find_active_round()
    if not active:
        print("No active round found.")
        sys.exit(0)

    round_id = active["id"]
    round_number = active["round_number"]
    print(f"Active round: {round_number} ({round_id})")
    print(f"Closes at: {active['closes_at']}")
    print(f"Weight: {active['round_weight']}")

    details = api_get(f"/astar-island/rounds/{round_id}")
    with open(LOG_DIR / f"round{round_number}_details.json", "w") as f:
        json.dump(details, f)

    budget = api_get("/astar-island/budget")
    print(f"Budget: {budget}")

    X_train_basic, X_train_phase, X_train_dsl, Y_train, gt_files, rule_names = load_training_data()
    print(f"Training on {len(gt_files)} ground-truth files, {len(X_train_basic)} labeled cells")
    model_config = args.model_config
    if model_config == "auto":
        model_config, overall = select_best_model_config()
        ranking = sorted(overall.items(), key=lambda item: item[1])
        print("Validated model ranking:")
        for name, score in ranking:
            print(f"  {name}: {score:.6f}")
    else:
        print(f"Using requested model config: {model_config}")

    model = build_model(model_config).fit(X_train_basic, X_train_phase, X_train_dsl, Y_train, rule_names)
    print(f"Selected model config: {model_config}")
    if model.use_dsl_model and model.dsl_model.rules:
        print("Top DSL rules:")
        for row in model.dsl_model.describe_rules(limit=8):
            print(f"  {row}")

    height = details["map_height"]
    width = details["map_width"]
    initial_states = details["initial_states"]

    for seed_idx, seed_state in enumerate(initial_states):
        initial_grid = seed_state["grid"]
        observations = load_round_observations(round_number, seed_idx)
        print(f"Seed {seed_idx}: {len(observations)} observations")

        prediction = build_prediction(initial_grid, model, observations)
        np.save(LOG_DIR / f"predictor_round{round_number}_seed{seed_idx}.npy", prediction)

        if args.no_submit:
            continue

        result, status = submit_prediction(round_id, seed_idx, prediction)
        print(f"  submit status={status} result={result}")
        time.sleep(0.25)

    print("Done.")


if __name__ == "__main__":
    main()
