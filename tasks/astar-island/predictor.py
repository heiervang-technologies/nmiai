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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import binary_dilation, binary_erosion, convolve, distance_transform_cdt
from sklearn.ensemble import HistGradientBoostingRegressor

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


class HybridPriorModel:
    def __init__(self):
        self.bucket = BucketModel()
        self.regressors = []

    def fit(self, X, Y):
        weights = entropy_weight(Y)
        self.bucket.fit(X, Y, weights)
        self.regressors = []
        for cls in range(6):
            reg = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.08,
                max_iter=120,
                min_samples_leaf=40,
                random_state=cls,
            )
            reg.fit(X, Y[:, cls], sample_weight=weights)
            self.regressors.append(reg)
        return self

    def predict_prior(self, X):
        bucket_pred, bucket_support = self.bucket.predict_with_support(X)
        gbr_pred = np.zeros((len(X), 6), dtype=np.float64)
        for cls, reg in enumerate(self.regressors):
            gbr_pred[:, cls] = reg.predict(X)
        gbr_pred = np.clip(gbr_pred, 0.0, None)
        gbr_sum = gbr_pred.sum(axis=1, keepdims=True)
        gbr_pred = np.divide(
            gbr_pred,
            gbr_sum,
            out=np.full_like(gbr_pred, 1.0 / 6.0),
            where=gbr_sum > 0,
        )

        # Bucket model is more stable with low data; trees capture interactions.
        blend = np.full((len(X), 1), 0.55, dtype=np.float64)
        blend[bucket_support < 30.0] = 0.40
        pred = blend * gbr_pred + (1.0 - blend) * bucket_pred
        pred = np.clip(pred, 0.0, None)
        pred_sum = pred.sum(axis=1, keepdims=True)
        pred = np.divide(
            pred,
            pred_sum,
            out=np.full_like(pred, 1.0 / 6.0),
            where=pred_sum > 0,
        )
        return pred


def load_training_data():
    gt_files = sorted(GT_DIR.glob("round*_seed*.json"))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found in {GT_DIR}")

    X_parts = []
    Y_parts = []
    for path in gt_files:
        data = json.loads(path.read_text())
        feature_maps = extract_feature_maps(data["initial_grid"])
        X_parts.append(feature_matrix_from_maps(feature_maps).reshape(-1, 7))
        Y_parts.append(np.asarray(data["ground_truth"], dtype=np.float64).reshape(-1, 6))
    return np.concatenate(X_parts, axis=0), np.concatenate(Y_parts, axis=0), gt_files


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
    X = feature_matrix_from_maps(maps).reshape(-1, 7)
    prior = model.predict_prior(X).reshape(maps["init"].shape + (6,))

    # Static per-seed overrides.
    ocean = maps["init"] == 10
    mountain = maps["init"] == 5
    prior[ocean] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    prior[mountain] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    counts = np.zeros_like(prior)
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

    init_type = maps["init_type"]
    tau = np.zeros(init_type.shape, dtype=np.float64)
    flat_prior = prior.reshape(-1, 6)
    flat_types = init_type.reshape(-1)
    tau[:] = np.array([tau_from_prior(p, t) for p, t in zip(flat_prior, flat_types)]).reshape(init_type.shape)

    posterior = counts + tau[:, :, None] * prior
    pred = posterior / posterior.sum(axis=2, keepdims=True)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-submit", action="store_true", help="Build predictions but do not submit.")
    args = parser.parse_args()

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

    X_train, Y_train, gt_files = load_training_data()
    print(f"Training on {len(gt_files)} ground-truth files, {len(X_train)} labeled cells")
    model = HybridPriorModel().fit(X_train, Y_train)

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
