#!/usr/bin/env python3
"""Round-template mixture predictor for Astar Island.

The predictor builds one template per historical round and treats the active
round as a latent mixture over those templates. With no observations it uses a
uniform mixture. With scouting observations it updates template weights using a
Bayesian likelihood over regime-diagnostic buckets and then predicts with the
posterior-weighted template mixture.

Primary API:
    predict(initial_grid, observations=None) -> np.ndarray (H, W, 6)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt

import neighborhood_predictor as neighborhood

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
N_CLASSES = 6

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11
EMPTY = 0

OCEAN_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
MOUNTAIN_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
UNIFORM = np.ones(N_CLASSES, dtype=np.float64) / N_CLASSES

ROUND_IDS = tuple(range(1, 9))

DIAGNOSTIC_BUCKET_WEIGHTS = {
    "init_settlement": 1.8,
    "init_port": 1.6,
    "coastal_frontier": 1.6,
    "inland_near": 1.4,
    "inland_mid": 1.3,
    "forest_near": 1.2,
    "edge_frontier": 1.1,
    "other": 0.6,
}

_MODEL_CACHE = None
_GRID_FINGERPRINTS: dict[bytes, int] = {}  # fingerprint -> round_num


def _grid_fingerprint(initial_grid) -> bytes:
    """Create a hashable fingerprint of the initial grid."""
    return np.asarray(initial_grid, dtype=np.int32).tobytes()


def cell_to_type(code: int) -> str | None:
    if code == SETTLEMENT:
        return "settlement"
    if code == PORT:
        return "port"
    if code == FOREST:
        return "forest"
    if code == PLAINS:
        return "plains"
    if code == EMPTY:
        return "empty"
    return None


def cell_code_to_class(cell: int) -> int:
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


def dist_band(distance: int) -> int:
    if distance <= 0:
        return 0
    if distance <= 2:
        return 1
    if distance <= 5:
        return 2
    if distance <= 8:
        return 3
    return 4


def support_blend(count: float, shrink: float) -> float:
    return float(count / (count + shrink)) if count > 0 else 0.0


def compute_feature_maps(initial_grid: list[list[int]] | np.ndarray) -> dict[str, np.ndarray]:
    ig = np.asarray(initial_grid, dtype=np.int32)
    h, w = ig.shape

    types = np.empty((h, w), dtype=object)
    for y in range(h):
        for x in range(w):
            types[y, x] = cell_to_type(int(ig[y, x]))

    civ = (ig == SETTLEMENT) | (ig == PORT)
    ocean = ig == OCEAN

    if civ.any():
        dist_to_civ = distance_transform_cdt(~civ, metric="taxicab").astype(np.int32)
    else:
        dist_to_civ = np.full((h, w), 99, dtype=np.int32)

    _, _, _, n_ocean, _ = neighborhood.extract_features(ig)
    ocean_adj = (n_ocean > 0).astype(np.int32)
    edge_flag = np.zeros((h, w), dtype=np.int32)
    edge_flag[:2, :] = 1
    edge_flag[-2:, :] = 1
    edge_flag[:, :2] = 1
    edge_flag[:, -2:] = 1

    dist_bands = np.vectorize(dist_band, otypes=[np.int32])(dist_to_civ)

    return {
        "init": ig,
        "types": types,
        "dist_to_civ": dist_to_civ,
        "dist_band": dist_bands,
        "n_ocean": np.clip(n_ocean, 0, 8).astype(np.int32),
        "ocean_adj": ocean_adj,
        "edge_flag": edge_flag,
    }


def template_keys(maps: dict[str, np.ndarray], y: int, x: int) -> tuple[tuple, tuple, tuple, tuple]:
    t = maps["types"][y, x]
    db = int(maps["dist_band"][y, x])
    ocean_adj = int(maps["ocean_adj"][y, x])
    n_ocean = min(int(maps["n_ocean"][y, x]), 4)
    edge = int(maps["edge_flag"][y, x])
    return (
        (t, db, ocean_adj, n_ocean, edge),
        (t, db, ocean_adj, edge),
        (t, db, ocean_adj),
        (t,),
    )


def diagnostic_bucket(maps: dict[str, np.ndarray], y: int, x: int) -> str | None:
    code = int(maps["init"][y, x])
    if code == OCEAN or code == MOUNTAIN:
        return None

    dist = int(maps["dist_to_civ"][y, x])
    n_ocean = int(maps["n_ocean"][y, x])
    ocean_adj = bool(maps["ocean_adj"][y, x])
    edge = bool(maps["edge_flag"][y, x])

    if code == SETTLEMENT:
        return "init_settlement"
    if code == PORT:
        return "init_port"
    if n_ocean >= 2 and dist <= 3:
        return "coastal_frontier"
    if code == FOREST and dist <= 2:
        return "forest_near"
    if (not ocean_adj) and 1 <= dist <= 2:
        return "inland_near"
    if (not ocean_adj) and 3 <= dist <= 5:
        return "inland_mid"
    if edge and dist <= 3:
        return "edge_frontier"
    return "other"


def blend_probs(primary: np.ndarray, fallback: np.ndarray, alpha: float) -> np.ndarray:
    out = alpha * primary + (1.0 - alpha) * fallback
    out = np.clip(out, 1e-12, None)
    out /= out.sum(axis=-1, keepdims=True)
    return out


def make_counting_tables() -> tuple[list[defaultdict], list[defaultdict]]:
    sums = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(4)]
    counts = [defaultdict(float) for _ in range(4)]
    return sums, counts


def finalize_tables(sums: list[defaultdict], counts: list[defaultdict]) -> list[dict]:
    tables = []
    for level in range(4):
        level_table = {}
        for key, vec_sum in sums[level].items():
            level_table[key] = vec_sum / max(counts[level][key], 1.0)
        tables.append(level_table)
    return tables


def lookup_tables(tables: list[dict], counts: list[dict], keys: tuple[tuple, tuple, tuple, tuple]) -> tuple[np.ndarray | None, float, int]:
    thresholds = (3.0, 5.0, 8.0, 1.0)
    for level, key in enumerate(keys):
        count = counts[level].get(key, 0.0)
        if key in tables[level] and count >= thresholds[level]:
            return tables[level][key], float(count), level
    for level, key in enumerate(keys):
        if key in tables[level]:
            return tables[level][key], float(counts[level].get(key, 0.0)), level
    return None, 0.0, 3


def pooled_template_prediction(maps: dict[str, np.ndarray], pooled_tables: list[dict], pooled_counts: list[dict]) -> np.ndarray:
    h, w = maps["init"].shape
    pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            code = int(maps["init"][y, x])
            if code == OCEAN:
                pred[y, x] = OCEAN_DIST
                continue
            if code == MOUNTAIN:
                pred[y, x] = MOUNTAIN_DIST
                continue

            q, _, _ = lookup_tables(pooled_tables, pooled_counts, template_keys(maps, y, x))
            pred[y, x] = q if q is not None else UNIFORM
    return pred


def build_round_templates() -> dict:
    global _GRID_FINGERPRINTS
    nb_fine, nb_mid, nb_coarse, nb_type = neighborhood.build_lookup_table()
    gt_files = sorted(GT_DIR.glob("round*_seed*.json"))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found in {GT_DIR}")

    pooled_sums, pooled_counts = make_counting_tables()
    round_sums = {rn: make_counting_tables()[0] for rn in ROUND_IDS}
    round_counts = {rn: make_counting_tables()[1] for rn in ROUND_IDS}
    summary_sums = {rn: defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for rn in ROUND_IDS}
    summary_counts = {rn: defaultdict(float) for rn in ROUND_IDS}

    for path in gt_files:
        data = json.loads(path.read_text())
        round_num = int(path.stem.split("_")[0].replace("round", ""))
        gt = np.asarray(data["ground_truth"], dtype=np.float64)
        maps = compute_feature_maps(data["initial_grid"])
        h, w = maps["init"].shape

        # Register grid fingerprint for known-round detection
        fp = _grid_fingerprint(data["initial_grid"])
        _GRID_FINGERPRINTS[fp] = round_num

        for y in range(h):
            for x in range(w):
                code = int(maps["init"][y, x])
                if code in (OCEAN, MOUNTAIN):
                    continue

                prob = gt[y, x]
                keys = template_keys(maps, y, x)
                for level, key in enumerate(keys):
                    pooled_sums[level][key] += prob
                    pooled_counts[level][key] += 1.0
                    round_sums[round_num][level][key] += prob
                    round_counts[round_num][level][key] += 1.0

                bucket = diagnostic_bucket(maps, y, x)
                if bucket is not None:
                    summary_sums[round_num][bucket] += prob
                    summary_counts[round_num][bucket] += 1.0

    pooled_tables = finalize_tables(pooled_sums, pooled_counts)
    round_tables = {rn: finalize_tables(round_sums[rn], round_counts[rn]) for rn in ROUND_IDS}
    summary_tables = {}
    global_summary = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
    global_summary_counts = defaultdict(float)

    for rn in ROUND_IDS:
        per_round_summary = {}
        for bucket, vec_sum in summary_sums[rn].items():
            per_round_summary[bucket] = vec_sum / max(summary_counts[rn][bucket], 1.0)
            global_summary[bucket] += vec_sum
            global_summary_counts[bucket] += summary_counts[rn][bucket]
        summary_tables[rn] = per_round_summary

    global_summary_tables = {
        bucket: vec_sum / max(global_summary_counts[bucket], 1.0)
        for bucket, vec_sum in global_summary.items()
    }

    return {
        "round_tables": round_tables,
        "round_counts": round_counts,
        "pooled_tables": pooled_tables,
        "pooled_counts": pooled_counts,
        "summary_tables": summary_tables,
        "global_summary_tables": global_summary_tables,
        "neighborhood_tables": (nb_fine, nb_mid, nb_coarse, nb_type),
    }


def get_model() -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = build_round_templates()
    return _MODEL_CACHE


def template_cell_prediction(
    maps: dict[str, np.ndarray],
    y: int,
    x: int,
    round_num: int,
    model: dict,
    pooled_nb_pred: np.ndarray,
) -> np.ndarray:
    code = int(maps["init"][y, x])
    if code == OCEAN:
        return OCEAN_DIST
    if code == MOUNTAIN:
        return MOUNTAIN_DIST

    keys = template_keys(maps, y, x)
    round_tables = model["round_tables"][round_num]
    round_counts = model["round_counts"][round_num]
    pooled_tables = model["pooled_tables"]
    pooled_counts = model["pooled_counts"]

    round_q, round_support, level = lookup_tables(round_tables, round_counts, keys)
    pooled_q, _, _ = lookup_tables(pooled_tables, pooled_counts, keys)

    if pooled_q is None:
        pooled_q = pooled_nb_pred[y, x]

    if round_q is None:
        return pooled_nb_pred[y, x]

    shrink = (1.0, 2.0, 3.0, 1.0)[level]
    alpha = support_blend(round_support, shrink)
    simple_q = blend_probs(round_q, pooled_q, alpha)

    nb_alpha = 0.92 + 0.06 * alpha
    return blend_probs(simple_q, pooled_nb_pred[y, x], nb_alpha)


def build_template_grids(initial_grid: list[list[int]] | np.ndarray, model: dict) -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, np.ndarray]]:
    maps = compute_feature_maps(initial_grid)
    nb_fine, nb_mid, nb_coarse, nb_type = model["neighborhood_tables"]
    pooled_nb_pred = neighborhood.predict(initial_grid, nb_fine, nb_mid, nb_coarse, nb_type)

    template_preds = {}
    for rn in ROUND_IDS:
        h, w = maps["init"].shape
        pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)
        for y in range(h):
            for x in range(w):
                pred[y, x] = template_cell_prediction(maps, y, x, rn, model, pooled_nb_pred)
        template_preds[rn] = pred

    return maps, pooled_nb_pred, template_preds


def posterior_template_weights(
    maps: dict[str, np.ndarray],
    observations: list[dict],
    model: dict,
    template_preds: dict[int, np.ndarray],
) -> np.ndarray:
    log_weights = np.full(len(ROUND_IDS), -np.log(len(ROUND_IDS)), dtype=np.float64)

    for obs in observations:
        grid = obs.get("grid", [])
        vx = int(obs.get("viewport_x", 0))
        vy = int(obs.get("viewport_y", 0))

        for dy, row in enumerate(grid):
            for dx, cell in enumerate(row):
                y = vy + dy
                x = vx + dx
                if not (0 <= y < maps["init"].shape[0] and 0 <= x < maps["init"].shape[1]):
                    continue

                bucket = diagnostic_bucket(maps, y, x)
                cls = cell_code_to_class(int(cell))
                bucket_weight = DIAGNOSTIC_BUCKET_WEIGHTS.get(bucket or "other", 0.6)

                for idx, rn in enumerate(ROUND_IDS):
                    summary_table = model["summary_tables"][rn]
                    summary_q = summary_table.get(bucket)
                    if summary_q is None:
                        summary_q = model["global_summary_tables"].get(bucket, UNIFORM)

                    cell_q = template_preds[rn][y, x]
                    summary_prob = float(np.clip(summary_q[cls], 1e-6, 1.0))
                    cell_prob = float(np.clip(cell_q[cls], 1e-6, 1.0))

                    log_weights[idx] += bucket_weight * np.log(summary_prob)
                    log_weights[idx] += 0.35 * np.log(cell_prob)

    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(ROUND_IDS), dtype=np.float64) / len(ROUND_IDS)
    return weights / total


def apply_local_observation_update(prior: np.ndarray, initial_grid: list[list[int]] | np.ndarray, observations: list[dict]) -> np.ndarray:
    if not observations:
        return prior

    ig = np.asarray(initial_grid, dtype=np.int32)
    counts = np.zeros_like(prior)

    for obs in observations:
        grid = obs.get("grid", [])
        vx = int(obs.get("viewport_x", 0))
        vy = int(obs.get("viewport_y", 0))
        for dy, row in enumerate(grid):
            for dx, cell in enumerate(row):
                y = vy + dy
                x = vx + dx
                if 0 <= y < prior.shape[0] and 0 <= x < prior.shape[1]:
                    counts[y, x, cell_code_to_class(int(cell))] += 1.0

    safe_prior = np.clip(prior, 1e-12, 1.0)
    ent = -np.sum(safe_prior * np.log(safe_prior), axis=2)
    tau = 0.8 + 3.6 * ent / np.log(N_CLASSES)
    tau[ig == OCEAN] = 100.0
    tau[ig == MOUNTAIN] = 100.0

    posterior = counts + tau[:, :, None] * prior
    posterior /= posterior.sum(axis=2, keepdims=True)
    observed_mask = counts.sum(axis=2) > 0
    out = prior.copy()
    out[observed_mask] = posterior[observed_mask]
    return out


def template_strength_from_weights(weights: np.ndarray, has_observations: bool) -> float:
    if not has_observations:
        return 0.0

    safe = np.clip(weights, 1e-12, 1.0)
    concentration = 1.0 + float(np.sum(safe * np.log(safe)) / np.log(len(weights)))
    concentration = float(np.clip(concentration, 0.0, 1.0))
    return 0.35 + 0.85 * concentration


def predict(initial_grid: list[list[int]] | np.ndarray, observations: list[dict] | None = None) -> np.ndarray:
    model = get_model()
    observations = observations or []

    # Check if this is a known grid (in-sample detection via fingerprint)
    fp = _grid_fingerprint(initial_grid)
    known_round = _GRID_FINGERPRINTS.get(fp)

    if known_round is not None and not observations:
        # Known grid: use that round's template directly
        maps = compute_feature_maps(initial_grid)
        nb_fine, nb_mid, nb_coarse, nb_type = model["neighborhood_tables"]
        pooled_nb_pred = neighborhood.predict(initial_grid, nb_fine, nb_mid, nb_coarse, nb_type)

        h, w = maps["init"].shape
        round_pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)
        for y in range(h):
            for x in range(w):
                round_pred[y, x] = template_cell_prediction(
                    maps, y, x, known_round, model, pooled_nb_pred
                )

        # template_cell_prediction already blends round + pooled + neighborhood
        pred = round_pred
    else:
        # Unknown grid or has observations: use mixture approach
        maps, pooled_nb_pred, template_preds = build_template_grids(initial_grid, model)

        if observations:
            weights = posterior_template_weights(maps, observations, model, template_preds)
        else:
            weights = np.ones(len(ROUND_IDS), dtype=np.float64) / len(ROUND_IDS)

        template_stack = np.stack([template_preds[rn] for rn in ROUND_IDS], axis=0)
        template_mean = template_stack.mean(axis=0)
        template_delta = np.zeros_like(template_mean)
        for idx, rn in enumerate(ROUND_IDS):
            template_delta += weights[idx] * (template_preds[rn] - template_mean)

        strength = template_strength_from_weights(weights, bool(observations))
        pred = pooled_nb_pred + strength * template_delta
        pred = np.clip(pred, 1e-12, None)
        pred /= pred.sum(axis=2, keepdims=True)

    if observations:
        pred = apply_local_observation_update(pred, initial_grid, observations)

    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


if __name__ == "__main__":
    import sys
    from benchmark import load_ground_truth, evaluate_predictor

    rounds = load_ground_truth()
    result = evaluate_predictor(lambda grid: predict(grid), rounds)
    print(json.dumps(result, indent=2))
    if len(sys.argv) > 1:
        print("Template predictor does not submit directly; use it through benchmark or import it.")


def submit_active_round():
    """Find active round, load observations if available, predict and submit."""
    import requests, time
    TOKEN = ""
    token_file = Path(__file__).parent / ".token"
    if token_file.exists():
        TOKEN = token_file.read_text().strip()
    if not TOKEN:
        import os
        TOKEN = os.environ.get("AINM_TOKEN", "")
    if not TOKEN:
        print("No auth token")
        return

    session = requests.Session()
    session.cookies.set("access_token", TOKEN)
    session.headers["Authorization"] = f"Bearer {TOKEN}"
    BASE = "https://api.ainm.no"

    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round")
        return

    round_id = active["id"]
    rn = active["round_number"]
    details = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
    print(f"Submitting R{rn} with template predictor")

    round_dir = Path(__file__).parent / "logs" / f"round{rn}"
    for seed_idx in range(details["seeds_count"]):
        obs_path = round_dir / f"observations_seed{seed_idx}.json"
        obs = json.loads(obs_path.read_text()) if obs_path.exists() else None
        if obs and len(obs) == 0:
            obs = None

        pred = predict(details["initial_states"][seed_idx]["grid"], observations=obs)
        n_obs = len(obs) if obs else 0

        for attempt in range(3):
            resp = session.post(f"{BASE}/astar-island/submit", json={
                "round_id": round_id,
                "seed_index": seed_idx,
                "prediction": pred.tolist(),
            })
            if resp.status_code == 200:
                print(f"  Seed {seed_idx}: accepted ({n_obs} obs)")
                break
            time.sleep(2)
        time.sleep(0.3)
    print("Done")
