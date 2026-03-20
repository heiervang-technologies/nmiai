#!/usr/bin/env python3
"""Entropy-calibrated neighborhood predictor for Astar Island.

Builds the same neighborhood buckets as ``neighborhood_predictor.py`` and then
fits per-bucket temperature corrections so the predicted entropy better matches
the observed ground-truth entropy of cells assigned to each effective bucket.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import neighborhood_predictor as base

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
N_CLASSES = base.N_CLASSES
LOG2_CLASSES = np.log2(N_CLASSES)

_MODEL_CACHE = None


def entropy_bits(prob):
    safe = np.clip(prob, 1e-12, 1.0)
    return float(-np.sum(safe * np.log2(safe)))


def temperature_scale(prob, temperature):
    safe = np.clip(prob, 1e-12, 1.0)
    logits = np.log(safe) / temperature
    logits -= logits.max()
    out = np.exp(logits)
    out /= out.sum()
    return out


def fit_temperature(prob, target_entropy):
    current = entropy_bits(prob)
    target = float(np.clip(target_entropy, 0.0, LOG2_CLASSES))
    if abs(current - target) < 1e-4:
        return 1.0

    if target > current:
        lo, hi = 1.0, 2.0
        while entropy_bits(temperature_scale(prob, hi)) < target and hi < 64.0:
            lo = hi
            hi *= 2.0
        hi = min(hi, 64.0)
    else:
        lo, hi = 0.2, 1.0
        while entropy_bits(temperature_scale(prob, lo)) > target and lo > 0.02:
            hi = lo
            lo *= 0.5
        lo = max(lo, 0.02)

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        mid_entropy = entropy_bits(temperature_scale(prob, mid))
        if mid_entropy < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_lookup_tables():
    fine_data = defaultdict(list)
    mid_data = defaultdict(list)
    coarse_data = defaultdict(list)
    type_data = defaultdict(list)

    for path in sorted(GT_DIR.glob("round*_seed*.json")):
        with open(path) as f:
            data = json.load(f)

        ig = np.array(data["initial_grid"])
        gt = np.array(data["ground_truth"], dtype=np.float64)
        types, n_settle, n_forest, n_ocean, dist_bins = base.extract_features(ig)
        h, w = ig.shape

        for y in range(h):
            for x in range(w):
                t = types[y, x]
                if t is None:
                    continue

                prob = gt[y, x]
                ns = min(int(n_settle[y, x]), 4)
                nf = min(int(n_forest[y, x]), 4)
                no = min(int(n_ocean[y, x]), 3)
                db = int(dist_bins[y, x])

                fine_data[(t, ns, nf, no, db)].append(prob)
                ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
                mid_data[(t, ns_bin, 1 if no > 0 else 0, db)].append(prob)
                coarse_data[(t, db)].append(prob)
                type_data[(t,)].append(prob)

    fine_table = {k: np.mean(v, axis=0) for k, v in fine_data.items() if len(v) >= 10}
    mid_table = {k: np.mean(v, axis=0) for k, v in mid_data.items() if len(v) >= 10}
    coarse_table = {k: np.mean(v, axis=0) for k, v in coarse_data.items()}
    type_table = {k: np.mean(v, axis=0) for k, v in type_data.items()}
    return fine_table, mid_table, coarse_table, type_table


def assign_bucket(t, ns, nf, no, db, fine_table, mid_table, coarse_table, type_table):
    fine_key = (t, ns, nf, no, db)
    if fine_key in fine_table:
        return ("fine", fine_key), fine_table[fine_key]

    ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
    mid_key = (t, ns_bin, 1 if no > 0 else 0, db)
    if mid_key in mid_table:
        return ("mid", mid_key), mid_table[mid_key]

    coarse_key = (t, db)
    if coarse_key in coarse_table:
        return ("coarse", coarse_key), coarse_table[coarse_key]

    type_key = (t,)
    return ("type", type_key), type_table[type_key]


def shrink_temperature(raw_temperature, support, shrink):
    alpha = support / (support + shrink)
    return float(np.exp(alpha * np.log(raw_temperature)))


def fit_temperature_tables(fine_table, mid_table, coarse_table, type_table):
    bucket_stats = defaultdict(lambda: {"entropy_weighted_sum": 0.0, "weight_sum": 0.0, "count": 0})
    dist_stats = defaultdict(lambda: {"entropy_weighted_sum": 0.0, "weight_sum": 0.0, "count": 0})
    type_stats = defaultdict(lambda: {"entropy_weighted_sum": 0.0, "weight_sum": 0.0, "count": 0})
    level_stats = defaultdict(lambda: {"entropy_weighted_sum": 0.0, "weight_sum": 0.0, "count": 0})

    for path in sorted(GT_DIR.glob("round*_seed*.json")):
        with open(path) as f:
            data = json.load(f)

        ig = np.array(data["initial_grid"])
        gt = np.array(data["ground_truth"], dtype=np.float64)
        types, n_settle, n_forest, n_ocean, dist_bins = base.extract_features(ig)
        h, w = ig.shape

        for y in range(h):
            for x in range(w):
                t = types[y, x]
                if t is None:
                    continue

                ns = min(int(n_settle[y, x]), 4)
                nf = min(int(n_forest[y, x]), 4)
                no = min(int(n_ocean[y, x]), 3)
                db = int(dist_bins[y, x])

                bucket_id, _ = assign_bucket(t, ns, nf, no, db, fine_table, mid_table, coarse_table, type_table)
                level = bucket_id[0]
                h_bits = entropy_bits(gt[y, x])
                weight = max(h_bits, 1e-3)

                for store, key in (
                    (bucket_stats, bucket_id),
                    (dist_stats, (level, t, db)),
                    (type_stats, (level, t)),
                    (level_stats, (level,)),
                ):
                    store[key]["entropy_weighted_sum"] += weight * h_bits
                    store[key]["weight_sum"] += weight
                    store[key]["count"] += 1

    def reference_distribution(bucket_id):
        level = bucket_id[0]
        key = bucket_id[1]
        if level == "fine":
            return fine_table[key]
        if level == "mid":
            return mid_table[key]
        if level == "coarse":
            return coarse_table[key]
        return type_table[key]

    def build_temperature_map(stats, shrink, min_count):
        out = {}
        for bucket_id, values in stats.items():
            count = values["count"]
            if count < min_count:
                continue

            if bucket_id in bucket_stats:
                prob = reference_distribution(bucket_id)
            elif bucket_id in dist_stats:
                level, cell_type, dist_bin = bucket_id
                if level in ("fine", "mid"):
                    candidates = []
                    if level == "fine":
                        candidates = [v for (t, _ns, _nf, _no, db), v in fine_table.items() if t == cell_type and db == dist_bin]
                    else:
                        candidates = [v for (t, _ns_bin, _coast, db), v in mid_table.items() if t == cell_type and db == dist_bin]
                    prob = np.mean(candidates, axis=0) if candidates else coarse_table[(cell_type, dist_bin)]
                elif level == "coarse":
                    prob = coarse_table[(cell_type, dist_bin)]
                else:
                    prob = type_table[(cell_type,)]
            elif bucket_id in type_stats:
                level, cell_type = bucket_id
                if level == "fine":
                    candidates = [v for (t, _ns, _nf, _no, _db), v in fine_table.items() if t == cell_type]
                    prob = np.mean(candidates, axis=0) if candidates else type_table[(cell_type,)]
                elif level == "mid":
                    candidates = [v for (t, _ns_bin, _coast, _db), v in mid_table.items() if t == cell_type]
                    prob = np.mean(candidates, axis=0) if candidates else type_table[(cell_type,)]
                elif level == "coarse":
                    candidates = [v for (t, _db), v in coarse_table.items() if t == cell_type]
                    prob = np.mean(candidates, axis=0) if candidates else type_table[(cell_type,)]
                else:
                    prob = type_table[(cell_type,)]
            else:
                level = bucket_id[0]
                if level == "fine":
                    candidates = list(fine_table.values())
                elif level == "mid":
                    candidates = list(mid_table.values())
                elif level == "coarse":
                    candidates = list(coarse_table.values())
                else:
                    candidates = list(type_table.values())
                prob = np.mean(candidates, axis=0)

            target_entropy = values["entropy_weighted_sum"] / values["weight_sum"]
            raw_temperature = fit_temperature(prob, target_entropy)
            out[bucket_id] = shrink_temperature(raw_temperature, count, shrink)
        return out

    return {
        "bucket": build_temperature_map(bucket_stats, shrink=30.0, min_count=6),
        "dist": build_temperature_map(dist_stats, shrink=60.0, min_count=15),
        "type": build_temperature_map(type_stats, shrink=90.0, min_count=30),
        "level": build_temperature_map(level_stats, shrink=150.0, min_count=1),
    }


def choose_temperature(bucket_id, bucket_temperatures):
    level = bucket_id[0]
    key = bucket_id[1]
    if bucket_id in bucket_temperatures["bucket"]:
        return bucket_temperatures["bucket"][bucket_id]

    if level in ("fine", "mid"):
        dist_key = (level, key[0], key[-1])
    elif level == "coarse":
        dist_key = (level, key[0], key[1])
    else:
        dist_key = None

    if dist_key is not None and dist_key in bucket_temperatures["dist"]:
        return bucket_temperatures["dist"][dist_key]

    type_key = (level, key[0])
    if type_key in bucket_temperatures["type"]:
        return bucket_temperatures["type"][type_key]

    return bucket_temperatures["level"].get((level,), 1.0)


def build_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    fine_table, mid_table, coarse_table, type_table = build_lookup_tables()
    bucket_temperatures = fit_temperature_tables(fine_table, mid_table, coarse_table, type_table)

    _MODEL_CACHE = {
        "fine_table": fine_table,
        "mid_table": mid_table,
        "coarse_table": coarse_table,
        "type_table": type_table,
        "bucket_temperatures": bucket_temperatures,
    }
    return _MODEL_CACHE


def predict(initial_grid):
    model = build_model()
    fine_table = model["fine_table"]
    mid_table = model["mid_table"]
    coarse_table = model["coarse_table"]
    type_table = model["type_table"]
    bucket_temperatures = model["bucket_temperatures"]

    ig = np.array(initial_grid)
    h, w = ig.shape
    types, n_settle, n_forest, n_ocean, dist_bins = base.extract_features(ig)
    pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code == base.OCEAN:
                pred[y, x] = base.OCEAN_DIST
                continue
            if code == base.MOUNTAIN:
                pred[y, x] = base.MOUNTAIN_DIST
                continue

            t = types[y, x]
            ns = min(int(n_settle[y, x]), 4)
            nf = min(int(n_forest[y, x]), 4)
            no = min(int(n_ocean[y, x]), 3)
            db = int(dist_bins[y, x])

            bucket_id, base_prob = assign_bucket(t, ns, nf, no, db, fine_table, mid_table, coarse_table, type_table)
            temperature = choose_temperature(bucket_id, bucket_temperatures)
            pred[y, x] = temperature_scale(base_prob, temperature)

    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred
