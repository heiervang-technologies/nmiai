#!/usr/bin/env python3
"""Compute a simple 2D Pareto frontier and best-so-far trace from a TSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--x", required=True, help="Column name for cost/axis X")
    parser.add_argument("--x-direction", choices=["min", "max"], required=True)
    parser.add_argument("--y", required=True, help="Column name for score/axis Y")
    parser.add_argument("--y-direction", choices=["min", "max"], required=True)
    parser.add_argument("--label", default="experiment_id", help="Column to use as point label")
    parser.add_argument("--time", help="Optional column for progress ordering; defaults to file order")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if row]


def to_number(value: str) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "min" else a > b


def no_worse(a: float, b: float, direction: str) -> bool:
    return a <= b if direction == "min" else a >= b


def dominates(a: dict, b: dict, x_direction: str, y_direction: str) -> bool:
    ax = a["x"]
    ay = a["y"]
    bx = b["x"]
    by = b["y"]
    if ax is None or ay is None or bx is None or by is None:
        return False
    return no_worse(ax, bx, x_direction) and no_worse(ay, by, y_direction) and (
        better(ax, bx, x_direction) or better(ay, by, y_direction)
    )


def build_frontier(points: list[dict], x_direction: str, y_direction: str) -> list[dict]:
    frontier = []
    for point in points:
        if point["x"] is None or point["y"] is None:
            continue
        if any(dominates(other, point, x_direction, y_direction) for other in points if other is not point):
            continue
        frontier.append(point)
    frontier.sort(key=lambda p: p["x"], reverse=(x_direction == "max"))
    return frontier


def build_progress(points: list[dict], y_direction: str) -> list[dict]:
    progress = []
    best = None
    for point in points:
        if point["y"] is None:
            continue
        if best is None or better(point["y"], best["y"], y_direction):
            best = point
        progress.append(
            {
                "time": point["time"],
                "label": point["label"],
                "current_y": point["y"],
                "best_label": best["label"],
                "best_y": best["y"],
            }
        )
    return progress


def main():
    args = parse_args()
    rows = load_rows(args.input)
    points = []
    for idx, row in enumerate(rows):
        points.append(
            {
                "index": idx,
                "label": row.get(args.label) or f"row-{idx}",
                "time": row.get(args.time) if args.time else str(idx),
                "x": to_number(row.get(args.x, "")),
                "y": to_number(row.get(args.y, "")),
                "row": row,
            }
        )

    frontier = build_frontier(points, args.x_direction, args.y_direction)
    progress = build_progress(points, args.y_direction)
    payload = {
        "input": str(args.input),
        "x": {"column": args.x, "direction": args.x_direction},
        "y": {"column": args.y, "direction": args.y_direction},
        "frontier": [
            {"label": p["label"], "x": p["x"], "y": p["y"], "time": p["time"]}
            for p in frontier
        ],
        "progress": progress,
        "points_considered": sum(1 for p in points if p["x"] is not None and p["y"] is not None),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
