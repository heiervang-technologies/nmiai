#!/usr/bin/env python3
"""Quick Pareto logger for MarkusNet runtime variants.

Runs submission-markusnet/run_fast.py on a small image set with different LM settings
and logs params + runtime + output count to CSV.
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_FAST = ROOT / "run_fast.py"
DEFAULT_OUTDIR = ROOT / "pareto_runs"


def count_active_layers(mask: str, total_layers: int = 12) -> int:
    if mask == "all":
        return total_layers
    parts = [x.strip() for x in mask.split(",") if x.strip()]
    if len(parts) != total_layers:
        raise ValueError(f"invalid mask length ({len(parts)}), expected {total_layers}")
    if any(x not in {"0", "1"} for x in parts):
        raise ValueError("mask must contain only 0/1")
    return sum(x == "1" for x in parts)


def estimate_params(active_layers: int, total_layers: int = 12) -> int:
    # Crude but stable proxy for comparison in sweeps.
    # Adjust with measured counts later if needed.
    total_params = 351_000_000
    lm_fraction = 0.55
    lm_params = int(total_params * lm_fraction)
    non_lm_params = total_params - lm_params
    kept_lm = int(lm_params * (active_layers / total_layers))
    return non_lm_params + kept_lm


def run_variant(input_dir: Path, output_json: Path, image_only: bool, no_chat: bool, mask: str) -> float:
    cmd = [
        "python3",
        str(RUN_FAST),
        "--input",
        str(input_dir),
        "--output",
        str(output_json),
        "--lm-layers-mask",
        mask,
    ]
    if image_only:
        cmd.append("--image-only")
    if no_chat:
        cmd.append("--no-chat-template")

    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory with a small eval subset of images")
    ap.add_argument("--variants", default="baseline,image_only,mask_linear_only")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--csv", default="pareto_log.csv")
    args = ap.parse_args()

    input_dir = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / args.csv

    variant_defs = {
        "baseline": dict(image_only=False, no_chat=False, mask="all"),
        "image_only": dict(image_only=True, no_chat=True, mask="all"),
        "mask_linear_only": dict(image_only=False, no_chat=False, mask="1,1,1,0,1,1,1,0,1,1,1,0"),
        "mask_top4": dict(image_only=False, no_chat=False, mask="0,0,0,0,0,0,0,0,1,1,1,1"),
    }

    selected = [x.strip() for x in args.variants.split(",") if x.strip()]
    rows = []
    for name in selected:
        if name not in variant_defs:
            raise ValueError(f"unknown variant: {name}")
        cfg = variant_defs[name]
        output_json = outdir / f"{name}.json"

        elapsed = run_variant(
            input_dir=input_dir,
            output_json=output_json,
            image_only=cfg["image_only"],
            no_chat=cfg["no_chat"],
            mask=cfg["mask"],
        )

        active_layers = count_active_layers(cfg["mask"])
        est_params = estimate_params(active_layers)

        with output_json.open("r", encoding="utf-8") as f:
            preds = json.load(f)

        row = {
            "variant": name,
            "image_only": int(cfg["image_only"]),
            "no_chat_template": int(cfg["no_chat"]),
            "lm_layers_mask": cfg["mask"],
            "active_lm_layers": active_layers,
            "est_params": est_params,
            "runtime_sec": round(elapsed, 3),
            "num_predictions": len(preds),
            "output_json": str(output_json),
        }
        rows.append(row)
        print(row)

    fieldnames = list(rows[0].keys()) if rows else []
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
