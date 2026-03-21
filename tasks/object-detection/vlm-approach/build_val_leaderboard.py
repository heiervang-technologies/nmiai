"""Evaluate known submission artifacts and build a clean validation leaderboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_stratified_map import append_csv, append_jsonl, evaluate_submission


ROOT = Path(__file__).resolve().parent.parent
VAL_IMAGES = ROOT / "data-creation" / "data" / "stratified_split" / "val" / "images"
VAL_LABELS = ROOT / "data-creation" / "data" / "stratified_split" / "val" / "labels"
DATASET_YAML = ROOT / "data-creation" / "data" / "stratified_split" / "dataset.yaml"
OUT_DIR = Path(__file__).resolve().parent / "val_eval"


def default_candidates() -> list[dict]:
    candidates = []
    seen = set()

    def add(path: Path, submission_type: str, model_name: str | None = None):
        resolved = path.resolve()
        if not resolved.exists() or resolved in seen:
            return
        seen.add(resolved)
        candidates.append(
            {
                "path": resolved,
                "submission_type": submission_type,
                "model_name": model_name or resolved.stem,
            }
        )

    for path in sorted(ROOT.glob("nmiai_*.zip")):
        add(path, "zip")

    for path in sorted(ROOT.glob("submission-*")):
        if (path / "run.py").exists():
            add(path, "dir", model_name=path.name)

    return candidates


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# Clean Val Leaderboard",
        "",
        "| Rank | Model | Combined | Detect mAP@0.5 | Class mAP@0.5 | Runtime (s) | Images | Source |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, result in enumerate(results, start=1):
        lines.append(
            "| {rank} | {model} | {combined:.4f} | {det:.4f} | {cls:.4f} | {runtime:.1f} | {images} | {source} |".format(
                rank=idx,
                model=result["model_name"],
                combined=result["combined_score"],
                det=result["detection_map50"],
                cls=result["classification_map50"],
                runtime=result["inference_seconds"],
                images=result["usable_images"],
                source=Path(result["submission_path"]).name,
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--images", type=Path, default=VAL_IMAGES)
    parser.add_argument("--labels", type=Path, default=VAL_LABELS)
    parser.add_argument("--dataset-yaml", type=Path, default=DATASET_YAML)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--path", type=Path, action="append", help="Optional explicit artifact paths to evaluate")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "leaderboard_runs.jsonl"
    csv_path = output_dir / "leaderboard_runs.csv"
    failures_path = output_dir / "leaderboard_failures.json"
    leaderboard_json = output_dir / "leaderboard_latest.json"
    leaderboard_md = output_dir / "leaderboard_latest.md"

    if args.path:
        candidates = []
        for raw_path in args.path:
            path = raw_path.resolve()
            submission_type = "zip" if path.suffix.lower() == ".zip" else "dir"
            candidates.append({
                "path": path,
                "submission_type": submission_type,
                "model_name": path.stem if submission_type == "zip" else path.name,
            })
    else:
        candidates = default_candidates()

    results = []
    failures = []
    for candidate in candidates:
        prediction_path = predictions_dir / f"{candidate['model_name']}.json"
        metadata = {"leaderboard_run": True}
        try:
            result = evaluate_submission(
                submission_path=candidate["path"],
                submission_type=candidate["submission_type"],
                images_dir=args.images,
                labels_dir=args.labels,
                dataset_yaml=args.dataset_yaml,
                output_json=prediction_path,
                model_name=candidate["model_name"],
                timeout=args.timeout,
                metadata=metadata,
            )
            results.append(result)
            append_jsonl(jsonl_path, result)
            append_csv(csv_path, result)
            print(
                f"[{len(results)}] {result['model_name']}: combined={result['combined_score']:.4f} "
                f"det={result['detection_map50']:.4f} cls={result['classification_map50']:.4f} "
                f"runtime={result['inference_seconds']:.1f}s images={result['usable_images']}"
            )
        except Exception as exc:  # pragma: no cover - operational failure path
            failure = {
                "model_name": candidate["model_name"],
                "submission_path": str(candidate["path"]),
                "submission_type": candidate["submission_type"],
                "error": repr(exc),
            }
            failures.append(failure)
            print(f"[FAIL] {candidate['model_name']}: {exc}")

    results.sort(
        key=lambda item: (
            item["combined_score"],
            item["classification_map50"],
            item["detection_map50"],
        ),
        reverse=True,
    )

    leaderboard_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    leaderboard_md.write_text(render_markdown(results), encoding="utf-8")
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved leaderboard to {leaderboard_json}")
    print(f"Saved markdown table to {leaderboard_md}")
    if failures:
        print(f"Recorded {len(failures)} failures in {failures_path}")


if __name__ == "__main__":
    main()
