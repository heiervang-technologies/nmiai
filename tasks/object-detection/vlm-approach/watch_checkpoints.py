"""Watch submission artifacts and MarkusNet checkpoints, evaluate on the clean val split."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch

from eval_stratified_map import append_csv, append_jsonl, evaluate_submission
from export_nf4 import GROUP_SIZE, quantize_tensor_nf4


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
VAL_IMAGES = ROOT / "data-creation" / "data" / "clean_split" / "val" / "images"
VAL_LABELS = ROOT / "data-creation" / "data" / "clean_split" / "val" / "labels"
DATASET_YAML = ROOT / "data-creation" / "data" / "clean_split" / "dataset.yaml"
WATCH_DIRS = [HERE / "training_output", HERE / "training_output_multitask"]
OUT_DIR = HERE / "val_watch"
MARKUSNET_TEMPLATE = ROOT / "submission-markusnet"


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def discover_submission_specs() -> list[dict]:
    specs = []
    for path in sorted(ROOT.glob("nmiai_*.zip")):
        specs.append(
            {
                "kind": "submission",
                "source_type": "zip",
                "model_name": path.stem,
                "source_path": path.resolve(),
                "fingerprint": fingerprint(path),
            }
        )
    for path in sorted(ROOT.glob("submission-*")):
        run_py = path / "run.py"
        if run_py.exists():
            specs.append(
                {
                    "kind": "submission",
                    "source_type": "dir",
                    "model_name": path.name,
                    "source_path": path.resolve(),
                    "fingerprint": fingerprint(run_py),
                }
            )
    return specs


def discover_checkpoint_specs() -> list[dict]:
    specs = []
    for watch_dir in WATCH_DIRS:
        if not watch_dir.exists():
            continue
        for checkpoint_path in sorted(watch_dir.glob("best/best.pt")):
            specs.append(build_checkpoint_spec(checkpoint_path))
        for checkpoint_path in sorted(watch_dir.glob("checkpoint-*/checkpoint.pt")):
            specs.append(build_checkpoint_spec(checkpoint_path))
    return specs


def build_checkpoint_spec(checkpoint_path: Path) -> dict:
    parent_name = checkpoint_path.parent.name
    group_name = checkpoint_path.parent.parent.name
    model_name = f"{group_name}_{parent_name}"
    return {
        "kind": "checkpoint",
        "source_type": "checkpoint",
        "model_name": model_name,
        "source_path": checkpoint_path.resolve(),
        "fingerprint": fingerprint(checkpoint_path),
    }


def state_key(spec: dict) -> str:
    return f"{spec['kind']}::{spec['source_path']}"


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_checkpoint_to_nf4(checkpoint_path: Path, output_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state"]
    nf4_state = {}
    fp16_state = {}
    for key, value in model_state.items():
        if "embed_tokens" in key or "lm_head" in key:
            continue
        if value.dim() >= 2 and value.numel() >= GROUP_SIZE:
            nf4_state[key] = quantize_tensor_nf4(value)
        else:
            fp16_state[key] = value.to(torch.float16) if value.is_floating_point() else value
    cls_fp16 = {key: value.to(torch.float16) for key, value in ckpt["cls_head_state"].items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "nf4_state": nf4_state,
            "fp16_state": fp16_state,
            "cls_head_state": cls_fp16,
            "accuracy": ckpt.get("accuracy", ckpt.get("val_acc", 0.0)),
            "global_step": ckpt.get("global_step", ckpt.get("step", 0)),
            "quantization": "nf4",
            "group_size": GROUP_SIZE,
            "source_checkpoint": str(checkpoint_path.resolve()),
        },
        output_path,
    )
    return {
        "global_step": ckpt.get("global_step", ckpt.get("step", 0)),
        "accuracy": ckpt.get("accuracy", ckpt.get("val_acc", 0.0)),
    }


def symlink_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def evaluate_checkpoint_spec(spec: dict, predictions_dir: Path, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="markusnet_watch_") as tmpdir:
        submission_dir = Path(tmpdir) / "submission-markusnet"
        submission_dir.mkdir(parents=True, exist_ok=True)
        symlink_or_copy(MARKUSNET_TEMPLATE / "run.py", submission_dir / "run.py")
        symlink_or_copy(MARKUSNET_TEMPLATE / "best.onnx", submission_dir / "best.onnx")
        export_metadata = export_checkpoint_to_nf4(spec["source_path"], submission_dir / "markusnet_351m_nf4.pt")
        prediction_path = predictions_dir / f"{spec['model_name']}.json"
        return evaluate_submission(
            submission_path=submission_dir,
            submission_type="dir",
            images_dir=VAL_IMAGES,
            labels_dir=VAL_LABELS,
            dataset_yaml=DATASET_YAML,
            output_json=prediction_path,
            model_name=spec["model_name"],
            timeout=timeout,
            metadata={"watch_source": "checkpoint", **export_metadata},
        )


def evaluate_submission_spec(spec: dict, predictions_dir: Path, timeout: int) -> dict:
    prediction_path = predictions_dir / f"{spec['model_name']}.json"
    return evaluate_submission(
        submission_path=spec["source_path"],
        submission_type=spec["source_type"],
        images_dir=VAL_IMAGES,
        labels_dir=VAL_LABELS,
        dataset_yaml=DATASET_YAML,
        output_json=prediction_path,
        model_name=spec["model_name"],
        timeout=timeout,
        metadata={"watch_source": "artifact"},
    )


def run_cycle(state: dict, output_dir: Path, timeout: int) -> dict:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "watch_results.csv"
    jsonl_path = output_dir / "watch_results.jsonl"

    specs = discover_submission_specs() + discover_checkpoint_specs()
    for spec in specs:
        key = state_key(spec)
        if state.get(key) == spec["fingerprint"]:
            continue
        print(f"Evaluating {spec['model_name']} from {spec['source_path']}")
        try:
            if spec["kind"] == "checkpoint":
                result = evaluate_checkpoint_spec(spec, predictions_dir, timeout)
            else:
                result = evaluate_submission_spec(spec, predictions_dir, timeout)
            append_jsonl(jsonl_path, result)
            append_csv(csv_path, result)
            state[key] = spec["fingerprint"]
            print(
                f"  combined={result['combined_score']:.4f} det={result['detection_map50']:.4f} "
                f"cls={result['classification_map50']:.4f} runtime={result['inference_seconds']:.1f}s"
            )
        except Exception as exc:  # pragma: no cover - operational failure path
            print(f"  FAILED: {exc}")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "watch_state.json"
    state = load_state(state_path)

    while True:
        state = run_cycle(state, output_dir, args.timeout)
        save_state(state_path, state)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
