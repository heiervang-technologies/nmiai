#!/usr/bin/env python3
"""Time-budgeted YOLO training on data mixtures.

Trains YOLOv8x with configurable data mixture, time budget, and
augmentation control. Runs eval_honest.py after training.

Usage:
  python train_mixture.py dataset.yaml --experiment-id exp001 --description "Polish det-only"
  python train_mixture.py dataset.yaml --max-minutes 30 --no-augment
  python train_mixture.py dataset.yaml --augment  # Phase 2 only
"""
import argparse
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "yolov8x.pt"  # COCO pretrained


def main():
    parser = argparse.ArgumentParser(description="Time-budgeted YOLO training")
    parser.add_argument("data", help="Path to dataset.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model (default: yolov8x.pt COCO pretrained)")
    parser.add_argument("--max-minutes", type=int, default=30, help="Time budget in minutes (default 30, max 60)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (early stopping will likely trigger first)")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--device", default="0", help="GPU device(s), e.g. '0' or '0,1' for multi-GPU")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation (Phase 2 only)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--project", default=str(SCRIPT_DIR / "mixture_runs"), help="Output directory")
    parser.add_argument("--name", default=None, help="Run name (defaults to experiment-id)")
    # Eval args (passed to eval_honest.py)
    parser.add_argument("--experiment-id", default="", help="Experiment ID")
    parser.add_argument("--description", default="", help="Description")
    parser.add_argument("--unique-images", type=int, default=0, help="Unique training images")
    parser.add_argument("--aug-images", type=int, default=0, help="Augmented copies")
    parser.add_argument("--data-sources", default="", help="Data sources")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation")
    args = parser.parse_args()

    # Clamp time budget
    max_minutes = min(args.max_minutes, 60)
    run_name = args.name or args.experiment_id or f"mix_{int(time.time())}"

    print(f"{'=' * 60}")
    print(f"Training: {run_name}")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Time budget: {max_minutes} minutes")
    print(f"Augmentation: {'ON' if args.augment else 'OFF (Phase 1)'}")
    print(f"{'=' * 60}")

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(args.model)

    # Augmentation settings
    if args.augment:
        # Phase 2: standard augmentation
        aug_kwargs = dict(
            mosaic=0.8,
            mixup=0.1,
            copy_paste=0.1,
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
        )
    else:
        # Phase 1: NO augmentation
        aug_kwargs = dict(
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            erasing=0.0,
        )

    t0 = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        patience=15,
        time=max_minutes / 60.0,  # ultralytics uses hours
        project=args.project,
        name=run_name,
        save_period=5,
        exist_ok=True,
        verbose=True,
        amp=True,
        **aug_kwargs,
    )
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed / 60:.1f} minutes")

    # Find best weights
    best_pt = Path(args.project) / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"WARNING: best.pt not found at {best_pt}")
        return

    print(f"Best weights: {best_pt}")

    # Run honest evaluation
    if not args.no_eval and args.experiment_id:
        print(f"\nRunning honest evaluation...")
        eval_cmd = [
            "python", str(SCRIPT_DIR / "eval_honest.py"),
            str(best_pt),
            "--experiment-id", args.experiment_id,
            "--description", args.description,
            "--unique-images", str(args.unique_images),
            "--aug-images", str(args.aug_images),
            "--data-sources", args.data_sources,
        ]
        # Leakage check against training images
        data_yaml = Path(args.data)
        if data_yaml.exists():
            import yaml
            with open(data_yaml) as f:
                cfg = yaml.safe_load(f)
            train_path = cfg.get("train", "")
            if train_path:
                train_images = Path(train_path)
                if not train_images.is_absolute():
                    train_images = data_yaml.parent / train_path
                if train_images.exists():
                    eval_cmd.extend(["--leakage-check", str(train_images)])

        subprocess.run(eval_cmd)

    # Run progress plot
    plot_script = SCRIPT_DIR / "plot_progress.py"
    if plot_script.exists():
        subprocess.run(["python", str(plot_script)])


if __name__ == "__main__":
    main()
