"""Package a submission ZIP for the NM i AI object detection challenge."""
import zipfile
import torch
from pathlib import Path


def strip_checkpoint(src_pt: Path, dst_pt: Path):
    """Remove optimizer state from checkpoint to reduce size."""
    ckpt = torch.load(src_pt, map_location='cpu', weights_only=False)
    # Remove optimizer to save space (412MB -> 138MB)
    ckpt_light = {k: v for k, v in ckpt.items() if k != 'optimizer'}
    torch.save(ckpt_light, dst_pt)
    print(f"Stripped: {src_pt.stat().st_size/1e6:.1f}MB -> {dst_pt.stat().st_size/1e6:.1f}MB")


def package(weights_path: Path, output_zip: Path, extra_files: list[tuple[Path, str]] = None):
    """Create submission ZIP with run.py and model weights."""
    run_py = Path(__file__).parent / "run.py"
    stripped = Path("/tmp/submission_best.pt")

    # Strip optimizer state
    strip_checkpoint(weights_path, stripped)

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # run.py must be at root
        zf.write(run_py, 'run.py')
        # Model weights
        zf.write(stripped, 'best.pt')
        # Extra files
        if extra_files:
            for src, arcname in extra_files:
                zf.write(src, arcname)

    size_mb = output_zip.stat().st_size / 1e6
    print(f"Created {output_zip} ({size_mb:.1f} MB)")
    if size_mb > 420:
        print("WARNING: Exceeds 420MB limit!")
    else:
        print(f"OK: {420 - size_mb:.1f} MB under limit")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                       default='/home/me/ht/nmiai/runs/detect/runs/yolov8x_1280_v1/weights/best.pt')
    parser.add_argument('--output', type=str,
                       default='/home/me/ht/nmiai/tasks/object-detection/yolo-approach/submission.zip')
    args = parser.parse_args()

    package(Path(args.weights), Path(args.output))
