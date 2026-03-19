"""
Package the hybrid YOLO + DINOv2 submission ZIP.

Collects all necessary files and creates the submission ZIP:
- run.py (hybrid inference pipeline)
- model.pt (fine-tuned YOLO weights)
- dinov2_vits14.pth (DINOv2 ViT-S weights)
- ref_embeddings.pth (pre-computed reference embeddings)
- linear_probe.pth (optional, trained linear classifier)

Usage: python package_submission.py [--yolo-weights PATH] [--output submission.zip]
"""

import argparse
import zipfile
from pathlib import Path
import shutil


SCRIPT_DIR = Path(__file__).parent


def package(yolo_weights: Path, output: Path, include_probe: bool = True):
    """Create submission ZIP."""
    files_to_include = []

    # Required: run.py
    run_py = SCRIPT_DIR / "run.py"
    assert run_py.exists(), f"run.py not found at {run_py}"
    files_to_include.append(("run.py", run_py))

    # Required: YOLO weights
    assert yolo_weights.exists(), f"YOLO weights not found at {yolo_weights}"
    files_to_include.append(("model.pt", yolo_weights))

    # Required: DINOv2 weights
    dinov2_weights = SCRIPT_DIR / "dinov2_vits14.pth"
    assert dinov2_weights.exists(), f"DINOv2 weights not found at {dinov2_weights}"
    files_to_include.append(("dinov2_vits14.pth", dinov2_weights))

    # Required: Reference embeddings (prefer our multi-angle version, fallback to data agent's)
    ref_emb = SCRIPT_DIR / "ref_embeddings.pth"
    if not ref_emb.exists():
        ref_emb = SCRIPT_DIR.parent / "data-creation" / "data" / "ref_embeddings.pth"
    assert ref_emb.exists(), f"Reference embeddings not found"
    files_to_include.append(("ref_embeddings.pth", ref_emb))

    # Optional: Linear probe
    if include_probe:
        probe = SCRIPT_DIR / "linear_probe.pth"
        if probe.exists():
            files_to_include.append(("linear_probe.pth", probe))
            print(f"Including linear probe: {probe.stat().st_size / 1024:.1f} KB")
        else:
            print("Linear probe not found, skipping (nearest-neighbor only)")

    # Calculate total size
    total_size = sum(f.stat().st_size for _, f in files_to_include)
    print(f"\nFiles to package:")
    for name, path in files_to_include:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {name}: {size_mb:.1f} MB")
    print(f"  Total uncompressed: {total_size / 1024 / 1024:.1f} MB")

    if total_size > 420 * 1024 * 1024:
        print(f"\nWARNING: Total size {total_size / 1024 / 1024:.1f} MB exceeds 420 MB limit!")
        print("Consider using a smaller YOLO model or removing optional components.")

    # Create ZIP
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in files_to_include:
            zf.write(path, name)

    zip_size = output.stat().st_size
    print(f"\nCreated: {output}")
    print(f"ZIP size: {zip_size / 1024 / 1024:.1f} MB")

    if zip_size > 420 * 1024 * 1024:
        print("ERROR: ZIP exceeds 420 MB limit!")
        return False

    print(f"Remaining budget: {(420 * 1024 * 1024 - zip_size) / 1024 / 1024:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=SCRIPT_DIR / "model.pt",
        help="Path to fine-tuned YOLO weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "submission" / "submission.zip",
        help="Output ZIP path",
    )
    parser.add_argument("--no-probe", action="store_true", help="Exclude linear probe")
    args = parser.parse_args()

    package(args.yolo_weights, args.output, include_probe=not args.no_probe)


if __name__ == "__main__":
    main()
