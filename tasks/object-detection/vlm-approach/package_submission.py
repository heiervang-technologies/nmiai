"""Package the dense detector + retrieval submission ZIP."""

import argparse
import zipfile
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent


def package(output: Path, include_probe: bool = True):
    """Create submission ZIP."""
    files_to_include = []

    run_py = SCRIPT_DIR / "run.py"
    helper_py = SCRIPT_DIR / "run_dense_retrieval.py"
    assert run_py.exists(), f"Submission entrypoint not found at {run_py}"
    assert helper_py.exists(), f"Dense retrieval runner not found at {helper_py}"
    files_to_include.append(("run.py", run_py))
    files_to_include.append(("run_dense_retrieval.py", helper_py))

    detector_a = SCRIPT_DIR.parent / "titan-models" / "yolo11x_v3.onnx"
    detector_b = SCRIPT_DIR.parent / "titan-models" / "yolo26x_v3.onnx"
    assert detector_a.exists(), f"Detector not found at {detector_a}"
    assert detector_b.exists(), f"Detector not found at {detector_b}"
    files_to_include.append(("yolo11x_v3.onnx", detector_a))
    files_to_include.append(("yolo26x_v3.onnx", detector_b))

    dinov2_weights = SCRIPT_DIR / "dinov2_vits14.pth"
    assert dinov2_weights.exists(), f"DINOv2 weights not found at {dinov2_weights}"
    files_to_include.append(("dinov2_vits14.pth", dinov2_weights))

    data_bank = SCRIPT_DIR.parent / "data-creation" / "data" / "ref_embeddings.pth"
    assert data_bank.exists(), f"Reference data bank not found at {data_bank}"
    files_to_include.append(("ref_embeddings_data.pth", data_bank))

    multi_bank = SCRIPT_DIR / "ref_embeddings.pth"
    assert multi_bank.exists(), f"Multi-angle reference bank not found at {multi_bank}"
    files_to_include.append(("ref_embeddings_multi.pth", multi_bank))

    if include_probe:
        probe = SCRIPT_DIR / "linear_probe.pth"
        if probe.exists():
            files_to_include.append(("linear_probe.pth", probe))
            print(f"Including linear probe: {probe.stat().st_size / 1024:.1f} KB")
        else:
            print("Linear probe not found, skipping")

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
        "--output",
        type=Path,
        default=SCRIPT_DIR / "submission" / "submission.zip",
        help="Output ZIP path",
    )
    parser.add_argument("--no-probe", action="store_true", help="Exclude linear probe")
    args = parser.parse_args()

    package(args.output, include_probe=not args.no_probe)


if __name__ == "__main__":
    main()
