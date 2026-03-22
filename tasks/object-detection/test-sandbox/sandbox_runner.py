"""
NM i AI 2026 Sandbox Simulator — Exact Competition Replica.

Mirrors the competition sandbox environment EXACTLY:
- Python 3.11
- Extracts submission ZIP (enforces file/size limits)
- AST-based security scan (blocked imports + calls)
- Runs run.py --input /data/images --output /predictions.json
- Enforces 360s hard timeout (300s inference + 60s loading grace)
- Validates output JSON format (flat COCO-style)
- Scores predictions: 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
- Reports per-image timing predictions
"""
import argparse
import ast
import copy
import json
import pathlib
import signal
import subprocess
import time
import zipfile

# ── Security scanning ────────────────────────────────────────────────────────

BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "ctypes", "builtins", "importlib",
    "pickle", "marshal", "shelve", "shutil", "yaml", "requests", "urllib",
    "http.client", "http", "multiprocessing", "threading", "signal", "gc",
    "code", "codeop", "pty",
})

BLOCKED_CALLS = frozenset({
    "eval", "exec", "compile", "__import__",
})

# getattr() is blocked when used with dangerous attribute names
DANGEROUS_GETATTR_NAMES = frozenset({
    "__import__", "__builtins__", "__subclasses__", "__globals__",
    "__code__", "__reduce__", "system", "popen", "exec",
})

# File limits from competition docs
MAX_ZIP_SIZE_MB = 420
MAX_FILES = 1000
MAX_PYTHON_FILES = 10
MAX_WEIGHT_FILES = 3
MAX_WEIGHT_SIZE_MB = 420
ALLOWED_EXTENSIONS = frozenset({
    ".py", ".json", ".yaml", ".yml", ".cfg",
    ".pt", ".pth", ".onnx", ".safetensors", ".npy",
})
WEIGHT_EXTENSIONS = frozenset({".pt", ".pth", ".onnx", ".safetensors", ".npy"})


def scan_python_file(path: pathlib.Path) -> list[str]:
    """AST-based security scan of a Python file. Returns list of violations."""
    violations = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        violations.append(f"  SyntaxError in {path.name}: {e}")
        return violations

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_MODULES or alias.name in BLOCKED_MODULES:
                    violations.append(
                        f"  {path.name}:{node.lineno} - blocked import: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BLOCKED_MODULES or node.module in BLOCKED_MODULES:
                    violations.append(
                        f"  {path.name}:{node.lineno} - blocked from-import: {node.module}"
                    )

        # Check blocked function calls
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                # Bare call: eval(), exec(), compile(), __import__()
                name = func.id
            # NOTE: Do NOT check ast.Attribute — model.eval() is legitimate PyTorch
            if name and name in BLOCKED_CALLS:
                violations.append(
                    f"  {path.name}:{node.lineno} - blocked call: {name}()"
                )
            # Check getattr() with dangerous attribute names (bare call only)
            if isinstance(func, ast.Name) and func.id == "getattr" and node.args and len(node.args) >= 2:
                attr_arg = node.args[1]
                if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                    if attr_arg.value in DANGEROUS_GETATTR_NAMES:
                        violations.append(
                            f"  {path.name}:{node.lineno} - blocked getattr with dangerous name: '{attr_arg.value}'"
                        )

    return violations


def scan_submission(submission_dir: pathlib.Path) -> tuple[bool, list[str]]:
    """Scan all Python files in submission for security violations."""
    all_violations = []
    for pyfile in submission_dir.rglob("*.py"):
        violations = scan_python_file(pyfile)
        all_violations.extend(violations)

    # Check for symlinks
    for p in submission_dir.rglob("*"):
        if p.is_symlink():
            all_violations.append(f"  Symlink found: {p.name}")

    # Check for binary executables (ELF, Mach-O, PE)
    for p in submission_dir.rglob("*"):
        if p.is_file() and p.suffix not in WEIGHT_EXTENSIONS:
            try:
                header = p.read_bytes()[:4]
                if header[:4] == b"\x7fELF":
                    all_violations.append(f"  ELF binary found: {p.name}")
                elif header[:2] == b"MZ":
                    all_violations.append(f"  PE binary found: {p.name}")
                elif header[:4] in (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf",
                                     b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe"):
                    all_violations.append(f"  Mach-O binary found: {p.name}")
            except Exception:
                pass

    # Check for path traversal in ZIP entries
    for p in submission_dir.rglob("*"):
        rel = str(p.relative_to(submission_dir))
        if ".." in rel or rel.startswith("/"):
            all_violations.append(f"  Path traversal attempt: {rel}")

    return len(all_violations) == 0, all_violations


def validate_file_limits(submission_dir: pathlib.Path) -> tuple[bool, list[str]]:
    """Check competition file limits."""
    issues = []

    all_files = [p for p in submission_dir.rglob("*") if p.is_file()]
    py_files = [p for p in all_files if p.suffix == ".py"]
    weight_files = [p for p in all_files if p.suffix in WEIGHT_EXTENSIONS]

    # Total file count
    if len(all_files) > MAX_FILES:
        issues.append(f"  Too many files: {len(all_files)} (limit: {MAX_FILES})")

    # Python file count
    if len(py_files) > MAX_PYTHON_FILES:
        issues.append(f"  Too many Python files: {len(py_files)} (limit: {MAX_PYTHON_FILES})")

    # Weight file count
    if len(weight_files) > MAX_WEIGHT_FILES:
        issues.append(f"  Too many weight files: {len(weight_files)} (limit: {MAX_WEIGHT_FILES})")
        for wf in weight_files:
            issues.append(f"    - {wf.name} ({wf.stat().st_size / 1024 / 1024:.1f} MB)")

    # Weight total size
    weight_total = sum(f.stat().st_size for f in weight_files)
    weight_total_mb = weight_total / (1024 * 1024)
    if weight_total_mb > MAX_WEIGHT_SIZE_MB:
        issues.append(f"  Weight files too large: {weight_total_mb:.1f} MB (limit: {MAX_WEIGHT_SIZE_MB} MB)")

    # Check allowed extensions
    for f in all_files:
        if f.suffix and f.suffix not in ALLOWED_EXTENSIONS:
            issues.append(f"  Disallowed file extension: {f.name} ({f.suffix})")

    # Info
    issues.append(f"  INFO: {len(all_files)} files, {len(py_files)} Python, {len(weight_files)} weights ({weight_total_mb:.1f} MB)")

    errors = [i for i in issues if not i.strip().startswith("INFO:")]
    return len(errors) == 0, issues


def validate_output(output_path: pathlib.Path) -> tuple[bool, list[str]]:
    """Validate the output JSON matches the REAL competition format.

    Competition expects FLAT COCO-style:
    [{"image_id": 42, "bbox": [x,y,w,h], "category_id": N, "score": F}, ...]

    image_id is an INTEGER (extracted from filename), NOT a string.
    """
    issues = []

    if not output_path.exists():
        return False, ["Output file not created"]

    try:
        data = json.loads(output_path.read_text())
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    if not isinstance(data, list):
        return False, [f"Output must be a JSON array, got {type(data).__name__}"]

    if len(data) == 0:
        issues.append("WARNING: No detections in output (empty array)")
        return True, issues

    # REJECT nested format
    first = data[0]
    if "predictions" in first:
        issues.append("WRONG FORMAT: Found 'predictions' key - server expects flat format")
        issues.append("Each detection must be its own entry: {image_id, bbox, category_id, score}")
        return False, issues

    # REJECT wrong field names
    if "confidence" in first and "score" not in first:
        issues.append("WRONG FIELD: Found 'confidence' - server expects 'score'")
        return False, issues

    # Validate entries (sample first 50 + last 10)
    required = {"image_id", "bbox", "category_id", "score"}
    check_indices = list(range(min(50, len(data)))) + list(range(max(0, len(data) - 10), len(data)))
    check_indices = sorted(set(check_indices))

    for i in check_indices:
        entry = data[i]
        missing = required - set(entry.keys())
        if missing:
            issues.append(f"Entry {i}: missing fields: {', '.join(sorted(missing))}")
            continue

        # image_id: official docs say int, but server accepts both string and int
        img_id = entry["image_id"]
        if isinstance(img_id, str) and i < 3:
            issues.append(
                f"WARNING: Entry {i}: image_id is string '{img_id}' - docs say int "
                f"(e.g., img_00042.jpg -> 42). Server accepts both but int is canonical."
            )
        elif not isinstance(img_id, (int, str)):
            issues.append(f"Entry {i}: image_id must be int or string, got {type(img_id).__name__}: {img_id}")

        # bbox
        if "bbox" in entry:
            bbox = entry["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                issues.append(f"Entry {i}: bbox must be [x,y,w,h] (4 elements)")
            elif not all(isinstance(v, (int, float)) for v in bbox):
                issues.append(f"Entry {i}: bbox values must be numbers")
            else:
                x, y, w, h = bbox
                if w < 0 or h < 0:
                    issues.append(f"Entry {i}: bbox has negative width/height: w={w}, h={h}")

        # category_id: 0-356 (356 = unknown_product)
        if "category_id" in entry:
            cid = entry["category_id"]
            if not isinstance(cid, int) or not (0 <= cid <= 356):
                issues.append(f"Entry {i}: category_id {cid} out of range 0-356")

        # score
        if "score" in entry:
            score = entry["score"]
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                issues.append(f"Entry {i}: score {score} out of range 0-1")

    # Summary stats
    unique_images = len(set(e.get("image_id", "") for e in data))
    categories = set(e.get("category_id", -1) for e in data)
    scores = [e.get("score", 0) for e in data if isinstance(e.get("score"), (int, float))]
    str_ids = sum(1 for e in data if isinstance(e.get("image_id"), str))
    int_ids = sum(1 for e in data if isinstance(e.get("image_id"), int))

    issues.append(f"INFO: {len(data)} detections across {unique_images} images")
    issues.append(f"INFO: image_id types: {int_ids} int, {str_ids} string")
    if scores:
        issues.append(f"INFO: Score range: {min(scores):.4f} - {max(scores):.4f}")
    issues.append(f"INFO: Categories used: {len(categories)} (range {min(categories)}-{max(categories)})")

    has_errors = any(not msg.strip().startswith(("INFO:", "WARNING:")) for msg in issues)
    return not has_errors, issues


def score_predictions(predictions_path: pathlib.Path, annotations_path: pathlib.Path) -> dict:
    """Score predictions using pycocotools mAP@0.5.

    Competition formula: Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

    Detection mAP: IoU >= 0.5, category ignored (all preds treated as single class)
    Classification mAP: IoU >= 0.5 AND correct category_id
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io
    import contextlib

    # Load ground truth
    with open(annotations_path) as f:
        gt_data = json.load(f)

    # Load predictions
    with open(predictions_path) as f:
        pred_data = json.load(f)

    if not pred_data:
        return {
            "detection_mAP": 0.0,
            "classification_mAP": 0.0,
            "final_score": 0.0,
            "num_predictions": 0,
            "num_gt": len(gt_data.get("annotations", [])),
        }

    # Normalize image_ids: if predictions use strings, convert to int
    img_name_to_id = {}
    for img in gt_data["images"]:
        img_name_to_id[img["file_name"]] = img["id"]

    for pred in pred_data:
        if isinstance(pred["image_id"], str):
            name = pred["image_id"]
            if name in img_name_to_id:
                pred["image_id"] = img_name_to_id[name]
            else:
                # Try stripping to get numeric id
                try:
                    pred["image_id"] = int(name.replace("img_", "").replace(".jpg", "").replace(".jpeg", "").replace(".png", ""))
                except ValueError:
                    pass

    # Ensure all predictions have required 'area' for cocoeval
    for pred in pred_data:
        if "area" not in pred:
            if "bbox" in pred and len(pred["bbox"]) == 4:
                pred["area"] = pred["bbox"][2] * pred["bbox"][3]
            else:
                pred["area"] = 0

    # Debug: verify image_id conversion worked
    gt_img_ids = {img["id"] for img in gt_data["images"]}
    pred_img_ids = {pred["image_id"] for pred in pred_data}
    matched = pred_img_ids & gt_img_ids
    unmatched = pred_img_ids - gt_img_ids
    print(f"  DEBUG: GT images: {len(gt_img_ids)}, Pred unique img_ids: {len(pred_img_ids)}")
    print(f"  DEBUG: Matched: {len(matched)}, Unmatched: {len(unmatched)}")
    if unmatched:
        print(f"  DEBUG: Sample unmatched: {list(unmatched)[:5]}")
    print(f"  DEBUG: Sample pred img_ids: {list(pred_img_ids)[:5]}")
    print(f"  DEBUG: Sample GT img_ids: {list(gt_img_ids)[:5]}")

    # Filter predictions to only include valid image_ids
    pred_data = [p for p in pred_data if p["image_id"] in gt_img_ids]
    if not pred_data:
        return {
            "detection_mAP": 0.0,
            "classification_mAP": 0.0,
            "final_score": 0.0,
            "num_predictions": 0,
            "num_gt": len(gt_data.get("annotations", [])),
            "error": "No predictions matched ground truth image_ids",
        }

    # Write predictions to temp file for loadRes (more reliable than passing list)
    import tempfile
    pred_tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    pred_tmp.write_text(json.dumps(pred_data))

    # ── 1. Classification mAP (category-aware) ──────────────────────────
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO()
        coco_gt.dataset = gt_data
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(str(pred_tmp))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.iouThrs = [0.5]  # Only IoU 0.5
        coco_eval.params.maxDets = [1, 100, 300]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # stats[0] = AP @ IoU=0.50 | area=all | maxDets=100
    classification_mAP = float(coco_eval.stats[0])

    # ── 2. Detection mAP (category-agnostic) ────────────────────────────
    # Remap everything to single class (category_id=0)
    gt_detection = copy.deepcopy(gt_data)
    for ann in gt_detection["annotations"]:
        ann["category_id"] = 0
    gt_detection["categories"] = [{"id": 0, "name": "product", "supercategory": "product"}]

    pred_detection = copy.deepcopy(pred_data)
    for pred in pred_detection:
        pred["category_id"] = 0

    pred_det_tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    pred_det_tmp.write_text(json.dumps(pred_detection))

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt_det = COCO()
        coco_gt_det.dataset = gt_detection
        coco_gt_det.createIndex()

        coco_dt_det = coco_gt_det.loadRes(str(pred_det_tmp))
        coco_eval_det = COCOeval(coco_gt_det, coco_dt_det, "bbox")
        coco_eval_det.params.iouThrs = [0.5]
        coco_eval_det.params.maxDets = [1, 100, 300]
        coco_eval_det.evaluate()
        coco_eval_det.accumulate()
        coco_eval_det.summarize()

    detection_mAP = float(coco_eval_det.stats[0])

    # ── 3. Final score ───────────────────────────────────────────────────
    final_score = 0.7 * detection_mAP + 0.3 * classification_mAP

    return {
        "detection_mAP": round(detection_mAP, 4),
        "classification_mAP": round(classification_mAP, 4),
        "final_score": round(final_score, 4),
        "final_score_pct": round(final_score * 100, 1),
        "num_predictions": len(pred_data),
        "num_gt": len(gt_data.get("annotations", [])),
    }


def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 Sandbox Simulator (Exact Replica)")
    parser.add_argument("--zip", default="/submission.zip", help="Path to submission ZIP")
    parser.add_argument("--input", default="/data/images", help="Path to test images")
    parser.add_argument("--output", default="/predictions.json", help="Output JSON path")
    parser.add_argument("--annotations", default="/data/annotations.json",
                        help="Path to COCO annotations for scoring")
    parser.add_argument("--timeout", type=int, default=360, help="Timeout in seconds (300s inference + 60s loading grace)")
    parser.add_argument("--skip-run", action="store_true", help="Only validate, don't run")
    args = parser.parse_args()

    zip_path = pathlib.Path(args.zip)
    input_dir = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    submission_dir = pathlib.Path("/submission")

    print("=" * 70)
    print("  NM i AI 2026 - Sandbox Simulator (EXACT COMPETITION REPLICA)")
    print("=" * 70)
    print(f"  Python: {__import__('platform').python_version()}")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        pass

    # ── Step 1: ZIP validation ────────────────────────────────────────────
    print(f"\n[1/8] ZIP Validation")
    if not zip_path.exists():
        print(f"  FAIL: ZIP not found at {zip_path}")
        exit(1)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    if zip_size_mb > MAX_ZIP_SIZE_MB:
        print(f"  FAIL: ZIP is {zip_size_mb:.1f}MB (limit: {MAX_ZIP_SIZE_MB}MB)")
        exit(1)
    print(f"  PASS: ZIP size {zip_size_mb:.1f}MB (limit: {MAX_ZIP_SIZE_MB}MB)")

    # ── Step 2: Extract ──────────────────────────────────────────────────
    print(f"\n[2/8] Extracting submission")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if "run.py" not in names:
            run_files = [n for n in names if n.endswith("run.py")]
            if run_files:
                print(f"  FAIL: run.py not at ZIP root. Found: {run_files}")
            else:
                print("  FAIL: run.py not found in ZIP")
            exit(1)
        zf.extractall(submission_dir)

    file_count = sum(1 for _ in submission_dir.rglob("*") if _.is_file())
    print(f"  PASS: Extracted {file_count} files, run.py at root")

    # ── Step 3: File limits ──────────────────────────────────────────────
    print(f"\n[3/8] File limits check")
    limits_ok, limit_issues = validate_file_limits(submission_dir)
    for issue in limit_issues:
        print(f"  {issue}")
    if not limits_ok:
        print("  FAIL: File limits exceeded")
        exit(1)
    print("  PASS: All file limits OK")

    # ── Step 4: Security scan ────────────────────────────────────────────
    print(f"\n[4/8] Security scan")
    clean, violations = scan_submission(submission_dir)
    if not clean:
        print(f"  FAIL: {len(violations)} security violations:")
        for v in violations[:20]:
            print(f"    {v}")
        if len(violations) > 20:
            print(f"    ... and {len(violations) - 20} more")
        exit(1)
    print("  PASS: No blocked imports, calls, symlinks, or binaries")

    # ── Step 5: Check test images ────────────────────────────────────────
    print(f"\n[5/8] Test images")
    if not input_dir.exists():
        print(f"  FAIL: Input directory not found: {input_dir}")
        exit(1)
    image_files = sorted(
        list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
    )
    print(f"  PASS: {len(image_files)} test images in {input_dir}")

    if args.skip_run:
        print(f"\n[6/8] Skipped (--skip-run)")
        print(f"\n[7/8] Skipped (--skip-run)")
        print("\nDone (validation only)")
        return

    # ── Step 6: Run inference ────────────────────────────────────────────
    print(f"\n[6/8] Running inference (timeout: {args.timeout}s)")
    cmd = ["python3", "run.py", "--input", str(input_dir), "--output", str(output_path)]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working dir: {submission_dir}")

    start = time.time()
    try:
        # Use Popen for proper process group kill on timeout
        proc = subprocess.Popen(
            cmd,
            cwd=str(submission_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=lambda: signal.signal(signal.SIGALRM, signal.SIG_DFL),
        )

        try:
            stdout, stderr = proc.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            elapsed = time.time() - start
            print(f"  FAIL: KILLED after {elapsed:.1f}s timeout (limit: {args.timeout}s)")
            print(f"  This submission would FAIL on the competition server.")
            if len(image_files) > 0:
                est_per_image = elapsed / len(image_files)
                print(f"  Estimated per-image: {est_per_image:.2f}s")
                print(f"  Need: < {args.timeout / len(image_files):.2f}s per image for {len(image_files)} images")
            exit(1)

        elapsed = time.time() - start

        if proc.returncode != 0:
            print(f"  FAIL: Exit code {proc.returncode} after {elapsed:.1f}s")
            if stderr:
                print("  STDERR (last 40 lines):")
                for line in stderr.strip().split("\n")[-40:]:
                    print(f"    {line}")
            if stdout:
                print("  STDOUT (last 10 lines):")
                for line in stdout.strip().split("\n")[-10:]:
                    print(f"    {line}")
            exit(1)

        print(f"  PASS: Completed in {elapsed:.1f}s (limit: {args.timeout}s)")

        # Timing analysis
        budget_pct = elapsed / args.timeout * 100
        if len(image_files) > 0:
            per_image = elapsed / len(image_files)
            print(f"  Timing: {per_image:.2f}s/image, {budget_pct:.0f}% of budget used")

            # Predict for different test set sizes
            for n in [50, 100, 150, 200, 250, 300]:
                predicted = per_image * n
                status = "OK" if predicted < args.timeout else "TIMEOUT!"
                print(f"    {n:>3} images: {predicted:>6.1f}s ({predicted/args.timeout*100:>5.1f}%) {status}")

        if budget_pct > 80:
            print(f"  WARNING: Used {budget_pct:.0f}% of timeout budget - RISKY for larger test sets!")
        elif budget_pct > 60:
            print(f"  WARNING: Used {budget_pct:.0f}% - may be tight for competition test set")

        # Print stdout for debugging
        if stdout and stdout.strip():
            print(f"\n  --- Submission stdout ---")
            for line in stdout.strip().split("\n"):
                print(f"    {line}")

    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAIL: Exception after {elapsed:.1f}s: {e}")
        exit(1)

    # ── Step 7: Validate output ──────────────────────────────────────────
    print(f"\n[7/8] Output validation")
    valid, issues = validate_output(output_path)
    for issue in issues:
        prefix = "  " if issue.strip().startswith(("INFO:", "WARNING:")) else "  "
        print(f"{prefix}{issue}")

    if not valid:
        print("  FAIL: Output validation failed")
        exit(1)
    print("  PASS: Output format valid")

    # ── Step 8: Score predictions ───────────────────────────────────────
    annotations_path = pathlib.Path(args.annotations)
    if annotations_path.exists():
        print(f"\n[8/8] Scoring (mAP@0.5)")
        try:
            scores = score_predictions(output_path, annotations_path)
            print(f"  Detection mAP@0.5:       {scores['detection_mAP']:.4f}")
            print(f"  Classification mAP@0.5:  {scores['classification_mAP']:.4f}")
            print(f"  ──────────────────────────────────")
            print(f"  Final Score:  {scores['final_score_pct']:.1f} / 100")
            print(f"    (0.7 × {scores['detection_mAP']:.4f} + 0.3 × {scores['classification_mAP']:.4f})")
            print(f"  Predictions: {scores['num_predictions']}, Ground truth: {scores['num_gt']}")
        except Exception as e:
            print(f"  ERROR: Scoring failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[8/8] Scoring skipped (no annotations at {annotations_path})")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ALL CHECKS PASSED - SUBMISSION READY")
    print(f"  Inference time: {elapsed:.1f}s / {args.timeout}s ({budget_pct:.0f}%)")
    print(f"  ZIP size: {zip_size_mb:.1f}MB / {MAX_ZIP_SIZE_MB}MB")
    if len(image_files) > 0:
        per_img = elapsed / len(image_files)
        est_250 = per_img * 250
        print(f"  Per-image: {per_img:.2f}s (est. {est_250:.0f}s for 250 test images)")
        if est_250 > args.timeout:
            print(f"  *** WARNING: Would likely TIMEOUT on 250-image test set! ***")
    if annotations_path.exists() and 'scores' in dir():
        try:
            print(f"  SCORE: {scores['final_score_pct']:.1f} / 100")
        except Exception:
            pass
    print("=" * 70)


if __name__ == "__main__":
    main()
