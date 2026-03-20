"""
NM i AI 2026 Sandbox Simulator.

Mirrors the competition sandbox environment:
- Extracts submission ZIP
- Enforces security restrictions (blocked imports)
- Runs run.py with --input / --output
- Enforces 300s timeout
- Validates output JSON format
- Reports pass/fail with details
"""
import argparse
import ast
import json
import pathlib
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
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in BLOCKED_CALLS:
                violations.append(
                    f"  {path.name}:{node.lineno} - blocked call: {name}()"
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

    # Check for binary executables
    for p in submission_dir.rglob("*"):
        if p.is_file() and not p.suffix:
            try:
                header = p.read_bytes()[:4]
                if header[:4] == b"\x7fELF":  # ELF binary
                    all_violations.append(f"  ELF binary found: {p.name}")
                elif header[:2] == b"MZ":  # PE binary
                    all_violations.append(f"  PE binary found: {p.name}")
            except Exception:
                pass

    return len(all_violations) == 0, all_violations


def validate_output(output_path: pathlib.Path) -> tuple[bool, list[str]]:
    """Validate the output JSON matches expected format."""
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

    # Server expects FLAT COCO-style format ONLY:
    # [{"image_id": "...", "bbox": [x,y,w,h], "category_id": N, "score": F}, ...]
    first = data[0]

    # REJECT nested format
    if "predictions" in first:
        issues.append("WRONG FORMAT: Found 'predictions' key — server expects flat format")
        issues.append("Each detection must be its own entry: {image_id, bbox, category_id, score}")
        return False, issues

    # REJECT wrong field names
    if "confidence" in first and "score" not in first:
        issues.append("WRONG FIELD: Found 'confidence' — server expects 'score'")
        return False, issues

    # Validate flat format
    required = {"image_id", "bbox", "category_id", "score"}
    for i, entry in enumerate(data[:20]):
        missing = required - set(entry.keys())
        if missing:
            issues.append(f"Entry {i}: missing fields: {', '.join(sorted(missing))}")
        if "bbox" in entry:
            bbox = entry["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                issues.append(f"Entry {i}: bbox must be [x,y,w,h] (4 elements)")
            elif not all(isinstance(v, (int, float)) for v in bbox):
                issues.append(f"Entry {i}: bbox values must be numbers")
        if "category_id" in entry:
            cid = entry["category_id"]
            if not isinstance(cid, int) or not (0 <= cid <= 355):
                issues.append(f"Entry {i}: category_id {cid} out of range 0-355")
        if "score" in entry:
            score = entry["score"]
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                issues.append(f"Entry {i}: score {score} out of range 0-1")
        if "image_id" in entry and not isinstance(entry["image_id"], str):
            issues.append(f"Entry {i}: image_id must be a filename string, not {type(entry['image_id']).__name__}")

    unique_images = len(set(e.get("image_id", "") for e in data))
    issues.append(f"INFO: {len(data)} detections across {unique_images} images")

    has_errors = any(not msg.startswith(("INFO:", "WARNING:")) for msg in issues)
    return not has_errors, issues


def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 Sandbox Simulator")
    parser.add_argument("--zip", default="/submission.zip", help="Path to submission ZIP")
    parser.add_argument("--input", default="/data/images", help="Path to test images")
    parser.add_argument("--output", default="/predictions.json", help="Output JSON path")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--skip-run", action="store_true", help="Only validate, don't run")
    args = parser.parse_args()

    zip_path = pathlib.Path(args.zip)
    input_dir = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    submission_dir = pathlib.Path("/submission")

    print("=" * 60)
    print("  NM i AI 2026 - Sandbox Simulator")
    print("=" * 60)

    # ── Step 1: ZIP validation ────────────────────────────────────────────
    print("\n[1/6] ZIP Validation")
    if not zip_path.exists():
        print(f"  FAIL: ZIP not found at {zip_path}")
        exit(1)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    if zip_size_mb > 420:
        print(f"  FAIL: ZIP is {zip_size_mb:.1f}MB (limit: 420MB)")
        exit(1)
    print(f"  PASS: ZIP size {zip_size_mb:.1f}MB (limit: 420MB)")

    # ── Step 2: Extract ──────────────────────────────────────────────────
    print("\n[2/6] Extracting submission")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if "run.py" not in names:
            # Check for nested run.py
            run_files = [n for n in names if n.endswith("run.py")]
            if run_files:
                print(f"  FAIL: run.py not at ZIP root. Found: {run_files}")
            else:
                print("  FAIL: run.py not found in ZIP")
            exit(1)
        zf.extractall(submission_dir)

    file_count = sum(1 for _ in submission_dir.rglob("*") if _.is_file())
    print(f"  PASS: Extracted {file_count} files, run.py at root")

    # ── Step 3: Security scan ────────────────────────────────────────────
    print("\n[3/6] Security scan")
    clean, violations = scan_submission(submission_dir)
    if not clean:
        print(f"  FAIL: {len(violations)} security violations:")
        for v in violations[:20]:
            print(f"    {v}")
        if len(violations) > 20:
            print(f"    ... and {len(violations) - 20} more")
        exit(1)
    print("  PASS: No blocked imports, calls, symlinks, or binaries")

    # ── Step 4: Check test images ────────────────────────────────────────
    print("\n[4/6] Test images")
    if not input_dir.exists():
        print(f"  FAIL: Input directory not found: {input_dir}")
        exit(1)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
    print(f"  PASS: {len(image_files)} test images in {input_dir}")

    if args.skip_run:
        print("\n[5/6] Skipped (--skip-run)")
        print("\n[6/6] Skipped (--skip-run)")
        print("\nDone (validation only)")
        return

    # ── Step 5: Run inference ────────────────────────────────────────────
    print(f"\n[5/6] Running inference (timeout: {args.timeout}s)")
    print(f"  Command: python3 run.py --input {input_dir} --output {output_path}")

    start = time.time()
    try:
        result = subprocess.run(
            ["python3", "run.py", "--input", str(input_dir), "--output", str(output_path)],
            cwd=str(submission_dir),
            timeout=args.timeout,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  FAIL: Exit code {result.returncode} after {elapsed:.1f}s")
            if result.stderr:
                print("  STDERR (last 30 lines):")
                for line in result.stderr.strip().split("\n")[-30:]:
                    print(f"    {line}")
            if result.stdout:
                print("  STDOUT (last 10 lines):")
                for line in result.stdout.strip().split("\n")[-10:]:
                    print(f"    {line}")
            exit(1)

        print(f"  PASS: Completed in {elapsed:.1f}s (limit: {args.timeout}s)")
        if elapsed > args.timeout * 0.8:
            print(f"  WARNING: Used {elapsed/args.timeout*100:.0f}% of timeout budget")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  FAIL: Timed out after {elapsed:.1f}s (limit: {args.timeout}s)")
        exit(1)

    # ── Step 6: Validate output ──────────────────────────────────────────
    print("\n[6/6] Output validation")
    valid, issues = validate_output(output_path)
    for issue in issues:
        prefix = "  " if issue.startswith(("INFO:", "WARNING:")) else "  "
        print(f"{prefix}{issue}")

    if not valid:
        print("  FAIL: Output validation failed")
        exit(1)
    print("  PASS: Output format valid")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED - READY TO SUBMIT")
    print(f"  Inference time: {elapsed:.1f}s / {args.timeout}s")
    print(f"  ZIP size: {zip_size_mb:.1f}MB / 420MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
