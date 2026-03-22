"""
Standalone output validator — matches the REAL competition server expectations.

Usage: python3 validate_output.py predictions.json

Competition expects FLAT COCO-style format:
[
  {"image_id": 42, "bbox": [x, y, w, h], "category_id": 42, "score": 0.95},
  ...
]

Per official docs:
- image_id: INTEGER (extracted from filename, e.g., img_00042.jpg -> 42)
- bbox: [x, y, width, height] in COCO format
- category_id: int 0-356 (356 = unknown_product)
- score: float 0-1
"""
import json
import pathlib


REQUIRED_FIELDS = {"image_id", "bbox", "category_id", "score"}


def validate(path: pathlib.Path) -> tuple[bool, list[str]]:
    issues = []
    data = json.loads(path.read_text())

    if not isinstance(data, list):
        return False, [f"Root must be JSON array, got {type(data).__name__}"]

    if len(data) == 0:
        issues.append("WARNING: Empty predictions array (0 detections)")
        return True, issues

    # Check first entry for format
    first = data[0]
    first_keys = set(first.keys())

    # REJECT nested format explicitly
    if "predictions" in first_keys:
        return False, [
            "WRONG FORMAT: Found 'predictions' key - this is nested format.",
            "Server expects FLAT format: each detection is its own entry.",
            "Expected: [{image_id, bbox, category_id, score}, ...]",
            f"Got: [{{{', '.join(first_keys)}}}, ...]",
        ]

    # Check for wrong field names
    if "confidence" in first_keys and "score" not in first_keys:
        return False, [
            "WRONG FIELD NAME: Found 'confidence' - server expects 'score'.",
            "Change 'confidence' to 'score' in your output.",
        ]

    # Check required fields
    missing = REQUIRED_FIELDS - first_keys
    if missing:
        return False, [
            f"Entry 0 missing fields: {', '.join(sorted(missing))}",
            f"Required: {', '.join(sorted(REQUIRED_FIELDS))}",
            f"Got: {', '.join(sorted(first_keys))}",
        ]

    # Validate all entries (sample first 50 + last 10)
    check_indices = list(range(min(50, len(data)))) + list(
        range(max(0, len(data) - 10), len(data))
    )
    check_indices = sorted(set(check_indices))

    for i in check_indices:
        entry = data[i]

        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in entry:
                issues.append(f"[{i}] missing '{field}'")
                continue

        # image_id: official docs say int, but server accepts both string and int
        img_id = entry.get("image_id")
        if isinstance(img_id, str) and i < 3:
            issues.append(
                f"WARNING: [{i}] image_id is string '{img_id}' - docs say int "
                f"(e.g., img_00042.jpg -> 42). Server accepts both but int is canonical."
            )
        elif not isinstance(img_id, (int, str)):
            issues.append(
                f"[{i}] image_id must be int or string, got {type(img_id).__name__}: {img_id}"
            )

        # bbox: must be list of 4 numbers
        bbox = entry.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            issues.append(f"[{i}] bbox must be [x,y,w,h] (4 elements), got {bbox}")
        elif not all(isinstance(v, (int, float)) for v in bbox):
            issues.append(f"[{i}] bbox values must be numbers, got {bbox}")
        else:
            x, y, w, h = bbox
            if w < 0 or h < 0:
                issues.append(f"[{i}] bbox has negative width/height: w={w}, h={h}")
            if x < 0 or y < 0:
                issues.append(
                    f"WARNING: [{i}] bbox has negative coordinates: x={x}, y={y}"
                )

        # category_id: must be int 0-356
        cid = entry.get("category_id")
        if not isinstance(cid, int):
            issues.append(
                f"[{i}] category_id must be int, got {type(cid).__name__}: {cid}"
            )
        elif not (0 <= cid <= 356):
            issues.append(f"[{i}] category_id={cid} out of range (need 0-356)")

        # score: must be float 0-1
        score = entry.get("score")
        if not isinstance(score, (int, float)):
            issues.append(
                f"[{i}] score must be number, got {type(score).__name__}: {score}"
            )
        elif not (0 <= score <= 1):
            issues.append(f"[{i}] score={score} out of range (need 0.0-1.0)")

        if len(issues) > 30:
            issues.append("... truncated (too many issues)")
            break

    # Summary stats
    unique_images = len(set(e.get("image_id", "") for e in data))
    categories = set(e.get("category_id", -1) for e in data)
    scores = [
        e.get("score", 0) for e in data if isinstance(e.get("score"), (int, float))
    ]
    str_ids = sum(1 for e in data if isinstance(e.get("image_id"), str))
    int_ids = sum(1 for e in data if isinstance(e.get("image_id"), int))

    print(f"  INFO: {len(data)} total detections, {unique_images} unique images")
    print(f"  INFO: image_id types: {int_ids} int, {str_ids} string")
    if scores:
        print(f"  INFO: Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(
        f"  INFO: Categories used: {len(categories)} (range {min(categories)}-{max(categories)})"
    )

    errors = [i for i in issues if not i.startswith("WARNING:")]
    ok = len(errors) == 0
    return ok, issues


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("file", type=pathlib.Path)
    args = p.parse_args()
    ok, issues = validate(args.file)
    for i in issues:
        print(f"  {'WARN' if i.startswith('WARNING:') else 'FAIL'}: {i}")
    if ok:
        print("  PASS: Output format valid")
    else:
        print("  FAIL: Output format INVALID")
    exit(0 if ok else 1)
