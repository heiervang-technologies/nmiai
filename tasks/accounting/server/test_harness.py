"""
Validation harness for accounting agent.
Replays real prompts through the full agent pipeline with a cheap/local LLM
and a mock Tripletex backend. Measures whether the agent:
  1) Classifies the task correctly (planner)
  2) Calls the right typed tools (not generic_api_call for everything)
  3) Action guards fire correctly (auto-balance, date injection, VAT fixes)
  4) Completes without circuit breaker / excessive errors

Usage:
  # Planner-only (no LLM, instant):
  uv run python test_harness.py --planner-only

  # Full replay with local model:
  uv run python test_harness.py --replay --model qwen3.5-27b --base-url http://192.168.8.170:30184/v1

  # Single family:
  uv run python test_harness.py --replay --family invoice --limit 5

  # List families and prompt counts:
  uv run python test_harness.py --list
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("harness")

LOG_DIR = Path(os.environ.get("LOG_DIR", "/tmp/accounting-logs"))
SUMMARY_FILE = LOG_DIR / "summary.jsonl"

# Expected entity/call patterns per family
FAMILY_EXPECTATIONS = {
    "invoice": {"entities": {"invoice": 1, "customer": 1}},
    "supplier": {"entities": {"supplier": 1}},
    "employee": {"entities": {"employee": 1}},
    "travel_expense": {"path_contains": ["/travelExpense"]},
    "project": {"entities": {"project": 1}},
    "voucher": {"entities": {"voucher": 1}},
    "salary": {"path_contains": ["/salary"]},
    "department": {"entities": {"department": 1}},
    "product": {"entities": {"product": 1}},
    "customer": {"entities": {"customer": 1}},
    "bank_reconciliation": {"path_contains": ["/bank/reconciliation"]},
    "cost_analysis": {"entities": {"activity": 1}},
}

_REVERSAL_PATTERN = re.compile(
    r"\b(reversal|reverse|reversed|returnert|devuelto|retourné|zurückgebucht)\b", re.IGNORECASE
)


def _check_family_expectations(family: str, prompt: str, assertions: dict, tool_calls: list[str]) -> list[dict]:
    """Compare actual results against FAMILY_EXPECTATIONS. Returns list of mismatches."""
    mismatches = []
    is_reversal = bool(_REVERSAL_PATTERN.search(prompt))

    if is_reversal and family == "invoice":
        # Payment reversal: expect exactly 1 invoice, must have /:payment call
        actual_invoices = assertions.get("entity_counts", {}).get("invoice", 0)
        if actual_invoices != 1:
            mismatches.append({"check": "reversal_invoice_count", "expected": 1, "got": actual_invoices, "severity": "error"})
        has_payment = any("/:payment" in tc for tc in tool_calls)
        if not has_payment:
            mismatches.append({"check": "reversal_payment_call", "expected": "/:payment call", "got": "missing", "severity": "error"})
        return mismatches

    spec = FAMILY_EXPECTATIONS.get(family)
    if not spec:
        return mismatches

    # Check expected entity counts
    for entity, expected_count in spec.get("entities", {}).items():
        actual = assertions.get("entity_counts", {}).get(entity, 0)
        if actual < expected_count:
            mismatches.append({"check": f"{entity}_count", "expected": f">={expected_count}", "got": actual, "severity": "warning"})

    # Check expected path patterns
    for path_pattern in spec.get("path_contains", []):
        found = any(path_pattern in tc for tc in tool_calls)
        if not found:
            mismatches.append({"check": f"path_{path_pattern}", "expected": f"call to {path_pattern}", "got": "missing", "severity": "warning"})

    return mismatches


def load_prompts(family: str = None, limit: int = None, timestamps: list[str] = None) -> list[dict]:
    """Load real prompts from competition logs."""
    prompts = []
    if not SUMMARY_FILE.exists():
        log.error(f"No summary file at {SUMMARY_FILE}")
        return []

    for line in SUMMARY_FILE.read_text().strip().split("\n"):
        summary = json.loads(line)
        ts = summary["ts"]
        if timestamps and ts not in timestamps:
            continue
        detail_file = LOG_DIR / f"{ts}.json"
        if not detail_file.exists():
            continue
        detail = json.loads(detail_file.read_text())
        entry = {
            "ts": ts,
            "prompt": detail["prompt"],
            "files": detail.get("files", []),
            "expected_family": summary.get("family"),
            "expected_confidence": summary.get("confidence"),
            "logged_api_calls": summary.get("api_calls", 0),
            "logged_api_errors": summary.get("api_errors", 0),
            "logged_elapsed": summary.get("elapsed", 0),
        }
        if family and entry["expected_family"] != family:
            continue
        prompts.append(entry)
        if limit and len(prompts) >= limit:
            break
    return prompts


def list_families():
    """Show distribution of families in logs."""
    if not SUMMARY_FILE.exists():
        print("No summary file found")
        return
    counts = {}
    for line in SUMMARY_FILE.read_text().strip().split("\n"):
        s = json.loads(line)
        fam = s.get("family", "unknown")
        counts[fam] = counts.get(fam, 0) + 1
    print(f"\n{'Family':<25} {'Count':>5}  {'Avg Errors':>10}")
    print("-" * 45)
    for fam, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{fam:<25} {count:>5}")
    print(f"\nTotal: {sum(counts.values())} prompts")


# ── Planner-only tests ──

def test_planner(prompts: list[dict]):
    """Test planner classification without LLM (keyword-only)."""
    # Force no LLM fallback by temporarily unsetting API keys
    saved_keys = {}
    for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
        if key in os.environ:
            saved_keys[key] = os.environ.pop(key)

    try:
        # Import fresh to pick up env changes
        from planner import plan_task
        results = {"match": 0, "mismatch": 0, "errors": []}

        for p in prompts:
            try:
                plan = plan_task(p["prompt"])
                actual = plan["family"]
                expected = p["expected_family"]
                if actual == expected:
                    results["match"] += 1
                else:
                    results["mismatch"] += 1
                    results["errors"].append({
                        "ts": p["ts"],
                        "prompt": p["prompt"][:100],
                        "expected": expected,
                        "got": actual,
                        "confidence": plan["confidence"],
                        "method": plan["method"],
                    })
            except Exception as e:
                results["errors"].append({"ts": p["ts"], "error": str(e)})

        total = results["match"] + results["mismatch"]
        pct = (results["match"] / total * 100) if total else 0
        print(f"\n=== Planner Classification ===")
        print(f"Correct: {results['match']}/{total} ({pct:.1f}%)")
        if results["errors"]:
            print(f"\nMismatches:")
            for e in results["errors"]:
                if "error" in e:
                    print(f"  [{e['ts']}] ERROR: {e['error']}")
                else:
                    print(f"  [{e['ts']}] expected={e['expected']} got={e['got']} ({e['method']}/{e['confidence']})")
                    print(f"    prompt: {e['prompt']}")
        return results
    finally:
        os.environ.update(saved_keys)


# ── Full replay with LLM + mock Tripletex ──

async def replay_single(prompt_entry: dict, mock_base_url: str) -> dict:
    """Replay a single prompt through the full agent."""
    from tripletex_client import TripletexClient
    from agent import run_agent
    from planner import plan_task
    from mock_tripletex import get_state

    client = TripletexClient(base_url=mock_base_url, session_token="mock-token")

    start = time.time()
    result = {}
    plan = None
    error = None
    assertions = {}
    expectation_mismatches = []

    try:
        plan = plan_task(prompt_entry["prompt"])
        files = None
        if prompt_entry.get("files"):
            files = [f for f in prompt_entry["files"] if f.get("content_base64")]
        result = await run_agent(client, prompt_entry["prompt"], files, playbook=plan.get("playbook"))
    except Exception as e:
        error = str(e)
        log.error(f"Replay failed for {prompt_entry['ts']}: {e}")
    finally:
        stats = client.get_stats()
        elapsed = time.time() - start
        await client.close()

    # Post-run assertions from mock state
    mock_st = get_state()
    assertions = mock_st.get_assertions()
    tool_calls = [c["path"] for c in stats.get("calls", [])]

    # Check family-specific expectations
    family = plan["family"] if plan else prompt_entry["expected_family"]
    expectation_mismatches = _check_family_expectations(
        family, prompt_entry["prompt"], assertions, tool_calls
    )

    # Scorer estimate
    from scorer_checks import estimate_score
    scorer = estimate_score(family, prompt_entry["prompt"], mock_st)

    return {
        "ts": prompt_entry["ts"],
        "expected_family": prompt_entry["expected_family"],
        "actual_family": plan["family"] if plan else None,
        "family_match": (plan["family"] == prompt_entry["expected_family"]) if plan else False,
        "api_calls": stats["total_calls"],
        "api_errors": stats["errors_4xx"],
        "elapsed": round(elapsed, 1),
        "error": error,
        "logged_api_calls": prompt_entry["logged_api_calls"],
        "logged_api_errors": prompt_entry["logged_api_errors"],
        "tool_calls": tool_calls,
        "final_message": (result.get("final_message") or "")[:200] if isinstance(result, dict) else "",
        "assertions": assertions,
        "expectation_mismatches": expectation_mismatches,
        "scorer": scorer,
    }


async def run_replay(prompts: list[dict], mock_port: int = 9876):
    """Run full replay with mock Tripletex server."""
    import uvicorn
    from mock_tripletex import app as mock_app, reset_state

    # Start mock server in background
    config = uvicorn.Config(mock_app, host="127.0.0.1", port=mock_port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)  # Let server start

    mock_base_url = f"http://127.0.0.1:{mock_port}/v2"
    results = []

    try:
        for i, p in enumerate(prompts):
            reset_state()  # Fresh state per prompt
            log.info(f"[{i+1}/{len(prompts)}] {p['expected_family']}: {p['prompt'][:80]}...")
            r = await replay_single(p, mock_base_url)
            results.append(r)

            assertions = r.get("assertions", {})
            assert_errs = assertions.get("error_count", 0)
            assert_warns = assertions.get("warning_count", 0)
            expect_misses = r.get("expectation_mismatches", [])
            scorer = r.get("scorer", {})

            has_issues = r["error"] or r["api_errors"] > 0 or assert_errs > 0 or any(m["severity"] == "error" for m in expect_misses)
            status = "ERR" if has_issues else "OK"
            fam_ok = "FAM-OK" if r["family_match"] else "FAM-MISS"

            extra = ""
            if assert_errs or assert_warns:
                extra += f" assert:{assert_errs}E/{assert_warns}W"
            if expect_misses:
                extra += f" expect-miss:{len(expect_misses)}"
            if scorer:
                sc_pct = scorer.get("correctness", 0)
                extra += f" score:{scorer.get('points_earned', 0)}/{scorer.get('max_points', '?')}={sc_pct:.0%}"

            print(f"  [{status}] [{fam_ok}] calls={r['api_calls']} errors={r['api_errors']}{extra} elapsed={r['elapsed']}s")
            if r["error"]:
                print(f"    ERROR: {r['error'][:150]}")
            for issue in assertions.get("issues", []):
                print(f"    ASSERT-{issue['severity'].upper()}: {issue['issue']}")
            for mm in expect_misses:
                print(f"    EXPECT-{mm['severity'].upper()}: {mm['check']} expected={mm['expected']} got={mm['got']}")
            for si in scorer.get("issues", []):
                print(f"    SCORER: {si}")
            for check in scorer.get("checks", []):
                if not check["passed"]:
                    print(f"    SCORE-FAIL: {check['label']} ({check['detail']})")
    finally:
        server.should_exit = True
        await server_task

    # Summary
    total = len(results)
    clean = sum(1 for r in results if not r["error"] and r["api_errors"] == 0)
    fam_match = sum(1 for r in results if r["family_match"])
    errors = sum(1 for r in results if r["error"])
    assert_errors = sum(r.get("assertions", {}).get("error_count", 0) for r in results)
    assert_warnings = sum(r.get("assertions", {}).get("warning_count", 0) for r in results)
    expect_misses = sum(len(r.get("expectation_mismatches", [])) for r in results)

    print(f"\n=== Replay Summary ===")
    print(f"Total: {total}")
    print(f"Clean (no errors): {clean}/{total} ({clean/total*100:.0f}%)")
    print(f"Family match: {fam_match}/{total} ({fam_match/total*100:.0f}%)")
    print(f"Failures: {errors}")
    print(f"Assertion errors/warnings: {assert_errors}E / {assert_warnings}W")
    print(f"Expectation mismatches: {expect_misses}")
    avg_elapsed = sum(r["elapsed"] for r in results) / total if total else 0
    print(f"Avg elapsed: {avg_elapsed:.1f}s")

    # Per-family breakdown
    families = {}
    for r in results:
        fam = r["expected_family"] or "unknown"
        if fam not in families:
            families[fam] = {"total": 0, "clean": 0, "errors": 0, "assert_errs": 0, "expect_miss": 0, "scorer_sum": 0.0}
        families[fam]["total"] += 1
        if not r["error"] and r["api_errors"] == 0:
            families[fam]["clean"] += 1
        if r["error"]:
            families[fam]["errors"] += 1
        families[fam]["assert_errs"] += r.get("assertions", {}).get("error_count", 0)
        families[fam]["expect_miss"] += len(r.get("expectation_mismatches", []))
        families[fam]["scorer_sum"] += r.get("scorer", {}).get("correctness", 0)

    print(f"\n{'Family':<25} {'Clean':>6} {'Total':>6} {'Rate':>6} {'AssrtE':>7} {'AvgScr':>7}")
    print("-" * 65)
    for fam in sorted(families, key=lambda f: families[f]["scorer_sum"] / max(families[f]["total"], 1)):
        f = families[fam]
        rate = f["clean"] / f["total"] * 100 if f["total"] else 0
        avg_score = f["scorer_sum"] / f["total"] * 100 if f["total"] else 0
        print(f"{fam:<25} {f['clean']:>6} {f['total']:>6} {rate:>5.0f}% {f['assert_errs']:>7} {avg_score:>6.0f}%")

    # Write results to file (family-specific to avoid overwrites)
    family_tag = results[0]["expected_family"] if results and results[0].get("expected_family") else "mixed"
    out_file = LOG_DIR / f"replay_{family_tag}.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    # Also append to aggregate
    agg_file = LOG_DIR / "replay_all.jsonl"
    with open(agg_file, "a") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nDetailed results: {out_file}")
    return results


def load_adversarial_prompts(family: str = None, limit: int = None) -> list[dict]:
    """Load adversarial prompts and convert to replay format."""
    from adversarial_prompts import get_prompts
    adv = get_prompts(family=family)
    prompts = []
    for i, ap in enumerate(adv):
        prompts.append({
            "ts": f"adversarial_{i:03d}",
            "prompt": ap["prompt"],
            "files": [],
            "expected_family": ap["family"],
            "expected_confidence": "high",
            "logged_api_calls": 0,
            "logged_api_errors": 0,
            "logged_elapsed": 0,
            "adversarial_meta": {
                "language": ap["language"],
                "difficulty": ap["difficulty"],
                "failure_mode_tested": ap["failure_mode_tested"],
                "expected_fields": ap.get("expected_fields", {}),
            },
        })
        if limit and len(prompts) >= limit:
            break
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Accounting agent validation harness")
    parser.add_argument("--planner-only", action="store_true", help="Only test planner (no LLM)")
    parser.add_argument("--replay", action="store_true", help="Full replay with LLM + mock Tripletex")
    parser.add_argument("--replay-adversarial", action="store_true", help="Replay adversarial prompts")
    parser.add_argument("--list", action="store_true", help="List families and counts")
    parser.add_argument("--family", type=str, help="Filter to single family")
    parser.add_argument("--limit", type=int, help="Max prompts to test")
    parser.add_argument("--model", type=str, help="LLM model name for replay")
    parser.add_argument("--base-url", type=str, help="LLM API base URL")
    parser.add_argument("--mock-port", type=int, default=9876, help="Port for mock Tripletex server")
    parser.add_argument("--timestamps", type=str, help="Comma-separated timestamps to replay specific prompts")
    args = parser.parse_args()

    if args.list:
        list_families()
        return

    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url

    if args.replay_adversarial:
        prompts = load_adversarial_prompts(family=args.family, limit=args.limit)
        if not prompts:
            print("No adversarial prompts found.")
            return
        print(f"Loaded {len(prompts)} adversarial prompts" + (f" (family={args.family})" if args.family else ""))
        asyncio.run(run_replay(prompts, mock_port=args.mock_port))
        return

    ts_list = args.timestamps.split(",") if args.timestamps else None
    prompts = load_prompts(family=args.family, limit=args.limit, timestamps=ts_list)
    if not prompts:
        print("No prompts found. Check LOG_DIR.")
        return

    print(f"Loaded {len(prompts)} prompts" + (f" (family={args.family})" if args.family else ""))

    if args.planner_only:
        test_planner(prompts)
    elif args.replay:
        asyncio.run(run_replay(prompts, mock_port=args.mock_port))
    else:
        # Default: planner test
        test_planner(prompts)


if __name__ == "__main__":
    main()
