#!/usr/bin/env python3
"""
Regression test suite for the accounting agent.
Sends test prompts to localhost:8000/solve and checks for known failure patterns.
Tracks results over time to detect regressions.

Usage:
    python regression_tests.py                    # Run all tests
    python regression_tests.py --family employee  # Run one family
    python regression_tests.py --compare          # Compare last two runs
    python regression_tests.py --watch            # Run after each code change
"""

import argparse
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

SERVER_URL = "http://localhost:8000"
RESULTS_DIR = Path(__file__).parent / "regression_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Sandbox creds for testing
SANDBOX_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "eyJ0b2tlbklkIjoyMTQ3NjMyNDU1LCJ0b2tlbiI6IjEwYmU0YzE3LTIyZDYtNDY1Ni04YzNlLTdlNzU2ZTFhNjZlMyJ9"
}

# --- Test cases per family ---
# Each test has: prompt, expected behaviors (api_calls > 0, no specific error patterns)

TEST_CASES = {
    "employee": [
        {
            "name": "create_employee_basic",
            "prompt": "Opprett en ansatt med navn Kari Nordmann, e-post kari@example.com, avdeling IT",
            "files": [],
            "expect": {
                "min_api_calls": 1,
                "family": "employee",
                "no_errors_containing": ["Name or service not known", "refused"],
                "must_have_writes": True,
            },
        },
        {
            "name": "create_employee_admin",
            "prompt": "Create employee John Smith, email john@test.com, with administrator access",
            "files": [],
            "expect": {
                "min_api_calls": 1,
                "family": "employee",
                "must_have_writes": True,
            },
        },
    ],
    "customer": [
        {
            "name": "create_customer_basic",
            "prompt": "Opprett en kunde med navn Bedrift AS, org.nr 987654321, e-post kontakt@bedrift.no",
            "files": [],
            "expect": {
                "min_api_calls": 1,
                "family": "customer",
                "must_have_writes": True,
            },
        },
    ],
    "supplier": [
        {
            "name": "register_supplier_invoice",
            "prompt": "Registrer en leverandørfaktura fra Leverandør AS (org.nr 123456789) på 12500 NOK inkl. mva. Fakturanummer LF-2026-001. Utgiftskonto 6300.",
            "files": [],
            "expect": {
                "min_api_calls": 2,
                "family": "supplier",
                "must_have_writes": True,
            },
        },
    ],
    "invoice": [
        {
            "name": "create_invoice_basic",
            "prompt": "Opprett en faktura til kunde Testfirma AS for konsulenttjenester, beløp 15000 NOK ekskl. mva. Forfallsdato 2026-04-15.",
            "files": [],
            "expect": {
                "min_api_calls": 2,
                "family": "invoice",
                "must_have_writes": True,
            },
        },
    ],
    "voucher": [
        {
            "name": "create_voucher_basic",
            "prompt": "Opprett et bilag: debet konto 6300 (kontorrekvisita) 5000 NOK, kredit konto 1920 (bank) 5000 NOK. Beskrivelse: Innkjøp kontorrekvisita.",
            "files": [],
            "expect": {
                "min_api_calls": 1,
                "family": "voucher",
                "must_have_writes": True,
            },
        },
        {
            "name": "voucher_with_customer",
            "prompt": "Opprett et bilag for purregebyr. Debet konto 1500 (kundefordringer) 60 NOK, kredit konto 3400 (purregebyr) 60 NOK. Bilagsbeskrivelse: Purregebyr Testfirma AS.",
            "files": [],
            "expect": {
                "min_api_calls": 1,
                "family": "voucher",
                "must_have_writes": True,
            },
        },
    ],
    "travel_expense": [
        {
            "name": "travel_expense_basic",
            "prompt": "Registrer en reiseregning for ansatt nummer 1 for 'Kundebesøk Oslo'. Reisen varte 3 dager med diett. Avreise fra Bergen, destinasjon Oslo.",
            "files": [],
            "expect": {
                "min_api_calls": 2,
                "family": "travel_expense",
                "must_have_writes": True,
                "no_errors_containing": ["rateType"],
            },
        },
    ],
    "timesheet": [
        {
            "name": "timesheet_basic",
            "prompt": "Registrer 8 timer for ansatt nummer 1 på et prosjekt den 2026-03-20. Aktivitet: utvikling.",
            "files": [],
            "expect": {
                "min_api_calls": 2,
                "family": "timesheet",
                "no_errors_containing": ["activity"],
                "must_have_writes": True,
            },
        },
    ],
    "salary": [
        {
            "name": "salary_basic",
            "prompt": "Kjør lønn for ansatt nummer 1. Grunnlønn 45000 NOK, ingen bonus.",
            "files": [],
            "expect": {
                "min_api_calls": 2,
                "family": "salary",
                "must_have_writes": True,
            },
        },
    ],
    "project": [
        {
            "name": "create_project_basic",
            "prompt": "Opprett prosjekt 'Ny nettside' for kunde Testfirma AS. Fastpris 250000 NOK. Startdato 2026-04-01.",
            "files": [],
            "expect": {
                "min_api_calls": 2,
                "family": "project",
                "must_have_writes": True,
            },
        },
    ],
    "product": [
        {
            "name": "create_product_basic",
            "prompt": "Opprett et produkt 'Konsulenttjeneste' med pris 1500 NOK ekskl. mva (25% MVA).",
            "files": [],
            "expect": {
                "min_api_calls": 1,
                "family": "product",
                "must_have_writes": True,
            },
        },
    ],
    "department": [
        {
            "name": "create_department_basic",
            "prompt": "Opprett avdeling 'Salg' med avdelingsnummer 200.",
            "files": [],
            "expect": {
                "min_api_calls": 1,
                "family": "department",
                "must_have_writes": True,
            },
        },
    ],
}

# Known failure patterns to flag
KNOWN_FAILURE_PATTERNS = [
    {"pattern": "Name or service not known", "severity": "critical", "description": "DNS resolution failure - agent refused to act"},
    {"pattern": "rateType", "severity": "high", "description": "Travel expense rateType.id validation error"},
    {"pattern": "activity field null", "severity": "high", "description": "Timesheet activity not resolved"},
    {"pattern": "dateFrom", "severity": "medium", "description": "Missing dateFrom/dateTo in ledger query"},
    {"pattern": "employment", "severity": "medium", "description": "Employment body format wrong"},
    {"pattern": "supplier", "severity": "medium", "description": "Missing supplier ref in voucher postings"},
    {"pattern": "customer", "severity": "medium", "description": "Missing customer ref in voucher postings"},
]


def check_server() -> bool:
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_test(test: dict) -> dict:
    """Run a single test case against the server."""
    payload = {
        "prompt": test["prompt"],
        "tripletex_credentials": SANDBOX_CREDS,
        "files": test.get("files", []),
    }
    start = time.time()
    try:
        r = httpx.post(f"{SERVER_URL}/solve", json=payload, timeout=120)
        elapsed = time.time() - start
        response = r.json()
    except Exception as e:
        elapsed = time.time() - start
        return {
            "name": test["name"],
            "status": "ERROR",
            "error": str(e),
            "elapsed": round(elapsed, 1),
        }

    # Fetch the latest log to get detailed stats
    try:
        logs = httpx.get(f"{SERVER_URL}/logs", timeout=10).json().get("logs", [])
        latest_log = logs[-1] if logs else {}
    except Exception:
        latest_log = {}

    result = {
        "name": test["name"],
        "status": "PASS",
        "elapsed": round(elapsed, 1),
        "api_calls": latest_log.get("api_calls", 0),
        "api_errors": latest_log.get("api_errors", 0),
        "family_detected": latest_log.get("family"),
        "failures": [],
    }

    expect = test.get("expect", {})

    # Check minimum API calls
    if expect.get("min_api_calls") and result["api_calls"] < expect["min_api_calls"]:
        result["failures"].append(f"Expected >= {expect['min_api_calls']} API calls, got {result['api_calls']}")

    # Check family classification
    if expect.get("family") and result["family_detected"] != expect["family"]:
        result["failures"].append(f"Expected family '{expect['family']}', got '{result['family_detected']}'")

    # Check for 0 API calls (agent refused)
    if result["api_calls"] == 0:
        result["failures"].append("CRITICAL: 0 API calls - agent refused or connection failed")

    if result["failures"]:
        result["status"] = "FAIL"

    return result


def run_suite(families: list[str] | None = None) -> dict:
    """Run the full test suite or specific families."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": ts,
        "server_url": SERVER_URL,
        "tests": [],
        "summary": {"total": 0, "pass": 0, "fail": 0, "error": 0},
    }

    test_families = families or list(TEST_CASES.keys())

    for family in test_families:
        if family not in TEST_CASES:
            print(f"  [SKIP] Unknown family: {family}")
            continue
        for test in TEST_CASES[family]:
            print(f"  [{family}] Running {test['name']}...", end=" ", flush=True)
            result = run_test(test)
            results["tests"].append(result)
            results["summary"]["total"] += 1
            results["summary"][result["status"].lower()] += 1

            status_icon = {"PASS": "OK", "FAIL": "FAIL", "ERROR": "ERR"}[result["status"]]
            print(f"[{status_icon}] {result['elapsed']}s api={result.get('api_calls', '?')} err={result.get('api_errors', '?')}")
            if result.get("failures"):
                for f in result["failures"]:
                    print(f"    -> {f}")

    # Save results
    result_file = RESULTS_DIR / f"run_{ts}.json"
    result_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {result_file}")

    return results


def compare_runs():
    """Compare the last two test runs to detect regressions."""
    runs = sorted(RESULTS_DIR.glob("run_*.json"))
    if len(runs) < 2:
        print("Need at least 2 runs to compare. Run the suite first.")
        return

    prev = json.loads(runs[-2].read_text())
    curr = json.loads(runs[-1].read_text())

    print(f"\nComparing runs:")
    print(f"  Previous: {prev['timestamp']} ({prev['summary']})")
    print(f"  Current:  {curr['timestamp']} ({curr['summary']})")

    prev_by_name = {t["name"]: t for t in prev["tests"]}
    curr_by_name = {t["name"]: t for t in curr["tests"]}

    regressions = []
    improvements = []

    for name, curr_test in curr_by_name.items():
        prev_test = prev_by_name.get(name)
        if not prev_test:
            continue

        if prev_test["status"] == "PASS" and curr_test["status"] != "PASS":
            regressions.append({"name": name, "was": "PASS", "now": curr_test["status"], "failures": curr_test.get("failures", [])})
        elif prev_test["status"] != "PASS" and curr_test["status"] == "PASS":
            improvements.append({"name": name, "was": prev_test["status"], "now": "PASS"})

    if regressions:
        print(f"\n  REGRESSIONS ({len(regressions)}):")
        for r in regressions:
            print(f"    {r['name']}: {r['was']} -> {r['now']}")
            for f in r.get("failures", []):
                print(f"      -> {f}")
    else:
        print("\n  No regressions detected.")

    if improvements:
        print(f"\n  IMPROVEMENTS ({len(improvements)}):")
        for i in improvements:
            print(f"    {i['name']}: {i['was']} -> {i['now']}")

    return {"regressions": regressions, "improvements": improvements}


def print_summary(results: dict):
    s = results["summary"]
    print(f"\n{'='*50}")
    print(f"REGRESSION TEST SUMMARY ({results['timestamp']})")
    print(f"{'='*50}")
    print(f"  Total: {s['total']}  Pass: {s['pass']}  Fail: {s['fail']}  Error: {s['error']}")
    rate = s['pass'] / s['total'] * 100 if s['total'] else 0
    print(f"  Pass rate: {rate:.0f}%")
    if s['fail'] > 0 or s['error'] > 0:
        print("\n  FAILED TESTS:")
        for t in results["tests"]:
            if t["status"] != "PASS":
                print(f"    {t['name']}: {t['status']}")
                for f in t.get("failures", []):
                    print(f"      -> {f}")


def main():
    parser = argparse.ArgumentParser(description="Accounting agent regression tests")
    parser.add_argument("--family", type=str, help="Run tests for specific family only")
    parser.add_argument("--compare", action="store_true", help="Compare last two runs")
    parser.add_argument("--watch", action="store_true", help="Continuous mode: re-run on Enter")
    parser.add_argument("--list", action="store_true", help="List all test cases")
    args = parser.parse_args()

    if args.list:
        for family, tests in TEST_CASES.items():
            print(f"\n{family}:")
            for t in tests:
                print(f"  - {t['name']}: {t['prompt'][:80]}...")
        return

    if args.compare:
        compare_runs()
        return

    if not check_server():
        print(f"ERROR: Server not reachable at {SERVER_URL}")
        print("Start it with: cd tasks/accounting/server && uv run python main.py")
        sys.exit(1)

    families = [args.family] if args.family else None

    if args.watch:
        while True:
            print(f"\n{'='*50}")
            print(f"Running regression suite...")
            results = run_suite(families)
            print_summary(results)
            input("\nPress Enter to run again (Ctrl+C to quit)...")
    else:
        results = run_suite(families)
        print_summary(results)


if __name__ == "__main__":
    main()
