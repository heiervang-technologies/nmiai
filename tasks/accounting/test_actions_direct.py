#!/usr/bin/env python3
"""
Direct action-layer tests — NO LLM calls, NO OpenRouter credits.
Tests the typed actions directly with hardcoded args against the sandbox API.
Validates that our action code produces correct API calls.

Usage: cd tasks/accounting/server && uv run python ../test_actions_direct.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "server"))

from tripletex_client import TripletexClient
from actions import ACTIONS

SANDBOX_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "eyJ0b2tlbklkIjoyMTQ3NjMyNDU1LCJ0b2tlbiI6IjEwYmU0YzE3LTIyZDYtNDY1Ni04YzNlLTdlNzU2ZTFhNjZlMyJ9",
}

# Direct action test cases — each calls the action function with specific args
TEST_CASES = [
    {
        "name": "discover_sandbox",
        "action": "discover_sandbox",
        "args": {},
        "expect_keys": ["departments", "employees"],
    },
    {
        "name": "create_customer",
        "action": "create_customer",
        "args": {"name": f"TestCo-{int(time.time())}", "email": "test@co.no", "organizationNumber": "999888777"},
        "expect_keys": ["value"],
    },
    {
        "name": "create_department",
        "action": "create_department",
        "args": {"name": f"TestDept-{int(time.time())}", "departmentNumber": str(int(time.time()) % 10000)},
        "expect_keys": ["value"],
    },
    {
        "name": "create_employee",
        "action": "create_employee",
        "args": {"firstName": "Test", "lastName": f"Employee{int(time.time())}", "email": f"test{int(time.time())}@example.com"},
        "expect_keys": ["value"],
    },
    {
        "name": "create_product",
        "action": "create_product",
        "args": {"name": f"TestProd-{int(time.time())}", "priceExcludingVat": 999.0},
        "expect_keys": ["value"],
    },
    {
        "name": "create_voucher",
        "action": "create_voucher",
        "args": {
            "description": f"Test voucher {int(time.time())}",
            "postings": [
                {"accountNumber": 6300, "amountGross": 1000.0},
                {"accountNumber": 1920, "amountGross": -1000.0},
            ],
        },
        "expect_keys": ["value"],
    },
    {
        "name": "create_supplier",
        "action": "create_supplier",
        "args": {"name": f"TestSupplier-{int(time.time())}"},
        "expect_keys": ["value"],
    },
    {
        "name": "process_salary",
        "action": "process_salary",
        "args": {"baseSalary": 40000, "bonus": 0},
        "expect_keys": ["value"],
    },
    {
        "name": "create_project",
        "action": "create_project",
        "args": {"name": f"TestProject-{int(time.time())}", "isInternal": True},
        "expect_keys": ["value"],
    },
    {
        "name": "setup_bank_account",
        "action": "setup_bank_account",
        "args": {},
        "expect_keys": ["status"],
    },
    {
        "name": "create_invoice",
        "action": "create_invoice",
        "args": {
            "customerName": f"InvoiceCust-{int(time.time())}",
            "orderLines": [{"description": "Test service", "count": 1, "unitPrice": 5000}],
        },
        "expect_keys": ["value"],
    },
]


async def run_test(test: dict, client: TripletexClient) -> dict:
    name = test["name"]
    action_fn = ACTIONS.get(test["action"])
    if not action_fn:
        return {"name": name, "status": "SKIP", "reason": f"Action {test['action']} not found"}

    start = time.time()
    try:
        result = await action_fn(client, test["args"])
        elapsed = time.time() - start
        calls = client.call_count
        errors = client.error_count

        # Check expected keys
        missing = [k for k in test.get("expect_keys", []) if k not in result]
        has_error = "error" in result and not any(k in result for k in test.get("expect_keys", []))

        status = "PASS"
        failures = []
        if missing:
            failures.append(f"Missing keys: {missing}")
            status = "FAIL"
        if has_error:
            failures.append(f"Error: {result['error'][:100]}")
            status = "FAIL"

        return {"name": name, "status": status, "elapsed": round(elapsed, 1),
                "calls": calls, "errors": errors, "failures": failures}
    except Exception as e:
        elapsed = time.time() - start
        return {"name": name, "status": "ERROR", "elapsed": round(elapsed, 1),
                "calls": client.call_count, "errors": client.error_count,
                "failures": [str(e)[:150]]}


async def main():
    print(f"Direct action tests against {SANDBOX_CREDS['base_url']}")
    print(f"NO LLM calls — testing action layer only\n")

    total = pass_count = fail_count = err_count = 0

    for test in TEST_CASES:
        # Fresh client per test to get accurate call counts
        client = TripletexClient(
            base_url=SANDBOX_CREDS["base_url"],
            session_token=SANDBOX_CREDS["session_token"],
        )
        try:
            result = await run_test(test, client)
        finally:
            await client.close()

        total += 1
        icon = {"PASS": "OK", "FAIL": "FAIL", "ERROR": "ERR", "SKIP": "SKIP"}[result["status"]]
        if result["status"] == "PASS":
            pass_count += 1
        elif result["status"] == "FAIL":
            fail_count += 1
        else:
            err_count += 1

        print(f"  [{icon:4s}] {result['name']:25s} {result.get('elapsed', 0):5.1f}s calls={result.get('calls', '?')} err={result.get('errors', '?')}")
        for f in result.get("failures", []):
            print(f"         -> {f}")

    print(f"\n{'='*50}")
    print(f"DIRECT ACTION TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Total: {total}  Pass: {pass_count}  Fail: {fail_count}  Error: {err_count}")
    rate = pass_count / total * 100 if total else 0
    print(f"  Pass rate: {rate:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
