"""
Scorer check map — replicates the competition's field-by-field k/n scoring locally.

Each family defines:
  - tier: multiplier (1, 2, or 3)
  - max_points: total points available
  - entity_checks: what entities must exist + which fields are checked
  - call_checks: required API call patterns
  - field_validators: functions that inspect mock state bodies for semantic correctness

Usage:
    from scorer_checks import estimate_score
    score = estimate_score(family, prompt, mock_state)
"""

import re
from datetime import datetime, timedelta


# ── Food keyword pattern (must match mock_tripletex.py) ──

_FOOD_KEYWORDS = re.compile(
    r"\b(mat|drikke|food|drink|beverage|catering|kantine|canteen|restaurant|"
    r"lunsj|lunch|middag|dinner|frokost|breakfast|kaffe|coffee|snack|"
    r"alimentaire|nourriture|comida|alimento|lebensmittel|getränke)\b", re.I
)

# ── Prompt field extractors ──

def _extract_name(prompt: str) -> tuple[str | None, str | None]:
    """Try to extract firstName/lastName from prompt."""
    # Patterns like "Ola Nordmann", "für Max Mustermann"
    m = re.search(r"(?:for|für|para|pour|per|av)\s+([A-ZÆØÅ]\w+)\s+([A-ZÆØÅ]\w+)", prompt)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _extract_org_number(prompt: str) -> str | None:
    m = re.search(r"(?:org\.?\s*(?:n[ºo°r]\.?|nr\.?|nummer)?\s*:?\s*)(\d{9})", prompt, re.I)
    return m.group(1) if m else None


def _extract_email(prompt: str) -> str | None:
    m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", prompt)
    return m.group(0) if m else None


def _extract_amounts(prompt: str) -> list[float]:
    """Extract NOK amounts from prompt."""
    amounts = []
    for m in re.finditer(r"(\d[\d\s.,]*)\s*(?:NOK|kr|nok)", prompt):
        raw = m.group(1).replace(" ", "").replace(",", ".")
        try:
            amounts.append(float(raw))
        except ValueError:
            pass
    return amounts


def _is_payment_task(prompt: str) -> bool:
    return bool(re.search(
        r"\b(register\s+payment|registrer\s+betaling|registre\s+el\s+pago|"
        r"enregistrer\s+le\s+paiement|Zahlung\s+registrieren|registrar\s+pagamento|"
        r"register\s+the\s+payment|betaling|payment|pago|paiement|Zahlung|pagamento)\b",
        prompt, re.I
    ))


def _is_credit_note_task(prompt: str) -> bool:
    return bool(re.search(
        r"\b(credit\s*note|kreditnota|Gutschrift|note\s+de\s+crédit|nota\s+de\s+crédito)\b",
        prompt, re.I
    ))


def _is_reversal_task(prompt: str) -> bool:
    return bool(re.search(
        r"\b(reversal|reverse|reversed|returnert|devuelto|retourné|zurückgebucht|"
        r"reverse\s+payment|reverser\s+betaling|payment\s+reversal|returned\s+by\s+the\s+bank)\b",
        prompt, re.I
    ))


def _is_order_task(prompt: str) -> bool:
    return bool(re.search(
        r"\b(order|pedido|ordre|Bestellung|commande|bestilling|"
        r"convert\s+to\s+invoice|konverter\s+til\s+faktura|converta\s+em\s+fatura)\b",
        prompt, re.I
    ))


def _is_send_task(prompt: str) -> bool:
    return bool(re.search(
        r"\b(send\s+invoice|send\s+faktura|enviar|envía|envie|envoyez|senden|"
        r"send\s+the\s+invoice|send\s+it)\b",
        prompt, re.I
    ))


# ── Scorer check definitions ──
#
# Based on:
# - Official scoring docs (scoring.md): employee example has 10 points
# - TASK_MAPPING.md: field lists per family
# - 316 live competition logs: actual POST/PUT bodies and failure patterns
# - ISSUES.md: confirmed scorer failure modes
# - family_scoreboard.json: observed dashboard scores and blockers

SCORER_CHECKS = {
    # ── TIER 1: Foundational (×1) ──

    "employee": {
        "tier": 1,
        "max_points": 10,
        "entity_checks": [
            {"type": "employee", "min": 1, "points": 2, "label": "Employee found"},
        ],
        "field_checks": [
            {"entity": "employee", "field": "firstName", "points": 1},
            {"entity": "employee", "field": "lastName", "points": 1},
            {"entity": "employee", "field": "email", "points": 1},
            # Admin role is 5 points — by far the biggest single check
            {"entity": "employee", "field": "userType", "expected": "ADMINISTRATOR", "points": 5},
        ],
        "call_checks": [],
        "notes": "Admin role (userType=ADMINISTRATOR) is 50% of points. Never downgrade to NO_ACCESS.",
    },

    "customer": {
        "tier": 1,
        "max_points": 8,
        "entity_checks": [
            {"type": "customer", "min": 1, "points": 2, "label": "Customer found"},
        ],
        "field_checks": [
            {"entity": "customer", "field": "name", "points": 1},
            {"entity": "customer", "field": "organizationNumber", "points": 1},
            {"entity": "customer", "field": "email", "points": 1},
            {"entity": "customer", "field": "postalAddress", "points": 2},
            {"entity": "customer", "field": "isPrivateIndividual", "points": 1},
        ],
        "call_checks": [],
    },

    "product": {
        "tier": 1,
        "max_points": 8,
        "entity_checks": [
            {"type": "product", "min": 1, "points": 2, "label": "Product found"},
        ],
        "field_checks": [
            {"entity": "product", "field": "name", "points": 1},
            {"entity": "product", "field": "number", "points": 1},
            {"entity": "product", "field": "priceExcludingVatCurrency", "points": 2},
            {"entity": "product", "field": "vatType", "points": 2},
        ],
        "call_checks": [],
        "notes": "VAT type cascade is the main failure mode. sandbox_valid_vat_type blocker.",
    },

    "supplier": {
        "tier": 1,
        "max_points": 8,
        "entity_checks": [
            {"type": "supplier", "min": 1, "points": 2, "label": "Supplier found"},
        ],
        "field_checks": [
            {"entity": "supplier", "field": "name", "points": 1},
            {"entity": "supplier", "field": "organizationNumber", "points": 1},
            {"entity": "supplier", "field": "email", "points": 1},
            {"entity": "supplier", "field": "postalAddress", "points": 1},
        ],
        "call_checks": [],
        "notes": "Supplier invoice (POST /incomingInvoice) returns 403. Use voucher fallback.",
    },

    "department": {
        "tier": 1,
        "max_points": 6,
        "entity_checks": [
            {"type": "department", "min": 1, "points": 2, "label": "Department found"},
        ],
        "field_checks": [
            {"entity": "department", "field": "name", "points": 2},
            {"entity": "department", "field": "departmentNumber", "points": 2},
        ],
        "call_checks": [],
        "notes": "departmentNumber collision with default is common. Auto-assign next available.",
    },

    # ── TIER 2: Multi-step workflows (×2) ──

    "invoice": {
        "tier": 2,
        "max_points": 8,
        "entity_checks": [
            {"type": "customer", "min": 1, "points": 1, "label": "Customer found"},
            {"type": "invoice", "min": 1, "points": 1, "label": "Invoice found"},
        ],
        "field_checks": [
            {"entity": "invoice", "field": "invoiceDate", "points": 1},
            {"entity": "invoice", "field": "invoiceDueDate", "points": 1},
            # orderLines checked individually
            {"entity": "invoice", "field": "orders", "points": 0},  # container
            {"entity": "orderLine", "field": "description", "points": 0.5},
            {"entity": "orderLine", "field": "unitPriceExcludingVatCurrency", "points": 0.5},
            {"entity": "orderLine", "field": "vatType", "points": 1},
            {"entity": "orderLine", "field": "count", "points": 0.5},
        ],
        "call_checks": [
            # Payment sub-task
            {"pattern": "/:payment", "condition": "is_payment", "points": 1.5, "label": "Payment registered"},
            # Credit note sub-task
            {"pattern": "/:createCreditNote", "condition": "is_credit_note", "points": 1.5, "label": "Credit note created"},
            # Send sub-task
            {"pattern": "/:send", "condition": "is_send", "points": 1, "label": "Invoice sent"},
        ],
        "notes": (
            "Issues #25 (food VAT on formation), #26 (double invoice on reversal), "
            "#23 (order→invoice missing invoiceDate). "
            "Payment amount must include VAT (×1.25 for 25% MVA). "
            "Reversal: must be on SAME invoice, not a second one."
        ),
    },

    "project": {
        "tier": 2,
        "max_points": 8,
        "entity_checks": [
            {"type": "customer", "min": 1, "points": 0.5, "label": "Customer found"},
            {"type": "employee", "min": 1, "points": 0.5, "label": "Project manager found"},
            {"type": "project", "min": 1, "points": 1, "label": "Project found"},
        ],
        "field_checks": [
            {"entity": "project", "field": "name", "points": 1},
            {"entity": "project", "field": "projectManager", "points": 1},
            {"entity": "project", "field": "customer", "points": 1},
            {"entity": "project", "field": "startDate", "points": 0.5},
            {"entity": "project", "field": "isInternal", "points": 0.5},
        ],
        "call_checks": [
            {"pattern": "/project/hourlyRates", "condition": "always", "points": 1, "label": "Hourly rates set"},
            {"pattern": "/:invoice", "condition": "is_invoice_task", "points": 1, "label": "Project invoiced"},
        ],
        "notes": (
            "Issue #24: fixed-price project. LLM uses hourlyRates.fixedRate instead of project.fixedprice. "
            "Action-layer intercept sets isFixedPrice on project when hourlyRates PUT detected."
        ),
    },

    "travel_expense": {
        "tier": 2,
        "max_points": 8,
        "entity_checks": [
            {"type": "employee", "min": 0, "points": 0, "label": "Employee exists"},  # often pre-existing
        ],
        "field_checks": [
            {"entity": "travelExpense", "field": "employee", "points": 1},
            {"entity": "travelExpense", "field": "title", "points": 0.5},
            {"entity": "travelExpense", "field": "travelDetails", "points": 1},
            {"entity": "perDiemCompensation", "field": "rateCategory", "points": 1},
            {"entity": "perDiemCompensation", "field": "rateType", "points": 0.5},
            {"entity": "perDiemCompensation", "field": "overnightAccommodation", "points": 0.5},
            {"entity": "cost", "field": "costCategory", "points": 1},
            {"entity": "cost", "field": "amountCurrencyIncVat", "points": 0.5},
        ],
        "call_checks": [
            {"pattern": "/:deliver", "condition": "always", "points": 2, "label": "Expense delivered"},
        ],
        "notes": (
            "Blockers: employee_time_access, rate_category_date_mismatch, travel_expense_kind. "
            "Missing: delivered_state, per_diem_completion. /expense/ must redirect to /travelExpense/."
        ),
    },

    "timesheet": {
        "tier": 2,
        "max_points": 8,
        "entity_checks": [
            {"type": "employee", "min": 0, "points": 0},
            {"type": "project", "min": 0, "points": 0},
            {"type": "activity", "min": 0, "points": 0},
        ],
        "field_checks": [
            {"entity": "timesheet", "field": "employee", "points": 1},
            {"entity": "timesheet", "field": "project", "points": 1},
            {"entity": "timesheet", "field": "activity", "points": 1},
            {"entity": "timesheet", "field": "hours", "points": 1},
            {"entity": "timesheet", "field": "date", "points": 0.5},
        ],
        "call_checks": [
            {"pattern": "/invoice", "condition": "is_invoice_task", "points": 2, "label": "Timesheet invoiced"},
        ],
        "notes": (
            "28.6% clean rate. Requires 5 entities correctly linked: employee, project, activity, hours, date. "
            "Blockers: sandbox_valid_vat_type, employee_time_access, activity_type_required, vat_account_mapping."
        ),
    },

    "salary": {
        "tier": 2,
        "max_points": 8,
        "entity_checks": [
            {"type": "voucher", "min": 1, "points": 2, "label": "Salary voucher found"},
        ],
        "field_checks": [
            {"entity": "voucher", "field": "postings", "points": 2},
            {"entity": "voucher", "field": "date", "points": 1},
            {"entity": "voucher", "field": "description", "points": 1},
        ],
        "call_checks": [],
        "notes": (
            "Issue #22: voucher-only fallback may miss scorer checks. "
            "Scorer may want real /salary/transaction. Currently 70% best observed."
        ),
    },

    # ── TIER 3: Complex scenarios (×3) ──

    "annual_close": {
        "tier": 3,
        "max_points": 8,
        "entity_checks": [
            {"type": "voucher", "min": 1, "points": 1, "label": "Closing voucher found"},
        ],
        "field_checks": [
            {"entity": "voucher", "field": "date", "points": 1},
            {"entity": "voucher", "field": "postings", "points": 2},
            # postings must balance
            {"entity": "voucher", "field": "postings_balance", "points": 2, "validator": "postings_balance"},
        ],
        "call_checks": [],
        "notes": "Depreciation, period close, P&L transfer. Multiple vouchers often needed.",
    },

    "voucher": {
        "tier": 2,
        "max_points": 8,
        "entity_checks": [
            {"type": "voucher", "min": 1, "points": 2, "label": "Voucher found"},
        ],
        "field_checks": [
            {"entity": "voucher", "field": "date", "points": 1},
            {"entity": "voucher", "field": "description", "points": 1},
            {"entity": "voucher", "field": "postings", "points": 2},
            {"entity": "voucher", "field": "postings_balance", "points": 2, "validator": "postings_balance"},
        ],
        "call_checks": [],
        "notes": "Dimension tasks use accountingDimensionName + accountingDimensionValue + voucher with dimension on expense posting only.",
    },

    "bank_reconciliation": {
        "tier": 3,
        "max_points": 8,
        "entity_checks": [],
        "field_checks": [
            {"entity": "voucher", "field": "postings", "points": 4},
        ],
        "call_checks": [
            {"pattern": "/bank/reconciliation", "condition": "always", "points": 2, "label": "Reconciliation posted"},
        ],
        "notes": "100% clean rate. CSV parsing + matching to invoices. Complex but working well.",
    },

    "cost_analysis": {
        "tier": 2,
        "max_points": 8,
        "entity_checks": [
            {"type": "activity", "min": 1, "points": 1, "label": "Activity found"},
            {"type": "voucher", "min": 1, "points": 1, "label": "Voucher found"},
        ],
        "field_checks": [
            {"entity": "voucher", "field": "postings", "points": 2},
            {"entity": "activity", "field": "name", "points": 1},
        ],
        "call_checks": [],
        "notes": "16.7% clean rate. Blocker: activity_type_required. Missing: activity, employee, project linkage.",
    },

    "ledger_correction": {
        "tier": 3,
        "max_points": 8,
        "entity_checks": [
            {"type": "voucher", "min": 1, "points": 2, "label": "Correction voucher found"},
        ],
        "field_checks": [
            {"entity": "voucher", "field": "postings", "points": 3},
            {"entity": "voucher", "field": "date", "points": 1},
            {"entity": "voucher", "field": "postings_balance", "points": 2, "validator": "postings_balance"},
        ],
        "call_checks": [],
        "notes": "70% best observed. Reversing entry + correct entry needed.",
    },
}


# ── Scoring engine ──

def estimate_score(family: str, prompt: str, mock_state) -> dict:
    """
    Estimate k/n score from mock state after a replay.

    Returns:
        {
            "family": str,
            "tier": int,
            "points_earned": float,
            "max_points": float,
            "correctness": float,          # points_earned / max_points
            "tier_score": float,            # correctness * tier
            "checks": [{"label": str, "passed": bool, "points": float, "detail": str}],
            "issues": [str],               # semantic issues found
        }
    """
    spec = SCORER_CHECKS.get(family)
    if not spec:
        return {
            "family": family, "tier": 1, "points_earned": 0, "max_points": 1,
            "correctness": 0, "tier_score": 0, "checks": [], "issues": [f"No scorer spec for family '{family}'"],
        }

    tier = spec["tier"]
    max_points = spec["max_points"]
    checks = []
    issues = []
    points = 0.0

    entities = mock_state.entities if hasattr(mock_state, "entities") else {}
    calls = mock_state.calls if hasattr(mock_state, "calls") else []

    # Collect all POST bodies by entity type
    post_bodies = {}
    for c in calls:
        if c.get("method") != "POST":
            continue
        path = (c.get("path") or "").lower()
        body = c.get("body") or {}
        for etype in entities:
            if etype in path or etype.replace("_", "") in path:
                post_bodies.setdefault(etype, []).append(body)
                break
        else:
            # Special paths
            if "voucher" in path:
                post_bodies.setdefault("voucher", []).append(body)
            elif "activity" in path:
                post_bodies.setdefault("activity", []).append(body)
            elif "invoice" in path and "incoming" not in path:
                post_bodies.setdefault("invoice", []).append(body)
            elif "order" in path:
                post_bodies.setdefault("order", []).append(body)
            elif "perdiem" in path:
                post_bodies.setdefault("perDiemCompensation", []).append(body)
            elif "cost" in path and "travelexpense" in path:
                post_bodies.setdefault("cost", []).append(body)

    # Also use entities from mock_state.entities (created via create_handler)
    for etype, elist in entities.items():
        if elist and etype not in post_bodies:
            post_bodies[etype] = elist

    # 1. Entity existence checks
    for ec in spec.get("entity_checks", []):
        etype = ec["type"]
        min_count = ec["min"]
        ec_points = ec["points"]
        label = ec.get("label", f"{etype} found")
        actual = len(entities.get(etype, []))

        if actual >= min_count and min_count > 0:
            points += ec_points
            checks.append({"label": label, "passed": True, "points": ec_points, "detail": f"{actual} found"})
        elif min_count > 0:
            checks.append({"label": label, "passed": False, "points": 0, "detail": f"expected >={min_count}, got {actual}"})
            issues.append(f"Missing {etype}: expected >={min_count}, got {actual}")

    # 2. Field presence checks
    for fc in spec.get("field_checks", []):
        etype = fc["entity"]
        field = fc["field"]
        fc_points = fc["points"]
        expected = fc.get("expected")
        validator = fc.get("validator")

        if fc_points == 0:
            continue  # Container field, skip

        bodies = post_bodies.get(etype, entities.get(etype, []))
        if not bodies:
            # Check if this is an orderLine check — look inside invoice/order bodies
            if etype == "orderLine":
                bodies = _extract_order_lines(post_bodies)

            if not bodies:
                checks.append({"label": f"{etype}.{field}", "passed": False, "points": 0,
                               "detail": f"no {etype} entities found"})
                continue

        # Special validators
        if validator == "postings_balance":
            balanced = _check_postings_balance(bodies)
            if balanced:
                points += fc_points
                checks.append({"label": "postings_balance", "passed": True, "points": fc_points, "detail": "balanced"})
            else:
                checks.append({"label": "postings_balance", "passed": False, "points": 0, "detail": "unbalanced"})
                issues.append("Voucher postings do not balance")
            continue

        # Standard field presence check
        found = False
        for body in bodies:
            val = body.get(field)
            if val is not None and val != "" and val != []:
                if expected is not None:
                    if str(val) == str(expected):
                        found = True
                        break
                else:
                    found = True
                    break

        if found:
            points += fc_points
            checks.append({"label": f"{etype}.{field}", "passed": True, "points": fc_points, "detail": "present"})
        else:
            detail = f"missing from {len(bodies)} {etype} bodies"
            if expected:
                detail += f" (expected={expected})"
            checks.append({"label": f"{etype}.{field}", "passed": False, "points": 0, "detail": detail})

    # 3. Call pattern checks
    call_paths = [c.get("path", "") for c in calls]
    for cc in spec.get("call_checks", []):
        pattern = cc["pattern"]
        condition = cc.get("condition", "always")
        cc_points = cc["points"]
        label = cc.get("label", f"Call to {pattern}")

        # Check if condition applies
        should_check = False
        if condition == "always":
            should_check = True
        elif condition == "is_payment":
            should_check = _is_payment_task(prompt)
        elif condition == "is_credit_note":
            should_check = _is_credit_note_task(prompt)
        elif condition == "is_send":
            should_check = _is_send_task(prompt)
        elif condition == "is_invoice_task":
            should_check = bool(re.search(r"\b(invoice|faktura|factura|fatura|Rechnung|facture)\b", prompt, re.I))

        if not should_check:
            continue

        found = any(pattern in p for p in call_paths)
        if found:
            points += cc_points
            checks.append({"label": label, "passed": True, "points": cc_points, "detail": "call found"})
        else:
            checks.append({"label": label, "passed": False, "points": 0, "detail": f"no call matching '{pattern}'"})
            issues.append(f"Missing call: {label}")

    # 4. Family-specific semantic checks
    family_issues = _family_semantic_checks(family, prompt, mock_state, post_bodies, call_paths)
    issues.extend(family_issues)

    correctness = points / max_points if max_points > 0 else 0
    tier_score = correctness * tier

    return {
        "family": family,
        "tier": tier,
        "points_earned": round(points, 1),
        "max_points": max_points,
        "correctness": round(correctness, 3),
        "tier_score": round(tier_score, 3),
        "checks": checks,
        "issues": issues,
    }


def _extract_order_lines(post_bodies: dict) -> list[dict]:
    """Extract orderLine dicts from invoice/order POST bodies."""
    lines = []
    for etype in ("invoice", "order"):
        for body in post_bodies.get(etype, []):
            for order in body.get("orders", [body]):
                lines.extend(order.get("orderLines", []))
    return lines


def _check_postings_balance(bodies: list[dict]) -> bool:
    """Check if all voucher postings sum to zero."""
    for body in bodies:
        postings = body.get("postings", [])
        if postings:
            total = sum(p.get("amountGross", 0) or 0 for p in postings)
            if abs(total) > 0.01:
                return False
    return True


def _family_semantic_checks(
    family: str, prompt: str, mock_state, post_bodies: dict, call_paths: list[str]
) -> list[str]:
    """Family-specific semantic validation beyond field presence."""
    issues = []
    entities = mock_state.entities if hasattr(mock_state, "entities") else {}

    if family == "invoice":
        # Check: invoiceDueDate != invoiceDate
        for body in post_bodies.get("invoice", []):
            inv_date = body.get("invoiceDate", "")
            due_date = body.get("invoiceDueDate", "")
            if inv_date and due_date and inv_date == due_date:
                issues.append("invoiceDueDate == invoiceDate (should be +30 days)")

        # Check: VAT type on order lines — food VAT on non-food
        for body in post_bodies.get("invoice", []):
            for order in body.get("orders", [body]):
                for line in order.get("orderLines", []):
                    vt = line.get("vatType", {})
                    if isinstance(vt, dict) and vt.get("id") == 31:
                        desc = line.get("description", "")
                        if not _FOOD_KEYWORDS.search(desc):
                            issues.append(f"Food VAT (id=31) on non-food: '{desc}'")

        # Check: payment reversal creates exactly 1 invoice
        if _is_reversal_task(prompt):
            invoice_posts = [c for c in mock_state.calls
                             if c.get("method") == "POST" and "invoice" in (c.get("path") or "").lower()
                             and "payment" not in (c.get("path") or "").lower()]
            if len(invoice_posts) > 1:
                issues.append(f"Payment reversal created {len(invoice_posts)} invoices (should be 1)")

        # Check: payment uses correct amount (incl VAT, not excl)
        if _is_payment_task(prompt):
            excl_vat = bool(re.search(
                r"\b(excl\.?\s*(?:MVA|VAT|TVA|IVA|MwSt)|hors\s+TVA|ohne\s+MwSt|sem\s+IVA|sin\s+IVA|ekskl\.?\s*mva)\b",
                prompt, re.I
            ))
            if excl_vat:
                # Check if payment amount was multiplied by 1.25
                for c in mock_state.calls:
                    if "/:payment" in (c.get("path") or ""):
                        params = c.get("params", {})
                        paid = params.get("paidAmount")
                        if paid:
                            try:
                                paid_f = float(paid)
                                amounts = _extract_amounts(prompt)
                                for amt in amounts:
                                    if abs(paid_f - amt) < 1:  # Used excl-VAT directly
                                        issues.append(
                                            f"Payment amount {paid_f} matches excl-VAT {amt} — should be {amt * 1.25} (incl 25% MVA)"
                                        )
                            except (ValueError, TypeError):
                                pass

        # Check: order→invoice has invoiceDate
        for c in mock_state.calls:
            path = c.get("path", "")
            if c.get("method") == "PUT" and "/:invoice" in path and "/invoice/" not in path:
                params = c.get("params", {})
                if not params.get("invoiceDate"):
                    issues.append("order→invoice PUT missing invoiceDate param")

    elif family == "employee":
        # Check: userType not downgraded to NO_ACCESS
        for body in post_bodies.get("employee", []):
            ut = body.get("userType", "")
            if ut == "NO_ACCESS":
                issues.append("Employee userType=NO_ACCESS (should be ADMINISTRATOR or STANDARD)")

    elif family == "travel_expense":
        # Check: /:deliver was called
        has_deliver = any("/:deliver" in p for p in call_paths)
        if not has_deliver:
            issues.append("Travel expense not delivered (missing /:deliver)")

        # Check: isForeignTravel derived from country
        for body in post_bodies.get("travelExpense", entities.get("travel_expense", [])):
            details = body.get("travelDetails", {})
            if isinstance(details, dict):
                country = details.get("departureCountryCode") or details.get("returnCountryCode")
                is_foreign = body.get("isForeignTravel")
                if country and country != "NO" and not is_foreign:
                    issues.append(f"Foreign travel (country={country}) but isForeignTravel not set")

    elif family == "salary":
        # Check: postings have salary accounts (5000-series)
        for body in post_bodies.get("voucher", []):
            postings = body.get("postings", [])
            has_salary_account = any(
                str(p.get("account", {}).get("number", "")).startswith("5")
                for p in postings if isinstance(p.get("account"), dict)
            )
            if postings and not has_salary_account:
                issues.append("Salary voucher has no 5000-series account")

    elif family == "project":
        # Check: fixed-price project has isFixedPrice
        if re.search(r"\b(fest\s*pris|fixed\s*price|precio\s*fijo|prix\s*fixe|Festpreis)\b", prompt, re.I):
            for body in post_bodies.get("project", entities.get("project", [])):
                if not body.get("isFixedPrice"):
                    issues.append("Fixed-price project missing isFixedPrice=true")

    return issues


def format_score_report(result: dict) -> str:
    """Format a score estimate as a readable string."""
    lines = []
    lines.append(f"Family: {result['family']} (Tier {result['tier']})")
    lines.append(f"Score: {result['points_earned']}/{result['max_points']} = {result['correctness']:.1%}")
    lines.append(f"Tier score: {result['tier_score']:.2f} / {result['tier'] * 1.0:.0f}.0 max (no efficiency bonus)")
    lines.append("")

    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        pts = f"+{check['points']}" if check["passed"] else "+0"
        lines.append(f"  [{status}] {check['label']:<35} {pts:>4}  ({check['detail']})")

    if result["issues"]:
        lines.append("")
        lines.append("Semantic issues:")
        for issue in result["issues"]:
            lines.append(f"  - {issue}")

    return "\n".join(lines)
