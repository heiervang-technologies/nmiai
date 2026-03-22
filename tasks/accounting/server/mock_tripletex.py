"""
Mock Tripletex API server for offline testing.
Returns plausible responses for all endpoints the agent uses.
Tracks all calls for assertion in tests.
"""

import json
import logging
import re
from datetime import date

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

log = logging.getLogger(__name__)

# Auto-incrementing IDs
_next_id = {"value": 100}

def _id():
    _next_id["value"] += 1
    return _next_id["value"]


class MockState:
    """Tracks all API calls and stores created entities."""
    def __init__(self):
        self.calls: list[dict] = []
        self.validation_errors: list[dict] = []  # Semantic issues found in request bodies
        self.entities: dict[str, list[dict]] = {
            "employee": [],
            "customer": [],
            "supplier": [],
            "product": [],
            "department": [],
            "invoice": [],
            "order": [],
            "project": [],
            "voucher": [],
            "activity": [],
            "timesheet": [],
        }
        # Pre-populate company
        self.company = {
            "id": 1, "name": "Test Company AS", "organizationNumber": "999888777",
            "bankAccountNumber": "1234.56.78903",
        }
        # Pre-populate some accounts
        self.accounts = [
            {"id": i, "number": n, "name": name, "vatType": vat, "ledgerType": lt}
            for i, (n, name, vat, lt) in enumerate([
                (1500, "Kundefordringer", None, "CUSTOMER"),
                (1920, "Bankkonto", None, "GENERAL"),
                (2400, "Leverandørgjeld", None, "VENDOR"),
                (2710, "Utgående MVA", None, "GENERAL"),
                (3000, "Salgsinntekt", {"id": 3}, "GENERAL"),
                (4000, "Varekostnad", {"id": 3}, "GENERAL"),
                (5000, "Lønn", None, "GENERAL"),
                (6000, "Avskrivning", None, "GENERAL"),
                (7000, "Kontorkostnader", {"id": 3}, "GENERAL"),
                (8300, "Skattekostnad", None, "GENERAL"),
            ], start=1)
        ]
        self.vat_types = [
            {"id": 3, "name": "Høy sats", "percentage": 25.0},
            {"id": 5, "name": "Fritatt", "percentage": 0.0},
            {"id": 6, "name": "Utenfor MVA", "percentage": 0.0},
            {"id": 31, "name": "Mat og drikke", "percentage": 15.0},
        ]
        self.payment_types = [
            {"id": 1, "description": "Bankoverføring"},
            {"id": 2, "description": "Kontant"},
        ]

    def log_call(self, method: str, path: str, params: dict = None, body: dict = None):
        self.calls.append({"method": method, "path": path, "params": params, "body": body})

    def log_validation(self, path: str, issue: str, severity: str = "warning", body: dict = None):
        """Record a semantic validation issue (doesn't block the request)."""
        self.validation_errors.append({
            "path": path, "issue": issue, "severity": severity,
            "body_preview": str(body)[:200] if body else None,
        })

    def get_assertions(self) -> dict:
        """Run post-request assertions and return results."""
        issues = list(self.validation_errors)

        # Entity count assertions
        invoice_count = len(self.entities["invoice"])
        customer_count = len(self.entities["customer"])
        supplier_count = len(self.entities["supplier"])

        # Check for duplicate invoices (payment reversal should create 1, not 2+)
        invoice_posts = [c for c in self.calls if c["method"] == "POST" and "invoice" in (c.get("path") or "").lower() and "payment" not in (c.get("path") or "").lower()]
        if len(invoice_posts) > 3:
            issues.append({"path": "POST /invoice", "issue": f"Possible loop: {len(invoice_posts)} invoice POST calls", "severity": "error"})

        # Check for excessive API calls
        total_calls = len(self.calls)
        if total_calls > 20:
            issues.append({"path": "*", "issue": f"Excessive API calls: {total_calls} (typical: 4-12)", "severity": "warning"})
        if total_calls > 40:
            issues.append({"path": "*", "issue": f"Runaway loop: {total_calls} API calls", "severity": "error"})

        # Check for repeated identical calls (loop detection)
        from collections import Counter
        call_sigs = Counter()
        for c in self.calls:
            sig = f"{c['method']} {c.get('path', '')}"
            call_sigs[sig] += 1
        for sig, count in call_sigs.items():
            if count > 4 and "GET" not in sig:
                issues.append({"path": sig, "issue": f"Repeated {count}x — possible LLM loop", "severity": "error"})

        return {
            "issues": issues,
            "entity_counts": {k: len(v) for k, v in self.entities.items() if v},
            "total_calls": total_calls,
            "error_count": len([i for i in issues if i["severity"] == "error"]),
            "warning_count": len([i for i in issues if i["severity"] == "warning"]),
        }

    def reset(self):
        self.__init__()


state = MockState()


def _wrap(value):
    """Wrap a single entity in Tripletex response format."""
    return {"value": value}


def _wrap_list(values):
    """Wrap a list of entities in Tripletex response format."""
    return {"fullResultSize": len(values), "from": 0, "count": len(values), "values": values}


def _extract_params(request) -> dict:
    return dict(request.query_params)


async def _read_body(request) -> dict | None:
    try:
        return await request.json()
    except Exception:
        return None


# --- Route handlers ---

async def company_handler(request):
    params = _extract_params(request)
    state.log_call("GET", request.url.path, params=params)
    return JSONResponse(_wrap(state.company))


async def account_handler(request):
    params = _extract_params(request)
    state.log_call("GET", request.url.path, params=params)
    # Filter by account number range if provided
    accts = state.accounts
    if "accountNumberFrom" in params:
        lo = int(params["accountNumberFrom"])
        accts = [a for a in accts if a["number"] >= lo]
    if "accountNumberTo" in params:
        hi = int(params["accountNumberTo"])
        accts = [a for a in accts if a["number"] <= hi]
    return JSONResponse(_wrap_list(accts))


async def vat_type_handler(request):
    state.log_call("GET", request.url.path)
    return JSONResponse(_wrap_list(state.vat_types))


async def payment_type_handler(request):
    state.log_call("GET", request.url.path)
    return JSONResponse(_wrap_list(state.payment_types))


async def list_handler(request):
    """Generic list handler for GET endpoints."""
    params = _extract_params(request)
    path = request.url.path
    state.log_call("GET", path, params=params)
    # Figure out entity type from path
    for etype in state.entities:
        if etype in path.lower() or etype.replace("_", "") in path.lower():
            return JSONResponse(_wrap_list(state.entities[etype]))
    # Ledger posting
    if "/ledger/posting" in path:
        return JSONResponse(_wrap_list([
            {"account": {"id": 5, "number": 4000}, "amountGross": 50000, "date": "2026-01-15", "description": "Varekjøp"},
            {"account": {"id": 5, "number": 4000}, "amountGross": 75000, "date": "2026-02-15", "description": "Varekjøp"},
            {"account": {"id": 7, "number": 5000}, "amountGross": 30000, "date": "2026-01-15", "description": "Lønn"},
            {"account": {"id": 7, "number": 5000}, "amountGross": 35000, "date": "2026-02-15", "description": "Lønn"},
        ]))
    if "/ledger/voucher" in path:
        return JSONResponse(_wrap_list(state.entities.get("voucher", [])))
    return JSONResponse(_wrap_list([]))


async def get_by_id_handler(request):
    """Handle GET /entity/{id}"""
    params = _extract_params(request)
    path = request.url.path
    state.log_call("GET", path, params=params)
    eid = int(request.path_params.get("id", 0))
    for etype, entities in state.entities.items():
        if etype in path.lower():
            for e in entities:
                if e.get("id") == eid:
                    return JSONResponse(_wrap(e))
    # Return a plausible default
    return JSONResponse(_wrap({"id": eid, "name": f"Entity {eid}"}))


_FOOD_KEYWORDS = re.compile(
    r"\b(mat|drikke|food|drink|beverage|catering|kantine|canteen|restaurant|"
    r"lunsj|lunch|middag|dinner|frokost|breakfast|kaffe|coffee|snack|"
    r"alimentaire|nourriture|comida|alimento|lebensmittel|getränke)\b", re.I
)


async def create_handler(request):
    """Generic POST handler — creates entity and returns it."""
    body = await _read_body(request)
    path = request.url.path
    state.log_call("POST", path, body=body)

    # --- Semantic validation (log issues, don't block) ---
    if body:
        path_lower = path.lower()

        # 1. Invoice/order orderLines: vatType required, food VAT on non-food
        if ("invoice" in path_lower or "order" in path_lower) and "voucher" not in path_lower:
            for order in body.get("orders", [body]):
                for line in order.get("orderLines", []):
                    vt = line.get("vatType")
                    if not vt or not vt.get("id"):
                        state.log_validation(path, f"Missing vatType on orderLine: {line.get('description', '?')}", "error", body)
                    elif vt.get("id") == 31:
                        desc = line.get("description", "")
                        if not _FOOD_KEYWORDS.search(desc):
                            state.log_validation(path, f"Wrong VAT: food rate (id=31) on non-food item '{desc}'", "error", body)
            # 6. invoiceDueDate >= invoiceDate
            inv_date = body.get("invoiceDate", "")
            due_date = body.get("invoiceDueDate", "")
            if inv_date and due_date and due_date < inv_date:
                state.log_validation(path, f"invoiceDueDate ({due_date}) < invoiceDate ({inv_date})", "error", body)
            if inv_date and due_date and due_date == inv_date:
                state.log_validation(path, f"invoiceDueDate == invoiceDate ({inv_date}) — should be +30 days", "warning", body)

        # 2-3. Voucher: balance check + forbidden fields
        if "voucher" in path_lower:
            _VOUCHER_FORBIDDEN = {"comment", "vendorInvoiceNumber", "externalVoucherNumber", "invoiceNumber"}
            found_forbidden = _VOUCHER_FORBIDDEN & set(body.keys())
            if found_forbidden:
                state.log_validation(path, f"Forbidden fields on voucher: {found_forbidden}", "error", body)
            postings = body.get("postings", [])
            if postings:
                total = sum(p.get("amountGross", 0) or 0 for p in postings)
                if abs(total) > 0.01:
                    state.log_validation(path, f"Unbalanced voucher: postings sum to {total}", "error", body)

        # 7. Employee: firstName + lastName required
        if "employee" in path_lower:
            if not body.get("firstName"):
                state.log_validation(path, "Employee missing firstName", "error", body)
            if not body.get("lastName"):
                state.log_validation(path, "Employee missing lastName", "error", body)

    entity = dict(body or {})
    entity["id"] = _id()
    entity.setdefault("name", "Created Entity")

    # Store in appropriate bucket
    for etype in state.entities:
        if etype in path.lower() or etype.replace("_", "") in path.lower():
            state.entities[etype].append(entity)
            break
    else:
        if "voucher" in path.lower():
            state.entities["voucher"].append(entity)
        elif "activity" in path.lower():
            state.entities["activity"].append(entity)

    return JSONResponse(_wrap(entity), status_code=201)


async def update_handler(request):
    """Generic PUT handler."""
    body = await _read_body(request)
    params = _extract_params(request)
    path = request.url.path
    state.log_call("PUT", path, params=params, body=body)

    eid = request.path_params.get("id", _id())

    # Special: invoice payment
    if "/:payment" in path:
        return JSONResponse(_wrap({"id": eid, "paymentComplete": True}))
    # Special: order to invoice
    if "/:invoice" in path:
        # 4. invoiceDate MUST be in params
        if not params.get("invoiceDate"):
            state.log_validation(path, "order→invoice missing invoiceDate param (real API returns 422)", "error", params)
        inv = {"id": _id(), "invoiceNumber": 10001, "invoiceDate": params.get("invoiceDate", str(date.today()))}
        state.entities["invoice"].append(inv)
        return JSONResponse(_wrap(inv))
    # Special: invoice send
    if "/:send" in path:
        # 5. sendType MUST be in params
        if not params.get("sendType"):
            state.log_validation(path, "/:send missing sendType param", "warning", params)
        return JSONResponse(_wrap({"id": eid, "sent": True}))
    # Special: createReminder
    if "/:createReminder" in path:
        return JSONResponse(_wrap({"id": eid, "reminderCreated": True}))
    # Special: deliver
    if "/:deliver" in path:
        return JSONResponse(_wrap({"id": eid, "delivered": True}))

    return JSONResponse(_wrap({"id": eid, **(body or {})}))


async def delete_handler(request):
    path = request.url.path
    state.log_call("DELETE", path)
    return JSONResponse({"status": 200})


async def sales_modules_handler(request):
    state.log_call("GET", request.url.path)
    return JSONResponse(_wrap_list([
        {"id": 1, "name": "Invoice", "isActive": True},
        {"id": 2, "name": "Project", "isActive": True},
    ]))


async def catchall_get(request):
    params = _extract_params(request)
    path = request.url.path
    state.log_call("GET", path, params=params)
    return JSONResponse(_wrap_list([]))


async def catchall_post(request):
    body = await _read_body(request)
    path = request.url.path
    state.log_call("POST", path, body=body)
    # Run same voucher/invoice validations on catchall
    if body:
        path_lower = path.lower()
        if "voucher" in path_lower:
            found_forbidden = {"comment", "vendorInvoiceNumber", "externalVoucherNumber", "invoiceNumber"} & set(body.keys())
            if found_forbidden:
                state.log_validation(path, f"Forbidden fields on voucher: {found_forbidden}", "error", body)
            postings = body.get("postings", [])
            if postings:
                total = sum(p.get("amountGross", 0) or 0 for p in postings)
                if abs(total) > 0.01:
                    state.log_validation(path, f"Unbalanced voucher: postings sum to {total}", "error", body)
        if ("invoice" in path_lower or "order" in path_lower) and "voucher" not in path_lower:
            for order in body.get("orders", [body]):
                for line in order.get("orderLines", []):
                    vt = line.get("vatType")
                    if vt and vt.get("id") == 31:
                        desc = line.get("description", "")
                        if not _FOOD_KEYWORDS.search(desc):
                            state.log_validation(path, f"Wrong VAT: food rate (id=31) on non-food item '{desc}'", "error", body)
    entity = dict(body or {})
    entity["id"] = _id()
    return JSONResponse(_wrap(entity), status_code=201)


async def catchall_put(request):
    body = await _read_body(request)
    params = _extract_params(request)
    path = request.url.path
    state.log_call("PUT", path, params=params, body=body)
    return JSONResponse(_wrap({"id": _id(), **(body or {})}))


# --- App ---

# Order matters: more specific routes first
routes = [
    Route("/v2/company/{rest:path}", company_handler, methods=["GET"]),
    Route("/v2/ledger/account", account_handler, methods=["GET"]),
    Route("/v2/ledger/vatType", vat_type_handler, methods=["GET"]),
    Route("/v2/invoice/paymentType", payment_type_handler, methods=["GET"]),
    Route("/v2/company/salesmodules", sales_modules_handler, methods=["GET"]),
    # CRUD by ID
    Route("/v2/{entity}/{id:int}/{action:path}", update_handler, methods=["PUT"]),
    Route("/v2/{entity}/{id:int}", get_by_id_handler, methods=["GET"]),
    Route("/v2/{entity}/{id:int}", update_handler, methods=["PUT"]),
    Route("/v2/{entity}/{id:int}", delete_handler, methods=["DELETE"]),
    # List + Create
    Route("/v2/{entity}/{sub:path}", list_handler, methods=["GET"]),
    Route("/v2/{entity}/{sub:path}", create_handler, methods=["POST"]),
    Route("/v2/{entity}/{sub:path}", update_handler, methods=["PUT"]),
    Route("/v2/{entity}", list_handler, methods=["GET"]),
    Route("/v2/{entity}", create_handler, methods=["POST"]),
]

app = Starlette(routes=routes)


def get_state() -> MockState:
    return state


def reset_state():
    global state, _next_id
    _next_id["value"] = 100
    state.reset()
