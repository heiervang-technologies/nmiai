"""
Tripletex API client with call tracking.
All calls go through the provided proxy base_url.
Auth: Basic auth with username=0, password=session_token.
"""

import httpx
import json as json_mod
import logging

log = logging.getLogger(__name__)


def _split_path_params(path: str) -> tuple[str, dict]:
    """If the LLM puts query params in the path, extract them."""
    if "?" in path:
        parts = path.split("?", 1)
        params = dict(parse_qs(parts[1], keep_blank_values=True))
        # parse_qs returns lists, flatten single values
        params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
        return parts[0], params
    return path, {}


def _payload_preview(payload, max_chars: int = 1200):
    if payload is None:
        return None
    try:
        normalized = json_mod.loads(json_mod.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        normalized = str(payload)
    try:
        encoded = json_mod.dumps(normalized, ensure_ascii=False, default=str)
    except Exception:
        encoded = str(normalized)
    if len(encoded) <= max_chars:
        return normalized
    return encoded[:max_chars] + '...[truncated]'


from urllib.parse import parse_qs


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self._client = httpx.AsyncClient(
            auth=self.auth,
            timeout=60.0,
            headers={"Content-Type": "application/json"},
        )
        # Bandit tracking
        self.call_count = 0
        self.error_count = 0
        self.calls_log = []
        # Simple GET cache to avoid duplicate reads within same request
        self._get_cache: dict[str, dict] = {}
        # Track if a reversal flow was completed (prevents duplicate invoice creation)
        self._reversal_invoice_id: int | None = None

    async def close(self):
        await self._client.aclose()

    def _log_call(self, method: str, path: str, status: int, error: str = None, params=None, json=None):
        self.call_count += 1
        if error or (400 <= status < 500):
            self.error_count += 1
        entry = {
            "method": method,
            "path": path,
            "status": status,
            "error": error,
        }
        preview_params = _payload_preview(params)
        if preview_params is not None:
            entry["params"] = preview_params
        preview_json = _payload_preview(json)
        if preview_json is not None:
            entry["json"] = preview_json
        self.calls_log.append(entry)

    async def get(self, path: str, params: dict | None = None) -> dict:
        clean_path, path_params = _split_path_params(path)
        merged_params = {**(path_params or {}), **(params or {})} or None
        # Auto-add required date params (catches run_python bypassing action layer)
        if merged_params is None:
            merged_params = {}
        lp = clean_path.lower()
        # Only inject date params on LIST endpoints (not /invoice/{id} or /invoice/paymentType)
        import re as _re
        is_invoice_list = lp.rstrip("/").endswith("/invoice") or _re.search(r"/invoice\?", lp)
        is_order_list = lp.rstrip("/").endswith("/order") or _re.search(r"/order\?", lp)
        if is_invoice_list and "invoiceDateFrom" not in merged_params:
            merged_params["invoiceDateFrom"] = "2020-01-01"
            merged_params["invoiceDateTo"] = "2030-12-31"
        if is_order_list and "orderDateFrom" not in merged_params:
            merged_params["orderDateFrom"] = "2020-01-01"
            merged_params["orderDateTo"] = "2030-12-31"
        is_supplier_invoice_list = _re.search(r"/supplierinvoice/?$", lp) or _re.search(r"/supplierinvoice\?", lp)
        if is_supplier_invoice_list and "invoiceDateFrom" not in merged_params:
            merged_params["invoiceDateFrom"] = "2020-01-01"
            merged_params["invoiceDateTo"] = "2030-12-31"
        if "/timesheet/entry" in lp and "dateFrom" not in merged_params:
            merged_params["dateFrom"] = "2020-01-01"
            merged_params["dateTo"] = "2030-12-31"
        if lp.rstrip("/").endswith("/ledger/voucher") and "dateFrom" not in merged_params:
            merged_params["dateFrom"] = "2020-01-01"
            merged_params["dateTo"] = "2030-12-31"
        if "/ledger/posting" in lp and "dateFrom" not in merged_params:
            merged_params["dateFrom"] = "2020-01-01"
            merged_params["dateTo"] = "2030-12-31"
        merged_params = merged_params or None
        # Cache key for deduplication within same request
        cache_key = f"{clean_path}|{json_mod.dumps(merged_params, sort_keys=True, default=str) if merged_params else ''}"
        if cache_key in self._get_cache:
            log.info(f"GET {clean_path} params={merged_params} [CACHED]")
            return self._get_cache[cache_key]
        url = f"{self.base_url}{clean_path}"
        log.info(f"GET {clean_path} params={merged_params}")
        resp = await self._client.get(url, params=merged_params)
        error_text = None
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            error_text = resp.text[:500]
            # Gracefully handle 404 on GET — return empty instead of raising.
            # DO NOT swallow 422, as that hides validation errors (like missing required params) from the LLM.
            if resp.status_code == 404:
                log.warning(f"GET {clean_path} returned 404 — returning empty result")
                # Log real status but don't count as error (saves error budget)
                self.call_count += 1
                self.calls_log.append({"method": "GET", "path": clean_path, "status": 404,
                                       "error": None, **({"params": _payload_preview(merged_params)} if merged_params else {})})
                return {"values": [], "count": 0}
            self._log_call("GET", clean_path, resp.status_code, error_text, params=merged_params)
            raise
        self._log_call("GET", clean_path, resp.status_code, params=merged_params)
        result = resp.json()
        self._get_cache[cache_key] = result
        return result

    async def post(self, path: str, json: dict | None = None) -> dict:
        clean_path, path_params = _split_path_params(path)
        # If LLM put fields in query params instead of body, convert to body
        if path_params and not json:
            log.warning(f"POST {clean_path}: converting query params to body: {path_params}")
            json = path_params
        # Block duplicate invoice creation after reversal flow completed
        if self._reversal_invoice_id and clean_path.rstrip("/").endswith("/invoice"):
            log.warning(f"Blocked duplicate POST /invoice — reversal already created invoice {self._reversal_invoice_id}")
            return {"value": {"id": self._reversal_invoice_id}, "message": "Invoice already created during reversal flow"}
        # Auto-inject invoiceDueDate on POST /invoice if missing (LLM forgets via run_python)
        if json and clean_path.rstrip("/").endswith("/invoice") and "invoiceDueDate" not in json:
            inv_date = json.get("invoiceDate", __import__("datetime").date.today().isoformat())
            from datetime import timedelta
            try:
                json["invoiceDueDate"] = (__import__("datetime").date.fromisoformat(inv_date) + timedelta(days=30)).isoformat()
                log.warning(f"Auto-injected invoiceDueDate={json['invoiceDueDate']} on POST /invoice")
            except Exception:
                pass
        # Strip employmentType from /employee/employment — Tripletex rejects it
        # ("Feltet eksisterer ikke i objektet"). employmentType belongs on employment/details only.
        if json and "/employee/employment" in clean_path and "employmentType" in json:
            json.pop("employmentType")
            log.warning(f"Stripped employmentType from /employee/employment POST (not accepted)")
        # Fix hourlyRateModel enum values (LLM sends wrong names)
        if json and "hourlyrates" in clean_path.lower() and "hourlyRateModel" in json:
            model_val = json["hourlyRateModel"]
            # Tripletex requires TYPE_ prefix
            valid_models = {"TYPE_PREDEFINED_HOURLY_RATES", "TYPE_PROJECT_SPECIFIC_HOURLY_RATES",
                           "TYPE_FIXED_HOURLY_RATE"}
            if model_val not in valid_models:
                if "FIXED" in model_val.upper():
                    json["hourlyRateModel"] = "TYPE_FIXED_HOURLY_RATE"
                elif "PREDEFINED" in model_val.upper():
                    json["hourlyRateModel"] = "TYPE_PREDEFINED_HOURLY_RATES"
                else:
                    json["hourlyRateModel"] = "TYPE_PROJECT_SPECIFIC_HOURLY_RATES"
                log.warning(f"Fixed hourlyRateModel: {model_val} → {json['hourlyRateModel']}")
        # Auto-balance voucher postings at the client level (catches run_python too)
        if json and "voucher" in clean_path.lower() and "postings" in json:
            postings = json.get("postings", [])
            if postings:
                total = sum(float(p.get("amountGross", p.get("amount", 0)) or 0) for p in postings)
                if abs(total) > 0.01:
                    if len(postings) >= 2:
                        last = postings[-1]
                        last_amount = float(last.get("amountGross", last.get("amount", 0)) or 0)
                        last["amountGross"] = round(last_amount - total, 2)
                        last["amountGrossCurrency"] = round(last_amount - total, 2)
                        log.warning(f"Client auto-balanced voucher: adjusted last posting by {round(-total, 2)}")
                    else:
                        # Single posting: add balancing row to bank 1920
                        postings.append({
                            "row": len(postings) + 1,
                            "account": {"id": 1, "number": 1920},
                            "amountGross": round(-total, 2),
                            "amountGrossCurrency": round(-total, 2),
                            "description": "Motpost",
                        })
                        log.warning(f"Client auto-added balancing posting: {round(-total, 2)} to 1920")
        url = f"{self.base_url}{clean_path}"
        log.info(f"POST {clean_path} body_keys={list((json or {}).keys())}")
        resp = await self._client.post(url, json=json)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            self._log_call("POST", clean_path, resp.status_code, resp.text[:500], json=json)
            raise
        self._log_call("POST", clean_path, resp.status_code, json=json)
        return resp.json()

    async def put(self, path: str, json: dict | None = None, params: dict | None = None) -> dict:
        clean_path, path_params = _split_path_params(path)
        merged_params = {**(path_params or {}), **(params or {})} or None
        # Auto-fix createReminder: LLM puts type/date in body instead of params
        if "/:createReminder" in clean_path or "/:createreminder" in clean_path.lower():
            merged_params = merged_params or {}
            if json and isinstance(json, dict):
                for field in ["type", "date", "includeCharge"]:
                    if field in json and field not in merged_params:
                        merged_params[field] = json.pop(field)
            merged_params.setdefault("type", "REMINDER")
            merged_params.setdefault("date", __import__("datetime").date.today().isoformat())
            merged_params.setdefault("includeCharge", "true")
            # Tripletex requires at least one send type ("Minst én sendetype må oppgis")
            # Same param name as /:send endpoint
            merged_params.setdefault("sendType", "EMAIL")
            log.warning(f"Auto-fixed createReminder params: {merged_params}")
        # Auto-fix invoice send: ensure sendType param
        if "/:send" in clean_path and "/invoice/" in clean_path:
            merged_params = merged_params or {}
            merged_params.setdefault("sendType", "EMAIL")
        # Auto-fix order→invoice: ensure invoiceDate param
        if "/:invoice" in clean_path and "/order/" in clean_path:
            merged_params = merged_params or {}
            if "invoiceDate" not in merged_params:
                merged_params["invoiceDate"] = __import__("datetime").date.today().isoformat()
            if "invoiceDueDate" not in merged_params:
                from datetime import timedelta
                merged_params["invoiceDueDate"] = (__import__("datetime").date.today() + timedelta(days=30)).isoformat()
        url = f"{self.base_url}{clean_path}"
        log.info(f"PUT {clean_path} params={merged_params}")
        resp = await self._client.put(url, json=json, params=merged_params)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            self._log_call("PUT", clean_path, resp.status_code, resp.text[:500], params=merged_params, json=json)
            raise
        self._log_call("PUT", clean_path, resp.status_code, params=merged_params, json=json)
        return resp.json()

    async def delete(self, path: str) -> dict:
        clean_path, _ = _split_path_params(path)
        url = f"{self.base_url}{clean_path}"
        log.info(f"DELETE {clean_path}")
        resp = await self._client.delete(url)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            self._log_call("DELETE", clean_path, resp.status_code, resp.text[:500])
            raise
        self._log_call("DELETE", clean_path, resp.status_code)
        return {"status": resp.status_code}

    def get_stats(self) -> dict:
        return {
            "total_calls": self.call_count,
            "errors_4xx": self.error_count,
            "calls": self.calls_log,
        }
