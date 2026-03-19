"""
Tripletex API client. All calls go through the provided proxy base_url.
Auth: Basic auth with username=0, password=session_token.
"""

import httpx
import logging
from typing import Any

log = logging.getLogger(__name__)


class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self._client = httpx.AsyncClient(
            auth=self.auth,
            timeout=60.0,
            headers={"Content-Type": "application/json"},
        )

    async def close(self):
        await self._client.aclose()

    async def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        log.info(f"GET {url} params={params}")
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def post(self, path: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        log.info(f"POST {url}")
        resp = await self._client.post(url, json=json)
        resp.raise_for_status()
        return resp.json()

    async def put(self, path: str, json: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        log.info(f"PUT {url}")
        resp = await self._client.put(url, json=json)
        resp.raise_for_status()
        return resp.json()

    async def delete(self, path: str) -> int:
        url = f"{self.base_url}{path}"
        log.info(f"DELETE {url}")
        resp = await self._client.delete(url)
        resp.raise_for_status()
        return resp.status_code

    # --- Convenience methods ---

    async def list_all(self, path: str, params: dict | None = None) -> list[dict]:
        """Fetch all pages from a list endpoint."""
        params = dict(params or {})
        params.setdefault("count", 1000)
        params.setdefault("from", 0)
        result = await self.get(path, params)
        return result.get("values", [])

    async def search(self, path: str, **kwargs) -> list[dict]:
        """Search endpoint with keyword args as query params."""
        return await self.list_all(path, params=kwargs)

    # --- Employee ---

    async def create_employee(self, data: dict) -> dict:
        return await self.post("/employee", json=data)

    async def get_employees(self, **params) -> list[dict]:
        return await self.list_all("/employee", params=params)

    async def update_employee(self, employee_id: int, data: dict) -> dict:
        return await self.put(f"/employee/{employee_id}", json=data)

    # --- Customer ---

    async def create_customer(self, data: dict) -> dict:
        return await self.post("/customer", json=data)

    async def get_customers(self, **params) -> list[dict]:
        return await self.list_all("/customer", params=params)

    # --- Product ---

    async def create_product(self, data: dict) -> dict:
        return await self.post("/product", json=data)

    async def get_products(self, **params) -> list[dict]:
        return await self.list_all("/product", params=params)

    # --- Invoice ---

    async def create_invoice(self, data: dict) -> dict:
        return await self.post("/invoice", json=data)

    async def get_invoices(self, **params) -> list[dict]:
        return await self.list_all("/invoice", params=params)

    async def create_order(self, data: dict) -> dict:
        return await self.post("/order", json=data)

    async def invoice_order(self, order_id: int) -> dict:
        """Convert an order to an invoice."""
        return await self.put(f"/invoice/{order_id}/:invoice")

    async def register_payment(self, invoice_id: int, data: dict) -> dict:
        return await self.post(f"/invoice/{invoice_id}/:payment", json=data)

    # --- Project ---

    async def create_project(self, data: dict) -> dict:
        return await self.post("/project", json=data)

    async def get_projects(self, **params) -> list[dict]:
        return await self.list_all("/project", params=params)

    # --- Department ---

    async def create_department(self, data: dict) -> dict:
        return await self.post("/department", json=data)

    async def get_departments(self, **params) -> list[dict]:
        return await self.list_all("/department", params=params)

    # --- Travel Expense ---

    async def create_travel_expense(self, data: dict) -> dict:
        return await self.post("/travelExpense", json=data)

    async def get_travel_expenses(self, **params) -> list[dict]:
        return await self.list_all("/travelExpense", params=params)

    async def delete_travel_expense(self, expense_id: int) -> int:
        return await self.delete(f"/travelExpense/{expense_id}")

    # --- Ledger ---

    async def get_accounts(self, **params) -> list[dict]:
        return await self.list_all("/ledger/account", params=params)

    async def create_voucher(self, data: dict) -> dict:
        return await self.post("/ledger/voucher", json=data)
