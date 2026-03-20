"""
Tripletex API client with call tracking.
All calls go through the provided proxy base_url.
Auth: Basic auth with username=0, password=session_token.
"""

import httpx
import json as json_mod
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import Any

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

    async def close(self):
        await self._client.aclose()

    def _log_call(self, method: str, path: str, status: int, error: str = None):
        self.call_count += 1
        if error or (400 <= status < 500):
            self.error_count += 1
        self.calls_log.append({
            "method": method, "path": path,
            "status": status, "error": error,
        })

    async def get(self, path: str, params: dict | None = None) -> dict:
        clean_path, path_params = _split_path_params(path)
        merged_params = {**(path_params or {}), **(params or {})} or None
        url = f"{self.base_url}{clean_path}"
        log.info(f"GET {clean_path} params={merged_params}")
        try:
            resp = await self._client.get(url, params=merged_params)
            self._log_call("GET", clean_path, resp.status_code)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            self._log_call("GET", clean_path, e.response.status_code, e.response.text[:500])
            raise

    async def post(self, path: str, json: dict | None = None) -> dict:
        clean_path, path_params = _split_path_params(path)
        # If LLM put fields in query params instead of body, convert to body
        if path_params and not json:
            log.warning(f"POST {clean_path}: converting query params to body: {path_params}")
            json = path_params
        url = f"{self.base_url}{clean_path}"
        log.info(f"POST {clean_path} body_keys={list((json or {}).keys())}")
        try:
            resp = await self._client.post(url, json=json)
            self._log_call("POST", clean_path, resp.status_code)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            self._log_call("POST", clean_path, e.response.status_code, e.response.text[:500])
            raise

    async def put(self, path: str, json: dict | None = None, params: dict | None = None) -> dict:
        clean_path, path_params = _split_path_params(path)
        merged_params = {**(path_params or {}), **(params or {})} or None
        url = f"{self.base_url}{clean_path}"
        log.info(f"PUT {clean_path} params={merged_params}")
        try:
            resp = await self._client.put(url, json=json, params=merged_params)
            self._log_call("PUT", clean_path, resp.status_code)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            self._log_call("PUT", clean_path, e.response.status_code, e.response.text[:500])
            raise

    async def delete(self, path: str) -> dict:
        clean_path, _ = _split_path_params(path)
        url = f"{self.base_url}{clean_path}"
        log.info(f"DELETE {clean_path}")
        try:
            resp = await self._client.delete(url)
            self._log_call("DELETE", clean_path, resp.status_code)
            resp.raise_for_status()
            return {"status": resp.status_code}
        except httpx.HTTPStatusError as e:
            self._log_call("DELETE", clean_path, e.response.status_code, e.response.text[:500])
            raise

    def get_stats(self) -> dict:
        return {
            "total_calls": self.call_count,
            "errors_4xx": self.error_count,
            "calls": self.calls_log,
        }
