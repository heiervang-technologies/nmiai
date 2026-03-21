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
