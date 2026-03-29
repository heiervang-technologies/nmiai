"""Tests for tasks/accounting/server/tripletex_client.py — pure-Python helpers.

Covers: _split_path_params, _payload_preview, TripletexClient.__init__,
        TripletexClient._log_call, TripletexClient.get_stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Point import path at the server directory so we can import without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"))

from tripletex_client import (
    TripletexClient,
    _payload_preview,
    _split_path_params,
)


# ---------------------------------------------------------------------------
# _split_path_params
# ---------------------------------------------------------------------------

class TestSplitPathParams:
    def test_no_query_string_returns_original_path_empty_dict(self):
        path, params = _split_path_params("/v2/invoice")
        assert path == "/v2/invoice"
        assert params == {}

    def test_single_param_extracted(self):
        path, params = _split_path_params("/v2/invoice?invoiceDateFrom=2020-01-01")
        assert path == "/v2/invoice"
        assert params["invoiceDateFrom"] == "2020-01-01"

    def test_multiple_params_extracted(self):
        path, params = _split_path_params(
            "/v2/invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31"
        )
        assert path == "/v2/invoice"
        assert params["invoiceDateFrom"] == "2020-01-01"
        assert params["invoiceDateTo"] == "2030-12-31"

    def test_multi_value_param_returns_list(self):
        path, params = _split_path_params("/v2/order?status=open&status=closed")
        assert path == "/v2/order"
        assert params["status"] == ["open", "closed"]

    def test_single_value_returned_as_scalar(self):
        path, params = _split_path_params("/v2/employee?id=42")
        assert params["id"] == "42"
        assert not isinstance(params["id"], list)

    def test_empty_path_no_params(self):
        path, params = _split_path_params("/")
        assert path == "/"
        assert params == {}

    def test_only_question_mark_no_params(self):
        path, params = _split_path_params("/v2/something?")
        assert path == "/v2/something"
        assert isinstance(params, dict)

    def test_path_preserved_exactly(self):
        path, params = _split_path_params("/v2/ledger/account?from=0&count=100")
        assert path == "/v2/ledger/account"

    def test_no_mutation_of_original_string(self):
        original = "/v2/invoice?a=1"
        _split_path_params(original)
        assert original == "/v2/invoice?a=1"


# ---------------------------------------------------------------------------
# _payload_preview
# ---------------------------------------------------------------------------

class TestPayloadPreview:
    def test_none_returns_none(self):
        assert _payload_preview(None) is None

    def test_small_dict_returned_normalized(self):
        result = _payload_preview({"key": "value", "num": 42})
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_empty_dict_returned(self):
        result = _payload_preview({})
        assert result == {}

    def test_small_list_returned(self):
        result = _payload_preview([1, 2, 3])
        assert result == [1, 2, 3]

    def test_long_payload_truncated(self):
        big = "x" * 2000
        result = _payload_preview(big)
        # JSON-encoded string is longer than 1200 chars → truncated
        assert isinstance(result, str)
        assert result.endswith("...[truncated]")

    def test_payload_within_max_chars_not_truncated(self):
        short = {"a": "b"}
        result = _payload_preview(short, max_chars=1200)
        # Short payload → returned as-is (normalized dict)
        assert "...[truncated]" not in str(result)

    def test_custom_max_chars_triggers_truncation(self):
        payload = "hello world"
        # JSON of "hello world" is 13 chars; set max_chars=5 → truncated
        result = _payload_preview(payload, max_chars=5)
        assert isinstance(result, str)
        assert result.endswith("...[truncated]")

    def test_non_serializable_object_handled(self):
        class Custom:
            pass
        result = _payload_preview(Custom())
        # default=str fallback should produce something non-None
        assert result is not None

    def test_nested_dict_normalized(self):
        nested = {"outer": {"inner": 99}}
        result = _payload_preview(nested)
        assert result["outer"]["inner"] == 99


# ---------------------------------------------------------------------------
# TripletexClient.__init__
# ---------------------------------------------------------------------------

class TestTripletexClientInit:
    def test_base_url_trailing_slash_stripped(self):
        client = TripletexClient("http://localhost:8080/", "token123")
        assert client.base_url == "http://localhost:8080"

    def test_base_url_without_slash_unchanged(self):
        client = TripletexClient("http://localhost:8080", "token123")
        assert client.base_url == "http://localhost:8080"

    def test_auth_tuple_uses_zero_as_username(self):
        client = TripletexClient("http://localhost", "mytoken")
        assert client.auth == ("0", "mytoken")

    def test_initial_call_count_zero(self):
        client = TripletexClient("http://localhost", "t")
        assert client.call_count == 0

    def test_initial_error_count_zero(self):
        client = TripletexClient("http://localhost", "t")
        assert client.error_count == 0

    def test_initial_calls_log_empty(self):
        client = TripletexClient("http://localhost", "t")
        assert client.calls_log == []

    def test_reversal_invoice_id_initially_none(self):
        client = TripletexClient("http://localhost", "t")
        assert client._reversal_invoice_id is None


# ---------------------------------------------------------------------------
# TripletexClient._log_call
# ---------------------------------------------------------------------------

class TestLogCall:
    def _client(self) -> TripletexClient:
        return TripletexClient("http://localhost", "t")

    def test_increments_call_count(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 200)
        assert c.call_count == 1

    def test_multiple_calls_accumulate(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 200)
        c._log_call("POST", "/v2/employee", 201)
        assert c.call_count == 2

    def test_4xx_status_increments_error_count(self):
        c = self._client()
        c._log_call("POST", "/v2/employee", 422)
        assert c.error_count == 1

    def test_400_is_4xx(self):
        c = self._client()
        c._log_call("DELETE", "/v2/invoice/1", 400)
        assert c.error_count == 1

    def test_499_is_4xx(self):
        c = self._client()
        c._log_call("GET", "/v2/x", 499)
        assert c.error_count == 1

    def test_500_not_4xx_no_error_increment(self):
        c = self._client()
        c._log_call("GET", "/v2/x", 500)
        assert c.error_count == 0

    def test_error_string_increments_error_count(self):
        c = self._client()
        c._log_call("GET", "/v2/x", 200, error="network error")
        assert c.error_count == 1

    def test_200_no_error_no_increment(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 200)
        assert c.error_count == 0

    def test_call_appended_to_log(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 200)
        assert len(c.calls_log) == 1
        entry = c.calls_log[0]
        assert entry["method"] == "GET"
        assert entry["path"] == "/v2/invoice"
        assert entry["status"] == 200

    def test_params_included_when_not_none(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 200, params={"from": "0"})
        assert "params" in c.calls_log[0]
        assert c.calls_log[0]["params"]["from"] == "0"

    def test_json_included_when_not_none(self):
        c = self._client()
        c._log_call("POST", "/v2/employee", 201, json={"firstName": "Ola"})
        assert "json" in c.calls_log[0]
        assert c.calls_log[0]["json"]["firstName"] == "Ola"

    def test_none_params_not_in_entry(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 200, params=None)
        assert "params" not in c.calls_log[0]

    def test_none_json_not_in_entry(self):
        c = self._client()
        c._log_call("POST", "/v2/invoice", 201, json=None)
        assert "json" not in c.calls_log[0]

    def test_error_field_in_entry(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 500, error="timeout")
        assert c.calls_log[0]["error"] == "timeout"

    def test_error_none_stored_in_entry(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 200)
        assert c.calls_log[0]["error"] is None


# ---------------------------------------------------------------------------
# TripletexClient.get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def _client(self) -> TripletexClient:
        return TripletexClient("http://localhost", "t")

    def test_empty_stats(self):
        c = self._client()
        stats = c.get_stats()
        assert stats["total_calls"] == 0
        assert stats["errors_4xx"] == 0
        assert stats["calls"] == []

    def test_stats_reflect_logged_calls(self):
        c = self._client()
        c._log_call("GET", "/v2/invoice", 200)
        c._log_call("POST", "/v2/employee", 422)
        stats = c.get_stats()
        assert stats["total_calls"] == 2
        assert stats["errors_4xx"] == 1

    def test_calls_list_in_stats(self):
        c = self._client()
        c._log_call("DELETE", "/v2/invoice/1", 200)
        stats = c.get_stats()
        assert len(stats["calls"]) == 1
        assert stats["calls"][0]["method"] == "DELETE"
