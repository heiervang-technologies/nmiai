"""Tests for tasks/accounting/server/mock_tripletex.py — state and helpers.

Covers: MockState construction, log_call, log_validation, get_assertions,
reset, _wrap, _wrap_list.
All pure / stateful-but-in-memory — no network or file system access.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock starlette before importing mock_tripletex (it's only used for HTTP routing,
# not needed for the state/helper functions we're testing).
for mod in ("starlette", "starlette.applications", "starlette.requests",
            "starlette.responses", "starlette.routing"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"))

from mock_tripletex import MockState, _wrap, _wrap_list


# ---------------------------------------------------------------------------
# _wrap
# ---------------------------------------------------------------------------

class TestWrap:
    def test_wraps_dict(self):
        result = _wrap({"id": 1, "name": "foo"})
        assert result == {"value": {"id": 1, "name": "foo"}}

    def test_wraps_none(self):
        assert _wrap(None) == {"value": None}

    def test_wraps_int(self):
        assert _wrap(42) == {"value": 42}


# ---------------------------------------------------------------------------
# _wrap_list
# ---------------------------------------------------------------------------

class TestWrapList:
    def test_empty_list(self):
        result = _wrap_list([])
        assert result["fullResultSize"] == 0
        assert result["count"] == 0
        assert result["values"] == []

    def test_single_item(self):
        result = _wrap_list([{"id": 1}])
        assert result["fullResultSize"] == 1
        assert result["count"] == 1
        assert result["values"] == [{"id": 1}]

    def test_multiple_items(self):
        items = [{"id": i} for i in range(5)]
        result = _wrap_list(items)
        assert result["fullResultSize"] == 5
        assert result["count"] == 5
        assert result["values"] == items

    def test_from_is_zero(self):
        result = _wrap_list([{"id": 1}])
        assert result["from"] == 0


# ---------------------------------------------------------------------------
# MockState construction
# ---------------------------------------------------------------------------

class TestMockStateConstruction:
    def test_empty_calls(self):
        state = MockState()
        assert state.calls == []

    def test_empty_validation_errors(self):
        state = MockState()
        assert state.validation_errors == []

    def test_has_entity_categories(self):
        state = MockState()
        for key in ("employee", "customer", "supplier", "invoice", "project", "voucher"):
            assert key in state.entities
            assert state.entities[key] == []

    def test_has_company(self):
        state = MockState()
        assert state.company["name"] == "Test Company AS"

    def test_has_accounts(self):
        state = MockState()
        assert len(state.accounts) > 0

    def test_has_vat_types(self):
        state = MockState()
        assert len(state.vat_types) > 0


# ---------------------------------------------------------------------------
# MockState.log_call
# ---------------------------------------------------------------------------

class TestLogCall:
    def test_appends_to_calls(self):
        state = MockState()
        state.log_call("GET", "/employee", params={})
        assert len(state.calls) == 1

    def test_stores_method_and_path(self):
        state = MockState()
        state.log_call("POST", "/invoice", body={"amount": 100})
        c = state.calls[0]
        assert c["method"] == "POST"
        assert c["path"] == "/invoice"

    def test_multiple_calls(self):
        state = MockState()
        state.log_call("GET", "/a")
        state.log_call("POST", "/b")
        state.log_call("PUT", "/c")
        assert len(state.calls) == 3


# ---------------------------------------------------------------------------
# MockState.log_validation
# ---------------------------------------------------------------------------

class TestLogValidation:
    def test_appends_to_validation_errors(self):
        state = MockState()
        state.log_validation("/invoice", "Missing VAT")
        assert len(state.validation_errors) == 1

    def test_stores_issue_text(self):
        state = MockState()
        state.log_validation("/customer", "Duplicate customer", severity="error")
        err = state.validation_errors[0]
        assert err["issue"] == "Duplicate customer"
        assert err["severity"] == "error"

    def test_default_severity_warning(self):
        state = MockState()
        state.log_validation("/foo", "some issue")
        assert state.validation_errors[0]["severity"] == "warning"


# ---------------------------------------------------------------------------
# MockState.get_assertions
# ---------------------------------------------------------------------------

class TestGetAssertions:
    def test_clean_state_no_issues(self):
        state = MockState()
        result = state.get_assertions()
        assert result["error_count"] == 0
        assert result["total_calls"] == 0

    def test_returns_dict_with_required_keys(self):
        state = MockState()
        result = state.get_assertions()
        for key in ("issues", "entity_counts", "total_calls", "error_count", "warning_count"):
            assert key in result

    def test_excessive_calls_triggers_warning(self):
        state = MockState()
        for i in range(25):  # > 20
            state.log_call("GET", "/employee")
        result = state.get_assertions()
        assert result["warning_count"] > 0

    def test_runaway_loop_triggers_error(self):
        state = MockState()
        for i in range(45):  # > 40
            state.log_call("GET", "/employee")
        result = state.get_assertions()
        assert result["error_count"] > 0

    def test_repeated_post_triggers_error(self):
        state = MockState()
        for i in range(6):  # > 4 non-GET calls with same signature
            state.log_call("POST", "/invoice")
        result = state.get_assertions()
        assert result["error_count"] > 0

    def test_validation_errors_included_in_issues(self):
        state = MockState()
        state.log_validation("/foo", "bad thing", severity="error")
        result = state.get_assertions()
        assert result["error_count"] >= 1

    def test_entity_counts_populated(self):
        state = MockState()
        state.entities["customer"].append({"id": 1, "name": "Corp"})
        result = state.get_assertions()
        assert result["entity_counts"].get("customer", 0) == 1


# ---------------------------------------------------------------------------
# MockState.reset
# ---------------------------------------------------------------------------

class TestMockStateReset:
    def test_reset_clears_calls(self):
        state = MockState()
        state.log_call("GET", "/foo")
        state.reset()
        assert state.calls == []

    def test_reset_clears_validation_errors(self):
        state = MockState()
        state.log_validation("/foo", "issue")
        state.reset()
        assert state.validation_errors == []

    def test_reset_clears_entities(self):
        state = MockState()
        state.entities["customer"].append({"id": 1})
        state.reset()
        assert state.entities["customer"] == []
