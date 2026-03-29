"""Tests for tasks/accounting/server/actions.py — pure helper functions."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"),
)

from actions import (
    _build_project_update_payload,
    _error_mentions,
    _error_text,
    _today,
)


# ---------------------------------------------------------------------------
# _today
# ---------------------------------------------------------------------------

class TestToday:
    def test_returns_string(self):
        assert isinstance(_today(), str)

    def test_iso_format(self):
        result = _today()
        # Should be YYYY-MM-DD
        assert len(result) == 10
        assert result[4] == "-"
        assert result[7] == "-"

    def test_matches_date_today(self):
        assert _today() == date.today().isoformat()


# ---------------------------------------------------------------------------
# _error_text
# ---------------------------------------------------------------------------

class TestErrorText:
    def test_simple_exception(self):
        exc = ValueError("something went wrong")
        assert _error_text(exc) == "something went wrong"

    def test_exception_with_response_text(self):
        exc = Exception("base message")
        mock_resp = type("R", (), {"text": "detailed error from server"})()
        exc.response = mock_resp
        assert _error_text(exc) == "detailed error from server"

    def test_exception_with_broken_response(self):
        exc = Exception("fallback message")
        # response.text raises an exception
        class BadResp:
            @property
            def text(self):
                raise AttributeError("no text")
        exc.response = BadResp()
        # Should fall back to str(exc)
        assert _error_text(exc) == "fallback message"


# ---------------------------------------------------------------------------
# _error_mentions
# ---------------------------------------------------------------------------

class TestErrorMentions:
    def test_finds_needle_in_exception(self):
        exc = ValueError("NotFound: the customer does not exist")
        assert _error_mentions(exc, "notfound") is True

    def test_case_insensitive(self):
        exc = ValueError("UNAUTHORIZED access")
        assert _error_mentions(exc, "unauthorized") is True

    def test_returns_false_when_not_found(self):
        exc = ValueError("something unrelated")
        assert _error_mentions(exc, "notfound") is False

    def test_multiple_needles_any_matches(self):
        exc = ValueError("permission denied")
        assert _error_mentions(exc, "forbidden", "denied", "unauthorized") is True

    def test_multiple_needles_none_match(self):
        exc = ValueError("something else")
        assert _error_mentions(exc, "forbidden", "denied", "unauthorized") is False


# ---------------------------------------------------------------------------
# _build_project_update_payload
# ---------------------------------------------------------------------------

class TestBuildProjectUpdatePayload:
    def _project(self, **overrides) -> dict:
        base = {
            "id": 42,
            "version": 1,
            "name": "My Project",
            "number": "P-001",
            "displayName": "My Project",
            "description": "Test project",
            "projectManager": {"id": 10},
            "startDate": "2026-01-01",
            "endDate": "2026-12-31",
            "customer": {"id": 5},
            "isClosed": False,
            "isReadyForInvoicing": False,
            "isInternal": False,
            "isOffer": False,
            "isFixedPrice": False,
            "extra_field": "should be stripped",
            "another_extra": 999,
        }
        base.update(overrides)
        return base

    def test_strips_extra_fields(self):
        result = _build_project_update_payload(self._project())
        assert "extra_field" not in result
        assert "another_extra" not in result

    def test_keeps_id(self):
        result = _build_project_update_payload(self._project(id=99))
        assert result["id"] == 99

    def test_keeps_name(self):
        result = _build_project_update_payload(self._project(name="Test"))
        assert result["name"] == "Test"

    def test_none_fields_excluded(self):
        proj = self._project()
        proj["fixedprice"] = None
        result = _build_project_update_payload(proj)
        assert "fixedprice" not in result

    def test_zero_value_kept(self):
        proj = self._project()
        proj["fixedprice"] = 0
        result = _build_project_update_payload(proj)
        # Zero is falsy so gets excluded by the `if project_obj.get(field) is not None` check
        # and `get(field)` with value 0 returns 0, which is truthy in `is not None` check
        # Actually: 0 is not None → True, so it should be kept
        assert result.get("fixedprice") == 0

    def test_empty_project_returns_empty_dict(self):
        result = _build_project_update_payload({})
        assert result == {}

    def test_returns_dict(self):
        result = _build_project_update_payload(self._project())
        assert isinstance(result, dict)
