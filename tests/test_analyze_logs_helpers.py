"""Tests for tasks/accounting/analyze_logs.py — pure helper functions.

Covers: parse_embedded_json, successful_writes, extract_validation_entries,
        infer_family_from_prompt.

These are pure/local functions that require no network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting"))

from analyze_logs import (
    extract_validation_entries,
    infer_family_from_prompt,
    load_playbooks,
    parse_embedded_json,
    successful_writes,
)


# ---------------------------------------------------------------------------
# parse_embedded_json
# ---------------------------------------------------------------------------

class TestParseEmbeddedJson:
    def test_none_returns_none(self):
        assert parse_embedded_json(None) is None

    def test_empty_string_returns_none(self):
        assert parse_embedded_json("") is None

    def test_valid_json_parsed(self):
        result = parse_embedded_json('{"key": "val"}')
        assert result == {"key": "val"}

    def test_valid_json_with_number(self):
        result = parse_embedded_json('{"count": 42}')
        assert result["count"] == 42

    def test_invalid_json_returns_none(self):
        assert parse_embedded_json("not json at all") is None

    def test_malformed_json_returns_none(self):
        assert parse_embedded_json('{"key": }') is None

    def test_json_list_parsed(self):
        result = parse_embedded_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_empty_object_parsed(self):
        result = parse_embedded_json('{}')
        assert result == {}

    def test_nested_json(self):
        data = {"validationMessages": [{"field": "firstName", "message": "required"}]}
        result = parse_embedded_json(json.dumps(data))
        assert result["validationMessages"][0]["field"] == "firstName"


# ---------------------------------------------------------------------------
# successful_writes
# ---------------------------------------------------------------------------

class TestSuccessfulWrites:
    def test_empty_list_returns_zero(self):
        assert successful_writes([]) == 0

    def test_get_200_not_counted(self):
        assert successful_writes([{"method": "GET", "status": 200}]) == 0

    def test_post_201_counted(self):
        assert successful_writes([{"method": "POST", "status": 201}]) == 1

    def test_put_200_counted(self):
        assert successful_writes([{"method": "PUT", "status": 200}]) == 1

    def test_delete_200_counted(self):
        assert successful_writes([{"method": "DELETE", "status": 200}]) == 1

    def test_post_422_not_counted(self):
        assert successful_writes([{"method": "POST", "status": 422}]) == 0

    def test_post_500_not_counted(self):
        assert successful_writes([{"method": "POST", "status": 500}]) == 0

    def test_mixed_calls(self):
        calls = [
            {"method": "POST", "status": 201},
            {"method": "PUT", "status": 200},
            {"method": "GET", "status": 200},
            {"method": "POST", "status": 422},
            {"method": "DELETE", "status": 200},
        ]
        assert successful_writes(calls) == 3

    def test_all_failed(self):
        calls = [{"method": "POST", "status": 400}, {"method": "PUT", "status": 500}]
        assert successful_writes(calls) == 0


# ---------------------------------------------------------------------------
# extract_validation_entries
# ---------------------------------------------------------------------------

class TestExtractValidationEntries:
    def test_empty_call_returns_empty(self):
        call = {"method": "GET", "status": 200, "path": "/v2/invoice"}
        assert extract_validation_entries(call) == []

    def test_validation_messages_extracted(self):
        call = {
            "method": "POST",
            "status": 422,
            "path": "/v2/employee",
            "error": json.dumps({
                "validationMessages": [
                    {"field": "firstName", "message": "required"},
                ]
            }),
        }
        entries = extract_validation_entries(call)
        assert len(entries) == 1
        assert entries[0]["field"] == "firstName"
        assert entries[0]["message"] == "required"
        assert entries[0]["path"] == "/v2/employee"

    def test_multiple_validation_messages(self):
        call = {
            "method": "POST",
            "status": 422,
            "path": "/v2/employee",
            "error": json.dumps({
                "validationMessages": [
                    {"field": "firstName", "message": "required"},
                    {"field": "email", "message": "invalid format"},
                ]
            }),
        }
        entries = extract_validation_entries(call)
        assert len(entries) == 2

    def test_plain_error_string_extracted(self):
        call = {"method": "POST", "status": 500, "path": "/v2/x", "error": "timeout"}
        entries = extract_validation_entries(call)
        assert len(entries) == 1
        assert entries[0]["message"] == "timeout"
        assert entries[0]["field"] == ""

    def test_4xx_status_with_no_error_adds_entry(self):
        call = {"method": "GET", "status": 404, "path": "/v2/invoice/999"}
        entries = extract_validation_entries(call)
        assert len(entries) == 1
        assert "404" in entries[0]["message"]


# ---------------------------------------------------------------------------
# infer_family_from_prompt + load_playbooks
# ---------------------------------------------------------------------------

class TestInferFamilyFromPrompt:
    @pytest.fixture(scope="class")
    def playbooks(self):
        return load_playbooks()

    def test_load_playbooks_returns_dict(self, playbooks):
        assert isinstance(playbooks, dict)

    def test_playbooks_non_empty(self, playbooks):
        assert len(playbooks) > 0

    def test_invoice_keyword_detected(self, playbooks):
        family, confidence, score = infer_family_from_prompt(
            "Create an invoice for customer Acme AS",
            playbooks,
        )
        assert family == "invoice"

    def test_returns_none_for_gibberish(self, playbooks):
        family, confidence, score = infer_family_from_prompt(
            "xyzxyz random gibberish 12345",
            playbooks,
        )
        assert family is None

    def test_confidence_low_for_no_match(self, playbooks):
        _, confidence, _ = infer_family_from_prompt("no match here", playbooks)
        assert confidence == "low"

    def test_score_is_float(self, playbooks):
        _, _, score = infer_family_from_prompt("invoice customer", playbooks)
        assert isinstance(score, float)

    def test_returns_tuple_of_three(self, playbooks):
        result = infer_family_from_prompt("some text", playbooks)
        assert isinstance(result, tuple)
        assert len(result) == 3
