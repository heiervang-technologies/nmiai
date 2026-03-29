"""Tests for tasks/accounting/server/parse_openapi.py — extract_endpoint_card."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"),
)

from parse_openapi import extract_endpoint_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec(**overrides) -> dict:
    """Minimal OpenAPI spec with a components/schemas section."""
    base = {"components": {"schemas": {}}}
    base.update(overrides)
    return base


def _path_data(method: str = "get", summary: str = "Get something", params=None, body=None) -> dict:
    op = {"summary": summary}
    if params:
        op["parameters"] = params
    if body:
        op["requestBody"] = body
    return {method: op}


# ---------------------------------------------------------------------------
# Basic card structure
# ---------------------------------------------------------------------------

class TestExtractEndpointCard:
    def test_returns_list(self):
        cards = extract_endpoint_card(_spec(), "/test", _path_data())
        assert isinstance(cards, list)

    def test_single_get_returns_one_card(self):
        cards = extract_endpoint_card(_spec(), "/test", _path_data("get"))
        assert len(cards) == 1

    def test_card_has_required_keys(self):
        cards = extract_endpoint_card(_spec(), "/test", _path_data("get", "Fetch resource"))
        card = cards[0]
        assert "method" in card
        assert "path" in card
        assert "summary" in card
        assert "required_params" in card
        assert "optional_params" in card

    def test_method_is_uppercase(self):
        cards = extract_endpoint_card(_spec(), "/test", _path_data("post"))
        assert cards[0]["method"] == "POST"

    def test_path_preserved(self):
        cards = extract_endpoint_card(_spec(), "/invoice/{id}", _path_data())
        assert cards[0]["path"] == "/invoice/{id}"

    def test_summary_preserved(self):
        cards = extract_endpoint_card(_spec(), "/test", _path_data(summary="Get invoice by ID"))
        assert cards[0]["summary"] == "Get invoice by ID"

    def test_multiple_methods_returns_multiple_cards(self):
        path_data = {
            "get": {"summary": "Get"},
            "post": {"summary": "Create"},
        }
        cards = extract_endpoint_card(_spec(), "/test", path_data)
        assert len(cards) == 2
        methods = {c["method"] for c in cards}
        assert methods == {"GET", "POST"}

    def test_non_http_methods_ignored(self):
        path_data = {
            "get": {"summary": "Get"},
            "x-custom": {"summary": "Not a method"},
            "parameters": [],  # OpenAPI shared parameters
        }
        cards = extract_endpoint_card(_spec(), "/test", path_data)
        assert len(cards) == 1
        assert cards[0]["method"] == "GET"

    def test_empty_path_data_returns_empty(self):
        cards = extract_endpoint_card(_spec(), "/test", {})
        assert cards == []


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

class TestParameters:
    def test_required_path_param(self):
        params = [{"name": "id", "in": "path", "required": True}]
        cards = extract_endpoint_card(_spec(), "/test", _path_data(params=params))
        assert any("id" in p for p in cards[0]["required_params"])

    def test_optional_query_param(self):
        params = [{"name": "limit", "in": "query", "required": False}]
        cards = extract_endpoint_card(_spec(), "/test", _path_data(params=params))
        assert any("limit" in p for p in cards[0]["optional_params"])

    def test_required_param_not_in_optional(self):
        params = [{"name": "id", "in": "path", "required": True}]
        cards = extract_endpoint_card(_spec(), "/test", _path_data(params=params))
        card = cards[0]
        assert len(card["required_params"]) == 1
        assert len(card["optional_params"]) == 0

    def test_optional_capped_at_five(self):
        params = [{"name": f"p{i}", "in": "query"} for i in range(10)]
        cards = extract_endpoint_card(_spec(), "/test", _path_data(params=params))
        assert len(cards[0]["optional_params"]) <= 5

    def test_no_params_returns_empty_lists(self):
        cards = extract_endpoint_card(_spec(), "/test", _path_data())
        assert cards[0]["required_params"] == []
        assert cards[0]["optional_params"] == []


# ---------------------------------------------------------------------------
# Request body handling
# ---------------------------------------------------------------------------

class TestRequestBody:
    def test_body_with_inline_schema(self):
        body = {
            "content": {
                "application/json": {
                    "schema": {
                        "properties": {
                            "name": {"type": "string"},
                            "amount": {"type": "number"},
                        }
                    }
                }
            }
        }
        cards = extract_endpoint_card(_spec(), "/test", _path_data("post", body=body))
        assert cards[0]["body_fields"] is not None
        assert "name" in cards[0]["body_fields"]

    def test_body_with_ref_schema(self):
        spec = _spec(components={
            "schemas": {
                "Invoice": {
                    "properties": {
                        "id": {"type": "integer"},
                        "date": {"type": "string"},
                    }
                }
            }
        })
        body = {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Invoice"}
                }
            }
        }
        cards = extract_endpoint_card(spec, "/invoice", _path_data("post", body=body))
        assert cards[0]["body_fields"] is not None
        assert "id" in cards[0]["body_fields"]

    def test_no_body_gives_none_body_fields(self):
        cards = extract_endpoint_card(_spec(), "/test", _path_data("get"))
        assert cards[0]["body_fields"] is None

    def test_body_fields_truncated_at_500(self):
        long_props = {f"field{i}": {"type": "string"} for i in range(100)}
        body = {
            "content": {
                "application/json": {
                    "schema": {"properties": long_props}
                }
            }
        }
        cards = extract_endpoint_card(_spec(), "/test", _path_data("post", body=body))
        if cards[0]["body_fields"]:
            assert len(cards[0]["body_fields"]) <= 500
