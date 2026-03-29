"""Tests for tasks/accounting/server/parse_openapi.py — pure helper functions.

Covers: extract_endpoint_card, FAMILY_PATHS structure.
These are pure functions that require no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"))

from parse_openapi import extract_endpoint_card, FAMILY_PATHS


# ---------------------------------------------------------------------------
# extract_endpoint_card
# ---------------------------------------------------------------------------

class TestExtractEndpointCard:
    def _spec(self, schemas=None):
        """Minimal OpenAPI spec."""
        return {"components": {"schemas": schemas or {}}}

    def test_get_method_returns_card(self):
        path_data = {
            "get": {
                "summary": "List employees",
                "parameters": [],
            }
        }
        cards = extract_endpoint_card(self._spec(), "/employee", path_data)
        assert len(cards) == 1
        assert cards[0]["method"] == "GET"

    def test_post_method_returns_card(self):
        path_data = {
            "post": {
                "summary": "Create employee",
                "parameters": [],
            }
        }
        cards = extract_endpoint_card(self._spec(), "/employee", path_data)
        assert len(cards) == 1
        assert cards[0]["method"] == "POST"

    def test_put_and_delete_methods_included(self):
        path_data = {
            "put": {"summary": "Update", "parameters": []},
            "delete": {"summary": "Delete", "parameters": []},
        }
        cards = extract_endpoint_card(self._spec(), "/employee/{id}", path_data)
        methods = {c["method"] for c in cards}
        assert "PUT" in methods
        assert "DELETE" in methods

    def test_non_http_keys_ignored(self):
        path_data = {
            "get": {"summary": "List", "parameters": []},
            "parameters": [{"name": "id", "in": "path"}],  # OpenAPI shared param
            "summary": "top level",
        }
        cards = extract_endpoint_card(self._spec(), "/items", path_data)
        assert all(c["method"] in ("GET", "POST", "PUT", "DELETE") for c in cards)

    def test_path_field_set_correctly(self):
        path_data = {"get": {"summary": "Fetch", "parameters": []}}
        cards = extract_endpoint_card(self._spec(), "/v2/invoice", path_data)
        assert cards[0]["path"] == "/v2/invoice"

    def test_summary_extracted(self):
        path_data = {"get": {"summary": "Get invoice by ID", "parameters": []}}
        cards = extract_endpoint_card(self._spec(), "/invoice/{id}", path_data)
        assert cards[0]["summary"] == "Get invoice by ID"

    def test_no_summary_gives_empty_string(self):
        path_data = {"get": {"parameters": []}}
        cards = extract_endpoint_card(self._spec(), "/x", path_data)
        assert cards[0]["summary"] == ""

    def test_required_param_extracted(self):
        path_data = {
            "get": {
                "summary": "",
                "parameters": [
                    {"name": "id", "in": "path", "required": True},
                ],
            }
        }
        cards = extract_endpoint_card(self._spec(), "/employee/{id}", path_data)
        assert any("id" in p for p in cards[0]["required_params"])

    def test_optional_query_param_extracted(self):
        path_data = {
            "get": {
                "summary": "",
                "parameters": [
                    {"name": "from", "in": "query"},
                    {"name": "count", "in": "query"},
                ],
            }
        }
        cards = extract_endpoint_card(self._spec(), "/employee", path_data)
        assert len(cards[0]["optional_params"]) == 2

    def test_required_param_not_in_optional(self):
        path_data = {
            "get": {
                "summary": "",
                "parameters": [
                    {"name": "id", "in": "path", "required": True},
                    {"name": "fields", "in": "query"},
                ],
            }
        }
        cards = extract_endpoint_card(self._spec(), "/employee/{id}", path_data)
        optional_names = [p for p in cards[0]["optional_params"] if "id" in p and "REQUIRED" in p]
        assert not optional_names  # required param should not appear in optional

    def test_body_fields_from_ref_schema(self):
        spec = {
            "components": {
                "schemas": {
                    "Employee": {
                        "properties": {
                            "firstName": {"type": "string"},
                            "lastName": {"type": "string"},
                            "email": {"type": "string"},
                        }
                    }
                }
            }
        }
        path_data = {
            "post": {
                "summary": "Create",
                "parameters": [],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Employee"}
                        }
                    }
                },
            }
        }
        cards = extract_endpoint_card(spec, "/employee", path_data)
        assert cards[0]["body_fields"] is not None
        assert "firstName" in cards[0]["body_fields"]

    def test_body_fields_none_when_no_request_body(self):
        path_data = {"get": {"summary": "", "parameters": []}}
        cards = extract_endpoint_card(self._spec(), "/x", path_data)
        assert cards[0]["body_fields"] is None

    def test_empty_path_data_returns_empty_list(self):
        cards = extract_endpoint_card(self._spec(), "/x", {})
        assert cards == []

    def test_multiple_methods_returns_multiple_cards(self):
        path_data = {
            "get": {"summary": "List", "parameters": []},
            "post": {"summary": "Create", "parameters": []},
        }
        cards = extract_endpoint_card(self._spec(), "/employee", path_data)
        assert len(cards) == 2


# ---------------------------------------------------------------------------
# FAMILY_PATHS structure
# ---------------------------------------------------------------------------

class TestFamilyPaths:
    def test_is_dict(self):
        assert isinstance(FAMILY_PATHS, dict)

    def test_non_empty(self):
        assert len(FAMILY_PATHS) > 0

    def test_invoice_family_present(self):
        assert "invoice" in FAMILY_PATHS

    def test_employee_family_present(self):
        assert "employee" in FAMILY_PATHS

    def test_each_family_has_list_of_paths(self):
        for family, paths in FAMILY_PATHS.items():
            assert isinstance(paths, list), f"{family} should be a list"
            assert len(paths) > 0, f"{family} should have at least one path"

    def test_all_paths_start_with_slash(self):
        for family, paths in FAMILY_PATHS.items():
            for path in paths:
                assert path.startswith("/"), f"{family}: {path} should start with /"

    def test_bank_family_present(self):
        assert "bank" in FAMILY_PATHS

    def test_travel_expense_family_present(self):
        assert "travel_expense" in FAMILY_PATHS
