"""Tests for tasks/accounting/analyze_logs.py — additional pure helpers.

Covers: _normalize_text, _compile_keyword_pattern, collect_omissions, normalize_pattern.
All pure functions requiring no file system or network access.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting"))

from analyze_logs import (
    _compile_keyword_pattern,
    _normalize_text,
    collect_omissions,
    normalize_pattern,
)


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercases_ascii(self):
        assert _normalize_text("Hello World") == "hello world"

    def test_strips_diacritics_umlaut(self):
        result = _normalize_text("café")
        assert "e" in result
        assert "é" not in result

    def test_strips_diacritics_tilde(self):
        result = _normalize_text("España")
        assert "n" in result

    def test_lowercases_result(self):
        result = _normalize_text("UPPER")
        assert result == result.lower()

    def test_empty_string(self):
        assert _normalize_text("") == ""

    def test_plain_ascii_unchanged(self):
        assert _normalize_text("hello world") == "hello world"

    def test_returns_string(self):
        assert isinstance(_normalize_text("test"), str)


# ---------------------------------------------------------------------------
# _compile_keyword_pattern
# ---------------------------------------------------------------------------

class TestCompileKeywordPattern:
    def test_returns_compiled_pattern(self):
        result = _compile_keyword_pattern("invoice")
        assert hasattr(result, "search")

    def test_matches_exact_keyword(self):
        pattern = _compile_keyword_pattern("invoice")
        assert pattern.search("pay the invoice now")

    def test_no_match_on_unrelated(self):
        pattern = _compile_keyword_pattern("invoice")
        assert not pattern.search("nothing relevant here")

    def test_empty_keyword_returns_no_match_on_text(self):
        pattern = _compile_keyword_pattern("")
        # The $^ pattern doesn't match non-empty text
        assert not pattern.search("any text")
        assert not pattern.search("invoice payment")

    def test_multi_word_keyword_matches(self):
        pattern = _compile_keyword_pattern("credit note")
        assert pattern.search("create a credit note for the client")

    def test_case_insensitive(self):
        pattern = _compile_keyword_pattern("invoice")
        assert pattern.search("INVOICE")
        assert pattern.search("Invoice")

    def test_word_boundary_prefix_restriction(self):
        # Should not match in the middle of a word (word preceded by \w)
        pattern = _compile_keyword_pattern("in")
        # "in" should not match as part of "invoice" due to (?<!\w) lookbehind
        # but it should match standalone "in"
        assert pattern.search(" in the ")

    def test_allows_suffix_extension(self):
        # The pattern uses \w* so "invoices" should match the keyword "invoice"
        pattern = _compile_keyword_pattern("invoice")
        assert pattern.search("three invoices pending")


# ---------------------------------------------------------------------------
# collect_omissions
# ---------------------------------------------------------------------------

class TestCollectOmissions:
    def test_empty_string_returns_empty(self):
        assert collect_omissions("") == []

    def test_not_added_phrase_detected(self):
        result = collect_omissions("The item was not added to the system")
        assert "explicit_missing_followup" in result

    def test_open_phrase_detected(self):
        result = collect_omissions("The task is still open")
        assert "left_open" in result

    def test_ready_to_deliver_detected(self):
        result = collect_omissions("ready to deliver the goods")
        assert "not_delivered" in result

    def test_no_match_returns_empty(self):
        result = collect_omissions("The task was completed successfully")
        assert result == []

    def test_multiple_patterns_can_match(self):
        # Both "not added" and "open" present
        result = collect_omissions("Item not added, ticket still open")
        assert "explicit_missing_followup" in result
        assert "left_open" in result

    def test_none_message_returns_empty(self):
        # collect_omissions checks `if final_message`
        result = collect_omissions("")
        assert result == []

    def test_aun_no_detected(self):
        result = collect_omissions("aún no ha sido procesado")
        assert "explicit_missing_followup" in result

    def test_case_insensitive_not_added(self):
        result = collect_omissions("NOT ADDED to the registry")
        assert "explicit_missing_followup" in result


# ---------------------------------------------------------------------------
# normalize_pattern
# ---------------------------------------------------------------------------

class TestNormalizePattern:
    def test_field_and_message_both_present(self):
        entry = {"path": "/api/invoice", "field": "vatType", "message": "vatType is required"}
        result = normalize_pattern(entry)
        assert "/api/invoice" in result
        assert "vatType" in result
        assert "is required" in result

    def test_field_and_message_format(self):
        entry = {"path": "/api/invoice", "field": "amount", "message": "cannot be zero"}
        result = normalize_pattern(entry)
        assert "->" in result

    def test_message_only_with_path(self):
        entry = {"path": "/api/payment", "field": "", "message": "not found"}
        result = normalize_pattern(entry)
        assert "/api/payment" in result
        assert "not found" in result

    def test_message_only_no_path(self):
        entry = {"path": "", "field": "", "message": "Generic error"}
        result = normalize_pattern(entry)
        assert result == "Generic error"

    def test_path_only_no_message_no_field(self):
        entry = {"path": "/api/item", "field": "", "message": ""}
        result = normalize_pattern(entry)
        assert "/api/item" in result

    def test_all_empty_returns_unknown_error(self):
        entry = {"path": "", "field": "", "message": ""}
        result = normalize_pattern(entry)
        assert result == "unknown_error"

    def test_normalizes_whitespace_in_message(self):
        entry = {"path": "", "field": "", "message": "some   extra   spaces"}
        result = normalize_pattern(entry)
        # whitespace collapsed
        assert "  " not in result

    def test_returns_string(self):
        entry = {"path": "/x", "field": "f", "message": "m"}
        assert isinstance(normalize_pattern(entry), str)
