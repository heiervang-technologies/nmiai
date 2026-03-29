"""Tests for planner.py — classify_by_keywords and _extract_message_text.

These are pure functions that need no API key or network access.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"))

from planner import classify_by_keywords, _extract_message_text


# ---------------------------------------------------------------------------
# classify_by_keywords
# ---------------------------------------------------------------------------

class TestClassifyByKeywords:
    def test_invoice_prompt_returns_invoice(self):
        family, confidence = classify_by_keywords("Create an invoice for customer Acme AS")
        assert family == "invoice"

    def test_employee_prompt_returns_employee(self):
        family, confidence = classify_by_keywords("Add a new employee named Ola Nordmann")
        assert family == "employee"

    def test_customer_prompt_returns_customer(self):
        family, confidence = classify_by_keywords("Create a new customer with name Foo AS")
        assert family == "customer"

    def test_unrecognized_prompt_returns_none(self):
        family, confidence = classify_by_keywords("random gibberish xyz abc 123")
        assert family is None

    def test_unrecognized_prompt_confidence_is_low(self):
        _, confidence = classify_by_keywords("unknown nonsense")
        assert confidence == "low"

    def test_high_score_gives_high_confidence(self):
        # "faktura" is a strong invoice keyword in Norwegian
        _, confidence = classify_by_keywords("Lag en faktura til kunde Norges Tekniker AS med 3 ordrelinjer")
        assert confidence in ("high", "medium")

    def test_returns_tuple_of_two(self):
        result = classify_by_keywords("some prompt")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_family_is_str_or_none(self):
        family, _ = classify_by_keywords("hello")
        assert family is None or isinstance(family, str)

    def test_confidence_is_one_of_three_values(self):
        _, confidence = classify_by_keywords("any prompt text here")
        assert confidence in ("high", "medium", "low")

    def test_empty_string_returns_none(self):
        family, confidence = classify_by_keywords("")
        assert family is None
        assert confidence == "low"

    def test_department_prompt(self):
        family, _ = classify_by_keywords("Create a new department called IT")
        assert family == "department"

    def test_product_prompt(self):
        family, _ = classify_by_keywords("Create a new product with price 100 NOK")
        assert family == "product"

    def test_norwegian_invoice_keyword(self):
        # "faktura" is Norwegian for invoice
        family, _ = classify_by_keywords("Send faktura til kunde")
        assert family == "invoice"

    def test_bank_reconciliation_strong_keyword(self):
        # "bankutskrift" is a high-weight keyword
        family, _ = classify_by_keywords("Avstem bankutskrift for januar")
        assert family == "bank_reconciliation"


# ---------------------------------------------------------------------------
# _extract_message_text
# ---------------------------------------------------------------------------

class TestExtractMessageText:
    def test_none_returns_empty_string(self):
        assert _extract_message_text(None) == ""

    def test_string_returned_as_is(self):
        assert _extract_message_text("hello world") == "hello world"

    def test_empty_string(self):
        assert _extract_message_text("") == ""

    def test_list_of_dicts_with_text_joined(self):
        content = [{"text": "block1"}, {"text": "block2"}]
        result = _extract_message_text(content)
        assert "block1" in result
        assert "block2" in result

    def test_list_items_separated_by_newline(self):
        content = [{"text": "a"}, {"text": "b"}]
        result = _extract_message_text(content)
        assert result == "a\nb"

    def test_list_dict_missing_text_skipped(self):
        content = [{"type": "tool_use"}, {"text": "actual"}]
        result = _extract_message_text(content)
        assert result == "actual"

    def test_list_with_string_items(self):
        content = [{"text": "a"}, "b"]
        result = _extract_message_text(content)
        assert "a" in result
        assert "b" in result

    def test_empty_list_returns_empty(self):
        assert _extract_message_text([]) == ""

    def test_non_string_non_list_uses_str(self):
        result = _extract_message_text(42)
        assert result == "42"

    def test_returns_string(self):
        assert isinstance(_extract_message_text("text"), str)
        assert isinstance(_extract_message_text(None), str)
        assert isinstance(_extract_message_text([{"text": "x"}]), str)
