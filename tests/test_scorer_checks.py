"""Tests for tasks/accounting/server/scorer_checks.py — pure helper functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"),
)

from scorer_checks import (
    _check_postings_balance,
    _extract_amounts,
    _extract_email,
    _extract_name,
    _extract_order_lines,
    _extract_org_number,
    _is_credit_note_task,
    _is_order_task,
    _is_payment_task,
    _is_reversal_task,
    _is_send_task,
    format_score_report,
)


# ---------------------------------------------------------------------------
# _extract_name
# ---------------------------------------------------------------------------

class TestExtractName:
    def test_english_for_prefix(self):
        first, last = _extract_name("Create invoice for Ola Nordmann")
        assert first == "Ola"
        assert last == "Nordmann"

    def test_german_fur_prefix(self):
        first, last = _extract_name("Erstellen Sie eine Rechnung für Max Mustermann")
        assert first == "Max"
        assert last == "Mustermann"

    def test_no_name_returns_none(self):
        first, last = _extract_name("Create an invoice without a name")
        assert first is None
        assert last is None

    def test_lowercase_first_letter_not_matched(self):
        first, last = _extract_name("for ola nordmann")
        assert first is None  # lowercase first letter doesn't match [A-ZÆØÅ]

    def test_nordic_letters(self):
        first, last = _extract_name("Create for Åse Ødegård")
        assert first == "Åse"
        assert last == "Ødegård"


# ---------------------------------------------------------------------------
# _extract_org_number
# ---------------------------------------------------------------------------

class TestExtractOrgNumber:
    def test_extracts_9_digit_number(self):
        result = _extract_org_number("Org nr: 123456789")
        assert result == "123456789"

    def test_extracts_with_org_nummer_prefix(self):
        result = _extract_org_number("org nummer: 987654321")
        assert result == "987654321"

    def test_no_org_number_returns_none(self):
        result = _extract_org_number("No number here")
        assert result is None

    def test_8_digit_not_matched(self):
        result = _extract_org_number("org nr: 12345678")
        assert result is None

    def test_case_insensitive(self):
        result = _extract_org_number("ORG NR 111222333")
        assert result == "111222333"


# ---------------------------------------------------------------------------
# _extract_email
# ---------------------------------------------------------------------------

class TestExtractEmail:
    def test_simple_email(self):
        assert _extract_email("Contact: user@example.com") == "user@example.com"

    def test_email_with_dots_and_plus(self):
        result = _extract_email("Send to first.last+tag@sub.domain.org")
        assert result == "first.last+tag@sub.domain.org"

    def test_no_email_returns_none(self):
        assert _extract_email("No email here") is None

    def test_email_in_context(self):
        result = _extract_email("Please send to invoice@company.no for billing")
        assert result == "invoice@company.no"


# ---------------------------------------------------------------------------
# _extract_amounts
# ---------------------------------------------------------------------------

class TestExtractAmounts:
    def test_simple_nok_amount(self):
        amounts = _extract_amounts("Payment of 1000 NOK")
        assert 1000.0 in amounts

    def test_kr_suffix(self):
        amounts = _extract_amounts("Amount: 500 kr")
        assert 500.0 in amounts

    def test_multiple_amounts(self):
        amounts = _extract_amounts("500 NOK and 1000 kr")
        assert len(amounts) == 2

    def test_decimal_amount(self):
        amounts = _extract_amounts("Total 1234.50 NOK")
        assert any(abs(a - 1234.50) < 0.01 for a in amounts)

    def test_no_amounts_returns_empty(self):
        amounts = _extract_amounts("No money mentioned here")
        assert amounts == []


# ---------------------------------------------------------------------------
# Boolean task detectors
# ---------------------------------------------------------------------------

class TestIsPaymentTask:
    def test_register_payment(self):
        assert _is_payment_task("register payment of 500 NOK") is True

    def test_betaling_norwegian(self):
        assert _is_payment_task("registrer betaling") is True

    def test_unrelated_returns_false(self):
        assert _is_payment_task("Create a new customer") is False


class TestIsCreditNoteTask:
    def test_credit_note_english(self):
        assert _is_credit_note_task("Issue a credit note") is True

    def test_kreditnota_norwegian(self):
        assert _is_credit_note_task("opprett kreditnota") is True

    def test_unrelated_returns_false(self):
        assert _is_credit_note_task("Send invoice to customer") is False


class TestIsReversalTask:
    def test_reversal_english(self):
        assert _is_reversal_task("reverse the payment") is True

    def test_returned_by_bank(self):
        assert _is_reversal_task("payment returned by the bank") is True

    def test_unrelated_returns_false(self):
        assert _is_reversal_task("Create new invoice") is False


class TestIsOrderTask:
    def test_order_english(self):
        assert _is_order_task("Create a new order for client") is True

    def test_bestilling_norwegian(self):
        assert _is_order_task("opprett bestilling") is True

    def test_convert_to_invoice(self):
        assert _is_order_task("convert to invoice") is True

    def test_unrelated_returns_false(self):
        assert _is_order_task("register payment") is False


class TestIsSendTask:
    def test_send_invoice(self):
        assert _is_send_task("send invoice to customer") is True

    def test_send_it(self):
        assert _is_send_task("send it to the client") is True

    def test_unrelated_returns_false(self):
        assert _is_send_task("create new customer") is False


# ---------------------------------------------------------------------------
# _extract_order_lines
# ---------------------------------------------------------------------------

class TestExtractOrderLines:
    def test_empty_dict_returns_empty(self):
        assert _extract_order_lines({}) == []

    def test_extracts_from_invoice(self):
        bodies = {"invoice": [{"orderLines": [{"product": "X", "qty": 1}]}]}
        result = _extract_order_lines(bodies)
        assert len(result) == 1
        assert result[0]["product"] == "X"

    def test_extracts_from_nested_orders(self):
        bodies = {
            "order": [{"orders": [{"orderLines": [{"product": "A"}, {"product": "B"}]}]}]
        }
        result = _extract_order_lines(bodies)
        assert len(result) == 2

    def test_other_entity_types_ignored(self):
        bodies = {"customer": [{"orderLines": [{"product": "X"}]}]}
        assert _extract_order_lines(bodies) == []


# ---------------------------------------------------------------------------
# _check_postings_balance
# ---------------------------------------------------------------------------

class TestCheckPostingsBalance:
    def test_empty_list_returns_true(self):
        assert _check_postings_balance([]) is True

    def test_balanced_postings(self):
        bodies = [{"postings": [{"amountGross": 100}, {"amountGross": -100}]}]
        assert _check_postings_balance(bodies) is True

    def test_unbalanced_postings(self):
        bodies = [{"postings": [{"amountGross": 100}, {"amountGross": -50}]}]
        assert _check_postings_balance(bodies) is False

    def test_zero_postings_returns_true(self):
        bodies = [{"postings": []}]
        assert _check_postings_balance(bodies) is True

    def test_no_postings_key_skipped(self):
        bodies = [{"other": "data"}]
        assert _check_postings_balance(bodies) is True


# ---------------------------------------------------------------------------
# format_score_report
# ---------------------------------------------------------------------------

class TestFormatScoreReport:
    def _make_result(self, **overrides) -> dict:
        base = {
            "family": "invoice",
            "tier": 2,
            "points_earned": 3,
            "max_points": 5,
            "correctness": 0.6,
            "tier_score": 1.2,
            "checks": [],
            "issues": [],
        }
        base.update(overrides)
        return base

    def test_returns_string(self):
        assert isinstance(format_score_report(self._make_result()), str)

    def test_contains_family(self):
        result = format_score_report(self._make_result(family="supplier"))
        assert "supplier" in result

    def test_contains_score_fraction(self):
        result = format_score_report(self._make_result(points_earned=3, max_points=5))
        assert "3/5" in result

    def test_check_pass_label_present(self):
        check = {"passed": True, "label": "has_customer", "points": 2, "detail": "ok"}
        result = format_score_report(self._make_result(checks=[check]))
        assert "PASS" in result
        assert "has_customer" in result

    def test_check_fail_label_present(self):
        check = {"passed": False, "label": "has_due_date", "points": 1, "detail": "missing"}
        result = format_score_report(self._make_result(checks=[check]))
        assert "FAIL" in result

    def test_issues_section_appears_when_present(self):
        result = format_score_report(self._make_result(issues=["Missing VAT type"]))
        assert "Missing VAT type" in result
        assert "Semantic issues" in result

    def test_no_issues_section_when_empty(self):
        result = format_score_report(self._make_result(issues=[]))
        assert "Semantic issues" not in result
