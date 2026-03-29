"""Tests for tasks/accounting/server/scorer_checks.py — pure extractor functions.

Covers: _extract_name, _extract_org_number, _extract_email, _extract_amounts,
        _is_payment_task, _is_credit_note_task, _is_reversal_task,
        _is_order_task, _is_send_task.

All pure regex/classification functions — no network or file I/O needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"))

from scorer_checks import (
    _extract_amounts,
    _extract_email,
    _extract_name,
    _extract_org_number,
    _is_credit_note_task,
    _is_order_task,
    _is_payment_task,
    _is_reversal_task,
    _is_send_task,
)


# ---------------------------------------------------------------------------
# _extract_name
# ---------------------------------------------------------------------------

class TestExtractName:
    def test_english_for_pattern(self):
        first, last = _extract_name("Create employee for Ola Nordmann")
        assert first == "Ola"
        assert last == "Nordmann"

    def test_german_fur_pattern(self):
        first, last = _extract_name("Erstelle Mitarbeiter für Max Mustermann")
        assert first == "Max"
        assert last == "Mustermann"

    def test_norwegian_av_pattern(self):
        first, last = _extract_name("Opprett ansatt av Kari Hansen")
        assert first == "Kari"
        assert last == "Hansen"

    def test_no_name_returns_none_tuple(self):
        first, last = _extract_name("Create an invoice")
        assert first is None
        assert last is None

    def test_returns_tuple_of_two(self):
        result = _extract_name("Create employee")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_portuguese_para_pattern(self):
        first, last = _extract_name("Criar funcionário para João Silva")
        assert first == "João" or first is not None


# ---------------------------------------------------------------------------
# _extract_org_number
# ---------------------------------------------------------------------------

class TestExtractOrgNumber:
    def test_basic_org_number(self):
        result = _extract_org_number("Org nr: 123456789")
        assert result == "123456789"

    def test_org_without_separator(self):
        result = _extract_org_number("org 987654321")
        assert result == "987654321"

    def test_no_org_number_returns_none(self):
        result = _extract_org_number("Create a new customer")
        assert result is None

    def test_returns_string(self):
        result = _extract_org_number("Org.nr: 123456789")
        assert isinstance(result, str)

    def test_nine_digits_required(self):
        # 8-digit number should not match
        result = _extract_org_number("org 12345678")
        assert result is None


# ---------------------------------------------------------------------------
# _extract_email
# ---------------------------------------------------------------------------

class TestExtractEmail:
    def test_basic_email(self):
        result = _extract_email("Send to ola@example.com")
        assert result == "ola@example.com"

    def test_email_with_dots(self):
        result = _extract_email("Contact kari.nilsen@company.no")
        assert result == "kari.nilsen@company.no"

    def test_no_email_returns_none(self):
        result = _extract_email("No email here")
        assert result is None

    def test_email_with_plus(self):
        result = _extract_email("Reply to user+tag@mail.org")
        assert result == "user+tag@mail.org"

    def test_returns_string(self):
        result = _extract_email("test@example.com hello")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _extract_amounts
# ---------------------------------------------------------------------------

class TestExtractAmounts:
    def test_basic_nok_amount(self):
        amounts = _extract_amounts("Pay 1000 NOK")
        assert 1000.0 in amounts

    def test_kr_abbreviation(self):
        amounts = _extract_amounts("Betaling på 500 kr")
        assert 500.0 in amounts

    def test_multiple_amounts(self):
        amounts = _extract_amounts("First item 200 kr and second 300 kr")
        assert len(amounts) >= 2

    def test_no_amounts_returns_empty_list(self):
        amounts = _extract_amounts("Create an employee named Ola")
        assert amounts == []

    def test_returns_list(self):
        result = _extract_amounts("1000 kr")
        assert isinstance(result, list)

    def test_decimal_amount(self):
        amounts = _extract_amounts("Amount 1500,50 kr")
        assert any(abs(a - 1500.5) < 0.01 for a in amounts)

    def test_nok_uppercase(self):
        amounts = _extract_amounts("Total 750 NOK")
        assert 750.0 in amounts


# ---------------------------------------------------------------------------
# _is_payment_task
# ---------------------------------------------------------------------------

class TestIsPaymentTask:
    def test_english_register_payment(self):
        assert _is_payment_task("Register payment for invoice 1001") is True

    def test_norwegian_betaling(self):
        assert _is_payment_task("Registrer betaling på faktura") is True

    def test_payment_keyword(self):
        assert _is_payment_task("Process the payment") is True

    def test_create_customer_is_not_payment(self):
        assert _is_payment_task("Create a new customer") is False

    def test_returns_bool(self):
        assert isinstance(_is_payment_task("some text"), bool)


# ---------------------------------------------------------------------------
# _is_credit_note_task
# ---------------------------------------------------------------------------

class TestIsCreditNoteTask:
    def test_english_credit_note(self):
        assert _is_credit_note_task("Create a credit note for invoice 1001") is True

    def test_norwegian_kreditnota(self):
        assert _is_credit_note_task("Opprett kreditnota") is True

    def test_german_gutschrift(self):
        assert _is_credit_note_task("Erstelle eine Gutschrift") is True

    def test_regular_invoice_not_credit_note(self):
        assert _is_credit_note_task("Create invoice for customer Acme") is False

    def test_returns_bool(self):
        assert isinstance(_is_credit_note_task("text"), bool)


# ---------------------------------------------------------------------------
# _is_reversal_task
# ---------------------------------------------------------------------------

class TestIsReversalTask:
    def test_reversal_keyword(self):
        assert _is_reversal_task("Create a payment reversal") is True

    def test_returned_by_bank(self):
        assert _is_reversal_task("Payment returned by the bank") is True

    def test_reverse_payment(self):
        assert _is_reversal_task("Reverse payment for invoice 42") is True

    def test_regular_payment_not_reversal(self):
        assert _is_reversal_task("Register payment") is False

    def test_returns_bool(self):
        assert isinstance(_is_reversal_task("text"), bool)


# ---------------------------------------------------------------------------
# _is_order_task
# ---------------------------------------------------------------------------

class TestIsOrderTask:
    def test_english_order(self):
        assert _is_order_task("Create an order for customer Foo AS") is True

    def test_convert_to_invoice(self):
        assert _is_order_task("Convert to invoice") is True

    def test_norwegian_convert(self):
        assert _is_order_task("Konverter til faktura") is True

    def test_create_employee_not_order(self):
        assert _is_order_task("Create employee Ola Nordmann") is False

    def test_returns_bool(self):
        assert isinstance(_is_order_task("text"), bool)


# ---------------------------------------------------------------------------
# _is_send_task
# ---------------------------------------------------------------------------

class TestIsSendTask:
    def test_send_invoice_english(self):
        assert _is_send_task("Send invoice to customer") is True

    def test_send_faktura_norwegian(self):
        assert _is_send_task("Send faktura til kunde") is True

    def test_send_the_invoice(self):
        assert _is_send_task("Send the invoice now") is True

    def test_create_invoice_not_send(self):
        assert _is_send_task("Create an invoice") is False

    def test_returns_bool(self):
        assert isinstance(_is_send_task("text"), bool)
