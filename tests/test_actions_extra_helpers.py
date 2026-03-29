"""Tests for actions.py — extended pure helper functions.

Covers: _money, _contains_ci, _invoice_matches, _resolve_default_vat_id,
        _find_vat_type_id, _requested_vat_percentages, _default_due_date,
        _overdue_dates, _is_placeholder_employee, _next_date_iso,
        _pick_travel_payment_type, _pick_travel_cost_category.

These are all pure synchronous functions that require no network access.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"))

from actions import (
    _contains_ci,
    _default_due_date,
    _find_vat_type_id,
    _invoice_matches,
    _is_placeholder_employee,
    _money,
    _next_date_iso,
    _overdue_dates,
    _pick_travel_cost_category,
    _pick_travel_payment_type,
    _requested_vat_percentages,
    _resolve_default_vat_id,
)


# ---------------------------------------------------------------------------
# _money
# ---------------------------------------------------------------------------

class TestMoney:
    def test_rounds_to_two_decimals(self):
        assert _money(10.5555) == 10.56

    def test_integer_input(self):
        assert _money(100) == 100.0

    def test_zero(self):
        assert _money(0) == 0.0

    def test_negative(self):
        assert _money(-5.999) == -6.0

    def test_returns_float(self):
        assert isinstance(_money(10), float)

    def test_already_two_decimals_unchanged(self):
        assert _money(3.14) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# _contains_ci
# ---------------------------------------------------------------------------

class TestContainsCi:
    def test_basic_match(self):
        assert _contains_ci("Hello World", "hello") is True

    def test_case_insensitive(self):
        assert _contains_ci("NASDAQ", "nasdaq") is True

    def test_no_match(self):
        assert _contains_ci("apple", "orange") is False

    def test_none_haystack(self):
        assert _contains_ci(None, "needle") is False

    def test_none_needle(self):
        assert _contains_ci("haystack", None) is False

    def test_both_none(self):
        assert _contains_ci(None, None) is False

    def test_empty_haystack(self):
        assert _contains_ci("", "x") is False

    def test_empty_needle(self):
        assert _contains_ci("haystack", "") is False

    def test_substring_match(self):
        assert _contains_ci("Ola Nordmann AS", "nordmann") is True


# ---------------------------------------------------------------------------
# _invoice_matches
# ---------------------------------------------------------------------------

class TestInvoiceMatches:
    def _inv(self, inv_num=None, customer_name=None, org_num=None):
        return {
            "invoiceNumber": inv_num,
            "customer": {
                "name": customer_name,
                "organizationNumber": org_num,
            },
        }

    def test_empty_args_always_matches(self):
        inv = self._inv(inv_num=1001, customer_name="Acme AS")
        assert _invoice_matches(inv, {}) is True

    def test_invoice_number_match(self):
        inv = self._inv(inv_num=1234)
        assert _invoice_matches(inv, {"invoiceNumber": "1234"}) is True

    def test_invoice_number_no_match(self):
        inv = self._inv(inv_num=1234)
        assert _invoice_matches(inv, {"invoiceNumber": "9999"}) is False

    def test_customer_name_match_case_insensitive(self):
        inv = self._inv(customer_name="Acme AS")
        assert _invoice_matches(inv, {"customerName": "acme"}) is True

    def test_customer_name_no_match(self):
        inv = self._inv(customer_name="Acme AS")
        assert _invoice_matches(inv, {"customerName": "Boring Corp"}) is False

    def test_org_number_match(self):
        inv = self._inv(org_num="123456789")
        assert _invoice_matches(inv, {"customerOrgNumber": "123456789"}) is True

    def test_org_number_no_match(self):
        inv = self._inv(org_num="123456789")
        assert _invoice_matches(inv, {"customerOrgNumber": "999999999"}) is False

    def test_all_args_must_pass(self):
        inv = self._inv(inv_num=42, customer_name="Foo AS", org_num="111")
        # invoiceNumber matches but org doesn't
        assert _invoice_matches(inv, {"invoiceNumber": "42", "customerOrgNumber": "999"}) is False


# ---------------------------------------------------------------------------
# _resolve_default_vat_id
# ---------------------------------------------------------------------------

class TestResolveDefaultVatId:
    def _vat_types(self, vts):
        return {"values": vts}

    def test_fallback_when_no_vat_types(self):
        assert _resolve_default_vat_id(self._vat_types([]), fallback=7) == 7

    def test_returns_25_percent_vat_id(self):
        vts = [{"id": 3, "name": "Høy sats", "percentage": 25}]
        result = _resolve_default_vat_id(self._vat_types(vts))
        assert result == 3

    def test_prefers_outgoing_vat(self):
        vts = [
            {"id": 10, "name": "Inngående MVA 25%", "percentage": 25},
            {"id": 11, "name": "Utgående MVA 25%", "percentage": 25},
        ]
        result = _resolve_default_vat_id(self._vat_types(vts))
        assert result == 11  # "utgående" scores higher


# ---------------------------------------------------------------------------
# _find_vat_type_id
# ---------------------------------------------------------------------------

class TestFindVatTypeId:
    def _vat(self, id, name, percentage):
        return {"id": id, "name": name, "percentage": percentage}

    def test_returns_none_for_empty(self):
        result = _find_vat_type_id({"values": []}, percentage=25, prefer_outgoing=False)
        assert result is None

    def test_finds_matching_percentage(self):
        vts = {"values": [self._vat(5, "Standard 25%", 25)]}
        result = _find_vat_type_id(vts, percentage=25, prefer_outgoing=False)
        assert result == 5

    def test_filters_out_wrong_percentage(self):
        vts = {"values": [self._vat(1, "Zero VAT", 0), self._vat(2, "25%", 25)]}
        result = _find_vat_type_id(vts, percentage=0, prefer_outgoing=False)
        assert result == 1

    def test_prefers_outgoing_when_flag_set(self):
        vts = {"values": [
            self._vat(1, "Inngående MVA", 25),
            self._vat(2, "Utgående MVA høy sats", 25),
        ]}
        result = _find_vat_type_id(vts, percentage=25, prefer_outgoing=True)
        assert result == 2


# ---------------------------------------------------------------------------
# _requested_vat_percentages
# ---------------------------------------------------------------------------

class TestRequestedVatPercentages:
    def test_none_returns_25(self):
        assert _requested_vat_percentages(None) == [25]

    def test_25_returns_25(self):
        assert _requested_vat_percentages(25) == [25]

    def test_0_returns_0(self):
        assert _requested_vat_percentages(0) == [0]

    def test_12_returns_12(self):
        assert _requested_vat_percentages(12) == [12]

    def test_15_returns_15(self):
        assert _requested_vat_percentages(15) == [15]

    def test_legacy_id_3_maps_to_25(self):
        assert _requested_vat_percentages(3) == [25]

    def test_legacy_id_31_maps_to_15(self):
        assert _requested_vat_percentages(31) == [15]

    def test_unknown_returns_empty(self):
        assert _requested_vat_percentages(999) == []


# ---------------------------------------------------------------------------
# _default_due_date
# ---------------------------------------------------------------------------

class TestDefaultDueDate:
    def test_adds_30_days(self):
        result = _default_due_date("2026-01-01")
        assert result == "2026-01-31"

    def test_crosses_month_boundary(self):
        result = _default_due_date("2026-03-15")
        expected = (date(2026, 3, 15) + timedelta(days=30)).isoformat()
        assert result == expected

    def test_invalid_date_returns_original(self):
        result = _default_due_date("not-a-date")
        assert result == "not-a-date"

    def test_returns_string(self):
        assert isinstance(_default_due_date("2026-01-01"), str)


# ---------------------------------------------------------------------------
# _overdue_dates
# ---------------------------------------------------------------------------

class TestOverdueDates:
    def test_returns_two_strings(self):
        invoice_date, due_date = _overdue_dates()
        assert isinstance(invoice_date, str)
        assert isinstance(due_date, str)

    def test_invoice_date_before_due_date(self):
        invoice_date, due_date = _overdue_dates()
        assert invoice_date < due_date

    def test_due_date_in_the_past(self):
        _, due_date = _overdue_dates()
        assert due_date < date.today().isoformat()

    def test_invoice_date_is_iso_format(self):
        invoice_date, _ = _overdue_dates()
        date.fromisoformat(invoice_date)  # should not raise


# ---------------------------------------------------------------------------
# _is_placeholder_employee
# ---------------------------------------------------------------------------

class TestIsPlaceholderEmployee:
    def test_none_is_placeholder(self):
        assert _is_placeholder_employee(None) is True

    def test_empty_dict_is_placeholder(self):
        assert _is_placeholder_employee({}) is True

    def test_employee_with_email_not_placeholder(self):
        emp = {"email": "ola@example.com"}
        assert _is_placeholder_employee(emp) is False

    def test_employee_with_department_not_placeholder(self):
        emp = {"department": {"id": 1}}
        assert _is_placeholder_employee(emp) is False

    def test_employee_with_employment_not_placeholder(self):
        emp = {"employments": [{"startDate": "2026-01-01"}]}
        assert _is_placeholder_employee(emp) is False

    def test_whitespace_only_email_is_placeholder(self):
        emp = {"email": "   "}
        assert _is_placeholder_employee(emp) is True


# ---------------------------------------------------------------------------
# _next_date_iso
# ---------------------------------------------------------------------------

class TestNextDateIso:
    def test_next_day(self):
        assert _next_date_iso("2026-03-15") == "2026-03-16"

    def test_crosses_month_boundary(self):
        assert _next_date_iso("2026-03-31") == "2026-04-01"

    def test_crosses_year_boundary(self):
        assert _next_date_iso("2026-12-31") == "2027-01-01"

    def test_returns_string(self):
        assert isinstance(_next_date_iso("2026-01-01"), str)


# ---------------------------------------------------------------------------
# _pick_travel_payment_type
# ---------------------------------------------------------------------------

class TestPickTravelPaymentType:
    def test_empty_list_returns_none(self):
        assert _pick_travel_payment_type([]) is None

    def test_picks_item_with_ansatt(self):
        items = [
            {"id": 1, "description": "Firma betaler"},
            {"id": 2, "description": "Utlegg av ansatt"},
        ]
        result = _pick_travel_payment_type(items)
        assert result == 2

    def test_skips_item_with_no_id(self):
        items = [{"description": "No ID"}, {"id": 5, "description": "Utlegg"}]
        result = _pick_travel_payment_type(items)
        assert result == 5

    def test_single_item_returned(self):
        items = [{"id": 42, "description": "Something"}]
        assert _pick_travel_payment_type(items) == 42


# ---------------------------------------------------------------------------
# _pick_travel_cost_category
# ---------------------------------------------------------------------------

class TestPickTravelCostCategory:
    def test_empty_returns_none(self):
        assert _pick_travel_cost_category([]) is None

    def test_picks_item_with_show_on_travel(self):
        items = [
            {"id": 1, "description": "Internal", "showOnTravelExpenses": False},
            {"id": 2, "description": "Fly", "showOnTravelExpenses": True},
        ]
        assert _pick_travel_cost_category(items) == 2

    def test_inactive_item_deprioritized(self):
        items = [
            {"id": 1, "description": "Active", "showOnTravelExpenses": True, "isInactive": False},
            {"id": 2, "description": "Inactive", "showOnTravelExpenses": True, "isInactive": True},
        ]
        assert _pick_travel_cost_category(items) == 1

    def test_skips_item_with_no_id(self):
        items = [{"description": "No ID"}, {"id": 9, "description": "With ID", "showOnTravelExpenses": True}]
        result = _pick_travel_cost_category(items)
        assert result == 9
