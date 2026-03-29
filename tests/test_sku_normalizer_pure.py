"""Pure-function tests for object-detection/data-creation/analyze_sku_pl.py.

Covers: normalize_name, extract_brand

Both are pure string functions — no file system, network, or external data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_OD_DATA_DIR = str(
    Path(__file__).resolve().parent.parent
    / "tasks" / "object-detection" / "data-creation"
)
sys.path.insert(0, _OD_DATA_DIR)

from analyze_sku_pl import normalize_name, extract_brand


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------

class TestNormalizeName:
    def test_lowercases(self):
        assert normalize_name("COCA-COLA") == "coca-cola"

    def test_strips_whitespace(self):
        assert normalize_name("  melk  ") == "melk"

    def test_removes_grams(self):
        result = normalize_name("Sjokolade 200g")
        assert "200g" not in result
        assert "g" not in result.split()

    def test_removes_kg(self):
        result = normalize_name("Poteter 2kg")
        assert "2kg" not in result

    def test_removes_ml(self):
        result = normalize_name("Brus 500ml")
        assert "500ml" not in result

    def test_removes_liter(self):
        result = normalize_name("Juice 1l")
        assert "1l" not in result or "1" not in result

    def test_collapses_whitespace(self):
        result = normalize_name("Melk   hvit   fullmelk")
        assert "  " not in result

    def test_returns_string(self):
        result = normalize_name("Juice 1L")
        assert isinstance(result, str)

    def test_empty_string(self):
        result = normalize_name("")
        assert result == ""

    def test_strips_and_lowers(self):
        result = normalize_name("  Egg  ")
        assert result == "egg"

    def test_unit_at_start(self):
        result = normalize_name("500g havregryn")
        assert "500g" not in result

    def test_stk_removed(self):
        result = normalize_name("Kapsler 12stk")
        assert "12" not in result or "stk" not in result


# ---------------------------------------------------------------------------
# extract_brand
# ---------------------------------------------------------------------------

class TestExtractBrand:
    def test_coca_cola_found(self):
        result = extract_brand("Coca-Cola Original 330ml")
        assert result == "coca-cola"

    def test_nestle_variant_found(self):
        result = extract_brand("Nestlé KitKat 4er")
        assert result in ("nestlé", "nestle", "kitkat")

    def test_unknown_brand_returns_none(self):
        result = extract_brand("Ukjent produkt")
        assert result is None

    def test_case_insensitive(self):
        result = extract_brand("OREO Chocolate Cookies")
        assert result == "oreo"

    def test_haribo_found(self):
        result = extract_brand("Haribo Goldbears 100g")
        assert result == "haribo"

    def test_empty_string_returns_none(self):
        result = extract_brand("")
        assert result is None

    def test_returns_string_or_none(self):
        result = extract_brand("Arla Melk")
        assert result is None or isinstance(result, str)

    def test_kitkat_found(self):
        result = extract_brand("KitKat Chunky 40g")
        assert result == "kitkat"
