"""Tests for tasks/accounting/server/planner.py — pure helper functions."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

# Set dummy API key so module-level OpenAI() call doesn't fail
os.environ.setdefault("OPENAI_API_KEY", "test-key")

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"),
)

from planner import (
    _compile_keyword_pattern,
    _dedupe_keywords,
    _normalize_text,
    FAMILY_PRIORITY,
    PLAYBOOKS,
)


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercase(self):
        assert _normalize_text("FAKTURA") == "faktura"

    def test_strips_combining_accents(self):
        # é = e + combining acute (U+0301) after NFKD decomposition
        result = _normalize_text("résumé")
        assert result == "resume"

    def test_nordic_chars_decomposed(self):
        # ø is not decomposable to ASCII, but combining chars stripped
        result = _normalize_text("Åse")
        # 'Å' decomposes to 'A' + combining ring, which gets stripped
        assert result == "ase"

    def test_already_normalized(self):
        assert _normalize_text("faktura") == "faktura"

    def test_empty_string(self):
        assert _normalize_text("") == ""

    def test_numbers_preserved(self):
        result = _normalize_text("Invoice123")
        assert "123" in result

    def test_whitespace_preserved(self):
        result = _normalize_text("create invoice")
        assert " " in result


# ---------------------------------------------------------------------------
# _dedupe_keywords
# ---------------------------------------------------------------------------

class TestDedupeKeywords:
    def test_empty_list(self):
        assert _dedupe_keywords([]) == []

    def test_no_duplicates(self):
        kws = ["faktura", "ordre", "kunde"]
        result = _dedupe_keywords(kws)
        assert len(result) == 3

    def test_exact_duplicates_removed(self):
        kws = ["faktura", "faktura"]
        result = _dedupe_keywords(kws)
        assert len(result) == 1

    def test_case_insensitive_dedup(self):
        kws = ["Faktura", "faktura", "FAKTURA"]
        result = _dedupe_keywords(kws)
        assert len(result) == 1

    def test_first_occurrence_kept(self):
        kws = ["Faktura", "faktura"]
        result = _dedupe_keywords(kws)
        assert result[0] == "Faktura"

    def test_distinct_after_normalization(self):
        kws = ["résumé", "resume"]
        result = _dedupe_keywords(kws)
        # Both normalize to "resume"
        assert len(result) == 1

    def test_order_preserved_for_unique(self):
        kws = ["abc", "xyz", "def"]
        result = _dedupe_keywords(kws)
        assert result == ["abc", "xyz", "def"]


# ---------------------------------------------------------------------------
# _compile_keyword_pattern
# ---------------------------------------------------------------------------

class TestCompileKeywordPattern:
    def test_returns_pattern(self):
        pat = _compile_keyword_pattern("faktura")
        assert isinstance(pat, re.Pattern)

    def test_matches_exact_word(self):
        pat = _compile_keyword_pattern("faktura")
        assert pat.search("send faktura til kunde")

    def test_matches_inflected_form(self):
        # Pattern allows suffix chars after the keyword
        pat = _compile_keyword_pattern("faktura")
        assert pat.search("fakturaen er sendt")

    def test_word_boundary_on_left(self):
        pat = _compile_keyword_pattern("faktura")
        # Should NOT match when 'faktura' is a suffix of another word
        assert not pat.search("xfaktura")

    def test_case_insensitive(self):
        pat = _compile_keyword_pattern("Faktura")
        assert pat.search("FAKTURA")

    def test_empty_string_returns_no_match_pattern(self):
        pat = _compile_keyword_pattern("")
        # Should not match anything
        assert not pat.search("faktura")

    def test_multi_word_keyword(self):
        pat = _compile_keyword_pattern("create invoice")
        assert pat.search("create invoice for customer")


# ---------------------------------------------------------------------------
# Registry sanity checks
# ---------------------------------------------------------------------------

class TestRegistries:
    def test_family_priority_non_empty(self):
        assert len(FAMILY_PRIORITY) > 0

    def test_invoice_has_high_priority(self):
        assert FAMILY_PRIORITY.get("invoice", 0) > FAMILY_PRIORITY.get("customer", 0)

    def test_playbooks_loaded(self):
        # At least one playbook should be loaded from the playbooks/ directory
        assert len(PLAYBOOKS) > 0

    def test_each_playbook_has_family_key(self):
        for family, pb in PLAYBOOKS.items():
            assert "family" in pb, f"Playbook {family} missing 'family' key"
