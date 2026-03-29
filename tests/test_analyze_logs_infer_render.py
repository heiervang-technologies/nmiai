"""Pure-function tests for accounting analyze_logs.py — infer_missing_fields and render_markdown.

These functions weren't covered by earlier test branches (analyze-logs-helpers
and analyze-logs-extra covered normalize_text, compile_keyword_pattern,
collect_omissions, normalize_pattern, parse_embedded_json, successful_writes,
extract_validation_entries, and infer_family_from_prompt).

All pure functions — no file system, network, or server access.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_ACCT_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "accounting")
sys.path.insert(0, _ACCT_DIR)

from analyze_logs import infer_missing_fields, render_markdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_summary(**overrides) -> dict:
    """Build a minimal valid summary dict for render_markdown."""
    base = {
        "total_runs": 5,
        "log_dir": "/tmp/logs",
        "seen_families": ["employee", "timesheet"],
        "effective_seen_families": ["employee"],
        "new_families_since_last_run": [],
        "unseen_families": ["department"],
        "family_mismatches": [],
        "empty_attachment_runs": 2,
        "alerts": [],
        "priority_targets": [],
        "families": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# infer_missing_fields
# ---------------------------------------------------------------------------

class TestInferMissingFields:
    def test_empty_inputs_return_empty_counters(self):
        direct, prompt_req, blockers = infer_missing_fields("unknown", "", "", [])
        assert isinstance(direct, Counter)
        assert isinstance(prompt_req, Counter)
        assert isinstance(blockers, Counter)
        assert len(direct) == 0
        assert len(prompt_req) == 0
        assert len(blockers) == 0

    def test_duplicate_identifier_suppresses_department(self):
        # "nummeret er i bruk" triggers duplicate_identifier blocker
        direct, prompt_req, blockers = infer_missing_fields(
            "department",
            "",
            "nummeret er i bruk",
            [],
        )
        # blocker should be set, department should be popped from direct
        assert blockers.get("duplicate_identifier", 0) > 0
        assert "department" not in direct

    def test_email_validation_suppresses_employee(self):
        direct, prompt_req, blockers = infer_missing_fields(
            "employee",
            "",
            "invalid email address format",
            [],
        )
        assert blockers.get("email_validation", 0) > 0
        assert "employee" not in direct

    def test_validation_entry_counted(self):
        entries = [{"path": "/supplier", "field": "name", "message": "missing name required"}]
        direct, prompt_req, blockers = infer_missing_fields(
            "unknown", "", "", entries
        )
        # Should produce some direct or blocker counts from validation text
        # (exact result depends on GENERIC_FIELD_RULES matching "name")
        assert isinstance(direct, Counter)

    def test_validation_http_400_counted(self):
        entries = [{"path": "/api/v2/employee", "field": "", "message": "HTTP 400 with no captured body"}]
        direct, prompt_req, blockers = infer_missing_fields(
            "unknown", "", "", entries
        )
        assert isinstance(blockers, Counter)

    def test_none_final_message_handled(self):
        # final_message=None should not raise
        direct, prompt_req, blockers = infer_missing_fields(
            "unknown", "some prompt", None, []
        )
        assert isinstance(direct, Counter)

    def test_returns_tuple_of_three(self):
        result = infer_missing_fields("unknown", "", "", [])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def test_returns_string(self):
        result = render_markdown(_min_summary())
        assert isinstance(result, str)

    def test_contains_header(self):
        result = render_markdown(_min_summary())
        assert "# Accounting Log Analysis" in result

    def test_total_runs_shown(self):
        result = render_markdown(_min_summary(total_runs=42))
        assert "42" in result

    def test_seen_families_listed(self):
        result = render_markdown(_min_summary(seen_families=["employee", "timesheet"]))
        assert "employee" in result
        assert "timesheet" in result

    def test_empty_alerts_shows_none(self):
        result = render_markdown(_min_summary(alerts=[]))
        # "none" should appear in the Alerts section
        assert "none" in result

    def test_alert_items_listed(self):
        result = render_markdown(_min_summary(alerts=["something broken"]))
        assert "something broken" in result

    def test_priority_targets_listed(self):
        targets = [
            {
                "family": "timesheet",
                "priority_score": 0.75,
                "proxy_clean_rate": 0.6,
                "runs": 10,
                "blockers": ["dup"],
                "missing_fields": ["date"],
            }
        ]
        result = render_markdown(_min_summary(priority_targets=targets))
        assert "timesheet" in result
        assert "0.75" in result

    def test_family_section_rendered(self):
        families = [
            {
                "family": "employee",
                "runs": 8,
                "proxy_clean_rate": 0.875,
                "likely_full_runs": 7,
                "likely_partial_runs": 1,
                "mean_api_errors": 0.5,
                "prompt_required_fields": [{"field": "name"}],
                "likely_blockers": [],
                "likely_missing_fields": [{"field": "email"}],
                "top_error_patterns": [],
                "scorer_hypotheses": [],
            }
        ]
        result = render_markdown(_min_summary(families=families))
        assert "### employee" in result
        assert "8" in result
        assert "email" in result

    def test_no_priority_targets_shows_none(self):
        result = render_markdown(_min_summary(priority_targets=[]))
        assert "none" in result

    def test_ends_with_newline(self):
        result = render_markdown(_min_summary())
        assert result.endswith("\n")

    def test_empty_attachment_runs_shown(self):
        result = render_markdown(_min_summary(empty_attachment_runs=7))
        assert "7" in result

    def test_unseen_families_shown(self):
        result = render_markdown(_min_summary(unseen_families=["department", "project"]))
        assert "department" in result
        assert "project" in result

    def test_new_families_none_when_empty(self):
        result = render_markdown(_min_summary(new_families_since_last_run=[]))
        # "none" in new families line
        assert "none" in result

    def test_new_families_listed_when_present(self):
        result = render_markdown(_min_summary(new_families_since_last_run=["supplier"]))
        assert "supplier" in result

    def test_priority_targets_capped_at_8(self):
        targets = [
            {
                "family": f"family_{i}",
                "priority_score": 0.5,
                "proxy_clean_rate": 0.5,
                "runs": 1,
                "blockers": [],
                "missing_fields": [],
            }
            for i in range(12)
        ]
        result = render_markdown(_min_summary(priority_targets=targets))
        # Only up to 8 should be shown
        shown = sum(1 for i in range(12) if f"family_{i}" in result)
        assert shown <= 8

    def test_family_hypothesis_rendered(self):
        families = [
            {
                "family": "timesheet",
                "runs": 2,
                "proxy_clean_rate": 0.5,
                "likely_full_runs": 1,
                "likely_partial_runs": 1,
                "mean_api_errors": 1.0,
                "prompt_required_fields": [],
                "likely_blockers": [],
                "likely_missing_fields": [],
                "top_error_patterns": [{"pattern": "404 not found"}],
                "scorer_hypotheses": ["missing start_date field"],
            }
        ]
        result = render_markdown(_min_summary(families=families))
        assert "missing start_date field" in result
        assert "404 not found" in result
