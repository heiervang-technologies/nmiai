"""Tests for solver.py and snail_runner.py pure helpers.

Covers:
  - solver.full_coverage_viewports
  - solver.cell_code_to_class
  - snail_runner.build_agent_prompt
All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting" / "server"))

from solver import cell_code_to_class, full_coverage_viewports
from snail_runner import build_agent_prompt


# ---------------------------------------------------------------------------
# solver.full_coverage_viewports
# ---------------------------------------------------------------------------

class TestFullCoverageViewports:
    def test_returns_list(self):
        result = full_coverage_viewports()
        assert isinstance(result, list)

    def test_returns_9_viewports(self):
        result = full_coverage_viewports()
        assert len(result) == 9

    def test_each_viewport_is_4_tuple(self):
        for vp in full_coverage_viewports():
            assert len(vp) == 4

    def test_viewport_size_is_15x15(self):
        for vx, vy, vw, vh in full_coverage_viewports():
            assert vw == 15
            assert vh == 15

    def test_covers_x_range(self):
        # x positions should include 0 and some larger value
        xs = {vp[0] for vp in full_coverage_viewports()}
        assert 0 in xs
        assert max(xs) > 0

    def test_covers_y_range(self):
        ys = {vp[1] for vp in full_coverage_viewports()}
        assert 0 in ys
        assert max(ys) > 0


# ---------------------------------------------------------------------------
# solver.cell_code_to_class
# ---------------------------------------------------------------------------

class TestSolverCellCodeToClass:
    def test_ocean_maps_to_0(self):
        assert cell_code_to_class(10) == 0

    def test_plains_maps_to_0(self):
        assert cell_code_to_class(11) == 0

    def test_empty_maps_to_0(self):
        assert cell_code_to_class(0) == 0

    def test_settlement_maps_to_1(self):
        assert cell_code_to_class(1) == 1

    def test_port_maps_to_2(self):
        assert cell_code_to_class(2) == 2

    def test_ruin_maps_to_3(self):
        assert cell_code_to_class(3) == 3

    def test_forest_maps_to_4(self):
        assert cell_code_to_class(4) == 4

    def test_mountain_maps_to_5(self):
        assert cell_code_to_class(5) == 5

    def test_unknown_maps_to_0(self):
        assert cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# snail_runner.build_agent_prompt
# ---------------------------------------------------------------------------

class TestBuildAgentPrompt:
    def test_returns_string(self):
        result = build_agent_prompt("Create invoice", "http://api", "token123")
        assert isinstance(result, str)

    def test_includes_task_prompt(self):
        result = build_agent_prompt("Pay the vendor", "http://api", "tok")
        assert "Pay the vendor" in result

    def test_includes_base_url(self):
        result = build_agent_prompt("task", "https://sandbox.tripletex.no", "tok")
        assert "https://sandbox.tripletex.no" in result

    def test_includes_session_token(self):
        result = build_agent_prompt("task", "http://api", "my-secret-token")
        assert "my-secret-token" in result

    def test_no_files_omits_attached_section(self):
        result = build_agent_prompt("task", "http://api", "tok")
        assert "ATTACHED FILES" not in result

    def test_with_files_includes_attached_section(self):
        files = [{"filename": "invoice.pdf", "mime_type": "application/pdf"}]
        result = build_agent_prompt("task", "http://api", "tok", files=files)
        assert "ATTACHED FILES" in result
        assert "invoice.pdf" in result

    def test_with_multiple_files(self):
        files = [
            {"filename": "a.pdf", "mime_type": "application/pdf"},
            {"filename": "b.csv", "mime_type": "text/csv"},
        ]
        result = build_agent_prompt("task", "http://api", "tok", files=files)
        assert "a.pdf" in result
        assert "b.csv" in result

    def test_contains_api_cheat_sheet(self):
        result = build_agent_prompt("task", "http://api", "tok")
        assert "API CHEAT SHEET" in result

    def test_contains_workflow_section(self):
        result = build_agent_prompt("task", "http://api", "tok")
        assert "WORKFLOW" in result
