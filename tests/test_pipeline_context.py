"""Tests for pipeline state dict pattern used by PhaseHandlers."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestPipelineState:
    """Verify state dict state passing between phases."""

    def test_default_initialization(self):
        state = {
            "library": [],
            "filtered_library": [],
            "docked_candidates": [],
            "md_results": [],
            "fep_results": [],
            "n_total": 0,
            "n_filtered": 0,
        }
        assert state["library"] == []
        assert state["filtered_library"] == []
        assert state["docked_candidates"] == []
        assert state["md_results"] == []
        assert state["fep_results"] == []
        assert state["n_total"] == 0
        assert state["n_filtered"] == 0

    def test_custom_values(self):
        mock_audit = MagicMock()
        state = {
            "library": ["lib1", "lib2"],
            "filtered_library": ["filt1"],
            "docked_candidates": ["docked1"],
            "md_results": ["md1"],
            "fep_results": ["fep1"],
            "audit": mock_audit,
            "n_total": 10,
            "n_filtered": 5,
        }
        assert state["library"] == ["lib1", "lib2"]
        assert state["filtered_library"] == ["filt1"]
        assert state["docked_candidates"] == ["docked1"]
        assert state["md_results"] == ["md1"]
        assert state["fep_results"] == ["fep1"]
        assert state["audit"] is mock_audit
        assert state["n_total"] == 10
        assert state["n_filtered"] == 5

    def test_mutability_of_lists(self):
        state = {
            "library": [],
            "filtered_library": [],
            "docked_candidates": [],
        }
        state["library"].append("new")
        assert state["library"] == ["new"]
        state["filtered_library"].append("filt")
        assert state["filtered_library"] == ["filt"]

    def test_state_passing_through_phases(self):
        state = {
            "library": [],
            "filtered_library": [],
            "docked_candidates": [],
        }
        state = self._phase_one(state)
        state = self._phase_two(state)
        assert state["library"] == ["comp1", "comp2"]
        assert state["filtered_library"] == ["comp1"]
        assert state["docked_candidates"] == ["comp1"]

    def _phase_one(self, state: dict) -> dict:
        state["library"] = ["comp1", "comp2"]
        return state

    def _phase_two(self, state: dict) -> dict:
        state["filtered_library"] = [c for c in state["library"] if c != "comp2"]
        state["docked_candidates"] = state["filtered_library"]
        return state

    def test_fep_results_independent_from_docked(self):
        state = {
            "docked_candidates": ["a", "b", "c"],
            "fep_results": ["a"],
        }
        assert len(state["docked_candidates"]) == 3
        assert len(state["fep_results"]) == 1
