from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Test Synthesis Planner
from autoantibiotic.synthesis_planner import (
    SynthesisPlanner,
    SynthesisResult,
)


class TestSynthesisResult:
    """Tests for SynthesisResult data container."""

    def test_default_values(self):
        """Test default values for SynthesisResult."""
        result = SynthesisResult(
            synthesizable=True,
            confidence=0.85,
        )
        assert result.synthesizable is True
        assert result.confidence == 0.85
        assert result.routes == []
        assert result.error is None

    def test_with_routes(self):
        """Test SynthesisResult with routes."""
        routes = [
            {"reactants": ["A", "B"], "products": ["C"], "confidence": 0.9},
        ]
        result = SynthesisResult(
            synthesizable=True,
            confidence=0.9,
            routes=routes,
        )
        assert len(result.routes) == 1
        assert result.routes[0]["reactants"] == ["A", "B"]

    def test_with_error(self):
        """Test SynthesisResult with error message."""
        result = SynthesisResult(
            synthesizable=False,
            confidence=0.0,
            error="API key missing",
        )
        assert result.error == "API key missing"

    def test_repr(self):
        """Test __repr__ output."""
        result = SynthesisResult(
            synthesizable=True,
            confidence=0.85,
            routes=[{"confidence": 0.9}],
        )
        repr_str = repr(result)
        assert "synthesizable=True" in repr_str
        assert "confidence=0.85" in repr_str
        assert "routes=1" in repr_str


class TestSynthesisPlanner:
    """Tests for SynthesisPlanner class."""

    @pytest.fixture
    def planner(self):
        """Create a SynthesisPlanner instance."""
        with patch("autoantibiotic.synthesis_planner._HAVE_REQUESTS", True):
            planner = SynthesisPlanner(api_key="test_api_key")
        return planner

    @pytest.fixture
    def planner_no_key(self):
        """Create a SynthesisPlanner without API key."""
        with patch("autoantibiotic.synthesis_planner._HAVE_REQUESTS", True):
            planner = SynthesisPlanner()
        return planner

    def test_check_synthesizability_invalid_smiles(self):
        """Test that invalid SMILES returns synthesizable=False."""
        planner = SynthesisPlanner()
        result = planner.check_synthesizability("invalid_smiles_string")
        assert result.synthesizable is False
        assert result.error is not None

    def test_check_synthesizability_no_api_key(self):
        """Test that missing API key returns synthesizable=False."""
        planner = SynthesisPlanner()
        result = planner.check_synthesizability("CC(=O)OC")
        assert result.synthesizable is False

    @patch("autoantibiotic.synthesis_planner.requests")
    def test_check_synthesizability_api_success(self, mock_requests):
        """Test successful API query."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "rxn_123",
            "reactants": ["CC(=O)O", "CCN"],
            "products": ["CC(=O)OC"],
            "confidence": 0.85,
        }
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response

        with patch("autoantibiotic.synthesis_planner._HAVE_REQUESTS", True):
            planner = SynthesisPlanner(api_key="test_key")
            result = planner.check_synthesizability("CC(=O)OC")

        assert result.synthesizable is True
        assert result.confidence == 0.85
        assert len(result.routes) == 1

    def test_clear_cache(self):
        """Test cache clearing."""
        planner = SynthesisPlanner(api_key="test_key")
        planner._cache["test"] = "value"
        planner.clear_cache()
        assert len(planner._cache) == 0

    def test_heuristic_sa_score_valid(self):
        """Test heuristic SA score computation."""
        planner = SynthesisPlanner()
        result = planner._heuristic_sa_score("CC(=O)OC")
        assert result is not None
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_heuristic_sa_score_invalid(self):
        """Test heuristic SA score with invalid input."""
        planner = SynthesisPlanner()
        result = planner._heuristic_sa_score("invalid")
        assert result is not None
        assert result.synthesizable is False

    def test_parse_api_response_list(self):
        """Test parsing list-format API response."""
        planner = SynthesisPlanner()
        data = [
            {"reactants": ["A"], "products": ["B"], "confidence": 0.8},
            {"reactants": ["C"], "products": ["D"], "confidence": 0.6},
        ]
        routes = planner._parse_api_response(data, max_routes=3)
        assert len(routes) == 2
        assert routes[0]["reactants"] == ["A"]
        assert routes[1]["products"] == ["D"]

    def test_parse_api_response_single(self):
        """Test parsing single-item API response."""
        planner = SynthesisPlanner()
        data = {
            "id": "rxn_123",
            "reactants": ["A", "B"],
            "products": ["C"],
            "confidence": 0.9,
        }
        routes = planner._parse_api_response(data, max_routes=3)
        assert len(routes) == 1
        assert routes[0]["reactants"] == ["A", "B"]

    def test_parse_api_response_max_routes(self):
        """Test that max_routes limits the number of routes returned."""
        planner = SynthesisPlanner()
        data = [
            {"confidence": 0.1},
            {"confidence": 0.2},
            {"confidence": 0.3},
            {"confidence": 0.4},
            {"confidence": 0.5},
        ]
        routes = planner._parse_api_response(data, max_routes=2)
        assert len(routes) == 2
