from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Test JT-VAE generative design
from autoantibiotic.generative_design import (
    JTVAE,
    generate_novel_scaffolds,
)


class TestJTVAE:
    """Tests for JTVAE class."""

    @pytest.fixture
    def jtvae(self):
        """Create a JTVAE instance."""
        return JTVAE(model_path="", device="cpu")

    def test_init_empty_model(self):
        """Test initialization without model path."""
        jtvae = JTVAE()
        assert jtvae._model is None

    def test_init_with_model_path(self):
        """Test initialization with model path."""
        with patch("autoantibiotic.generative_design._HAVE_TORCH", False):
            jtvae = JTVAE(model_path="/nonexistent/model.pt")
            assert jtvae._model is None

    @pytest.mark.skip(reason="Requires torch installation")
    def test_load_model(self):
        """Test loading a model from file."""
        with patch("autoantibiotic.generative_design._HAVE_TORCH", True):
            jtvae = JTVAE(model_path="/nonexistent/model.pt")
            assert jtvae._model is None

    def test_clear_cache(self):
        """Test cache clearing."""
        jtvae = JTVAE()
        jtvae.clear_cache()
        # No exception should be raised
        assert True

    def test_generate_novel_scaffolds_heuristic(self):
        """Test heuristic generation when torch is unavailable."""
        with patch("autoantibiotic.generative_design._HAVE_TORCH", False):
            jtvae = JTVAE()
            scaffolds = jtvae.generate_novel_scaffolds(
                core_smiles="CC1=CC=CC=C1",
                n_samples=5,
            )
            assert isinstance(scaffolds, list)
            assert len(scaffolds) <= 5

    def test_heuristic_generation_empty(self):
        """Test heuristic generation with empty input."""
        jtvae = JTVAE()
        scaffolds = jtvae.generate_novel_scaffolds(
            core_smiles="",
            n_samples=10,
        )
        assert isinstance(scaffolds, list)
        assert len(scaffolds) == 0

    def test_heuristic_generation_invalid_smiles(self):
        """Test heuristic generation with invalid SMILES."""
        jtvae = JTVAE()
        scaffolds = jtvae.generate_novel_scaffolds(
            core_smiles="invalid_smiles",
            n_samples=10,
        )
        assert isinstance(scaffolds, list)


class TestGenerateNovelScaffolds:
    """Tests for the convenience wrapper function."""

    @pytest.mark.skip(reason="Requires torch installation")
    def test_generate_novel_scaffolds_wrapper(self):
        """Test the generate_novel_scaffolds wrapper function."""
        scaffolds = generate_novel_scaffolds(
            core_smiles="CC1=CC=CC=C1",
            n_samples=5,
        )
        assert isinstance(scaffolds, list)
        assert len(scaffolds) <= 5
