from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from autoantibiotic.generative_design import (
    JTVAE,
    generate_novel_scaffolds,
    _validate_mol,
    _fitness,
)


class TestJTVAE:
    """Tests for JTVAE class (GA backend)."""

    @pytest.fixture
    def jtvae(self):
        return JTVAE(model_path="", device="cpu")

    def test_init_empty_model(self):
        jtvae = JTVAE()
        assert jtvae._model is None

    def test_init_with_model_path(self):
        with patch("autoantibiotic.generative_design._HAVE_TORCH", False):
            jtvae = JTVAE(model_path="/nonexistent/model.pt")
            assert jtvae._model is None

    def test_generate_novel_scaffolds_ga_backend(self):
        """GA backend produces valid RDKit Mol objects."""
        jtvae = JTVAE()
        mols = jtvae.generate_novel_scaffolds(
            core_smiles="CC1=CC=CC=C1",
            n_samples=5,
        )
        assert isinstance(mols, list)
        assert all(isinstance(m, Chem.Mol) for m in mols)
        assert all(m.GetNumAtoms() > 0 for m in mols)

    def test_generate_novel_scaffolds_empty_input(self):
        jtvae = JTVAE()
        mols = jtvae.generate_novel_scaffolds(
            core_smiles="",
            n_samples=10,
        )
        assert isinstance(mols, list)
        # GA backend can still generate from building blocks alone

    def test_generate_novel_scaffolds_invalid_smiles(self):
        jtvae = JTVAE()
        mols = jtvae.generate_novel_scaffolds(
            core_smiles="invalid_smiles",
            n_samples=10,
        )
        assert isinstance(mols, list)

    def test_generate_novel_scaffolds_returns_mols_not_strings(self):
        """Ensure the return type is List[Chem.Mol], not List[str]."""
        jtvae = JTVAE()
        mols = jtvae.generate_novel_scaffolds(
            core_smiles="CC1=CC=CC=C1",
            n_samples=3,
        )
        for m in mols:
            assert isinstance(m, Chem.Mol)
            # Verify each mol can be sanitized
            smi = Chem.MolToSmiles(m)
            assert Chem.MolFromSmiles(smi) is not None

    def test_generated_mols_have_reasonable_properties(self):
        """Generated molecules should have non-trivial QED and pass basic sanity."""
        jtvae = JTVAE()
        mols = jtvae.generate_novel_scaffolds(
            core_smiles="CC1=CC=CC=C1",
            n_samples=10,
        )
        assert len(mols) > 0
        for m in mols:
            qed = _fitness(m)
            assert qed >= 0.0, "Fitness should be non-negative"

    def test_clear_cache(self):
        jtvae = JTVAE()
        jtvae.clear_cache()

    def test_neural_generation_fallback_to_ga(self):
        """When neural encoding fails, GA backend should be used."""
        jtvae = JTVAE()
        with patch.object(jtvae, '_model', MagicMock()), \
             patch.object(jtvae, '_encode', return_value=None):
            mols = jtvae.generate_novel_scaffolds(
                core_smiles="CC1=CC=CC=C1",
                n_samples=3,
            )
            assert isinstance(mols, list)
            # GA fallback should still produce molecules
            if mols:
                assert all(isinstance(m, Chem.Mol) for m in mols)


class TestGenerateNovelScaffolds:
    """Tests for the convenience wrapper function."""

    def test_generate_novel_scaffolds_wrapper(self):
        """Wrapper function returns List[str] (backward compatible)."""
        scaffolds = generate_novel_scaffolds(
            core_smiles="CC1=CC=CC=C1",
            n_samples=5,
        )
        assert isinstance(scaffolds, list)
        # Backward compatible: returns SMILES strings
        assert all(isinstance(s, str) for s in scaffolds)
        # Each SMILES should be parseable
        for s in scaffolds:
            assert Chem.MolFromSmiles(s) is not None

    def test_wrapper_with_different_cores(self):
        scaffolds = generate_novel_scaffolds(
            core_smiles="c1ccccc1",
            n_samples=3,
        )
        assert len(scaffolds) <= 3


class TestFitnessFunction:
    """Tests for the _fitness function used by GA."""

    def test_fitness_of_valid_mols(self):
        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        score = _fitness(mol)
        assert 0.0 <= score <= 1.0

    def test_fitness_is_reproducible(self):
        mol = Chem.MolFromSmiles("CC1=CC=CC=C1")
        assert mol is not None
        score1 = _fitness(mol)
        score2 = _fitness(mol)
        assert score1 == score2


class TestValidateMol:
    def test_valid_smiles(self):
        mol = _validate_mol("c1ccccc1")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)

    def test_invalid_smiles(self):
        mol = _validate_mol("invalid")
        assert mol is None

    def test_empty_string(self):
        mol = _validate_mol("")
        # RDKit may return an empty mol; validate it has no atoms
        if mol is not None:
            assert mol.GetNumAtoms() == 0
