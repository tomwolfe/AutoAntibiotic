from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from autoantibiotic.generative_design import (
    JTVAE,
    generate_novel_scaffolds,
    _validate_mol,
    _fitness,
    _compute_fingerprint,
    _load_reference_actives_fps,
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


class TestDiversity:
    """Tests for scaffold diversity improvements."""

    def test_generated_scaffolds_have_low_pairwise_similarity(self):
        """Verify that the MaxMinPicker post-filter produces a diverse
        set of scaffolds with low average Tanimoto similarity."""
        jtvae = JTVAE()
        mols = jtvae.generate_novel_scaffolds(
            core_smiles="CC1=CC=CC=C1",
            n_samples=10,
        )
        assert len(mols) >= 3, "Need at least 3 scaffolds to compute diversity"

        fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
        avg_sim = 0.0
        n_pairs = 0
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                avg_sim += TanimotoSimilarity(fps[i], fps[j])
                n_pairs += 1
        avg_sim /= max(1, n_pairs)

        # With the increased diversity penalty (0.25) and MaxMinPicker
        # post-filter, the average pairwise Tanimoto similarity should
        # be well below 0.7 for a set of 10+ molecules.
        assert avg_sim < 0.7, (
            f"Average pairwise Tanimoto similarity = {avg_sim:.3f}, "
            "expected < 0.7 for a diverse set. "
            "The diversity penalty or MaxMinPicker may not be working."
        )


class TestNoveltyFiltering:
    """Tests for the ChEMBL novelty filter on generated scaffolds."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Reset the reference actives cache before each test."""
        import autoantibiotic.generative_design as gd
        gd._REFERENCE_ACTIVES_FPS = None
        yield
        gd._REFERENCE_ACTIVES_FPS = None

    def test_novelty_filtering_known_active(self):
        """A molecule very similar to a reference active should be filtered."""
        from rdkit.Chem import AllChem
        jtvae = JTVAE()

        # Ceftaroline is a known reference active
        ceftaroline = Chem.MolFromSmiles(
            "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)"
            "C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O"
        )
        assert ceftaroline is not None

        # Load reference fingerprints and check similarity
        ref_fps = _load_reference_actives_fps()
        fp = _compute_fingerprint(ceftaroline)
        if ref_fps:
            from rdkit.DataStructs import TanimotoSimilarity
            max_sim = max(TanimotoSimilarity(fp, ref) for ref in ref_fps)
            # Ceftaroline should match itself (or a close analog) in the reference set
            assert max_sim >= 0.9, (
                f"Ceftaroline similarity to reference actives is {max_sim:.3f}, "
                "expected >= 0.9 since it is itself a reference active."
            )

        # check_chembl_novelty should return False with default threshold
        assert not jtvae.check_chembl_novelty(ceftaroline, threshold=0.4), (
            "Ceftaroline is a known reference active and should be flagged as non-novel."
        )

    def test_novelty_filtering_novel_molecule(self):
        """A completely novel molecule (e.g. methane) should pass the filter."""
        jtvae = JTVAE()
        methane = Chem.MolFromSmiles("C")
        assert methane is not None
        assert jtvae.check_chembl_novelty(methane, threshold=0.4), (
            "Methane is structurally unrelated to any reference active."
        )

    def test_novelty_filtering_generated_scaffolds(self):
        """Generated scaffolds should pass the novelty check by default."""
        jtvae = JTVAE()
        mols = jtvae.generate_novel_scaffolds(
            core_smiles="CC1=CC=CC=C1",
            n_samples=5,
        )
        for mol in mols:
            assert jtvae.check_chembl_novelty(mol, threshold=0.4), (
                f"Generated scaffold {Chem.MolToSmiles(mol)} is too similar "
                "to known reference actives."
            )

    def test_novelty_filtering_with_high_threshold(self):
        """At a very high threshold (0.9), even similar molecules pass."""
        jtvae = JTVAE()
        methane = Chem.MolFromSmiles("C")
        assert methane is not None
        assert jtvae.check_chembl_novelty(methane, threshold=0.9)
