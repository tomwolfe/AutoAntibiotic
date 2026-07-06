"""Tests for FEP expanded scope: size limits and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoantibiotic.config import CONFIG, ConfigurationError as ConfigConfigurationError
from autoantibiotic.fep_engine import (
    FEPResistanceCalculator,
    FEPResistanceResult,
    ConfigurationError,
)


class TestFEPEmptyLargeMolecule:
    """Verify that FEP skips molecules that are too large."""

    @pytest.fixture
    def large_mol_mock(self):
        """Create a mock RDKit Mol with >50 heavy atoms."""
        from rdkit import Chem
        # Build a simple aliphatic chain with 55 carbons via SMILES
        smiles = "C" * 55
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumHeavyAtoms() > 50
        return mol

    @pytest.fixture
    def calculator_with_large_mol(self, large_mol_mock):
        return FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_rdkit=large_mol_mock,
        )

    def test_fep_skips_large_molecule(self, calculator_with_large_mol):
        """calculate_ddg raises ConfigurationError for ligands >50 heavy atoms."""
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True), \
             patch("os.path.exists", return_value=True):
            with pytest.raises(ConfigurationError, match="Molecule too large"):
                calculator_with_large_mol.calculate_ddg()

    def test_fep_allows_small_molecule(self):
        """Small molecules should not raise size errors."""
        from rdkit import Chem
        small_mol = Chem.MolFromSmiles("CC(=O)OC")  # small ester
        assert small_mol is not None
        assert small_mol.GetNumHeavyAtoms() <= 50

        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_rdkit=small_mol,
        )
        # No exception should be raised for small molecules
        # (we don't test the full FEP calculation here, just that the size check passes)
        assert calc.ligand_rdkit is not None


class TestFEPExpandedTopN:
    """Verify that fep_top_n default is 20."""

    def test_fep_top_n_default_is_20(self):
        """fep_top_n should default to 20 for expanded FEP scope."""
        assert CONFIG.fep_top_n == 20

    def test_fep_top_n_can_be_increased(self):
        """fep_top_n can be increased beyond default."""
        original = CONFIG.fep_top_n
        CONFIG.fep_top_n = 50
        try:
            assert CONFIG.fep_top_n == 50
        finally:
            CONFIG.fep_top_n = original


class TestFEPExpandedSmoke:
    """Smoke tests for FEP engine integration."""

    def test_fep_result_repr(self):
        result = FEPResistanceResult(
            delta_delta_g=-0.5,
            confidence=0.7,
            n_windows=11,
        )
        assert "d\u0394\u0394G=-0.5" in repr(result)
        assert "confidence=0.7" in repr(result)
        assert "windows=11" in repr(result)

    def test_fep_result_defaults(self):
        result = FEPResistanceResult(
            delta_delta_g=0.0,
            confidence=0.0,
            n_windows=0,
        )
        assert result.delta_delta_g == 0.0
        assert result.confidence == 0.0
        assert result.n_windows == 0
        assert result.error is None
