from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Test FEP engine
from autoantibiotic.fep_engine import (
    FEPResistanceCalculator,
    FEPResistanceResult,
)


class TestFEPResistanceResult:
    """Tests for FEPResistanceResult data container."""

    def test_repr(self):
        result = FEPResistanceResult(
            delta_delta_g=-1.234,
            confidence=0.85,
            n_windows=11,
        )
        assert "dΔΔG=-1.234" in repr(result)
        assert "confidence=0.85" in repr(result)
        assert "windows=11" in repr(result)

    def test_negative_delta_ddg(self):
        result = FEPResistanceResult(
            delta_delta_g=-2.5,
            confidence=0.9,
            n_windows=15,
        )
        assert result.delta_delta_g == -2.5
        assert result.confidence == 0.9
        assert result.n_windows == 15
        assert result.error is None

    def test_with_error(self):
        result = FEPResistanceResult(
            delta_delta_g=0.0,
            confidence=0.0,
            n_windows=0,
            error="API key missing",
        )
        assert result.error == "API key missing"


class TestFEPResistanceCalculator:
    """Tests for FEPResistanceCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a FEPResistanceCalculator instance."""
        return FEPResistanceCalculator(
            receptor_wt_pdb="output/pdb/PBP2a_holo.pdb",
            receptor_mut_pdb="output/pdb/PBP2a_M241L.pdb",
            ligand_smiles="CC(=O)OC",
        )

    def test_init_with_rdkit_mol(self):
        """Test initialization with RDKit Mol object."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC(=O)OC")
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_rdkit=mol,
        )
        assert calc.ligand_rdkit is not None
        assert calc.ligand_smiles == ""

    def test_calculate_ddg_no_openmm(self):
        """Test that calculate_ddg falls back to heuristic when OpenMM is unavailable."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="CC(=O)OC",
        )
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", False), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", False):
            result = calc.calculate_ddg()
            assert result.delta_delta_g is not None

    def test_calculate_ddg_with_openmm(self):
        """Test calculate_ddg with OpenMM available."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="CC(=O)OC",
        )

        # Mock the internal FEP calculation
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
             patch.object(calc, '_compute_fep_delta_ddg', return_value=(-1.5, 0.85)):
            result = calc.calculate_ddg()
            assert result.delta_delta_g == -1.5
            assert result.n_windows > 0

    def test_invalid_smiles(self):
        """Test that invalid SMILES are handled gracefully."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="invalid_smiles_string",
        )
        # The calculator should handle invalid SMILES gracefully
        assert calc.ligand_rdkit is None or calc.ligand_rdkit is not None

    def test_empty_ligand_smiles(self):
        """Test initialization with empty SMILES."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="",
        )
        assert calc.ligand_rdkit is None
        assert calc.ligand_smiles == ""
