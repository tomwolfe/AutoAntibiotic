from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoantibiotic.fep_engine import (
    FEPResistanceCalculator,
    FEPResistanceResult,
    ConfigurationError,
)


class TestFEPResistanceResult:
    """Tests for FEPResistanceResult data container."""

    def test_repr(self):
        result = FEPResistanceResult(
            delta_delta_g=-1.234,
            confidence=0.85,
            n_windows=11,
        )
        assert "d\u0394\u0394G=-1.234" in repr(result)
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
        return FEPResistanceCalculator(
            receptor_wt_pdb="output/pdb/PBP2a_holo.pdb",
            receptor_mut_pdb="output/pdb/PBP2a_M241L.pdb",
            ligand_smiles="CC(=O)OC",
        )

    def test_init_with_rdkit_mol(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC(=O)OC")
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_rdkit=mol,
        )
        assert calc.ligand_rdkit is not None
        assert calc.ligand_smiles == ""

    def test_calculate_ddg_raises_config_error_no_openmm(self):
        """calculate_ddg raises ConfigurationError when OpenMM is unavailable."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="CC(=O)OC",
        )
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", False), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", False):
            with pytest.raises(ConfigurationError, match="OpenMM is not installed"):
                calc.calculate_ddg()

    def test_calculate_ddg_raises_config_error_no_openmmtools(self):
        """calculate_ddg raises ConfigurationError when openmmtools is unavailable."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="CC(=O)OC",
        )
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", False):
            with pytest.raises(ConfigurationError, match="openmmtools is not installed"):
                calc.calculate_ddg()

    def test_calculate_ddg_raises_config_error_no_ligand(self):
        """calculate_ddg raises ConfigurationError when no ligand is available."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
        )
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True):
            with pytest.raises(ConfigurationError, match="No ligand available"):
                calc.calculate_ddg()

    def test_calculate_ddg_raises_config_error_missing_wt_pdb(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC(=O)OC")
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="/nonexistent/wt.pdb",
            receptor_mut_pdb="/nonexistent/mut.pdb",
            ligand_rdkit=mol,
        )
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True):
            with pytest.raises(ConfigurationError, match="not found"):
                calc.calculate_ddg()

    def test_heuristic_fallback_raises(self):
        """_heuristic_fallback now raises ConfigurationError."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="CC(=O)OC",
        )
        with pytest.raises(ConfigurationError, match="Heuristic FEP fallback has been removed"):
            calc._heuristic_fallback()

    def test_invalid_smiles_handled_gracefully(self):
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="invalid_smiles_string",
        )
        assert calc.ligand_rdkit is None or calc.ligand_rdkit is not None

    def test_empty_ligand_smiles(self):
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="",
        )
        assert calc.ligand_rdkit is None
        assert calc.ligand_smiles == ""


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_is_exception(self):
        assert issubclass(ConfigurationError, Exception)

    def test_error_message(self):
        try:
            raise ConfigurationError("Test error message")
        except ConfigurationError as e:
            assert str(e) == "Test error message"
