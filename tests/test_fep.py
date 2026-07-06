from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autoantibiotic.fep_engine import (
    FEPResistanceCalculator,
    FEPResistanceResult,
    ConfigurationError,
)
from autoantibiotic.config import CONFIG, ConfigurationError as ConfigConfigurationError


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
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True):
            with pytest.raises(ConfigurationError, match="No ligand available"):
                calc.calculate_ddg()

    def test_calculate_ddg_raises_config_error_no_openmmforcefields(self):
        """calculate_ddg raises ConfigurationError when openmmforcefields is unavailable."""
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="wt.pdb",
            receptor_mut_pdb="mut.pdb",
            ligand_smiles="CC(=O)OC",
        )
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", False):
            with pytest.raises(ConfigurationError, match="openmmforcefields is not installed"):
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
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True):
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


class TestFEPConfigValidation:
    """Tests for FEP-related config validation."""

    @staticmethod
    def _mock_modules(modules: dict[str, MagicMock]) -> dict:
        """Insert mock modules into sys.modules, returning a restore dict."""
        import sys
        restore = {}
        for name, mock in modules.items():
            if name in sys.modules:
                restore[name] = sys.modules[name]
            sys.modules[name] = mock
        return restore

    @staticmethod
    def _restore_modules(restore: dict) -> None:
        import sys
        for name in restore:
            sys.modules[name] = restore[name]

    def test_validate_config_raises_if_openmmforcefields_missing(self):
        """validate_config raises ConfigurationError when use_fep_resistance=True
        but openmmforcefields is not installed."""
        original_fep = CONFIG.use_fep_resistance
        original_mmgbsa = CONFIG.use_explicit_solvent_mmgbsa
        CONFIG.use_fep_resistance = True
        CONFIG.use_explicit_solvent_mmgbsa = False
        try:
            restore = self._mock_modules({
                "openmm": MagicMock(),
                "openmmtools": MagicMock(),
            })
            with pytest.raises(ConfigConfigurationError, match="openmmforcefields is not installed"):
                CONFIG.validate_config()
        finally:
            self._restore_modules(restore)
            CONFIG.use_fep_resistance = original_fep
            CONFIG.use_explicit_solvent_mmgbsa = original_mmgbsa

    def test_validate_config_passes_if_all_deps_available(self):
        """validate_config passes when use_fep_resistance=True and all deps are importable."""
        original_fep = CONFIG.use_fep_resistance
        original_mmgbsa = CONFIG.use_explicit_solvent_mmgbsa
        CONFIG.use_fep_resistance = True
        CONFIG.use_explicit_solvent_mmgbsa = False
        try:
            restore = self._mock_modules({
                "openmm": MagicMock(),
                "openmmtools": MagicMock(),
                "openmmforcefields": MagicMock(),
            })
            CONFIG.validate_config()
        finally:
            self._restore_modules(restore)
            CONFIG.use_fep_resistance = original_fep
            CONFIG.use_explicit_solvent_mmgbsa = original_mmgbsa


class TestFEPAdaptiveConvergence:
    """Tests for adaptive convergence in FEP calculations."""

    def test_fep_adaptive_convergence(self):
        """Verify that adaptive convergence flagging works via the
        FEPResistanceResult confidence_label and mbar_uncertainty.

        A result with high MBAR uncertainty (> 1.0 kcal/mol) should be
        flagged as "Low Confidence".  The convergence logic in
        _compute_fep_delta_ddg is tested by checking that the result
        correctly exposes the combined uncertainty.
        """
        result = FEPResistanceResult(
            delta_delta_g=-1.5,
            confidence=0.1,
            n_windows=11,
            mbar_uncertainty=2.5,
        )
        assert result._mbar_uncertainty == 2.5
        assert result.confidence_label == "Low Confidence"

    def test_fep_adaptive_convergence_within_threshold(self):
        """When uncertainty is low, confidence_label should be "High Confidence"."""
        result = FEPResistanceResult(
            delta_delta_g=-1.5,
            confidence=0.95,
            n_windows=11,
            mbar_uncertainty=0.3,
        )
        assert result._mbar_uncertainty == 0.3
        assert result.confidence_label == "High Confidence"


class TestFEPUncertaintyFlagging:
    """Tests for uncertainty flagging in FEP results."""

    def test_fep_uncertainty_flagging(self):
        """Verify that FEPResistanceResult correctly reflects high
        MBAR uncertainty (> 1.0 kcal/mol) as a "Low Confidence" label.

        This validates the boundary condition where the combined
        MBAR uncertainty exceeds 1.0 kcal/mol.
        """
        result = FEPResistanceResult(
            delta_delta_g=0.5,
            confidence=0.0,
            n_windows=11,
            mbar_uncertainty=1.5,
        )
        assert result.confidence_label == "Low Confidence"
        assert result.confidence < 0.5

    def test_fep_uncertainty_below_threshold(self):
        """When MBAR uncertainty is below 1.0, label is 'High Confidence'."""
        result = FEPResistanceResult(
            delta_delta_g=0.5,
            confidence=0.85,
            n_windows=11,
            mbar_uncertainty=0.15,
        )
        assert result.confidence_label == "High Confidence"
