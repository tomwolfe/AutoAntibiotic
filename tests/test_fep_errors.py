from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoantibiotic.fep_engine import (
    ConfigurationError,
    FEPConvergenceError,
    FEPResistanceCalculator,
    FETopologyError,
    FEResourceError,
)
from autoantibiotic.config import CONFIG


class TestFEPCustomExceptions:
    """Verify the new FEP exception types."""

    def test_fep_convergence_error_is_exception(self):
        assert issubclass(FEPConvergenceError, Exception)

    def test_fep_topology_error_is_exception(self):
        assert issubclass(FETopologyError, Exception)

    def test_fep_resource_error_is_exception(self):
        assert issubclass(FEResourceError, Exception)

    def test_fep_convergence_error_message(self):
        exc = FEPConvergenceError("MBAR did not converge")
        assert "MBAR did not converge" in str(exc)

    def test_fep_topology_error_message(self):
        exc = FETopologyError("Missing LIG residue")
        assert "Missing LIG residue" in str(exc)

    def test_fep_resource_error_message(self):
        exc = FEResourceError("CUDA out of memory")
        assert "CUDA out of memory" in str(exc)


class TestFEPErrorMapping:
    """Verify that calculate_ddg maps OpenMM errors to custom exceptions."""

    @pytest.fixture
    def calculator(self):
        return FEPResistanceCalculator(
            receptor_wt_pdb="/fake/wt.pdb",
            receptor_mut_pdb="/fake/mut.pdb",
            ligand_smiles="CC(=O)OC",
        )

    def test_topology_error_raised(self, calculator):
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True), \
             patch.object(calculator, "_pre_screen_initial_energy", return_value=None), \
             patch("autoantibiotic.fep_engine.os.path.exists", return_value=True), \
             patch.object(calculator, "_compute_fep_delta_ddg",
                          side_effect=Exception("GAFF atom type not found for ligand")):
            with pytest.raises(FETopologyError) as excinfo:
                calculator.calculate_ddg()
            assert "GAFF" in str(excinfo.value)

    def test_convergence_error_raised(self, calculator):
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True), \
             patch.object(calculator, "_pre_screen_initial_energy", return_value=None), \
             patch("autoantibiotic.fep_engine.os.path.exists", return_value=True), \
             patch.object(calculator, "_compute_fep_delta_ddg",
                          side_effect=Exception("MBAR estimate failed due to poor overlap")):
            with pytest.raises(FEPConvergenceError) as excinfo:
                calculator.calculate_ddg()
            assert "MBAR" in str(excinfo.value) or "poor overlap" in str(excinfo.value)

    def test_resource_error_raised(self, calculator):
        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
             patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True), \
             patch.object(calculator, "_pre_screen_initial_energy", return_value=None), \
             patch("autoantibiotic.fep_engine.os.path.exists", return_value=True), \
             patch.object(calculator, "_compute_fep_delta_ddg",
                          side_effect=Exception("CUDA out of memory. Try reducing system size.")):
            with pytest.raises(FEResourceError) as excinfo:
                calculator.calculate_ddg()
            assert "CUDA" in str(excinfo.value) or "memory" in str(excinfo.value)


class TestFEPRetryWithIncreasedWindows:
    """Verify retry_with_increased_windows."""

    def test_retry_increases_lambda_windows(self):
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="/fake/wt.pdb",
            receptor_mut_pdb="/fake/mut.pdb",
            ligand_smiles="CC(=O)OC",
        )
        original_windows = CONFIG.fep_lambda_windows

        dummy_result = MagicMock()
        dummy_result.delta_delta_g = -1.5
        dummy_result.confidence = 0.85

        with patch.object(calc, "calculate_ddg", return_value=dummy_result) as mock_calc:
            result = calc.retry_with_increased_windows(extra_windows=4)

            assert mock_calc.call_count == 1
            assert result.delta_delta_g == -1.5
            assert result.confidence == 0.85
            assert CONFIG.fep_lambda_windows == original_windows

    def test_retry_restores_original_windows_on_error(self):
        calc = FEPResistanceCalculator(
            receptor_wt_pdb="/fake/wt.pdb",
            receptor_mut_pdb="/fake/mut.pdb",
            ligand_smiles="CC(=O)OC",
        )
        original_windows = CONFIG.fep_lambda_windows

        with patch.object(calc, "calculate_ddg",
                          side_effect=FEPConvergenceError("still failing")):
            with pytest.raises(FEPConvergenceError):
                calc.retry_with_increased_windows(extra_windows=4)

        assert CONFIG.fep_lambda_windows == original_windows, (
            "Lambda windows should be restored after error"
        )
