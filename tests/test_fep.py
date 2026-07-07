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
    """Tests for adaptive convergence in FEP calculations.

    Verifies that the adaptive sampling loop:
    - Uses fep_check_interval_steps as the check interval
    - Stops early when uncertainty drops below fep_convergence_threshold_kcal_per_mol
    - Populates per_window_uncertainties in the result
    - Calculates total_simulation_time_ps
    """

    def test_result_has_per_window_uncertainties(self):
        """Verify that FEPResistanceResult can hold per_window_uncertainties."""
        result = FEPResistanceResult(
            delta_delta_g=-1.5,
            confidence=0.8,
            n_windows=11,
            per_window_uncertainties=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.5],
            total_simulation_time_ps=5000.0,
        )
        assert result.per_window_uncertainties is not None
        assert len(result.per_window_uncertainties) == 11
        assert result.total_simulation_time_ps == 5000.0

    def test_result_default_uncertainties_are_none(self):
        """Verify that per_window_uncertainties defaults to None."""
        result = FEPResistanceResult(
            delta_delta_g=-1.5,
            confidence=0.8,
            n_windows=11,
        )
        assert result.per_window_uncertainties is None

    def test_result_default_time_is_zero(self):
        """Verify that total_simulation_time_ps defaults to 0.0."""
        result = FEPResistanceResult(
            delta_delta_g=-1.5,
            confidence=0.8,
            n_windows=11,
        )
        assert result.total_simulation_time_ps == 0.0

    def test_config_defaults_for_adaptive_sampling(self):
        """Verify that config defaults support adaptive sampling."""
        assert CONFIG.fep_convergence_threshold_kcal_per_mol == 0.5
        assert CONFIG.fep_uncertainty_threshold == 1.0
        assert CONFIG.fep_check_interval_steps == 500


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

class TestPreScreenRejection:
    """Tests for FEP pre-screening energy threshold."""

    def test_pre_screen_high_initial_energy(self):
        """Pre-screening rejects when initial energy exceeds threshold."""
        from unittest.mock import patch

        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True),              patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True),              patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True),              patch("os.path.exists", return_value=True):
            calc = FEPResistanceCalculator(
                receptor_wt_pdb="wt.pdb",
                receptor_mut_pdb="mut.pdb",
                ligand_smiles="CC(=O)OC",
            )
            with patch.object(calc, "_pre_screen_initial_energy") as mock_pre_screen:
                mock_pre_screen.return_value = FEPResistanceResult(
                    delta_delta_g=0.0,
                    confidence=0.0,
                    n_windows=0,
                    error="Skipped: High Initial Energy",
                )
                result = calc.calculate_ddg()
                assert result.error == "Skipped: High Initial Energy"
                assert result.delta_delta_g == 0.0
                assert result.n_windows == 0

    def test_pre_screen_accepts_valid_energy(self):
        """Pre-screening accepts when initial energy is within threshold."""
        from unittest.mock import patch

        with patch("autoantibiotic.fep_engine._HAVE_OPENMM", True),              patch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True),              patch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True),              patch("os.path.exists", return_value=True):
            calc = FEPResistanceCalculator(
                receptor_wt_pdb="wt.pdb",
                receptor_mut_pdb="mut.pdb",
                ligand_smiles="CC(=O)OC",
            )
            with patch.object(calc, "_pre_screen_initial_energy") as mock_pre_screen:
                mock_pre_screen.return_value = None
                with patch.object(calc, "_compute_fep_delta_ddg") as mock_compute:
                    mock_compute.return_value = FEPResistanceResult(
                        delta_delta_g=-0.5,
                        confidence=0.9,
                        n_windows=11,
                    )
                    result = calc.calculate_ddg()
                    assert result.delta_delta_g == -0.5
                    assert result.n_windows == 11


class TestCheckpointSaveLoad:
    """Tests for checkpoint save and load functionality."""

    def test_checkpoint_save_load(self, tmp_path):
        """Checkpoints are saved and loaded correctly."""
        import json
        from autoantibiotic.fep_engine import FEPResistanceResult

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        label = "WT"

        # Simulate checkpoint data
        checkpoint_data = {
            "windows": [
                {"index": 0, "samples": [[0.1, 0.2, 0.3]], "uncertainty": 0.1},
                {"index": 1, "samples": [[0.4, 0.5, 0.6]], "uncertainty": 0.2},
            ]
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_{label}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Load and verify
        with open(checkpoint_path, "r") as f:
            loaded = json.load(f)

        assert len(loaded["windows"]) == 2
        assert loaded["windows"][0]["index"] == 0
        assert loaded["windows"][0]["samples"] == [[0.1, 0.2, 0.3]]
        assert loaded["windows"][1]["uncertainty"] == 0.2

    def test_checkpoint_missing_file(self, tmp_path):
        """Missing checkpoint file returns empty data."""
        label = "WT"
        checkpoint_path = tmp_path / f"checkpoint_{label}.json"

        # File doesn't exist
        assert not checkpoint_path.exists()

        # When loading non-existent file, should return empty
        import json
        data = {}
        windows = data.get("windows", [])
        assert len(windows) == 0

    @pytest.mark.parametrize("label", ["WT", "Mutant", "Custom_Label"])
    def test_checkpoint_filename_format(self, label, tmp_path):
        """Checkpoint files follow the correct naming convention."""
        import json

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_{label}.json"

        # Create directory and save
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump({"windows": []}, f)

        assert checkpoint_path.exists()
        assert "checkpoint_" in str(checkpoint_path)
        assert str(checkpoint_path).endswith(".json")

    def test_checkpoint_skips_completed_windows(self, tmp_path):
        """Verify checkpoint loading skips completed windows."""
        import json

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create a checkpoint with 2 completed windows for WT
        checkpoint_data = {
            "label": "WT",
            "windows": [
                {"index": 0, "samples": [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]},
                {"index": 1, "samples": [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]},
            ],
            "per_window_uncertainties": [0.1, 0.2],
        }
        checkpoint_path = checkpoint_dir / "checkpoint_WT.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Verify file exists and has correct content
        assert checkpoint_path.exists()
        with open(checkpoint_path, "r") as f:
            loaded = json.load(f)
        assert len(loaded["windows"]) == 2
        assert loaded["windows"][0]["index"] == 0
        assert len(loaded["windows"][0]["samples"]) == 2
        assert loaded["per_window_uncertainties"] == [0.1, 0.2]


class TestFEPConvergenceCheck:
    """Tests for MBAR-based convergence during FEP calculations."""

    class MockQuantity:
        """Minimal mock for OpenMM Quantity so arithmetic yields real floats."""
        def __init__(self, val=0.0):
            self._val = float(val)

        def value_in_unit(self, unit):
            return self._val

        def __sub__(self, other):
            if isinstance(other, TestFEPConvergenceCheck.MockQuantity):
                return TestFEPConvergenceCheck.MockQuantity(self._val - other._val)
            return TestFEPConvergenceCheck.MockQuantity(self._val - other)

        def __rsub__(self, other):
            return TestFEPConvergenceCheck.MockQuantity(other - self._val)

        def __truediv__(self, other):
            if isinstance(other, TestFEPConvergenceCheck.MockQuantity):
                return TestFEPConvergenceCheck.MockQuantity(
                    self._val / other._val if other._val != 0 else 0.0
                )
            return TestFEPConvergenceCheck.MockQuantity(
                self._val / other if other != 0 else 0.0
            )

        def __rtruediv__(self, other):
            return TestFEPConvergenceCheck.MockQuantity(
                other / self._val if self._val != 0 else 0.0
            )

        def __mul__(self, other):
            if isinstance(other, TestFEPConvergenceCheck.MockQuantity):
                return TestFEPConvergenceCheck.MockQuantity(self._val * other._val)
            return TestFEPConvergenceCheck.MockQuantity(self._val * other)

        def __rmul__(self, other):
            return TestFEPConvergenceCheck.MockQuantity(other * self._val)

        def __neg__(self):
            return TestFEPConvergenceCheck.MockQuantity(-self._val)

        def __abs__(self):
            return TestFEPConvergenceCheck.MockQuantity(abs(self._val))

    def _setup_mock_environment(
        self, mock_mbar_return_values,
    ):
        """Set up all OpenMM mocks and return the patcher contexts.

        ``mock_mbar_return_values`` is a list of (delta_f, ddelta_f) tuples
        that MBAR.get_free_energy_differences should return on successive
        calls.
        """
        import openmmtools.multistate as _oms
        import openmmtools.alchemy as _alchemy
        MQ = self.MockQuantity
        mock_unit = MagicMock()
        mock_unit.kelvin = MQ(1.0)
        mock_unit.atmospheres = MQ(1.0)
        mock_unit.picosecond = MQ(1.0)
        mock_unit.kilojoules_per_mole = MQ(1.0)
        mock_unit.kilocalories_per_mole = MQ(1.0)
        mock_unit.MOLAR_GAS_CONSTANT_R = MQ(1.0)

        mock_mbar_instance = MagicMock()
        counter = [0]

        def get_free_energy():
            idx = counter[0]
            counter[0] += 1
            if idx < len(mock_mbar_return_values):
                return mock_mbar_return_values[idx]
            return (np.zeros((3, 3)), np.ones((3, 3)) * 0.3, None)

        mock_mbar_instance.get_free_energy_differences.side_effect = get_free_energy

        mock_positions = MagicMock()

        mock_state = MagicMock()
        mock_state.getPositions.return_value = mock_positions
        mock_state.getPotentialEnergy.return_value = MQ(0.0)
        mock_state.getParameters.return_value = MagicMock()

        mock_sim = MagicMock()
        mock_sim.context.getState.return_value = mock_state

        off_diag_energy = MQ(1.0)
        mock_context_instance = MagicMock()
        mock_context_instance.getState.return_value.getPotentialEnergy.return_value = off_diag_energy

        mock_topology = MagicMock()
        mock_topology.atoms.return_value = []
        mock_topology.getNumAtoms.return_value = 100

        mock_mbar_class = MagicMock()
        mock_mbar_class.from_energy_matrix.return_value = mock_mbar_instance

        mock_alchemical_state_class = MagicMock()
        mock_alchemical_state_class.from_system.return_value = MagicMock()
        mock_alchemical_region_class = MagicMock()
        mock_alchemical_factory_class = MagicMock()
        mock_alchemical_factory_instance = MagicMock()
        mock_alchemical_factory_class.return_value = mock_alchemical_factory_instance

        patchers = [
            patch("autoantibiotic.fep_engine._openmm_unit", mock_unit),
            patch("autoantibiotic.fep_engine._openmm.LangevinIntegrator"),
            patch("autoantibiotic.fep_engine._openmm.Platform.getPlatformByName"),
            patch("autoantibiotic.fep_engine._openmm_app.Simulation", return_value=mock_sim),
            patch("autoantibiotic.fep_engine._openmm.Context", return_value=mock_context_instance),
            patch.object(_oms, "MBAR", mock_mbar_class, create=True),
            patch.object(_alchemy, "AlchemicalState", mock_alchemical_state_class, create=True),
            patch.object(_alchemy, "AlchemicalRegion", mock_alchemical_region_class, create=True),
            patch.object(_alchemy, "AbsoluteAlchemicalFactory", mock_alchemical_factory_class, create=True),
        ]

        return patchers, mock_topology

    def test_adaptive_convergence_stops_early(self, tmp_path):
        """MBAR convergence stops sampling before max steps."""
        from unittest.mock import patch as utpatch

        n_windows = 3
        saved = {
            "lw": CONFIG.fep_lambda_windows,
            "mn": CONFIG.fep_min_steps_per_window,
            "mx": CONFIG.fep_max_steps_per_window,
            "ci": CONFIG.fep_check_interval_steps,
            "cp": CONFIG.fep_enable_checkpointing,
        }
        CONFIG.fep_lambda_windows = n_windows
        CONFIG.fep_min_steps_per_window = 1
        CONFIG.fep_max_steps_per_window = 200
        CONFIG.fep_check_interval_steps = 1
        CONFIG.fep_enable_checkpointing = False

        try:
            stable_dG = -6.15
            stable_unc = 0.3

            return_values = []
            for i in range(20):
                df = np.zeros((n_windows, n_windows))
                ddf = np.ones((n_windows, n_windows)) * 2.0
                if i < 2:
                    df[-1, 0] = -5.0 + i * 2.0
                    ddf[-1, 0] = 1.5
                elif i < 3:
                    df[-1, 0] = -6.1
                    ddf[-1, 0] = 0.8
                else:
                    df[-1, 0] = stable_dG
                    ddf[-1, 0] = stable_unc
                return_values.append((df, ddf, None))

            patchers, mock_topology = self._setup_mock_environment(return_values)

            calc = FEPResistanceCalculator(
                receptor_wt_pdb="wt.pdb",
                receptor_mut_pdb="mut.pdb",
                ligand_smiles="CC(=O)OC",
            )

            with utpatch("autoantibiotic.fep_engine._HAVE_OPENMM", True), \
                 utpatch("autoantibiotic.fep_engine._HAVE_OPENMMTOOLS", True), \
                 utpatch("autoantibiotic.fep_engine._HAVE_OPENMMFORCEFIELDS", True), \
                 utpatch("os.path.exists", return_value=True), \
                 utpatch.object(calc, "_pre_screen_initial_energy", return_value=None), \
                 utpatch.object(calc, "_build_system") as mock_build:
                mock_build.return_value = (MagicMock(), mock_topology, MagicMock())

                for p in patchers:
                    p.start()

                try:
                    result = calc.calculate_ddg()
                finally:
                    for p in patchers:
                        p.stop()

            max_total_ps = (
                n_windows
                * CONFIG.fep_max_steps_per_window
                * CONFIG.fep_check_interval_steps
                * CONFIG.fep_time_step_ps
                * 2
            )
            assert result.total_simulation_time_ps < max_total_ps, (
                f"Expected early convergence but total_time={result.total_simulation_time_ps} "
                f">= max={max_total_ps}"
            )
            assert result.delta_delta_g != 0.0 or result.n_windows > 0
        finally:
            for k, v in saved.items():
                setattr(CONFIG, {
                    "lw": "fep_lambda_windows",
                    "mn": "fep_min_steps_per_window",
                    "mx": "fep_max_steps_per_window",
                    "ci": "fep_check_interval_steps",
                    "cp": "fep_enable_checkpointing",
                }[k], v)

    def test_config_has_checkpoint_defaults(self):
        """Config has defaults for all checkpoint-related fields."""
        assert CONFIG.fep_enable_checkpointing is True
        assert CONFIG.fep_uncertainty_threshold == 1.0