"""Tests for configuration validation and sanitization."""

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoantibiotic.config import CONFIG, ConfigurationError, PipelineConfig


class TestPipelineConfigValidation:
    """Verify that :meth:`PipelineConfig.validate_config` raises
    :class:`ConfigurationError` for invalid parameter values."""

    def _config_with(self, **overrides: object) -> PipelineConfig:
        cfg = copy.deepcopy(CONFIG)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    # ── IFP distance thresholds ──────────────────────────────────

    @pytest.mark.parametrize("field", ["ifp_hba_dist", "ifp_hbd_dist", "ifp_hyd_dist", "ifp_pi_dist"])
    def test_ifp_distance_positive(self, field: str) -> None:
        cfg = self._config_with(**{field: -1.0})
        with pytest.raises(ConfigurationError, match="must be > 0"):
            cfg.validate_config()

    def test_ifp_distance_zero(self) -> None:
        cfg = self._config_with(ifp_hba_dist=0.0)
        with pytest.raises(ConfigurationError, match="must be > 0"):
            cfg.validate_config()

    # ── Probability thresholds (must be in [0, 1]) ───────────────

    @pytest.mark.parametrize("field", [
        "ml_admet_herg_threshold", "max_dropout_rate",
        "consensus_vina_weight", "consensus_shape_weight",
        "ifp_similarity_threshold", "fep_ifp_threshold",
    ])
    def test_probability_threshold_out_of_range(self, field: str) -> None:
        cfg = self._config_with(**{field: 1.5})
        with pytest.raises(ConfigurationError, match="must be in \\[0, 1\\]"):
            cfg.validate_config()

    def test_probability_threshold_negative(self) -> None:
        cfg = self._config_with(ml_admet_herg_threshold=-0.1)
        with pytest.raises(ConfigurationError, match="must be in \\[0, 1\\]"):
            cfg.validate_config()

    # ── FEP physical parameters ──────────────────────────────────

    @pytest.mark.parametrize("field", [
        "fep_nonbonded_cutoff_nm", "fep_solvent_padding_nm",
        "fep_collision_rate_per_ps", "fep_pressure_atm",
        "fep_ewald_error_tolerance",
    ])
    def test_fep_positive_required(self, field: str) -> None:
        cfg = self._config_with(**{field: 0.0})
        with pytest.raises(ConfigurationError, match="must be > 0"):
            cfg.validate_config()

    def test_fep_ionic_strength_non_negative(self) -> None:
        cfg = self._config_with(fep_ionic_strength_molar=-0.1)
        with pytest.raises(ConfigurationError, match="must be >= 0"):
            cfg.validate_config()

    def test_fep_min_samples_mbar_too_small(self) -> None:
        cfg = self._config_with(fep_min_samples_mbar=5)
        with pytest.raises(ConfigurationError, match="must be >= 10"):
            cfg.validate_config()

    def test_fep_max_heavy_atoms_invalid(self) -> None:
        cfg = self._config_with(fep_max_heavy_atoms=0)
        with pytest.raises(ConfigurationError, match="must be >= 1"):
            cfg.validate_config()

    def test_fep_max_smiles_length_invalid(self) -> None:
        cfg = self._config_with(fep_max_smiles_length=0)
        with pytest.raises(ConfigurationError, match="must be >= 1"):
            cfg.validate_config()

    # ── Box sizes ────────────────────────────────────────────────

    @pytest.mark.parametrize("field", [
        "allosteric_box_size", "active_box_size",
        "offtarget_box_size", "redocking_box_size",
    ])
    def test_box_size_positive(self, field: str) -> None:
        cfg = self._config_with(**{field: (0.0, 15.0, 15.0)})
        with pytest.raises(ConfigurationError, match="must be > 0"):
            cfg.validate_config()

    def test_dynamic_box_padding_positive(self) -> None:
        cfg = self._config_with(dynamic_box_padding=-1.0)
        with pytest.raises(ConfigurationError, match="must be > 0"):
            cfg.validate_config()

    # ── Valid config should pass ─────────────────────────────────

    def test_default_config_passes(self) -> None:
        cfg = copy.deepcopy(CONFIG)
        cfg.dry_run = True
        cfg.validate_config()  # should not raise

    def test_dry_run_skips_validation(self) -> None:
        cfg = copy.deepcopy(CONFIG)
        cfg.dry_run = True
        cfg.ifp_hba_dist = -999.0
        cfg.validate_config()  # should not raise when dry_run=True


class TestConfigSanitization:
    """Verify that the magic numbers from scoring_metrics and fep_engine
    have been successfully migrated to PipelineConfig."""

    def test_ifp_distances_in_config(self) -> None:
        assert hasattr(CONFIG, "ifp_hba_dist")
        assert hasattr(CONFIG, "ifp_hbd_dist")
        assert hasattr(CONFIG, "ifp_hyd_dist")
        assert hasattr(CONFIG, "ifp_pi_dist")
        assert CONFIG.ifp_hba_dist == 3.5
        assert CONFIG.ifp_hbd_dist == 3.5
        assert CONFIG.ifp_hyd_dist == 4.5
        assert CONFIG.ifp_pi_dist == 5.5

    def test_fep_params_in_config(self) -> None:
        assert hasattr(CONFIG, "fep_collision_rate_per_ps")
        assert hasattr(CONFIG, "fep_nonbonded_cutoff_nm")
        assert hasattr(CONFIG, "fep_solvent_padding_nm")
        assert hasattr(CONFIG, "fep_ionic_strength_molar")
        assert hasattr(CONFIG, "fep_pressure_atm")
        assert hasattr(CONFIG, "fep_min_samples_mbar")
        assert hasattr(CONFIG, "fep_max_heavy_atoms")
        assert hasattr(CONFIG, "fep_max_smiles_length")
        assert hasattr(CONFIG, "fep_minimization_iterations")
        assert hasattr(CONFIG, "fep_ewald_error_tolerance")

    def test_admet_reference_csv_in_config(self) -> None:
        assert hasattr(CONFIG, "admet_reference_csv")
        assert CONFIG.admet_reference_csv == "data/admet_reference_curated.csv"


class TestFEPDependencyValidation:
    """Tests for FEP dependency validation with mocked imports."""

    def _config_with_fep(self) -> PipelineConfig:
        import copy
        cfg = copy.deepcopy(CONFIG)
        cfg.dry_run = False
        cfg.use_fep_resistance = True
        return cfg

    def test_fep_raises_config_error_when_openmmforcefields_missing(self) -> None:
        """When openmmforcefields is missing, validate_config raises ConfigurationError."""
        cfg = self._config_with_fep()
        with patch.dict("sys.modules", {
            "openmmforcefields": None,
            "openmmforcefields.generators": None,
        }):
            with pytest.raises(ConfigurationError, match="openmmforcefields"):
                cfg.validate_config()

    def test_fep_raises_config_error_when_gaff_generator_fails(self) -> None:
        """When GAFFTemplateGenerator instantiation fails, validate_config raises ConfigurationError."""
        cfg = self._config_with_fep()
        mock_ommf = MagicMock()
        mock_ommf.generators = MagicMock()
        mock_ommf.generators.GAFFTemplateGenerator = MagicMock(side_effect=ImportError("no gaff"))
        with patch.dict("sys.modules", {
            "openmm": MagicMock(),
            "openmmtools": MagicMock(),
            "openmmforcefields": mock_ommf,
            "openmmforcefields.generators": mock_ommf.generators,
        }):
            with pytest.raises(ConfigurationError, match="openmmforcefields"):
                cfg.validate_config()

    def test_fep_dry_run_skips_validation(self) -> None:
        """When dry_run=True, FEP dependency checks are skipped."""
        import copy
        cfg = copy.deepcopy(CONFIG)
        cfg.use_fep_resistance = True
        cfg.dry_run = True
        cfg.validate_config()  # should not raise


class TestCheckFEPCompatibility:
    """Tests for check_fep_compatibility in io_utils."""

    def test_check_fep_compatibility_logs_versions(self, caplog) -> None:
        """check_fep_compatibility logs versions of FEP packages."""
        from autoantibiotic.io_utils import check_fep_compatibility
        import logging
        caplog.set_level(logging.INFO)
        with patch.dict("sys.modules", {
            "openmm": MagicMock(__version__="7.6.0"),
            "openmmtools": MagicMock(__version__="1.0.0"),
            "openmmforcefields": MagicMock(__version__="1.0.0"),
        }):
            check_fep_compatibility()
        assert "openmm version: 7.6.0" in caplog.text
        assert "openmmtools version: 1.0.0" in caplog.text
        assert "openmmforcefields version: 1.0.0" in caplog.text

    def test_check_fep_compatibility_runs_without_error(self) -> None:
        """check_fep_compatibility runs without raising when packages are available."""
        from autoantibiotic.io_utils import check_fep_compatibility
        with patch.dict("sys.modules", {
            "openmm": MagicMock(__version__="8.1.0"),
            "openmmtools": MagicMock(__version__="1.1.0"),
            "openmmforcefields": MagicMock(__version__="0.12.0"),
        }):
            check_fep_compatibility()
