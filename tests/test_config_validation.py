"""Tests for configuration validation of profiles and dependencies."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from autoantibiotic.config import (
    CONFIG,
    ConfigurationError,
    PipelineConfig,
    PipelineProfile,
)
from autoantibiotic.io_utils import check_openmm_availability


class TestStandardProfileIsLightweight:
    """Verify that PipelineProfile.STANDARD does NOT require OpenMM."""

    def _make_config(self, profile: PipelineProfile) -> PipelineConfig:
        cfg = copy.deepcopy(CONFIG)
        cfg.apply_profile(profile)
        cfg.dry_run = False
        return cfg

    def test_standard_validate_passes_without_openmm(self) -> None:
        """STANDARD profile should validate successfully even when OpenMM
        is mocked as missing."""
        cfg = self._make_config(PipelineProfile.STANDARD)
        with patch.dict("sys.modules", {
            "openmm": None,
            "openmmtools": None,
            "openmmforcefields": None,
            "pdbfixer": None,
        }):
            cfg.validate_config()  # should not raise

    def test_standard_has_fep_disabled(self) -> None:
        """STANDARD profile should have use_fep_resistance=False."""
        cfg = self._make_config(PipelineProfile.STANDARD)
        assert cfg.use_fep_resistance is False

    def test_standard_has_explicit_solvent_disabled(self) -> None:
        """STANDARD profile should have use_explicit_solvent_mmgbsa=False."""
        cfg = self._make_config(PipelineProfile.STANDARD)
        assert cfg.use_explicit_solvent_mmgbsa is False

    def test_standard_has_gnn_rescoring_disabled(self) -> None:
        """STANDARD profile should have use_gnn_rescoring=False."""
        cfg = self._make_config(PipelineProfile.STANDARD)
        assert cfg.use_gnn_rescoring is False

    def test_standard_has_ml_admet_enabled(self) -> None:
        """STANDARD profile should have use_ml_admet=True (lightweight)."""
        cfg = self._make_config(PipelineProfile.STANDARD)
        assert cfg.use_ml_admet is True

    def test_standard_has_moderate_exhaustiveness(self) -> None:
        """STANDARD profile should have vina_exhaustiveness=8."""
        cfg = self._make_config(PipelineProfile.STANDARD)
        assert cfg.vina_exhaustiveness == 8


class TestProductionFEPRequiresOpenMM:
    """Verify that PipelineProfile.PRODUCTION_FEP raises
    ConfigurationError if OpenMM dependencies are missing."""

    def _make_config(self) -> PipelineConfig:
        cfg = copy.deepcopy(CONFIG)
        cfg.apply_profile(PipelineProfile.PRODUCTION_FEP)
        cfg.dry_run = False
        return cfg

    def test_production_fep_raises_if_openmm_missing(self) -> None:
        """PRODUCTION_FEP should raise ConfigurationError when openmm
        is mocked as missing."""
        cfg = self._make_config()
        with patch.dict("sys.modules", {
            "openmm": None,
            "openmmtools": None,
            "openmmforcefields": None,
        }):
            with pytest.raises(ConfigurationError, match="OpenMM"):
                cfg.validate_config()

    def test_production_fep_raises_if_openmmtools_missing(self) -> None:
        """PRODUCTION_FEP should raise ConfigurationError when openmmtools
        is mocked as missing (openmm is available)."""
        cfg = self._make_config()
        with patch.dict("sys.modules", {
            "openmm": MagicMock(),
            "openmmtools": None,
            "pdbfixer": MagicMock(),
        }):
            with pytest.raises(ConfigurationError, match="openmmtools"):
                cfg.validate_config()

    def test_production_fep_raises_if_openmmforcefields_missing(self) -> None:
        """PRODUCTION_FEP should raise ConfigurationError when
        openmmforcefields is mocked as missing."""
        cfg = self._make_config()
        with patch.dict("sys.modules", {
            "openmm": MagicMock(),
            "openmmtools": MagicMock(),
            "openmmforcefields": None,
            "openmmforcefields.generators": None,
            "pdbfixer": MagicMock(),
        }):
            with pytest.raises(ConfigurationError, match="openmmforcefields"):
                cfg.validate_config()

    def test_production_fep_has_fep_enabled(self) -> None:
        """PRODUCTION_FEP profile should have use_fep_resistance=True."""
        cfg = self._make_config()
        assert cfg.use_fep_resistance is True

    def test_production_fep_has_explicit_solvent_enabled(self) -> None:
        """PRODUCTION_FEP profile should have use_explicit_solvent_mmgbsa=True."""
        cfg = self._make_config()
        assert cfg.use_explicit_solvent_mmgbsa is True


class TestExplicitSolventValidation:
    """Verify that use_explicit_solvent_mmgbsa=True triggers strict
    dependency checking."""

    def _make_config(self) -> PipelineConfig:
        cfg = copy.deepcopy(CONFIG)
        cfg.use_explicit_solvent_mmgbsa = True
        cfg.dry_run = False
        return cfg

    def test_explicit_solvent_raises_if_openmm_missing(self) -> None:
        """Should raise ConfigurationError when openmm is missing."""
        cfg = self._make_config()
        with patch.dict("sys.modules", {"openmm": None}):
            with pytest.raises(ConfigurationError, match="OpenMM"):
                cfg.validate_config()

    def test_explicit_solvent_raises_if_pdbfixer_missing(self) -> None:
        """Should raise ConfigurationError when pdbfixer is missing."""
        cfg = self._make_config()
        with patch.dict("sys.modules", {
            "openmm": MagicMock(),
            "pdbfixer": None,
        }):
            with pytest.raises(ConfigurationError, match="pdbfixer"):
                cfg.validate_config()


class TestCLIFlagValidation:
    """Verify that CLI flags trigger strict validation.

    These tests simulate the CLI-override logic that ``main()`` performs
    rather than importing ``main()`` directly, because the full import
    chain (scikit-learn, numpy, etc.) can cause C-extension loading
    issues in certain test environments.
    """

    def test_cli_fep_flag_triggers_validation_error(self) -> None:
        """--use-fep-resistance should trigger ConfigurationError when
        OpenMM dependencies are missing (simulated via config)."""
        cfg = copy.deepcopy(CONFIG)
        cfg.apply_profile(PipelineProfile.STANDARD)
        cfg.use_fep_resistance = True  # --use-fep-resistance CLI flag
        cfg.dry_run = False
        with patch.dict("sys.modules", {
            "openmm": None,
            "openmmtools": None,
            "openmmforcefields": None,
        }):
            with pytest.raises(ConfigurationError, match="OpenMM"):
                cfg.validate_config()

    def test_cli_explicit_solvent_flag_triggers_validation_error(self) -> None:
        """--use-explicit-solvent should trigger ConfigurationError when
        OpenMM dependencies are missing (simulated via config)."""
        cfg = copy.deepcopy(CONFIG)
        cfg.apply_profile(PipelineProfile.STANDARD)
        cfg.use_explicit_solvent_mmgbsa = True  # --use-explicit-solvent CLI flag
        cfg.dry_run = False
        with patch.dict("sys.modules", {"openmm": None}):
            with pytest.raises(ConfigurationError, match="OpenMM"):
                cfg.validate_config()

    def test_cli_flags_override_standard_profile(self) -> None:
        """CLI flags (--use-fep-resistance) should override STANDARD
        profile defaults (simulated via config)."""
        cfg = copy.deepcopy(CONFIG)
        cfg.apply_profile(PipelineProfile.STANDARD)
        # STANDARD profile sets use_fep_resistance=False
        assert cfg.use_fep_resistance is False
        # CLI flag overrides to True
        cfg.use_fep_resistance = True
        assert cfg.use_fep_resistance is True

    def test_dry_run_skips_fep_validation(self) -> None:
        """--dry-run should skip FEP dependency checks even when
        use_fep_resistance=True."""
        cfg = copy.deepcopy(CONFIG)
        cfg.use_fep_resistance = True
        cfg.dry_run = True
        with patch.dict("sys.modules", {
            "openmm": None,
            "openmmtools": None,
            "openmmforcefields": None,
        }):
            cfg.validate_config()  # should not raise


class TestCheckOpenMMAvailability:
    """Tests for check_openmm_availability in io_utils."""

    def test_returns_true_when_all_available(self) -> None:
        """Returns (True, '') when all packages are available."""
        with patch.dict("sys.modules", {
            "openmm": MagicMock(),
            "openmmtools": MagicMock(),
            "openmmforcefields": MagicMock(),
        }):
            ok, msg = check_openmm_availability()
            assert ok is True
            assert msg == ""

    def test_returns_false_when_openmm_missing(self) -> None:
        """Returns (False, ...) when openmm is missing."""
        with patch.dict("sys.modules", {
            "openmm": None,
            "openmmtools": MagicMock(),
            "openmmforcefields": MagicMock(),
        }):
            ok, msg = check_openmm_availability()
            assert ok is False
            assert "openmm" in msg.lower()

    def test_returns_false_when_all_missing(self) -> None:
        """Returns (False, ...) when all packages are missing."""
        with patch.dict("sys.modules", {
            "openmm": None,
            "openmmtools": None,
            "openmmforcefields": None,
        }):
            ok, msg = check_openmm_availability()
            assert ok is False
            assert "openmm" in msg
            assert "openmmtools" in msg
            assert "openmmforcefields" in msg


class TestBackwardCompatibility:
    """Verify backward compatibility for users who manually set
    use_fep_resistance=True in scripts."""

    def test_manual_fep_true_still_raises_if_deps_missing(self) -> None:
        """Manually setting use_fep_resistance=True should still raise
        ConfigurationError if deps are missing."""
        cfg = copy.deepcopy(CONFIG)
        cfg.use_fep_resistance = True
        cfg.dry_run = False
        with patch.dict("sys.modules", {
            "openmm": None,
            "openmmtools": None,
            "openmmforcefields": None,
        }):
            with pytest.raises(ConfigurationError, match="OpenMM"):
                cfg.validate_config()

    def test_manual_explicit_solvent_true_still_raises_if_deps_missing(self) -> None:
        """Manually setting use_explicit_solvent_mmgbsa=True should still
        raise ConfigurationError if deps are missing."""
        cfg = copy.deepcopy(CONFIG)
        cfg.use_explicit_solvent_mmgbsa = True
        cfg.dry_run = False
        with patch.dict("sys.modules", {"openmm": None}):
            with pytest.raises(ConfigurationError, match="OpenMM"):
                cfg.validate_config()
