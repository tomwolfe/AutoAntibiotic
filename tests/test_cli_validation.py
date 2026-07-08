"""Tests for the --validate-inputs CLI flag and validate_pipeline_inputs()."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from autoantibiotic.config import PipelineConfig
from autoantibiotic.io_utils import validate_pipeline_inputs


class TestValidatePipelineInputs:
    """Tests for validate_pipeline_inputs()."""

    def test_valid_config_returns_no_errors(self) -> None:
        """A config with valid inputs should pass validation with at most warnings."""
        cfg = PipelineConfig(
            output_dir=Path(tempfile.mkdtemp()),
            reference_antibiotics={
                "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
                "Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NCC3)C(=O)O)C(=O)O)(C)O",
            },
        )
        issues = validate_pipeline_inputs(cfg)
        # Skip binary errors since they depend on the environment
        non_binary_errors = [e for e in issues["errors"] if "binary" not in e.lower() and "not found" not in e.lower()]
        assert len(non_binary_errors) == 0, f"Unexpected errors: {non_binary_errors}"

    def test_output_dir_not_writable_returns_error(self) -> None:
        """An unwritable output directory should produce an error."""
        cfg = PipelineConfig(output_dir=Path("/nonexistent_dir_xyz"))
        issues = validate_pipeline_inputs(cfg)
        errors = issues["errors"]
        error_texts = " ".join(errors).lower()
        assert any("writ" in e.lower() or "not" in e.lower() for e in errors), \
            f"Expected writability error, got: {errors}"

    def test_invalid_reference_smiles_returns_error(self) -> None:
        """Invalid SMILES in reference_antibiotics should be flagged."""
        cfg = PipelineConfig(
            output_dir=Path(tempfile.mkdtemp()),
            reference_antibiotics={"test": "not_a_valid_smiles_!!!"},
        )
        issues = validate_pipeline_inputs(cfg)
        error_texts = " ".join(issues["errors"]).lower()
        assert "invalid" in error_texts or "smiles" in error_texts

    def test_invalid_brics_smiles_returns_error(self) -> None:
        """Invalid SMILES in brics_building_blocks should be flagged."""
        cfg = PipelineConfig(
            output_dir=Path(tempfile.mkdtemp()),
            brics_building_blocks=["invalid_smiles_!!!"],
        )
        issues = validate_pipeline_inputs(cfg)
        error_texts = " ".join(issues["errors"]).lower()
        assert "invalid" in error_texts or "smiles" in error_texts

    def test_missing_ensemble_dir_returns_warning(self) -> None:
        """A non-existent ensemble_structures_dir should produce a warning, not error."""
        cfg = PipelineConfig(
            output_dir=Path(tempfile.mkdtemp()),
            ensemble_structures_dir=Path("/nonexistent_ensemble_dir"),
        )
        issues = validate_pipeline_inputs(cfg)
        warning_texts = " ".join(issues["warnings"]).lower()
        assert "does not exist" in warning_texts or "ensemble" in warning_texts

    def test_binary_check_included(self) -> None:
        """Binary availability should be checked."""
        cfg = PipelineConfig(output_dir=Path(tempfile.mkdtemp()))
        with patch("autoantibiotic.io_utils.BinaryManager.validate_all") as mock_val:
            mock_val.return_value = {
                "vina": True, "gnina": False,
                "obabel": True, "prepare_receptor": False,
            }
            issues = validate_pipeline_inputs(cfg)
        error_texts = " ".join(issues["errors"]).lower()
        assert "gnina" in error_texts or "binary" in error_texts

    def test_issues_dict_structure(self) -> None:
        """The return dict should have 'errors' and 'warnings' keys with list values."""
        cfg = PipelineConfig(output_dir=Path(tempfile.mkdtemp()))
        issues = validate_pipeline_inputs(cfg)
        assert "errors" in issues
        assert "warnings" in issues
        assert isinstance(issues["errors"], list)
        assert isinstance(issues["warnings"], list)


class TestCLIValidateInputs:
    """Test that --validate-inputs is properly wired in main()."""

    def test_validate_inputs_argument_accepted(self) -> None:
        """The --validate-inputs flag should be parseable."""
        from autoantibiotic.main import main
        with patch("autoantibiotic.main.validate_pipeline_inputs") as mock_val:
            mock_val.return_value = {"errors": [], "warnings": []}
            with patch("autoantibiotic.main.print") as mock_print:
                with pytest.raises(SystemExit) as excinfo:
                    main(["--validate-inputs"])
                assert excinfo.value.code == 0

    def test_validate_inputs_with_errors_exits_1(self) -> None:
        """If validation has errors, exit code should be 1."""
        from autoantibiotic.main import main
        with patch("autoantibiotic.main.validate_pipeline_inputs") as mock_val:
            mock_val.return_value = {
                "errors": ["Binary 'vina' not found"],
                "warnings": [],
            }
            with pytest.raises(SystemExit) as excinfo:
                main(["--validate-inputs"])
            assert excinfo.value.code == 1
