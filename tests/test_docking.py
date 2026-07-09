"""Unit tests for GNINA integration and ensemble docking."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.config import CONFIG
from autoantibiotic.models import CompoundRecord
from autoantibiotic.docking import (
    _run_docking_tool,
    dock_compound,
    dock_compound_ensemble,
    _parallel_dock_ensemble,
    prepare_ligand_pdbqt,
)
from autoantibiotic.io_utils import (
    DockingParseError,
    DockingResultValidator,
    ToolExecutor,
    parse_gnina_energy,
    parse_vina_energy,
    ToolResult,
    VinaError,
    OpenBabelError,
    _classify_tool_error,
)
from autoantibiotic.config import ConfigurationError


# ── GNINA output parsing ───────────────────────────────────────────

class TestParseGninaEnergy:
    """``parse_gnina_energy`` extracts CNNscore from GNINA output."""

    def test_cnnscore_parsed(self) -> None:
        stdout = (
            "CNNscore    :   0.8567\n"
            "CNNaffinity :   7.2345\n"
        )
        score = parse_gnina_energy(stdout)
        assert score == pytest.approx(0.8567)

    def test_cnnaffinity_fallback(self) -> None:
        stdout = "CNNaffinity :   7.2345\n"
        score = parse_gnina_energy(stdout)
        assert score == pytest.approx(7.2345)

    def test_no_score_returns_none(self) -> None:
        assert parse_gnina_energy("No GNINA output") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_gnina_energy("") is None

    def test_multiple_modes_returns_first_cnnscore(self) -> None:
        stdout = (
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
            "CNNscore    :   0.9123\n"
            "CNNaffinity :   8.4567\n"
            "   2       -7.500       1.234      2.345\n"
            "CNNscore    :   0.8500\n"
            "CNNaffinity :   7.8000\n"
        )
        score = parse_gnina_energy(stdout)
        assert score == pytest.approx(0.9123)

    def test_vina_stdout_not_parsed_as_gnina(self) -> None:
        """Vina's table should not produce a false positive GNINA score."""
        stdout = (
            "mode |   affinity | dist from best mode\n"
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
        )
        score = parse_gnina_energy(stdout)
        assert score is None


# ── GNINA subprocess mocking ───────────────────────────────────────

_GNINA_SUCCESS_STDOUT = (
    "Reading input ... done.\n"
    "Setting up the scoring function ... done.\n"
    "CNNscore    :   0.9123\n"
    "CNNaffinity :   8.4567\n"
    "done.\n"
)

_GNINA_SUCCESS_STDERR = ""

_GNINA_FAILURE_STDERR = "Error: Could not open receptor file."


def _make_tool_result(
    returncode: int = 0,
    stdout: str = _GNINA_SUCCESS_STDOUT,
    stderr: str = _GNINA_SUCCESS_STDERR,
) -> ToolResult:
    return ToolResult(returncode=returncode, stdout=stdout, stderr=stderr)


class TestRunGninaDocking:
    """``_run_docking_tool("gnina", …)`` invokes GNINA and returns the CNNscore."""

    @pytest.fixture(autouse=True)
    def reset_gnina_config(self) -> None:
        saved = (
            CONFIG.gnina_binary_path, CONFIG.dry_run,
            CONFIG.validate_docking_binaries_on_startup,
        )
        CONFIG.dry_run = False
        CONFIG.validate_docking_binaries_on_startup = False
        yield
        CONFIG.gnina_binary_path, CONFIG.dry_run, CONFIG.validate_docking_binaries_on_startup = saved

    def test_successful_docking_returns_cnnscore(self) -> None:
        with patch("autoantibiotic.io_utils.ToolExecutor.run", return_value=_make_tool_result()):
            score = _run_docking_tool(
                "gnina",
                receptor_pdbqt="rec.pdbqt",
                ligand_pdbqt="lig.pdbqt",
                output_pdbqt="out.pdbqt",
                center=np.array([0.0, 0.0, 0.0]),
                box_size=(20.0, 20.0, 20.0),
            )
        assert score == pytest.approx(0.9123)

    def test_nonzero_returncode_raises_error(self) -> None:
        with patch(
            "autoantibiotic.docking.ToolExecutor.run",
            return_value=_make_tool_result(returncode=1, stderr=_GNINA_FAILURE_STDERR),
        ):
            with pytest.raises(DockingParseError):
                _run_docking_tool(
                    "gnina",
                    receptor_pdbqt="rec.pdbqt",
                    ligand_pdbqt="lig.pdbqt",
                    output_pdbqt="out.pdbqt",
                    center=np.array([0.0, 0.0, 0.0]),
                    box_size=(20.0, 20.0, 20.0),
                )

    def test_dry_run_returns_random_score(self) -> None:
        saved = CONFIG.dry_run
        CONFIG.dry_run = True
        try:
            score = _run_docking_tool(
                "gnina",
                receptor_pdbqt="rec.pdbqt",
                ligand_pdbqt="lig.pdbqt",
                output_pdbqt="out.pdbqt",
                center=np.array([0.0, 0.0, 0.0]),
                box_size=(20.0, 20.0, 20.0),
            )
            assert score is not None
            assert 0.5 <= score <= 0.95
        finally:
            CONFIG.dry_run = saved

    def test_uses_configured_binary_path(self) -> None:
        saved_path = CONFIG.gnina_binary_path
        CONFIG.gnina_binary_path = "/custom/path/gnina"
        try:
            with patch("autoantibiotic.docking.ToolExecutor.run") as mock_run:
                mock_run.return_value = _make_tool_result()
                _run_docking_tool(
                    "gnina",
                    receptor_pdbqt="rec.pdbqt",
                    ligand_pdbqt="lig.pdbqt",
                    output_pdbqt="out.pdbqt",
                    center=np.array([1.0, 2.0, 3.0]),
                    box_size=(15.0, 15.0, 15.0),
                )
                binary_arg = mock_run.call_args[0][0]
                assert binary_arg == "/custom/path/gnina"
        finally:
            CONFIG.gnina_binary_path = saved_path


class TestParseVinaEnergy:
    """``parse_vina_energy`` still works correctly (regression)."""

    def test_stdout_mode_line(self) -> None:
        stdout = (
            "mode |   affinity | dist from best mode\n"
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
            "   2       -7.500       1.234      2.345\n"
        )
        energy = parse_vina_energy(stdout)
        assert energy == pytest.approx(-8.123)

    def test_affinity_fallback(self) -> None:
        stdout = "Affinity: -9.456 (kcal/mol)"
        energy = parse_vina_energy(stdout)
        assert energy == pytest.approx(-9.456)

    def test_no_energy_returns_none(self) -> None:
        assert parse_vina_energy("No docking results") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_vina_energy("") is None


# ── dock_compound with GNINA ───────────────────────────────────────

class TestDockCompoundWithGnina:
    """``dock_compound`` uses GNINA when config says so."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.work_dir = str(tmp_path)
        self.saved_config = (
            CONFIG.use_gnina, CONFIG.gnina_binary_path, CONFIG.dry_run,
            CONFIG.validate_docking_binaries_on_startup,
        )
        CONFIG.dry_run = False
        CONFIG.use_gnina = True
        CONFIG.gnina_binary_path = "gnina"
        CONFIG.validate_docking_binaries_on_startup = False
        self.record = CompoundRecord(
            compound_id="TEST-GNINA",
            smiles="c1ccccc1O",
            mol=Chem.MolFromSmiles("c1ccccc1O"),
        )
        yield
        CONFIG.use_gnina, CONFIG.gnina_binary_path, CONFIG.dry_run, CONFIG.validate_docking_binaries_on_startup = self.saved_config

    def test_gnina_success_returns_score(self) -> None:
        with patch("autoantibiotic.docking.prepare_ligand_pdbqt", return_value=True):
            with patch("autoantibiotic.docking._run_docking_tool", return_value=0.9123):
                energy, method = dock_compound(
                    self.record,
                    receptor_pdbqt="rec.pdbqt",
                    center=np.array([0.0, 0.0, 0.0]),
                    box_size=(20.0, 20.0, 20.0),
                    work_dir=self.work_dir,
                    tag="test",
                )
        assert energy == pytest.approx(0.9123)
        assert method == "GNINA"

    def test_gnina_failure_falls_back_to_vina(self) -> None:
        def _mock_dock(tool_name, *args, **kwargs):
            return None if tool_name == "gnina" else -8.5

        with patch("autoantibiotic.docking.prepare_ligand_pdbqt", return_value=True):
            with patch("autoantibiotic.docking._run_docking_tool", side_effect=_mock_dock):
                energy, method = dock_compound(
                    self.record,
                    receptor_pdbqt="rec.pdbqt",
                    center=np.array([0.0, 0.0, 0.0]),
                    box_size=(20.0, 20.0, 20.0),
                    work_dir=self.work_dir,
                    tag="test",
                )
        assert energy == pytest.approx(-8.5)
        assert method == "Vina"

    def test_both_fail_return_none(self) -> None:
        with patch("autoantibiotic.docking.prepare_ligand_pdbqt", return_value=True):
            with patch("autoantibiotic.docking._run_docking_tool", return_value=None):
                energy, method = dock_compound(
                    self.record,
                    receptor_pdbqt="rec.pdbqt",
                    center=np.array([0.0, 0.0, 0.0]),
                    box_size=(20.0, 20.0, 20.0),
                    work_dir=self.work_dir,
                    tag="test",
                )
        assert energy is None
        assert method == "Vina"

    def test_returns_method_string_in_dry_run_gnina(self) -> None:
        CONFIG.dry_run = True
        CONFIG.use_gnina = True
        with patch("autoantibiotic.docking.prepare_ligand_pdbqt", return_value=True):
            with patch("autoantibiotic.docking._run_docking_tool") as mock_dock:
                energy, method = dock_compound(
                    self.record,
                    receptor_pdbqt="rec.pdbqt",
                    center=np.array([0.0, 0.0, 0.0]),
                    box_size=(20.0, 20.0, 20.0),
                    work_dir=self.work_dir,
                    tag="test",
                )
                assert method == "GNINA"
                assert energy is not None

    def test_returns_method_string_in_dry_run_vina(self) -> None:
        CONFIG.dry_run = True
        CONFIG.use_gnina = False
        with patch("autoantibiotic.docking.prepare_ligand_pdbqt", return_value=True):
            with patch("autoantibiotic.docking._run_docking_tool") as mock_dock:
                energy, method = dock_compound(
                    self.record,
                    receptor_pdbqt="rec.pdbqt",
                    center=np.array([0.0, 0.0, 0.0]),
                    box_size=(20.0, 20.0, 20.0),
                    work_dir=self.work_dir,
                    tag="test",
                )
                assert method == "Vina"
                assert energy is not None

    def test_gnina_not_used_when_disabled(self) -> None:
        CONFIG.use_gnina = False
        with patch("autoantibiotic.docking.prepare_ligand_pdbqt", return_value=True):
            with patch("autoantibiotic.docking._run_docking_tool", return_value=-7.2) as mock_dock:
                energy, method = dock_compound(
                    self.record,
                    receptor_pdbqt="rec.pdbqt",
                    center=np.array([0.0, 0.0, 0.0]),
                    box_size=(20.0, 20.0, 20.0),
                    work_dir=self.work_dir,
                    tag="test",
                )
                assert mock_dock.call_args[0][0] == "vina"
                assert method == "Vina"


# ── Ensemble docking ───────────────────────────────────────────────

class TestDockCompoundEnsemble:
    """``dock_compound_ensemble`` aggregates scores correctly."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path) -> None:
        self.work_dir = str(tmp_path)
        self.saved_config = (
            CONFIG.consensus_scoring_method, CONFIG.use_gnina, CONFIG.dry_run,
        )
        CONFIG.dry_run = False
        CONFIG.use_gnina = False
        self.record = CompoundRecord(
            compound_id="TEST-ENS",
            smiles="c1ccccc1O",
            mol=Chem.MolFromSmiles("c1ccccc1O"),
        )
        yield
        CONFIG.consensus_scoring_method, CONFIG.use_gnina, CONFIG.dry_run = self.saved_config

    def _mock_dock_compound(self, *args, **kwargs) -> Tuple[Optional[float], str]:
        tag = args[5] if len(args) > 5 else ""
        if "ens0" in tag:
            return -8.0, "Vina"
        elif "ens1" in tag:
            return -6.0, "Vina"
        elif "ens2" in tag:
            return -10.0, "Vina"
        return None, "None"

    def test_mean_consensus(self) -> None:
        CONFIG.consensus_scoring_method = "mean"
        with patch("autoantibiotic.docking.dock_compound", side_effect=self._mock_dock_compound):
            score, method = dock_compound_ensemble(
                self.record,
                receptor_pdbqt_list=["r1.pdbqt", "r2.pdbqt", "r3.pdbqt"],
                center_list=[np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])],
                box_size=(20.0, 20.0, 20.0),
                work_dir=self.work_dir,
                tag="ens",
            )
        assert score == pytest.approx(-8.0)
        assert method == "Vina"

    def test_min_consensus(self) -> None:
        CONFIG.consensus_scoring_method = "min"
        with patch("autoantibiotic.docking.dock_compound", side_effect=self._mock_dock_compound):
            score, method = dock_compound_ensemble(
                self.record,
                receptor_pdbqt_list=["r1.pdbqt", "r2.pdbqt", "r3.pdbqt"],
                center_list=[np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])],
                box_size=(20.0, 20.0, 20.0),
                work_dir=self.work_dir,
                tag="ens",
            )
        assert score == pytest.approx(-10.0)
        assert method == "Vina"

    def test_median_consensus(self) -> None:
        CONFIG.consensus_scoring_method = "median"
        with patch("autoantibiotic.docking.dock_compound", side_effect=self._mock_dock_compound):
            score, method = dock_compound_ensemble(
                self.record,
                receptor_pdbqt_list=["r1.pdbqt", "r2.pdbqt", "r3.pdbqt"],
                center_list=[np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])],
                box_size=(20.0, 20.0, 20.0),
                work_dir=self.work_dir,
                tag="ens",
            )
        assert score == pytest.approx(-8.0)
        assert method == "Vina"

    def test_all_fail_returns_none(self) -> None:
        with patch("autoantibiotic.docking.dock_compound", return_value=(None, "None")):
            score, method = dock_compound_ensemble(
                self.record,
                receptor_pdbqt_list=["r1.pdbqt"],
                center_list=[np.array([0, 0, 0])],
                box_size=(20.0, 20.0, 20.0),
                work_dir=self.work_dir,
                tag="ens",
            )
        assert score is None
        assert method == "None"

    def test_some_fail_uses_remaining(self) -> None:
        def _partial(*args, **kwargs) -> Tuple[Optional[float], str]:
            tag = args[5] if len(args) > 5 else ""
            return (-9.0, "Vina") if "ens0" in tag else (None, "None")

        CONFIG.consensus_scoring_method = "mean"
        with patch("autoantibiotic.docking.dock_compound", side_effect=_partial):
            score, method = dock_compound_ensemble(
                self.record,
                receptor_pdbqt_list=["r1.pdbqt", "r2.pdbqt"],
                center_list=[np.array([0, 0, 0]), np.array([1, 1, 1])],
                box_size=(20.0, 20.0, 20.0),
                work_dir=self.work_dir,
                tag="ens",
            )
        assert score == pytest.approx(-9.0)
        assert method == "Vina"

    def test_first_fail_uses_remaining(self) -> None:
        def _first_fails(*args, **kwargs) -> Tuple[Optional[float], str]:
            tag = args[5] if len(args) > 5 else ""
            if "ens0" in tag:
                return None, "None"
            return -7.0, "Vina"

        CONFIG.consensus_scoring_method = "mean"
        with patch("autoantibiotic.docking.dock_compound", side_effect=_first_fails):
            score, method = dock_compound_ensemble(
                self.record,
                receptor_pdbqt_list=["r1.pdbqt", "r2.pdbqt"],
                center_list=[np.array([0, 0, 0]), np.array([1, 1, 1])],
                box_size=(20.0, 20.0, 20.0),
                work_dir=self.work_dir,
                tag="ens",
            )
        assert score == pytest.approx(-7.0)
        assert method == "Vina"


# ── Error classification tests ──────────────────────────────────────

class TestErrorClassification:
    """Tests for _classify_tool_error, VinaError, and OpenBabelError."""

    def test_classify_vina_no_file(self) -> None:
        msg = _classify_tool_error("vina", "Error: Could not open receptor file")
        assert msg is not None
        assert "file not found" in msg.lower()

    def test_classify_vina_bad_alloc(self) -> None:
        msg = _classify_tool_error("vina", "std::bad_alloc")
        assert msg is not None
        assert "memory" in msg.lower()

    def test_classify_vina_no_match(self) -> None:
        msg = _classify_tool_error("vina", "everything is fine")
        assert msg is None

    def test_classify_gnina_cuda_error(self) -> None:
        msg = _classify_tool_error("gnina", "CUDA error: out of memory")
        assert msg is not None
        assert "cuda" in msg.lower()

    def test_classify_obabel_cannot_convert(self) -> None:
        msg = _classify_tool_error("obabel", "Cannot convert from format XYZ")
        assert msg is not None
        assert "cannot convert" in msg.lower()

    def test_classify_prepare_receptor_error(self) -> None:
        msg = _classify_tool_error("prepare_receptor", "Error: missing atoms")
        assert msg is not None
        assert "prepare_receptor" in msg.lower()

    def test_classify_unknown_tool(self) -> None:
        msg = _classify_tool_error("blarg", "anything")
        assert msg is None

    def test_vina_error_is_exception(self) -> None:
        err = VinaError("test error")
        assert isinstance(err, Exception)
        assert "test error" in str(err)

    def test_openbabel_error_is_exception(self) -> None:
        err = OpenBabelError("test error")
        assert isinstance(err, Exception)
        assert "test error" in str(err)


# ── DockingResultValidator tests ────────────────────────────────────

class TestDockingResultValidator:
    """Tests for the structured DockingResultValidator."""

    @pytest.fixture
    def validator(self) -> DockingResultValidator:
        return DockingResultValidator()

    # ── parse_vina ─────────────────────────────────────────────────

    def test_vina_tabular_output(self, validator: DockingResultValidator) -> None:
        stdout = (
            "mode |   affinity | dist from best mode\n"
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
            "   2       -7.500       1.234      2.345\n"
        )
        energy = validator.parse_vina(stdout)
        assert energy == pytest.approx(-8.123)

    def test_vina_tabular_without_header(self, validator: DockingResultValidator) -> None:
        """Should still parse mode lines even without detecting header."""
        stdout = (
            "   1       -8.123       0.000      0.000\n"
            "   2       -7.500       1.234      2.345\n"
        )
        energy = validator.parse_vina(stdout)
        assert energy == pytest.approx(-8.123)

    def test_vina_single_line_affinity(self, validator: DockingResultValidator) -> None:
        stdout = "Affinity: -9.456 (kcal/mol)"
        energy = validator.parse_vina(stdout)
        assert energy == pytest.approx(-9.456)

    def test_vina_affinity_no_parenthetical(self, validator: DockingResultValidator) -> None:
        stdout = "Affinity: -5.234"
        energy = validator.parse_vina(stdout)
        assert energy == pytest.approx(-5.234)

    def test_vina_malformed_output_no_numbers(self, validator: DockingResultValidator) -> None:
        stdout = (
            "mode |   affinity\n"
            "-----+----------\n"
            "   a       b\n"
        )
        assert validator.parse_vina(stdout) is None

    def test_vina_error_keyword_fatal(self, validator: DockingResultValidator) -> None:
        stdout = "Fatal Error: something went wrong\n"
        assert validator.parse_vina(stdout) is None

    def test_vina_error_keyword_segfault(self, validator: DockingResultValidator) -> None:
        stdout = "Segmentation fault\n"
        assert validator.parse_vina(stdout) is None

    def test_vina_empty_string(self, validator: DockingResultValidator) -> None:
        assert validator.parse_vina("") is None

    def test_vina_whitespace_only(self, validator: DockingResultValidator) -> None:
        assert validator.parse_vina("   \n   \n") is None

    def test_vina_mixed_success_and_error(self, validator: DockingResultValidator) -> None:
        """Output with partial success but also an error keyword."""
        stdout = (
            "mode |   affinity | dist from best mode\n"
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
            "Error: Could not open receptor file\n"
        )
        assert validator.parse_vina(stdout) is None

    def test_vina_missing_columns(self, validator: DockingResultValidator) -> None:
        stdout = (
            "mode |   affinity\n"
            "-----+----------\n"
            "   1       -8.1\n"
            "   2\n"
        )
        energy = validator.parse_vina(stdout)
        assert energy == pytest.approx(-8.1)

    def test_vina_logs_warning_on_multiple_modes(self, validator: DockingResultValidator, caplog) -> None:
        import logging
        caplog.set_level(logging.WARNING)
        stdout = (
            "mode |   affinity | dist from best mode\n"
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
            "   2       -7.500       1.234      2.345\n"
        )
        energy = validator.parse_vina(stdout)
        assert energy == pytest.approx(-8.123)
        assert any("Multiple docking modes found" in rec.message for rec in caplog.records)

    # ── parse_gnina ────────────────────────────────────────────────

    def test_gnina_cnnscore_parsed(self, validator: DockingResultValidator) -> None:
        stdout = (
            "CNNscore    :   0.8567\n"
            "CNNaffinity :   7.2345\n"
        )
        score = validator.parse_gnina(stdout)
        assert score == pytest.approx(0.8567)

    def test_gnina_cnnaffinity_fallback(self, validator: DockingResultValidator) -> None:
        stdout = "CNNaffinity :   7.2345\n"
        score = validator.parse_gnina(stdout)
        assert score == pytest.approx(7.2345)

    def test_gnina_no_score_returns_none(self, validator: DockingResultValidator) -> None:
        assert validator.parse_gnina("No GNINA output") is None

    def test_gnina_empty_string(self, validator: DockingResultValidator) -> None:
        assert validator.parse_gnina("") is None

    def test_gnina_multi_mode_returns_first_cnnscore(self, validator: DockingResultValidator) -> None:
        stdout = (
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
            "CNNscore    :   0.9123\n"
            "CNNaffinity :   8.4567\n"
            "   2       -7.500       1.234      2.345\n"
            "CNNscore    :   0.8500\n"
            "CNNaffinity :   7.8000\n"
        )
        score = validator.parse_gnina(stdout)
        assert score == pytest.approx(0.9123)

    def test_gnina_missing_cnnscore_fallback_cnnaffinity(self, validator: DockingResultValidator) -> None:
        stdout = (
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
            "CNNaffinity :   8.4567\n"
        )
        score = validator.parse_gnina(stdout)
        assert score == pytest.approx(8.4567)

    def test_gnina_nan_values_ignored(self, validator: DockingResultValidator) -> None:
        stdout = (
            "CNNscore    :   nan\n"
            "CNNaffinity :   7.2345\n"
        )
        score = validator.parse_gnina(stdout)
        assert score == pytest.approx(7.2345)

    def test_gnina_cuda_error(self, validator: DockingResultValidator) -> None:
        stdout = (
            "CUDA error: out of memory\n"
            "CNNscore    :   0.8567\n"
        )
        assert validator.parse_gnina(stdout) is None

    def test_gnina_cudna_error_variant(self, validator: DockingResultValidator) -> None:
        stdout = "cudaError: all CUDA-capable devices are busy or unavailable\n"
        assert validator.parse_gnina(stdout) is None

    def test_gnina_fatal_error(self, validator: DockingResultValidator) -> None:
        stdout = "Fatal Error: could not read input\n"
        assert validator.parse_gnina(stdout) is None

    def test_gnina_partial_output_no_crash(self, validator: DockingResultValidator) -> None:
        """Partial output from a timeout scenario should not crash."""
        stdout = "Reading input ...\n"
        assert validator.parse_gnina(stdout) is None

    def test_gnina_vina_stdout_not_parsed(self, validator: DockingResultValidator) -> None:
        """Vina's table should not produce a false positive GNINA score."""
        stdout = (
            "mode |   affinity | dist from best mode\n"
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
        )
        assert validator.parse_gnina(stdout) is None

    # ── validate_binary_health ─────────────────────────────────────

    def test_vina_version_valid(self, validator: DockingResultValidator) -> None:
        assert validator.validate_binary_health("vina", "Vina 1.2.3")

    def test_vina_autodock_version_valid(self, validator: DockingResultValidator) -> None:
        assert validator.validate_binary_health("vina", "AutoDock Vina 1.2.3")

    def test_vina_version_unknown(self, validator: DockingResultValidator) -> None:
        assert not validator.validate_binary_health("vina", "Vina 0.9.0")

    def test_vina_version_empty(self, validator: DockingResultValidator) -> None:
        assert not validator.validate_binary_health("vina", "")

    def test_gnina_version_valid(self, validator: DockingResultValidator) -> None:
        assert validator.validate_binary_health("gnina", "GNINA 1.1")

    def test_gnina_version_with_path(self, validator: DockingResultValidator) -> None:
        assert validator.validate_binary_health("gnina", "gnina 1.0.1")

    def test_gnina_version_unknown(self, validator: DockingResultValidator) -> None:
        assert not validator.validate_binary_health("gnina", "GNINA 0.9")

    def test_gnina_version_empty(self, validator: DockingResultValidator) -> None:
        assert not validator.validate_binary_health("gnina", "")

    def test_unknown_tool_name(self, validator: DockingResultValidator) -> None:
        assert not validator.validate_binary_health("blarg", "blarg 1.0")

    # ── DockingParseError ─────────────────────────────────────────

    def test_docking_parse_error_is_exception(self) -> None:
        err = DockingParseError("test error")
        assert isinstance(err, Exception)
        assert "test error" in str(err)
