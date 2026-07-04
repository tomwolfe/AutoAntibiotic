"""Integration tests for the AutoAntibiotic discovery pipeline.

Tests exercise the full pipeline in dry-run mode with all external
dependencies (PDB download, subprocess calls) mocked.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from autoantibiotic.config import CONFIG, CompoundRecord
from autoantibiotic.main import main
from autoantibiotic.library_gen import generate_candidate_library, apply_filters


_DUMMY_RECORDS = [
    CompoundRecord(
        compound_id="TEST-001", smiles="c1ccccc1O",
        mol=Chem.MolFromSmiles("c1ccccc1O"),
        passes_lipinski=True, qed_score=0.85, max_similarity=0.0,
    ),
    CompoundRecord(
        compound_id="TEST-002", smiles="c1ccccc1C(=O)O",
        mol=Chem.MolFromSmiles("c1ccccc1C(=O)O"),
        passes_lipinski=True, qed_score=0.80, max_similarity=0.0,
    ),
    CompoundRecord(
        compound_id="TEST-003", smiles="c1ccc(O)cc1",
        mol=Chem.MolFromSmiles("c1ccc(O)cc1"),
        passes_lipinski=True, qed_score=0.90, max_similarity=0.0,
    ),
]


def _mock_prepare_targets(pdb_dir, work_dir, deps):
    """Return a synthetic target dictionary without downloading PDBs."""
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    dummy_pdb = os.path.join(pdb_dir, "dummy.pdb")
    if not os.path.exists(dummy_pdb):
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A 237       1.500   1.500   1.500  1.00  0.00           C\nEND\n")
    dummy_pdbqt = dummy_pdb.replace(".pdb", ".pdbqt")
    if not os.path.exists(dummy_pdbqt):
        # minimal valid PDBQT
        with open(dummy_pdbqt, "w") as f:
            f.write("ROOT\nENDROOT\nTORSDOF 0\n")

    return {
        "holo_pdb": dummy_pdb,
        "PBP2a": {
            "pdbqt": dummy_pdbqt,
            "allosteric_center": np.array([2.5, 2.5, 2.5]),
            "active_center": np.array([4.5, 4.5, 4.5]),
        },
        "trypsin": {
            "pdbqt": dummy_pdbqt,
            "active_center": np.array([6.5, 6.5, 6.5]),
        },
        "CES1": {
            "pdbqt": dummy_pdbqt,
            "active_center": np.array([9.5, 9.5, 9.5]),
        },
    }


@pytest.fixture(autouse=True)
def temp_output_dir(monkeypatch: pytest.MonkeyPatch) -> str:
    """Use a temporary output directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(CONFIG, "output_dir", Path(tmpdir))
        yield tmpdir


@pytest.fixture(autouse=True)
def mock_deps_and_targets():
    """Mock dependency verification, target preparation, and redocking."""
    deps = {
        "rdkit": True, "meeko": True, "Bio": True,
        "vina": True, "obabel": True, "prepare_receptor": True,
        "USE_VINA": True, "USE_OBABEL": True,
    }
    with patch("autoantibiotic.main.verify_dependencies", return_value=deps):
        with patch("autoantibiotic.main.prepare_targets", side_effect=_mock_prepare_targets):
            with patch("autoantibiotic.main.run_redocking_validation", return_value=(False, None)):
                with patch("autoantibiotic.main.generate_candidate_library", return_value=_DUMMY_RECORDS):
                    with patch("autoantibiotic.main.apply_filters", return_value=_DUMMY_RECORDS):
                        yield


EXPECTED_CSV_COLUMNS = [
    "Compound_ID",
    "SMILES",
    "PBP2a_Allosteric_Energy",
    "PBP2a_Active_Energy",
    "Human_Trypsin_Energy",
    "Human_CES1_Energy",
    "Shape_Score",
    "Selectivity_Index",
    "Max_Similarity",
    "Passes_Lipinski",
    "QED_Score",
    "Scoring_Method",
    "Binding_Mode_Notes",
]


class TestFullPipelineDryRun:
    """End-to-end dry-run integration tests."""

    def test_full_pipeline_dry_run(self) -> None:
        """Pipeline in dry-run mode produces a valid CSV report."""
        main(["--dry-run"])

        csv_path = CONFIG.output_dir / "top_candidates.csv"
        assert csv_path.exists(), f"CSV report not found at {csv_path}"

        df = pd.read_csv(csv_path)
        for col in EXPECTED_CSV_COLUMNS:
            assert col in df.columns, f"Missing column {col} in CSV"

        assert len(df) > 0, "CSV should contain at least one compound"

    def test_dry_run_csv_columns_match_expected(self) -> None:
        """CSV columns exactly match the expected schema."""
        main(["--dry-run"])
        df = pd.read_csv(CONFIG.output_dir / "top_candidates.csv")
        assert list(df.columns) == EXPECTED_CSV_COLUMNS, (
            f"Column mismatch.\n"
            f"Expected: {EXPECTED_CSV_COLUMNS}\n"
            f"Got:      {list(df.columns)}"
        )

    def test_dry_run_produces_unique_compound_ids(self) -> None:
        """All compound IDs in dry-run output should be unique."""
        main(["--dry-run"])
        df = pd.read_csv(CONFIG.output_dir / "top_candidates.csv")
        assert df["Compound_ID"].is_unique, "Duplicate compound IDs found"

    def test_dry_run_energy_values_not_na(self) -> None:
        """Dry-run should produce mock energy values (not N/A) for PBP2a."""
        main(["--dry-run"])
        df = pd.read_csv(CONFIG.output_dir / "top_candidates.csv")
        assert df["PBP2a_Allosteric_Energy"].notna().all(), (
            "Allosteric energies should not be N/A in dry-run"
        )

    def test_dry_run_html_report_generated(self) -> None:
        """HTML report should exist after pipeline completion."""
        main(["--dry-run"])
        html_path = CONFIG.output_dir / "report.html"
        assert html_path.exists(), "HTML report not found"

    def test_dry_run_html_report_contains_plotly(self) -> None:
        """HTML report should embed interactive Plotly charts."""
        main(["--dry-run"])
        html_path = CONFIG.output_dir / "report.html"
        content = html_path.read_text()
        assert "plotly" in content, "HTML report missing Plotly JavaScript"
        assert "Plotly" in content, "HTML report missing Plotly traces"
        assert "PCA" in content, "HTML report missing PCA diversity plot"

    def test_dry_run_html_report_contains_candidate_table(self) -> None:
        """HTML report should contain the top-candidates table."""
        main(["--dry-run"])
        html_path = CONFIG.output_dir / "report.html"
        content = html_path.read_text()
        assert "<table>" in content, "HTML report missing candidates table"
        assert "TEST-" in content, "HTML report missing compound IDs"

    def test_dry_run_log_file_generated(self) -> None:
        """Pipeline log file should exist."""
        main(["--dry-run"])
        log_path = CONFIG.output_dir / "pipeline.log"
        assert log_path.exists(), "Pipeline log not found"


class TestLibraryLipinskiCompliance:
    """Verify that generated candidates pass Lipinski filters."""

    def test_generated_library_passes_lipinski(self) -> None:
        """Candidates from generate_candidate_library should have a high
        proportion passing Lipinski Rule-of-5."""
        records = generate_candidate_library(target_count=20, seed=42)
        assert len(records) > 0, "Library generation should produce compounds"

        passed = apply_filters(records)
        assert len(passed) > 0, "At least one compound should pass filters"

        for rec in passed:
            assert rec.passes_lipinski, (
                f"{rec.compound_id} failed Lipinski filter"
            )
            assert rec.qed_score >= CONFIG.qed_threshold, (
                f"{rec.compound_id} QED {rec.qed_score:.3f} < {CONFIG.qed_threshold}"
            )

    def test_lipinski_compliant_molecules_have_valid_properties(self) -> None:
        """Molecules passing filters should have measurable ADMET properties."""
        records = generate_candidate_library(target_count=20, seed=42)
        passed = apply_filters(records)

        for rec in passed:
            assert rec.max_similarity >= 0.0, "Similarity should be non-negative"
            assert rec.qed_score > 0.0, "QED score should be positive"
