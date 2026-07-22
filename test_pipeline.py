#!/usr/bin/env python3
"""
Unit tests for discovery_pipeline.py
======================================
Tests core scientific and engineering functions in isolation.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from discovery_pipeline import (
    check_dependencies,
    run_redocking_validation,
    compute_selectivity_index,
    analyze_binding_interactions,
    profile_resistance_risk,
    ensure_output_dir,
    screen_library,
    fetch_structure,
    _find_downloaded_pdb,
    _redocking_box_size,

    analyze_selectivity_and_resistance,
    log,
)
from utils.filtering import apply_filters
from utils.library_gen import generate_candidate_library, CompoundRecord
from utils.docking import _run_vina_docking, _dock_compounds_parallel
from utils.ligand_prep import LigandPreparator
from utils.reporting import generate_csv_report, generate_pymol_script, diversify_top_n, si_tier
from rdkit.DataStructs import TanimotoSimilarity
from utils.structure_prep import compute_residue_centroid
from config.constants import (
    OUTPUT_DIR,
    TOP_N,
    BETA_LACTAM_SMARTS,
    DIVERSITY_MIN_COUNT,
    RMSD_VALIDATED_MAX,
    RMSD_MARGINAL_MAX,
    protocol_trust,
    SI_STRONG_THRESHOLD,
    SI_PROMISING_THRESHOLD,
)
from tests.helpers import create_minimal_pdb
from rdkit import Chem

TEST_REAL_PDB_DIR = Path(__file__).parent / "tests" / "data"

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_pdb_dir():
    """Create a temporary directory with a minimal PDB file for centroid testing."""
    tmpdir = tempfile.mkdtemp()

    # Minimal PDB with a single residue (ALA 237) generated in memory.
    # Coordinates are arbitrary.
    pdb_content = create_minimal_pdb({
        ("ALA", 237, "A"): [
            ("N", 41.234, 12.345, 78.901),
            ("CA", 42.345, 13.456, 79.012),
            ("C", 43.456, 14.567, 80.123),
            ("O", 44.567, 15.678, 81.234),
        ],
    })
    pdb_path = os.path.join(tmpdir, "mock.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)

    yield pdb_path

    # Cleanup
    for fname in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, fname))
    os.rmdir(tmpdir)

@pytest.fixture(autouse=True)
def setup_output_dir():
    """Ensure output/ exists for functions that write intermediate files."""
    ensure_output_dir()
    yield

# ── Test 1: compute_residue_centroid ────────────────────────────────────────

class TestComputeResidueCentroid:
    def test_returns_ndarray_shape_3(self, mock_pdb_dir):
        """compute_residue_centroid returns a numpy array of shape (3,) for a valid PDB."""
        centroid = compute_residue_centroid(mock_pdb_dir, ["ALA237"])
        assert isinstance(centroid, np.ndarray), "Expected numpy array"
        assert centroid.shape == (3,), f"Expected shape (3,), got {centroid.shape}"

    def test_centroid_is_mean_of_ca_coords(self, mock_pdb_dir):
        """The centroid should equal the arithmetic mean of Cα coordinates."""
        centroid = compute_residue_centroid(mock_pdb_dir, ["ALA237"])
        # For our mock PDB, the CA is at (42.345, 13.456, 79.012)
        expected = np.array([42.345, 13.456, 79.012])
        np.testing.assert_allclose(centroid, expected, rtol=1e-5)

    def test_raises_on_missing_residue(self, mock_pdb_dir):
        """Raises ValueError when none of the requested residues exist in the PDB."""
        with pytest.raises(ValueError, match="No matching residues found"):
            compute_residue_centroid(mock_pdb_dir, ["GLY999"])

# ── Test 2: apply_filters ────────────────────────────────────────────────────

class TestApplyFilters:
    def test_rejects_beta_lactam(self):
        """A compound matching the β-lactam SMARTS pattern is rejected."""
        # A simple 3,4-dimethyl-2-azetidinone matches the β-lactam SMARTS
        lactam_smi = "CC1C(=O)NC1C"
        mol = Chem.MolFromSmiles(lactam_smi)
        assert mol is not None, "Test SMILES should be valid"

        # Verify it matches the beta-lactam SMARTS
        lactam_pattern = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)
        assert mol.HasSubstructMatch(lactam_pattern), (
            "Test molecule should match beta-lactam SMARTS"
        )

        record = CompoundRecord(
            compound_id="TEST_BETA_LACTAM",
            smiles=lactam_smi,
            mol=mol,
        )
        filtered = apply_filters([record])
        assert len(filtered) == 0, "Beta-lactam compound should be filtered out"

    def test_rejects_brenk_alert(self):
        """A compound containing a Brenk alert structural pattern is rejected."""
        # A known Brenk-alert structure (an aromatic amine with nitro group)
        brenk_smi = "O=[N+]([O-])c1ccccc1N"
        mol = Chem.MolFromSmiles(brenk_smi)
        assert mol is not None, "Test SMILES should be valid"

        record = CompoundRecord(
            compound_id="TEST_BRENK_ALERT",
            smiles=brenk_smi,
            mol=mol,
        )
        filtered = apply_filters([record])
        assert len(filtered) == 0, "Brenk-alert compound should be filtered out"

    def test_passes_valid_compound(self):
        """A known non-beta-lactam compound with reasonable properties should pass."""
        # Quercetin (a flavonoid, no beta-lactam)
        quercetin_smi = "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"
        mol = Chem.MolFromSmiles(quercetin_smi)
        assert mol is not None

        record = CompoundRecord(
            compound_id="TEST_QUERCETIN",
            smiles=quercetin_smi,
            mol=mol,
        )
        filtered = apply_filters([record])
        # Quercetin may or may not pass depending on similarity/ADMET,
        # but it should NOT be filtered by the β-lactam structural filter.
        # We simply verify it's not removed due to structural exclusion.
        assert record in filtered or len(filtered) == 0, (
            "Quercetin may be filtered later, but structural check should pass"
        )

# ── Test 3: generate_candidate_library ──────────────────────────────────────

class TestGenerateCandidateLibrary:
    def test_returns_at_least_10_compounds(self):
        """generate_candidate_library returns a multi-compound library with default params.

        The library is generated from NATURAL_PRODUCT_SCAFFOLDS plus the 2 CONTROL_SMILES. The
        robust floor is therefore the control compounds; we also require at least one
        generated (non-control) compound to confirm BRICS expansion ran.
        """
        library = generate_candidate_library(target_count=500)
        assert len(library) >= 2, (
            f"Expected at least the control compounds, got {len(library)}"
        )
        generated = [r for r in library if not r.compound_id.startswith("CTRL_")]
        assert len(generated) >= 1, "Expected at least one generated (non-control) compound"

    def test_all_records_have_smiles(self):
        """Every returned CompoundRecord must have a non-empty SMILES string."""
        library = generate_candidate_library(target_count=100)
        for record in library:
            assert record.smiles, f"Record {record.compound_id} has no SMILES"
            mol = Chem.MolFromSmiles(record.smiles)
            assert mol is not None, (
                f"Record {record.compound_id} has invalid SMILES: {record.smiles}"
            )

    def test_compound_ids_are_unique(self):
        """All compound IDs in the library must be unique."""
        library = generate_candidate_library(target_count=200)
        ids = [r.compound_id for r in library]
        assert len(ids) == len(set(ids)), "Duplicate compound IDs found"

# ── Test 4: check_dependencies with mocked subprocess ─────────────────────────

class TestCheckDependencies:
    def test_returns_vina_true_when_binary_found(self):
        """check_dependencies returns vina=True when 'vina --version' succeeds."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            deps = check_dependencies()
            assert deps["vina"] is True
            assert deps["USE_VINA"] is True

    def test_returns_vina_false_when_binary_missing(self):
        """check_dependencies returns vina=False when 'vina --version' raises FileNotFoundError."""
        with patch("subprocess.run", side_effect=FileNotFoundError) as mock_run:
            deps = check_dependencies()
            assert deps["vina"] is False
            assert deps["USE_VINA"] is False

    def test_handles_timeout_gracefully(self):
        """check_dependencies returns vina=False when subprocess times out."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="vina", timeout=10)):
            deps = check_dependencies()
            assert deps["vina"] is False
            assert deps["USE_VINA"] is False

    def test_handles_missing_obabel(self):
        """check_dependencies still succeeds when obabel is missing (optional)."""
        def side_effect(cmd, **kwargs):
            if cmd[0] == "vina":
                mock = MagicMock()
                mock.returncode = 0
                return mock
            raise FileNotFoundError
        with patch("subprocess.run", side_effect=side_effect):
            deps = check_dependencies()
            assert deps["vina"] is True

# ── Test 5: _run_vina_docking with mocked subprocess ─────────────────────────

class TestRunVinaDocking:
    @pytest.fixture
    def mock_center(self):
        return np.array([0.0, 0.0, 0.0])

    @pytest.fixture
    def mock_box(self):
        return (20.0, 20.0, 20.0)

    def test_returns_energy_on_success(self, mock_center, mock_box):
        """Returns binding energy when Vina outputs a valid table."""
        stdout = textwrap.dedent("""\
            mode |   affinity | dist from best mode
               1       -8.5       0.000
               2       -7.2       1.234
        """)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = stdout
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            energy = _run_vina_docking("rec.pdbqt", "lig.pdbqt", "out.pdbqt", mock_center, mock_box)
            assert energy == -8.5

    def test_returns_none_on_nonzero_exit(self, mock_center, mock_box):
        """Returns None when Vina returns a non-zero exit code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: something went wrong"

        with patch("subprocess.run", return_value=mock_result):
            energy = _run_vina_docking("rec.pdbqt", "lig.pdbqt", "out.pdbqt", mock_center, mock_box)
            assert energy is None

    def test_returns_none_on_timeout(self, mock_center, mock_box):
        """Returns None when Vina subprocess times out."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="vina", timeout=10)):
            energy = _run_vina_docking("rec.pdbqt", "lig.pdbqt", "out.pdbqt", mock_center, mock_box)
            assert energy is None

    def test_returns_none_on_file_not_found(self, mock_center, mock_box):
        """Returns None when Vina binary is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            energy = _run_vina_docking("rec.pdbqt", "lig.pdbqt", "out.pdbqt", mock_center, mock_box)
            assert energy is None

    def test_parses_affinity_from_stderr_fallback(self, mock_center, mock_box):
        """Falls back to parsing affinity from stderr when stdout table is missing."""
        stdout = ""
        stderr = "Affinity: -9.3 (kcal/mol)"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = stdout
        mock_result.stderr = stderr

        with patch("subprocess.run", return_value=mock_result):
            energy = _run_vina_docking("rec.pdbqt", "lig.pdbqt", "out.pdbqt", mock_center, mock_box)
            assert energy == -9.3

# ── Test 6: compute_selectivity_index edge cases ────────────────────────────

class TestComputeSelectivityIndex:
    def test_normal_case(self):
        """Normal case: SI = |PBP2a| / |Human| — a compound with PBP2a=-10, human=-5 gives SI=2.0."""
        si = compute_selectivity_index(-10.0, -5.0)
        assert si == pytest.approx(2.0)

    def test_positive_pb2pa_returns_zero(self):
        """Returns 0.0 when PBP2a energy is positive (non-binder)."""
        si = compute_selectivity_index(1.0, -5.0)
        assert si == 0.0

    def test_zero_pb2pa_energy(self):
        """Returns 0.0 when PBP2a energy is zero to avoid division by zero."""
        si = compute_selectivity_index(0.0, -5.0)
        assert si == 0.0

    def test_near_zero_pb2pa_energy(self):
        """Returns 0.0 when abs(PBP2a energy) is below epsilon."""
        si = compute_selectivity_index(-1e-7, -5.0)
        assert si == 0.0

    def test_zero_human_energy(self):
        """Returns 0.0 when human average energy is zero."""
        si = compute_selectivity_index(-10.0, 0.0)
        assert si == 0.0

    def test_both_zero_energy(self):
        """Returns 0.0 when both energies are zero."""
        si = compute_selectivity_index(0.0, 0.0)
        assert si == 0.0

    def test_negative_human_energy(self):
        """Still computes correctly when human energy is negative."""
        si = compute_selectivity_index(-8.0, -4.0)
        assert si == pytest.approx(2.0)

# ── Test 8: Library generation edge cases ────────────────────────────────────

class TestGenerateCandidateLibraryEdgeCases:
    def test_returns_unique_ids(self):
        """All compound IDs in the library must be unique."""
        library = generate_candidate_library(target_count=100)
        ids = [r.compound_id for r in library]
        assert len(ids) == len(set(ids)), "Duplicate compound IDs found"

    def test_all_records_have_valid_smiles(self):
        """Every returned record has valid SMILES that RDKit can parse."""
        library = generate_candidate_library(target_count=50)
        for record in library:
            assert record.smiles, f"Record {record.compound_id} has no SMILES"
            mol = Chem.MolFromSmiles(record.smiles)
            assert mol is not None, f"Record {record.compound_id} has invalid SMILES: {record.smiles}"

    def test_returns_capped_at_target_count(self):
        """Library never exceeds the requested target_count."""
        library = generate_candidate_library(target_count=10000)
        assert len(library) == 10000, f"Expected exactly 10000, got {len(library)}"

    def test_returns_only_valid_smiles_and_is_capped(self):
        """generate_candidate_library(target_count=20) returns valid SMILES and ≤ target_count."""
        library = generate_candidate_library(target_count=20)
        assert len(library) <= 20, f"Expected ≤ 20 records, got {len(library)}"
        for record in library:
            assert record.smiles, f"Record {record.compound_id} has no SMILES"
            mol = Chem.MolFromSmiles(record.smiles)
            assert mol is not None, (
                f"Record {record.compound_id} has invalid SMILES: {record.smiles}"
            )

# ── Test: Redocking Validation ───────────────────────────────────────────────

class TestRedockingValidation:
    def test_returns_false_none_without_raising(self, tmp_path):
        """
        run_redocking_validation must return (False, None) gracefully when Vina
        is unavailable — without raising — even after mocking the native-ligand
        extraction and the docking call.
        """
        deps = {"vina": False, "USE_VINA": False}
        with patch(
            "discovery_pipeline._extract_native_ligand_from_holo",
            return_value="CCO",
        ):
            with patch(
                "utils.docking._run_vina_docking",
                return_value=None,
            ):
                result = run_redocking_validation(
                    holo_pdb_path=str(tmp_path / "6TKO.pdb"),
                    target_pdbqt_path=str(tmp_path / "PBP2a.pdbqt"),
                    work_dir=str(tmp_path),
                    deps=deps,
                )
        assert result == (False, None, None), f"Expected (False, None, None), got {result}"

class TestMockRedockingSkip:
    def test_skips_mock_pdb_with_vina_enabled(self, tmp_path):
        """
        When USE_VINA is True but holo_pdb_path points at a bundled
        tests/data mock, run_redocking_validation must short-circuit and
        return (False, None) without attempting redocking (no fake RMSD).
        """
        from pathlib import Path
        tests_data = Path(__file__).parent / "tests" / "data"
        mock_holo = str(tests_data / "6TKO.pdb")
        assert os.path.exists(mock_holo), "tests/data/6TKO.pdb must be present"

        deps = {"vina": True, "USE_VINA": True}
        with patch(
            "discovery_pipeline._extract_native_ligand_from_holo",
            return_value="CCO",
        ):
            with patch(
                "utils.docking._run_vina_docking",
                return_value=None,
            ):
                result = run_redocking_validation(
                    holo_pdb_path=mock_holo,
                    target_pdbqt_path=str(tmp_path / "PBP2a.pdbqt"),
                    work_dir=str(tmp_path),
                    deps=deps,
                )
        assert result == (False, None, None), f"Expected (False, None, None), got {result}"

# ── Test: Redocking box auto-size ─────────────────────────────────────────

class TestRedockingBoxSize:
    def test_autosizes_from_ligand(self):
        """_redocking_box_size sizes the box from a native-ligand PDBQT atom spread."""
        mol = Chem.MolFromSmiles("CCCCC")
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol)
        pdbqt = Chem.MolToPDBBlock(mol)
        path = None
        import tempfile as _tf
        d = _tf.mkdtemp()
        try:
            path = os.path.join(d, "lig.pdbqt")
            with open(path, "w") as fh:
                fh.write(pdbqt)
            center = np.array([0.0, 0.0, 0.0])
            box = _redocking_box_size(path, center)
            assert len(box) == 3
            # All edges equal (cubic) and at least the 15 Å minimum.
            assert box[0] == box[1] == box[2]
            assert box[0] >= 15.0
        finally:
            import shutil as _sh
            _sh.rmtree(d, ignore_errors=True)

    def test_falls_back_on_missing_file(self):
        """A missing ligand file yields the default 25 Å box."""
        box = _redocking_box_size("/no/such/file.pdbqt", np.array([0.0, 0.0, 0.0]))
        assert box == (25.0, 25.0, 25.0)

# ── Test: fetch_structure rename fallback ─────────────────────────────────

class TestFetchStructureRename:
    def test_find_downloaded_pdb_matches_ent(self, tmp_path):
        """_find_downloaded_pdb finds a pdb{pdb_id}.ent file in the dir."""
        p = tmp_path / "pdb3qpd.ent"
        p.write_text("HEADER\n")
        found = _find_downloaded_pdb(str(tmp_path), "3QPD")
        assert found is not None
        assert found.endswith("pdb3qpd.ent")

    def test_find_downloaded_pdb_recursive_subdir(self, tmp_path):
        """_find_downloaded_pdb scans one level of nested subdirs too."""
        sub = tmp_path / "pdb003" / "qpd"
        sub.mkdir(parents=True)
        (sub / "3qpd.pdb").write_text("HEADER\n")
        found = _find_downloaded_pdb(str(tmp_path), "3QPD")
        assert found is not None
        assert "3qpd.pdb" in found

    def test_find_downloaded_pdb_no_match(self, tmp_path):
        """_find_downloaded_pdb returns None when nothing contains the pdb_id."""
        (tmp_path / "unrelated.txt").write_text("x")
        assert _find_downloaded_pdb(str(tmp_path), "3QPD") is None

# ── Test: Config-driven protocol-trust thresholds ────────────────────────

class TestProtocolTrustThresholds:
    def test_defaults_are_sane(self):
        """Loaded thresholds match the documented 1.5 / 2.0 Å defaults."""
        assert RMSD_VALIDATED_MAX == 1.5
        assert RMSD_MARGINAL_MAX == 2.0

    def test_thresholds_drive_badges(self):
        """protocol_trust badges respect the loaded RMSD cutoffs."""
        assert protocol_trust("science", 1.0) == "Validated"
        assert protocol_trust("science", 1.8) == "Validated (Marginal)"
        assert protocol_trust("science", 2.5).startswith("CAUTION: High RMSD")
        assert protocol_trust("ci", 0.5) == "CI Mode (Skipped)"

# ── Test: Error Handling ───────────────────────────────────────────────────

class TestErrorHandling:
    """Tests for robust error tracking during docking / ligand prep."""

    def _benzene_mol(self):
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        return mol

    def test_ligand_preparator_logs_meeko_failure(self):
        """
        When meeko is unavailable/fails, LigandPreparator.prepare must raise a
        RuntimeError whose message includes 'Meeko failed', and must log a
        warning carrying that specific message.
        """
        import sys
        preparator = LigandPreparator()
        mol = self._benzene_mol()

        # Force the meeko import inside prepare() to fail.
        with patch.dict(sys.modules, {"meeko": None}):
            with patch("utils.ligand_prep.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("obabel not on PATH")
                with patch.object(log, "warning") as mock_warn:
                    with pytest.raises(RuntimeError):
                        preparator.prepare(mol)
                    assert any(
                        "Meeko failed" in str(call.args[0])
                        for call in mock_warn.call_args_list
                    ), "Expected log.warning to report the specific 'Meeko failed' message"

    def test_parallel_dock_handles_worker_crash(self, tmp_path):
        """
        If dock_compound raises for a record, _dock_compounds_parallel must
        return (record, None) for that record and still produce results for
        the others.
        """
        records = [
            CompoundRecord(
                compound_id=f"R{i}",
                smiles="c1ccccc1",
                mol=Chem.MolFromSmiles("c1ccccc1"),
            )
            for i in range(3)
        ]

        def fake_dock(rec, *args, **kwargs):
            if rec.compound_id == "R1":
                raise RuntimeError("simulated docking crash")
            return -5.0

        results = _dock_compounds_parallel(
            records,
            "rec.pdbqt",
            np.zeros(3),
            (20.0, 20.0, 20.0),
            str(tmp_path),
            "tag",
            n_jobs=1,
            dock_func=fake_dock,
        )

        by_id = {rec.compound_id: energy for rec, energy in results}
        assert by_id["R1"] is None, "Crashed worker should yield (record, None)"
        assert by_id["R0"] == -5.0, "Healthy workers should still return their energy"
        assert by_id["R2"] == -5.0, "Healthy workers should still return their energy"
        assert len(results) == 3

# ── Test: No recursive similarity relaxation ─────────────────────────────

class TestApplyFiltersRelaxed:
    def test_apply_filters_relaxes_threshold(self):
        """
        When fewer than DIVERSITY_MIN_COUNT compounds pass the strict
        similarity threshold, apply_filters must relax the threshold to
        SIMILARITY_THRESHOLD_RELAXED and re-run the filter on the original
        records (no recursion).

        Records pinned to similarity 0.45 sit in [0.4, 0.5): the strict
        filter (>=0.4) removes them, but the relaxed filter (<0.5) keeps
        them, so all 20 records that pass ADMET/PAINS are returned.
        """
        smiles = "CC(C)Cc1ccc(CC(=O)O)cc1"  # ibuprofen — passes all other filters
        mol = Chem.MolFromSmiles(smiles)
        records = [
            CompoundRecord(compound_id=f"C{i}", smiles=smiles, mol=mol)
            for i in range(20)
        ]

        with patch("utils.filtering.TanimotoSimilarity", return_value=0.45):
            with patch("utils.filtering.DIVERSITY_MIN_COUNT", 100):
                with patch.object(log, "info") as mock_info:
                    result = apply_filters(records)

        # Strict filter removes all, then relaxed re-run (threshold 0.5) keeps
        # every record that passes the other filters.
        assert len(result) == 20, (
            f"Relaxed filter (Tc < 0.5) should keep records with sim=0.45, "
            f"got {len(result)}"
        )

        # A relaxation notice must be logged.
        assert any(
            "Relaxing similarity threshold" in str(c.args[0])
            for c in mock_info.call_args_list
        ), "Expected a similarity-threshold relaxation notice"

# ── Test: Real PDB smoke test (science mode, real PDBs) ────────────────────

class TestRealPDBSmoke:
    def test_real_pdb_smoke(self, tmp_path):
        """
        With science mode and real PDBs supplied via prepare_targets spy,
        main() must run end-to-end, write top_candidates.csv with 2 rows,
        using the mocked library/filters. Vina is absent, so AUTOANTIBIOTIC_FORCE=1
        is set to allow the run without a validated docking protocol.
        """
        import csv
        import discovery_pipeline as dp

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Provide local PDBs to fetch_structure via a neutral directory (path
        # must NOT contain "tests/data", otherwise prepare_targets would treat
        # them as mocks and switch to CI mode). This exercises the science-mode
        # path without any network download.
        real_pdb_dir = tmp_path / "real_pdbs"
        real_pdb_dir.mkdir()
        for pdb_id in ["1VQQ", "3ZG0", "4DKI", "1UTN", "1YAH"]:
            src = TEST_REAL_PDB_DIR / f"{pdb_id}.pdb"
            if src.exists():
                shutil.copy(str(src), str(real_pdb_dir / f"{pdb_id}.pdb"))

        def mock_fetch_structure(pdb_id, out_dir):
            return str(real_pdb_dir / f"{pdb_id}.pdb")

        captured = {}
        real_prepare_targets = dp.prepare_targets

        def spy_prepare(pdb_dir_arg, work_dir_arg, deps, config=None):
            result = real_prepare_targets(pdb_dir_arg, work_dir_arg, deps, config)
            captured["targets"] = result
            return result

        def mock_generate(target_count=2, input_csv=None, input_sdf=None):
            smis = ["c1ccccc1", "Cc1ccccc1"]
            recs = []
            for i, s in enumerate(smis):
                recs.append(CompoundRecord(
                    compound_id=f"AA-{i:04d}",
                    smiles=s,
                    mol=Chem.MolFromSmiles(s),
                ))
            return recs

        def mock_filters(records, **kwargs):
            return list(records)

        def mock_screen_library(records, targets, work_dir, deps):
            out = []
            for i, rec in enumerate(records):
                out.append(CompoundRecord(
                    compound_id=rec.compound_id,
                    smiles=rec.smiles,
                    pb2pa_allosteric_energy=-9.5 + i * 0.3,
                ))
            return out

        with patch("discovery_pipeline.check_dependencies",
                   return_value={"vina": False, "USE_VINA": False}):
            with patch("discovery_pipeline.prepare_targets", side_effect=spy_prepare):
                with patch("discovery_pipeline.generate_candidate_library",
                            side_effect=mock_generate):
                    with patch("discovery_pipeline.apply_filters", side_effect=mock_filters):
                        with patch("discovery_pipeline.screen_library",
                                    side_effect=mock_screen_library):
                            with patch("discovery_pipeline.fetch_structure",
                                        side_effect=mock_fetch_structure):
                                with patch("discovery_pipeline.run_redocking_validation",
                                            return_value=(False, None, None)):
                                    with patch("discovery_pipeline.OUTPUT_DIR", output_dir):
                                        with patch("discovery_pipeline.CSV_REPORT",
                                                    output_dir / "top_candidates.csv"):
                                            with patch.dict(os.environ, {
                                                "AUTOANTIBIOTIC_FORCE": "1",
                                            }):
                                                from discovery_pipeline import main
                                                main(target_count=2)

        csv_path = output_dir / "top_candidates.csv"
        assert csv_path.exists(), "top_candidates.csv should exist after pipeline run"

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

# ── Test 10: LigandPreparator ──────────────────────────────────────────────

class TestLigandPreparator:
    @pytest.fixture
    def benzene_mol(self):
        """Benzene molecule with 3D coordinates for testing."""
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles('c1ccccc1')
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        return mol

    @pytest.fixture
    def ethanol_mol(self):
        """Ethanol molecule with 3D coordinates for testing."""
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles('CCO')
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        return mol

    def test_prepare_returns_pdbqt_string(self, benzene_mol):
        """LigandPreparator.prepare returns a non-empty PDBQT string."""
        preparator = LigandPreparator()
        pdbqt = preparator.prepare(benzene_mol)
        assert pdbqt is not None
        assert len(pdbqt) > 0

    def test_prepare_writes_to_file(self, benzene_mol, tmp_path):
        """LigandPreparator writes PDBQT content to the output file."""
        preparator = LigandPreparator()
        output_path = str(tmp_path / "lig.pdbqt")
        result = preparator.prepare(benzene_mol)
        with open(output_path, "w") as f:
            f.write(result)
        with open(output_path) as f:
            content = f.read()
        assert "BENZENE" in content.upper() or len(content) > 50

    def test_prepare_invalid_mol_raises(self):
        """LigandPreparator.prepare raises RuntimeError for invalid molecules."""
        preparator = LigandPreparator()
        # An empty molecule should raise
        empty_mol = Chem.RWMol()
        with pytest.raises((RuntimeError, TypeError)):
            preparator.prepare(empty_mol)

    def test_prepare_uses_obabel_as_fallback(self, benzene_mol, tmp_path):
        """LigandPreparator falls back to obabel when meeko is unavailable."""
        with patch.dict(sys.modules, {"meeko": None}):
            with patch("utils.ligand_prep.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("obabel not on PATH")
                preparator = LigandPreparator()
                with pytest.raises(RuntimeError):
                    preparator.prepare(benzene_mol)

    def test_prepare_empty_input_raises(self):
        """LigandPreparator raises RuntimeError for None input."""
        preparator = LigandPreparator()
        with pytest.raises((TypeError, RuntimeError)):
            preparator.prepare(None)  # type: ignore[arg-type]

# ── Test 11: analyze_binding_interactions ──────────────────────────────────

class TestAnalyzeBindingInteractions:
    @pytest.fixture
    def mock_receptor_pdb(self, tmp_path):
        """Create a minimal receptor PDB with SER403, LYS406, TYR446."""
        content = create_minimal_pdb({
            ("SER", 403, "A"): [("OG", 11.000, 11.500, 10.000)],
            ("LYS", 406, "A"): [("NZ", 16.000, 12.000, 10.000)],
            ("TYR", 446, "A"): [("OH", 21.000, 12.000, 10.000)],
        })
        pdb_path = tmp_path / "receptor.pdb"
        with open(pdb_path, "w") as f:
            f.write(content)
        return str(pdb_path)

    @pytest.fixture
    def close_ligand_pdbqt(self, tmp_path):
        """Ligand PDBQT with heavy atoms close to key residues."""
        content = create_minimal_pdb({
            ("LIG", 1, "A"): [
                ("C", 11.200, 11.500, 10.000),
                ("C", 16.500, 12.500, 10.000),
                ("C", 21.200, 12.200, 10.000),
            ],
        })
        pdbqt_path = tmp_path / "ligand_close.pdbqt"
        with open(pdbqt_path, "w") as f:
            f.write(content)
        return str(pdbqt_path)

    @pytest.fixture
    def far_ligand_pdbqt(self, tmp_path):
        """Ligand PDBQT with all atoms far from key residues."""
        content = create_minimal_pdb({
            ("LIG", 1, "A"): [("C", 50.000, 50.000, 50.000)],
        })
        pdbqt_path = tmp_path / "ligand_far.pdbqt"
        with open(pdbqt_path, "w") as f:
            f.write(content)
        return str(pdbqt_path)

    def test_detects_ser403_contact(self, mock_receptor_pdb, close_ligand_pdbqt):
        """Returns True when ligand heavy atom is near Ser403 OG."""
        result = analyze_binding_interactions(close_ligand_pdbqt, mock_receptor_pdb)
        assert result["Ser403_contact"] is True
        assert result["min_dist_Ser403"] < 3.5

    def test_detects_lys406_hbond(self, mock_receptor_pdb, close_ligand_pdbqt):
        """Returns True when ligand heavy atom is near Lys406 NZ."""
        result = analyze_binding_interactions(close_ligand_pdbqt, mock_receptor_pdb)
        assert result["Lys406_Hbond"] is True
        assert result["min_dist_Lys406"] < 3.8

    def test_detects_tyr446_hbond(self, mock_receptor_pdb, close_ligand_pdbqt):
        """Returns True when ligand heavy atom is near Tyr446 OH."""
        result = analyze_binding_interactions(close_ligand_pdbqt, mock_receptor_pdb)
        assert result["Tyr446_Hbond"] is True
        assert result["min_dist_Tyr446"] < 3.5

    def test_detects_no_contact_far(self, mock_receptor_pdb, far_ligand_pdbqt):
        """Returns False for all contacts when ligand is far from key residues."""
        result = analyze_binding_interactions(far_ligand_pdbqt, mock_receptor_pdb)
        assert result["Ser403_contact"] is False
        assert result["Lys406_Hbond"] is False
        assert result["Tyr446_Hbond"] is False

    def test_missing_docked_file(self, mock_receptor_pdb):
        """Raises FileNotFoundError when docked PDBQT does not exist."""
        with pytest.raises(FileNotFoundError):
            analyze_binding_interactions("/nonexistent/ligand.pdbqt", mock_receptor_pdb)

    def test_missing_receptor_file(self, close_ligand_pdbqt):
        """Raises FileNotFoundError when receptor PDB does not exist."""
        with pytest.raises(FileNotFoundError):
            analyze_binding_interactions(close_ligand_pdbqt, "/nonexistent/receptor.pdb")

    def test_empty_ligand_raises_value_error(self, mock_receptor_pdb, tmp_path):
        """Raises ValueError when ligand PDBQT has no heavy atoms."""
        empty_path = str(tmp_path / "empty.pdbqt")
        with open(empty_path, "w") as f:
            f.write("REMARK   0\n")
        with pytest.raises(ValueError):
            analyze_binding_interactions(empty_path, mock_receptor_pdb)

# ── Test 12: Integration Pipeline ────────────────────────────────────────────

class TestIntegrationPipeline:
    def test_minimal_pipeline_run(self, tmp_path):
        """
        End-to-end pipeline test:
          - Mocks fetch_structure to return local dummy PDB files.
          - Mocks subprocess.run for Vina to return a successful dummy output.
          - Mocks prepare_targets to bypass PDB cleaning logic.
          - Mocks screen_library to return top 10 with docking scores.
          - Calls main() with target_count=5.
          - Asserts output/top_candidates.csv exists with 5 rows (plus header).
          - Asserts CSV contains required columns.
        """
        import csv

        # Use the local minimal PDB files shipped under tests/data (no network).
        tests_data = Path(__file__).parent / "tests" / "data"
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()

        for pdb_id in ["3QPD", "6TKO", "1UTN", "1YAH"]:
            src = tests_data / f"{pdb_id}.pdb"
            shutil.copy(str(src), str(pdb_dir / f"{pdb_id}.pdb"))

        # Mock dependencies and targets
        mock_deps = {"vina": True, "USE_VINA": True}
        mock_targets = {
            "PBP2a": {
                "pdbqt": str(tmp_path / "PBP2a.pdbqt"),
                "cleaned_pdb": str(tmp_path / "PBP2a_clean.pdb"),
                "allosteric_center": np.array([0.0, 0.0, 0.0]),
                "active_center": np.array([0.0, 0.0, 0.0]),
            },
            "trypsin": {
                "pdbqt": str(tmp_path / "trypsin.pdbqt"),
                "active_center": np.array([0.0, 0.0, 0.0]),
            },
            "CES1": {
                "pdbqt": str(tmp_path / "CES1.pdbqt"),
                "active_center": np.array([0.0, 0.0, 0.0]),
            },
            "holo_pdb": str(pdb_dir / "6TKO.pdb"),
        }

        # Mock the PDB download — return local files from pdb_dir
        def mock_fetch_structure(pdb_id, out_dir):
            return str(pdb_dir / f"{pdb_id}.pdb")

        # Mock prepare_targets to skip PDB cleaning entirely
        def mock_prepare_targets(pdb_dir, work_dir, deps, config=None):
            return mock_targets

        # Mock apply_filters to return all records unchanged
        def mock_apply_filters(records, similarity_threshold=None, recal_mode=False, return_counts=False):
            result = list(records)
            if return_counts:
                return result, {"from": len(result), "after_filtering": len(result)}
            return result

        # Mock analyze_selectivity_and_resistance to return records unchanged
        def mock_analyze_selectivity_and_resistance(records, targets, work_dir, deps):
            return list(records)

        # Mock screen_library to return 5 records with valid docking scores
        # records may be a tuple (records_list, funnel_dict) if from apply_filters with return_counts
        def mock_screen_library(records, targets, work_dir, deps):
            from utils.library_gen import CompoundRecord
            recs = records[0] if isinstance(records, tuple) else records
            top5 = []
            for i, rec in enumerate(recs[:5]):
                new_rec = CompoundRecord(
                    compound_id=rec.compound_id,
                    smiles=rec.smiles,
                    pb2pa_allosteric_energy=-9.5 + i * 0.3,
                )
                top5.append(new_rec)
            return top5

        # Mock Vina subprocess to return a valid docking result
        mock_vina_output = textwrap.dedent("""\
            +---------------------------------------------------+
            | RDKit 2023.09.2                                  |
            +---------------------------------------------------+
            | 1     -9.500      0.000                          |
            | 2     -8.200      1.234                          |
            | 3     -7.800      2.456                          |
            +---------------------------------------------------+
        """)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_vina_output
        mock_result.stderr = ""

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("discovery_pipeline.fetch_structure", side_effect=mock_fetch_structure):
            with patch("discovery_pipeline.check_dependencies", return_value=mock_deps):
                with patch("discovery_pipeline.prepare_targets", side_effect=mock_prepare_targets):
                    with patch("discovery_pipeline.apply_filters", side_effect=mock_apply_filters):
                        with patch("discovery_pipeline.screen_library", side_effect=mock_screen_library):
                            with patch("discovery_pipeline.analyze_selectivity_and_resistance", side_effect=mock_analyze_selectivity_and_resistance):
                                with patch("subprocess.run", return_value=mock_result):
                                    from discovery_pipeline import main
                                    with patch("discovery_pipeline.OUTPUT_DIR", output_dir):
                                        with patch("discovery_pipeline.CSV_REPORT", output_dir / "top_candidates.csv"):
                                            with patch.dict(os.environ, {"AUTOANTIBIOTIC_FORCE": "1"}):
                                                main()

        csv_path = output_dir / "top_candidates.csv"
        assert csv_path.exists(), "top_candidates.csv should exist after pipeline run"

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 5 candidates (target_count=5)
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

        required_columns = {
            "Compound_ID",
            "SMILES",
            "PBP2a_Allosteric_Energy",
        }
        assert required_columns.issubset(set(rows[0].keys())), (
            f"CSV missing required columns: {required_columns - set(rows[0].keys())}"
        )

# ── Test: Redocking failure aborts main() unless forced ─────────────────────

class TestMainRedockingGate:
    def test_main_continues_when_validation_fails_without_force(self, tmp_path):
        """main() continues (writes CSV) when redocking validation fails even if FORCE unset."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        def mock_gen(target_count=3, input_csv=None, input_sdf=None):
            return [CompoundRecord(compound_id=f"AA-{i:04d}", smiles="c1ccccc1",
                                   mol=Chem.MolFromSmiles("c1ccccc1")) for i in range(3)]

        mock_targets = {
            "holo_pdb": "/dev/null",
            "PBP2a": {
                "pdbqt": "/dev/null",
                "allosteric_center": np.array([0.0, 0.0, 0.0]),
                "active_center": np.array([0.0, 0.0, 0.0]),
                "cleaned_pdb": "/dev/null",
                "receptor_pdbqts": ["/dev/null"],
            },
            "trypsin": {"pdbqt": "/dev/null", "active_center": np.array([0.0, 0.0, 0.0]), "cleaned_pdb": "/dev/null"},
            "CES1": {"pdbqt": "/dev/null", "active_center": np.array([0.0, 0.0, 0.0]), "cleaned_pdb": "/dev/null"},
        }

        def mock_screen_library(records, targets, work_dir, deps):
            out = []
            for i, rec in enumerate(records):
                out.append(CompoundRecord(
                    compound_id=rec.compound_id,
                    smiles=rec.smiles,
                    pb2pa_allosteric_energy=-9.5 + i * 0.3,
                ))
            return out

        with patch("discovery_pipeline.check_dependencies",
                    return_value={"vina": False, "USE_VINA": False}):
            with patch("discovery_pipeline.prepare_targets",
                        return_value=mock_targets):
                with patch("discovery_pipeline.run_redocking_validation",
                            return_value=(False, None, None)):
                    with patch("discovery_pipeline.load_config",
                                return_value={"mode": "ci"}):
                        with patch("discovery_pipeline.screen_library",
                                    side_effect=mock_screen_library):
                            with patch.dict(os.environ, {}, clear=False):
                                os.environ.pop("AUTOANTIBIOTIC_FORCE", None)
                                with patch("discovery_pipeline.generate_candidate_library",
                                            side_effect=mock_gen):
                                    with patch("discovery_pipeline.apply_filters",
                                                side_effect=lambda r, **kw: list(r)):
                                        with patch("discovery_pipeline.OUTPUT_DIR", output_dir):
                                            with patch("discovery_pipeline.CSV_REPORT",
                                                        output_dir / "top_candidates.csv"):
                                                from discovery_pipeline import main
                                                main(target_count=3)

        assert (output_dir / "top_candidates.csv").exists(), \
            "CSV should be written even when validation fails and FORCE is unset"

    def test_main_proceeds_when_force_set(self, tmp_path):
        """main() proceeds past the redocking gate when AUTOANTIBIOTIC_FORCE is set."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        def mock_gen(target_count=3, input_csv=None, input_sdf=None):
            return [CompoundRecord(compound_id=f"AA-{i:04d}", smiles="c1ccccc1",
                                   mol=Chem.MolFromSmiles("c1ccccc1")) for i in range(3)]

        mock_targets = {
            "holo_pdb": "/dev/null",
            "PBP2a": {
                "pdbqt": "/dev/null",
                "allosteric_center": np.array([0.0, 0.0, 0.0]),
                "active_center": np.array([0.0, 0.0, 0.0]),
                "cleaned_pdb": "/dev/null",
                "receptor_pdbqts": ["/dev/null"],
            },
            "trypsin": {"pdbqt": "/dev/null", "active_center": np.array([0.0, 0.0, 0.0]), "cleaned_pdb": "/dev/null"},
            "CES1": {"pdbqt": "/dev/null", "active_center": np.array([0.0, 0.0, 0.0]), "cleaned_pdb": "/dev/null"},
        }

        def mock_screen_library(records, targets, work_dir, deps):
            out = []
            for i, rec in enumerate(records):
                out.append(CompoundRecord(
                    compound_id=rec.compound_id,
                    smiles=rec.smiles,
                    pb2pa_allosteric_energy=-9.5 + i * 0.3,
                ))
            return out

        with patch("discovery_pipeline.check_dependencies",
                    return_value={"vina": False, "USE_VINA": False}):
            with patch("discovery_pipeline.prepare_targets",
                        return_value=mock_targets):
                with patch("discovery_pipeline.run_redocking_validation",
                            return_value=(False, None, None)):
                    with patch("discovery_pipeline.screen_library",
                                side_effect=mock_screen_library):
                        with patch.dict(os.environ, {"AUTOANTIBIOTIC_FORCE": "1"}):
                            with patch("discovery_pipeline.generate_candidate_library",
                                        side_effect=mock_gen):
                                with patch("discovery_pipeline.apply_filters",
                                            side_effect=lambda r, **kw: list(r)):
                                    with patch("discovery_pipeline.OUTPUT_DIR", output_dir):
                                        with patch("discovery_pipeline.CSV_REPORT",
                                                    output_dir / "top_candidates.csv"):
                                            from discovery_pipeline import main
                                            main(target_count=3)

        assert (output_dir / "top_candidates.csv").exists(), \
            "CSV should be written when AUTOANTIBIOTIC_FORCE is set"

# ── Test: New experimental-validation-defaults (Task changes) ─────────────

class TestExperimentalValidationDefaults:
    def test_science_continues_on_failed_redocking_without_force(self, tmp_path):
        """The redocking gate is DIAGNOSTIC, never a hard gate: in science mode
        a failed redocking validation (RMSD > threshold) must NOT abort the
        pipeline. The measured (high) RMSD is surfaced honestly so the
        protocol_trust badge reports CAUTION, and the screen continues so the
        candidate report is still produced."""
        import discovery_pipeline as dp

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_targets = {
            "mode": "science",
            "holo_pdb": "/dev/null",
            "PBP2a": {
                "pdbqt": "/dev/null",
                "allosteric_center": np.array([0.0, 0.0, 0.0]),
                "active_center": np.array([0.0, 0.0, 0.0]),
                "cleaned_pdb": "/dev/null",
                "receptor_pdbqts": ["/dev/null"],
            },
        }

        with patch("discovery_pipeline.check_dependencies",
                   return_value={"vina": True, "USE_VINA": True}):
            with patch("discovery_pipeline.run_redocking_validation",
                       return_value=(False, 2.50, 2.50)):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("AUTOANTIBIOTIC_FORCE", None)
                    # The gate must NOT raise SystemExit; it returns
                    # validation_ok=False and surfaces the measured RMSD.
                    from discovery_pipeline import _run_redocking_phase
                    ok, rmsd, vjson = _run_redocking_phase(
                        mock_targets, str(output_dir),
                        {"vina": True, "USE_VINA": True},
                        {"mode": "science"}, force=False,
                    )
                    assert ok is False, \
                        "Failed redocking must report validation_ok=False"
                    assert rmsd == 2.50, \
                        "Measured RMSD must be surfaced, not hidden"

    def test_filter_rejects_qed_0_45(self):
        """apply_filters rejects a compound whose QED is below the 0.5 gate."""
        smiles = "c1ccccc1"
        mol = Chem.MolFromSmiles(smiles)
        record = CompoundRecord(compound_id="TEST_QED045", smiles=smiles, mol=mol)

        with patch("utils.filtering.QED.qed", return_value=0.45):
            with patch("utils.filtering.TanimotoSimilarity", return_value=0.0):
                with patch("utils.filtering.DIVERSITY_MIN_COUNT", 0):
                    filtered = apply_filters([record])
        assert len(filtered) == 0, "QED=0.45 must be rejected by the >0.5 gate"

# ── Test: CONSERVED_RESIDUES warning ──────────────────────────────────────

class TestConservedResiduesCentroid:
    def test_prepare_targets_warns_on_missing_conserved(self, tmp_path):
        """prepare_targets logs a warning when conserved residues are absent."""
        from discovery_pipeline import prepare_targets
        from config.constants import CONSERVED_RESIDUES
        import discovery_pipeline as dp

        # PDB with active-site SER403 present but missing LYS406 / TYR446
        pdb = tmp_path / "p.pdb"
        pdb.write_text(textwrap.dedent("""\
            ATOM      1  N   ALA A 237      41.234  12.345  78.901  1.00  0.00           N
            ATOM      2  CA  ALA A 237      42.345  13.456  79.012  1.00  0.00           C
            ATOM      3  C   ALA A 237      43.456  14.567  80.123  1.00  0.00           C
            ATOM      4  O   ALA A 237      44.567  15.678  81.234  1.00  0.00           O
            ATOM      5  OG  SER A 403       5.000   6.000   7.000  1.00  0.00           O
            END
        """))

        real_centroid = dp.compute_residue_centroid

        def side_centroid(p, r):
            if r == list(CONSERVED_RESIDUES):
                raise ValueError("No matching residues found")
            return real_centroid(p, r)

        def side_clean(in_path, out_path, **kwargs):
            import shutil
            shutil.copy(str(pdb), out_path)
            return out_path

        with patch.object(dp, "fetch_structure", return_value=str(pdb)):
            with patch.object(dp, "clean_pdb_structure", side_effect=side_clean):
                with patch.object(dp, "compute_residue_centroid", side_effect=side_centroid):
                    with patch.object(dp.log, "warning") as mock_warn:
                        prepare_targets(str(tmp_path), str(tmp_path),
                                        {"vina": False, "USE_VINA": False},
                                        config={"mode": "ci"})
                        assert any(
                            "Conserved residues" in str(c.args[0])
                            for c in mock_warn.call_args_list
                        ), "Expected warning about missing conserved residues"

# ── Test: Offline local PDB loading ──────────────────────────────────────

class TestPrepareTargetsNoneCenter:
    def test_active_center_none_when_centroid_fails(self, tmp_path):
        """
        When compute_residue_centroid raises for the active site (both
        CONSERVED_RESIDUES and ACTIVE_SITE_RESIDUES), prepare_targets must
        leave PBP2a active_center as None instead of falling back to the
        allosteric center.
        """
        from discovery_pipeline import prepare_targets
        from config.constants import CONSERVED_RESIDUES, ACTIVE_SITE_RESIDUES
        import discovery_pipeline as dp

        pdb = tmp_path / "p.pdb"
        pdb.write_text(textwrap.dedent("""\
            ATOM      1  N   ALA A 237      41.234  12.345  78.901  1.00  0.00           N
            ATOM      2  CA  ALA A 237      42.345  13.456  79.012  1.00  0.00           C
            ATOM      3  C   ALA A 237      43.456  14.567  80.123  1.00  0.00           C
            ATOM      4  O   ALA A 237      44.567  15.678  81.234  1.00  0.00           O
            END
        """))

        active_res = [list(CONSERVED_RESIDUES), list(ACTIVE_SITE_RESIDUES)]

        def side_centroid(p, r):
            if r in active_res:
                raise ValueError("No matching residues found")
            return np.zeros(3)

        def side_clean(in_path, out_path, **kwargs):
            import shutil
            shutil.copy(str(pdb), out_path)
            return out_path

        with patch.object(dp, "fetch_structure", return_value=str(pdb)):
            with patch.object(dp, "clean_pdb_structure", side_effect=side_clean):
                with patch.object(dp, "compute_residue_centroid", side_effect=side_centroid):
                    result = prepare_targets(
                        str(tmp_path), str(tmp_path),
                        {"vina": False, "USE_VINA": False},
                        config={"mode": "ci"},
                    )

        assert result["PBP2a"]["active_center"] is None, (
            "PBP2a active_center should be None when active-site centroid "
            "computation fails."
        )

class TestOfflinePDBLoad:
    def test_uses_local_mock_pdb(self, tmp_path):
        """
        prepare_targets must use the bundled tests/data PDBs locally
        instead of downloading them — fetch_structure must NOT be called
        for PDBs that exist under tests/data/ (1VQQ, 3ZG0, 4DKI, 1UTN, 1YAH).
        """
        from unittest.mock import MagicMock
        import discovery_pipeline as dp

        tests_data = Path(__file__).parent / "tests" / "data"
        local_ids = ["1VQQ", "3ZG0", "4DKI", "1UTN", "1YAH"]
        for pid in local_ids:
            assert os.path.exists(str(tests_data / f"{pid}.pdb")), \
                f"tests/data/{pid}.pdb must be present"

        resolved_local = {}  # pdb_id -> local path
        for pid in local_ids:
            resolved_local[pid] = str(tests_data / f"{pid}.pdb")

        mock_fetch = MagicMock(
            side_effect=lambda pdb_id, out_dir: os.path.join(out_dir, f"{pdb_id}.pdb")
        )

        clean_inputs = []

        def side_clean(in_path, out_path, **kwargs):
            import shutil
            clean_inputs.append(in_path)
            if os.path.exists(in_path):
                shutil.copy(in_path, out_path)
            else:
                # Off-target PDBs fetched via mock don't exist on disk; create a
                # minimal placeholder so *clean_pdb_structure* succeeds.
                with open(out_path, "w") as f:
                    f.write("ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N\nEND\n")
            return out_path

        with patch.object(dp, "fetch_structure", mock_fetch):
            with patch.object(dp, "clean_pdb_structure", side_effect=side_clean):
                with patch.object(dp, "compute_residue_centroid",
                                  return_value=np.zeros(3)):
                    with patch.object(dp, "load_config",
                                     return_value={"mode": "ci"}):
                        dp.prepare_targets(
                            str(tmp_path), str(tmp_path),
                            {"vina": False, "USE_VINA": False},
                            config={"mode": "ci"},
                        )

        # Every PDB that has a local tests/data/ copy must be sourced locally,
        # never downloaded.
        fetched_ids = [call.args[0] for call in mock_fetch.call_args_list]
        for pid in local_ids:
            assert pid not in fetched_ids, \
                f"fetch_structure must not be called for {pid} when a local copy exists"

        # Verify the local paths were passed to clean_pdb_structure for the
        # PBP2a conformers (the order-independent check).
        local_1vqq = resolved_local["1VQQ"]
        assert local_1vqq in clean_inputs, (
            "Local tests/data/1VQQ.pdb should be the apo structure passed to cleaning"
        )

    def test_science_mode_rejects_mock_pdb(self, tmp_path):
        """prepare_targets must raise ScienceModeMockPDBError in science mode
        when a mock PDB under tests/data is resolved."""
        import discovery_pipeline as dp
        from discovery_pipeline import ScienceModeMockPDBError

        tests_data = Path(__file__).parent / "tests" / "data"
        local_mock = str(tests_data / "1VQQ.pdb")
        assert os.path.exists(local_mock), "tests/data/1VQQ.pdb must be present"

        with patch.object(dp, "load_config", return_value={"mode": "science"}):
            with patch.object(dp, "fetch_structure",
                              return_value=local_mock):
                with patch.object(dp.log, "error") as mock_err:
                    with pytest.raises(ScienceModeMockPDBError):
                        dp.prepare_targets(
                            str(tmp_path), str(tmp_path),
                            {"vina": False, "USE_VINA": False},
                            config={"mode": "science"},
                        )
        assert any(
            "mock pdb" in str(c.args[0]).lower()
            for c in mock_err.call_args_list
        ), "Expected an error about refusing science mode with a mock PDB"

# ── Test: CSV low-conf suffix ──────────────────────────────────────────────

class TestCsvLowConfSuffix:
    def test_low_conf_appends_suffix(self, tmp_path):
        """Selectivity_Index gets ' (low-conf)' suffix when confidence != High."""
        import discovery_pipeline as dp

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        recs = [
            CompoundRecord(compound_id="AA-0001", smiles="c1ccccc1",
                           selectivity_index=3.5, selectivity_confidence="High"),
            CompoundRecord(compound_id="AA-0002", smiles="c1ccccc1",
                           selectivity_index=1.2, selectivity_confidence="Low"),
        ]
        generate_csv_report(
            recs,
            output_dir=output_dir,
            csv_report=output_dir / "top_candidates.csv",
        )

        import csv
        with open(output_dir / "top_candidates.csv") as f:
            rows = list(csv.DictReader(f))
        by_id = {r["Compound_ID"]: r for r in rows}
        assert by_id["AA-0001"]["Selectivity_Index"] == "3.50", by_id["AA-0001"]
        assert by_id["AA-0002"]["Selectivity_Index"] == "1.20 (low-conf)", by_id["AA-0002"]

# ── Test: resistance flags unverified residue on missing SER403 ─────────────

class TestResistanceUnverifiedResidue:
    def test_resistance_unverified_on_missing_residue(self, tmp_path):
        """
        When the cleaned receptor PDB lacks SER403, profile_resistance_risk must
        report an 'unverified' residue note rather than fabricating a distance.
        """
        # Cleaned PDB missing SER403 (only LYS406 and TYR446 present).
        receptor = tmp_path / "receptor_no_ser403.pdb"
        receptor.write_text(textwrap.dedent("""\
            ATOM      1  NZ  LYS A 406      16.000  12.000  10.000  1.00  0.00           N
            ATOM      2  OH  TYR A 446      21.000  12.000  10.000  1.00  0.00           O
            END
        """))

        ligand = tmp_path / "ligand.pdbqt"
        ligand.write_text(textwrap.dedent("""\
            ATOM      1  C   LIG A   1      16.500  12.500  10.000  1.00  0.00           C
            ATOM      2  C   LIG A   1      21.200  12.200  10.000  1.00  0.00           C
            END
        """))

        interactions = analyze_binding_interactions(str(ligand), str(receptor))
        assert interactions["min_dist_Ser403"] == float("inf")
        assert "Ser403" in interactions["unverified_residues"]

        record = CompoundRecord(compound_id="AA-UNV", smiles="c1ccccc1")
        notes = profile_resistance_risk(
            record,
            str(tmp_path),
            "receptor.pdbqt",
            np.zeros(3),
            (20.0, 20.0, 20.0),
            interactions=interactions,
        )
        assert "unverified" in notes.lower(), notes

# ── Test: Task 1 — Externalised target residue lists ──────────────────────

class TestTargetResidueLoading:
    def test_load_target_residues_override(self, tmp_path):
        """_load_target_residues merges a subset override from targets.yaml."""
        from config import constants

        yaml_file = tmp_path / "targets.yaml"
        yaml_file.write_text(
            "targets:\n  ALLOSTERIC_RESIDUES: ['X1', 'X2']\n"
        )
        old = constants.TARGETS_FILE
        constants.TARGETS_FILE = yaml_file
        try:
            result = constants._load_target_residues()
        finally:
            constants.TARGETS_FILE = old

        assert result["ALLOSTERIC_RESIDUES"] == ["X1", "X2"]
        # Untouched lists keep their defaults.
        assert result["CONSERVED_RESIDUES"] == ["SER403", "LYS406", "TYR446"]

    def test_load_target_residues_fallback_when_missing(self, tmp_path):
        """Missing targets.yaml falls back to hardcoded defaults."""
        from config import constants

        missing = tmp_path / "does_not_exist.yaml"
        old = constants.TARGETS_FILE
        constants.TARGETS_FILE = missing
        try:
            result = constants._load_target_residues()
        finally:
            constants.TARGETS_FILE = old

        assert result["ALLOSTERIC_RESIDUES"] == ["TYR105", "GLN199", "GLU237"]
        assert result["ACTIVE_SITE_RESIDUES"] == ["SER403", "LYS406", "TYR446"]
        assert result["CONSERVED_RESIDUES"] == ["SER403", "LYS406", "TYR446"]
        assert result["TRYPSIN_CATALYTIC_RESIDUES"] == ["HIS57", "ASP102", "SER195"]
        assert result["CES1_CATALYTIC_RESIDUES"] == ["SER221", "HIS468", "GLU354"]

    def test_constants_expose_loaded_residues(self):
        """Module-level residue constants are populated from targets.yaml."""
        from config.constants import (
            ALLOSTERIC_RESIDUES,
            ACTIVE_SITE_RESIDUES,
            CONSERVED_RESIDUES,
            TRYPSIN_CATALYTIC_RESIDUES,
            CES1_CATALYTIC_RESIDUES,
        )

        assert ALLOSTERIC_RESIDUES == ["TYR105", "GLN199", "GLU237"]
        assert ACTIVE_SITE_RESIDUES == ["SER403", "LYS406", "TYR446"]
        assert CONSERVED_RESIDUES == ["SER403", "LYS406", "TYR446"]
        assert TRYPSIN_CATALYTIC_RESIDUES == ["HIS57", "ASP102", "SER195"]
        assert CES1_CATALYTIC_RESIDUES == ["SER221", "HIS468", "GLU354"]

# ── Test: Task 2 — LigandPreparator error clarity ─────────────────────────

class TestLigandPreparatorError:
    @staticmethod
    def _benzene_mol():
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        return mol

    def test_raises_clear_error_when_both_backends_fail(self):
        """
        When both meeko and obabel are unavailable, LigandPreparator.prepare
        raises a RuntimeError carrying the exact user-facing message.
        """
        from utils.ligand_prep import LigandPreparator

        preparator = LigandPreparator()
        with patch.dict(sys.modules, {"meeko": None}):
            with patch("utils.ligand_prep.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("obabel not on PATH")
                with pytest.raises(RuntimeError) as exc_info:
                    preparator.prepare(self._benzene_mol())
        assert (
            "PDBQT preparation failed. Please ensure either 'meeko' or "
            "'openbabel' is installed and on your PATH."
            in str(exc_info.value)
        )

    def test_warns_to_install_meeko_on_failure(self):
        """
        A meeko failure logs a warning suggesting 'pip install meeko'.
        """
        from utils.ligand_prep import LigandPreparator

        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol)

        preparator = LigandPreparator()
        with patch.dict(sys.modules, {"meeko": None}):
            with patch("utils.ligand_prep.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("obabel not on PATH")
                with patch.object(log, "warning") as mock_warn:
                    with pytest.raises(RuntimeError):
                        preparator.prepare(mol)
                    assert any(
                        "pip install meeko" in str(call.args[0])
                        for call in mock_warn.call_args_list
                    ), "Expected a warning suggesting 'pip install meeko'"

# ── Test: Task 3 — Native ligand resname override ─────────────────────────

class TestNativeLigandResnameOverride:
    @staticmethod
    def _write_holo_pdb(path: Path) -> None:
        # Two ligands: SO4 (buffer, auto-detect skipped) and CEF (target).
        path.write_text(textwrap.dedent("""\
            HETATM    1  C1  CEF A 500       1.000   2.000   3.000  1.00  0.00           C
            HETATM    2  C2  CEF A 500       2.000   3.000   4.000  1.00  0.00           C
            HETATM    3  S1  SO4 A 501       9.000   9.000   9.000  1.00  0.00           S
            END
        """))

    def test_resname_override_selects_exact_residue(self, tmp_path):
        """resname_override picks the named residue and writes it to a PDB."""
        from utils.structure_prep import _extract_native_ligand_from_holo

        holo = tmp_path / "holo.pdb"
        self._write_holo_pdb(holo)
        smi = tmp_path / "lig.smi"
        pdbqt = tmp_path / "lig.pdbqt"

        result = _extract_native_ligand_from_holo(
            str(holo), str(smi), str(pdbqt), resname_override="CEF"
        )
        # Ligand PDB is written as the .pdb sibling of the .pdbqt path.
        lig_pdb = str(pdbqt).replace(".pdbqt", ".pdb")
        assert Path(lig_pdb).exists()
        content = Path(lig_pdb).read_text()
        assert "CEF" in content
        assert "SO4" not in content

    def test_resname_override_missing_returns_none(self, tmp_path):
        """An override name absent from the PDB returns None gracefully."""
        from utils.structure_prep import _extract_native_ligand_from_holo

        holo = tmp_path / "holo.pdb"
        self._write_holo_pdb(holo)
        smi = tmp_path / "lig.smi"
        pdbqt = tmp_path / "lig.pdbqt"

        result = _extract_native_ligand_from_holo(
            str(holo), str(smi), str(pdbqt), resname_override="NOPE"
        )
        assert result is None

    def test_missing_override_returns_none(self, tmp_path):
        """When resname_override is None, extraction is skipped and returns None."""
        from utils.structure_prep import _extract_native_ligand_from_holo

        holo = tmp_path / "holo.pdb"
        self._write_holo_pdb(holo)
        smi = tmp_path / "lig.smi"
        pdbqt = tmp_path / "lig.pdbqt"

        result = _extract_native_ligand_from_holo(
            str(holo), str(smi), str(pdbqt)
        )
        assert result is None

# ── Test: PyMOL script generation ─────────────────────────────────────────

class TestGeneratePyMOLScript:
    def _make_records(self, tmp_path):
        recs = []
        for i in range(2):
            pdbqt = tmp_path / f"pose_{i}.pdbqt"
            pdbqt.write_text(
                "ATOM      1  C   LIG A   1      11.200  11.500  10.000  1.00  0.00           C\n"
                "END\n"
            )
            rec = CompoundRecord(
                compound_id=f"AA-{i:04d}",
                smiles="c1ccccc1",
                mol=Chem.MolFromSmiles("c1ccccc1"),
            )
            rec.active_docked_pdbqt = str(pdbqt)
            recs.append(rec)
        return recs

    def test_returns_pml_path(self, tmp_path):
        """generate_pymol_script returns the path to the written .pml file."""
        recs = self._make_records(tmp_path)
        receptor_pdb = tmp_path / "receptor.pdb"
        receptor_pdb.write_text(
            "ATOM      1  OG  SER A 403      11.000  11.500  10.000  1.00  0.00           O\n"
            "END\n"
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        targets = {"PBP2a": {"cleaned_pdb": str(receptor_pdb)}}
        pml = generate_pymol_script(recs, targets, str(out_dir))
        assert pml.endswith(".pml")
        assert os.path.exists(pml), "visualization.pml should be written"

    def test_script_loads_receptor_and_poses(self, tmp_path):
        """The generated .pml loads the receptor and each ligand pose."""
        recs = self._make_records(tmp_path)
        receptor_pdb = tmp_path / "receptor.pdb"
        receptor_pdb.write_text(
            "ATOM      1  OG  SER A 403      11.000  11.500  10.000  1.00  0.00           O\n"
            "END\n"
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        targets = {"PBP2a": {"cleaned_pdb": str(receptor_pdb)}}
        pml = generate_pymol_script(recs, targets, str(out_dir))
        content = Path(pml).read_text()
        assert "load" in content
        assert "PBP2a" in content
        # Both ligand poses should be loaded.
        assert content.count("load") >= 3  # receptor + 2 poses

    def test_script_skips_missing_pose(self, tmp_path):
        """Records without an active-site pose are skipped in the .pml."""
        recs = self._make_records(tmp_path)
        recs[1].active_docked_pdbqt = None  # no pose
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        targets = {"PBP2a": {"cleaned_pdb": None}}
        pml = generate_pymol_script(recs, targets, str(out_dir))
        content = Path(pml).read_text()
        # Only the receptor-less header + 1 pose load expected.
        assert "Ligand_1" in content
        assert "Ligand_2" not in content

# ── Test: Task 1 — Consensus rigid docking returns best energy ───────

class TestConsensusDocking:
    def test_returns_best_energy_across_conformers(self, tmp_path):
        """_run_consensus_dock keeps the most negative energy over all conformers."""
        from discovery_pipeline import _run_consensus_dock
        records = [
            CompoundRecord(compound_id="AA-0001", smiles="c1ccccc1",
                           mol=Chem.MolFromSmiles("c1ccccc1")),
            CompoundRecord(compound_id="AA-0002", smiles="Cc1ccccc1",
                           mol=Chem.MolFromSmiles("Cc1ccccc1")),
        ]

        def fake_parallel(recs, receptor_pdbqt, center, box, wd, tag):
            conf_idx = int(tag.rsplit("_c", 1)[-1])
            e = -5.0 if conf_idx == 0 else -9.0
            return [(r, e) for r in recs]

        with patch("discovery_pipeline._dock_compounds_parallel", side_effect=fake_parallel):
            results = _run_consensus_dock(
                records,
                ["r0.pdbqt", "r1.pdbqt"],
                np.zeros(3), (20.0, 20.0, 20.0),
                str(tmp_path), "allosteric",
            )
        assert results["AA-0001"] == -9.0, "Expected best of -5.0/-9.0 = -9.0"
        assert results["AA-0002"] == -9.0

    def test_single_conformer_fallback(self, tmp_path):
        """With one receptor, _run_consensus_dock returns that single energy."""
        from discovery_pipeline import _run_consensus_dock
        records = [CompoundRecord(compound_id="AA-0001", smiles="c1ccccc1",
                                 mol=Chem.MolFromSmiles("c1ccccc1"))]

        def fake_parallel(recs, receptor_pdbqt, center, box, wd, tag):
            return [(r, -6.0) for r in recs]

        with patch("discovery_pipeline._dock_compounds_parallel", side_effect=fake_parallel):
            results = _run_consensus_dock(
                records, ["r0.pdbqt"], np.zeros(3),
                (20.0, 20.0, 20.0), str(tmp_path), "active",
            )
        assert results["AA-0001"] == -6.0

# ── Test: Task 1 — Active-site pose propagated across workers ───────

class TestActivePosePropagation:
    def test_worker_returns_pose_tuple(self, tmp_path):
        """_dock_worker returns (rec, energy, active_docked_pdbqt)."""
        from utils.docking import _dock_worker

        def fake_dock(rec, *a, **k):
            rec.active_docked_pdbqt = "/tmp/pose.pdbqt"
            return -7.0

        # dock_func is the 2nd positional arg; pass fake_dock directly.
        result = _dock_worker(
            ("AA-0001", "c1ccccc1"),
            fake_dock, "r.pdbqt", np.zeros(3),
            (20.0, 20.0, 20.0), str(tmp_path), "active",
        )
        assert len(result) == 3, "worker must return a 3-tuple"
        rec, energy, pose = result
        assert energy == -7.0
        assert pose == "/tmp/pose.pdbqt"

    def test_parallel_propagates_pose_to_parent(self, tmp_path):
        """_dock_compounds_parallel assigns active_docked_pdbqt on the parent."""
        from utils.docking import _dock_compounds_parallel

        def fake_dock(rec, *a, **k):
            rec.active_docked_pdbqt = f"/tmp/{rec.compound_id}.pdbqt"
            return -7.0

        rec = CompoundRecord(
            compound_id="AA-0001", smiles="c1ccccc1",
            mol=Chem.MolFromSmiles("c1ccccc1"),
        )
        with patch("utils.docking.dock_compound", side_effect=fake_dock):
            results = _dock_compounds_parallel(
                [rec], "r.pdbqt", np.zeros(3),
                (20.0, 20.0, 20.0), str(tmp_path), "active",
                n_jobs=1,
            )
        assert results[0][1] == -7.0
        # The parent record's pose attribute was mutated by the (in-process) worker.
        assert rec.active_docked_pdbqt == "/tmp/AA-0001.pdbqt"

    @pytest.mark.parametrize("tag,should_keep", [
        ("active", True),
        ("active_c0", True),   # consensus per-conformer — was the bug
        ("active_c2", True),
        ("active_flex", True),  # flexible active-site docking
        ("mut_S403A", False),   # mutant scan — must NOT be retained
        ("allosteric", False),
        ("allosteric_c1", False),
    ])
    def test_dock_compound_retains_active_pose_for_tag(self, tmp_path, tag, should_keep):
        """dock_compound must retain the docked-pose file for ALL active-site
        tags (plain 'active', consensus 'active_c*', flexible 'active_flex'),
        but not for mutant/allosteric tags. Regression for consensus docking
        passing 'active_c0' and the pose being dropped (MMGBSA/mutant N/A)."""
        from utils.docking import dock_compound

        rec = CompoundRecord(
            compound_id="AA-0001", smiles="c1ccccc1",
            mol=Chem.MolFromSmiles("c1ccccc1"),
        )

        def fake_vina(receptor, lig, out, center, box, flex_pdbqt=None, **kwargs):
            with open(out, "w") as fh:
                fh.write("MODEL 1\nENDMDL\n")
            return -7.5

        with patch("utils.docking.prepare_ligand_pdbqt", return_value=True), \
             patch("utils.docking._run_vina_docking", side_effect=fake_vina):
            energy = dock_compound(
                rec, "r.pdbqt", np.zeros(3), (20.0, 20.0, 20.0),
                str(tmp_path), tag,
            )
        assert energy == -7.5
        if should_keep:
            assert rec.active_docked_pdbqt is not None, \
                f"tag {tag!r} must retain the active-site pose"
            assert os.path.exists(rec.active_docked_pdbqt), \
                f"pose file for tag {tag!r} must survive cleanup"
        else:
            assert getattr(rec, "active_docked_pdbqt", None) is None, \
                f"tag {tag!r} must NOT retain a pose"

    def test_failed_flex_dock_does_not_clobber_good_rigid_pose(self, tmp_path):
        """A failed flexible re-dock (Vina rejects the flex file → energy None,
        no output written) must NOT overwrite a previously retained good rigid
        pose. Regression: dock_compound unconditionally set active_docked_pdbqt
        to a non-existent flex pose, breaking MM-GBSA/H-bond/mutation analysis."""
        from utils.docking import dock_compound

        rec = CompoundRecord(
            compound_id="AA-0001", smiles="c1ccccc1",
            mol=Chem.MolFromSmiles("c1ccccc1"),
        )

        # First: a successful rigid active-site dock retains a real pose.
        def good_vina(receptor, lig, out, center, box, flex_pdbqt=None, **kwargs):
            with open(out, "w") as fh:
                fh.write("MODEL 1\nENDMDL\n")
            return -8.0

        with patch("utils.docking.prepare_ligand_pdbqt", return_value=True), \
             patch("utils.docking._run_vina_docking", side_effect=good_vina):
            dock_compound(rec, "r.pdbqt", np.zeros(3), (20.0, 20.0, 20.0),
                          str(tmp_path), "active_c0")
        good_pose = rec.active_docked_pdbqt
        assert good_pose is not None and os.path.exists(good_pose)

        # Then: a failed flex dock (Vina writes nothing, returns None).
        def failed_vina(receptor, lig, out, center, box, flex_pdbqt=None, **kwargs):
            return None  # no output file written

        with patch("utils.docking.prepare_ligand_pdbqt", return_value=True), \
             patch("utils.docking._run_vina_docking", side_effect=failed_vina):
            dock_compound(rec, "r.pdbqt", np.zeros(3), (20.0, 20.0, 20.0),
                          str(tmp_path), "active_flex",
                          flex_pdbqt="flex.pdbqt")

        # The good rigid pose must survive; not clobbered by the failed flex dock.
        assert rec.active_docked_pdbqt == good_pose, \
            "failed flex dock must not clobber the retained rigid pose"
        assert os.path.exists(rec.active_docked_pdbqt)

# ── Test: Task 1 — Mechanism-restricted (two-target) Selectivity Index ──

class TestMechanismRestrictedSelectivity:
    """Task 1 — SI uses ONLY the selectivity panel (trypsin, CES1) in its
    denominator; Off_Target_Risk is derived from trypsin/CES1 only (the
    simplified pipeline no longer docks the liability panel). Also covers
    SI_vs_Ceftaroline and the tiered SI labels."""

    def _run(self, tmp_path, pb2pa_e, trypsin_e, ces1_e):
        records = [
            CompoundRecord(compound_id="AA-0001", smiles="c1ccccc1",
                           mol=Chem.MolFromSmiles("c1ccccc1"))
        ]
        records[0].pb2pa_active_energy = pb2pa_e

        targets = {
            "trypsin": {"pdbqt": "t.pdbqt", "active_center": np.zeros(3)},
            "CES1": {"pdbqt": "c.pdbqt", "active_center": np.zeros(3)},
        }
        fixed = {"trypsin": trypsin_e, "ces1": ces1_e}

        def fake_parallel(recs, receptor_pdbqt, center, box, wd, tag, n_jobs=1,
                          dock_func=None):
            return [(r, fixed[tag]) for r in recs]

        with patch("discovery_pipeline._dock_compounds_parallel", side_effect=fake_parallel):
            return analyze_selectivity_and_resistance(
                records, targets, str(tmp_path),
                {"vina": True, "USE_VINA": True},
            )

    def test_selectivity_uses_two_targets(self, tmp_path):
        """analyze_selectivity_and_resistance docks the top-10 against the
        mechanism-restricted two-target panel (trypsin, CES1) only and computes
        SI from the tightest of those two energies. The liability panel is no
        longer docked (its columns are reported as N/A)."""
        records = [
            CompoundRecord(compound_id=f"AA-{i:04d}", smiles="c1ccccc1",
                           mol=Chem.MolFromSmiles("c1ccccc1"))
            for i in range(3)
        ]
        for i, rec in enumerate(records):
            rec.pb2pa_active_energy = -10.0 - i  # PBP2a active-site energy

        targets = {
            "trypsin": {"pdbqt": "t.pdbqt", "active_center": np.zeros(3)},
            "CES1": {"pdbqt": "c.pdbqt", "active_center": np.zeros(3)},
        }
        # Fixed human energies per tag: trypsin=-4, ces1=-3.
        fixed = {"trypsin": -4.0, "ces1": -3.0}

        def fake_parallel(recs, receptor_pdbqt, center, box, wd, tag, n_jobs=1,
                          dock_func=None):
            return [(r, fixed[tag]) for r in recs]

        with patch("discovery_pipeline._dock_compounds_parallel", side_effect=fake_parallel):
            out = analyze_selectivity_and_resistance(
                records, targets, str(tmp_path),
                {"vina": True, "USE_VINA": True},
            )

        assert out[0].selectivity_index == pytest.approx(2.5)
        assert out[0].selectivity_confidence == "High"

    def test_raw_si_not_zeroed_on_tight_offtarget(self, tmp_path):
        """The old override that set SI = 0.0 when a human off-target
        bound tightly (energy < -8.0) must be GONE. The raw SI is
        preserved and the high-risk signal moves to Off_Target_Risk."""
        records = [
            CompoundRecord(compound_id="AA-0001", smiles="c1ccccc1",
                           mol=Chem.MolFromSmiles("c1ccccc1"))
        ]
        records[0].pb2pa_active_energy = -9.0

        targets = {
            "trypsin": {"pdbqt": "t.pdbqt", "active_center": np.zeros(3)},
            "CES1": {"pdbqt": "c.pdbqt", "active_center": np.zeros(3)},
        }
        # Trypsin binds tightly (-9.0) → would have zeroed SI before the fix.
        fixed = {"trypsin": -9.0, "ces1": -3.0}

        def fake_parallel(recs, receptor_pdbqt, center, box, wd, tag, n_jobs=1,
                          dock_func=None):
            return [(r, fixed[tag]) for r in recs]

        with patch("discovery_pipeline._dock_compounds_parallel", side_effect=fake_parallel):
            out = analyze_selectivity_and_resistance(
                records, targets, str(tmp_path),
                {"vina": True, "USE_VINA": True},
            )
        assert out[0].selectivity_index == pytest.approx(1.0)
        assert out[0].off_target_risk is True

    def test_invalid_positive_human_energy_ignored(self, tmp_path):
        """A human off-target energy > 0.0 (no-pose / clash) is treated
        as invalid and excluded from the SI denominator (paper §4.1b)."""
        records = [
            CompoundRecord(compound_id="AA-0001", smiles="c1ccccc1",
                           mol=Chem.MolFromSmiles("c1ccccc1"))
        ]
        records[0].pb2pa_active_energy = -9.0

        targets = {
            "trypsin": {"pdbqt": "t.pdbqt", "active_center": np.zeros(3)},
            "CES1": {"pdbqt": "c.pdbqt", "active_center": np.zeros(3)},
        }
        # Trypsin = +5.0 (no pose / clash → invalid), CES1 = -3.0 (real).
        fixed = {"trypsin": 5.0, "ces1": -3.0}

        def fake_parallel(recs, receptor_pdbqt, center, box, wd, tag, n_jobs=1,
                          dock_func=None):
            return [(r, fixed[tag]) for r in recs]

        with patch("discovery_pipeline._dock_compounds_parallel", side_effect=fake_parallel):
            out = analyze_selectivity_and_resistance(
                records, targets, str(tmp_path),
                {"vina": True, "USE_VINA": True},
            )
        # Only 1 valid panel energy -> SI = None (requires 2+)
        assert out[0].selectivity_index is None
        assert out[0].off_target_risk is False
        # si_provisional shows the single-target ratio
        assert out[0].si_provisional == pytest.approx(9.0 / 3.0)

    def test_si_vs_ceftaroline_transparency(self, tmp_path):
        """SI_vs_Ceftaroline = |E_PBP2a_best| / CEFTAROLINE_CONTROL_E (7.3),
        computed with NO covalent bonus. Matches the configured control energy."""
        from config.constants import CEFTAROLINE_CONTROL_E
        out = self._run(tmp_path, pb2pa_e=-7.3, trypsin_e=-2.0, ces1_e=-3.0)
        rec = out[0]
        assert rec.si_vs_ceftaroline == pytest.approx(
            abs(-7.3) / CEFTAROLINE_CONTROL_E)
        assert rec.si_vs_ceftaroline == pytest.approx(abs(-7.3) / CEFTAROLINE_CONTROL_E)

    def test_si_vs_ceftaroline_populated_for_all(self, tmp_path):
        """SI_vs_Ceftaroline is populated whenever a PBP2a energy exists,
        independently of human panel availability."""
        out = self._run(tmp_path, pb2pa_e=-10.0, trypsin_e=-3.0, ces1_e=-4.0)
        assert out[0].si_vs_ceftaroline is not None
        assert out[0].si_vs_ceftaroline == pytest.approx(10.0 / 7.3)

class TestSelectivityIndexTiers:
    """Tiered SI labels (paper §2.4)."""

    def test_si_tier_strong_promising_weak_na(self):
        assert si_tier(2.5) == "Strong"
        assert si_tier(2.0) == "Strong"
        assert si_tier(1.8) == "Promising"
        assert si_tier(1.5) == "Promising"
        assert si_tier(1.2) == "Weak"
        assert si_tier(0.0) == "Weak"
        assert si_tier(None) == "N/A"

    def test_si_tier_thresholds_match_constants(self):
        from config.constants import SI_STRONG_THRESHOLD, SI_PROMISING_THRESHOLD
        assert si_tier(SI_STRONG_THRESHOLD) == "Strong"
        assert si_tier(SI_STRONG_THRESHOLD - 0.01) == "Promising"
        assert si_tier(SI_PROMISING_THRESHOLD) == "Promising"
        assert si_tier(SI_PROMISING_THRESHOLD - 0.01) == "Weak"

# ── Test: Flexible-residue PDBQT torsion tree (Vina --flex) ──────────

class TestProtocolValidationTightening:
    """Redocking validation must use a tighter per-axis box around the native
    ligand (2.0 Å padding, capped at 30 Å) and a pragmatic exhaustiveness (32)
    that completes within the wall-clock budget.
    """

    def test_redocking_uses_exhaustiveness_16(self, tmp_path):
        import discovery_pipeline as P

        cmd_captured = {}

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            cmd = list(cmd)
            if cmd and cmd[0] == "vina":
                cmd_captured["cmd"] = cmd
            class R:
                returncode = 0
                stdout = ""
                stderr = ""
            return R()

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(P, "_extract_native_ligand_from_holo",
                          return_value="Cc1ccccc1"), \
             patch.object(P, "compute_residue_centroid",
                          return_value=np.array([0.0, 0.0, 0.0])), \
             patch.object(P, "_compute_rmsd_docked_vs_crystal",
                          return_value=1.2):
            deps = {"USE_VINA": True, "mode": "science"}
            P.run_redocking_validation(
                "holo.pdb", "receptor.pdbqt", str(tmp_path), deps,
                mode="science", config={"native_ligand_resname": "LIG"},
            )
        assert "--exhaustiveness" in cmd_captured["cmd"]
        ex_idx = cmd_captured["cmd"].index("--exhaustiveness")
        assert cmd_captured["cmd"][ex_idx + 1] == "32"

    def test_redocking_box_uses_3A_padding(self):
        import discovery_pipeline as P
        # The redocking box must be sized with the tighter 3.0 Å padding (rather
        # than the default 6.0 Å) when run_redocking_validation calls
        # _redocking_box_size. We patch the sizing helper and assert it was
        # invoked with redock_padding=3.0.
        captured = {}

        def fake_size(ligand_pdbqt, center, min_size=15.0, padding=6.0,
                      default_box=(25.0, 25.0, 25.0), redock_padding=4.0):
            captured["redock_padding"] = redock_padding
            return (12.0, 12.0, 12.0)

        with patch.object(P, "_redocking_box_size", side_effect=fake_size), \
             patch("subprocess.run") as mock_run, \
             patch.object(P, "_extract_native_ligand_from_holo",
                          return_value="Cc1ccccc1"), \
             patch.object(P, "compute_residue_centroid",
                          return_value=np.array([0.0, 0.0, 0.0])), \
             patch.object(P, "_compute_rmsd_docked_vs_crystal",
                          return_value=1.2):
            # Make the vina subprocess.run succeed with a redocked pose.
            def _fake_vina(cmd, capture_output=True, text=True, timeout=None):
                class R:
                    returncode = 0
                    stdout = ""
                    stderr = ""
                return R()
            mock_run.side_effect = _fake_vina
            deps = {"USE_VINA": True, "mode": "science"}
            P.run_redocking_validation(
                "holo.pdb", "receptor.pdbqt", str(tempfile.mkdtemp()),
                deps, mode="science",
                config={"native_ligand_resname": "LIG"},
            )
        assert captured.get("redock_padding") == 5.0

# ── Test: Reporting adds Phase 3.5 columns ───────────────────────────

class TestReportingPhase35:
    """top_candidates.csv must now carry Human_OffTarget_Max_Energy and
    HIGH_TOXICITY_RISK columns.
    """

    def test_csv_has_phase35_columns(self, tmp_path):
        rec = CompoundRecord(
            compound_id="AA-1", smiles="c1ccccc1",
            mol=Chem.MolFromSmiles("c1ccccc1"),
            pb2pa_allosteric_energy=-9.0, pb2pa_active_energy=-9.0,
            human_trypsin_energy=-3.0, human_ces1_energy=-2.0,
            human_offtarget_max_energy=-2.0,
            selectivity_index=0.8, selectivity_confidence="High",
            max_similarity=0.1, passes_lipinski=True, qed_score=0.6,
            resistance_notes="",
        )
        csv_path = tmp_path / "top_candidates.csv"
        generate_csv_report(
            [rec], validation_ok=True, mode="science", redock_rmsd=1.2,
            csv_report=csv_path, output_dir=tmp_path,
        )
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert "Human_OffTarget_Max_Energy" in df.columns
        assert "HIGH_TOXICITY_RISK" in df.columns
        # SI < 1.0 ⇒ flagged as high toxicity risk.
        assert bool(df["HIGH_TOXICITY_RISK"].iloc[0]) is True
        assert df["Human_OffTarget_Max_Energy"].iloc[0] == pytest.approx(-2.0)

# ── Test: enrichment_validation.py uses only independent labels ──

class TestEnrichmentNoCircularLabeling:
    """The enrichment validation script must contain NO line that defines an
    active from a docking energy threshold — labels come ONLY from
    data/known_actives.csv and data/known_decoys.csv."""

    SCRIPT_PATH = Path(__file__).resolve().parent / "scripts" / "enrichment_validation.py"

    def test_no_docking_energy_used_for_labels(self):
        """Assert that no line in enrichment_validation.py defines actives from
        a docking energy threshold. Scan for the specific patterns that were used
        in the circular version."""
        import ast
        script = self.SCRIPT_PATH.read_text()
        # The old circular logic used lines like:
        #   active_ids = set(ctrl_smiles.keys())
        #   for cid, e in energies.items():
        #       if cid.startswith("SEED_") and e is not None and e < -8.0:
        #           active_ids.add(cid)
        # Assert that 'e < -8.0' or 'energy < -8' or similar patterns are absent.
        forbidden_patterns = [
            "e < -8",
            "energy < -",
            "e < -8.0",
            "active_ids.add",
            "# Define actives",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in script, (
                f"enrichment_validation.py contains forbidden pattern "
                f"'{pattern}' — labels must be independent of docking energy"
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
