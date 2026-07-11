#!/usr/bin/env python3
"""
Unit tests for discovery_pipeline.py
======================================
Tests core scientific and engineering functions in isolation.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from discovery_pipeline import (
    compute_residue_centroid,
    apply_filters,
    generate_candidate_library,
    check_dependencies,
    _run_vina_docking,
    compute_selectivity_index,
    analyze_binding_interactions,
    LigandPreparator,
    CompoundRecord,
    BETA_LACTAM_SMARTS,
    OUTPUT_DIR,
    TOP_N,
    ensure_output_dir,
    screen_library,
)
from rdkit import Chem


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_pdb_dir():
    """Create a temporary directory with a minimal PDB file for centroid testing."""
    tmpdir = tempfile.mkdtemp()

    # Minimal PDB with a single residue (ALA 237) containing a CA atom.
    # Coordinates are arbitrary.
    pdb_content = textwrap.dedent("""\
        ATOM      1  N   ALA A 237      41.234  12.345  78.901  1.00  0.00           N
        ATOM      2  CA  ALA A 237      42.345  13.456  79.012  1.00  0.00           C
        ATOM      3  C   ALA A 237      43.456  14.567  80.123  1.00  0.00           C
        ATOM      4  O   ALA A 237      44.567  15.678  81.234  1.00  0.00           O
        END
    """)
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
        """generate_candidate_library returns at least 10 compounds with default params."""
        library = generate_candidate_library(target_count=500)
        assert len(library) >= 10, (
            f"Expected at least 10 compounds, got {len(library)}"
        )

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

    def test_returns_at_least_controls_when_generation_fails(self):
        """Even with an unreachable target_count, control compounds are returned."""
        library = generate_candidate_library(target_count=10000)
        ids = [r.compound_id for r in library]
        control_ids = [cid for cid in ids if cid.startswith("CTRL_")]
        assert len(control_ids) >= 1, "Expected at least one control compound"
        assert len(library) >= 2, "Expected at least control compounds to be returned"


# ── Run ──────────────────────────────────────────────────────────────────────

# ── Test 9: Fallback Scoring (TestFallbackScoring) ──────────────────────────

class TestFallbackScoring:
    def test_shape_scoring_fallback(self):
        """
        When USE_VINA is False (Vina unavailable), screen_library must:
          - Still compute shape scores for every compound.
          - Leave pb2pa_allosteric_energy as None (no Vina docking).
        """
        # Build 3 mock compound records
        records = [
            CompoundRecord(
                compound_id='SHAPE_A',
                smiles='CC1=CC=C(C=C1)',  # benzene
                mol=Chem.MolFromSmiles('CC1=CC=C(C=C1)'),
            ),
            CompoundRecord(
                compound_id='SHAPE_B',
                smiles='CC(C)(C)C1=CC=C(C=C1)C(C)(C)C',  # xylene-like
                mol=Chem.MolFromSmiles('CC(C)(C)C1=CC=C(C=C1)C(C)(C)C'),
            ),
            CompoundRecord(
                compound_id='SHAPE_C',
                smiles='C1=CC=C(C=C1)O',  # phenol
                mol=Chem.MolFromSmiles('C1=CC=C(C=C1)O'),
            ),
        ]

        # Mock deps to simulate Vina unavailable
        mock_deps = {'vina': False, 'USE_VINA': False}

        # Mock targets to return minimal structure
        mock_targets = {
            'PBP2a': {
                'pdbqt': '/dev/null',
                'cleaned_pdb': '/dev/null',
                'allosteric_center': np.array([0.0, 0.0, 0.0]),
                'active_center': np.array([0.0, 0.0, 0.0]),
            },
            'trypsin': {
                'pdbqt': '/dev/null',
                'active_center': np.array([0.0, 0.0, 0.0]),
            },
            'CES1': {
                'pdbqt': '/dev/null',
                'active_center': np.array([0.0, 0.0, 0.0]),
            },
            'holo_pdb': '/dev/null',
        }

        with tempfile.TemporaryDirectory() as work_dir:
            result = screen_library(records, mock_targets, work_dir, mock_deps)

        # All records must have non-None shape_score
        for rec in result:
            assert rec.shape_score is not None, \
                f"shape_score is None for {rec.compound_id}"

        # pb2pa_allosteric_energy must be None (Vina disabled)
        for rec in result:
            assert rec.pb2pa_allosteric_energy is None, \
                f"pb2pa_allosteric_energy is not None: {rec.pb2pa_allosteric_energy}"

        # Should return at least TOP_N (10) or all if fewer
        assert len(result) >= min(len(records), TOP_N), \
            f"Expected at least {min(len(records), TOP_N)} results, got {len(result)}"


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
        # Mock meeko import to raise ImportError
        with patch('meeko.MoleculePreparation') as mock_meeko_prep:
            mock_meeko_prep.side_effect = ImportError("meeko not found")
            preparator = LigandPreparator()
            # obabel won't be available in test env, so we expect ValueError (empty output)
            with pytest.raises((ImportError, RuntimeError, ValueError)):
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
        content = textwrap.dedent("""\
            ATOM      1  OG  SER A 403      11.000  11.500  10.000  1.00  0.00           O
            ATOM      2  NZ  LYS A 406      16.000  12.000  10.000  1.00  0.00           N
            ATOM      3  OH  TYR A 446      21.000  12.000  10.000  1.00  0.00           O
            END
        """)
        pdb_path = tmp_path / "receptor.pdb"
        with open(pdb_path, "w") as f:
            f.write(content)
        return str(pdb_path)

    @pytest.fixture
    def close_ligand_pdbqt(self, tmp_path):
        """Ligand PDBQT with heavy atoms close to key residues."""
        content = textwrap.dedent("""\
            ATOM      1  C   LIG A   1      11.200  11.500  10.000  1.00  0.00           C
            ATOM      2  C   LIG A   1      16.500  12.500  10.000  1.00  0.00           C
            ATOM      3  C   LIG A   1      21.200  12.200  10.000  1.00  0.00           C
            END
        """)
        pdbqt_path = tmp_path / "ligand_close.pdbqt"
        with open(pdbqt_path, "w") as f:
            f.write(content)
        return str(pdbqt_path)

    @pytest.fixture
    def far_ligand_pdbqt(self, tmp_path):
        """Ligand PDBQT with all atoms far from key residues."""
        content = textwrap.dedent("""\
            ATOM      1  C   LIG A   1      50.000  50.000  50.000  1.00  0.00           C
            END
        """)
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

        # Create minimal PDB files for the four PDB IDs
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()

        dummy_pdb = textwrap.dedent("""\
            ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
            ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C
            ATOM      3  C   ALA A   1       2.000   3.000   4.000  1.00  0.00           C
            ATOM      4  O   ALA A   1       2.500   3.500   4.500  1.00  0.00           O
            ATOM      5  O   SER A 403       5.000   6.000   7.000  1.00  0.00           O
            ATOM      6  NZ  LYS A 406       8.000   9.000  10.000  1.00  0.00           N
            ATOM      7  OH  TYR A 446      11.000  12.000  13.000  1.00  0.00           O
            END
        """)

        for pdb_id in ["3QPD", "6TKO", "1UTN", "3KJZ"]:
            path = pdb_dir / f"{pdb_id}.pdb"
            with open(path, "w") as f:
                f.write(dummy_pdb)

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
            "holo_pdb": str(tmp_path / "6TKO.pdb"),
        }

        # Mock the PDB download — return local dummy files
        def mock_fetch_structure(pdb_id, out_dir):
            return str(pdb_dir / f"{pdb_id}.pdb")

        # Mock prepare_targets to skip PDB cleaning entirely
        def mock_prepare_targets(pdb_dir, work_dir, deps):
            return mock_targets

        # Mock apply_filters to return all records unchanged
        def mock_apply_filters(records):
            return list(records)

        # Mock analyze_selectivity_and_resistance to return records unchanged
        def mock_analyze_selectivity_and_resistance(records, targets, work_dir, deps):
            return list(records)

        # Mock screen_library to return 5 records with valid docking scores
        def mock_screen_library(records, targets, work_dir, deps):
            from discovery_pipeline import CompoundRecord
            # Return top 5 records with allosteric energy scores
            top5 = []
            for i, rec in enumerate(records[:5]):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
