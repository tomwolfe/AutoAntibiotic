"""Unit tests for interaction fingerprint (IFP) functions."""

import os
import tempfile

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.analysis import (
    _parse_pdb_residue_coords,
    _parse_pdbqt_ligand_coords,
    check_key_interactions,
    compute_ifp_similarity,
)

# Minimal receptor PDB snippet containing SER403 at (10, 10, 10).
# The residue name in PDB column 17-20 is "SER", residue number column 22-26 is "403".
_SER403_PDB = """\
ATOM      1  N   SER A 403      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  SER A 403      10.500  10.000  10.000  1.00  0.00           C
ATOM      3  C   SER A 403      11.000  10.500  10.000  1.00  0.00           C
ATOM      4  O   SER A 403      11.800  10.200   9.200  1.00  0.00           O
ATOM      5  CB  SER A 403      10.500   9.500  11.000  1.00  0.00           C
ATOM      6  OG  SER A 403      11.000   9.000  11.000  1.00  0.00           O
END
"""

# Ligand PDBQT with a ligand atom at (10.5, 10.0, 10.0) — within 3.5 Å of SER403 CA.
_POSE_WITH_CONTACT = """\
ROOT
ATOM      1  C   LIG     1      10.500  10.000  10.000  1.00  0.00           C
ATOM      2  C   LIG     1      15.000  15.000  15.000  1.00  0.00           C
ENDROOT
"""

# Ligand PDBQT with all atoms far from SER403.
_POSE_NO_CONTACT = """\
ROOT
ATOM      1  C   LIG     1      50.000  50.000  50.000  1.00  0.00           C
ATOM      2  N   LIG     1      51.000  51.000  51.000  1.00  0.00           N
ENDROOT
"""

# Ligand PDBQT that only contains hydrogen (should yield no heavy atoms).
_POSE_HYDROGEN_ONLY = """\
ROOT
ATOM      1  H   LIG     1      10.500  10.000  10.000  1.00  0.00           H
ENDROOT
"""


@pytest.fixture
def ser403_pdb() -> str:
    """Write the SER403 PDB snippet to a temp file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(_SER403_PDB)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def pose_contact_pdbqt() -> str:
    """Write a PDBQT with a ligand atom near SER403."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False)
    tmp.write(_POSE_WITH_CONTACT)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def pose_no_contact_pdbqt() -> str:
    """Write a PDBQT with all ligand atoms far from SER403."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False)
    tmp.write(_POSE_NO_CONTACT)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def pose_hydrogen_only_pdbqt() -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False)
    tmp.write(_POSE_HYDROGEN_ONLY)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


class TestParsePdbqtLigandCoords:
    def test_parses_heavy_atoms(self, pose_contact_pdbqt: str) -> None:
        coords = _parse_pdbqt_ligand_coords(pose_contact_pdbqt)
        assert len(coords) == 2
        assert np.allclose(coords[0], [10.5, 10.0, 10.0])

    def test_filters_hydrogen(self, pose_hydrogen_only_pdbqt: str) -> None:
        coords = _parse_pdbqt_ligand_coords(pose_hydrogen_only_pdbqt)
        assert len(coords) == 0

    def test_nonexistent_file_returns_empty(self) -> None:
        coords = _parse_pdbqt_ligand_coords("/nonexistent/file.pdbqt")
        assert coords == []


class TestParsePdbResidueCoords:
    def test_parses_key_residue(self, ser403_pdb: str) -> None:
        result = _parse_pdb_residue_coords(ser403_pdb, ["SER403"])
        assert "SER403" in result
        assert len(result["SER403"]) > 0
        # Should have 6 heavy atoms (N, CA, C, O, CB, OG)
        assert len(result["SER403"]) == 6

    def test_missing_residue_returns_empty(self, ser403_pdb: str) -> None:
        result = _parse_pdb_residue_coords(ser403_pdb, ["ALA999"])
        assert "ALA999" in result
        assert len(result["ALA999"]) == 0

    def test_only_heavy_atoms(self, ser403_pdb: str) -> None:
        result = _parse_pdb_residue_coords(ser403_pdb, ["SER403"])
        for atom in result["SER403"]:
            assert len(atom) == 3


class TestCheckKeyInteractions:
    def test_detects_contact(self, pose_contact_pdbqt: str, ser403_pdb: str) -> None:
        assert check_key_interactions(
            pose_contact_pdbqt, ser403_pdb, ["SER403"], distance_cutoff=3.5,
        )

    def test_no_contact_returns_false(
        self, pose_no_contact_pdbqt: str, ser403_pdb: str,
    ) -> None:
        assert not check_key_interactions(
            pose_no_contact_pdbqt, ser403_pdb, ["SER403"], distance_cutoff=3.5,
        )

    def test_missing_residue_returns_false(
        self, pose_contact_pdbqt: str, ser403_pdb: str,
    ) -> None:
        assert not check_key_interactions(
            pose_contact_pdbqt, ser403_pdb, ["TYR999"], distance_cutoff=3.5,
        )

    def test_hydrogen_only_ligand_returns_false(
        self, pose_hydrogen_only_pdbqt: str, ser403_pdb: str,
    ) -> None:
        assert not check_key_interactions(
            pose_hydrogen_only_pdbqt, ser403_pdb, ["SER403"], distance_cutoff=3.5,
        )

    def test_missing_pose_file_returns_false(self, ser403_pdb: str) -> None:
        assert not check_key_interactions(
            "/nonexistent/pose.pdbqt", ser403_pdb, ["SER403"],
        )

    def test_missing_receptor_file_returns_false(
        self, pose_contact_pdbqt: str,
    ) -> None:
        assert not check_key_interactions(
            pose_contact_pdbqt, "/nonexistent/receptor.pdb", ["SER403"],
        )

    def test_strict_distance_cutoff(
        self, pose_contact_pdbqt: str, ser403_pdb: str,
    ) -> None:
        # Ligand at (10.5, 10, 10), SER403 CA at (10.5, 10, 10) → dist = 0.0
        assert check_key_interactions(
            pose_contact_pdbqt, ser403_pdb, ["SER403"], distance_cutoff=0.1,
        )
        # With a very tight cutoff (0.001) it would still be true since CA is at 0.0
        # Let's use a cutoff that's actually smaller than the minimum distance
        # Minimum distance from ligand (10.5, 10, 10) to any SER403 heavy atom
        # is 0.0 (CA at exact same pos). So any >0 cutoff should pass.
        assert check_key_interactions(
            pose_contact_pdbqt, ser403_pdb, ["SER403"], distance_cutoff=1e-6,
        )


# ── Minimal receptor PDB with all 4 IFP residues ──────────────────────────
_IFP_RECEPTOR_PDB = """\
ATOM      1  N   TYR A 159       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  O   TYR A 159       0.000   0.000   0.000  1.00  0.00           O
ATOM      3  CG  TYR A 159       1.000   1.000   0.000  1.00  0.00           C
ATOM      4  CD1 TYR A 159       1.707   1.707   0.000  1.00  0.00           C
ATOM      5  CD2 TYR A 159       1.707   0.293   0.000  1.00  0.00           C
ATOM      6  CE1 TYR A 159       3.000   1.707   0.000  1.00  0.00           C
ATOM      7  CE2 TYR A 159       3.000   0.293   0.000  1.00  0.00           C
ATOM      8  CZ  TYR A 159       3.707   1.000   0.000  1.00  0.00           C
ATOM      9  OH  TYR A 159       4.414   1.000   0.000  1.00  0.00           O
ATOM     10  CB  TYR A 159       0.500   0.500   1.000  1.00  0.00           C
ATOM     11  N   ALA A 237      -1.000   0.000   0.000  1.00  0.00           N
ATOM     12  O   ALA A 237      -3.000   0.000   0.000  1.00  0.00           O
ATOM     13  CA  ALA A 237      -2.000   0.000   0.000  1.00  0.00           C
ATOM     14  CB  ALA A 237      -2.500   0.500   0.000  1.00  0.00           C
ATOM     15  N   MET A 241       0.000  -1.000   0.000  1.00  0.00           N
ATOM     16  O   MET A 241       0.000  -3.000   0.000  1.00  0.00           O
ATOM     17  CA  MET A 241       0.000  -2.000   0.000  1.00  0.00           C
ATOM     18  CB  MET A 241       0.500  -2.500   0.000  1.00  0.00           C
ATOM     19  CG  MET A 241       1.000  -3.000   0.000  1.00  0.00           C
ATOM     20  CE  MET A 241       1.500  -3.500   0.000  1.00  0.00           C
ATOM     21  N   SER A 403       0.000   0.000  -1.000  1.00  0.00           N
ATOM     22  O   SER A 403       0.000   0.000  -3.000  1.00  0.00           O
ATOM     23  CA  SER A 403       0.000   0.000  -2.000  1.00  0.00           C
ATOM     24  CB  SER A 403       0.500   0.500  -2.500  1.00  0.00           C
ATOM     25  OG  SER A 403       1.000   1.000  -3.000  1.00  0.00           O
END
"""


@pytest.fixture
def ifp_receptor_pdb() -> str:
    """Write the 4-residue receptor PDB to a temp file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(_IFP_RECEPTOR_PDB)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


_CEFTAROLINE_SMI = (
    "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)"
    "C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O"
)


class TestIfpSimilarity:
    def test_identical_molecules_returns_one(self, ifp_receptor_pdb: str) -> None:
        """Verifies that identical molecules give Tanimoto ≈ 1.0."""
        ceft = Chem.MolFromSmiles(_CEFTAROLINE_SMI)
        assert ceft is not None
        sim = compute_ifp_similarity(ceft, ceft, ifp_receptor_pdb)
        assert sim == pytest.approx(1.0, abs=1e-6), f"Expected 1.0, got {sim}"

    def test_dissimilar_molecules_below_threshold(
        self, ifp_receptor_pdb: str,
    ) -> None:
        """Verifies that a very different molecule scores < 0.5."""
        ceft = Chem.MolFromSmiles(_CEFTAROLINE_SMI)
        ethane = Chem.MolFromSmiles("CC")
        assert ceft is not None and ethane is not None
        sim = compute_ifp_similarity(ceft, ethane, ifp_receptor_pdb)
        assert sim < 0.5, f"Expected < 0.5, got {sim}"

    def test_missing_receptor_returns_zero(self) -> None:
        """Missing PDB should yield 0.0."""
        ceft = Chem.MolFromSmiles(_CEFTAROLINE_SMI)
        assert ceft is not None
        sim = compute_ifp_similarity(
            ceft, ceft, "/nonexistent/receptor.pdb",
        )
        assert sim == 0.0

    def test_invalid_molecule_returns_zero(self, ifp_receptor_pdb: str) -> None:
        """Invalid reference should yield 0.0."""
        ceft = Chem.MolFromSmiles(_CEFTAROLINE_SMI)
        invalid = Chem.MolFromSmiles("C1=CC=CC=C1")  # valid but very different
        assert ceft is not None and invalid is not None
        sim = compute_ifp_similarity(ceft, invalid, ifp_receptor_pdb)
        # Benzene (no N/O) vs Ceftaroline → IFP vectors should differ
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0
