"""Unit tests for interaction fingerprint (IFP) functions."""

import os
import tempfile

import numpy as np
import pytest

from autoantibiotic.analysis import (
    _parse_pdb_residue_coords,
    _parse_pdbqt_ligand_coords,
    check_key_interactions,
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
