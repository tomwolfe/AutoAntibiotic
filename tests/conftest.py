"""Shared pytest fixtures for the AutoAntibiotic test suite."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rdkit import Chem
from rdkit.Chem import AllChem

# Minimal PDB snippet with ASN159, GLU237, ARG241 (allosteric site)
# and SER403 (active site) for centroid tests.
_MINIMAL_PDB = """\
ATOM      1  N   ASN A 159       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ASN A 159       1.500   1.500   1.500  1.00  0.00           C
ATOM      3  C   ASN A 159       2.500   2.000   1.800  1.00  0.00           C
ATOM      4  O   ASN A 159       3.200   2.800   1.200  1.00  0.00           O
ATOM      5  N   GLU A 237       2.000   2.000   2.000  1.00  0.00           N
ATOM      6  CA  GLU A 237       2.500   2.500   2.500  1.00  0.00           C
ATOM      7  C   GLU A 237       3.500   3.000   2.800  1.00  0.00           C
ATOM      8  O   GLU A 237       4.200   3.800   2.200  1.00  0.00           O
ATOM      9  N   ARG A 241       3.000   3.000   3.000  1.00  0.00           N
ATOM     10  CA  ARG A 241       3.500   3.500   3.500  1.00  0.00           C
ATOM     11  C   ARG A 241       4.500   4.000   3.800  1.00  0.00           C
ATOM     12  O   ARG A 241       5.200   4.800   3.200  1.00  0.00           O
ATOM     13  N   SER A 403       4.000   4.000   4.000  1.00  0.00           N
ATOM     14  CA  SER A 403       4.500   4.500   4.500  1.00  0.00           C
ATOM     15  C   SER A 403       5.500   5.000   4.800  1.00  0.00           C
ATOM     16  O   SER A 403       6.200   5.800   4.200  1.00  0.00           O
END
"""

BETA_LACTAM_SMARTS: str = "[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1"

# A beta-lactam (3,4-dimethylazetidin-2-one) that matches the SMARTS pattern
BETA_LACTAM_SMILES: str = "CC1C(=O)NC1C"

# A non-beta-lactam that should not match
NON_BETA_LACTAM_SMILES: str = "c1ccccc1O"


@pytest.fixture(scope="session")
def test_pdb_path() -> str:
    """Write the minimal PDB to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(_MINIMAL_PDB)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture(scope="session")
def beta_lactam_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles(BETA_LACTAM_SMILES)
    assert mol is not None
    Chem.SanitizeMol(mol)
    return mol


@pytest.fixture(scope="session")
def non_beta_lactam_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles(NON_BETA_LACTAM_SMILES)
    assert mol is not None
    Chem.SanitizeMol(mol)
    return mol
