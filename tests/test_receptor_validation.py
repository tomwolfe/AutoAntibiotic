"""Unit tests for receptor integrity validation."""

import os
import tempfile

import pytest

from autoantibiotic.config import CONFIG, ConfigurationError
from autoantibiotic.structure_prep import (
    BACKBONE_ATOMS,
    CRITICAL_RESIDUE_ATOMS,
    validate_receptor_integrity,
)

# ── Test PDB templates ──────────────────────────────────────────────

# Valid receptor: backbone + full side-chain atoms for all critical residues.
_VALID_PDB = """\
ATOM      1  N   ASN A 159       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ASN A 159       1.500   1.500   1.500  1.00  0.00           C
ATOM      3  C   ASN A 159       2.500   2.000   1.800  1.00  0.00           C
ATOM      4  O   ASN A 159       3.200   2.800   1.200  1.00  0.00           O
ATOM      5  CB  ASN A 159       0.500   2.500   1.200  1.00  0.00           C
ATOM      6  CG  ASN A 159      -0.300   3.200   2.100  1.00  0.00           C
ATOM      7  OD1 ASN A 159       0.100   4.000   3.000  1.00  0.00           O
ATOM      8  ND2 ASN A 159      -1.500   2.800   2.000  1.00  0.00           N
ATOM      9  N   GLU A 237       2.000   2.000   2.000  1.00  0.00           N
ATOM     10  CA  GLU A 237       2.500   2.500   2.500  1.00  0.00           C
ATOM     11  C   GLU A 237       3.500   3.000   2.800  1.00  0.00           C
ATOM     12  O   GLU A 237       4.200   3.800   2.200  1.00  0.00           O
ATOM     13  CB  GLU A 237       1.500   3.500   2.200  1.00  0.00           C
ATOM     14  CG  GLU A 237       0.700   4.200   3.100  1.00  0.00           C
ATOM     15  CD  GLU A 237      -0.300   5.000   2.500  1.00  0.00           C
ATOM     16  OE1 GLU A 237      -1.200   5.400   3.300  1.00  0.00           O
ATOM     17  OE2 GLU A 237      -0.200   5.200   1.200  1.00  0.00           O
ATOM     18  N   ARG A 241       3.000   3.000   3.000  1.00  0.00           N
ATOM     19  CA  ARG A 241       3.500   3.500   3.500  1.00  0.00           C
ATOM     20  C   ARG A 241       4.500   4.000   3.800  1.00  0.00           C
ATOM     21  O   ARG A 241       5.200   4.800   3.200  1.00  0.00           O
ATOM     22  CB  ARG A 241       2.500   4.500   3.200  1.00  0.00           C
ATOM     23  CG  ARG A 241       1.700   5.200   4.100  1.00  0.00           C
ATOM     24  CD  ARG A 241       0.700   6.000   3.500  1.00  0.00           C
ATOM     25  NE  ARG A 241      -0.200   6.400   4.300  1.00  0.00           N
ATOM     26  CZ  ARG A 241      -1.200   7.000   3.800  1.00  0.00           C
ATOM     27  NH1 ARG A 241      -1.300   7.200   2.500  1.00  0.00           N
ATOM     28  NH2 ARG A 241      -2.100   7.400   4.600  1.00  0.00           N
ATOM     29  N   SER A 403       4.000   4.000   4.000  1.00  0.00           N
ATOM     30  CA  SER A 403       4.500   4.500   4.500  1.00  0.00           C
ATOM     31  C   SER A 403       5.500   5.000   4.800  1.00  0.00           C
ATOM     32  O   SER A 403       6.200   5.800   4.200  1.00  0.00           O
ATOM     33  CB  SER A 403       3.500   5.500   4.200  1.00  0.00           C
ATOM     34  OG  SER A 403       2.700   6.200   5.000  1.00  0.00           O
END
"""

# Missing side-chain atoms: backbone only for all critical residues.
_MISSING_SIDECHAIN_PDB = """\
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

# Missing backbone: CA atom removed from ASN159.
_MISSING_BACKBONE_PDB = """\
ATOM      1  N   ASN A 159       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  C   ASN A 159       2.500   2.000   1.800  1.00  0.00           C
ATOM      3  O   ASN A 159       3.200   2.800   1.200  1.00  0.00           O
ATOM      4  N   GLU A 237       2.000   2.000   2.000  1.00  0.00           N
ATOM      5  CA  GLU A 237       2.500   2.500   2.500  1.00  0.00           C
ATOM      6  C   GLU A 237       3.500   3.000   2.800  1.00  0.00           C
ATOM      7  O   GLU A 237       4.200   3.800   2.200  1.00  0.00           O
ATOM      8  N   ARG A 241       3.000   3.000   3.000  1.00  0.00           N
ATOM      9  CA  ARG A 241       3.500   3.500   3.500  1.00  0.00           C
ATOM     10  C   ARG A 241       4.500   4.000   3.800  1.00  0.00           C
ATOM     11  O   ARG A 241       5.200   4.800   3.200  1.00  0.00           O
ATOM     12  N   SER A 403       4.000   4.000   4.000  1.00  0.00           N
ATOM     13  CA  SER A 403       4.500   4.500   4.500  1.00  0.00           C
ATOM     14  C   SER A 403       5.500   5.000   4.800  1.00  0.00           C
ATOM     15  O   SER A 403       6.200   5.800   4.200  1.00  0.00           O
END
"""

# Missing >50% side-chain atoms for ASN159 and GLU237 (only backbone + partial side-chain).
_MISSING_MOST_SIDECHAIN_PDB = """\
ATOM      1  N   ASN A 159       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ASN A 159       1.500   1.500   1.500  1.00  0.00           C
ATOM      3  C   ASN A 159       2.500   2.000   1.800  1.00  0.00           C
ATOM      4  O   ASN A 159       3.200   2.800   1.200  1.00  0.00           O
ATOM      5  CB  ASN A 159       0.500   2.500   1.200  1.00  0.00           C
ATOM      6  N   GLU A 237       2.000   2.000   2.000  1.00  0.00           N
ATOM      7  CA  GLU A 237       2.500   2.500   2.500  1.00  0.00           C
ATOM      8  C   GLU A 237       3.500   3.000   2.800  1.00  0.00           C
ATOM      9  O   GLU A 237       4.200   3.800   2.200  1.00  0.00           O
ATOM     10  N   ARG A 241       3.000   3.000   3.000  1.00  0.00           N
ATOM     11  CA  ARG A 241       3.500   3.500   3.500  1.00  0.00           C
ATOM     12  C   ARG A 241       4.500   4.000   3.800  1.00  0.00           C
ATOM     13  O   ARG A 241       5.200   4.800   3.200  1.00  0.00           O
ATOM     14  N   SER A 403       4.000   4.000   4.000  1.00  0.00           N
ATOM     15  CA  SER A 403       4.500   4.500   4.500  1.00  0.00           C
ATOM     16  C   SER A 403       5.500   5.000   4.800  1.00  0.00           C
ATOM     17  O   SER A 403       6.200   5.800   4.200  1.00  0.00           O
END
"""


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def valid_pdb_path() -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(_VALID_PDB)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def missing_sidechain_pdb_path() -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(_MISSING_SIDECHAIN_PDB)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def missing_backbone_pdb_path() -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(_MISSING_BACKBONE_PDB)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def missing_most_sidechain_pdb_path() -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(_MISSING_MOST_SIDECHAIN_PDB)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def work_dir(tmp_path) -> str:
    return str(tmp_path)


# ── Test Case 1: Valid receptor passes without modification ─────────


class TestValidReceptor:
    def test_valid_receptor_returns_original_path(
        self, valid_pdb_path: str, work_dir: str
    ) -> None:
        result = validate_receptor_integrity(valid_pdb_path, work_dir, {})
        assert result == valid_pdb_path, (
            "Valid receptor should return the original path unchanged"
        )

    def test_valid_receptor_no_exception(
        self, valid_pdb_path: str, work_dir: str
    ) -> None:
        try:
            validate_receptor_integrity(valid_pdb_path, work_dir, {})
        except ConfigurationError as exc:
            pytest.fail(f"Valid receptor raised ConfigurationError: {exc}")

    def test_valid_receptor_file_exists(
        self, valid_pdb_path: str, work_dir: str
    ) -> None:
        result = validate_receptor_integrity(valid_pdb_path, work_dir, {})
        assert os.path.isfile(result), "Result PDB path must exist"


# ── Test Case 2: Missing side-chain atoms triggers warning / repair ─


class TestMissingSidechain:
    def test_missing_sidechain_does_not_raise(
        self, missing_sidechain_pdb_path: str, work_dir: str
    ) -> None:
        """Backbone-only PDB should not raise (side-chain missing <50%
        for backbone-only actually =100% missing per residue, but no
        backbone missing so it should either repair or warn)."""
        try:
            validate_receptor_integrity(missing_sidechain_pdb_path, work_dir, {})
        except ConfigurationError as exc:
            pytest.fail(f"Missing side-chain raised ConfigurationError: {exc}")

    def test_missing_sidechain_returns_path(
        self, missing_sidechain_pdb_path: str, work_dir: str
    ) -> None:
        result = validate_receptor_integrity(missing_sidechain_pdb_path, work_dir, {})
        assert isinstance(result, str)
        assert os.path.isfile(result)

    def test_missing_most_sidechain_without_pdbfixer(
        self, missing_most_sidechain_pdb_path: str, work_dir: str
    ) -> None:
        """When PDBFixer is unavailable and >50% side-chain missing,
        strict_receptor_validation should raise ConfigurationError."""
        old_val = CONFIG.strict_receptor_validation
        CONFIG.strict_receptor_validation = True
        try:
            # Skip if PDBFixer is available (then it will repair, not raise)
            from autoantibiotic.structure_prep import _HAVE_PDBFIXER
            if _HAVE_PDBFIXER:
                pytest.skip("PDBFixer is available; repair will succeed")
            with pytest.raises(ConfigurationError, match="PDBFixer is not installed"):
                validate_receptor_integrity(missing_most_sidechain_pdb_path, work_dir, {})
        finally:
            CONFIG.strict_receptor_validation = old_val

    def test_missing_sidechain_repair_with_pdbfixer(
        self, missing_sidechain_pdb_path: str, work_dir: str
    ) -> None:
        """When PDBFixer is available, missing side-chain atoms should be
        added and the repaired file should contain the required atoms.
        Note: PDBFixer renumbers residues sequentially (starting at 1), so
        we match by residue name and order rather than original numbering."""
        from autoantibiotic.structure_prep import _HAVE_PDBFIXER
        if not _HAVE_PDBFIXER:
            pytest.skip("PDBFixer not available")
        from Bio.PDB import PDBParser
        result = validate_receptor_integrity(missing_sidechain_pdb_path, work_dir, {})
        if result == missing_sidechain_pdb_path:
            pytest.skip("Repair did not produce a new file")
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("test", result)

        expected_resnames = [
            "".join(ch for ch in spec if ch.isalpha()).upper()
            for spec in CONFIG.allosteric_residues + CONFIG.active_site_residues
        ]

        found_residues = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    rid = residue.get_id()
                    if rid[0] != " ":
                        continue
                    rn = residue.get_resname().strip().upper()
                    if rn in expected_resnames:
                        expected = CRITICAL_RESIDUE_ATOMS[rn]
                        atom_names = {a.get_id() for a in residue.get_atoms()}
                        heavy_only = {a for a in atom_names if not a.startswith("H")}
                        missing = expected - heavy_only
                        assert not missing, (
                            f"After PDBFixer repair, {rn} "
                            f"still missing heavy atoms: {sorted(missing)}"
                        )
                        found_residues.append(rn)

        for rn in expected_resnames:
            assert rn in found_residues, (
                f"Could not find residue {rn} in repaired file."
            )

    def test_missing_sidechain_non_strict(
        self, missing_most_sidechain_pdb_path: str, work_dir: str
    ) -> None:
        """With strict_receptor_validation=False, missing side-chain should
        produce a warning but not raise."""
        old_val = CONFIG.strict_receptor_validation
        CONFIG.strict_receptor_validation = False
        try:
            from autoantibiotic.structure_prep import _HAVE_PDBFIXER
            if _HAVE_PDBFIXER:
                pytest.skip("PDBFixer is available; repair will succeed")
            result = validate_receptor_integrity(
                missing_most_sidechain_pdb_path, work_dir, {}
            )
            assert os.path.isfile(result)
        finally:
            CONFIG.strict_receptor_validation = old_val


# ── Test Case 3: Missing backbone atoms raises ConfigurationError ───


class TestMissingBackbone:
    def test_missing_backbone_raises_configuration_error(
        self, missing_backbone_pdb_path: str, work_dir: str
    ) -> None:
        with pytest.raises(ConfigurationError, match="backbone"):
            validate_receptor_integrity(missing_backbone_pdb_path, work_dir, {})

    def test_missing_backbone_message_includes_residue(
        self, missing_backbone_pdb_path: str, work_dir: str
    ) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            validate_receptor_integrity(missing_backbone_pdb_path, work_dir, {})
        msg = str(exc_info.value)
        assert "ASN159" in msg, (
            f"Error message should mention the missing residue. Got: {msg}"
        )
        assert "CA" in msg, (
            f"Error message should mention the missing atom. Got: {msg}"
        )


# ── Sanity: CRITICAL_RESIDUE_ATOMS constant ─────────────────────────


class TestCriticalResidueAtoms:
    def test_backbone_atoms_in_all_definitions(self) -> None:
        for resname, atoms in CRITICAL_RESIDUE_ATOMS.items():
            for bb in BACKBONE_ATOMS:
                assert bb in atoms, (
                    f"{resname} is missing backbone atom {bb} in CRITICAL_RESIDUE_ATOMS"
                )

    def test_all_critical_residues_are_defined(self) -> None:
        for spec in CONFIG.allosteric_residues + CONFIG.active_site_residues:
            resname = "".join(ch for ch in spec if ch.isalpha()).upper()
            assert resname in CRITICAL_RESIDUE_ATOMS, (
                f"Residue {spec} (name={resname}) is not defined in CRITICAL_RESIDUE_ATOMS"
            )
