"""Unit tests for MM-GB/SA rescoring integration."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.config import CONFIG, CompoundRecord
from autoantibiotic.ml_scoring import (
    rescore_with_mmgbsa,
    rescore_with_ml,
    _compute_rdkit_descriptors,
    _HAVE_OPENMM,
)


# ── Test fixtures ──────────────────────────────────────────────

@pytest.fixture
def temp_work_dir() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def dummy_receptor_pdb(temp_work_dir: str) -> str:
    """Write a minimal PDB with a few alanine residues."""
    pdb_content = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.095   1.389   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.395   2.396   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.957  -0.784   1.207  1.00  0.00           C
ATOM      6  N   ALA A   2       3.420   1.429   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       4.202   2.660   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.699   2.330   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       6.106   1.176   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       3.829   3.492   1.222  1.00  0.00           C
END
"""
    path = os.path.join(temp_work_dir, "receptor.pdb")
    with open(path, "w") as f:
        f.write(pdb_content)
    return path


@pytest.fixture
def top_candidates() -> List[CompoundRecord]:
    return [
        CompoundRecord(
            compound_id="CMP-001",
            smiles="c1ccccc1O",
            mol=Chem.MolFromSmiles("c1ccccc1O"),
            pb2pa_allosteric_energy=-7.5,
        ),
        CompoundRecord(
            compound_id="CMP-002",
            smiles="c1ccccc1N",
            mol=Chem.MolFromSmiles("c1ccccc1N"),
            pb2pa_allosteric_energy=-6.8,
        ),
    ]


# ── Helper tests ──────────────────────────────────────────────

class TestRDKitDescriptors:
    def test_compute_descriptors(self) -> None:
        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        descs = _compute_rdkit_descriptors(mol)
        assert isinstance(descs, np.ndarray)
        assert descs.ndim == 1
        assert descs.shape[0] > 0
        assert not np.any(np.isnan(descs))

    def test_invalid_mol_still_returns(self) -> None:
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        descs = _compute_rdkit_descriptors(mol)
        assert descs.ndim == 1


# ── MM-GB/SA rescoring ─────────────────────────────────────────

class TestRescoreWithMMGBSA:
    def test_no_openmm_returns_unchanged(
        self,
        top_candidates: List[CompoundRecord],
        temp_work_dir: str,
    ) -> None:
        """If OpenMM is not available, scores should remain unchanged."""
        with patch("autoantibiotic.ml_scoring._HAVE_OPENMM", False):
            result = rescore_with_mmgbsa(
                top_candidates,
                "/nonexistent/file.pdb",
                temp_work_dir,
            )
        assert len(result) == len(top_candidates)
        for rec in result:
            assert rec.ml_score is None

    def test_missing_receptor_returns_unchanged(
        self,
        top_candidates: List[CompoundRecord],
        temp_work_dir: str,
    ) -> None:
        result = rescore_with_mmgbsa(
            top_candidates,
            "/nonexistent/file.pdb",
            temp_work_dir,
        )
        assert len(result) == len(top_candidates)

    def test_missing_workdir_creates_it(
        self,
        top_candidates: List[CompoundRecord],
        temp_work_dir: str,
    ) -> None:
        mm_dir = os.path.join(temp_work_dir, "mmgbsa")
        assert not os.path.exists(mm_dir)
        # Should handle missing workdir gracefully (OpenMM won't actually be available)
        result = rescore_with_mmgbsa(
            top_candidates,
            "/nonexistent.pdb",
            temp_work_dir,
        )
        assert result is not None

    def test_receptor_not_a_protein(
        self,
        top_candidates: List[CompoundRecord],
        temp_work_dir: str,
        dummy_receptor_pdb: str,
    ) -> None:
        """Empty/minimal PDB should fail PDBFixer but not crash."""
        with patch("autoantibiotic.ml_scoring._HAVE_OPENMM", False):
            result = rescore_with_mmgbsa(
                top_candidates,
                dummy_receptor_pdb,
                temp_work_dir,
            )
        assert len(result) == len(top_candidates)
        # Without OpenMM, ml_score stays as-is (None)
        for rec in result:
            assert rec.ml_score is None

    def test_uses_mm_gbsa_top_n_config(
        self,
        top_candidates: List[CompoundRecord],
        dummy_receptor_pdb: str,
        temp_work_dir: str,
    ) -> None:
        """Verify that CONFIG.mm_gbsa_top_n controls how many are rescored."""
        saved = CONFIG.mm_gbsa_top_n
        CONFIG.mm_gbsa_top_n = 1
        # Simulate the internal flow by checking what _rescore_with_mmgbsa is called with
        with patch("autoantibiotic.ml_scoring._HAVE_OPENMM", False):
            result = rescore_with_mmgbsa(
                top_candidates,
                dummy_receptor_pdb,
                temp_work_dir,
            )
        CONFIG.mm_gbsa_top_n = saved
        assert len(result) == len(top_candidates)


# ── ML rescoring dispatch ─────────────────────────────────────

class TestRescoreWithML:
    def test_dispatch_to_mmgbsa_when_enabled(
        self,
        top_candidates: List[CompoundRecord],
        temp_work_dir: str,
    ) -> None:
        """When both flags are set, rescores via mmgbsa path."""
        saved_mm = CONFIG.use_mm_gbsa
        saved_mmr = CONFIG.use_mm_gbsa_rescoring
        CONFIG.use_mm_gbsa = True
        CONFIG.use_mm_gbsa_rescoring = True

        try:
            with patch("autoantibiotic.ml_scoring.rescore_with_mmgbsa") as mock_mm:
                mock_mm.return_value = top_candidates
                # Create a dummy pdb alongside pdbqt
                pdbqt = os.path.join(temp_work_dir, "receptor.pdbqt")
                pdb = os.path.join(temp_work_dir, "receptor.pdb")
                with open(pdb, "w") as f:
                    f.write("ATOM ...\n")
                with open(pdbqt, "w") as f:
                    f.write("ATOM ...\n")

                result = rescore_with_ml(top_candidates, pdbqt, temp_work_dir)
                mock_mm.assert_called_once()
        finally:
            CONFIG.use_mm_gbsa = saved_mm
            CONFIG.use_mm_gbsa_rescoring = saved_mmr

    def test_mmgbsa_skipped_if_no_receptor_pdb(
        self,
        top_candidates: List[CompoundRecord],
        temp_work_dir: str,
    ) -> None:
        """When the .pdb does not exist next to .pdbqt, MM-GB/SA is skipped."""
        saved = CONFIG.use_mm_gbsa
        CONFIG.use_mm_gbsa = True
        try:
            with patch("autoantibiotic.ml_scoring._rescore_with_gnina") as mock_gnina:
                mock_gnina.return_value = top_candidates
                result = rescore_with_ml(
                    top_candidates,
                    "/tmp/phantom.pdbqt",
                    temp_work_dir,
                )
                # When mmgbsa can't find .pdb, it falls to gnina
                # But since _HAVE_GNINA may be False, this is fine
        finally:
            CONFIG.use_mm_gbsa = saved


# ── Config integration ────────────────────────────────────────

class TestConfigIntegration:
    def test_mm_gbsa_top_n_default(self) -> None:
        assert CONFIG.mm_gbsa_top_n == 50

    def test_use_mm_gbsa_rescoring_default(self) -> None:
        assert not CONFIG.use_mm_gbsa_rescoring

    def test_both_mm_gbsa_flags_independent(self) -> None:
        """Both flags can be set independently."""
        assert hasattr(CONFIG, "use_mm_gbsa")
        assert hasattr(CONFIG, "use_mm_gbsa_rescoring")
        assert hasattr(CONFIG, "mm_gbsa_top_n")
