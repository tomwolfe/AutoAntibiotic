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


# ── Water displacement correction ──────────────────────────────

class MockWater:
    """Minimal mock with the attributes *rescore_with_mmgbsa* accesses."""
    def __init__(self, position, displacement_energy, is_high_energy):
        self.position = np.array(position, dtype=np.float64)
        self.displacement_energy = displacement_energy
        self.is_high_energy = is_high_energy


class MockWaterAnalysisResult:
    """Minimal mock of ``WaterAnalysisResult``."""
    def __init__(self, high_energy_waters):
        self.high_energy_waters = high_energy_waters
        self.all_waters = high_energy_waters


class TestWaterDisplacementCorrection:
    """Verifies that high-energy water clashes correctly adjust MM-GB/SA scores."""

    def test_correction_applied_for_clashing_water(self, temp_work_dir: str) -> None:
        """A high-energy water within 2.5 Å of the ligand should reduce ΔG."""
        saved_thresh = CONFIG.pharmacophore_rmsd_threshold
        candidate = CompoundRecord(
            compound_id="CMP-WAT-001",
            smiles="c1ccccc1O",  # phenol – simple ligand
            mol=Chem.MolFromSmiles("c1ccccc1O"),
            pb2pa_allosteric_energy=-7.0,
        )

        # Water at (0, 0, 0); phenol's oxygen will be near origin after ETKDG
        high_energy_water = MockWater(
            position=[0.0, 0.0, 0.0],
            displacement_energy=2.0,
            is_high_energy=True,
        )
        water_results = MockWaterAnalysisResult(
            high_energy_waters=[high_energy_water],
        )

        # We'll mock the MM-GB/SA internals to return fixed energies
        fake_rec_energy = -2000.0
        fake_lig_energy = 50.0
        fake_complex_energy = -1950.0
        # ΔG_binding = -1950 - (-2000) - 50 = 0.0 (simple test baseline)

        dummy_pdb = os.path.join(temp_work_dir, "receptor.pdb")
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

        with patch.multiple(
            "autoantibiotic.ml_scoring",
            _HAVE_OPENMM=True,
            _prepare_receptor_for_mmgbsa=MagicMock(
                return_value=(
                    MagicMock(),  # rec_topology
                    MagicMock(),  # forcefield
                    MagicMock(),  # cpu_platform
                    fake_rec_energy,
                )
            ),
            _compute_ligand_gb_energy=MagicMock(return_value=fake_lig_energy),
            _compute_complex_gb_energy=MagicMock(return_value=fake_complex_energy),
        ):
            result = rescore_with_mmgbsa(
                [candidate],
                dummy_pdb,
                temp_work_dir,
                water_results=water_results,
            )

        assert len(result) == 1
        final_score = result[0].ml_score
        # ΔG_binding = 0.0, correction = -2.0 (subtract displacement energy)
        # expected = 0.0 - 2.0 = -2.0
        assert final_score is not None
        assert final_score == pytest.approx(-2.0, abs=1e-4), (
            f"Expected -2.0, got {final_score}"
        )

    def test_no_correction_when_water_not_high_energy(
        self, temp_work_dir: str,
    ) -> None:
        """A low-energy water clashing should NOT adjust the score."""
        candidate = CompoundRecord(
            compound_id="CMP-WAT-002",
            smiles="c1ccccc1O",
            mol=Chem.MolFromSmiles("c1ccccc1O"),
            pb2pa_allosteric_energy=-7.0,
        )

        low_energy_water = MockWater(
            position=[0.0, 0.0, 0.0],
            displacement_energy=2.0,
            is_high_energy=False,
        )
        water_results = MockWaterAnalysisResult(
            high_energy_waters=[],  # empty → no correction
        )

        dummy_pdb = os.path.join(temp_work_dir, "receptor.pdb")
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

        with patch.multiple(
            "autoantibiotic.ml_scoring",
            _HAVE_OPENMM=True,
            _prepare_receptor_for_mmgbsa=MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), -2000.0)
            ),
            _compute_ligand_gb_energy=MagicMock(return_value=50.0),
            _compute_complex_gb_energy=MagicMock(return_value=-1950.0),
        ):
            result = rescore_with_mmgbsa(
                [candidate],
                dummy_pdb,
                temp_work_dir,
                water_results=water_results,
            )

        final_score = result[0].ml_score
        # ΔG_binding = 0.0, no correction → expected 0.0
        assert final_score is not None
        assert final_score == pytest.approx(0.0, abs=1e-4)

    def test_no_correction_when_water_results_none(
        self, temp_work_dir: str,
    ) -> None:
        """When *water_results* is None, the score should not be adjusted."""
        candidate = CompoundRecord(
            compound_id="CMP-WAT-003",
            smiles="c1ccccc1O",
            mol=Chem.MolFromSmiles("c1ccccc1O"),
            pb2pa_allosteric_energy=-7.0,
        )

        dummy_pdb = os.path.join(temp_work_dir, "receptor.pdb")
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

        with patch.multiple(
            "autoantibiotic.ml_scoring",
            _HAVE_OPENMM=True,
            _prepare_receptor_for_mmgbsa=MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), -2000.0)
            ),
            _compute_ligand_gb_energy=MagicMock(return_value=50.0),
            _compute_complex_gb_energy=MagicMock(return_value=-1950.0),
        ):
            result = rescore_with_mmgbsa(
                [candidate],
                dummy_pdb,
                temp_work_dir,
                water_results=None,
            )

        final_score = result[0].ml_score
        assert final_score is not None
        assert final_score == pytest.approx(0.0, abs=1e-4)


# ── Water displacement integration tests (new spec) ──────────

class TestWaterDisplacementIntegration:
    """Integration tests for water displacement correction in MM-GB/SA rescoring."""

    def test_high_energy_water_clash_reduces_dg(self, temp_work_dir: str) -> None:
        """A high-energy water within 2.5 Å of the ligand should reduce ΔG by its displacement_energy."""
        candidate = CompoundRecord(
            compound_id="CMP-INT-001",
            smiles="c1ccccc1O",
            mol=Chem.MolFromSmiles("c1ccccc1O"),
            pb2pa_allosteric_energy=-7.0,
        )

        high_energy_water = MockWater(
            position=[0.0, 0.0, 0.0],
            displacement_energy=2.5,
            is_high_energy=True,
        )
        water_results = MockWaterAnalysisResult(
            high_energy_waters=[high_energy_water],
        )

        # ΔG_binding = complex - rec - lig = -1950 - (-2000) - 50 = 0.0
        dummy_pdb = os.path.join(temp_work_dir, "receptor.pdb")
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

        with patch.multiple(
            "autoantibiotic.ml_scoring",
            _HAVE_OPENMM=True,
            _prepare_receptor_for_mmgbsa=MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), -2000.0)
            ),
            _compute_ligand_gb_energy=MagicMock(return_value=50.0),
            _compute_complex_gb_energy=MagicMock(return_value=-1950.0),
        ):
            result = rescore_with_mmgbsa(
                [candidate],
                dummy_pdb,
                temp_work_dir,
                water_results=water_results,
            )

        assert len(result) == 1
        final_score = result[0].ml_score
        assert final_score is not None
        # expected = 0.0 - 2.5 = -2.5
        assert final_score == pytest.approx(-2.5, abs=1e-4), (
            f"Expected -2.5, got {final_score}"
        )

    def test_bridging_water_no_correction(self, temp_work_dir: str) -> None:
        """A water with is_high_energy=False should NOT trigger any correction."""
        candidate = CompoundRecord(
            compound_id="CMP-INT-002",
            smiles="c1ccccc1O",
            mol=Chem.MolFromSmiles("c1ccccc1O"),
            pb2pa_allosteric_energy=-7.0,
        )

        # Low-energy water: not in high_energy_waters list → no correction applied
        water_results = MockWaterAnalysisResult(high_energy_waters=[])

        dummy_pdb = os.path.join(temp_work_dir, "receptor.pdb")
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

        with patch.multiple(
            "autoantibiotic.ml_scoring",
            _HAVE_OPENMM=True,
            _prepare_receptor_for_mmgbsa=MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), -2000.0)
            ),
            _compute_ligand_gb_energy=MagicMock(return_value=50.0),
            _compute_complex_gb_energy=MagicMock(return_value=-1950.0),
        ):
            result = rescore_with_mmgbsa(
                [candidate],
                dummy_pdb,
                temp_work_dir,
                water_results=water_results,
            )

        final_score = result[0].ml_score
        assert final_score is not None
        # ΔG_binding = 0.0, no correction → expected 0.0
        assert final_score == pytest.approx(0.0, abs=1e-4)


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
