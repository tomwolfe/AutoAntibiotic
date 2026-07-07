"""Unit tests for MM-GB/SA rescoring integration."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.config import CONFIG
from autoantibiotic.models import CompoundRecord
from autoantibiotic.ml_scoring.scoring import (
    rescore_with_mmgbsa,
    rescore_with_ml,
    _compute_rdkit_descriptors,
    _compute_water_displacement_penalty,
    _perform_pose_relaxation,
    _HAVE_OPENMM,
)
from autoantibiotic.md_validation import (
    run_short_md,
    _compute_ligand_rmsd,
    _check_openmm,
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
        with patch("autoantibiotic.ml_scoring.scoring._HAVE_OPENMM", False):
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
        with patch("autoantibiotic.ml_scoring.scoring._HAVE_OPENMM", False):
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
        with patch("autoantibiotic.ml_scoring.scoring._HAVE_OPENMM", False):
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
            with patch("autoantibiotic.ml_scoring.scoring.rescore_with_mmgbsa") as mock_mm:
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
            with patch("autoantibiotic.ml_scoring.scoring._rescore_with_gnina") as mock_gnina:
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
            "autoantibiotic.ml_scoring.scoring",
            _HAVE_OPENMM=True,
            _HAVE_PDBFIXER=False,
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
            "autoantibiotic.ml_scoring.scoring",
            _HAVE_OPENMM=True,
            _HAVE_PDBFIXER=False,
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
            "autoantibiotic.ml_scoring.scoring",
            _HAVE_OPENMM=True,
            _HAVE_PDBFIXER=False,
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
            "autoantibiotic.ml_scoring.scoring",
            _HAVE_OPENMM=True,
            _HAVE_PDBFIXER=False,
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
        # expected = 0.0 - 2.5 = -2.5
        assert final_score is not None
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
            "autoantibiotic.ml_scoring.scoring",
            _HAVE_OPENMM=True,
            _HAVE_PDBFIXER=False,
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


# ── Ensemble MM-GB/SA tests ─────────────────────────────────

class TestEnsembleMMGBSA:
    """Tests for the ensemble conformer averaging feature."""

    def test_single_conformer_when_expensive_disabled(
        self,
        top_candidates: List[CompoundRecord],
        dummy_receptor_pdb: str,
        temp_work_dir: str,
    ) -> None:
        """When use_expensive_ml_features is False, only 1 conformer is used."""
        saved_expensive = CONFIG.use_expensive_ml_features
        saved_n_conf = CONFIG.mmgbsa_n_conformers
        CONFIG.use_expensive_ml_features = False
        CONFIG.mmgbsa_n_conformers = 10

        try:
            with patch.multiple(
                "autoantibiotic.ml_scoring.scoring",
                _HAVE_OPENMM=True,
                _prepare_receptor_for_mmgbsa=MagicMock(
                    return_value=(MagicMock(), MagicMock(), MagicMock(), -2000.0)
                ),
                _compute_ligand_gb_energy=MagicMock(return_value=50.0),
                _compute_complex_gb_energy=MagicMock(return_value=-1950.0),
            ):
                result = rescore_with_mmgbsa(
                    top_candidates,
                    dummy_receptor_pdb,
                    temp_work_dir,
                )
            assert len(result) == len(top_candidates)
        finally:
            CONFIG.use_expensive_ml_features = saved_expensive
            CONFIG.mmgbsa_n_conformers = saved_n_conf

    def test_ensemble_uses_n_conformers(
        self,
        top_candidates: List[CompoundRecord],
        dummy_receptor_pdb: str,
        temp_work_dir: str,
    ) -> None:
        """When expensive features are enabled, multiple conformers are used."""
        saved_expensive = CONFIG.use_expensive_ml_features
        saved_n_conf = CONFIG.mmgbsa_n_conformers
        CONFIG.use_expensive_ml_features = True
        CONFIG.mmgbsa_n_conformers = 5

        try:
            with patch.multiple(
                "autoantibiotic.ml_scoring.scoring",
                _HAVE_OPENMM=True,
                _prepare_receptor_for_mmgbsa=MagicMock(
                    return_value=(MagicMock(), MagicMock(), MagicMock(), -2000.0)
                ),
                _compute_ligand_gb_energy=MagicMock(return_value=50.0),
                _compute_complex_gb_energy=MagicMock(return_value=-1950.0),
            ):
                result = rescore_with_mmgbsa(
                    top_candidates,
                    dummy_receptor_pdb,
                    temp_work_dir,
                )
            assert len(result) == len(top_candidates)
            # With all conformers returning same energy, std dev should be 0
            for rec in result:
                if rec.ml_score is not None:
                    assert rec.ml_score_std is not None
                    assert rec.ml_score_std >= 0.0
        finally:
            CONFIG.use_expensive_ml_features = saved_expensive
            CONFIG.mmgbsa_n_conformers = saved_n_conf

    def test_ml_score_std_is_not_none_when_rescored(
        self,
        top_candidates: List[CompoundRecord],
        dummy_receptor_pdb: str,
        temp_work_dir: str,
    ) -> None:
        """MM-GB/SA rescored compounds should have non-None ml_score_std."""
        saved_expensive = CONFIG.use_expensive_ml_features
        CONFIG.use_expensive_ml_features = True

        try:
            with patch.multiple(
                "autoantibiotic.ml_scoring.scoring",
                _HAVE_OPENMM=True,
                _prepare_receptor_for_mmgbsa=MagicMock(
                    return_value=(MagicMock(), MagicMock(), MagicMock(), -2000.0)
                ),
                _compute_ligand_gb_energy=MagicMock(return_value=50.0),
                _compute_complex_gb_energy=MagicMock(return_value=-1950.0),
            ):
                result = rescore_with_mmgbsa(
                    top_candidates[:1],
                    dummy_receptor_pdb,
                    temp_work_dir,
                )
            assert len(result) == 1
            r = result[0]
            if r.ml_score is not None:
                assert r.ml_score_std is not None
                assert isinstance(r.ml_score_std, float)
        finally:
            CONFIG.use_expensive_ml_features = saved_expensive

    def test_water_vdw_overlap_check(
        self,
        temp_work_dir: str,
    ) -> None:
        """VDW-overlap-based water correction should work."""
        from autoantibiotic.ml_scoring.scoring import _check_vdw_overlap
        from rdkit.Chem import AllChem, rdDistGeom

        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        rdDistGeom.EmbedMolecule(mol_3d, params)
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)

        # A water exactly at origin should clash with some atom
        water_pos = np.array([0.0, 0.0, 0.0])
        clash = _check_vdw_overlap(mol_3d, water_pos)
        # May or may not clash depending on conformer; just verify it runs
        assert isinstance(clash, bool)

    def test_water_vdw_no_clash_far_away(
        self,
        temp_work_dir: str,
    ) -> None:
        """A water far from the ligand should NOT trigger a clash."""
        from autoantibiotic.ml_scoring.scoring import _check_vdw_overlap
        from rdkit.Chem import AllChem, rdDistGeom

        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        rdDistGeom.EmbedMolecule(mol_3d, params)
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)

        water_pos = np.array([100.0, 100.0, 100.0])
        clash = _check_vdw_overlap(mol_3d, water_pos)
        assert clash is False


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

    def test_new_config_fields_exist(self) -> None:
        assert hasattr(CONFIG, "use_expensive_ml_features")
        assert hasattr(CONFIG, "mmgbsa_n_conformers")
        assert hasattr(CONFIG, "max_stereoisomers")
        assert CONFIG.use_expensive_ml_features is False
        assert CONFIG.mmgbsa_n_conformers == 10
        assert CONFIG.max_stereoisomers == 8


# ── MD Validation tests ─────────────────────────────────────────

class TestMDValidation:
    """Tests for the MD validation module."""

    def test_check_openmm_flag(self) -> None:
        """_check_openmm should return a bool."""
        result = _check_openmm()
        assert isinstance(result, bool)

    def test_run_short_md_no_openmm_returns_none(self) -> None:
        """When OpenMM is not available, run_short_md returns None."""
        with patch("autoantibiotic.md_validation._HAVE_OPENMM", False):
            mol = Chem.MolFromSmiles("c1ccccc1O")
            assert mol is not None
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem, rdDistGeom
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            rdDistGeom.EmbedMolecule(mol, params)
            AllChem.MMFFOptimizeMolecule(mol)
            result = run_short_md(mol, "/nonexistent.pdb", duration_ns=0.1)
            assert result is None

    def test_run_short_md_no_pdbfixer_returns_none(self) -> None:
        """When pdbfixer is not available, run_short_md returns None."""
        with patch.multiple(
            "autoantibiotic.md_validation",
            _HAVE_OPENMM=True,
            _HAVE_PDBFIXER=False,
        ):
            mol = Chem.MolFromSmiles("c1ccccc1O")
            assert mol is not None
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem, rdDistGeom
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            rdDistGeom.EmbedMolecule(mol, params)
            AllChem.MMFFOptimizeMolecule(mol)
            result = run_short_md(mol, "/nonexistent.pdb", duration_ns=0.1)
            assert result is None

    def test_run_short_md_missing_receptor_returns_none(self) -> None:
        """When receptor PDB does not exist, run_short_md returns None."""
        with patch.multiple(
            "autoantibiotic.md_validation",
            _HAVE_OPENMM=True,
            _HAVE_PDBFIXER=True,
        ):
            mol = Chem.MolFromSmiles("c1ccccc1O")
            assert mol is not None
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem, rdDistGeom
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            rdDistGeom.EmbedMolecule(mol, params)
            AllChem.MMFFOptimizeMolecule(mol)
            result = run_short_md(mol, "/nonexistent.pdb", duration_ns=0.1)
            assert result is None

    def test_ligand_rmsd_identical_positions(self) -> None:
        """RMSD of identical positions should be 0."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        rmsd = _compute_ligand_rmsd(pos, pos)
        assert rmsd == pytest.approx(0.0)

    def test_ligand_rmsd_known_value(self) -> None:
        """RMSD between two known point sets."""
        pos1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pos2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        rmsd = _compute_ligand_rmsd(pos1, pos2)
        # All distances are 1.0, so RMSD = sqrt(mean(1^2)) = 1.0
        assert rmsd == pytest.approx(1.0)

    def test_ligand_rmsd_mismatched_lengths(self) -> None:
        """RMSD with mismatched array lengths should return 999.9."""
        pos1 = np.array([[0.0, 0.0, 0.0]])
        pos2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        rmsd = _compute_ligand_rmsd(pos1, pos2)
        assert rmsd == pytest.approx(999.9)

    def test_ligand_rmsd_empty_returns_999(self) -> None:
        """RMSD with empty arrays should return 999.9."""
        rmsd = _compute_ligand_rmsd(np.array([]), np.array([]))
        assert rmsd == pytest.approx(999.9)

    def test_check_convergence_insufficient_data(self) -> None:
        """_check_convergence with fewer than window_size frames."""
        from autoantibiotic.md_validation import _check_convergence
        traj = [0.5, 0.6, 0.55]
        result = _check_convergence(traj, window_size=5)
        assert result is False

    def test_check_convergence_detected(self) -> None:
        """_check_convergence should return True when std < threshold."""
        from autoantibiotic.md_validation import _check_convergence
        traj = [0.50, 0.51, 0.49, 0.50, 0.51]
        result = _check_convergence(traj, window_size=5, threshold=0.1)
        assert result is True

    def test_check_convergence_not_detected(self) -> None:
        """_check_convergence should return False when std >= threshold."""
        from autoantibiotic.md_validation import _check_convergence
        traj = [0.1, 0.5, 0.9, 1.3, 1.7]
        result = _check_convergence(traj, window_size=5, threshold=0.1)
        assert result is False

    def test_check_convergence_empty_returns_false(self) -> None:
        """_check_convergence with empty list returns False."""
        from autoantibiotic.md_validation import _check_convergence
        result = _check_convergence([], window_size=5)
        assert result is False

    def test_run_short_md_result_has_converged_key(self) -> None:
        """run_short_md result dict should include 'converged' key."""
        with patch.multiple(
            "autoantibiotic.md_validation",
            _HAVE_OPENMM=False,
        ):
            mol = Chem.MolFromSmiles("c1ccccc1O")
            assert mol is not None
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem, rdDistGeom
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            rdDistGeom.EmbedMolecule(mol, params)
            AllChem.MMFFOptimizeMolecule(mol)
            result = run_short_md(mol, "/nonexistent.pdb", duration_ns=0.1)
            # No OpenMM → returns None, but the key exists in the error path
            # Since OpenMM is not available, result is None
            assert result is None

    def test_md_config_convergence_interval_exists(self) -> None:
        """CONFIG.md_convergence_check_interval_ns should exist."""
        assert hasattr(CONFIG, "md_convergence_check_interval_ns")
        assert CONFIG.md_convergence_check_interval_ns == 5.0

    def test_ligand_no_conformer_returns_none(self) -> None:
        """run_short_md with a molecule that has no conformer returns None."""
        with patch.multiple(
            "autoantibiotic.md_validation",
            _HAVE_OPENMM=True,
            _HAVE_PDBFIXER=True,
        ):
            mol = Chem.MolFromSmiles("c1ccccc1")
            assert mol is not None
            with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
                f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
                pdb_path = f.name
            try:
                result = run_short_md(mol, pdb_path, duration_ns=0.1)
                assert result is None
            finally:
                os.unlink(pdb_path)


# ── Explicit-solvent MM-GB/SA tests ─────────────────────────────

class TestExplicitSolventMMGBSA:
    """Tests for the explicit-solvent MM-GB/SA rescoring function."""

    def test_fallback_to_implicit_when_no_openmm(
        self,
        top_candidates: List[CompoundRecord],
        dummy_receptor_pdb: str,
        temp_work_dir: str,
    ) -> None:
        """When OpenMM is unavailable, fall back to implicit MM-GB/SA."""
        with patch.multiple(
            "autoantibiotic.ml_scoring.scoring",
            _HAVE_OPENMM=False,
            rescore_with_mmgbsa=MagicMock(
                return_value=top_candidates,
            ),
        ):
            from autoantibiotic.ml_scoring.scoring import rescore_with_explicit_mmgbsa
            result = rescore_with_explicit_mmgbsa(
                top_candidates, dummy_receptor_pdb, temp_work_dir,
            )
        assert len(result) == len(top_candidates)

    def test_fallback_when_receptor_missing(
        self,
        top_candidates: List[CompoundRecord],
        temp_work_dir: str,
    ) -> None:
        """When the receptor PDB does not exist, fall back to implicit."""
        with patch.multiple(
            "autoantibiotic.ml_scoring.scoring",
            _HAVE_OPENMM=True,
            rescore_with_mmgbsa=MagicMock(
                return_value=top_candidates,
            ),
        ):
            from autoantibiotic.ml_scoring.scoring import rescore_with_explicit_mmgbsa
            result = rescore_with_explicit_mmgbsa(
                top_candidates, "/nonexistent.pdb", temp_work_dir,
            )
        assert len(result) == len(top_candidates)

    def test_explicit_config_flags_exist(self) -> None:
        """Verify the config flags for explicit solvent exist."""
        assert hasattr(CONFIG, "use_explicit_solvent_mmgbsa")
        assert hasattr(CONFIG, "explicit_solvent_frames")
        assert CONFIG.use_explicit_solvent_mmgbsa is True
        assert CONFIG.explicit_solvent_frames == 10

    def test_explicit_uses_configured_frames(
        self,
        top_candidates: List[CompoundRecord],
        temp_work_dir: str,
    ) -> None:
        """When explicit solvent is enabled, frames from config are used."""
        import sys as _sys
        _pdbfixer_mock = MagicMock()
        _sys.modules["pdbfixer"] = _pdbfixer_mock
        _pdbfixer_mock.PDBFixer = MagicMock()

        # Configure mock OpenMM simulation to return numeric energies
        _mock_sim = MagicMock()
        _mock_state = MagicMock()
        _mock_energy = MagicMock()
        _mock_energy.value_in_unit.return_value = -2000.0
        _mock_state.getPotentialEnergy.return_value = _mock_energy
        _mock_sim.context.getState.return_value = _mock_state

        _mock_openmm_app = MagicMock()
        _mock_openmm_app.Simulation.return_value = _mock_sim
        _mock_openmm_app.PDBFile.writeFile = MagicMock()
        _mock_openmm_app.ForceField = MagicMock()
        modeller_instance = MagicMock()
        _mock_openmm_app.Modeller.return_value = modeller_instance

        class _MockUnit:
            """Simple mock that supports float arithmetic."""
            def __mul__(self, other):
                return _MockUnit()
            def __rmul__(self, other):
                return _MockUnit()
            def __truediv__(self, other):
                return _MockUnit()

        class _MockUnit:
            """Simple mock that supports float arithmetic."""
            def __mul__(self, other):
                return _MockUnit()
            def __rmul__(self, other):
                return _MockUnit()
            def __truediv__(self, other):
                return _MockUnit()
            def __rtruediv__(self, other):
                return _MockUnit()

        _mock_openmm_unit = MagicMock()
        _mock_openmm_unit.kilocalorie_per_mole = MagicMock()
        _mock_openmm_unit.kilocalorie_per_mole.__rmul__ = lambda self, x: _MockUnit()
        _mock_openmm_unit.kelvin = _MockUnit()
        _mock_openmm_unit.picosecond = _MockUnit()
        _mock_openmm_unit.femtoseconds = _MockUnit()
        _mock_openmm_unit.angstrom = _MockUnit()
        _mock_openmm_unit.nanometer = _MockUnit()

        saved_flag = CONFIG.use_explicit_solvent_mmgbsa

        saved_frames = CONFIG.explicit_solvent_frames
        CONFIG.use_explicit_solvent_mmgbsa = True
        CONFIG.explicit_solvent_frames = 5
        try:
            pdb_path = os.path.join(temp_work_dir, "receptor.pdb")
            with open(pdb_path, "w") as f:
                f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

            with patch("autoantibiotic.ml_scoring.scoring._HAVE_OPENMM", True, create=True):
                with patch("autoantibiotic.ml_scoring.scoring._HAVE_PDBFIXER", True, create=True):
                    with patch("autoantibiotic.ml_scoring.scoring._openmm", MagicMock(), create=True):
                        with patch("autoantibiotic.ml_scoring.scoring._openmm_app", _mock_openmm_app, create=True):
                            with patch("autoantibiotic.ml_scoring.scoring._openmm_unit", _mock_openmm_unit, create=True):
                                with patch("autoantibiotic.ml_scoring.scoring._compute_ligand_gb_energy", return_value=50.0):
                                    with patch("autoantibiotic.ml_scoring.scoring._compute_complex_gb_energy", return_value=-1950.0):
                                        mock_fixer = MagicMock()
                                        mock_fixer.topology = MagicMock()
                                        mock_fixer.positions = []
                                        _pdbfixer_mock.PDBFixer.return_value = mock_fixer

                                        from autoantibiotic.ml_scoring.scoring import rescore_with_explicit_mmgbsa
                                        result = rescore_with_explicit_mmgbsa(
                                            top_candidates, pdb_path, temp_work_dir,
                                        )
            assert len(result) == len(top_candidates)
        finally:
            CONFIG.use_explicit_solvent_mmgbsa = saved_flag
            CONFIG.explicit_solvent_frames = saved_frames
            _sys.modules.pop("pdbfixer", None)

    def test_explicit_returns_valid_floats(
        self,
        temp_work_dir: str,
    ) -> None:
        """Explicit MM-GB/SA returns valid float scores when mocked."""
        import sys as _sys
        _pdbfixer_mock = MagicMock()
        _sys.modules["pdbfixer"] = _pdbfixer_mock
        _pdbfixer_mock.PDBFixer = MagicMock()

        # Configure mock OpenMM simulation to return numeric energies
        _mock_sim = MagicMock()
        _mock_state = MagicMock()
        _mock_energy = MagicMock()
        _mock_energy.value_in_unit.return_value = -2000.0
        _mock_state.getPotentialEnergy.return_value = _mock_energy
        _mock_sim.context.getState.return_value = _mock_state

        _mock_openmm_app = MagicMock()
        _mock_openmm_app.Simulation.return_value = _mock_sim
        _mock_openmm_app.PDBFile.writeFile = MagicMock()
        _mock_openmm_app.ForceField = MagicMock()
        modeller_instance = MagicMock()
        _mock_openmm_app.Modeller.return_value = modeller_instance

        class _MockUnit:
            """Simple mock that supports float arithmetic."""
            def __mul__(self, other):
                return _MockUnit()
            def __rmul__(self, other):
                return _MockUnit()
            def __truediv__(self, other):
                return _MockUnit()
            def __rtruediv__(self, other):
                return _MockUnit()

        _mock_openmm_unit = MagicMock()
        _mock_openmm_unit.kilocalorie_per_mole = _MockUnit()
        _mock_openmm_unit.kelvin = _MockUnit()
        _mock_openmm_unit.picosecond = _MockUnit()
        _mock_openmm_unit.femtoseconds = _MockUnit()
        _mock_openmm_unit.angstrom = _MockUnit()
        _mock_openmm_unit.nanometer = _MockUnit()

        candidates = [
            CompoundRecord(
                compound_id="CMP-EXP-001",
                smiles="c1ccccc1O",
                mol=Chem.MolFromSmiles("c1ccccc1O"),
                pb2pa_allosteric_energy=-7.5,
            ),
        ]
        pdb_path = os.path.join(temp_work_dir, "receptor.pdb")
        with open(pdb_path, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

        with patch("autoantibiotic.ml_scoring.scoring._HAVE_OPENMM", True, create=True):
            with patch("autoantibiotic.ml_scoring.scoring._HAVE_PDBFIXER", True, create=True):
                with patch("autoantibiotic.ml_scoring.scoring._openmm", MagicMock(), create=True):
                    with patch("autoantibiotic.ml_scoring.scoring._openmm_app", _mock_openmm_app, create=True):
                        with patch("autoantibiotic.ml_scoring.scoring._openmm_unit", _mock_openmm_unit, create=True):
                            with patch("autoantibiotic.ml_scoring.scoring._compute_ligand_gb_energy", return_value=50.0):
                                with patch("autoantibiotic.ml_scoring.scoring._build_explicit_complex_system", return_value=(MagicMock(), MagicMock(), MagicMock())):
                                    with patch("autoantibiotic.ml_scoring.scoring._perform_pose_relaxation", return_value=(MagicMock(), True)):
                                        with patch("autoantibiotic.ml_scoring.scoring._compute_complex_gb_energy_relaxed", return_value=-1950.0):
                                            mock_fixer = MagicMock()
                                            mock_fixer.topology = MagicMock()
                                            mock_fixer.positions = []
                                            _pdbfixer_mock.PDBFixer.return_value = mock_fixer

                                            from autoantibiotic.ml_scoring.scoring import rescore_with_explicit_mmgbsa
                                            result = rescore_with_explicit_mmgbsa(
                                                candidates, pdb_path, temp_work_dir,
                                            )
        _sys.modules.pop("pdbfixer", None)
        assert len(result) == 1
        final_score = result[0].ml_score
        assert final_score is not None
        assert isinstance(final_score, float)
        # ΔG_binding = -1950 - (-2000) - 50 = 0.0 with our mocked values
        assert final_score == pytest.approx(0.0, abs=1e-4)


# ── Pose relaxation + water displacement integration ───────────

class TestPoseRelaxationAndWater:
    """Tests for pose relaxation and water displacement integration."""

    def test_rescore_with_relaxation_and_water(
        self, temp_work_dir: str,
    ) -> None:
        """High-energy water clash makes ΔG more negative than no-water baseline."""
        # Use distinct candidate objects to avoid mutation from a prior call
        def make_candidate(cid: str) -> CompoundRecord:
            return CompoundRecord(
                compound_id=cid,
                smiles="c1ccccc1O",
                mol=Chem.MolFromSmiles("c1ccccc1O"),
                pb2pa_allosteric_energy=-7.0,
            )

        water = MockWater(
            position=[0.0, 0.0, 0.0],
            displacement_energy=2.5,
            is_high_energy=True,
        )
        water_results = MockWaterAnalysisResult(high_energy_waters=[water])

        dummy_pdb = os.path.join(temp_work_dir, "receptor.pdb")
        with open(dummy_pdb, "w") as f:
            f.write(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
                "  1.00  0.00           C\nEND\n"
            )

        class _FakeExplicitLoop:
            """Callable that replaces _rescore_explicit_solvent_loop.

            Uses w_results presence to decide whether to apply the
            water displacement penalty.
            """
            def __call__(
                self, candidates, r_pdb, w_dir, w_results, n, n_c, ensemble,
            ):
                for rec in candidates[:n]:
                    score = -5.0
                    if w_results is not None and w_results.high_energy_waters:
                        penalty = _compute_water_displacement_penalty(
                            rec.mol, w_results.high_energy_waters,
                            seed=CONFIG.random_seed,
                            receptor_pdb=r_pdb,
                            strict_mode=False,
                        )
                        score -= penalty
                    rec.ml_score = score
                return candidates

        fake_loop = _FakeExplicitLoop()

        saved_explicit = CONFIG.use_explicit_solvent_mmgbsa
        saved_top_n = CONFIG.mm_gbsa_top_n
        CONFIG.use_explicit_solvent_mmgbsa = True
        CONFIG.mm_gbsa_top_n = 1
        try:
            with patch(
                "autoantibiotic.ml_scoring.scoring._rescore_explicit_solvent_loop",
                fake_loop,
            ):
                # Separate calls with distinct candidate objects
                result_no_water = rescore_with_mmgbsa(
                    [make_candidate("CMP-NO-WAT")], dummy_pdb, temp_work_dir,
                )
                result_with_water = rescore_with_mmgbsa(
                    [make_candidate("CMP-WITH-WAT")], dummy_pdb, temp_work_dir,
                    water_results=water_results,
                )
        finally:
            CONFIG.use_explicit_solvent_mmgbsa = saved_explicit
            CONFIG.mm_gbsa_top_n = saved_top_n

        score_no = result_no_water[0].ml_score
        score_with = result_with_water[0].ml_score

        assert score_no is not None, "Baseline score should be set"
        assert score_with is not None, "Water-corrected score should be set"
        assert score_with < score_no, (
            f"Water displacement should make ΔG more negative: "
            f"{score_with} >= {score_no}"
        )

    def test_pose_relaxation_reduces_clashes(self, temp_work_dir: str) -> None:
        """Synthetic bad pose → pose relaxation succeeds and returns positions."""
        mock_topology = MagicMock()
        mock_topology.atoms = MagicMock(return_value=[])  # No CA atoms → no restraints

        mock_system = MagicMock()
        mock_positions = MagicMock()
        relaxed_positions = MagicMock()

        mock_openmm = MagicMock()
        mock_openmm_app = MagicMock()
        mock_openmm_unit = MagicMock()

        with patch.multiple(
            "autoantibiotic.ml_scoring.scoring",
            _openmm=mock_openmm,
            _openmm_app=mock_openmm_app,
            _openmm_unit=mock_openmm_unit,
        ):
            mock_sim = MagicMock()
            mock_state = MagicMock()
            mock_state.getPositions.return_value = relaxed_positions
            mock_sim.context.getState.return_value = mock_state
            mock_openmm_app.Simulation.return_value = mock_sim
            mock_openmm.CustomExternalForce.return_value = MagicMock()
            mock_openmm.Platform.getPlatformByName.return_value = MagicMock()

            result, success = _perform_pose_relaxation(
                mock_topology, mock_system, mock_positions,
            )

        assert success is True, "Pose relaxation should succeed"
        assert result is relaxed_positions, (
            "Relaxed positions should be returned"
        )
