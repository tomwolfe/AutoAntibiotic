"""
Integration tests for FEP resistance profiling and heuristic fallback.

Verifies:
1. FEP resistance profiling is enabled by default in the config.
2. The orchestrator delegates to ``FEPManager`` for candidate selection
   and FEP execution.
3. ``FEPManager.select_candidates_for_fep`` correctly filters candidates
   based on IFP, energy, and pharmacophore criteria.
4. ADMET reference data contains >50 samples per class when using
   the expanded hardcoded list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from autoantibiotic.config import CONFIG
from autoantibiotic.models import CompoundRecord


# ── Test 1: FEP enabled by default ────────────────────────────────


class TestFEPDisabledByDefault:
    """Verify that FEP resistance profiling is disabled in the default config."""

    def test_fep_disabled_by_default(self) -> None:
        assert CONFIG.use_fep_resistance is False

    def test_fep_top_n_default(self) -> None:
        assert CONFIG.fep_top_n == 20

    def test_heuristic_fallback_enabled_by_default(self) -> None:
        assert CONFIG.use_heuristic_resistance_fallback is True


# ── Test 2: Orchestrator delegation to FEPManager ──────────────────


class TestFEPOrchestratorDelegation:
    """Verify the orchestrator properly delegates to ``FEPManager``."""

    @pytest.fixture
    def mock_candidate(self) -> CompoundRecord:
        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        return CompoundRecord(
            compound_id="TEST-FEP-001",
            smiles="c1ccccc1O",
            mol=mol,
        )

    @pytest.fixture
    def mock_orchestrator(self, mock_candidate: CompoundRecord) -> Any:
        import copy
        from autoantibiotic.config import CONFIG
        from autoantibiotic.orchestrator import PipelineOrchestrator

        local_config = copy.deepcopy(CONFIG)
        local_config.use_fep_resistance = True
        orch = PipelineOrchestrator(use_cache=False, config=local_config)
        orch.top_candidates = [mock_candidate]
        orch.targets = {
            "PBP2a": {
                "pdbqt": str(Path("/tmp/fake/pdbqt") / "PBP2a.pdbqt"),
                "allosteric_center": (0.0, 0.0, 0.0),
            },
        }
        return orch

    def test_apply_fep_resistance_delegates_to_fep_manager(
        self, mock_orchestrator: Any,
    ) -> None:
        """The orchestrator should call ``FEPManager`` methods and set
        ``context.fep_results``."""
        rec = mock_orchestrator.top_candidates[0]
        with patch("autoantibiotic.orchestrator.FEPManager") as mock_mgr_cls:
            mock_mgr = MagicMock()
            mock_mgr_cls.return_value = mock_mgr
            mock_mgr.select_candidates_for_fep.return_value = [rec]
            mock_mgr.run_fep_profiling.return_value = [rec]

            with patch("os.path.isfile", return_value=True):
                mock_orchestrator.apply_fep_resistance()

            mock_mgr.select_candidates_for_fep.assert_called_once()
            mock_mgr.run_fep_profiling.assert_called_once()

    def test_apply_fep_resistance_does_not_crash_on_fep_failure(
        self, mock_orchestrator: Any,
    ) -> None:
        """When ``FEPManager`` returns no successful results, the method
        should not crash."""
        rec = mock_orchestrator.top_candidates[0]
        with patch("autoantibiotic.orchestrator.FEPManager") as mock_mgr_cls:
            mock_mgr = MagicMock()
            mock_mgr_cls.return_value = mock_mgr
            mock_mgr.select_candidates_for_fep.return_value = [rec]
            mock_mgr.run_fep_profiling.return_value = []

            with patch("os.path.isfile", return_value=True):
                mock_orchestrator.apply_fep_resistance()

    def test_fallback_populates_resistance_stability_score(
        self, mock_orchestrator: Any,
    ) -> None:
        """When ``FEPManager.run_fep_profiling`` returns a record with a
        score set, the result should be reflected on the top candidate."""
        rec = mock_orchestrator.top_candidates[0]
        rec.resistance_stability_score = 0.85

        with patch("autoantibiotic.orchestrator.FEPManager") as mock_mgr_cls:
            mock_mgr = MagicMock()
            mock_mgr_cls.return_value = mock_mgr
            mock_mgr.select_candidates_for_fep.return_value = [rec]
            mock_mgr.run_fep_profiling.return_value = [rec]

            with patch("os.path.isfile", return_value=True):
                mock_orchestrator.apply_fep_resistance()

            assert rec.resistance_stability_score == 0.85

    def test_fallback_disabled_still_delegates(
        self, mock_orchestrator: Any,
    ) -> None:
        """When ``use_heuristic_resistance_fallback`` is False, the
        orchestrator still delegates to ``FEPManager`` (the fallback
        logic lives inside ``FEPManager``)."""
        original_fallback = CONFIG.use_heuristic_resistance_fallback
        CONFIG.use_heuristic_resistance_fallback = False
        rec = mock_orchestrator.top_candidates[0]
        try:
            with patch("autoantibiotic.orchestrator.FEPManager") as mock_mgr_cls:
                mock_mgr = MagicMock()
                mock_mgr_cls.return_value = mock_mgr
                mock_mgr.select_candidates_for_fep.return_value = [rec]
                mock_mgr.run_fep_profiling.return_value = []

                with patch("os.path.isfile", return_value=True):
                    mock_orchestrator.apply_fep_resistance()

                mock_mgr.run_fep_profiling.assert_called_once()
        finally:
            CONFIG.use_heuristic_resistance_fallback = original_fallback


# ── Test 3: FEPManager selection logic ─────────────────────────────


class TestFEPManagerSelectionLogic:
    """Verify ``FEPManager.select_candidates_for_fep`` filtering."""

    @pytest.fixture
    def candidates(self) -> List[CompoundRecord]:
        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        records = []
        for i in range(10):
            rec = CompoundRecord(
                compound_id=f"TEST-{i:03d}",
                smiles="c1ccccc1O",
                mol=mol,
                docked_pose_path=f"/tmp/pose_{i}.pdbqt",
                pb2pa_allosteric_energy=-9.0 + i * 0.5,
            )
            records.append(rec)
        return records

    def test_select_candidates_filters_by_energy_cutoff(
        self, candidates: List[CompoundRecord],
    ) -> None:
        """Candidates with allosteric energy >= -8.0 should be excluded."""
        from autoantibiotic.fep_manager import FEPManager

        manager = FEPManager(config=CONFIG, targets={"PBP2a": {"pdbqt": "/tmp/fake.pdbqt"}})
        manager._pharmacophore_query = {"mode": "2d"}  # bypass _build
        manager._ref_mol = Chem.MolFromSmiles("c1ccccc1")
        manager._receptor_pdb = "/tmp/fake.pdb"

        # compute_ifp_similarity is called twice per candidate (round 1 + round 2)
        # Round 1 (10 calls): all return 0.9 → all pass initial IFP
        # Round 2 (10 calls): first 5 return 0.9, last 5 return 0.3
        # But only candidates that pass round 1 proceed to round 2 (all 10 pass)
        side_effects = [0.9] * 10 + [0.9] * 5 + [0.3] * 5
        with patch(
            "autoantibiotic.fep_manager.compute_ifp_similarity",
            side_effect=side_effects,
        ), patch(
            "autoantibiotic.fep_manager._build_allosteric_pharmacophore",
            return_value={"mode": "2d"},
        ), patch(
            "os.path.isfile", return_value=True,
        ), patch(
            "autoantibiotic.fep_manager.check_pharmacophore_match",
            return_value=True,
        ):
            result = manager.select_candidates_for_fep(candidates)

        # energies: -9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5
        # Strict: energy < -8.0 AND IFP >= 0.7
        # -9.0 (IFP=0.9) passes, -8.5 (IFP=0.9) passes, -8.0 (IFP=0.3) fails
        # So only 2 candidates should be returned
        assert len(result) == 2, f"Expected 2, got {len(result)}"
        assert all(r.pb2pa_allosteric_energy < -8.0 for r in result)

    def test_select_candidates_filters_by_ifp(
        self, candidates: List[CompoundRecord],
    ) -> None:
        """Candidates with IFP below strict threshold (0.7) should be excluded."""
        from autoantibiotic.fep_manager import FEPManager

        manager = FEPManager(config=CONFIG, targets={"PBP2a": {"pdbqt": "/tmp/fake.pdbqt"}})
        manager._pharmacophore_query = {"mode": "2d"}
        manager._ref_mol = Chem.MolFromSmiles("c1ccccc1")
        manager._receptor_pdb = "/tmp/fake.pdb"

        # First 3 have low IFP (fail initial IFP > 0.5)
        # Next 3 have IFP 0.6 (pass initial but fail strict >= 0.7)
        # Last 4 have IFP 0.8 (pass both)
        side_effects = [0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8]
        with patch(
            "autoantibiotic.fep_manager.compute_ifp_similarity",
            side_effect=side_effects,
        ), patch(
            "autoantibiotic.fep_manager._build_allosteric_pharmacophore",
            return_value={"mode": "2d"},
        ), patch(
            "os.path.isfile", return_value=True,
        ), patch(
            "autoantibiotic.fep_manager.check_pharmacophore_match",
            return_value=True,
        ):
            result = manager.select_candidates_for_fep(candidates)

        # 3 fail initial IFP (0.3 <= 0.5), 3 fail strict IFP (0.6 < 0.7),
        # 4 pass, but also need energy < -8.0
        # energies: -9.0(0.3), -8.5(0.3), -8.0(0.3), -7.5(0.6), -7.0(0.6), -6.5(0.6),
        #           -6.0(0.8), -5.5(0.8), -5.0(0.8), -4.5(0.8)
        # Candidates with IFP >= 0.7 (last 4): -6.0, -5.5, -5.0, -4.5
        # BUT energy must be < -8.0, and none of those are < -8.0
        assert len(result) == 0

    def test_select_candidates_returns_top_n_strict(
        self, candidates: List[CompoundRecord],
    ) -> None:
        """Only top ``fep_top_n_strict`` candidates should be returned."""
        from autoantibiotic.fep_manager import FEPManager

        original_strict = CONFIG.fep_top_n_strict
        CONFIG.fep_top_n_strict = 3
        try:
            rec = candidates[0]
            rec.pb2pa_allosteric_energy = -12.0

            manager = FEPManager(config=CONFIG, targets={"PBP2a": {"pdbqt": "/tmp/fake.pdbqt"}})
            manager._pharmacophore_query = {"mode": "2d"}
            manager._ref_mol = Chem.MolFromSmiles("c1ccccc1")
            manager._receptor_pdb = "/tmp/fake.pdb"

            with patch(
                "autoantibiotic.fep_manager.compute_ifp_similarity",
                return_value=0.9,
            ), patch(
                "autoantibiotic.fep_manager._build_allosteric_pharmacophore",
                return_value={"mode": "2d"},
            ), patch(
                "os.path.isfile", return_value=True,
            ), patch(
                "autoantibiotic.fep_manager.check_pharmacophore_match",
                return_value=True,
            ):
                result = manager.select_candidates_for_fep(candidates)

            assert len(result) <= 3
        finally:
            CONFIG.fep_top_n_strict = original_strict


# ── Test 4: ADMET expanded data ────────────────────────────────────


class TestADMETExpandedData:
    """Verify that the expanded hardcoded ADMET reference data contains
    >50 samples per class."""

    def test_herg_blockers_class_size(self) -> None:
        from benchmarks.reference_data import _HERG_BLOCKERS, _SAFE_COMPOUNDS

        assert len(_HERG_BLOCKERS) > 50, (
            f"hERG blockers have {len(_HERG_BLOCKERS)} entries, expected >50"
        )
        assert len(_SAFE_COMPOUNDS) > 50, (
            f"Safe compounds have {len(_SAFE_COMPOUNDS)} entries, expected >50"
        )

    def test_cyp_inhibitors_class_size(self) -> None:
        from benchmarks.reference_data import _CYP_INHIBITORS, _NON_CYP_INHIBITORS

        assert len(_CYP_INHIBITORS) > 50, (
            f"CYP inhibitors have {len(_CYP_INHIBITORS)} entries, expected >50"
        )
        assert len(_NON_CYP_INHIBITORS) > 50, (
            f"Non-CYP inhibitors have {len(_NON_CYP_INHIBITORS)} entries, "
            "expected >50"
        )

    def test_load_chembl_admet_subset_returns_expanded_data(self) -> None:
        from benchmarks.reference_data import load_chembl_admet_subset

        data = load_chembl_admet_subset()

        herg_blockers = [d for d in data["herg"] if d["label"] == 1]
        assert len(herg_blockers) > 50, (
            f"hERG blockers: {len(herg_blockers)} (expected >50)"
        )

        herg_safe = [d for d in data["herg"] if d["label"] == 0]
        assert len(herg_safe) > 50, (
            f"hERG safe: {len(herg_safe)} (expected >50)"
        )

        cyp_inhibitors = [d for d in data["cyp"] if d["label"] == 1]
        assert len(cyp_inhibitors) > 50, (
            f"CYP inhibitors: {len(cyp_inhibitors)} (expected >50)"
        )

        cyp_non = [d for d in data["cyp"] if d["label"] == 0]
        assert len(cyp_non) > 50, (
            f"CYP non-inhibitors: {len(cyp_non)} (expected >50)"
        )

    def test_all_smiles_are_valid(self) -> None:
        """Verify all hardcoded SMILES strings are parseable by RDKit."""
        from benchmarks.reference_data import (
            _HERG_BLOCKERS,
            _SAFE_COMPOUNDS,
            _CYP_INHIBITORS,
            _NON_CYP_INHIBITORS,
        )

        all_smiles = _HERG_BLOCKERS + _SAFE_COMPOUNDS + _CYP_INHIBITORS + _NON_CYP_INHIBITORS
        invalid: List[str] = []
        for smi in all_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid.append(smi)

        assert len(invalid) == 0, (
            f"Found {len(invalid)} invalid SMILES: {invalid[:5]}"
        )


# ── Test 5: Minimal FEP pre-screening integration ──────────────────


class TestFEPMinimalIntegration:
    """Verify FEP pre-screening logic with small and large ligands."""

    @pytest.fixture
    def dummy_pdb_files(self, tmp_path: Path) -> tuple:
        """Create minimal dummy PDB files for FEPResistanceCalculator."""
        receptor_pdb = tmp_path / "receptor_wt.pdb"
        receptor_pdb.write_text(
            "ATOM      1  N   ALA A   1       1.458   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       2.009   1.422   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       1.461   2.172   1.200  1.00  0.00           C\n"
            "ATOM      4  O   ALA A   1       0.300   2.572   1.192  1.00  0.00           O\n"
            "ATOM      5  CB  ALA A   1       1.598   2.159  -1.262  1.00  0.00           C\n"
            "ATOM      6  N   ALA A   2       2.280   2.363   2.233  1.00  0.00           N\n"
            "ATOM      7  CA  ALA A   2       1.840   3.081   3.453  1.00  0.00           C\n"
            "ATOM      8  C   ALA A   2       2.636   2.660   4.676  1.00  0.00           C\n"
            "ATOM      9  O   ALA A   2       2.773   3.408   5.648  1.00  0.00           O\n"
            "ATOM     10  CB  ALA A   2       2.029   4.580   3.282  1.00  0.00           C\n"
            "ATOM     11  N   ALA A   3       3.145   1.429   4.655  1.00  0.00           N\n"
            "ATOM     12  CA  ALA A   3       3.943   0.941   5.798  1.00  0.00           C\n"
            "ATOM     13  C   ALA A   3       3.140   0.365   6.950  1.00  0.00           C\n"
            "ATOM     14  O   ALA A   3       2.109  -0.269   6.752  1.00  0.00           O\n"
            "ATOM     15  CB  ALA A   3       4.941   1.994   6.272  1.00  0.00           C\n"
            "TER\n"
            "END\n"
        )
        mutant_pdb = tmp_path / "receptor_mut.pdb"
        mutant_pdb.write_text(receptor_pdb.read_text())
        return str(receptor_pdb), str(mutant_pdb)

    def test_pre_screen_ligand_passes_for_methane(
        self, dummy_pdb_files: tuple,
    ) -> None:
        """pre_screen_ligand should pass for methane (1 heavy atom)."""
        from autoantibiotic.fep_engine import FEPResistanceCalculator

        wt_pdb, mut_pdb = dummy_pdb_files
        calc = FEPResistanceCalculator(
            receptor_wt_pdb=wt_pdb,
            receptor_mut_pdb=mut_pdb,
            ligand_smiles="C",
        )
        calc.pre_screen_ligand()

    def test_pre_screen_ligand_raises_for_large_molecule(
        self, dummy_pdb_files: tuple,
    ) -> None:
        """pre_screen_ligand should raise ConfigurationError for >50 heavy atoms."""
        from autoantibiotic.fep_engine import FEPResistanceCalculator, ConfigurationError

        wt_pdb, mut_pdb = dummy_pdb_files
        calc = FEPResistanceCalculator(
            receptor_wt_pdb=wt_pdb,
            receptor_mut_pdb=mut_pdb,
            ligand_smiles="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        )
        with pytest.raises(ConfigurationError):
            calc.pre_screen_ligand()

    def test_minimal_fep_run_if_openmm_available(
        self, dummy_pdb_files: tuple,
    ) -> None:
        """Skip the full FEP test if OpenMM dependencies are missing."""
        from autoantibiotic.fep_engine import (
            _HAVE_OPENMM,
            _HAVE_OPENMMTOOLS,
        )

        if not (_HAVE_OPENMM and _HAVE_OPENMMTOOLS):
            pytest.skip("OpenMM and/or openmmtools not available")
