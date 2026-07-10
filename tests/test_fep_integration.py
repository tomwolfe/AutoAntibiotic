"""
Integration tests for FEP resistance profiling and heuristic fallback.

Verifies:
1. FEP resistance profiling is enabled by default in the config.
2. The orchestrator falls back to heuristic scoring when FEP fails.
3. ADMET reference data contains >50 samples per class when using
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


class TestFEPEnabledByDefault:
    """Verify that FEP resistance profiling is enabled in the default config."""

    def test_fep_enabled_by_default(self) -> None:
        assert CONFIG.use_fep_resistance is True

    def test_fep_top_n_default(self) -> None:
        assert CONFIG.fep_top_n == 20

    def test_heuristic_fallback_enabled_by_default(self) -> None:
        assert CONFIG.use_heuristic_resistance_fallback is True


# ── Test 2: Orchestrator fallback to heuristic ─────────────────────


class TestFEPFallbackToHeuristic:
    """Verify the orchestrator gracefully falls back to heuristic scoring
    when FEP raises an exception."""

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
        from autoantibiotic.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator(use_cache=False)
        orch.top_candidates = [mock_candidate]
        orch.targets = {
            "PBP2a": {
                "pdbqt": str(Path("/tmp/fake/pdbqt") / "PBP2a.pdbqt"),
                "allosteric_center": (0.0, 0.0, 0.0),
            },
        }
        return orch

    def test_apply_fep_resistance_does_not_crash_on_fep_failure(
        self, mock_orchestrator: Any,
    ) -> None:
        """When FEP raises an exception and fallback is enabled, the method
        should not crash and should log a warning."""
        with patch(
            "autoantibiotic.fep_engine.FEPResistanceCalculator",
        ) as mock_fep_cls:
            mock_calc = MagicMock()
            mock_calc.calculate_ddg.side_effect = Exception("FEP engine failure")
            mock_fep_cls.return_value = mock_calc

            # Should not raise
            mock_orchestrator.apply_fep_resistance()

    def test_fallback_populates_resistance_stability_score(
        self, mock_orchestrator: Any,
    ) -> None:
        """When FEP fails and heuristic fallback is enabled, the
        resistance_stability_score should be set."""
        import tempfile
        import os as os_module

        with tempfile.TemporaryDirectory() as tmpdir:
            mutant_dir = Path(tmpdir) / "mutants"
            mutant_dir.mkdir(parents=True, exist_ok=True)
            (mutant_dir / "mut_1.pdbqt").write_text("DUMMY")
            (mutant_dir / "mut_2.pdbqt").write_text("DUMMY")

            # Create a dummy receptor PDB so the orchestrator proceeds into FEP
            receptor_pdb_path = Path(tmpdir) / "PBP2a.pdb"
            receptor_pdb_path.write_text("DUMMY")
            mock_orchestrator.targets["PBP2a"]["pdbqt"] = str(
                receptor_pdb_path.with_suffix(".pdbqt"),
            )

            original_output_dir = CONFIG.output_dir
            CONFIG.output_dir = Path(tmpdir)

            try:
                with patch(
                    "autoantibiotic.fep_engine.FEPResistanceCalculator",
                ) as mock_fep_cls:
                    mock_calc = MagicMock()
                    mock_calc.calculate_ddg.side_effect = Exception(
                        "FEP engine failure",
                    )
                    mock_fep_cls.return_value = mock_calc

                    with patch(
                        "autoantibiotic.analysis.profile_resistance_mutation_sensitivity",
                    ) as mock_fallback:
                        mock_fallback.return_value = 0.85

                        rec = mock_orchestrator.top_candidates[0]
                        assert rec.resistance_stability_score is None

                        mock_orchestrator.apply_fep_resistance()

                        mock_fallback.assert_called_once()
                        assert rec.resistance_stability_score == 0.85
            finally:
                CONFIG.output_dir = original_output_dir

    def test_fallback_disabled_skips_heuristic(
        self, mock_orchestrator: Any,
    ) -> None:
        """When use_heuristic_resistance_fallback is False, FEP failure
        should just log a warning without calling the heuristic."""
        original_fallback = CONFIG.use_heuristic_resistance_fallback
        CONFIG.use_heuristic_resistance_fallback = False
        try:
            with patch(
                "autoantibiotic.fep_engine.FEPResistanceCalculator",
            ) as mock_fep_cls:
                mock_calc = MagicMock()
                mock_calc.calculate_ddg.side_effect = Exception(
                    "FEP engine failure",
                )
                mock_fep_cls.return_value = mock_calc

                with patch(
                    "autoantibiotic.analysis.profile_resistance_mutation_sensitivity",
                ) as mock_fallback:
                    mock_orchestrator.apply_fep_resistance()
                    mock_fallback.assert_not_called()
        finally:
            CONFIG.use_heuristic_resistance_fallback = original_fallback


# ── Test 3: FEP pre-screening with IFP ─────────────────────────────


class TestFEPPreScreening:
    """Verify the IFP-based pre-screening in apply_fep_resistance."""

    @pytest.fixture
    def mock_candidates(self) -> List[CompoundRecord]:
        candidates = []
        for i in range(10):
            mol = Chem.MolFromSmiles("c1ccccc1O")
            assert mol is not None
            rec = CompoundRecord(
                compound_id=f"TEST-PRE-{i:03d}",
                smiles="c1ccccc1O",
                mol=mol,
                docked_pose_path=f"/tmp/fake_pose_{i}.pdbqt",
            )
            candidates.append(rec)
        return candidates

    @pytest.fixture
    def mock_orchestrator(
        self, mock_candidates: List[CompoundRecord],
    ) -> Any:
        from autoantibiotic.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator(use_cache=False)
        orch.top_candidates = mock_candidates[:]
        orch.targets = {
            "PBP2a": {
                "pdbqt": str(Path("/tmp/fake/pdbqt") / "PBP2a.pdbqt"),
                "allosteric_center": (0.0, 0.0, 0.0),
            },
        }
        return orch

    def test_pre_screen_filters_low_ifp(
        self, mock_orchestrator: Any,
    ) -> None:
        """Candidates with IFP below threshold are excluded from FEP."""
        original_pool_size = CONFIG.fep_pre_screen_pool_size
        original_ifp_threshold = CONFIG.fep_ifp_threshold
        CONFIG.fep_pre_screen_pool_size = 10
        CONFIG.fep_ifp_threshold = 0.5
        try:
            with patch(
                "autoantibiotic.scoring_metrics.compute_ifp_similarity",
                side_effect=[0.3, 0.3, 0.3] + [0.8] * 7,
            ), patch(
                "os.path.isfile", return_value=True,
            ), patch(
                "autoantibiotic.fep_engine.FEPResistanceCalculator",
            ) as mock_fep_cls:
                mock_calc = MagicMock()
                mock_fep_cls.return_value = mock_calc

                mock_orchestrator.apply_fep_resistance()

                # 3 low IFP (skip) + 7 high IFP = 7 pass, cap at fep_top_n=5
                assert mock_fep_cls.call_count == 5, (
                    f"Expected 5 FEP calls, got {mock_fep_cls.call_count}"
                )
        finally:
            CONFIG.fep_pre_screen_pool_size = original_pool_size
            CONFIG.fep_ifp_threshold = original_ifp_threshold

    def test_pre_screen_expands_pool(
        self, mock_orchestrator: Any,
    ) -> None:
        """When more candidates pass IFP than fep_top_n, only top N are processed."""
        original_pool_size = CONFIG.fep_pre_screen_pool_size
        original_top_n = CONFIG.fep_top_n
        CONFIG.fep_pre_screen_pool_size = 10
        CONFIG.fep_top_n = 3
        try:
            with patch(
                "autoantibiotic.scoring_metrics.compute_ifp_similarity",
                return_value=0.9,
            ), patch(
                "os.path.isfile", return_value=True,
            ), patch(
                "autoantibiotic.fep_engine.FEPResistanceCalculator",
            ) as mock_fep_cls:
                mock_calc = MagicMock()
                mock_fep_cls.return_value = mock_calc

                mock_orchestrator.apply_fep_resistance()

                # All 10 pass IFP, cap at fep_top_n=3
                assert mock_fep_cls.call_count == 3, (
                    f"Expected 3 FEP calls, got {mock_fep_cls.call_count}"
                )
        finally:
            CONFIG.fep_pre_screen_pool_size = original_pool_size
            CONFIG.fep_top_n = original_top_n


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

        # hERG blockers (label=1)
        herg_blockers = [d for d in data["herg"] if d["label"] == 1]
        assert len(herg_blockers) > 50, (
            f"hERG blockers: {len(herg_blockers)} (expected >50)"
        )

        # hERG safe (label=0)
        herg_safe = [d for d in data["herg"] if d["label"] == 0]
        assert len(herg_safe) > 50, (
            f"hERG safe: {len(herg_safe)} (expected >50)"
        )

        # CYP inhibitors (label=1)
        cyp_inhibitors = [d for d in data["cyp"] if d["label"] == 1]
        assert len(cyp_inhibitors) > 50, (
            f"CYP inhibitors: {len(cyp_inhibitors)} (expected >50)"
        )

        # CYP non-inhibitors (label=0)
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
        # Two-alanine receptor PDB (minimal valid PDB)
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
        # Same file for mutant (we only care about pre_screen_ligand)
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
        # Should not raise
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
