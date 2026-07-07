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
        assert CONFIG.fep_top_n == 5

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


# ── Test 3: ADMET expanded data ────────────────────────────────────


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
