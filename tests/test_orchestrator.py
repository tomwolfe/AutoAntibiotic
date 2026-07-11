"""Tests for PipelineOrchestrator benchmark check and engine integration."""

from unittest.mock import patch

import numpy as np
import pytest

from autoantibiotic.config import CONFIG
from autoantibiotic.models import CompoundRecord
from autoantibiotic.phases import DockingHandler


class TestBenchmarkCheck:
    """``DockingHandler._run_benchmark_check`` logs EF1% and ROC-AUC when
    benchmark_mode is True."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.saved_mode = CONFIG.benchmark_mode
        CONFIG.benchmark_mode = True
        yield
        CONFIG.benchmark_mode = self.saved_mode

    def test_benchmark_check_skipped_when_disabled(self) -> None:
        CONFIG.benchmark_mode = False
        handler = DockingHandler()
        state = {
            "docked_candidates": [
                CompoundRecord(compound_id="C1", smiles="c1ccccc1"),
            ],
        }
        result = handler._run_benchmark_check(state, CONFIG)
        assert result is state

    def test_benchmark_check_skipped_no_candidates(self) -> None:
        handler = DockingHandler()
        state = {"docked_candidates": []}
        result = handler._run_benchmark_check(state, CONFIG)
        assert result is state

    def test_benchmark_check_runs_and_logs(self, caplog) -> None:
        import logging
        caplog.set_level(logging.INFO)

        handler = DockingHandler()

        active_smiles = ["c1ccccc1", "c1ccccc1O", "c1ccccc1Cl", "c1ccccc1F",
                         "c1ccccc1Br", "c1ccccc1I"]
        inactive_smiles = ["CCO", "CCN", "CCCO", "CCCN", "CCCCO", "CCCCN",
                           "C1CCCCC1", "C1CCNCC1", "CCCCCC", "CCCCCCC"]

        recs = []
        for smi in active_smiles:
            rec = CompoundRecord(compound_id=f"ACTIVE_{smi}", smiles=smi)
            rec.pb2pa_allosteric_energy = -8.0
            recs.append(rec)
        for smi in inactive_smiles:
            rec = CompoundRecord(compound_id=f"INACTIVE_{smi}", smiles=smi)
            rec.pb2pa_allosteric_energy = -4.0
            recs.append(rec)

        state = {"docked_candidates": recs}

        with patch("benchmarks.reference_data.get_actives_smiles", return_value=active_smiles):
            with patch("benchmarks.reference_data.get_inactives_smiles", return_value=inactive_smiles):
                result = handler._run_benchmark_check(state, CONFIG)

        assert result is state
        assert any("EF1%" in rec.message for rec in caplog.records)
        assert any("ROC-AUC" in rec.message for rec in caplog.records)


class TestEngineIntegration:
    """DockingHandler creates an engine via get_engine in execute."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.saved = (CONFIG.dry_run, CONFIG.use_gnina)
        CONFIG.dry_run = True
        CONFIG.use_gnina = False
        yield
        CONFIG.dry_run, CONFIG.use_gnina = self.saved

    def test_screen_candidates_uses_engine(self) -> None:
        handler = DockingHandler()

        rec = CompoundRecord(
            compound_id="TEST",
            smiles="c1ccccc1",
            mol=None,
        )
        state = {
            "targets": {
                "PBP2a": {
                    "allosteric_center": np.array([0.0, 0.0, 0.0]),
                    "active_center": np.array([0.0, 0.0, 0.0]),
                    "pdbqt": "rec.pdbqt",
                },
            },
            "deps": {"USE_VINA": True},
            "cache": {},
            "use_cache": False,
            "water_results": None,
            "audit": None,
            "filtered_library": [rec],
        }

        with patch("autoantibiotic.phases.docking.screen_library", return_value=[rec]):
            result = handler.execute(state, CONFIG)

        assert result["docked_candidates"] == [rec]
