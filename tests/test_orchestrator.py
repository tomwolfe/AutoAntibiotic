"""Tests for PipelineOrchestrator benchmark check and engine integration."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from autoantibiotic.config import CONFIG, PipelineConfig
from autoantibiotic.orchestrator import PipelineOrchestrator
from autoantibiotic.pipeline_context import PipelineContext
from autoantibiotic.models import CompoundRecord


class TestBenchmarkCheck:
    """``_run_benchmark_check`` logs EF1% and ROC-AUC when benchmark_mode is True."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.saved_mode = CONFIG.benchmark_mode
        CONFIG.benchmark_mode = True
        yield
        CONFIG.benchmark_mode = self.saved_mode

    def test_benchmark_check_skipped_when_disabled(self) -> None:
        CONFIG.benchmark_mode = False
        orch = PipelineOrchestrator(config=CONFIG)
        ctx = PipelineContext()
        ctx.docked_candidates = [
            CompoundRecord(compound_id="C1", smiles="c1ccccc1"),
        ]
        result = orch._run_benchmark_check(ctx)
        assert result is ctx

    def test_benchmark_check_skipped_no_candidates(self) -> None:
        orch = PipelineOrchestrator(config=CONFIG)
        ctx = PipelineContext()
        result = orch._run_benchmark_check(ctx)
        assert result is ctx

    def test_benchmark_check_runs_and_logs(self, caplog) -> None:
        import logging
        caplog.set_level(logging.INFO)

        orch = PipelineOrchestrator(config=CONFIG)

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

        ctx = PipelineContext()
        ctx.docked_candidates = recs

        with patch("benchmarks.reference_data.get_actives_smiles", return_value=active_smiles):
            with patch("benchmarks.reference_data.get_inactives_smiles", return_value=inactive_smiles):
                result = orch._run_benchmark_check(ctx)

        assert result is ctx
        assert any("EF1%" in rec.message for rec in caplog.records)
        assert any("ROC-AUC" in rec.message for rec in caplog.records)


class TestEngineIntegration:
    """Orchestrator creates an engine via get_engine in _screen_candidates."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.saved = (CONFIG.dry_run, CONFIG.use_gnina)
        CONFIG.dry_run = True
        CONFIG.use_gnina = False
        yield
        CONFIG.dry_run, CONFIG.use_gnina = self.saved

    def test_screen_candidates_uses_engine(self) -> None:
        orch = PipelineOrchestrator(config=CONFIG)
        orch.targets = {
            "PBP2a": {
                "allosteric_center": np.array([0.0, 0.0, 0.0]),
                "active_center": np.array([0.0, 0.0, 0.0]),
                "pdbqt": "rec.pdbqt",
            },
        }
        orch.deps = {"USE_VINA": True}
        orch.use_cache = False
        orch.audit = None

        rec = CompoundRecord(
            compound_id="TEST",
            smiles="c1ccccc1",
            mol=None,
        )
        ctx = PipelineContext()
        ctx.filtered_library = [rec]
        ctx.library = [rec]

        # screen_library is called internally — mock it to return records quickly
        with patch("autoantibiotic.orchestrator.screen_library", return_value=[rec]):
            result = orch._screen_candidates(ctx)

        assert result.docked_candidates == [rec]
