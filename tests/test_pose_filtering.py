"""Tests for pose filtering via Interaction Fingerprints (IFP)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from autoantibiotic.config import PipelineConfig
from autoantibiotic.models import CompoundRecord
from autoantibiotic.orchestrator import PipelineOrchestrator


@pytest.fixture
def receptor_pdb() -> str:
    """Create a minimal receptor PDB file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write("ATOM      1  CA  SER A 403       1.000   1.000   1.000  1.00  0.00           C\nEND\n")
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def pose_pdbqt() -> str:
    """Create a minimal docked-pose PDBQT file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False)
    tmp.write("ROOT\nATOM      1  C   LIG     1       1.500   1.500   1.500  1.00  0.00           C\nENDROOT\n")
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


def _make_orchestrator(
    require_key_interactions: bool = True,
    top_candidates: list | None = None,
) -> PipelineOrchestrator:
    """Helper: build a PipelineOrchestrator with a minimal config."""
    cfg = PipelineConfig()
    cfg.require_key_interactions_for_rescoring = require_key_interactions
    cfg.dry_run = True

    orch = PipelineOrchestrator(config=cfg)
    orch.targets = {
        "PBP2a": {
            "pdbqt": "/tmp/dummy.pdbqt",
            "allosteric_center": np.array([0.0, 0.0, 0.0]),
            "active_center": np.array([0.0, 0.0, 0.0]),
        },
    }
    if top_candidates is not None:
        orch.top_candidates = top_candidates
    return orch


class TestFilterByKeyInteractions:
    """Unit tests for PipelineOrchestrator._filter_by_key_interactions."""

    def test_filter_removes_compound_without_interactions(
        self, receptor_pdb: str, pose_pdbqt: str,
    ) -> None:
        """Compounds with no key interactions are removed when flag is True."""
        rec = CompoundRecord(
            compound_id="TEST-001",
            smiles="c1ccccc1",
            docked_pose_path=pose_pdbqt,
        )
        orch = _make_orchestrator(
            require_key_interactions=True,
            top_candidates=[rec],
        )
        # Point receptor PDBQT to a real path so .pdb resolves to receptor_pdb
        orch.targets["PBP2a"]["pdbqt"] = receptor_pdb.replace(".pdb", ".pdbqt")
        # Create the dummy .pdbqt so replace works
        with open(orch.targets["PBP2a"]["pdbqt"], "w") as f:
            f.write("ROOT\nENDROOT\nTORSDOF 0\n")

        with patch(
            "autoantibiotic.orchestrator.check_key_interactions",
            return_value=False,
        ):
            orch._filter_by_key_interactions()

        assert len(orch.top_candidates) == 0

        # Cleanup
        os.unlink(orch.targets["PBP2a"]["pdbqt"])

    def test_filter_retains_compound_with_interactions(
        self, receptor_pdb: str, pose_pdbqt: str,
    ) -> None:
        """Compounds with key interactions are retained."""
        rec = CompoundRecord(
            compound_id="TEST-002",
            smiles="c1ccccc1O",
            docked_pose_path=pose_pdbqt,
        )
        orch = _make_orchestrator(
            require_key_interactions=True,
            top_candidates=[rec],
        )
        orch.targets["PBP2a"]["pdbqt"] = receptor_pdb.replace(".pdb", ".pdbqt")
        with open(orch.targets["PBP2a"]["pdbqt"], "w") as f:
            f.write("ROOT\nENDROOT\nTORSDOF 0\n")

        with patch(
            "autoantibiotic.orchestrator.check_key_interactions",
            return_value=True,
        ):
            orch._filter_by_key_interactions()

        assert len(orch.top_candidates) == 1
        assert orch.top_candidates[0].compound_id == "TEST-002"

        os.unlink(orch.targets["PBP2a"]["pdbqt"])

    def test_filter_bypassed_when_flag_false(
        self, receptor_pdb: str, pose_pdbqt: str,
    ) -> None:
        """Filter is skipped entirely when require_key_interactions_for_rescoring is False."""
        rec = CompoundRecord(
            compound_id="TEST-003",
            smiles="c1ccccc1",
            docked_pose_path=pose_pdbqt,
        )
        orch = _make_orchestrator(
            require_key_interactions=False,
            top_candidates=[rec],
        )
        orch.targets["PBP2a"]["pdbqt"] = receptor_pdb.replace(".pdb", ".pdbqt")
        with open(orch.targets["PBP2a"]["pdbqt"], "w") as f:
            f.write("ROOT\nENDROOT\nTORSDOF 0\n")

        # Even though we mock check_key_interactions to return False,
        # the filter should never call it.
        with patch(
            "autoantibiotic.orchestrator.check_key_interactions",
            return_value=False,
        ) as mocked:
            orch._filter_by_key_interactions()

        assert len(orch.top_candidates) == 1
        mocked.assert_not_called()

        os.unlink(orch.targets["PBP2a"]["pdbqt"])

    def test_fail_safe_keeps_compound_on_check_failure(
        self, receptor_pdb: str, pose_pdbqt: str,
    ) -> None:
        """If check_key_interactions raises, the compound is kept (fail-safe)."""
        rec = CompoundRecord(
            compound_id="TEST-004",
            smiles="c1ccccc1",
            docked_pose_path=pose_pdbqt,
        )
        orch = _make_orchestrator(
            require_key_interactions=True,
            top_candidates=[rec],
        )
        orch.targets["PBP2a"]["pdbqt"] = receptor_pdb.replace(".pdb", ".pdbqt")
        with open(orch.targets["PBP2a"]["pdbqt"], "w") as f:
            f.write("ROOT\nENDROOT\nTORSDOF 0\n")

        with patch(
            "autoantibiotic.orchestrator.check_key_interactions",
            side_effect=RuntimeError("Unexpected failure"),
        ):
            orch._filter_by_key_interactions()

        assert len(orch.top_candidates) == 1

        os.unlink(orch.targets["PBP2a"]["pdbqt"])

    def test_fail_safe_keeps_compound_when_pose_missing(self) -> None:
        """If docked_pose_path is None or missing, compound is kept (fail-safe)."""
        rec = CompoundRecord(
            compound_id="TEST-005",
            smiles="c1ccccc1",
            docked_pose_path=None,  # No pose file
        )
        orch = _make_orchestrator(
            require_key_interactions=True,
            top_candidates=[rec],
        )

        with patch(
            "autoantibiotic.orchestrator.check_key_interactions",
        ) as mocked:
            orch._filter_by_key_interactions()

        assert len(orch.top_candidates) == 1
        mocked.assert_not_called()

    def test_mixed_filtering(
        self, receptor_pdb: str, pose_pdbqt: str,
    ) -> None:
        """Only compounds with interactions survive when flag is True."""
        rec_with = CompoundRecord(
            compound_id="WITH-INT", smiles="c1ccccc1O",
            docked_pose_path=pose_pdbqt,
        )
        rec_without = CompoundRecord(
            compound_id="NO-INT", smiles="c1ccccc1",
            docked_pose_path=pose_pdbqt,
        )
        orch = _make_orchestrator(
            require_key_interactions=True,
            top_candidates=[rec_with, rec_without],
        )
        orch.targets["PBP2a"]["pdbqt"] = receptor_pdb.replace(".pdb", ".pdbqt")
        with open(orch.targets["PBP2a"]["pdbqt"], "w") as f:
            f.write("ROOT\nENDROOT\nTORSDOF 0\n")

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call (rec_with) → True, second call (rec_without) → False
            return call_count == 1

        with patch(
            "autoantibiotic.orchestrator.check_key_interactions",
            side_effect=_side_effect,
        ):
            orch._filter_by_key_interactions()

        assert len(orch.top_candidates) == 1
        assert orch.top_candidates[0].compound_id == "WITH-INT"

        os.unlink(orch.targets["PBP2a"]["pdbqt"])
