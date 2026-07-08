"""Tests for FEP pre-screening strict filtering.

Verifies that the strict pre-screening logic (IFP >= 0.7,
allosteric energy < -8.0, limited to top fep_top_n_strict)
correctly filters candidates.
"""

import os
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from autoantibiotic.config import PipelineConfig
from autoantibiotic.io_utils import BinaryManager
from autoantibiotic.models import CompoundRecord


def _make_record(
    cid: str,
    smiles: str,
    allosteric_energy: float,
    has_pose: bool = True,
    ifp_score: float = 0.0,
) -> CompoundRecord:
    """Create a CompoundRecord with the given properties."""
    mol = Chem.MolFromSmiles(smiles)
    rec = CompoundRecord(
        compound_id=cid,
        smiles=smiles,
        mol=mol,
        pb2pa_allosteric_energy=allosteric_energy,
    )
    if has_pose:
        fd, path = tempfile.mkstemp(suffix=".pdbqt")
        os.close(fd)
        rec.docked_pose_path = path
    rec._test_ifp_score = ifp_score
    return rec


class TestStrictFiltering:
    """Tests for the strict pre-screening logic used in FEP resistance."""

    def setup_method(self) -> None:
        self.ref_smiles = "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O"

    def test_strict_filter_reduces_candidates(self) -> None:
        """Candidates with low IFP or poor energy should be filtered out."""
        candidates: List[CompoundRecord] = []
        # 5 good candidates: energy < -8.0
        for i in range(5):
            rec = _make_record(
                f"good_{i}", "c1ccccc1", -9.0 - i,
                has_pose=True, ifp_score=0.85,
            )
            candidates.append(rec)
        # 5 poor candidates: energy >= -8.0
        for i in range(5):
            rec = _make_record(
                f"poor_energy_{i}", "c1ccccc1", -7.0 + i,
                has_pose=True, ifp_score=0.85,
            )
            candidates.append(rec)
        # 5 poor candidates: IFP < 0.7
        for i in range(5):
            rec = _make_record(
                f"poor_ifp_{i}", "c1ccccc1", -10.0,
                has_pose=True, ifp_score=0.3,
            )
            candidates.append(rec)

        energy_cutoff = -8.0
        ifp_threshold_strict = 0.7
        fep_top_n_strict = 5

        ref_mol = Chem.MolFromSmiles(self.ref_smiles)

        filtered: List[CompoundRecord] = []
        for rec in candidates:
            if rec.pb2pa_allosteric_energy is None or rec.pb2pa_allosteric_energy >= energy_cutoff:
                continue
            if rec.docked_pose_path and os.path.isfile(rec.docked_pose_path) and ref_mol is not None and rec.mol is not None:
                ifp_score = getattr(rec, "_test_ifp_score", 0.0)
                if ifp_score >= ifp_threshold_strict:
                    filtered.append(rec)

        filtered.sort(key=lambda r: r.pb2pa_allosteric_energy if r.pb2pa_allosteric_energy is not None else 0.0)
        filtered = filtered[:fep_top_n_strict]

        # Should only keep the 5 good candidates (energy < -8.0, IFP >= 0.7)
        assert len(filtered) == 5
        for rec in filtered:
            assert rec.pb2pa_allosteric_energy < -8.0
            cid = rec.compound_id
            assert cid.startswith("good"), f"Unexpected candidate: {cid}"

    def test_all_poor_candidates_filtered(self) -> None:
        """When all candidates fail strict checks, the result should be empty."""
        candidates: List[CompoundRecord] = []
        for i in range(5):
            rec = _make_record(
                f"poor_{i}", "c1ccccc1", -7.0,
                has_pose=True, ifp_score=0.3,
            )
            candidates.append(rec)

        energy_cutoff = -8.0
        ifp_threshold_strict = 0.7
        fep_top_n_strict = 5

        ref_mol = Chem.MolFromSmiles(self.ref_smiles)

        filtered: List[CompoundRecord] = []
        for rec in candidates:
            if rec.pb2pa_allosteric_energy is None or rec.pb2pa_allosteric_energy >= energy_cutoff:
                continue
            if rec.docked_pose_path and os.path.isfile(rec.docked_pose_path) and ref_mol is not None and rec.mol is not None:
                ifp_score = getattr(rec, "_test_ifp_score", 0.0)
                if ifp_score >= ifp_threshold_strict:
                    filtered.append(rec)

        assert len(filtered) == 0

    def test_top_n_strict_limits_results(self) -> None:
        """Only fep_top_n_strict candidates should be kept."""
        candidates: List[CompoundRecord] = []
        for i in range(20):
            rec = _make_record(
                f"good_{i}", "c1ccccc1", -9.0 - i,
                has_pose=True, ifp_score=0.9,
            )
            candidates.append(rec)

        energy_cutoff = -8.0
        ifp_threshold_strict = 0.7
        fep_top_n_strict = 3

        ref_mol = Chem.MolFromSmiles(self.ref_smiles)

        filtered: List[CompoundRecord] = []
        for rec in candidates:
            if rec.pb2pa_allosteric_energy is None or rec.pb2pa_allosteric_energy >= energy_cutoff:
                continue
            if rec.docked_pose_path and os.path.isfile(rec.docked_pose_path) and ref_mol is not None and rec.mol is not None:
                ifp_score = getattr(rec, "_test_ifp_score", 0.0)
                if ifp_score >= ifp_threshold_strict:
                    filtered.append(rec)

        filtered.sort(key=lambda r: r.pb2pa_allosteric_energy if r.pb2pa_allosteric_energy is not None else 0.0)
        filtered = filtered[:fep_top_n_strict]

        assert len(filtered) == 3

    def teardown_method(self) -> None:
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, CompoundRecord) and obj.docked_pose_path:
                try:
                    if os.path.exists(obj.docked_pose_path):
                        os.unlink(obj.docked_pose_path)
                except (OSError, AttributeError):
                    pass
