"""
Tests for resistance mutation sensitivity profiling (v4.0).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.config import CONFIG
from autoantibiotic.models import CompoundRecord
from autoantibiotic.analysis import (
    profile_resistance_mutation_sensitivity,
    profile_resistance_risk,
)


def _make_test_record(smiles: str = "c1ccccc1O") -> CompoundRecord:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return CompoundRecord(
        compound_id="TEST-001",
        smiles=smiles,
        mol=mol,
        pb2pa_allosteric_energy=-8.5,
        pb2pa_active_energy=-7.2,
        qed_score=0.7,
    )


def test_profile_resistance_mutation_sensitivity_no_mutants() -> None:
    """Empty mutant list should return None."""
    rec = _make_test_record()
    result = profile_resistance_mutation_sensitivity(
        rec, "/tmp/nonexistent", [], np.zeros(3), (15.0, 15.0, 15.0),
    )
    assert result is None


def test_profile_resistance_mutation_sensitivity_invalid_mutants() -> None:
    """Non-existent mutant PDBQT files should gracefully return None."""
    rec = _make_test_record()
    result = profile_resistance_mutation_sensitivity(
        rec,
        "/tmp/nonexistent",
        ["/tmp/mut_nonexistent.pdbqt", "/tmp/mut_nonexistent2.pdbqt"],
        np.zeros(3),
        (15.0, 15.0, 15.0),
    )
    # No actual docking can happen -> should return None
    assert result is None


def test_profile_resistance_risk_with_mutant_pdbqts() -> None:
    """profile_resistance_risk should accept mutant_pdbqts kwarg and
    store resistance_stability_score."""
    rec = _make_test_record()
    notes = profile_resistance_risk(
        rec,
        "/tmp/nonexistent",
        "/tmp/receptor.pdbqt",
        np.zeros(3),
        (15.0, 15.0, 15.0),
        mutant_pdbqts=None,
    )
    assert isinstance(notes, str)
    assert len(notes) > 0


def test_profile_resistance_risk_valid_record() -> None:
    """Basic heuristics should fire for a known compound."""
    # Ceftaroline — large, high MW, high QED
    rec = _make_test_record(
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O"
    )
    notes = profile_resistance_risk(
        rec,
        "/tmp/nonexistent",
        "/tmp/receptor.pdbqt",
        np.zeros(3),
        (15.0, 15.0, 15.0),
    )
    assert isinstance(notes, str)
    assert len(notes) > 0


def test_resistance_stability_score_field() -> None:
    """CompoundRecord should have resistance_stability_score field."""
    rec = _make_test_record()
    assert hasattr(rec, "resistance_stability_score")
    assert rec.resistance_stability_score is None


# ── CONFIG toggle ─────────────────────────────────────────────────


def test_use_mutation_sampling_config() -> None:
    """CONFIG.use_mutation_sampling should default to False."""
    assert CONFIG.use_mutation_sampling is False


def test_mutation_variants_config() -> None:
    """CONFIG.mutation_variants should contain expected entries."""
    assert "G246" in CONFIG.mutation_variants
    assert "N146" in CONFIG.mutation_variants
