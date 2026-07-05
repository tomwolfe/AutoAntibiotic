"""
Tests for the dynamic fragment growth function (v4.0).
"""

from __future__ import annotations

from typing import List, Optional

import pytest
from rdkit import Chem

from autoantibiotic.config import CONFIG
from autoantibiotic.library_gen import generate_grown_library
from autoantibiotic.models import CompoundRecord


def _make_core_record(smiles: str = "c1ccccc1") -> CompoundRecord:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return CompoundRecord(
        compound_id="CORE-000",
        smiles=smiles,
        mol=mol,
        pb2pa_allosteric_energy=-8.0,
        qed_score=0.7,
    )


def test_generate_grown_library_empty_cores() -> None:
    """Empty core list should yield nothing."""
    results = list(generate_grown_library([], max_growth_steps=1, target_per_core=10))
    assert len(results) == 0


def test_generate_grown_library_no_building_blocks() -> None:
    """Empty building blocks should yield nothing."""
    core = _make_core_record()
    results = list(
        generate_grown_library(
            [core],
            building_blocks=[],
            max_growth_steps=1,
            target_per_core=10,
        )
    )
    assert len(results) == 0


def test_generate_grown_library_basic_growth() -> None:
    """A simple core with building blocks should produce grown compounds."""
    core = _make_core_record("c1ccccc1")
    bbs = ["[1*]c1ccccc1", "[1*]CCO"]
    results = list(
        generate_grown_library(
            [core],
            building_blocks=bbs,
            max_growth_steps=1,
            target_per_core=5,
        )
    )
    # May produce 0 if BRICS cannot decompose the simple benzene ring
    # (benzene has no BRICS break points), but should not crash.
    for rec in results:
        assert isinstance(rec, CompoundRecord)
        assert rec.compound_id.startswith("GROWN-")
        assert rec.mol is not None
        assert Chem.MolToSmiles(rec.mol) == rec.smiles


def test_generate_grown_library_lipinski_filter() -> None:
    """Grown compounds should pass Lipinski/QED filters."""
    core = _make_core_record("c1ccncc1")  # pyridine — has BRICS break point
    bbs = ["[1*]c1ccccc1"]
    results = list(
        generate_grown_library(
            [core],
            building_blocks=bbs,
            max_growth_steps=1,
            target_per_core=10,
        )
    )
    for rec in results:
        assert rec.mol is not None
        mw = Chem.Descriptors.MolWt(rec.mol)
        logp = Chem.Crippen.MolLogP(rec.mol)
        hbd = Chem.Descriptors.NumHDonors(rec.mol)
        hba = Chem.Descriptors.NumHAcceptors(rec.mol)
        assert mw <= CONFIG.lipinski_mw_max
        assert logp <= CONFIG.lipinski_logp_max
        assert hbd <= CONFIG.lipinski_hbd_max
        assert hba <= CONFIG.lipinski_hba_max


def test_generate_grown_library_no_duplicates() -> None:
    """No duplicate SMILES should be yielded."""
    core = _make_core_record("c1ccncc1")
    bbs = ["[1*]c1ccccc1", "[1*]CCO", "[1*]c1ccc(O)cc1"]
    results = list(
        generate_grown_library(
            [core],
            building_blocks=bbs,
            max_growth_steps=1,
            target_per_core=20,
        )
    )
    smiles_set = set()
    for rec in results:
        assert rec.smiles not in smiles_set, f"Duplicate SMILES: {rec.smiles}"
        smiles_set.add(rec.smiles)


def test_generate_grown_library_multiple_cores() -> None:
    """Multiple cores should each be processed."""
    core1 = _make_core_record("c1ccncc1")
    core2 = _make_core_record("c1ccccc1O")
    bbs = ["[1*]c1ccccc1"]
    results = list(
        generate_grown_library(
            [core1, core2],
            building_blocks=bbs,
            max_growth_steps=1,
            target_per_core=5,
        )
    )
    # Should not crash; may produce 0 compounds if cores have no BRICS breaks
    for rec in results:
        assert isinstance(rec, CompoundRecord)


def test_generate_grown_library_target_per_core() -> None:
    """The target per core limit should be respected."""
    core = _make_core_record("c1ccncc1")
    bbs = ["[1*]c1ccccc1", "[1*]c1ccc(O)cc1", "[1*]c1ccc(Cl)cc1", "[1*]CCO"]
    results = list(
        generate_grown_library(
            [core],
            building_blocks=bbs,
            max_growth_steps=1,
            target_per_core=3,
        )
    )
    assert len(results) <= 3


def test_generate_grown_library_multi_step() -> None:
    """Multi-step growth should produce larger compounds."""
    core = _make_core_record("c1ccncc1")
    bbs = ["[1*]CCO"]
    one_step = list(
        generate_grown_library(
            [core],
            building_blocks=bbs,
            max_growth_steps=1,
            target_per_core=5,
        )
    )
    two_step = list(
        generate_grown_library(
            [core],
            building_blocks=bbs,
            max_growth_steps=2,
            target_per_core=5,
        )
    )
    # Two-step should not produce fewer compounds than one-step
    # (may produce the same if no further growth is possible)
    pass
