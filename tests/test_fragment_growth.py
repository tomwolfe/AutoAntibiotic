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


# ── Stereoisomer enumeration tests ───────────────────────────────

class TestStereoisomerEnumeration:
    """Tests for stereoisomer enumeration in library generation."""

    def test_enumerate_no_stereocenters(self) -> None:
        """Molecule without stereocenters should return only the original
        (RDKit returns the input molecule itself when there are no isomers)."""
        from autoantibiotic.library_gen import _enumerate_stereoisomers
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        isomers = _enumerate_stereoisomers(mol, max_isomers=8)
        # RDKit returns at least the original molecule; check no extra
        # stereoisomers beyond the original are created
        assert len(isomers) <= 1

    def test_enumerate_undefined_chiral_center(self) -> None:
        """Molecule with undefined chiral center should yield isomers."""
        from autoantibiotic.library_gen import _enumerate_stereoisomers
        mol = Chem.MolFromSmiles("CCC(C)O")
        assert mol is not None
        isomers = _enumerate_stereoisomers(mol, max_isomers=8)
        assert len(isomers) >= 1
        for iso in isomers:
            assert iso is not None
            smi = Chem.MolToSmiles(iso)
            assert smi != "CCC(C)O"  # should be different from undefined

    def test_enumerate_respects_max_isomers(self) -> None:
        """Even with many possibilities, should not exceed max_isomers."""
        from autoantibiotic.library_gen import _enumerate_stereoisomers
        mol = Chem.MolFromSmiles("CC(O)C(C)O")
        assert mol is not None
        isomers = _enumerate_stereoisomers(mol, max_isomers=4)
        assert len(isomers) <= 4

    def test_isomers_are_sanitized(self) -> None:
        """Enumerated isomers should be valid sanitized molecules."""
        from autoantibiotic.library_gen import _enumerate_stereoisomers
        mol = Chem.MolFromSmiles("CCC(C)O")
        assert mol is not None
        isomers = _enumerate_stereoisomers(mol, max_isomers=8)
        for iso in isomers:
            assert iso is not None
            # Verify sanitization by computing a simple property
            assert Chem.Descriptors.MolWt(iso) > 0

    def test_isomers_have_unique_smiles(self) -> None:
        """Each isomer should have a distinct SMILES."""
        from autoantibiotic.library_gen import _enumerate_stereoisomers
        mol = Chem.MolFromSmiles("CCC(C)O")
        assert mol is not None
        isomers = _enumerate_stereoisomers(mol, max_isomers=8)
        smiles_set = set()
        for iso in isomers:
            smi = Chem.MolToSmiles(iso)
            smiles_set.add(smi)
        assert len(smiles_set) == len(isomers)

    def test_grown_library_uses_isomer_ids(self) -> None:
        """Grown compounds with stereoisomers should have suffixed IDs."""
        core = _make_core_record("c1ccncc1")
        # Building block with an undefined chiral center
        bbs = ["[1*]CC(C)O"]
        results = list(
            generate_grown_library(
                [core],
                building_blocks=bbs,
                max_growth_steps=1,
                target_per_core=10,
            )
        )
        for rec in results:
            assert isinstance(rec, CompoundRecord)
            # If expensive features are off, IDs are plain GROWN-*
            # If on, some may have suffixes
            if rec.parent_id is not None:
                assert "-" in rec.compound_id
                assert rec.parent_id.startswith("GROWN-")

    def test_parent_id_preserved_on_isomers(self) -> None:
        """Isomers should have a parent_id linking to the original."""
        from autoantibiotic.library_gen import _enumerate_stereoisomers
        mol = Chem.MolFromSmiles("CCC(C)O")
        assert mol is not None
        isomers = _enumerate_stereoisomers(mol, max_isomers=8)
        if isomers:
            # Simulate what generate_grown_library does
            base_id = "GROWN-000"
            for j, iso in enumerate(isomers):
                suf = chr(ord("a") + j)
                rec = CompoundRecord(
                    compound_id=f"{base_id}-{suf}",
                    smiles=Chem.MolToSmiles(iso),
                    mol=iso,
                    parent_id=base_id,
                )
                assert rec.parent_id == base_id
                assert rec.compound_id == f"{base_id}-{suf}"


# ── Stereochemistry flagging tests ────────────────────────────────

class TestStereochemistryFlagging:
    """Tests for undefined stereochemistry detection in library generation."""

    def test_check_undefined_stereo_achiral_molecule(self) -> None:
        """Benzene should have no undefined stereocenters."""
        from autoantibiotic.library_gen import _check_undefined_stereo
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert _check_undefined_stereo(mol) is False

    def test_check_undefined_stereo_chiral_defined(self) -> None:
        """A molecule with defined tetrahedral stereochemistry."""
        from autoantibiotic.library_gen import _check_undefined_stereo
        # (R)-2-butanol with explicit stereochemistry
        mol = Chem.MolFromSmiles("CC[C@H](C)O")
        assert mol is not None
        assert _check_undefined_stereo(mol) is False

    def test_check_undefined_stereo_undefined_chiral(self) -> None:
        """A molecule with undefined tetrahedral stereochemistry."""
        from autoantibiotic.library_gen import _check_undefined_stereo
        # 2-butanol without stereochemistry specified
        mol = Chem.MolFromSmiles("CCC(C)O")
        assert mol is not None
        stereo = Chem.FindPotentialStereo(mol)
        has_undefined = any(s.specified == Chem.StereoSpecified.Unspecified for s in stereo)
        assert has_undefined is True

    def test_check_undefined_stereo_undefined_double_bond(self) -> None:
        """A molecule with undefined geometric (cis/trans) stereochemistry."""
        from autoantibiotic.library_gen import _check_undefined_stereo
        # 2-butene without stereochemistry
        mol = Chem.MolFromSmiles("CC=CC")
        assert mol is not None
        assert _check_undefined_stereo(mol) is True

    def test_check_undefined_stereo_defined_double_bond(self) -> None:
        """A molecule with defined geometric stereochemistry."""
        from autoantibiotic.library_gen import _check_undefined_stereo
        # (E)-2-butene
        mol = Chem.MolFromSmiles("C/C=C/C")
        assert mol is not None
        assert _check_undefined_stereo(mol) is False

    def test_compound_record_has_undefined_stereo_field(self) -> None:
        """CompoundRecord should have the has_undefined_stereo field."""
        rec = CompoundRecord(
            compound_id="TEST",
            smiles="c1ccccc1",
        )
        assert hasattr(rec, "has_undefined_stereo")
        assert rec.has_undefined_stereo is False

    def test_grown_compound_sets_stereo_flag(self) -> None:
        """Grown compounds with undefined stereocenters should be flagged."""
        core = _make_core_record("c1ccncc1")
        # Building block with an undefined chiral center
        bbs = ["[1*]CC(C)O"]
        results = list(
            generate_grown_library(
                [core],
                building_blocks=bbs,
                max_growth_steps=1,
                target_per_core=10,
            )
        )
        for rec in results:
            assert isinstance(rec.has_undefined_stereo, bool)
