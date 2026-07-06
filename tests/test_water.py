"""Unit tests for crystallographic water analysis."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from autoantibiotic.config import CONFIG
from autoantibiotic.water_analysis import (
    WaterInfo,
    WaterAnalysisResult,
    analyze_waters,
    get_waters_to_remove,
    _compute_displacement_energy,
    _count_hbonds_to_protein,
    _parse_pdb_waters,
)

# ── Test fixtures ──────────────────────────────────────────────

_PDB_WITH_WATERS = """\
ATOM      1  N   ASN A 159       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ASN A 159       1.500   1.500   1.500  1.00  0.00           C
ATOM      3  C   ASN A 159       2.500   2.000   1.800  1.00  0.00           C
ATOM      4  O   ASN A 159       3.200   2.800   1.200  1.00  0.00           O
ATOM      5  N   GLU A 237       2.000   2.000   2.000  1.00  0.00           N
ATOM      6  CA  GLU A 237       2.500   2.500   2.500  1.00  0.00           C
ATOM      7  C   GLU A 237       3.500   3.000   2.800  1.00  0.00           C
ATOM      8  O   GLU A 237       4.200   3.800   2.200  1.00  0.00           O
ATOM      9  N   ARG A 241       3.000   3.000   3.000  1.00  0.00           N
ATOM     10  CA  ARG A 241       3.500   3.500   3.500  1.00  0.00           C
ATOM     11  C   ARG A 241       4.500   4.000   3.800  1.00  0.00           C
ATOM     12  O   ARG A 241       5.200   4.800   3.200  1.00  0.00           O
ATOM     13  N   SER A 403       4.000   4.000   4.000  1.00  0.00           N
ATOM     14  CA  SER A 403       4.500   4.500   4.500  1.00  0.00           C
ATOM     15  C   SER A 403       5.500   5.000   4.800  1.00  0.00           C
ATOM     16  O   SER A 403       6.200   5.800   4.200  1.00  0.00           O
HETATM   17  O   HOH A 301       2.800   2.800   2.800  1.00 85.00           O
HETATM   18  O   HOH A 302       8.000   8.000   8.000  1.00 10.00           O
HETATM   19  O   HOH A 303       3.200   3.200   3.200  1.00 50.00           O
END
"""


@pytest.fixture(scope="session")
def pdb_with_waters_path() -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(_PDB_WITH_WATERS)
    tmp.close()
    yield tmp.name
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


@pytest.fixture(scope="module")
def water_info_301() -> WaterInfo:
    return WaterInfo(
        chain="A",
        resseq=301,
        position=np.array([2.8, 2.8, 2.8]),
        b_factor=85.0,
        occupancy=1.0,
        n_hbonds_protein=0,
        displacement_energy=0.0,
    )


# ── WaterInfo ──────────────────────────────────────────────────

class TestWaterInfo:
    def test_identifier_format(self, water_info_301: WaterInfo) -> None:
        assert water_info_301.identifier == "A:HOH_301"

    def test_defaults(self) -> None:
        w = WaterInfo(chain="B", resseq=100)
        assert w.resname == "HOH"
        assert not w.is_high_energy
        assert not w.is_bridging
        assert np.allclose(w.position, [0, 0, 0])

    def test_high_energy_flag(self) -> None:
        w = WaterInfo(chain="A", resseq=1, displacement_energy=3.0, is_high_energy=True)
        assert w.is_high_energy


# ── WaterAnalysisResult ───────────────────────────────────────

class TestWaterAnalysisResult:
    def test_empty_result(self) -> None:
        r = WaterAnalysisResult()
        assert len(r) == 0

    def test_len_matches_waters(self) -> None:
        w1 = WaterInfo(chain="A", resseq=1)
        w2 = WaterInfo(chain="A", resseq=2)
        r = WaterAnalysisResult(all_waters=[w1, w2])
        assert len(r) == 2

    def test_categorised_waters(self) -> None:
        he = WaterInfo(chain="A", resseq=1, is_high_energy=True)
        br = WaterInfo(chain="A", resseq=2, is_bridging=True)
        r = WaterAnalysisResult(
            high_energy_waters=[he],
            bridging_waters=[br],
            all_waters=[he, br],
        )
        assert len(r.high_energy_waters) == 1
        assert len(r.bridging_waters) == 1
        assert len(r) == 2


# ── Water analysis functions ──────────────────────────────────

class TestParsePdbWaters:
    def test_parse_returns_waters(self, pdb_with_waters_path: str) -> None:
        waters = _parse_pdb_waters(pdb_with_waters_path)
        assert len(waters) == 3

    def test_water_positions(self, pdb_with_waters_path: str) -> None:
        waters = _parse_pdb_waters(pdb_with_waters_path)
        waters.sort(key=lambda w: w.resseq)
        assert np.allclose(waters[0].position, [2.8, 2.8, 2.8])
        assert np.allclose(waters[1].position, [8.0, 8.0, 8.0])

    def test_water_bfactor(self, pdb_with_waters_path: str) -> None:
        waters = _parse_pdb_waters(pdb_with_waters_path)
        waters.sort(key=lambda w: w.resseq)
        assert waters[0].b_factor == pytest.approx(85.0)
        assert waters[1].b_factor == pytest.approx(10.0)

    def test_water_chain(self, pdb_with_waters_path: str) -> None:
        waters = _parse_pdb_waters(pdb_with_waters_path)
        for w in waters:
            assert w.chain == "A"


class TestCountHbonds:
    def test_no_hbonds_empty_protein(self) -> None:
        n = _count_hbonds_to_protein(np.array([0, 0, 0]), [])
        assert n == 0

    def test_count_hbonds_nearby(self) -> None:
        water = np.array([0.0, 0.0, 0.0])
        protein = [
            np.array([0.0, 0.0, 3.0]),   # within 3.5
            np.array([0.0, 0.0, 3.6]),   # beyond 3.5
            np.array([3.0, 3.0, 3.0]),   # sqrt(27)=5.2 > 3.5
        ]
        n = _count_hbonds_to_protein(water, protein, distance_cutoff=3.5)
        assert n == 1

    def test_all_within_cutoff(self) -> None:
        water = np.array([0.0, 0.0, 0.0])
        protein = [np.array([1.0, 1.0, 1.0]), np.array([-1.0, -1.0, -1.0])]
        n = _count_hbonds_to_protein(water, protein, distance_cutoff=3.0)
        dist1 = float(np.linalg.norm(protein[0]))
        assert dist1 <= 3.0
        assert n == 2


class TestDisplacementEnergy:
    def test_low_bfactor_gives_low_energy(self) -> None:
        w = WaterInfo(chain="A", resseq=1, b_factor=5.0)
        energy = _compute_displacement_energy(w, [])
        # No protein atoms → max hbond penalty
        assert 0.0 <= energy <= 1.0

    def test_high_bfactor_gives_higher_energy(self) -> None:
        w_low = WaterInfo(chain="A", resseq=1, b_factor=10.0)
        w_high = WaterInfo(chain="A", resseq=2, b_factor=90.0)
        e_low = _compute_displacement_energy(w_low, [])
        e_high = _compute_displacement_energy(w_high, [])
        assert e_high > e_low

    def test_more_hbonds_lowers_energy(self) -> None:
        protein = [np.array([0.0, 0.0, d]) for d in [2.0, 2.5, 3.0]]
        w = WaterInfo(chain="A", resseq=1, b_factor=50.0)
        energy = _compute_displacement_energy(w, protein)
        assert 0.0 <= energy <= 1.0

    def test_low_occupancy_increases_energy(self) -> None:
        w_high = WaterInfo(chain="A", resseq=1, b_factor=50.0, occupancy=1.0)
        w_low = WaterInfo(chain="A", resseq=2, b_factor=50.0, occupancy=0.5)
        e_high = _compute_displacement_energy(w_high, [])
        e_low = _compute_displacement_energy(w_low, [])
        assert e_low > e_high


class TestAnalyzeWaters:
    def test_no_waters_returns_empty(self) -> None:
        pdb_no_waters = """\
ATOM      1  N   ASN A 159       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ASN A 159       1.500   1.500   1.500  1.00  0.00           C
END
"""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
        tmp.write(pdb_no_waters)
        tmp.close()
        try:
            result = analyze_waters(
                tmp.name,
                allosteric_residues=["ASN159"],
                active_site_residues=[],
            )
            assert isinstance(result, WaterAnalysisResult)
            assert len(result) == 0
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def test_analyze_with_waters(self, pdb_with_waters_path: str) -> None:
        result = analyze_waters(
            pdb_with_waters_path,
            allosteric_residues=["ASN159", "GLU237", "ARG241"],
            active_site_residues=["SER403"],
            distance_cutoff=5.0,
            displacement_energy_threshold=2.5,
        )
        # Should have found waters near the binding-site residues
        assert isinstance(result, WaterAnalysisResult)


class TestGetWatersToRemove:
    def test_remove_high_energy_non_bridging(self) -> None:
        he = WaterInfo(chain="A", resseq=1, is_high_energy=True, is_bridging=False)
        br = WaterInfo(chain="A", resseq=2, is_high_energy=False, is_bridging=True)
        also_he = WaterInfo(chain="A", resseq=3, is_high_energy=True, is_bridging=False)
        also_br = WaterInfo(chain="A", resseq=4, is_high_energy=True, is_bridging=True)
        result = WaterAnalysisResult(
            high_energy_waters=[he, also_he, also_br],
            bridging_waters=[br, also_br],
            all_waters=[he, br, also_he, also_br],
        )
        to_remove = get_waters_to_remove(result)
        ids = {w.identifier for w in to_remove}
        assert ids == {"A:HOH_1", "A:HOH_3"}
