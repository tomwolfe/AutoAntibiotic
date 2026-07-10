"""Tests for the docking engine ABC and implementations."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from autoantibiotic.config import CONFIG, PipelineConfig
from autoantibiotic.docking import (
    DockingEngine,
    VinaEngine,
    GninaEngine,
    RdkitShapeEngine,
    get_engine,
)
from autoantibiotic.io_utils import ToolResult


# ── Abstract Base Class ───────────────────────────────────────────────

class TestDockingEngineABC:
    """DockingEngine cannot be instantiated directly."""

    def test_abstract_class_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            DockingEngine()  # type: ignore[abstract]


# ── Factory ───────────────────────────────────────────────────────────

class TestGetEngine:
    """Factory function returns correct engine types."""

    def test_returns_vina_engine(self) -> None:
        eng = get_engine("vina")
        assert isinstance(eng, VinaEngine)

    def test_returns_gnina_engine(self) -> None:
        eng = get_engine("gnina")
        assert isinstance(eng, GninaEngine)

    def test_returns_shape_engine(self) -> None:
        eng = get_engine("shape")
        assert isinstance(eng, RdkitShapeEngine)

    def test_unknown_name_returns_shape_engine(self) -> None:
        eng = get_engine("unknown")
        assert isinstance(eng, RdkitShapeEngine)


# ── VinaEngine ────────────────────────────────────────────────────────

class TestVinaEngine:
    """VinaEngine mocked subprocess calls."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.saved = (
            CONFIG.dry_run,
            CONFIG.validate_docking_binaries_on_startup,
            CONFIG.vina_exhaustiveness,
            CONFIG.vina_num_modes,
        )
        CONFIG.dry_run = False
        CONFIG.validate_docking_binaries_on_startup = False
        yield
        (
            CONFIG.dry_run,
            CONFIG.validate_docking_binaries_on_startup,
            CONFIG.vina_exhaustiveness,
            CONFIG.vina_num_modes,
        ) = self.saved

    def test_instantiate(self) -> None:
        eng = VinaEngine()
        assert eng.tool_name == "vina"
        assert eng.binary_path == "vina"

    def test_dry_run_returns_float(self) -> None:
        CONFIG.dry_run = True
        eng = VinaEngine()
        score = eng.dock(
            ligand_path="lig.pdbqt",
            receptor_path="rec.pdbqt",
            center=np.array([0.0, 0.0, 0.0]),
            box_size=(20.0, 20.0, 20.0),
        )
        assert score is not None
        assert -10.0 <= score <= -5.0

    def test_dock_success_returns_energy(self) -> None:
        mock_result = ToolResult(
            returncode=0,
            stdout=(
                "mode |   affinity | dist from best mode\n"
                "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
                "-----+------------+----------+----------\n"
                "   1       -8.123       0.000      0.000\n"
            ),
            stderr="",
        )
        with patch("autoantibiotic.docking.engines.ToolExecutor.run", return_value=mock_result):
            eng = VinaEngine()
            score = eng.dock(
                ligand_path="lig.pdbqt",
                receptor_path="rec.pdbqt",
                center=np.array([0.0, 0.0, 0.0]),
                box_size=(20.0, 20.0, 20.0),
            )
        assert score == pytest.approx(-8.123)

    def test_dock_failure_returns_none(self) -> None:
        mock_result = ToolResult(returncode=1, stdout="", stderr="Error: could not open receptor")
        with patch("autoantibiotic.docking.engines.ToolExecutor.run", return_value=mock_result):
            eng = VinaEngine()
            score = eng.dock(
                ligand_path="lig.pdbqt",
                receptor_path="rec.pdbqt",
                center=np.array([0.0, 0.0, 0.0]),
                box_size=(20.0, 20.0, 20.0),
            )
        assert score is None

    def test_prepare_ligand_too_large(self) -> None:
        eng = VinaEngine()
        mol = Chem.MolFromSmiles("C" * 200)
        assert mol is not None
        result = eng.prepare_ligand(mol, "/tmp/test.pdbqt")
        assert not result


# ── GninaEngine ───────────────────────────────────────────────────────

class TestGninaEngine:
    """GninaEngine mocked subprocess calls."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.saved = (
            CONFIG.dry_run,
            CONFIG.validate_docking_binaries_on_startup,
            CONFIG.gnina_binary_path,
        )
        CONFIG.dry_run = False
        CONFIG.validate_docking_binaries_on_startup = False
        CONFIG.gnina_binary_path = "gnina"
        yield
        (
            CONFIG.dry_run,
            CONFIG.validate_docking_binaries_on_startup,
            CONFIG.gnina_binary_path,
        ) = self.saved

    def test_instantiate(self) -> None:
        eng = GninaEngine()
        assert eng.tool_name == "gnina"
        assert eng.binary_path == "gnina"

    def test_dry_run_returns_float(self) -> None:
        CONFIG.dry_run = True
        eng = GninaEngine()
        score = eng.dock(
            ligand_path="lig.pdbqt",
            receptor_path="rec.pdbqt",
            center=np.array([0.0, 0.0, 0.0]),
            box_size=(20.0, 20.0, 20.0),
        )
        assert score is not None
        assert 0.5 <= score <= 0.95

    def test_dock_success_returns_cnnscore(self) -> None:
        mock_result = ToolResult(
            returncode=0,
            stdout="CNNscore    :   0.9123\nCNNaffinity :   8.4567\n",
            stderr="",
        )
        with patch("autoantibiotic.docking.engines.ToolExecutor.run", return_value=mock_result):
            eng = GninaEngine()
            score = eng.dock(
                ligand_path="lig.pdbqt",
                receptor_path="rec.pdbqt",
                center=np.array([0.0, 0.0, 0.0]),
                box_size=(20.0, 20.0, 20.0),
            )
        assert score == pytest.approx(0.9123)

    def test_dock_failure_returns_none(self) -> None:
        mock_result = ToolResult(returncode=1, stdout="", stderr="CUDA error")
        with patch("autoantibiotic.docking.engines.ToolExecutor.run", return_value=mock_result):
            eng = GninaEngine()
            score = eng.dock(
                ligand_path="lig.pdbqt",
                receptor_path="rec.pdbqt",
                center=np.array([0.0, 0.0, 0.0]),
                box_size=(20.0, 20.0, 20.0),
            )
        assert score is None

    def test_uses_configured_binary_path(self) -> None:
        CONFIG.gnina_binary_path = "/custom/path/gnina"
        eng = GninaEngine()
        assert eng.binary_path == "/custom/path/gnina"


# ── RdkitShapeEngine ──────────────────────────────────────────────────

class TestRdkitShapeEngine:
    """RdkitShapeEngine computes shape scores for known molecules."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.saved = CONFIG.dry_run
        CONFIG.dry_run = False
        yield
        CONFIG.dry_run = self.saved

    @pytest.fixture
    def benzene_mol(self) -> Chem.Mol:
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol

    @pytest.fixture
    def toluene_mol(self) -> Chem.Mol:
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        assert mol is not None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol

    def test_instantiate(self) -> None:
        eng = RdkitShapeEngine()
        assert eng._ref_mol is None

    def test_set_reference_mol(self, benzene_mol: Chem.Mol) -> None:
        eng = RdkitShapeEngine()
        eng.set_reference_mol(benzene_mol)
        assert eng._ref_mol is not None

    def test_prepare_ligand_always_true(self) -> None:
        eng = RdkitShapeEngine()
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        assert eng.prepare_ligand(mol, "/tmp/test.pdbqt")

    def test_prepare_receptor_returns_path(self) -> None:
        eng = RdkitShapeEngine()
        result = eng.prepare_receptor("/tmp/rec.pdb")
        assert result == "/tmp/rec.pdb"

    def test_dry_run_returns_float(self) -> None:
        CONFIG.dry_run = True
        eng = RdkitShapeEngine()
        score = eng.dock(
            ligand_path="lig.pdb",
            receptor_path="rec.pdb",
            center=np.array([0.0, 0.0, 0.0]),
            box_size=(20.0, 20.0, 20.0),
        )
        assert score is not None

    def test_dock_no_ref_mol_returns_none(self) -> None:
        eng = RdkitShapeEngine()
        eng._ref_mol = None
        with patch.object(Chem, "MolFromPDBFile", return_value=None):
            with patch.object(Chem, "MolFromSmiles", return_value=None):
                score = eng.dock(
                    ligand_path="nonexistent.pdb",
                    receptor_path="rec.pdb",
                    center=np.array([0.0, 0.0, 0.0]),
                    box_size=(20.0, 20.0, 20.0),
                )
        assert score is None

    def test_shape_score_reasonable(self, benzene_mol: Chem.Mol, toluene_mol: Chem.Mol) -> None:
        eng = RdkitShapeEngine()
        eng.set_reference_mol(benzene_mol)
        score = eng._compute_shape_score(toluene_mol, benzene_mol)
        if score is not None:
            assert score >= 0.0


# ── Engine factory integration ────────────────────────────────────────

class TestEngineIntegration:
    """End-to-end integration with get_engine factory."""

    def test_engine_implements_abc(self) -> None:
        for name in ("vina", "gnina", "shape"):
            eng = get_engine(name)
            assert isinstance(eng, DockingEngine)
            assert hasattr(eng, "dock")
            assert hasattr(eng, "prepare_receptor")
            assert hasattr(eng, "prepare_ligand")
