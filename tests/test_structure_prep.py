"""Unit tests for structure preparation utilities, including dynamic box sizing."""

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.structure_prep import (
    calculate_adaptive_box_size,
    get_ligand_max_dimension,
)


class TestCalculateAdaptiveBoxSize:
    """``calculate_adaptive_box_size`` computes box dims from coordinates."""

    def test_basic_box(self) -> None:
        coords = np.array([[0, 0, 0], [10, 10, 10]], dtype=float)
        sx, sy, sz = calculate_adaptive_box_size(coords, padding=2.0, minimum=5.0)
        assert sx == pytest.approx(14.0)
        assert sy == pytest.approx(14.0)
        assert sz == pytest.approx(14.0)

    def test_minimum_clamp(self) -> None:
        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        sx, sy, sz = calculate_adaptive_box_size(coords, padding=0.0, minimum=10.0)
        assert sx == pytest.approx(10.0)
        assert sy == pytest.approx(10.0)
        assert sz == pytest.approx(10.0)

    def test_single_point_returns_minimum(self) -> None:
        coords = np.array([[5, 5, 5]], dtype=float)
        sx, sy, sz = calculate_adaptive_box_size(coords, padding=2.0, minimum=8.0)
        assert sx == pytest.approx(8.0)
        assert sy == pytest.approx(8.0)
        assert sz == pytest.approx(8.0)

    def test_empty_array_returns_minimum(self) -> None:
        coords = np.empty((0, 3), dtype=float)
        sx, sy, sz = calculate_adaptive_box_size(coords, padding=2.0, minimum=6.0)
        assert sx == pytest.approx(6.0)
        assert sy == pytest.approx(6.0)
        assert sz == pytest.approx(6.0)

    def test_none_coords_returns_minimum(self) -> None:
        sx, sy, sz = calculate_adaptive_box_size(None, padding=2.0, minimum=6.0)  # type: ignore[arg-type]
        assert sx == pytest.approx(6.0)

    def test_non_uniform_dimensions(self) -> None:
        coords = np.array([[0, 0, 0], [5, 15, 3]], dtype=float)
        sx, sy, sz = calculate_adaptive_box_size(coords, padding=1.0, minimum=3.0)
        assert sx == pytest.approx(7.0)
        assert sy == pytest.approx(17.0)
        assert sz == pytest.approx(5.0)

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_adaptive_box_size(np.array([1, 2, 3]))


class TestGetLigandMaxDimension:
    """``get_ligand_max_dimension`` returns max heavy-atom distance."""

    def test_benzene(self) -> None:
        """Benzene diameter should be ~2.8 Å (C-C distance ~1.4 Å × 2)."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        params = Chem.rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        Chem.rdDistGeom.EmbedMolecule(mol, params)
        d = get_ligand_max_dimension(mol)
        assert d == pytest.approx(2.8, abs=0.3), f"Benzene max dim: {d:.2f}"

    def test_ceftaroline(self) -> None:
        """Ceftaroline is a large antibiotic ~15-20 Å."""
        smi = "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O"
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        mol = Chem.AddHs(mol)
        params = Chem.rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        status = Chem.rdDistGeom.EmbedMolecule(mol, params)
        assert status >= 0, "Ceftaroline embedding failed"
        d = get_ligand_max_dimension(mol)
        assert 10.0 <= d <= 25.0, f"Ceftaroline max dim: {d:.2f}"

    def test_single_heavy_atom_returns_zero(self) -> None:
        """Methane has 1 heavy atom → 0.0."""
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        d = get_ligand_max_dimension(mol)
        assert d == pytest.approx(0.0)

    def test_no_conformer_generates_one(self) -> None:
        """A molecule without a conformer should still return a value."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        d = get_ligand_max_dimension(mol)
        assert d == pytest.approx(2.8, abs=0.3), f"Benzene max dim (no initial conf): {d:.2f}"

    def test_empty_mol_returns_zero(self) -> None:
        mol = Chem.MolFromSmiles("")
        assert mol is not None
        d = get_ligand_max_dimension(mol)
        assert d == pytest.approx(0.0)

    def test_linear_molecule(self) -> None:
        """Octane should have a measurable end-to-end distance > 5 Å."""
        mol = Chem.MolFromSmiles("CCCCCCCC")
        mol = Chem.AddHs(mol)
        params = Chem.rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        status = Chem.rdDistGeom.EmbedMolecule(mol, params)
        assert status >= 0
        d = get_ligand_max_dimension(mol)
        assert d >= 5.0, f"Octane max dim: {d:.2f}"
        assert d <= 15.0, f"Octane max dim: {d:.2f}"
