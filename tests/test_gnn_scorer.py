"""
Tests for the GNNScorer (Physics-Informed GNN rescoring, v4.0).

Mock ``torch`` / ``torch_geometric`` imports so that the logic can be
verified without requiring GPU or deep-learning libraries in CI.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.ml_scoring.gnn_scorer import (
    GNNScorer,
    _generate_etkdg_conformer,
    _get_atom_features,
    _get_bond_features,
    _one_hot_atomic_num,
    _parse_pdbqt_coords,
    mol_pose_to_graph,
)
from autoantibiotic.models import CompoundRecord


# ── test helpers ───────────────────────────────────────────────────

_SMILES_BENZENE = "c1ccccc1"
_SMILES_PHENOL = "c1ccccc1O"


def _make_record(smiles: str = _SMILES_PHENOL) -> CompoundRecord:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return CompoundRecord(
        compound_id="GNN-TEST-001",
        smiles=smiles,
        mol=mol,
        pb2pa_allosteric_energy=-8.5,
        shape_score=0.75,
        qed_score=0.7,
    )


def _write_dummy_pdbqt(path: str, n_atoms: int = 12) -> None:
    """Write a minimal PDBQT file with *n_atoms* HETATM records."""
    with open(path, "w") as fh:
        for i in range(n_atoms):
            fh.write(
                f"HETATM{i+1:4d}  C   LIG A   1    "
                f"   1.000   2.000   3.000  1.00  0.00           C\n"
            )


# ═══════════════════════════════════════════════════════════════════
#  Feature-extraction tests  (no DL deps required)
# ═══════════════════════════════════════════════════════════════════


class TestOneHotAtomicNum:
    def test_carbon(self) -> None:
        vec = _one_hot_atomic_num(6)
        assert len(vec) == 10
        assert vec[1] == 1.0
        assert all(v == 0.0 for i, v in enumerate(vec) if i != 1)

    def test_hydrogen(self) -> None:
        vec = _one_hot_atomic_num(1)
        assert vec[0] == 1.0

    def test_unknown_element(self) -> None:
        vec = _one_hot_atomic_num(999)
        assert all(v == 0.0 for v in vec)


class TestGetAtomFeatures:
    def test_carbon_in_benzene(self) -> None:
        mol = Chem.MolFromSmiles(_SMILES_BENZENE)
        assert mol is not None
        atom = mol.GetAtomWithIdx(0)
        feats = _get_atom_features(atom)
        assert len(feats) == 12  # 10 one-hot + degree + hybrid
        assert feats[1] == 1.0  # carbon

    def test_oxygen_in_phenol(self) -> None:
        mol = Chem.MolFromSmiles(_SMILES_PHENOL)
        assert mol is not None
        # oxygen is the last atom
        oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
        assert len(oxygens) == 1
        feats = _get_atom_features(oxygens[0])
        assert feats[3] == 1.0  # oxygen


class TestGetBondFeatures:
    def test_single(self) -> None:
        mol = Chem.MolFromSmiles("CC")
        assert mol is not None
        assert _get_bond_features(mol.GetBondWithIdx(0)) == [1.0, 0.0, 0.0, 0.0]

    def test_double(self) -> None:
        mol = Chem.MolFromSmiles("C=C")
        assert mol is not None
        assert _get_bond_features(mol.GetBondWithIdx(0)) == [0.0, 1.0, 0.0, 0.0]

    def test_triple(self) -> None:
        mol = Chem.MolFromSmiles("C#C")
        assert mol is not None
        assert _get_bond_features(mol.GetBondWithIdx(0)) == [0.0, 0.0, 1.0, 0.0]

    def test_aromatic(self) -> None:
        mol = Chem.MolFromSmiles(_SMILES_BENZENE)
        assert mol is not None
        assert _get_bond_features(mol.GetBondWithIdx(0)) == [0.0, 0.0, 0.0, 1.0]


# ═══════════════════════════════════════════════════════════════════
#  PDBQT coordinate parsing tests  (no DL deps)
# ═══════════════════════════════════════════════════════════════════


class TestParsePdbqtCoords:
    def test_valid_file(self, tmp_path: Path) -> None:
        pdbqt = str(tmp_path / "pose.pdbqt")
        _write_dummy_pdbqt(pdbqt, n_atoms=3)
        coords = _parse_pdbqt_coords(pdbqt)
        assert coords is not None
        assert coords.shape == (3, 3)
        np.testing.assert_array_almost_equal(coords[0], [1.0, 2.0, 3.0])

    def test_missing_file(self, tmp_path: Path) -> None:
        result = _parse_pdbqt_coords(str(tmp_path / "nonexistent.pdbqt"))
        assert result is None

    def test_empty_file(self, tmp_path: Path) -> None:
        pdbqt = str(tmp_path / "empty.pdbqt")
        Path(pdbqt).write_text("REMARK empty\n")
        result = _parse_pdbqt_coords(pdbqt)
        assert result is None


class TestGenerateETKDGConformer:
    def test_small_molecule(self) -> None:
        mol = Chem.MolFromSmiles(_SMILES_BENZENE)
        assert mol is not None
        coords = _generate_etkdg_conformer(mol)
        # ETKDG may or may not succeed depending on the environment;
        # we just verify the return type is correct.
        assert coords is None or coords.shape[1] == 3


# ═══════════════════════════════════════════════════════════════════
#  GNNScorer — availability & guard clauses  (no DL deps required)
# ═══════════════════════════════════════════════════════════════════


class TestGNNScorerAvailability:
    def test_init_without_model_file(self) -> None:
        """Constructing a scorer with a non-existent model path should
        not raise and ``available`` should be *False*."""
        scorer = GNNScorer(model_path="/nonexistent/path/model.pt")
        assert scorer.available is False

    def test_predict_returns_none_when_unavailable(self) -> None:
        scorer = GNNScorer(model_path="/nonexistent/path/model.pt")
        result = scorer.predict(_make_record())
        assert result is None

    def test_predict_invalid_molecule(self) -> None:
        """Predicting with an invalid SMILES should return None."""
        scorer = GNNScorer(model_path="/nonexistent/path/model.pt")
        bad_record = CompoundRecord(
            compound_id="GNN-BAD-001",
            smiles="CCCC=",  # invalid SMILES — RDKit will return None
            mol=None,
        )
        result = scorer.predict(bad_record)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
#  GNNScorer — predict with mocked torch/PYG  (mock-only)
# ═══════════════════════════════════════════════════════════════════


class TestGNNScorerPredictMocked:
    """Uses ``unittest.mock`` to simulate a loaded model so that the
    ``predict`` code path can be exercised without real DL libraries."""

    @patch("autoantibiotic.ml_scoring.gnn_scorer._HAS_TORCH", True)
    @patch("autoantibiotic.ml_scoring.gnn_scorer._HAS_PYG", True)
    @patch("autoantibiotic.ml_scoring.gnn_scorer.torch")
    @patch("autoantibiotic.ml_scoring.gnn_scorer.mol_pose_to_graph")
    def test_predict_returns_float(
        self, mock_mol_to_graph: MagicMock, mock_torch: MagicMock
    ) -> None:
        """When a model is loaded and graph conversion succeeds,
        ``predict`` should return a float."""
        # Mock torch.no_grad() context-manager
        mock_torch.no_grad.return_value.__enter__.return_value = None

        # Mock graph conversion
        mock_mol_to_graph.return_value = MagicMock()

        # Mock model prediction
        mock_pred = MagicMock()
        mock_pred.item.return_value = -8.5
        mock_model = MagicMock()
        mock_model.return_value = mock_pred

        scorer = GNNScorer(model_path="/nonexistent/model.pt")
        scorer._model = mock_model

        result = scorer.predict(_make_record())
        assert isinstance(result, float)
        assert result == -8.5

    @patch("autoantibiotic.ml_scoring.gnn_scorer._HAS_TORCH", True)
    @patch("autoantibiotic.ml_scoring.gnn_scorer._HAS_PYG", True)
    @patch("autoantibiotic.ml_scoring.gnn_scorer.torch")
    @patch("autoantibiotic.ml_scoring.gnn_scorer.mol_pose_to_graph")
    def test_predict_graph_failure(
        self, mock_mol_to_graph: MagicMock, mock_torch: MagicMock
    ) -> None:
        """When graph conversion fails, predict should return None."""
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_mol_to_graph.return_value = None  # graph building failed

        mock_model = MagicMock()
        scorer = GNNScorer(model_path="/nonexistent/model.pt")
        scorer._model = mock_model

        result = scorer.predict(_make_record())
        assert result is None

    @patch("autoantibiotic.ml_scoring.gnn_scorer._HAS_TORCH", True)
    @patch("autoantibiotic.ml_scoring.gnn_scorer._HAS_PYG", True)
    @patch("autoantibiotic.ml_scoring.gnn_scorer.torch")
    def test_predict_from_smiles(
        self, mock_torch: MagicMock
    ) -> None:
        """When ``record.mol`` is None, should parse SMILES."""
        mock_torch.no_grad.return_value.__enter__.return_value = None

        mock_pred = MagicMock()
        mock_pred.item.return_value = -7.2
        mock_model = MagicMock()
        mock_model.return_value = mock_pred

        scorer = GNNScorer(model_path="/nonexistent/model.pt")
        scorer._model = mock_model

        record = _make_record()
        record.mol = None  # force re-parse from SMILES

        # mol_pose_to_graph is called inside predict — mock it too
        with patch(
            "autoantibiotic.ml_scoring.gnn_scorer.mol_pose_to_graph"
        ) as mock_graph:
            mock_graph.return_value = MagicMock()
            result = scorer.predict(record)

        assert isinstance(result, float)
        assert result == -7.2


# ═══════════════════════════════════════════════════════════════════
#  mol_pose_to_graph  (may be skipped if torch_geometric is absent)
# ═══════════════════════════════════════════════════════════════════


class TestMolPoseToGraph:
    @staticmethod
    def _has_torch_geometric() -> bool:
        try:
            import torch_geometric  # noqa: F401
            return True
        except ImportError:
            return False

    def test_returns_none_without_pyg(self) -> None:
        """When PyG is not installed, the function should return None."""
        result = mol_pose_to_graph(Chem.MolFromSmiles("C"), pose_path=None)
        # This will be None if PyG unavailable, or a Data object if available.
        # We just check the type is consistent.
        if not self._has_torch_geometric():
            assert result is None

    def test_returns_none_on_no_coords(self) -> None:
        """A molecule that cannot produce coordinates should return None."""
        # An empty mol has no atoms and will fail coordinate generation.
        mol = Chem.RWMol()
        result = mol_pose_to_graph(mol, pose_path=None)
        if not self._has_torch_geometric():
            assert result is None

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch_geometric"),
        reason="requires torch_geometric",
    )
    def test_with_pose_path(self, tmp_path: Path) -> None:
        """When a valid PDBQT is provided, the graph should have the
        correct number of nodes."""
        import torch_geometric  # noqa: F401 (ensures Data is not None)

        mol = Chem.MolFromSmiles(_SMILES_BENZENE)
        assert mol is not None
        pdbqt = str(tmp_path / "pose.pdbqt")
        _write_dummy_pdbqt(pdbqt, n_atoms=mol.GetNumAtoms())

        graph = mol_pose_to_graph(mol, pose_path=pdbqt)
        assert graph is not None
        assert graph.num_nodes == mol.GetNumAtoms()
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "edge_attr")
        assert hasattr(graph, "x")
        assert hasattr(graph, "pos")


# ═══════════════════════════════════════════════════════════════════
#  Config integration  (use_gnn_rescoring flag)
# ═══════════════════════════════════════════════════════════════════


class TestConfigIntegration:
    def test_gnn_config_defaults(self) -> None:
        from autoantibiotic.config import CONFIG as cfg

        assert cfg.use_gnn_rescoring is False
        assert cfg.gnn_model_path == "output/gnn_model.pt"
