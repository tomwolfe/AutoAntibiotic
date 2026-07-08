"""
Physics-Informed Graph Neural Network (GNN) Rescoring Module
=============================================================

Provides a GNN-based scorer that uses 3D molecular graphs from docked
poses to predict binding affinity.  Falls back to ETKDG conformers when
pose files are unavailable.

Requires ``torch`` and ``torch_geometric`` (install via
``pip install autoantibiotic[gnn]``).
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..config import CONFIG
from ..models import CompoundRecord

log = logging.getLogger(__name__)

# ── Lazy dependency checks ─────────────────────────────────────────
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_PYG = importlib.util.find_spec("torch_geometric") is not None

if _HAS_TORCH:
    import torch
else:
    torch = None  # type: ignore[assignment]

if _HAS_PYG:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GINConv, global_mean_pool
else:
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Data = None  # type: ignore[assignment]
    GINConv = None  # type: ignore[assignment]
    global_mean_pool = None  # type: ignore[assignment]

# ── Feature dimensions ─────────────────────────────────────────────
ATOM_FEATURE_DIM: int = 12  # 10 one-hot + degree + hybridisation
BOND_FEATURE_DIM: int = 4  # one-hot bond type


# ── Atom / bond feature helpers ────────────────────────────────────


def _one_hot_atomic_num(atomic_num: int) -> List[float]:
    """One-hot encode the most common elements found in drug-like molecules.

    Map: H→0, C→1, N→2, O→3, F→4, P→5, S→6, Cl→7, Br→8, I→9.
    """
    vec = [0.0] * 10
    common = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
    idx = common.get(atomic_num)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _get_atom_features(atom: Chem.Atom) -> List[float]:
    """Return a feature vector for a single RDKit atom.

    Concatenates a 10-element one-hot atomic-number vector with the
    atom's degree and hybridisation ordinal.
    """
    atomic_vec = _one_hot_atomic_num(atom.GetAtomicNum())
    degree = float(atom.GetDegree())
    hybridisation = float(atom.GetHybridization())
    return atomic_vec + [degree, hybridisation]


def _get_bond_features(bond: Chem.Bond) -> List[float]:
    """One-hot bond type: single, double, triple, aromatic."""
    bt = bond.GetBondType()
    return {
        Chem.BondType.SINGLE: [1.0, 0.0, 0.0, 0.0],
        Chem.BondType.DOUBLE: [0.0, 1.0, 0.0, 0.0],
        Chem.BondType.TRIPLE: [0.0, 0.0, 1.0, 0.0],
        Chem.BondType.AROMATIC: [0.0, 0.0, 0.0, 1.0],
    }.get(bt, [0.0, 0.0, 0.0, 0.0])


# ── 3D coordinate helpers ──────────────────────────────────────────


def _parse_pdbqt_coords(pose_path: str) -> Optional[np.ndarray]:
    """Extract heavy-atom 3D coordinates from a PDBQT file.

    Parameters
    ----------
    pose_path : str
        Path to a PDBQT file containing ``ATOM`` / ``HETATM`` records.

    Returns
    -------
    np.ndarray or None
        *(N, 3)* array of float32 coordinates, or *None* on failure.
    """
    try:
        coords: List[List[float]] = []
        with open(pose_path) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
        return np.array(coords, dtype=np.float32) if coords else None
    except (FileNotFoundError, ValueError, OSError):
        return None


def _generate_etkdg_conformer(mol: Chem.Mol) -> Optional[np.ndarray]:
    """Generate an ETKDG conformer and return its 3D coordinates.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule (will be hydrogenated internally).

    Returns
    -------
    np.ndarray or None
        *(N, 3)* array or *None* if embedding fails.
    """
    try:
        mol_with_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = CONFIG.random_seed
        result = AllChem.EmbedMolecule(mol_with_h, params)
        if result == 0:
            conf = mol_with_h.GetConformer()
            return np.array(
                [conf.GetAtomPosition(i) for i in range(mol_with_h.GetNumAtoms())],
                dtype=np.float32,
            )
    except Exception:
        pass
    return None


# ── Graph construction ─────────────────────────────────────────────


def mol_pose_to_graph(
    mol: Chem.Mol,
    pose_path: Optional[str] = None,
    pocket_residues: Optional[List[str]] = None,
) -> Optional[Data]:
    """Convert a molecule to a |torch_geometric| ``Data`` graph object.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    pose_path : str or None
        Path to a PDBQT file with the docked pose.  If *None* or missing,
        an ETKDG conformer is generated.
    pocket_residues : list of str or None
        Reserved for future pocket-aware encoding (currently unused).

    Returns
    -------
    torch_geometric.data.Data or None
        Graph with node features ``x``, 3D positions ``pos``, edge indices
        ``edge_index``, and edge features ``edge_attr``.  Returns *None*
        when |torch_geometric| is not installed or coordinate generation
        fails.
    """
    if Data is None:
        log.debug("torch_geometric not available; cannot build graph.")
        return None

    # 3D coordinates -------------------------------------------------
    coords = None
    if pose_path and os.path.exists(pose_path):
        coords = _parse_pdbqt_coords(pose_path)
    if coords is None:
        coords = _generate_etkdg_conformer(mol)
    if coords is None:
        log.warning("Could not obtain 3D coordinates for graph construction.")
        return None

    # Node features --------------------------------------------------
    atom_features = [_get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)  # type: ignore[union-attr]
    pos = torch.tensor(coords, dtype=torch.float)  # type: ignore[union-attr]

    # Edge indices & features ----------------------------------------
    edge_indices: List[List[int]] = []
    edge_features: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _get_bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # undirected
        edge_features.append(bf)
        edge_features.append(bf)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # type: ignore[union-attr]
    edge_attr = torch.tensor(edge_features, dtype=torch.float)  # type: ignore[union-attr]

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)  # type: ignore[union-attr]
    return data


# ── GNN model architecture (only defined when deps are available) ──

if _HAS_PYG and _HAS_TORCH:

    class _GNNModel(nn.Module):  # type: ignore[name-defined]
        """Graph Isomorphism Network (GIN) for binding-affinity regression.

        Three GINConv layers → global mean pool → MLP readout → scalar.
        """

        def __init__(
            self,
            node_dim: int = ATOM_FEATURE_DIM,
            edge_dim: int = BOND_FEATURE_DIM,
            hidden_dim: int = 64,
        ) -> None:
            super().__init__()
            self.conv1 = GINConv(  # type: ignore[union-attr]
                nn.Sequential(
                    nn.Linear(node_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=True,
            )
            self.conv2 = GINConv(  # type: ignore[union-attr]
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=True,
            )
            self.conv3 = GINConv(  # type: ignore[union-attr]
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=True,
            )
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, data: Data) -> torch.Tensor:  # type: ignore[name-defined]
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.conv1(x, edge_index)
            x = F.relu(x)  # type: ignore[name-defined]
            x = self.conv2(x, edge_index)
            x = F.relu(x)  # type: ignore[name-defined]
            x = self.conv3(x, edge_index)
            x = global_mean_pool(x, batch)  # type: ignore[union-attr]
            x = self.mlp(x)
            return x.squeeze(-1)

    _MODEL_CLS = _GNNModel
else:
    _MODEL_CLS = None  # type: ignore[assignment]


# ── Public scorer class ────────────────────────────────────────────


class GNNScorer:
    """Physics-Informed Graph Neural Network rescoring.

    Predicts binding affinity (kcal/mol) from 3D molecular graphs
    constructed from docked poses or ETKDG conformers.

    Parameters
    ----------
    model_path : str or None
        Path to a saved ``_GNNModel`` state dictionary.  Defaults to
        ``CONFIG.gnn_model_path``.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path: str = model_path or CONFIG.gnn_model_path
        self._model: Optional[object] = None
        self._load_model()

    # ── public API ──────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """``True`` when |torch_geometric| is installed and a model
        checkpoint has been loaded successfully."""
        return bool(_HAS_PYG and _HAS_TORCH and self._model is not None)

    def predict(self, record: CompoundRecord) -> Optional[float]:
        """Predict binding affinity for *record*.

        Parameters
        ----------
        record : CompoundRecord
            The compound to score.  Uses ``record.mol`` (or parses
            ``record.smiles``) and ``record.docked_pose_path``.

        Returns
        -------
        float or None
            Predicted binding affinity in kcal/mol (more negative means
            stronger predicted binder).  Returns *None* on any failure.
        """
        if not self.available:
            return None

        mol = record.mol
        if mol is None:
            try:
                mol = Chem.MolFromSmiles(record.smiles)
            except Exception:
                return None
        if mol is None:
            return None

        graph = mol_pose_to_graph(mol, pose_path=record.docked_pose_path)
        if graph is None:
            return None

        try:
            with torch.no_grad():  # type: ignore[union-attr]
                pred = self._model(graph)  # type: ignore[union-attr]
            return float(pred.item())
        except Exception as exc:
            log.warning("GNN prediction failed for %s: %s", record.compound_id, exc)
            return None

    # ── internal helpers ────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the trained GNN state dictionary from disk."""
        if not _HAS_PYG or not _HAS_TORCH or _MODEL_CLS is None:
            log.debug("GNN dependencies not available; skipping model load.")
            return
        if not os.path.exists(self._model_path):
            log.info("GNN model not found at '%s'.", self._model_path)
            return
        try:
            model = _MODEL_CLS()
            state = torch.load(  # type: ignore[union-attr]
                self._model_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)
            model.eval()
            self._model = model
            log.info("GNN model loaded from '%s'.", self._model_path)
        except Exception as exc:
            log.warning("Failed to load GNN model from '%s': %s", self._model_path, exc)
