from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rdkit import Chem


@dataclass
class ToolResult:
    """Result from an external tool execution."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class CompoundRecord:
    """Stores all computed properties for a single candidate."""
    compound_id: str
    smiles: str
    mol: Optional[Chem.Mol] = None

    pb2pa_allosteric_energy: Optional[float] = None
    pb2pa_active_energy: Optional[float] = None
    human_trypsin_energy: Optional[float] = None
    human_ces1_energy: Optional[float] = None

    selectivity_index: Optional[float] = None

    max_similarity: float = 0.0

    passes_lipinski: bool = False
    qed_score: float = 0.0
    passes_pains: bool = False

    resistance_notes: str = ""
    resistance_stability_score: Optional[float] = None

    ifp_score: Optional[float] = None

    shape_score: Optional[float] = None

    ml_score: Optional[float] = None
    ml_score_std: Optional[float] = None
    admet_flags: List[str] = field(default_factory=list)
    has_undefined_stereo: bool = False
    needs_manual_review: bool = False
    parent_id: Optional[str] = None

    # MD-derived dynamic stability features (for MetaScorer)
    md_ligand_rmsd: Optional[float] = None
    """Ligand heavy-atom RMSD (Å) from MD simulation, mean across frames."""
    md_pocket_rg_stability: Optional[float] = None
    """Fractional pocket Rg change from MD simulation (0=stable)."""
    md_converged: bool = False
    """Whether the MD simulation reached convergence during adaptive
    sampling (RMSD std + Rg std below threshold)."""

    water_displacement_energy: Optional[float] = None
    """Water displacement energy score from crystallographic water analysis."""

    docked_pose_path: Optional[str] = None
    """Path to the best docked pose PDBQT file for interaction fingerprint analysis."""
