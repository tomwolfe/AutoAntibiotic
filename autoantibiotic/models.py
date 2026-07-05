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
