"""Docking engine abstraction layer for AutoAntibiotic.

This package provides docking engine abstractions and compound-level
docking operations, along with pipeline-level screening orchestration.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from ..config import CONFIG, PipelineConfig
from ..models import CompoundRecord
from ..io_utils import PipelineAudit

try:
    from ..water_analysis import WaterAnalysisResult
except ImportError:
    WaterAnalysisResult = None  # type: ignore

from .base import DockingEngine
from .engines import VinaEngine, GninaEngine, RdkitShapeEngine
from .compound import (
    dock_compound,
    dock_compound_ensemble,
    prepare_ligand_pdbqt,
)
from .pipeline import (
    _apply_consensus_scoring,
    _apply_ml_rescoring,
    _build_flexible_ensemble,
    _compute_dynamic_box_params,
    _compute_rank_consensus,
    _compute_rmsd_docked_vs_crystal,
    _compute_shape_fallback_score,
    _extract_native_ligand_from_holo,
    _generate_flexible_conformers,
    _parallel_dock,
    _parallel_dock_ensemble,
    _prepare_flexible_receptors,
    _screen_ensemble,
    _screen_flexible,
    _screen_shape_fallback,
    _screen_standard,
    _worker_dock_wrapper,
    run_redocking_validation,
    screen_library,
)

__all__ = [
    "DockingEngine",
    "VinaEngine",
    "GninaEngine",
    "RdkitShapeEngine",
    "get_engine",
    "dock_compound",
    "dock_compound_ensemble",
    "prepare_ligand_pdbqt",
    "run_redocking_validation",
    "screen_library",
    "_parallel_dock",
]


def get_engine(name: str, config: Optional[PipelineConfig] = None) -> DockingEngine:
    """Factory: return a DockingEngine implementation by name.

    Args:
        name: ``"vina"``, ``"gnina"``, or ``"shape"``.
        config: Optional pipeline config (defaults to module-level CONFIG).

    Returns:
        A :class:`DockingEngine` instance ready for docking.
    """
    cfg = config or CONFIG
    if name == "vina":
        return VinaEngine(cfg)
    elif name == "gnina":
        return GninaEngine(cfg)
    else:
        return RdkitShapeEngine(cfg)
