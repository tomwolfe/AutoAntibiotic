"""
PipelineContext — DEPRECATED
=============================

``PipelineContext`` has been replaced by a plain ``dict`` managed
directly within :class:`~autoantibiotic.orchestrator.PipelineOrchestrator`.

This module is kept for backward compatibility but will be removed in a
future release. Importing ``PipelineContext`` now emits a deprecation
warning.
"""

from __future__ import annotations

import warnings as _warnings

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .io_utils import PipelineAudit
from .models import CompoundRecord


_warnings.warn(
    "PipelineContext is deprecated. "
    "Use the state dict managed by PipelineOrchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class PipelineContext:
    library: List[CompoundRecord] = field(default_factory=list)
    filtered_library: List[CompoundRecord] = field(default_factory=list)
    docked_candidates: List[CompoundRecord] = field(default_factory=list)
    md_results: List[CompoundRecord] = field(default_factory=list)
    fep_results: List[CompoundRecord] = field(default_factory=list)
    audit_log: Optional[PipelineAudit] = None
    n_total: int = 0
    n_filtered: int = 0
