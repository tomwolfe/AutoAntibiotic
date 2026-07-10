from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .io_utils import PipelineAudit
from .models import CompoundRecord


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
