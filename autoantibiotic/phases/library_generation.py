from typing import Any, Dict, List, Optional

from ..config import PipelineConfig
from ..io_utils import log, PipelineAudit
from ..library_gen import (
    apply_filters,
    generate_candidate_library,
    generate_pharmacophore_aware_library,
)
from ..models import CompoundRecord
from .base import PhaseHandler


class LibraryGenerationHandler(PhaseHandler):
    def execute(self, state: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
        log.info("─" * 3 + " Phase 2: Library Generation & Filtering " + "─" * 3)

        audit: Optional[PipelineAudit] = state.get("audit")

        if config.use_pharmacophore_filter:
            log.info("  Pharmacophore-constrained library generation enabled.")
            all_records: List[CompoundRecord] = list(
                generate_pharmacophore_aware_library(
                    target_count=config.library_target_count,
                    seed=config.random_seed,
                    config=config,
                )
            )
        else:
            all_records = list(
                generate_candidate_library(
                    target_count=config.library_target_count,
                    config=config,
                )
            )

        state["library"] = all_records
        state["n_total"] = len(all_records)

        filtered = apply_filters(all_records, audit=audit, config=config)
        state["filtered_library"] = filtered
        state["n_filtered"] = len(filtered)

        if audit is not None:
            audit.check_health(state["n_total"], "Library Filtering")

        if len(filtered) == 0:
            log.warning("  No compounds passed filters. Halting pipeline.")
            raise SystemExit(0)

        return state
