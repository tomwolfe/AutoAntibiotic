from typing import Any, Dict, List

from ..config import PipelineConfig
from ..models import CompoundRecord
from ..reporting import generate_csv_report, generate_html_report, generate_images
from .base import PhaseHandler


class ReportingHandler(PhaseHandler):
    def execute(self, state: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
        candidates: List[CompoundRecord] = state.get("docked_candidates", [])
        filtered_library: List[CompoundRecord] = state.get("filtered_library", [])

        generate_csv_report(candidates)

        top3 = candidates[:config.top_n_for_images]
        generate_images(top3)

        scored_for_top50 = [
            r for r in filtered_library
            if r.pb2pa_allosteric_energy is not None
        ]
        scored_for_top50.sort(key=lambda r: r.pb2pa_allosteric_energy)
        top_n = config.top_n_for_html_report
        top50 = (
            scored_for_top50[:top_n]
            if len(scored_for_top50) >= top_n
            else scored_for_top50
        )

        generate_html_report(candidates, top50, config.output_dir)
        return state
