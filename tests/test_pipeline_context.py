from __future__ import annotations

from unittest.mock import MagicMock

from autoantibiotic.pipeline_context import PipelineContext


class TestPipelineContext:
    """Verify PipelineContext dataclass state passing."""

    def test_default_initialization(self):
        ctx = PipelineContext()
        assert ctx.library == []
        assert ctx.filtered_library == []
        assert ctx.docked_candidates == []
        assert ctx.md_results == []
        assert ctx.fep_results == []
        assert ctx.audit_log is None
        assert ctx.n_total == 0
        assert ctx.n_filtered == 0

    def test_custom_values(self):
        mock_audit = MagicMock()
        ctx = PipelineContext(
            library=["lib1", "lib2"],
            filtered_library=["filt1"],
            docked_candidates=["docked1"],
            md_results=["md1"],
            fep_results=["fep1"],
            audit_log=mock_audit,
            n_total=10,
            n_filtered=5,
        )
        assert ctx.library == ["lib1", "lib2"]
        assert ctx.filtered_library == ["filt1"]
        assert ctx.docked_candidates == ["docked1"]
        assert ctx.md_results == ["md1"]
        assert ctx.fep_results == ["fep1"]
        assert ctx.audit_log is mock_audit
        assert ctx.n_total == 10
        assert ctx.n_filtered == 5

    def test_immutability_of_defaults(self):
        ctx = PipelineContext()
        ctx.library.append("new")
        assert ctx.library == ["new"]
        ctx.filtered_library.append("filt")
        assert ctx.filtered_library == ["filt"]

    def test_state_passing_through_pipeline(self):
        ctx = PipelineContext()
        ctx = self._phase_one(ctx)
        ctx = self._phase_two(ctx)
        assert ctx.library == ["comp1", "comp2"]
        assert ctx.filtered_library == ["comp1"]
        assert ctx.docked_candidates == ["comp1"]

    def _phase_one(self, ctx: PipelineContext) -> PipelineContext:
        ctx.library = ["comp1", "comp2"]
        return ctx

    def _phase_two(self, ctx: PipelineContext) -> PipelineContext:
        ctx.filtered_library = [c for c in ctx.library if c != "comp2"]
        ctx.docked_candidates = ctx.filtered_library
        return ctx

    def test_fep_results_independent_from_docked(self):
        ctx = PipelineContext()
        ctx.docked_candidates = ["a", "b", "c"]
        ctx.fep_results = ["a"]
        assert len(ctx.docked_candidates) == 3
        assert len(ctx.fep_results) == 1
