"""
AutoAntibiotic Discovery Pipeline v4.0
========================================
MRSA PBP2a Inhibitor Screening

Entry point: parses CLI arguments and delegates to PipelineOrchestrator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

from .config import CONFIG
from .orchestrator import PipelineOrchestrator


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments, configure, and run the pipeline.

    Usage::

        python -m autoantibiotic [--use-cache] [--dry-run]
    """
    parser = argparse.ArgumentParser(
        description="AutoAntibiotic Discovery Pipeline v3.2",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Skip re-docking if cache has results for a (compound_id, target) pair.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Limit library to 10 compounds and use mock docking energies.",
    )
    parser.add_argument(
        "--use-gnina", action="store_true",
        help="Use GNINA (deep-learning docking) instead of AutoDock Vina.",
    )
    parser.add_argument(
        "--ensemble-dir", type=str, default=None,
        help="Path to directory containing multiple receptor structures for ensemble docking.",
    )
    parser.add_argument(
        "--use-mm-gbsa", action="store_true",
        help="Use MM-GB/SA rescoring (requires OpenMM + AmberTools).",
    )
    parser.add_argument(
        "--use-mm-gbsa-rescoring", action="store_true",
        help="Use MM-GB/SA rescoring of top N docking candidates (alias for --use-mm-gbsa).",
    )
    parser.add_argument(
        "--flexible-docking", action="store_true",
        help="Enable induced-fit docking with flexible side-chain rotamers.",
    )
    parser.add_argument(
        "--no-water-analysis", action="store_true",
        help="Skip crystallographic water analysis.",
    )
    parser.add_argument(
        "--run-md-validation", action="store_true",
        help="Run short explicit-solvent MD validation on top candidates.",
    )
    parser.add_argument(
        "--use-mutation-sampling", action="store_true",
        help="Dock against mutant receptor variants to profile resistance risk.",
    )
    args = parser.parse_args(argv)

    # ── Apply CLI overrides to CONFIG ──
    if args.dry_run:
        CONFIG.dry_run = True
        CONFIG.library_target_count = 10

    if args.use_gnina:
        CONFIG.use_gnina = True

    if args.ensemble_dir:
        CONFIG.ensemble_mode = True
        CONFIG.ensemble_structures_dir = Path(args.ensemble_dir)

    if args.flexible_docking:
        CONFIG.flexible_docking = True

    if args.no_water_analysis:
        CONFIG.use_water_analysis = False

    if args.run_md_validation:
        CONFIG.md_validation_duration_ns = 10

    if args.use_mutation_sampling:
        CONFIG.use_mutation_sampling = True

    if args.use_mm_gbsa or args.use_mm_gbsa_rescoring:
        CONFIG.use_mm_gbsa = True
        CONFIG.use_mm_gbsa_rescoring = True
    orchestrator = PipelineOrchestrator(use_cache=args.use_cache)
    orchestrator.run()


if __name__ == "__main__":
    main()
