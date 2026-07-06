"""
AutoAntibiotic Discovery Pipeline v4.1
========================================
MRSA PBP2a Inhibitor Screening

Entry point: parses CLI arguments and delegates to PipelineOrchestrator.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Optional, List

from .config import CONFIG, PipelineConfig, ConfigurationError
from .orchestrator import PipelineOrchestrator


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments, configure, and run the pipeline.

    Creates a local copy of the module-level CONFIG using
    ``dataclasses.replace``, applies CLI overrides to the copy, and
    passes it to the orchestrator — avoiding side effects on the global
    singleton.
    """
    parser = argparse.ArgumentParser(
        description="AutoAntibiotic Discovery Pipeline v4.1",
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
        "--strict-scoring", action="store_true",
        help="Force explicit-solvent MM-GB/SA and increase explicit_solvent_frames to 20 for higher precision.",
    )
    parser.add_argument(
        "--retrain-model", type=str,
        help="Path to CSV file with {smiles, ic50} for active-learning retraining.",
    )
    parser.add_argument(
        "--use-mutation-sampling", action="store_true",
        help="Dock against mutant receptor variants to profile resistance risk.",
    )
    args = parser.parse_args(argv)

    # ── Create a local copy of CONFIG and apply CLI overrides ──
    cfg = copy.deepcopy(CONFIG)

    if args.dry_run:
        cfg.dry_run = True
        cfg.library_target_count = 10

    if args.use_gnina:
        cfg.use_gnina = True

    if args.ensemble_dir:
        cfg.ensemble_mode = True
        cfg.ensemble_structures_dir = Path(args.ensemble_dir)

    if args.flexible_docking:
        cfg.flexible_docking = True

    if args.no_water_analysis:
        cfg.use_water_analysis = False

    if args.run_md_validation:
        cfg.md_validation_duration_ns = 10

    if args.use_mutation_sampling:
        cfg.use_mutation_sampling = True

    if args.strict_scoring:
        cfg.use_strict_scoring = True
        cfg.explicit_solvent_frames = 20

    if args.retrain_model is not None:
        cfg.retrain_model_path = args.retrain_model

    # ── Validate configuration ──
    try:
        cfg.validate_config()
    except ConfigurationError as exc:
        print(f"Configuration Error: {exc}")
        raise

    orchestrator = PipelineOrchestrator(use_cache=args.use_cache, config=cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()
