"""
AutoAntibiotic Discovery Pipeline v3.2
========================================
MRSA PBP2a Inhibitor Screening

Entry point: parses CLI arguments and delegates to PipelineOrchestrator.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "--benchmark", action="store_true",
        help="Run enrichment benchmark instead of full library screening.",
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
    args = parser.parse_args(argv)

    # ── Apply CLI overrides to CONFIG ──
    if args.benchmark:
        CONFIG.benchmark_mode = True

    if args.dry_run:
        CONFIG.dry_run = True
        CONFIG.library_target_count = 10

    if args.use_gnina:
        CONFIG.docking.use_gnina = True

    if args.ensemble_dir:
        CONFIG.docking.ensemble_mode = True
        CONFIG.docking.ensemble_structures_dir = Path(args.ensemble_dir)

    if args.flexible_docking:
        CONFIG.docking.flexible_docking = True

    if args.no_water_analysis:
        CONFIG.water.use_water_analysis = False

    if args.use_mm_gbsa or args.use_mm_gbsa_rescoring:
        CONFIG.docking.use_mm_gbsa = True
        CONFIG.docking.use_mm_gbsa_rescoring = True

    # ── Benchmark mode (early exit) ──
    if args.benchmark:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(CONFIG.output_dir / CONFIG.reporting.pipeline_log_name),
            ],
        )
        from .io_utils import log
        log.info("─── Benchmark Mode ───")
        try:
            from benchmarks.run_enrichment_test import run_enrichment_test
            results = run_enrichment_test(
                n_decoys_per_active=CONFIG.benchmark_n_decoys,
                use_vina=CONFIG.docking.use_gnina,
            )
            log.info(f"  Benchmark complete: EF1%={results['ef1']:.3f}, "
                     f"ROC-AUC={results['roc_auc']:.3f}")
        except Exception as exc:
            log.error(f"  Benchmark failed: {exc}")
        log.info("Benchmark mode complete. Exiting.")
        return

    # ── Normal pipeline execution ──
    orchestrator = PipelineOrchestrator(use_cache=args.use_cache)
    orchestrator.run()


if __name__ == "__main__":
    main()
