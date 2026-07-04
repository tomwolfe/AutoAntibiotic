"""
AutoAntibiotic Discovery Pipeline v3.2
========================================
MRSA PBP2a Inhibitor Screening

Screens novel small-molecule libraries against MRSA PBP2a (allosteric + active sites)
with selectivity filtering against human serine hydrolases, ADMET profiling, and
resistance-risk analysis.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CONFIG
from .docking import run_redocking_validation, screen_library
from .analysis import analyze_selectivity_and_resistance
from .io_utils import (
    CacheManager,
    ensure_output_dir,
    log,
    set_global_seed,
    verify_dependencies,
)
from .library_gen import apply_filters, generate_candidate_library, generate_pharmacophore_aware_library
from .reporting import generate_csv_report, generate_html_report, generate_images, print_summary
from .structure_prep import prepare_targets


def main(argv: Optional[List[str]] = None) -> None:
    """Orchestrate the full discovery pipeline end-to-end.

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
    args = parser.parse_args(argv)

    if args.dry_run:
        CONFIG.dry_run = True
        CONFIG.library_target_count = 10

    if args.use_gnina:
        CONFIG.use_gnina = True

    if args.ensemble_dir:
        CONFIG.ensemble_mode = True
        CONFIG.ensemble_structures_dir = Path(args.ensemble_dir)

    ensure_output_dir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(CONFIG.output_dir / CONFIG.pipeline_log_name),
        ],
    )

    # Phase 0: Deterministic seeding
    set_global_seed(CONFIG.random_seed)

    use_cache = args.use_cache
    cache: Optional[CacheManager] = None
    if use_cache:
        cache = CacheManager(str(CONFIG.output_dir / CONFIG.cache_db_name))
        log.info(f"  Cache loaded ({len(cache)} entries).")
    else:
        log.info("  Cache disabled. Use --use-cache to enable.")

    deps = verify_dependencies()

    work_dir = str(CONFIG.work_dir)
    pdb_dir = str(CONFIG.pdb_dir)
    os.makedirs(work_dir, exist_ok=True)

    # ── Phase 1: Target preparation ──
    targets = prepare_targets(pdb_dir, work_dir, deps)

    # ── Phase 0: Redocking validation ──
    validation_ok, redock_rmsd = run_redocking_validation(
        holo_pdb_path=targets["holo_pdb"],
        target_pdbqt_path=targets["PBP2a"]["pdbqt"],
        work_dir=work_dir,
        deps=deps,
        center=targets["PBP2a"]["active_center"],
    )

    # ── Phase 2: Library generation & filtering ──
    if CONFIG.use_pharmacophore_filter:
        log.info("  Pharmacophore-constrained library generation enabled.")
        all_records = generate_pharmacophore_aware_library(
            target_count=CONFIG.library_target_count,
            seed=CONFIG.random_seed,
        )
    else:
        all_records = list(
            generate_candidate_library(target_count=CONFIG.library_target_count)
        )
    n_total = len(all_records)

    filtered = apply_filters(all_records)
    n_filtered = len(filtered)

    if n_filtered == 0:
        log.warning("  No compounds passed filters. Halting pipeline.")
        return

    # ── Phase 3: Virtual screening ──
    top10 = screen_library(filtered, targets, work_dir, deps, cache=cache, use_cache=use_cache)

    if not top10:
        log.warning("  No candidates after screening. Halting pipeline.")
        return

    # ── Phase 4: Selectivity & Resistance ──
    top10 = analyze_selectivity_and_resistance(
        top10, targets, work_dir, deps, cache=cache, use_cache=use_cache,
    )

    # ── Phase 5: Reporting & Artifacts ──
    generate_csv_report(top10)

    top3 = top10[:CONFIG.top_n_for_images]
    generate_images(top3)

    scored_for_top50 = [
        r for r in filtered
        if r.pb2pa_allosteric_energy is not None
    ]
    scored_for_top50.sort(key=lambda r: r.pb2pa_allosteric_energy)
    top50 = scored_for_top50[:CONFIG.top_n_for_html_report] if len(scored_for_top50) >= CONFIG.top_n_for_html_report else scored_for_top50

    generate_html_report(top10, top50, CONFIG.output_dir)

    if use_cache and cache is not None:
        cache.close()
        log.info(f"  Cache saved ({cache.__len__()} entries).")

    print_summary(
        n_total, n_filtered, top10,
        validation_ok, redock_rmsd, deps,
    )

    log.info("Pipeline complete. Exiting.")


if __name__ == "__main__":
    main()
