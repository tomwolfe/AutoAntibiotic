"""
AutoAntibiotic Discovery Pipeline v4.1
========================================
MRSA PBP2a Inhibitor Screening

Entry point: parses CLI arguments and delegates to PipelineOrchestrator.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Optional, List

from .config import CONFIG, PipelineConfig, ConfigurationError, PipelineProfile
from .io_utils import validate_pipeline_inputs, verify_dependencies
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
    parser.add_argument(
        "--use-fep-resistance", action="store_true",
        help="Use OpenMM FEP for resistance profiling (computationally expensive: requires "
        "OpenMM + openmmtools + openmmforcefields, ~hours per compound).",
    )
    parser.add_argument(
        "--use-explicit-solvent", action="store_true",
        help="Use explicit-solvent (TIP3P) MM-GB/SA rescoring with pose relaxation instead of "
        "implicit OBC2 (moderately expensive: ~minutes per compound with OpenMM).",
    )
    parser.add_argument(
        "--generative-mode", action="store_true",
        help="Use generative model (JT-VAE / graph-based) for novel scaffold design instead of "
        "BRICS recombination (computationally expensive: requires model inference).",
    )
    parser.add_argument(
        "--fep-top-n", type=int, default=None,
        help="Override CONFIG.fep_top_n — max number of top candidates for rigorous FEP.",
    )
    parser.add_argument(
        "--fep-ifp-threshold", type=float, default=None,
        help="Override CONFIG.fep_ifp_threshold — minimum IFP Tanimoto similarity to reference "
        "ligand for FEP pre-screening.",
    )
    parser.add_argument(
        "--validate-inputs", action="store_true",
        help="Validate all pipeline inputs (binaries, SMILES, directories) and exit. "
        "Returns exit code 0 on success, 1 if any issues are found.",
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="Print the final resolved configuration as JSON and exit.",
    )
    parser.add_argument(
        "--profile", type=str, default="standard",
        choices=[p.value for p in PipelineProfile],
        help="Pipeline profile: quick (lightweight test), standard (default), "
        "or production_fep (rigorous FEP-based resistance profiling).",
    )
    parser.add_argument(
        "--strict-deps", action="store_true",
        help="Enforce strict dependency checking: fail immediately if required "
        "binaries (Vina/GNINA) are missing.",
    )
    args = parser.parse_args(argv)

    # ── Create a local copy of CONFIG and apply CLI overrides ──
    cfg = copy.deepcopy(CONFIG)
    cfg.apply_profile(PipelineProfile(args.profile))

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

    if args.use_fep_resistance:
        cfg.use_fep_resistance = True

    if args.use_explicit_solvent:
        cfg.use_explicit_solvent_mmgbsa = True

    if args.generative_mode:
        cfg.generative_mode = True

    if args.fep_top_n is not None:
        cfg.fep_top_n = args.fep_top_n

    if args.fep_ifp_threshold is not None:
        cfg.fep_ifp_threshold = args.fep_ifp_threshold

    if args.strict_scoring:
        cfg.use_strict_scoring = True
        cfg.explicit_solvent_frames = 20

    if args.retrain_model is not None:
        cfg.retrain_model_path = args.retrain_model

    # ── Show resolved configuration and exit ──
    if args.show_config:
        print(json.dumps(cfg.__dict__, default=str, indent=2))
        raise SystemExit(0)

    # ── Validate inputs (if requested) ──
    if args.validate_inputs:
        strict_deps = (
            args.strict_deps
            or PipelineProfile(args.profile) == PipelineProfile.PRODUCTION_FEP
        )
        if strict_deps:
            verify_dependencies(cfg, strict=True)

        print("─── Input Validation Report ───")
        issues = validate_pipeline_inputs(cfg)
        has_errors = len(issues["errors"]) > 0
        has_warnings = len(issues["warnings"]) > 0

        if has_errors:
            print(f"\nErrors ({len(issues['errors'])}):")
            for err in issues["errors"]:
                print(f"  ✗  {err}")
        if has_warnings:
            print(f"\nWarnings ({len(issues['warnings'])}):")
            for warn in issues["warnings"]:
                print(f"  ⚠  {warn}")
        if not has_errors and not has_warnings:
            print("  ✓  All inputs are valid.")

        print(f"\nSummary: {len(issues['errors'])} error(s), "
              f"{len(issues['warnings'])} warning(s).")
        raise SystemExit(1 if has_errors else 0)

    # ── Validate configuration ──
    try:
        cfg.validate_config()
    except ConfigurationError as exc:
        print(f"Configuration Error: {exc}")
        raise

    # Always validate OpenMM deps when advanced features are explicitly
    # enabled, even during dry-run, so users get early feedback.
    if cfg.use_fep_resistance or cfg.use_explicit_solvent_mmgbsa:
        from .io_utils import check_openmm_availability
        omm_ok, omm_msg = check_openmm_availability()
        if not omm_ok:
            print(f"Dependency Error: {omm_msg}")
            raise ConfigurationError(omm_msg)

    orchestrator = PipelineOrchestrator(use_cache=args.use_cache, config=cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()
