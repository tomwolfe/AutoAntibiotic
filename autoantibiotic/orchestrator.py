"""
Pipeline Orchestrator
======================
Encapsulates the full discovery pipeline as a class with phase-by-phase methods.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CONFIG, PipelineConfig
from .models import CompoundRecord
from .docking import run_redocking_validation, screen_library
from .analysis import analyze_selectivity_and_resistance
from .io_utils import (
    ensure_output_dir,
    load_json_cache,
    log,
    save_json_cache,
    set_global_seed,
    verify_dependencies,
)
from .library_gen import apply_filters, generate_candidate_library, generate_pharmacophore_aware_library
from .reporting import generate_csv_report, generate_html_report, generate_images, print_summary
from .structure_prep import prepare_targets

try:
    from .water_analysis import analyze_waters
    _HAVE_WATER = True
except ImportError:
    _HAVE_WATER = False


class PipelineOrchestrator:
    """Manages the end-to-end AutoAntibiotic discovery pipeline.

    Each major phase is implemented as a method, allowing subclasses to
    override individual steps.

    Parameters
    ----------
    use_cache : bool
        Whether to persist docking results to a JSON cache.
    config : PipelineConfig
        Configuration object. Defaults to the module-level ``CONFIG`` singleton.
    """

    def __init__(self, use_cache: bool = False, config: PipelineConfig = CONFIG) -> None:
        self.use_cache = use_cache
        self.config = config
        self.cache: Dict[str, float] = {}
        self.deps: Dict[str, Any] = {}
        self.targets: Dict[str, Any] = {}
        self.water_results: Any = None
        self.all_records: List[CompoundRecord] = []
        self.filtered: List[CompoundRecord] = []
        self.top_candidates: List[CompoundRecord] = []
        self.validation_ok: bool = False
        self.redock_rmsd: Optional[float] = None
        self.n_total: int = 0
        self.n_filtered: int = 0

    def run(self) -> None:
        """Execute the full pipeline from preparation through reporting."""
        ensure_output_dir()
        self._setup_logging()

        log.info("─── AutoAntibiotic Pipeline v3.2 ───")

        self.prepare_environment()
        self.run_water_analysis()
        self.prepare_targets()
        self.run_redocking_validation()
        self.generate_and_filter_library()
        self.screen_candidates()
        self.analyze_selectivity()
        self.generate_reports()
        self._finalize()

    # ── Phase methods ──────────────────────────────────────────────

    def prepare_environment(self) -> None:
        """Phase 0: Seeding, cache loading, dependency check."""
        set_global_seed(self.config.random_seed)

        if self.use_cache:
            cache_path = self.config.output_dir / self.config.cache_name
            self.cache = load_json_cache(cache_path)
            log.info(f"  Cache loaded ({len(self.cache)} entries).")
        else:
            log.info("  Cache disabled. Use --use-cache to enable.")

        self.deps = verify_dependencies()
        os.makedirs(self.config.work_dir, exist_ok=True)

    def run_water_analysis(self) -> None:
        """Phase 0.5: Crystallographic water analysis (if available)."""
        self.water_results = None
        if not (self.config.use_water_analysis and _HAVE_WATER):
            if self.config.use_water_analysis:
                log.info("  Water analysis module not available (install Bio.PDB).")
            return

        pdb_dir = self.config.pdb_dir
        holo_pdb_id = self.config.pdb_ids.get("PBP2a_holo", "6TKO")
        holo_pdb_path = os.path.join(str(pdb_dir), f"{holo_pdb_id}.pdb")

        if os.path.exists(holo_pdb_path):
            log.info("─── Phase 0.5: Crystallographic Water Analysis ───")
            self._analyze_waters(holo_pdb_path)
        else:
            log.info("  Holo PDB not yet downloaded (will analyse after Phase 1).")

    def prepare_targets(self) -> None:
        """Phase 1: Download and prepare receptor structures."""
        pdb_dir = str(self.config.pdb_dir)
        work_dir = str(self.config.work_dir)
        self.targets = prepare_targets(pdb_dir, work_dir, self.deps, water_results=self.water_results)

        if (
            self.config.use_water_analysis
            and _HAVE_WATER
            and self.water_results is None
        ):
            holo_pdb_path = self.targets.get("holo_pdb", "")
            if holo_pdb_path and os.path.exists(holo_pdb_path):
                log.info("─── Phase 0.5 (retry): Crystallographic Water Analysis ───")
                self._analyze_waters(holo_pdb_path)

    def run_redocking_validation(self) -> None:
        """Phase 0 (after target prep): protocol validation via redocking."""
        self.validation_ok, self.redock_rmsd = run_redocking_validation(
            holo_pdb_path=self.targets["holo_pdb"],
            target_pdbqt_path=self.targets["PBP2a"]["pdbqt"],
            work_dir=str(self.config.work_dir),
            deps=self.deps,
            center=self.targets["PBP2a"]["active_center"],
        )

    def generate_and_filter_library(self) -> None:
        """Phase 2: Compound library generation and filtering."""
        log.info("─── Phase 2: Library Generation & Filtering ───")
        if self.config.use_pharmacophore_filter:
            log.info("  Pharmacophore-constrained library generation enabled.")
            self.all_records = generate_pharmacophore_aware_library(
                target_count=self.config.library_target_count,
                seed=self.config.random_seed,
            )
        else:
            self.all_records = list(
                generate_candidate_library(target_count=self.config.library_target_count)
            )
        self.n_total = len(self.all_records)

        self.filtered = apply_filters(self.all_records)
        self.n_filtered = len(self.filtered)

        if self.n_filtered == 0:
            log.warning("  No compounds passed filters. Halting pipeline.")
            raise SystemExit(0)

    def screen_candidates(self) -> None:
        """Phase 3: Virtual screening (docking + ML rescoring)."""
        self.top_candidates = screen_library(
            self.filtered, self.targets, str(self.config.work_dir),
            self.deps, cache=self.cache, use_cache=self.use_cache,
            water_results=self.water_results,
        )

        if not self.top_candidates:
            log.warning("  No candidates after screening. Halting pipeline.")
            raise SystemExit(0)

    def analyze_selectivity(self) -> None:
        """Phase 4: Selectivity filtering and resistance analysis."""
        self.top_candidates = analyze_selectivity_and_resistance(
            self.top_candidates, self.targets, str(self.config.work_dir),
            self.deps, cache=self.cache, use_cache=self.use_cache,
        )

    def generate_reports(self) -> None:
        """Phase 5: CSV, images, and HTML report generation."""
        generate_csv_report(self.top_candidates)

        top3 = self.top_candidates[:self.config.top_n_for_images]
        generate_images(top3)

        scored_for_top50 = [
            r for r in self.filtered
            if r.pb2pa_allosteric_energy is not None
        ]
        scored_for_top50.sort(key=lambda r: r.pb2pa_allosteric_energy)
        top_n = self.config.top_n_for_html_report
        top50 = (
            scored_for_top50[:top_n]
            if len(scored_for_top50) >= top_n
            else scored_for_top50
        )

        generate_html_report(self.top_candidates, top50, self.config.output_dir)

    # ── Internal helpers ───────────────────────────────────────────

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.output_dir / self.config.pipeline_log_name),
            ],
        )

    def _analyze_waters(self, holo_pdb_path: str) -> None:
        try:
            self.water_results = analyze_waters(
                holo_pdb_path,
                allosteric_residues=self.config.flexible_residues_allosteric,
                active_site_residues=self.config.flexible_residues_active,
                distance_cutoff=self.config.water_distance_cutoff,
                displacement_energy_threshold=self.config.water_displacement_energy_threshold,
            )
            if self.water_results and self.water_results.high_energy_waters:
                log.info(
                    f"  -> {len(self.water_results.high_energy_waters)} high-energy "
                    f"waters flagged for displacement; "
                    f"{len(self.water_results.bridging_waters)} bridging waters retained."
                )
        except Exception as exc:
            log.warning(f"  Water analysis failed: {exc}")

    def _finalize(self) -> None:
        """Save cache and print summary."""
        if self.use_cache:
            cache_path = self.config.output_dir / self.config.cache_name
            save_json_cache(cache_path, self.cache)
            log.info(f"  Cache saved ({len(self.cache)} entries).")

        print_summary(
            self.n_total, self.n_filtered, self.top_candidates,
            self.validation_ok, self.redock_rmsd, self.deps,
        )

        log.info("Pipeline complete. Exiting.")
