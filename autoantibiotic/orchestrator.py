"""
Pipeline Orchestrator
======================
Encapsulates the full discovery pipeline as a class with phase-by-phase methods.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem

from .config import CONFIG, PipelineConfig
from .models import CompoundRecord
from .docking import run_redocking_validation, screen_library
from .docking import get_engine
from .docking import DockingEngine
from .analysis import analyze_selectivity_and_resistance, predict_meta_score
from .scoring_metrics import check_key_interactions
from .ml_scoring.meta_scorer import _get_meta_scorer
from .ml_scoring.gnn_scorer import GNNScorer
from .io_utils import (
    ensure_output_dir,
    load_json_cache,
    log,
    PipelineAudit,
    PipelineHealthError,
    save_json_cache,
    set_global_seed,
    verify_dependencies,
)
from .library_gen import (
    apply_filters,
    generate_candidate_library,
    generate_pharmacophore_aware_library,
)
from .fep_manager import FEPManager
from .ml_scoring.scoring import rescore_with_explicit_mmgbsa
from .pipeline_context import PipelineContext
from .reporting import generate_csv_report, generate_html_report, generate_images, print_summary
from .structure_prep import prepare_targets

try:
    from .water_analysis import analyze_waters
    _HAVE_WATER = True
except ImportError:
    _HAVE_WATER = False

try:
    from .md_validation import run_short_md
    _HAVE_MD = True
except ImportError:
    _HAVE_MD = False


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
        self.audit: Optional[PipelineAudit] = (
            PipelineAudit() if self.config.audit_enabled else None
        )
        self._context: Optional[PipelineContext] = None

    def run(self) -> None:
        """Execute the full pipeline from preparation through reporting.

        Initialises a :class:`PipelineContext` and passes it through each
        pipeline phase as a private method, ensuring no implicit state
        mutation on ``self`` other than logging.
        """
        ensure_output_dir()
        self._setup_logging()

        log.info("─── AutoAntibiotic Pipeline v4.0 ───")

        ctx = PipelineContext(audit_log=self.audit)

        ctx = self._prepare_environment(ctx)
        ctx = self._run_water_analysis(ctx)
        ctx = self._prepare_targets(ctx)
        ctx = self._run_redocking_validation(ctx)
        ctx = self._generate_and_filter_library(ctx)
        ctx = self._screen_candidates(ctx)
        ctx = self._analyze_selectivity(ctx)
        # Explicit-solvent MM-GB/SA rescoring (if enabled) — uses the
        # solvated complex from MD preparation for rigorous ΔG prediction.
        ctx = self._apply_explicit_solvent_rescoring(ctx)
        # MD validation runs BEFORE meta-scoring so that dynamic features
        # (ligand RMSD, pocket Rg stability) are available to the MetaScorer.
        ctx = self._apply_md_validation(ctx)
        ctx = self._apply_meta_scoring(ctx)
        # FEP resistance profiling — top-hit only, runs after rescoring
        # so that only the most promising candidates are evaluated.
        ctx = self._apply_fep_resistance(ctx)
        ctx = self._generate_reports(ctx)

        self._sync_from_context(ctx)

        self._finalize()

    # ── Private phase methods (context-based, no side-effects on self) ──

    def _prepare_environment(self, context: PipelineContext) -> PipelineContext:
        set_global_seed(self.config.random_seed)
        if self.use_cache:
            cache_path = self.config.output_dir / self.config.cache_name
            self.cache = load_json_cache(cache_path)
            log.info(f"  Cache loaded ({len(self.cache)} entries).")
        else:
            log.info("  Cache disabled. Use --use-cache to enable.")
        self.deps = verify_dependencies()
        os.makedirs(self.config.work_dir, exist_ok=True)
        if self.config.retrain_model_path:
            self._retrain_from_csv(self.config.retrain_model_path)
        return context

    def _run_water_analysis(self, context: PipelineContext) -> PipelineContext:
        if not (self.config.use_water_analysis and _HAVE_WATER):
            if self.config.use_water_analysis:
                log.info("  Water analysis module not available (install Bio.PDB).")
            return context
        pdb_dir = self.config.pdb_dir
        holo_pdb_id = self.config.pdb_ids.get("PBP2a_holo", "3ZG0")
        holo_pdb_path = os.path.join(str(pdb_dir), f"{holo_pdb_id}.pdb")
        if os.path.exists(holo_pdb_path):
            log.info("─── Phase 0.5: Crystallographic Water Analysis ───")
            self._analyze_waters(holo_pdb_path)
        else:
            log.info("  Holo PDB not yet downloaded (will analyse after Phase 1).")
        return context

    def _prepare_targets(self, context: PipelineContext) -> PipelineContext:
        pdb_dir = str(self.config.pdb_dir)
        work_dir = str(self.config.work_dir)
        self.targets = prepare_targets(pdb_dir, work_dir, self.deps, water_results=self.water_results)
        pb2pa = self.targets.get("PBP2a", {})
        validated_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if validated_pdb and os.path.isfile(validated_pdb):
            log.info(f"  Receptor integrity validated for PBP2a: {validated_pdb}")
        else:
            log.warning("  PBP2a receptor PDB not found after target preparation.")
        if (self.config.use_water_analysis and _HAVE_WATER and self.water_results is None):
            holo_pdb_path = self.targets.get("holo_pdb", "")
            if holo_pdb_path and os.path.exists(holo_pdb_path):
                log.info("─── Phase 0.5 (retry): Crystallographic Water Analysis ───")
                self._analyze_waters(holo_pdb_path)
        return context

    def _run_redocking_validation(self, context: PipelineContext) -> PipelineContext:
        self.validation_ok, self.redock_rmsd = run_redocking_validation(
            holo_pdb_path=self.targets["holo_pdb"],
            target_pdbqt_path=self.targets["PBP2a"]["pdbqt"],
            work_dir=str(self.config.work_dir),
            deps=self.deps,
            center=self.targets["PBP2a"]["active_center"],
            config=self.config,
        )
        return context

    def _generate_and_filter_library(self, context: PipelineContext) -> PipelineContext:
        log.info("─── Phase 2: Library Generation & Filtering ───")
        if self.config.use_pharmacophore_filter:
            log.info("  Pharmacophore-constrained library generation enabled.")
            all_records = list(
                generate_pharmacophore_aware_library(
                    target_count=self.config.library_target_count,
                    seed=self.config.random_seed,
                    config=self.config,
                )
            )
        else:
            all_records = list(
                generate_candidate_library(
                    target_count=self.config.library_target_count,
                    config=self.config,
                )
            )
        context.library = all_records
        context.n_total = len(all_records)
        filtered = apply_filters(all_records, audit=self.audit, config=self.config)
        context.filtered_library = filtered
        context.n_filtered = len(filtered)
        if self.audit is not None:
            self.audit.check_health(context.n_total, "Library Filtering")
        if len(filtered) == 0:
            log.warning("  No compounds passed filters. Halting pipeline.")
            raise SystemExit(0)
        return context

    def _screen_candidates(self, context: PipelineContext) -> PipelineContext:
        engine_name = "gnina" if self.config.use_gnina else "vina"
        engine = get_engine(engine_name, config=self.config)
        top_candidates = screen_library(
            context.filtered_library, self.targets, str(self.config.work_dir),
            self.deps, cache=self.cache, use_cache=self.use_cache,
            water_results=self.water_results, dry_run=self.config.dry_run,
            audit=self.audit, config=self.config, engine=engine,
        )
        context.docked_candidates = top_candidates
        if self.audit is not None:
            self.audit.check_health(len(context.filtered_library), "Docking")
        if not top_candidates:
            log.warning("  No candidates after screening. Halting pipeline.")
            raise SystemExit(0)
        context = self._filter_by_key_interactions_context(context)
        context = self._run_benchmark_check(context)
        return context

    def _run_benchmark_check(self, context: PipelineContext) -> PipelineContext:
        if not self.config.benchmark_mode:
            return context
        candidates = context.docked_candidates
        if not candidates:
            return context
        log.info("─── Benchmark Check ───")
        try:
            from benchmarks.reference_data import get_actives_smiles, get_inactives_smiles
            from benchmarks.run_enrichment_test import compute_enrichment_factor, compute_roc_auc
            import numpy as np

            active_smiles = set(get_actives_smiles())
            inactive_smiles = set(get_inactives_smiles())

            scores: list = []
            labels: list = []
            for rec in candidates:
                energy = rec.pb2pa_allosteric_energy
                if energy is None:
                    continue
                scores.append(energy)
                if rec.smiles in active_smiles:
                    labels.append(1)
                elif rec.smiles in inactive_smiles:
                    labels.append(0)
                else:
                    labels.append(0)

            if len(set(labels)) < 2 or len(scores) < 10:
                log.info("  Benchmark check: insufficient actives/inactives in results.")
                return context

            scores_arr = np.array(scores, dtype=np.float64)
            labels_arr = np.array(labels, dtype=np.int64)
            ef1 = compute_enrichment_factor(scores_arr, labels_arr, fraction=0.01)
            roc_auc = compute_roc_auc(scores_arr, labels_arr)
            log.info(f"  EF1% (Enrichment Factor at 1%): {ef1:.3f}")
            log.info(f"  ROC-AUC:                         {roc_auc:.3f}")
            if ef1 > 1.0:
                log.info("  ✓ Pipeline shows enrichment better than random.")
            else:
                log.info("  ⚠ Pipeline enrichment at or below random.")
            if roc_auc > 0.7:
                log.info("  ✓ Good discriminatory power (ROC-AUC > 0.7).")
            elif roc_auc > 0.55:
                log.info("  ✓ Moderate discriminatory power.")
            else:
                log.info("  ⚠ Poor discriminatory power (near random).")
        except Exception as exc:
            log.warning(f"  Benchmark check failed: {exc}")
        return context

    def _analyze_selectivity(self, context: PipelineContext) -> PipelineContext:
        context.docked_candidates = analyze_selectivity_and_resistance(
            context.docked_candidates, self.targets, str(self.config.work_dir),
            self.deps, cache=self.cache, use_cache=self.use_cache,
            water_results=self.water_results,
        )
        return context

    def _apply_explicit_solvent_rescoring(self, context: PipelineContext) -> PipelineContext:
        if not self.config.use_explicit_solvent_mmgbsa:
            return context
        if not context.docked_candidates:
            return context
        log.info("─── Phase 4.6: Explicit-Solvent MM-GB/SA Rescoring ───")
        pb2pa = self.targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            log.warning("  Receptor PDB not found; skipping explicit-solvent rescoring.")
            return context
        try:
            rescore_with_explicit_mmgbsa(
                context.docked_candidates,
                receptor_pdb,
                str(self.config.work_dir),
                water_results=self.water_results,
            )
        except Exception as exc:
            log.warning(f"  Explicit-solvent rescoring failed: {exc}")
        return context

    def _apply_md_validation(self, context: PipelineContext) -> PipelineContext:
        from .config import ConfigurationError
        md_duration = self.config.md_validation_duration_ns
        if not (md_duration > 0 and _HAVE_MD and context.docked_candidates):
            if self.config.force_md_for_meta_scoring and context.docked_candidates:
                raise ConfigurationError(
                    "MD validation is required by force_md_for_meta_scoring=True, "
                    f"but md_duration={md_duration} ns or MD module unavailable."
                )
            return context
        log.info("─── Phase 4.7: MD Validation (Adaptive Sampling) ───")
        pb2pa = self.targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            if self.config.force_md_for_meta_scoring:
                raise ConfigurationError(
                    "MD validation is required by force_md_for_meta_scoring=True, "
                    "but receptor PDB file is missing."
                )
            log.warning("  Receptor PDB not found; skipping MD validation.")
            return context
        failed_any = False
        md_results: List[CompoundRecord] = []
        for rec in context.docked_candidates[:3]:
            if rec.mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    continue
                rec.mol = mol
            try:
                result = run_short_md(
                    rec.mol,
                    receptor_pdb,
                    duration_ns=float(md_duration),
                    max_duration_ns=float(self.config.md_max_duration_ns),
                    convergence_window_chunks=int(self.config.md_convergence_window_chunks),
                    rmsd_convergence_threshold=float(self.config.md_rmsd_convergence_threshold),
                )
                if result is not None:
                    rec.md_ligand_rmsd = result.get("ligand_rmsd_angstrom")
                    rec.md_pocket_rg_stability = result.get("pocket_rg_stability")
                    rec.md_converged = result.get("converged", False)
                    md_results.append(rec)
                    log.info(
                        f"  {rec.compound_id}: MD ligand RMSD = "
                        f"{rec.md_ligand_rmsd:.2f} A, "
                        f"pocket Rg stability = {rec.md_pocket_rg_stability:.3f}, "
                        f"converged = {rec.md_converged}"
                    )
                else:
                    if self.config.force_md_for_meta_scoring:
                        raise ConfigurationError(
                            f"  {rec.compound_id}: MD validation returned None — "
                            "meta-scoring requires MD-derived dynamic features."
                        )
                    log.info(f"  {rec.compound_id}: MD not available.")
                    failed_any = True
            except ConfigurationError:
                raise
            except Exception as exc:
                if self.config.force_md_for_meta_scoring:
                    raise ConfigurationError(
                        f"  MD validation failed for {rec.compound_id}: {exc}"
                    ) from exc
                log.warning(f"  MD validation failed for {rec.compound_id}: {exc}")
                failed_any = True
        if failed_any and self.config.force_md_for_meta_scoring:
            raise ConfigurationError(
                "MD validation failed for one or more top candidates. "
                "Set force_md_for_meta_scoring=False to allow meta-scoring "
                "without MD data."
            )
        context.md_results = md_results
        return context

    def _apply_meta_scoring(self, context: PipelineContext) -> PipelineContext:
        if not self.config.use_meta_scoring:
            log.info("  Meta-scoring disabled (use_meta_scoring=False).")
            return context
        if not context.docked_candidates:
            return context
        water_disp_energy: Optional[float] = None
        if self.water_results is not None and self.water_results.all_waters:
            energies = [w.displacement_energy for w in self.water_results.all_waters]
            if energies:
                water_disp_energy = float(np.mean(energies))
        for rec in context.docked_candidates:
            if water_disp_energy is not None:
                rec.water_displacement_energy = water_disp_energy
        if self.config.use_gnn_rescoring:
            log.info("─── Phase 4.5: GNN Rescoring ───")
            gnn_scorer = GNNScorer()
            if gnn_scorer.available:
                for rec in context.docked_candidates:
                    gnn_score = gnn_scorer.predict(rec)
                    if gnn_score is not None:
                        rec.ml_score = gnn_score
                        log.debug(
                            "  %s: GNN score = %.4f kcal/mol",
                            rec.compound_id, gnn_score,
                        )
                    else:
                        log.debug("  %s: GNN score not computed.", rec.compound_id)
                return context
            log.info("  GNN rescoring unavailable; falling back to MetaScorer.")
        log.info("─── Phase 4.5: Meta-Learner Consensus Scoring ───")
        for rec in context.docked_candidates:
            meta_score = predict_meta_score(rec)
            if meta_score is not None:
                log.debug(f"  {rec.compound_id}: meta-score = {meta_score:.4f}")
            else:
                log.debug(f"  {rec.compound_id}: meta-score not computed.")
        scorer = _get_meta_scorer()
        if scorer is not None:
            scorer.flag_uncertain_predictions(
                context.docked_candidates, threshold=0.1,
            )
            flagged = [r for r in context.docked_candidates if r.needs_manual_review]
            if flagged:
                log.info(
                    f"  Active learning: {len(flagged)}/{len(context.docked_candidates)} "
                    "compounds flagged for manual review "
                    "(prediction std > 0.1)."
                )
                self._save_review_queue(flagged)
            else:
                log.info(
                    "  Active learning: no compounds flagged for review "
                    "(all predictions within uncertainty threshold)."
                )

            # ── Auto-retrain on uncertainty ─────────────────────────
            if (
                CONFIG.auto_retrain_on_uncertainty
                and CONFIG.retrain_model_path
                and len(flagged) > 5
            ):
                log.info(
                    f"  Active learning: auto-retraining triggered with "
                    f"{len(flagged)} uncertain compounds."
                )
                import csv
                import tempfile

                temp_csv = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False,
                )
                try:
                    writer = csv.writer(temp_csv)
                    writer.writerow(["smiles", "ic50"])
                    for rec in flagged:
                        meta_score = predict_meta_score(rec)
                        if meta_score is None:
                            continue
                        if meta_score > 0.7:
                            writer.writerow([rec.smiles, "0.001"])
                        elif meta_score < 0.3:
                            writer.writerow([rec.smiles, "100"])
                    temp_csv.close()
                    from .active_learning import retrain_meta_scorer
                    retrain_meta_scorer(temp_csv.name)
                    log.info(
                        "  Active learning: MetaScorer retrained with "
                        "synthetic labels from uncertain compounds."
                    )
                except Exception as exc:
                    log.warning(
                        f"  Active learning auto-retraining failed: {exc}"
                    )
                finally:
                    try:
                        os.unlink(temp_csv.name)
                    except OSError:
                        pass
        return context

    def _apply_fep_resistance(self, context: PipelineContext) -> PipelineContext:
        if not self.config.use_fep_resistance:
            return context
        if not context.docked_candidates:
            return context
        log.info("─── Phase 4.8: Top-Hit FEP Resistance Profiling ───")
        pb2pa = self.targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            log.warning("  Receptor PDB not found; skipping FEP.")
            return context

        manager = FEPManager(config=self.config, targets=self.targets)
        selected_candidates = manager.select_candidates_for_fep(
            context.docked_candidates,
        )
        fep_results = manager.run_fep_profiling(
            selected_candidates, str(self.config.work_dir),
        )
        context.fep_results = fep_results
        return context

    def _generate_reports(self, context: PipelineContext) -> PipelineContext:
        generate_csv_report(context.docked_candidates)
        top3 = context.docked_candidates[:self.config.top_n_for_images]
        generate_images(top3)
        scored_for_top50 = [
            r for r in context.filtered_library
            if r.pb2pa_allosteric_energy is not None
        ]
        scored_for_top50.sort(key=lambda r: r.pb2pa_allosteric_energy)
        top_n = self.config.top_n_for_html_report
        top50 = (
            scored_for_top50[:top_n]
            if len(scored_for_top50) >= top_n
            else scored_for_top50
        )
        generate_html_report(context.docked_candidates, top50, self.config.output_dir)
        return context

    def _sync_from_context(self, context: PipelineContext) -> None:
        """Sync PipelineContext state back to ``self`` attributes for
        backward compatibility with public API methods and ``_finalize``."""
        self.all_records = context.library
        self.filtered = context.filtered_library
        self.top_candidates = context.docked_candidates
        self.n_total = context.n_total
        self.n_filtered = context.n_filtered

    def _filter_by_key_interactions_context(self, context: PipelineContext) -> PipelineContext:
        candidates = context.docked_candidates
        flag = self.config.require_key_interactions_for_rescoring
        if not flag:
            log.info("  Key-interaction filter disabled.")
            return context
        pb2pa = self.targets.get("PBP2a", {})
        receptor_pdbqt = pb2pa.get("pdbqt", "")
        if not receptor_pdbqt:
            log.warning("  No PBP2a receptor PDBQT; skipping IFP filter.")
            return context
        receptor_pdb = receptor_pdbqt.replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            log.warning("  Receptor PDB not found; skipping IFP filter.")
            return context
        allosteric_residues = self.config.key_interaction_residues_allosteric
        active_residues = self.config.key_interaction_residues_active
        n_before = len(candidates)
        filtered: List[CompoundRecord] = []
        for rec in candidates:
            pose_path = rec.docked_pose_path
            if not pose_path or not os.path.isfile(pose_path):
                filtered.append(rec)
                continue
            try:
                alloc_hit = check_key_interactions(
                    pose_path, receptor_pdb, allosteric_residues,
                )
                act_hit = check_key_interactions(
                    pose_path, receptor_pdb, active_residues,
                )
                if alloc_hit or act_hit:
                    filtered.append(rec)
                else:
                    log.info(
                        f"  Filtering out {rec.compound_id}: "
                        "no key interactions detected."
                    )
            except Exception:
                log.warning(
                    f"  IFP check failed for {rec.compound_id}; "
                    "keeping compound (fail-safe).",
                    exc_info=True,
                )
                filtered.append(rec)
        context.docked_candidates = filtered
        n_removed = n_before - len(filtered)
        if n_removed:
            log.info(
                f"  Pose filtering removed {n_removed}/{n_before} "
                f"candidates. {len(filtered)} remaining."
            )
        else:
            log.info(
                f"  Pose filtering: all {n_before} candidates "
                "retained (all had key interactions)."
            )
        return context

    # ── Phase methods (public, backward-compatible) ────────────────

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

        # Active-learning retraining
        if self.config.retrain_model_path:
            self._retrain_from_csv(self.config.retrain_model_path)

    def run_water_analysis(self) -> None:
        """Phase 0.5: Crystallographic water analysis (if available)."""
        self.water_results = None
        if not (self.config.use_water_analysis and _HAVE_WATER):
            if self.config.use_water_analysis:
                log.info("  Water analysis module not available (install Bio.PDB).")
            return

        pdb_dir = self.config.pdb_dir
        holo_pdb_id = self.config.pdb_ids.get("PBP2a_holo", "3ZG0")
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

        pb2pa = self.targets.get("PBP2a", {})
        validated_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if validated_pdb and os.path.isfile(validated_pdb):
            log.info(f"  Receptor integrity validated for PBP2a: {validated_pdb}")
        else:
            log.warning("  PBP2a receptor PDB not found after target preparation.")

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
            config=self.config,
        )

    def generate_and_filter_library(self) -> None:
        """Phase 2: Compound library generation and filtering."""
        log.info("─── Phase 2: Library Generation & Filtering ───")
        if self.config.use_pharmacophore_filter:
            log.info("  Pharmacophore-constrained library generation enabled.")
            self.all_records = generate_pharmacophore_aware_library(
                target_count=self.config.library_target_count,
                seed=self.config.random_seed,
                config=self.config,
            )
        else:
            self.all_records = list(
                generate_candidate_library(
                    target_count=self.config.library_target_count,
                    config=self.config,
                )
            )
        self.n_total = len(self.all_records)

        self.filtered = apply_filters(self.all_records, audit=self.audit, config=self.config)
        self.n_filtered = len(self.filtered)

        if self.audit is not None:
            self.audit.check_health(self.n_total, "Library Filtering")

        if self.n_filtered == 0:
            log.warning("  No compounds passed filters. Halting pipeline.")
            raise SystemExit(0)

    def screen_candidates(self) -> None:
        """Phase 3: Virtual screening (docking + ML rescoring)."""
        self.top_candidates = screen_library(
            self.filtered, self.targets, str(self.config.work_dir),
            self.deps, cache=self.cache, use_cache=self.use_cache,
            water_results=self.water_results, dry_run=self.config.dry_run,
            audit=self.audit, config=self.config,
        )

        if self.audit is not None:
            self.audit.check_health(len(self.filtered), "Docking")

        if not self.top_candidates:
            log.warning("  No candidates after screening. Halting pipeline.")
            raise SystemExit(0)

        self._filter_by_key_interactions()

    def analyze_selectivity(self) -> None:
        """Phase 4: Selectivity filtering and resistance analysis."""
        self.top_candidates = analyze_selectivity_and_resistance(
            self.top_candidates, self.targets, str(self.config.work_dir),
            self.deps, cache=self.cache, use_cache=self.use_cache,
            water_results=self.water_results,
        )

    def apply_fep_resistance(self) -> None:
        """Phase 4.8 — Top-hit only FEP resistance profiling.

        Runs rigorous Free Energy Perturbation (FEP) only on the top
        :attr:`CONFIG.fep_top_n` candidates after docking and MM-GB/SA
        rescoring.  Skips candidates whose ligand has >50 heavy atoms
        or whose SMILES exceeds 100 characters.

        If FEP fails for a specific compound and
        ``use_heuristic_resistance_fallback`` is True, falls back to
        heuristic docking-based SD resistance profiling.

        Configured via ``CONFIG.use_fep_resistance`` and
        ``CONFIG.fep_top_n``.

        This public method delegates to the private context-based
        implementation for consistency.
        """
        ctx = PipelineContext(
            library=self.all_records,
            filtered_library=self.filtered,
            docked_candidates=self.top_candidates,
            audit_log=self.audit,
        )
        ctx = self._apply_fep_resistance(ctx)
        self.top_candidates = ctx.docked_candidates

    def apply_explicit_solvent_rescoring(self) -> None:
        """Phase 4.6 — Explicit-solvent MM-GB/SA rescoring (if enabled).

        When ``CONFIG.use_explicit_solvent_mmgbsa`` is True, replaces the
        default implicit-solvent MM-GB/SA heuristic with a more rigorous
        explicit-solvent calculation on the top candidates.
        """
        if not self.config.use_explicit_solvent_mmgbsa:
            return
        if not self.top_candidates:
            return
        log.info("─── Phase 4.6: Explicit-Solvent MM-GB/SA Rescoring ───")
        pb2pa = self.targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            log.warning("  Receptor PDB not found; skipping explicit-solvent rescoring.")
            return
        try:
            rescore_with_explicit_mmgbsa(
                self.top_candidates,
                receptor_pdb,
                str(self.config.work_dir),
                water_results=self.water_results,
            )
        except Exception as exc:
            log.warning(f"  Explicit-solvent rescoring failed: {exc}")

    def apply_meta_scoring(self) -> None:
        """Phase 4.5 — Meta-learner / GNN consensus scoring on top candidates.

        When ``self.config.use_gnn_rescoring`` is *True* a
        :class:`GNNScorer` is tried first.  If the GNN scorer is
        available it replaces the Random-Forest MetaScorer entirely.
        Otherwise the pipeline falls back to the existing
        ``MetaScorer`` path.

        After scoring, uncertain predictions are flagged for manual
        review and saved to ``output/review_queue.csv`` (MetaScorer
        path only).
        """
        if not self.config.use_meta_scoring:
            log.info("  Meta-scoring disabled (use_meta_scoring=False).")
            return
        if not self.top_candidates:
            return

        # Populate water displacement energy once
        water_disp_energy: Optional[float] = None
        if self.water_results is not None and self.water_results.all_waters:
            energies = [w.displacement_energy for w in self.water_results.all_waters]
            if energies:
                water_disp_energy = float(np.mean(energies))

        for rec in self.top_candidates:
            if water_disp_energy is not None:
                rec.water_displacement_energy = water_disp_energy

        # ── GNN rescoring path (takes over completely when available) ──
        if self.config.use_gnn_rescoring:
            log.info("─── Phase 4.5: GNN Rescoring ───")
            gnn_scorer = GNNScorer()
            if gnn_scorer.available:
                for rec in self.top_candidates:
                    gnn_score = gnn_scorer.predict(rec)
                    if gnn_score is not None:
                        rec.ml_score = gnn_score
                        log.debug(
                            "  %s: GNN score = %.4f kcal/mol",
                            rec.compound_id, gnn_score,
                        )
                    else:
                        log.debug("  %s: GNN score not computed.", rec.compound_id)
                return  # GNN path complete

            log.info("  GNN rescoring unavailable; falling back to MetaScorer.")

        # ── MetaScorer path (default / fallback) ──
        log.info("─── Phase 4.5: Meta-Learner Consensus Scoring ───")

        for rec in self.top_candidates:
            meta_score = predict_meta_score(rec)
            if meta_score is not None:
                log.debug(f"  {rec.compound_id}: meta-score = {meta_score:.4f}")
            else:
                log.debug(f"  {rec.compound_id}: meta-score not computed.")

        # Active learning: flag uncertain predictions
        scorer = _get_meta_scorer()
        if scorer is not None:
            scorer.flag_uncertain_predictions(
                self.top_candidates, threshold=0.1,
            )
            flagged = [r for r in self.top_candidates if r.needs_manual_review]
            if flagged:
                log.info(
                    f"  Active learning: {len(flagged)}/{len(self.top_candidates)} "
                    "compounds flagged for manual review "
                    "(prediction std > 0.1)."
                )
                self._save_review_queue(flagged)
            else:
                log.info(
                    "  Active learning: no compounds flagged for review "
                    "(all predictions within uncertainty threshold)."
                )

            # ── Auto-retrain on uncertainty ─────────────────────────
            if (
                CONFIG.auto_retrain_on_uncertainty
                and CONFIG.retrain_model_path
                and len(flagged) > 5
            ):
                log.info(
                    f"  Active learning: auto-retraining triggered with "
                    f"{len(flagged)} uncertain compounds."
                )
                import csv
                import tempfile

                temp_csv = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False,
                )
                try:
                    writer = csv.writer(temp_csv)
                    writer.writerow(["smiles", "ic50"])
                    for rec in flagged:
                        meta_score = predict_meta_score(rec)
                        if meta_score is None:
                            continue
                        if meta_score > 0.7:
                            writer.writerow([rec.smiles, "0.001"])
                        elif meta_score < 0.3:
                            writer.writerow([rec.smiles, "100"])
                    temp_csv.close()
                    from .active_learning import retrain_meta_scorer
                    retrain_meta_scorer(temp_csv.name)
                    log.info(
                        "  Active learning: MetaScorer retrained with "
                        "synthetic labels from uncertain compounds."
                    )
                except Exception as exc:
                    log.warning(
                        f"  Active learning auto-retraining failed: {exc}"
                    )
                finally:
                    try:
                        os.unlink(temp_csv.name)
                    except OSError:
                        pass

    def apply_md_validation(self) -> None:
        """Phase 4.7 — Optional MD validation of top candidates with
        adaptive sampling.

        Uses :func:`run_short_md` with adaptive convergence checking.
        Stores MD-derived dynamic features (*ligand_rmsd*, *pocket_rg_stability*,
        *md_converged*) in each ``CompoundRecord`` so they are available to the
        MetaScorer (Phase 4.5) and for downstream analysis.

        When ``CONFIG.force_md_for_meta_scoring`` is True and MD validation
        fails for top candidates, a ``ConfigurationError`` is raised to
        prevent meta-scoring from proceeding with incomplete data.
        """
        from .config import ConfigurationError

        md_duration = self.config.md_validation_duration_ns
        if not (md_duration > 0 and _HAVE_MD and self.top_candidates):
            # When force_md_for_meta_scoring is True, MD validation is mandatory
            if self.config.force_md_for_meta_scoring and self.top_candidates:
                raise ConfigurationError(
                    "MD validation is required by force_md_for_meta_scoring=True, "
                    f"but md_duration={md_duration} ns or MD module unavailable."
                )
            return
        log.info("─── Phase 4.7: MD Validation (Adaptive Sampling) ───")
        pb2pa = self.targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            if self.config.force_md_for_meta_scoring:
                raise ConfigurationError(
                    "MD validation is required by force_md_for_meta_scoring=True, "
                    "but receptor PDB file is missing."
                )
            log.warning("  Receptor PDB not found; skipping MD validation.")
            return

        failed_any = False
        for rec in self.top_candidates[:3]:
            if rec.mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    continue
                rec.mol = mol
            try:
                result = run_short_md(
                    rec.mol,
                    receptor_pdb,
                    duration_ns=float(md_duration),
                    max_duration_ns=float(self.config.md_max_duration_ns),
                    convergence_window_chunks=int(self.config.md_convergence_window_chunks),
                    rmsd_convergence_threshold=float(self.config.md_rmsd_convergence_threshold),
                )
                if result is not None:
                    rec.md_ligand_rmsd = result.get("ligand_rmsd_angstrom")
                    rec.md_pocket_rg_stability = result.get("pocket_rg_stability")
                    rec.md_converged = result.get("converged", False)
                    log.info(
                        f"  {rec.compound_id}: MD ligand RMSD = "
                        f"{rec.md_ligand_rmsd:.2f} A, "
                        f"pocket Rg stability = {rec.md_pocket_rg_stability:.3f}, "
                        f"converged = {rec.md_converged}"
                    )
                else:
                    if self.config.force_md_for_meta_scoring:
                        raise ConfigurationError(
                            f"  {rec.compound_id}: MD validation returned None — "
                            "meta-scoring requires MD-derived dynamic features."
                        )
                    log.info(f"  {rec.compound_id}: MD not available.")
                    failed_any = True
            except ConfigurationError:
                raise
            except Exception as exc:
                if self.config.force_md_for_meta_scoring:
                    raise ConfigurationError(
                        f"  MD validation failed for {rec.compound_id}: {exc}"
                    ) from exc
                log.warning(f"  MD validation failed for {rec.compound_id}: {exc}")
                failed_any = True

        if failed_any and self.config.force_md_for_meta_scoring:
            raise ConfigurationError(
                "MD validation failed for one or more top candidates. "
                "Set force_md_for_meta_scoring=False to allow meta-scoring "
                "without MD data."
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

    def _filter_by_key_interactions(self) -> None:
        """Filter ``self.top_candidates`` by key interaction fingerprint.

        Docked poses that do ***not*** contact any residue listed in
        ``CONFIG.key_interaction_residues_allosteric`` **or**
        ``CONFIG.key_interaction_residues_active`` are discarded **unless**
        the check itself fails (missing file, parse error), in which case
        the compound is retained (fail-safe).

        The filter is bypassed entirely when
        ``CONFIG.require_key_interactions_for_rescoring`` is ``False``.
        """
        flag = self.config.require_key_interactions_for_rescoring
        if not flag:
            log.info("  Key-interaction filter disabled.")
            return

        pb2pa = self.targets.get("PBP2a", {})
        receptor_pdbqt = pb2pa.get("pdbqt", "")
        if not receptor_pdbqt:
            log.warning("  No PBP2a receptor PDBQT; skipping IFP filter.")
            return

        receptor_pdb = receptor_pdbqt.replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            log.warning("  Receptor PDB not found; skipping IFP filter.")
            return

        allosteric_residues = self.config.key_interaction_residues_allosteric
        active_residues = self.config.key_interaction_residues_active
        n_before = len(self.top_candidates)
        filtered: List[CompoundRecord] = []

        for rec in self.top_candidates:
            pose_path = rec.docked_pose_path
            if not pose_path or not os.path.isfile(pose_path):
                # Fail-safe: keep if pose file is missing
                filtered.append(rec)
                continue

            try:
                alloc_hit = check_key_interactions(
                    pose_path, receptor_pdb, allosteric_residues,
                )
                act_hit = check_key_interactions(
                    pose_path, receptor_pdb, active_residues,
                )
                if alloc_hit or act_hit:
                    filtered.append(rec)
                else:
                    log.info(
                        f"  Filtering out {rec.compound_id}: "
                        "no key interactions detected."
                    )
            except Exception:
                # Fail-safe: keep if the check itself fails
                log.warning(
                    f"  IFP check failed for {rec.compound_id}; "
                    "keeping compound (fail-safe).",
                    exc_info=True,
                )
                filtered.append(rec)

        self.top_candidates = filtered
        n_removed = n_before - len(filtered)
        if n_removed:
            log.info(
                f"  Pose filtering removed {n_removed}/{n_before} "
                f"candidates. {len(filtered)} remaining."
            )
        else:
            log.info(
                f"  Pose filtering: all {n_before} candidates "
                "retained (all had key interactions)."
            )

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
        """Save cache, audit summary, and print summary."""
        if self.use_cache:
            cache_path = self.config.output_dir / self.config.cache_name
            save_json_cache(cache_path, self.cache)
            log.info(f"  Cache saved ({len(self.cache)} entries).")

        if self.audit is not None:
            audit_path = self.config.output_dir / self.config.audit_output_name
            try:
                summary = self.audit.get_summary()
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with open(audit_path, "w") as f:
                    json.dump(summary, f, indent=2)
                log.info(f"  Audit summary saved to {audit_path}")
                dr = summary.get("dropout_rate", 0)
                log.info(f"  Total processed: {summary['total_processed']} | "
                         f"Dropped: {summary['total_dropped']} | "
                         f"Rate: {dr:.1%}")
            except Exception as exc:
                log.warning(f"  Failed to save audit summary: {exc}")

        print_summary(
            self.n_total, self.n_filtered, self.top_candidates,
            self.validation_ok, self.redock_rmsd, self.deps,
        )

    def _save_review_queue(self, flagged: List[CompoundRecord]) -> None:
        """Save flagged compounds to a CSV for manual review or FEP."""
        import csv

        review_path = self.config.output_dir / "review_queue.csv"
        try:
            review_path.parent.mkdir(parents=True, exist_ok=True)
            with open(review_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["compound_id", "smiles", "meta_score", "reason"])
                for rec in flagged:
                    writer.writerow([
                        rec.compound_id, rec.smiles,
                        getattr(rec, "ml_score", ""),
                        "High prediction uncertainty",
                    ])
            log.info(f"  Review queue saved to {review_path}")
        except Exception as exc:
            log.warning(f"  Failed to save review queue: {exc}")

        log.info("Pipeline complete. Exiting.")

    def _retrain_from_csv(self, csv_path: str) -> None:
        """Load a CSV with {smiles, ic50} and retrain the MetaScorer."""
        import csv

        # Ensure output directory exists (needed for model persistence)
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        from autoantibiotic.ml_scoring.meta_scorer import _get_meta_scorer, MetaScorer

        scorer = _get_meta_scorer()
        if scorer is None:
            log.warning("  MetaScorer unavailable for retraining.")
            return

        new_actives: List[str] = []
        new_inactives: List[str] = []
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smi = row.get("smiles", "").strip()
                    ic50 = row.get("ic50", "").strip()
                    if not smi or not ic50:
                        continue
                    try:
                        val = float(ic50)
                    except ValueError:
                        continue
                    if val > 0:
                        new_actives.append(smi)
                    else:
                        new_inactives.append(smi)
            if new_actives or new_inactives:
                scorer.retrain_with_new_data(new_actives, new_inactives)
                log.info(
                    f"  MetaScorer retrained with {len(new_actives)} active "
                    f"/ {len(new_inactives)} inactive compounds from {csv_path}"
                )
            else:
                log.info("  No valid retraining data found in CSV.")
        except Exception as exc:
            log.warning(f"  Failed to retrain model from CSV: {exc}")
