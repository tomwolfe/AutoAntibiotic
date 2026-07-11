import csv
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem

from ..config import ConfigurationError, PipelineConfig
from ..analysis import analyze_selectivity_and_resistance, predict_meta_score
from ..fep_manager import FEPManager
from ..io_utils import log
from ..ml_scoring.gnn_scorer import GNNScorer
from ..ml_scoring.meta_scorer import _get_meta_scorer
from ..ml_scoring.scoring import rescore_with_explicit_mmgbsa
from ..models import CompoundRecord
from .base import PhaseHandler

try:
    from ..md_validation import run_short_md
    _HAVE_MD = True
except ImportError:
    _HAVE_MD = False


class AnalysisHandler(PhaseHandler):
    def execute(self, state: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
        state = self._analyze_selectivity(state, config)
        state = self._apply_explicit_solvent_rescoring(state, config)
        state = self._apply_md_validation(state, config)
        state = self._apply_meta_scoring(state, config)
        state = self._apply_fep_resistance(state, config)
        return state

    def _analyze_selectivity(
        self, state: Dict[str, Any], config: PipelineConfig,
    ) -> Dict[str, Any]:
        targets: Dict[str, Any] = state["targets"]
        deps: Dict[str, Any] = state["deps"]
        cache: Dict[str, float] = state.get("cache", {})
        use_cache: bool = state.get("use_cache", False)
        water_results: Any = state.get("water_results")
        candidates: List[CompoundRecord] = state.get("docked_candidates", [])

        analyzed = analyze_selectivity_and_resistance(
            candidates, targets, str(config.work_dir),
            deps, cache=cache, use_cache=use_cache,
            water_results=water_results,
        )
        state["docked_candidates"] = analyzed
        return state

    def _apply_explicit_solvent_rescoring(
        self, state: Dict[str, Any], config: PipelineConfig,
    ) -> Dict[str, Any]:
        if not config.use_explicit_solvent_mmgbsa:
            return state
        candidates: List[CompoundRecord] = state.get("docked_candidates", [])
        if not candidates:
            return state
        log.info("─" * 3 + " Phase 4.6: Explicit-Solvent MM-GB/SA Rescoring " + "─" * 3)
        targets: Dict[str, Any] = state.get("targets", {})
        water_results: Any = state.get("water_results")
        pb2pa = targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            log.warning("  Receptor PDB not found; skipping explicit-solvent rescoring.")
            return state
        try:
            rescore_with_explicit_mmgbsa(
                candidates,
                receptor_pdb,
                str(config.work_dir),
                water_results=water_results,
            )
        except Exception as exc:
            log.warning(f"  Explicit-solvent rescoring failed: {exc}")
        return state

    def _apply_md_validation(
        self, state: Dict[str, Any], config: PipelineConfig,
    ) -> Dict[str, Any]:
        md_duration = config.md_validation_duration_ns
        candidates: List[CompoundRecord] = state.get("docked_candidates", [])
        if not (md_duration > 0 and _HAVE_MD and candidates):
            if config.force_md_for_meta_scoring and candidates:
                raise ConfigurationError(
                    "MD validation is required by force_md_for_meta_scoring=True, "
                    f"but md_duration={md_duration} ns or MD module unavailable."
                )
            return state
        log.info("─" * 3 + " Phase 4.7: MD Validation (Adaptive Sampling) " + "─" * 3)
        targets: Dict[str, Any] = state.get("targets", {})
        pb2pa = targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            if config.force_md_for_meta_scoring:
                raise ConfigurationError(
                    "MD validation is required by force_md_for_meta_scoring=True, "
                    "but receptor PDB file is missing."
                )
            log.warning("  Receptor PDB not found; skipping MD validation.")
            return state

        failed_any = False
        md_results: List[CompoundRecord] = []
        for rec in candidates[:3]:
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
                    max_duration_ns=float(config.md_max_duration_ns),
                    convergence_window_chunks=int(config.md_convergence_window_chunks),
                    rmsd_convergence_threshold=float(config.md_rmsd_convergence_threshold),
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
                    if config.force_md_for_meta_scoring:
                        raise ConfigurationError(
                            f"  {rec.compound_id}: MD validation returned None — "
                            "meta-scoring requires MD-derived dynamic features."
                        )
                    log.info(f"  {rec.compound_id}: MD not available.")
                    failed_any = True
            except ConfigurationError:
                raise
            except Exception as exc:
                if config.force_md_for_meta_scoring:
                    raise ConfigurationError(
                        f"  MD validation failed for {rec.compound_id}: {exc}"
                    ) from exc
                log.warning(f"  MD validation failed for {rec.compound_id}: {exc}")
                failed_any = True

        if failed_any and config.force_md_for_meta_scoring:
            raise ConfigurationError(
                "MD validation failed for one or more top candidates. "
                "Set force_md_for_meta_scoring=False to allow meta-scoring "
                "without MD data."
            )
        state["md_results"] = md_results
        return state

    def _apply_meta_scoring(
        self, state: Dict[str, Any], config: PipelineConfig,
    ) -> Dict[str, Any]:
        if not config.use_meta_scoring:
            log.info("  Meta-scoring disabled (use_meta_scoring=False).")
            return state
        candidates: List[CompoundRecord] = state.get("docked_candidates", [])
        if not candidates:
            return state
        water_results: Any = state.get("water_results")

        water_disp_energy: Optional[float] = None
        if water_results is not None and water_results.all_waters:
            energies = [w.displacement_energy for w in water_results.all_waters]
            if energies:
                water_disp_energy = float(np.mean(energies))

        for rec in candidates:
            if water_disp_energy is not None:
                rec.water_displacement_energy = water_disp_energy

        if config.use_gnn_rescoring:
            log.info("─" * 3 + " Phase 4.5: GNN Rescoring " + "─" * 3)
            gnn_scorer = GNNScorer()
            if gnn_scorer.available:
                for rec in candidates:
                    gnn_score = gnn_scorer.predict(rec)
                    if gnn_score is not None:
                        rec.ml_score = gnn_score
                        log.debug(
                            "  %s: GNN score = %.4f kcal/mol",
                            rec.compound_id, gnn_score,
                        )
                    else:
                        log.debug("  %s: GNN score not computed.", rec.compound_id)
                return state
            log.info("  GNN rescoring unavailable; falling back to MetaScorer.")

        log.info("─" * 3 + " Phase 4.5: Meta-Learner Consensus Scoring " + "─" * 3)

        for rec in candidates:
            meta_score = predict_meta_score(rec)
            if meta_score is not None:
                log.debug(f"  {rec.compound_id}: meta-score = {meta_score:.4f}")
            else:
                log.debug(f"  {rec.compound_id}: meta-score not computed.")

        scorer = _get_meta_scorer()
        if scorer is not None:
            scorer.flag_uncertain_predictions(
                candidates, threshold=0.1,
            )
            flagged = [r for r in candidates if r.needs_manual_review]
            if flagged:
                log.info(
                    f"  Active learning: {len(flagged)}/{len(candidates)} "
                    "compounds flagged for manual review "
                    "(prediction std > 0.1)."
                )
                self._save_review_queue(flagged, config)
            else:
                log.info(
                    "  Active learning: no compounds flagged for review "
                    "(all predictions within uncertainty threshold)."
                )

            from ..config import CONFIG
            if (
                CONFIG.auto_retrain_on_uncertainty
                and CONFIG.retrain_model_path
                and len(flagged) > 5
            ):
                log.info(
                    f"  Active learning: auto-retraining triggered with "
                    f"{len(flagged)} uncertain compounds."
                )
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
                    from ..active_learning import retrain_meta_scorer
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

        return state

    def _apply_fep_resistance(
        self, state: Dict[str, Any], config: PipelineConfig,
    ) -> Dict[str, Any]:
        if not config.use_fep_resistance:
            return state
        candidates: List[CompoundRecord] = state.get("docked_candidates", [])
        if not candidates:
            return state
        log.info("─" * 3 + " Phase 4.8: Top-Hit FEP Resistance Profiling " + "─" * 3)
        targets: Dict[str, Any] = state.get("targets", {})
        pb2pa = targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("pdbqt", "").replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            log.warning("  Receptor PDB not found; skipping FEP.")
            return state

        manager = FEPManager(config=config, targets=targets)
        selected_candidates = manager.select_candidates_for_fep(candidates)
        fep_results = manager.run_fep_profiling(
            selected_candidates, str(config.work_dir),
        )
        state["fep_results"] = fep_results
        return state

    def _save_review_queue(
        self, flagged: List[CompoundRecord], config: PipelineConfig,
    ) -> None:
        review_path = config.output_dir / "review_queue.csv"
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
