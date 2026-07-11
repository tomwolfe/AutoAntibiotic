import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import PipelineConfig
from ..docking import get_engine, screen_library
from ..io_utils import log, PipelineAudit
from ..models import CompoundRecord
from ..scoring_metrics import check_key_interactions
from .base import PhaseHandler


class DockingHandler(PhaseHandler):
    def execute(self, state: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
        targets: Dict[str, Any] = state["targets"]
        deps: Dict[str, Any] = state["deps"]
        cache: Dict[str, float] = state.get("cache", {})
        use_cache: bool = state.get("use_cache", False)
        water_results: Any = state.get("water_results")
        audit: Optional[PipelineAudit] = state.get("audit")
        filtered_library: List[CompoundRecord] = state.get("filtered_library", [])

        engine_name = "gnina" if config.use_gnina else "vina"
        engine = get_engine(engine_name, config=config)

        top_candidates = screen_library(
            filtered_library, targets, str(config.work_dir),
            deps, cache=cache, use_cache=use_cache,
            water_results=water_results, dry_run=config.dry_run,
            audit=audit, config=config, engine=engine,
        )

        state["docked_candidates"] = top_candidates

        if audit is not None:
            audit.check_health(len(filtered_library), "Docking")

        if not top_candidates:
            log.warning("  No candidates after screening. Halting pipeline.")
            raise SystemExit(0)

        state = self._filter_by_key_interactions(state, config)
        state = self._run_benchmark_check(state, config)
        return state

    def _filter_by_key_interactions(
        self, state: Dict[str, Any], config: PipelineConfig,
    ) -> Dict[str, Any]:
        candidates: List[CompoundRecord] = state.get("docked_candidates", [])
        flag = config.require_key_interactions_for_rescoring
        if not flag:
            log.info("  Key-interaction filter disabled.")
            return state

        targets: Dict[str, Any] = state.get("targets", {})
        pb2pa = targets.get("PBP2a", {})
        receptor_pdbqt = pb2pa.get("pdbqt", "")
        if not receptor_pdbqt:
            log.warning("  No PBP2a receptor PDBQT; skipping IFP filter.")
            return state

        receptor_pdb = receptor_pdbqt.replace(".pdbqt", ".pdb")
        if not os.path.isfile(receptor_pdb):
            log.warning("  Receptor PDB not found; skipping IFP filter.")
            return state

        allosteric_residues = config.key_interaction_residues_allosteric
        active_residues = config.key_interaction_residues_active
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

        state["docked_candidates"] = filtered
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
        return state

    def _run_benchmark_check(
        self, state: Dict[str, Any], config: PipelineConfig,
    ) -> Dict[str, Any]:
        if not config.benchmark_mode:
            return state
        candidates: List[CompoundRecord] = state.get("docked_candidates", [])
        if not candidates:
            return state
        log.info("─" * 3 + " Benchmark Check " + "─" * 3)
        try:
            from benchmarks.reference_data import get_actives_smiles, get_inactives_smiles
            from benchmarks.run_enrichment_test import compute_enrichment_factor, compute_roc_auc

            active_smiles = set(get_actives_smiles())
            inactive_smiles = set(get_inactives_smiles())

            scores: List[float] = []
            labels: List[int] = []
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
                return state

            scores_arr = np.array(scores, dtype=np.float64)
            labels_arr = np.array(labels, dtype=np.int64)
            ef1 = compute_enrichment_factor(scores_arr, labels_arr, fraction=0.01)
            roc_auc = compute_roc_auc(scores_arr, labels_arr)
            log.info(f"  EF1% (Enrichment Factor at 1%): {ef1:.3f}")
            log.info(f"  ROC-AUC:                         {roc_auc:.3f}")
            if ef1 > 1.0:
                log.info("  \u2713 Pipeline shows enrichment better than random.")
            else:
                log.info("  \u26a0 Pipeline enrichment at or below random.")
            if roc_auc > 0.7:
                log.info("  \u2713 Good discriminatory power (ROC-AUC > 0.7).")
            elif roc_auc > 0.55:
                log.info("  \u2713 Moderate discriminatory power.")
            else:
                log.info("  \u26a0 Poor discriminatory power (near random).")
        except Exception as exc:
            log.warning(f"  Benchmark check failed: {exc}")
        return state
