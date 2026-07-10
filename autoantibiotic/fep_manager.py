"""
FEP Manager
============
Encapsulates all FEP-related logic: candidate pre-screening, execution,
error handling, and heuristic fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdkit import Chem

from .config import PipelineConfig
from .models import CompoundRecord
from .io_utils import log
from .library_gen import check_pharmacophore_match, _build_allosteric_pharmacophore
from .scoring_metrics import compute_ifp_similarity


class FEPManager:
    """Manages FEP resistance profiling candidate selection and execution.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.
    targets : Dict[str, Any]
        Prepared target structures (e.g. ``PBP2a`` receptor paths, centers).
    """

    def __init__(self, config: PipelineConfig, targets: Dict[str, Any]) -> None:
        self.config = config
        self.targets = targets
        self._pharmacophore_query: Optional[Dict[str, Any]] = None
        self._ref_mol: Optional[Chem.Mol] = None
        self._receptor_pdb: Optional[str] = None

    # ── Lazy-loaded helpers ─────────────────────────────────────────

    @property
    def pharmacophore_query(self) -> Optional[Dict[str, Any]]:
        if self._pharmacophore_query is None:
            self._pharmacophore_query = _build_allosteric_pharmacophore(
                config=self.config,
            )
        return self._pharmacophore_query

    @property
    def ref_mol(self) -> Optional[Chem.Mol]:
        if self._ref_mol is None:
            ref_smiles = self.config.reference_antibiotics.get("Ceftaroline", "")
            if ref_smiles:
                self._ref_mol = Chem.MolFromSmiles(ref_smiles)
        return self._ref_mol

    @property
    def receptor_pdb(self) -> str:
        if self._receptor_pdb is None:
            pb2pa = self.targets.get("PBP2a", {})
            pdbqt = pb2pa.get("pdbqt", "")
            self._receptor_pdb = pdbqt.replace(".pdbqt", ".pdb")
        return self._receptor_pdb

    # ── Public API ──────────────────────────────────────────────────

    def select_candidates_for_fep(
        self, candidates: List[CompoundRecord],
    ) -> List[CompoundRecord]:
        """Pre-screen and select candidates for rigorous FEP profiling.

        Applies the full pre-screening pipeline:
          1. Limits to ``fep_pre_screen_pool_size`` candidates.
          2. Pharmacophore filter (min_matches=2).
          3. IFP similarity > ``fep_ifp_threshold``.
          4. Strict IFP >= 0.7 and allosteric energy < -8.0 kcal/mol.
          5. Sort by binding energy and return top ``fep_top_n_strict``.

        Parameters
        ----------
        candidates : List[CompoundRecord]
            Full list of docked candidates.

        Returns
        -------
        List[CompoundRecord]
            Filtered and sorted candidates ready for FEP execution.
        """
        pool_size = self.config.fep_pre_screen_pool_size
        candidates_pool = candidates[:pool_size]
        pharmacophore_query = self.pharmacophore_query
        ref_mol = self.ref_mol
        receptor_pdb = self.receptor_pdb

        fep_candidates: List[CompoundRecord] = []
        n_pharmacophore_filtered = 0
        n_ifp_filtered = 0

        for rec in candidates_pool:
            pharmacophore_ok = False
            if rec.mol is not None and pharmacophore_query is not None:
                if check_pharmacophore_match(
                    rec.mol,
                    query=pharmacophore_query,
                    min_matches=2,
                    tolerance=self.config.pharmacophore_tolerance,
                    config=self.config,
                ):
                    pharmacophore_ok = True
            if not pharmacophore_ok:
                n_pharmacophore_filtered += 1
                log.info(
                    f"  {rec.compound_id}: Failed pharmacophore match "
                    f"(min_matches=2) — Skipped FEP pre-screen."
                )
                continue

            ifp_ok = False
            pose_path = rec.docked_pose_path
            if (pose_path and os.path.isfile(pose_path)
                    and ref_mol is not None and rec.mol is not None):
                try:
                    ifp_score = compute_ifp_similarity(rec.mol, ref_mol, receptor_pdb)
                    if ifp_score > self.config.fep_ifp_threshold:
                        ifp_ok = True
                    else:
                        log.warning(
                            f"  {rec.compound_id}: IFP similarity {ifp_score:.3f} "
                            f"<= {self.config.fep_ifp_threshold} — Skipped FEP pre-screen."
                        )
                except Exception:
                    log.warning(
                        f"  {rec.compound_id}: IFP calculation failed — "
                        "Skipped FEP pre-screen."
                    )
            else:
                log.warning(
                    f"  {rec.compound_id}: No docked pose or reference ligand "
                    "available — Skipped FEP pre-screen."
                )
            if not ifp_ok:
                n_ifp_filtered += 1
                continue

            fep_candidates.append(rec)

        total_pre = len(candidates_pool)
        n_passed = len(fep_candidates)
        log.info(
            f"  FEP pre-screening: {n_passed}/{total_pre} candidates passed "
            f"({n_pharmacophore_filtered} failed pharmacophore, "
            f"{n_ifp_filtered} failed IFP similarity)."
        )

        # Strict pre-screening
        strict_threshold = 0.7
        energy_cutoff = -8.0
        strict_candidates: List[CompoundRecord] = []
        for rec in fep_candidates:
            if (rec.pb2pa_allosteric_energy is None
                    or rec.pb2pa_allosteric_energy >= energy_cutoff):
                continue
            if (rec.docked_pose_path and os.path.isfile(rec.docked_pose_path)
                    and ref_mol is not None and rec.mol is not None):
                try:
                    ifp_score = compute_ifp_similarity(rec.mol, ref_mol, receptor_pdb)
                    if ifp_score >= strict_threshold:
                        strict_candidates.append(rec)
                except Exception:
                    continue
            else:
                continue

        strict_candidates.sort(
            key=lambda r: (
                r.pb2pa_allosteric_energy
                if r.pb2pa_allosteric_energy is not None
                else 0.0
            ),
        )
        strict_candidates = strict_candidates[:self.config.fep_top_n_strict]

        n_skipped = len(fep_candidates) - len(strict_candidates)
        if n_skipped > 0:
            log.info(
                f"  Strict pre-screening skipped {n_skipped}/{len(fep_candidates)} "
                f"candidates (IFP < {strict_threshold} or allosteric energy "
                f">= {energy_cutoff} kcal/mol)."
            )

        log.info(
            f"  Pre-screening selected {len(strict_candidates)}/"
            f"{len(candidates_pool)} candidates for FEP (pharmacophore "
            f"min_matches=2, IFP > {self.config.fep_ifp_threshold}, "
            f"strict IFP >= {strict_threshold})."
        )
        return strict_candidates

    def run_fep_profiling(
        self,
        candidates: List[CompoundRecord],
        work_dir: str,
    ) -> List[CompoundRecord]:
        """Run rigorous FEP on the given candidates with error handling.

        For each candidate:
          - Parses SMILES if needed.
          - Checks heavy-atom count (<= 50).
          - Validates pharmacophore match.
          - Runs ``FEPResistanceCalculator.calculate_ddg()``.
          - Handles ``FEPConvergenceError`` (retry with increased windows),
            ``FETopologyError``, ``FEResourceError``, and generic errors.
          - Falls back to heuristic resistance profiling when
            ``config.use_heuristic_resistance_fallback`` is ``True``.

        Parameters
        ----------
        candidates : List[CompoundRecord]
            Pre-screened candidates for FEP profiling.
        work_dir : str
            Working directory for auxiliary files.

        Returns
        -------
        List[CompoundRecord]
            Candidates with ``resistance_stability_score`` populated.
        """
        from .fep_engine import (
            FEPResistanceCalculator,
            ConfigurationError as FEPConfigError,
            FEPConvergenceError,
            FETopologyError,
            FEResourceError,
        )
        from .analysis import profile_resistance_mutation_sensitivity

        pharmacophore_query = self.pharmacophore_query
        ref_mol = self.ref_mol
        receptor_pdb = self.receptor_pdb

        pb2pa = self.targets.get("PBP2a", {})
        mutant_pdbqts: List[str] = []
        mutant_dir = self.config.output_dir / "mutants"
        if mutant_dir.exists():
            mutant_pdbqts = sorted(str(p) for p in mutant_dir.glob("*.pdbqt"))
        center = pb2pa.get("allosteric_center", (0.0, 0.0, 0.0))
        box_size = self.config.allosteric_box_size

        fep_results: List[CompoundRecord] = []
        for rec in candidates:
            if rec.mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    log.warning(
                        f"  {rec.compound_id}: Cannot parse SMILES; skipping FEP."
                    )
                    continue
                rec.mol = mol

            if rec.mol.GetNumHeavyAtoms() > 50:
                log.info(
                    f"  {rec.compound_id}: {rec.mol.GetNumHeavyAtoms()} heavy atoms "
                    "(>50) — skipping FEP."
                )
                continue

            if pharmacophore_query is not None and not self._check_pharmacophore_match(
                rec.mol,
            ):
                log.info(
                    f"  {rec.compound_id}: Failed pharmacophore match — "
                    "Skipped FEP."
                )
                continue

            try:
                calc = FEPResistanceCalculator(
                    receptor_wt_pdb=receptor_pdb,
                    receptor_mut_pdb=receptor_pdb,
                    ligand_rdkit=rec.mol,
                )
                result = calc.calculate_ddg()
                rec.resistance_stability_score = result.delta_delta_g
                fep_results.append(rec)
                log.info(
                    f"  {rec.compound_id}: FEP ΔΔG = {result.delta_delta_g:.3f} "
                    f"kcal/mol (confidence={result.confidence:.2f})"
                )
            except FETopologyError as exc:
                log.warning(
                    f"  {rec.compound_id}: FEP topology error ({exc}) — "
                    "skipping compound (invalid input, no heuristic fallback)."
                )
            except FEPConvergenceError as exc:
                self._handle_convergence_error(
                    rec, exc, receptor_pdb, fep_results,
                    work_dir, mutant_pdbqts, center, box_size,
                )
            except FEResourceError as exc:
                log.warning(
                    f"  {rec.compound_id}: FEP resource error ({exc}) — "
                    "skipping compound."
                )
            except (FEPConfigError, Exception) as exc:
                self._handle_generic_error(
                    rec, exc, fep_results,
                    work_dir, mutant_pdbqts, center, box_size,
                )

        return fep_results

    # ── Internal helpers ────────────────────────────────────────────

    def _check_pharmacophore_match(self, mol: Chem.Mol) -> bool:
        return check_pharmacophore_match(
            mol,
            query=self.pharmacophore_query,
            min_matches=self.config.pharmacophore_min_matches,
            tolerance=self.config.pharmacophore_tolerance,
            config=self.config,
        )

    def _handle_convergence_error(
        self,
        rec: CompoundRecord,
        exc: Exception,
        receptor_pdb: str,
        fep_results: List[CompoundRecord],
        work_dir: str,
        mutant_pdbqts: List[str],
        center: Any,
        box_size: Any,
    ) -> None:
        from .fep_engine import (
            FEPResistanceCalculator,
            FEPConvergenceError,
        )
        from .analysis import profile_resistance_mutation_sensitivity

        log.warning(
            f"  {rec.compound_id}: FEP convergence error ({exc}) — "
            "retrying with increased lambda windows."
        )
        try:
            retry_calc = FEPResistanceCalculator(
                receptor_wt_pdb=receptor_pdb,
                receptor_mut_pdb=receptor_pdb,
                ligand_rdkit=rec.mol,
            )
            retry_result = retry_calc.retry_with_increased_windows()
            rec.resistance_stability_score = retry_result.delta_delta_g
            fep_results.append(rec)
            log.info(
                f"  {rec.compound_id}: FEP ΔΔG (retry) = "
                f"{retry_result.delta_delta_g:.3f} kcal/mol "
                f"(confidence={retry_result.confidence:.2f})"
            )
        except (FEPConvergenceError, Exception) as retry_exc:
            self._maybe_apply_heuristic_fallback(
                rec, retry_exc, fep_results,
                work_dir, mutant_pdbqts, center, box_size,
            )

    def _handle_generic_error(
        self,
        rec: CompoundRecord,
        exc: Exception,
        fep_results: List[CompoundRecord],
        work_dir: str,
        mutant_pdbqts: List[str],
        center: Any,
        box_size: Any,
    ) -> None:
        from .analysis import profile_resistance_mutation_sensitivity

        if self.config.use_heuristic_resistance_fallback and mutant_pdbqts:
            log.warning(
                f"  {rec.compound_id}: FEP failed ({exc}), "
                "falling back to heuristic resistance profiling."
            )
            try:
                heuristic_score = profile_resistance_mutation_sensitivity(
                    rec, work_dir, mutant_pdbqts, center, box_size,
                )
                rec.resistance_stability_score = heuristic_score
                if heuristic_score is not None:
                    fep_results.append(rec)
                    log.info(
                        f"  {rec.compound_id}: Heuristic resistance score = "
                        f"{heuristic_score:.3f}"
                    )
                else:
                    log.warning(
                        f"  {rec.compound_id}: Heuristic fallback also "
                        "returned None."
                    )
            except Exception as he:
                log.warning(
                    f"  {rec.compound_id}: Heuristic fallback also failed "
                    f"({he})."
                )
        else:
            if not mutant_pdbqts and self.config.use_heuristic_resistance_fallback:
                log.warning(
                    f"  {rec.compound_id}: FEP failed ({exc}) — "
                    "no mutant PDBQTs available for heuristic fallback."
                )
            else:
                log.warning(f"  {rec.compound_id}: FEP skipped — {exc}")

    def _maybe_apply_heuristic_fallback(
        self,
        rec: CompoundRecord,
        retry_exc: Exception,
        fep_results: List[CompoundRecord],
        work_dir: str,
        mutant_pdbqts: List[str],
        center: Any,
        box_size: Any,
    ) -> None:
        from .analysis import profile_resistance_mutation_sensitivity

        if self.config.use_heuristic_resistance_fallback and mutant_pdbqts:
            log.warning(
                f"  {rec.compound_id}: FEP retry also failed ({retry_exc}), "
                "falling back to heuristic resistance profiling."
            )
            try:
                heuristic_score = profile_resistance_mutation_sensitivity(
                    rec, work_dir, mutant_pdbqts, center, box_size,
                )
                rec.resistance_stability_score = heuristic_score
                if heuristic_score is not None:
                    fep_results.append(rec)
                    log.info(
                        f"  {rec.compound_id}: Heuristic resistance score = "
                        f"{heuristic_score:.3f}"
                    )
                else:
                    log.warning(
                        f"  {rec.compound_id}: Heuristic fallback also "
                        "returned None."
                    )
            except Exception as he:
                log.warning(
                    f"  {rec.compound_id}: Heuristic fallback also failed "
                    f"({he})."
                )
        else:
            if not mutant_pdbqts and self.config.use_heuristic_resistance_fallback:
                log.warning(
                    f"  {rec.compound_id}: FEP retry failed — "
                    "no mutant PDBQTs available for heuristic fallback."
                )
            else:
                log.warning(
                    f"  {rec.compound_id}: FEP retry also failed — {retry_exc}"
                )
