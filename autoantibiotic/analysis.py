"""
Analysis Module
================
Top-level analysis orchestration: selectivity, resistance profiling,
and consensus scoring.

Most ML/ADMET implementation details have been moved to submodules
(``ml_scoring.meta_scorer`` and ``admet.predictors``) but are
re-exported here for backward compatibility.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdDistGeom, rdMolAlign, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from .config import CONFIG, ConfigurationError
from .models import CompoundRecord
from .docking import _parallel_dock
from .io_utils import log
from .scoring_metrics import (
    check_key_interactions,
    compute_ifp_similarity,
    compute_pharmacophore_score,
    _parse_pdbqt_ligand_coords,
    _parse_pdb_residue_coords,
)

# ── Backward-compatible re-exports from submodules ─────────────────
# These classes / functions have been moved to ``ml_scoring.meta_scorer``
# and ``admet.predictors`` but are re-imported here so that existing
# imports from ``autoantibiotic.analysis`` continue to work.

# ML scoring (MetaScorer)
from .ml_scoring.meta_scorer import (      # noqa: F401
    MetaScorer,
    _get_meta_scorer,
    predict_meta_score,
)

# ADMET predictors
from .admet.predictors import (            # noqa: F401
    ChemBERTaEmbedder,
    MLADMETPredictor,
    _get_chemberta_embedder,
    _get_ml_admet_predictor,
    _has_basic_nitrogen,
    predict_admet_profile,
    predict_cyp_inhibition,
    predict_herg_risk,
    predict_herg_ml,
    predict_logs,
)

_CacheLike = Optional[Dict[str, float]]


def compute_consensus_score(
    vina_energy: Optional[float],
    shape_score: Optional[float],
    vina_weight: float = CONFIG.consensus_vina_weight,
    shape_weight: float = CONFIG.consensus_shape_weight,
) -> Optional[float]:
    """Compute a weighted consensus score from Vina and Shape scores.

    Returns ``w_vina * |vina| + w_shape * shape`` if both are available,
    or whichever single score is present, or ``None`` if neither exists.
    """
    if vina_energy is not None and shape_score is not None:
        return vina_weight * abs(vina_energy) + shape_weight * shape_score
    if vina_energy is not None:
        return abs(vina_energy)
    if shape_score is not None:
        return shape_score
    return None


def compute_selectivity_index(
    pb2pa_energy: float, human_avg_energy: float,
) -> float:
    """Selectivity Index (SI).

    SI = |PBP2a Energy| / |Human Avg Energy|

    Returns 0.0 if either energy is non-negative or human average is near zero.
    """
    if pb2pa_energy >= 0 or human_avg_energy >= 0:
        return 0.0
    return abs(pb2pa_energy) / abs(human_avg_energy) if abs(human_avg_energy) > 1e-6 else 0.0


def profile_resistance_mutation_sensitivity(
    record: CompoundRecord,
    work_dir: str,
    mutant_pdbqts: List[str],
    center: np.ndarray,
    box_size: tuple,
) -> Optional[float]:
    """Dock a candidate against multiple mutant receptor variants and
    compute the standard deviation of binding energies.

    A high standard deviation indicates that the compound's binding
    affinity is sensitive to mutational changes — i.e. elevated
    resistance risk.

    When ``CONFIG.use_fep_resistance`` is True, the function uses
    OpenMM-based Free Energy Perturbation (FEP) to compute ΔΔG
    between wild-type and mutant receptor binding the same ligand.
    If FEP is unavailable or fails, the original heuristic standard
    deviation approach is used as a fallback.

    Returns
    -------
    Optional[float]
        Standard deviation of binding energies across mutants, or
        ``None`` if fewer than 2 valid energies are computed.
    """
    if not mutant_pdbqts:
        return None

    # ── FEP-based resistance profiling ──
    if CONFIG.use_fep_resistance:
        from .fep_engine import FEPResistanceCalculator

        try:
            # Use first mutant PDBQT as the "wild-type" reference
            # (in practice, this should be the actual WT PDB)
            wt_pdbqt = mutant_pdbqts[0]
            wt_pdb = wt_pdbqt.replace(".pdbqt", ".pdb")

            # Build RDKit mol from the record
            ligand_mol = record.mol
            if ligand_mol is None:
                ligand_mol = Chem.MolFromSmiles(record.smiles)
            if ligand_mol is None:
                ligand_mol = Chem.MolFromSmiles(
                    record.smiles,
                    sanitize=False,
                )
                if ligand_mol is None:
                    return None

            # Create FEP calculator and compute ΔΔG
            calc = FEPResistanceCalculator(
                receptor_wt_pdb=wt_pdb,
                receptor_mut_pdb=mutant_pdbqts[0],
                ligand_rdkit=ligand_mol,
            )
            result = calc.calculate_ddg()

            # Convert ΔΔG to an equivalent "stability score" for
            # consistency with the existing API
            if result.delta_delta_g is not None:
                # Use absolute ΔΔG as the stability score
                return abs(result.delta_delta_g)
        except ConfigurationError:
            raise
        except Exception as exc:
            log.warning(
                f"  FEP resistance profiling failed: {exc}. "
                "Falling back to heuristic."
            )

    # ── Heuristic fallback (original behaviour) ──
    energies: List[float] = []
    for i, mut_pdbqt in enumerate(mutant_pdbqts):
        from .docking import dock_compound

        e = dock_compound(
            record, mut_pdbqt, center, box_size,
            work_dir, f"mut_{i}",
        )
        if e is not None:
            energies.append(e)

    if len(energies) < 2:
        return None

    return float(np.std(energies, ddof=1))


def profile_resistance_risk(
    record: CompoundRecord,
    work_dir: str,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: tuple,
    mutant_pdbqts: Optional[List[str]] = None,
) -> str:
    """Energy-based heuristic proxy for resistance-risk profiling."""
    notes: List[str] = []

    act_thresh = CONFIG.resistance_energy_active_threshold
    allo_thresh = CONFIG.resistance_energy_allosteric_threshold
    mw_thresh = CONFIG.resistance_mw_threshold
    rot_thresh = CONFIG.resistance_rot_threshold
    qed_thresh = CONFIG.resistance_qed_threshold

    if record.pb2pa_active_energy is not None and record.pb2pa_active_energy < act_thresh:
        notes.append("Energy profile suggests interaction near catalytic Ser403 (heuristic proxy).")

    if record.pb2pa_allosteric_energy is not None and record.pb2pa_allosteric_energy < allo_thresh:
        if record.pb2pa_active_energy is None or record.pb2pa_active_energy > act_thresh:
            notes.append("Allosteric binder (Ala237/Met241/Tyr159 pocket). Novel mechanism.")

    if record.mol is not None:
        mw = Descriptors.MolWt(record.mol)
        if mw > mw_thresh:
            notes.append(f"High MW (>{mw_thresh:.0f}) — broad interaction surface, may contact multiple residues.")
        n_rot = Descriptors.NumRotatableBonds(record.mol)
        if n_rot < rot_thresh:
            notes.append(f"Rigid scaffold — reduced entropic penalty, may enhance binding specificity.")

    if record.qed_score > qed_thresh:
        notes.append(f"High drug-likeness (QED > {qed_thresh}) — good developability profile.")

    # Mutation-sensitivity sampling
    if CONFIG.use_mutation_sampling and mutant_pdbqts:
        mut_std = profile_resistance_mutation_sensitivity(
            record, work_dir, mutant_pdbqts, center, box_size,
        )
        record.resistance_stability_score = mut_std
        if mut_std is not None:
            notes.append(
                f"Mutation binding-energy std = {mut_std:.2f} kcal/mol"
                f" ({'HIGH' if mut_std > 1.0 else 'MODERATE' if mut_std > 0.5 else 'LOW'} resistance risk)."
            )

    if not notes:
        notes.append("No specific resistance flags identified.")

    return "; ".join(notes)


def analyze_selectivity_and_resistance(
    top10: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: _CacheLike = None,
    use_cache: bool = False,
    water_results: Any = None,
) -> List[CompoundRecord]:
    """Phase 4 — Selectivity & Resistance Analysis.

    Docks top 10 against human off-targets, computes SI, profiles resistance risk.

    Parameters
    ----------
    water_results : WaterAnalysisResult, optional
        Result from crystallographic water analysis.  When provided,
        ``rec.water_displacement_energy`` is populated from the mean
        displacement energy of all waters.
    """
    log.info("─── Phase 4: Selectivity & Resistance Analysis ───")

    use_vina = deps.get("USE_VINA", False)
    if not use_vina:
        log.warning("  Vina unavailable — skipping selectivity docking. Flagging all as uncertain.")
        for rec in top10:
            rec.selectivity_index = 1.0
            rec.resistance_notes = "Selectivity not assessed (Vina unavailable)."
        return top10

    trypsin_target = targets.get("trypsin")
    ces1_target = targets.get("CES1")
    if trypsin_target is None or ces1_target is None:
        log.warning("  Off-target data missing — skipping selectivity docking.")
        for rec in top10:
            rec.selectivity_index = 1.0
            rec.resistance_notes = "Selectivity not assessed (off-target data missing)."
        return top10

    log.info("  Docking top 10 vs Human Trypsin (1UTN)…")
    trypsin_center = trypsin_target.get("active_center", np.array([0.0, 0.0, 0.0]))
    trypisn_items = [(r.compound_id, r.smiles) for r in top10]
    trypsin_results = _parallel_dock(
        trypisn_items, targets["trypsin"]["pdbqt"],
        trypsin_center, CONFIG.offtarget_box_size,
        work_dir, "trypsin", n_jobs=min(4, len(top10)),
        cache=cache, use_cache=use_cache,
    )
    cid_map = {r.compound_id: r for r in top10}
    for cid, energy in trypsin_results:
        if cid in cid_map:
            cid_map[cid].human_trypsin_energy = energy

    log.info("  Docking top 10 vs Human Carboxylesterase 1 (1YA4)…")
    ces1_center = ces1_target.get("active_center", np.array([0.0, 0.0, 0.0]))
    ces1_items = [(r.compound_id, r.smiles) for r in top10]
    ces1_results = _parallel_dock(
        ces1_items, targets["CES1"]["pdbqt"],
        ces1_center, CONFIG.offtarget_box_size,
        work_dir, "ces1", n_jobs=min(4, len(top10)),
        cache=cache, use_cache=use_cache,
    )
    for cid, energy in ces1_results:
        if cid in cid_map:
            cid_map[cid].human_ces1_energy = energy

    for rec in top10:
        energies_human = [
            e for e in (rec.human_trypsin_energy, rec.human_ces1_energy)
            if e is not None
        ]
        if not energies_human:
            log.warning(f"  {rec.compound_id}: No human docking data. SI = N/A.")
            rec.selectivity_index = 1.0
            continue

        human_avg = np.mean(energies_human)
        pb2pa_best = (
            rec.pb2pa_active_energy if rec.pb2pa_active_energy is not None
            else rec.pb2pa_allosteric_energy
        )
        if pb2pa_best is None:
            rec.selectivity_index = 1.0
            continue

        si = compute_selectivity_index(pb2pa_best, human_avg)
        rec.selectivity_index = si

        if si < CONFIG.selectivity_index_threshold:
            log.warning(
                f"  {rec.compound_id}: Low selectivity (SI = {si:.2f} < {CONFIG.selectivity_index_threshold}). "
                "Flagged for off-target risk."
            )
        else:
            log.info(f"  {rec.compound_id}: SI = {si:.2f} (pass).")

    pb2pa = targets["PBP2a"]
    receptor_pdb = pb2pa["pdbqt"].replace(".pdbqt", ".pdb")
    if not os.path.isfile(receptor_pdb):
        receptor_pdb = os.path.join(
            os.path.dirname(pb2pa["pdbqt"]),
            "PBP2a_clean.pdb",
        )

    log.info("  Checking key interactions for top candidates…")

    mutant_pdbqts: Optional[List[str]] = None
    if CONFIG.use_mutation_sampling:
        mutant_dir = Path(CONFIG.output_dir) / "mutants"
        if mutant_dir.exists():
            mutant_pdbqts = sorted(str(p) for p in mutant_dir.glob("*.pdbqt"))
            if mutant_pdbqts:
                log.info(f"  Mutation-sampling enabled: {len(mutant_pdbqts)} mutant variants.")
            else:
                log.info("  Mutation-sampling enabled but no mutant PDBQTs found.")

    for rec in top10:
        rec.resistance_notes = profile_resistance_risk(
            rec, work_dir,
            pb2pa["pdbqt"],
            pb2pa["allosteric_center"],
            CONFIG.allosteric_box_size,
            mutant_pdbqts=mutant_pdbqts if CONFIG.use_mutation_sampling else None,
        )

        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol
        mol = Chem.MolFromSmiles(rec.smiles) if rec.mol is None else Chem.RWMol(rec.mol)
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = CONFIG.random_seed
        if rdDistGeom.EmbedMolecule(mol, params) >= 0:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".pdbqt", delete=False,
                ) as tmp:
                    tmp_pdbqt = tmp.name
                    conf = mol.GetConformer()
                    tmp.write("ROOT\n")
                    for i in range(mol.GetNumAtoms()):
                        atom = mol.GetAtomWithIdx(i)
                        if atom.GetAtomicNum() == 1:
                            continue
                        pt = conf.GetAtomPosition(i)
                        elem = atom.GetSymbol()
                        tmp.write(
                            f"ATOM  {i+1:5d} {elem:<4s} LIG     1    "
                            f"{pt.x:8.3f}{pt.y:8.3f}{pt.z:8.3f}  "
                            f"1.00  0.00          {elem:>2s}\n"
                        )
                    tmp.write("ENDROOT\n")
                try:
                    allosteric_hits = (
                        CONFIG.min_key_interactions > 0
                        and check_key_interactions(
                            tmp_pdbqt, receptor_pdb,
                            CONFIG.key_interaction_residues_allosteric,
                        )
                    )
                    active_hits = check_key_interactions(
                        tmp_pdbqt, receptor_pdb,
                        CONFIG.key_interaction_residues_active,
                    )
                    if not (allosteric_hits or active_hits):
                        rec.resistance_notes += (
                            "; Warning: No key interactions detected"
                        )
                finally:
                    try:
                        os.unlink(tmp_pdbqt)
                    except OSError:
                        pass
            except Exception as exc:
                log.debug(f"  IFP check failed for {rec.compound_id}: {exc}")

        try:
            ref_smi = CONFIG.control_smiles.get("Ceftaroline", "")
            if ref_smi:
                ref_mol = Chem.MolFromSmiles(ref_smi)
                if ref_mol is not None and rec.mol is not None:
                    rec.ifp_score = compute_ifp_similarity(
                        rec.mol, ref_mol, receptor_pdb,
                    )
                    if rec.ifp_score < CONFIG.ifp_similarity_threshold:
                        rec.resistance_notes += (
                            f"; Warning: Low IFP similarity to reference ligand "
                            f"({rec.ifp_score:.2f})"
                        )
        except Exception as exc:
            log.debug(f"  IFP similarity failed for {rec.compound_id}: {exc}")

        try:
            if water_results is not None and hasattr(water_results, "all_waters") and water_results.all_waters:
                energies = [w.displacement_energy for w in water_results.all_waters]
                if energies:
                    rec.water_displacement_energy = float(np.mean(energies))
        except Exception as exc:
            log.debug(f"  Water displacement energy failed for {rec.compound_id}: {exc}")

    # ── ADMET profiling on top 10 ──
    log.info("  Computing ADMET profiles for top candidates…")
    for rec in top10:
        predict_admet_profile(rec)
        log.debug(f"  {rec.compound_id}: {rec.admet_flags}")

    log.info("─── Phase 4 complete ───")
    return top10
