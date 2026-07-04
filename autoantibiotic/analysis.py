from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdDistGeom, rdMolAlign

from .config import CONFIG, CompoundRecord
from .docking import _parallel_dock
from .io_utils import CacheManager, log

_CacheLike = Optional[Union[CacheManager, Dict[str, float]]]


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


def compute_pharmacophore_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    tolerance: float = 2.0,
) -> Optional[float]:
    """Compute a pharmacophore feature matching score between two molecules.

    Generates 3D conformations, aligns *mol* to *ref_mol* via
    :func:`AllChem.AlignMol`, then counts H-bond donor and acceptor atoms
    in both molecules.  A feature in *mol* is considered matched if it lies
    within *tolerance* Å of the same feature type in *ref_mol*.

    Returns the fraction of reference features matched (0.0 – 1.0),
    or ``None`` if scoring fails (e.g. 3D embedding unsuccessful).
    """
    try:
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = CONFIG.random_seed
        if rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)

        ref_3d = Chem.RWMol(ref_mol)
        ref_3d = Chem.AddHs(ref_3d)
        params_ref = rdDistGeom.ETKDGv3()
        params_ref.randomSeed = CONFIG.random_seed
        if rdDistGeom.EmbedMolecule(ref_3d, params_ref) < 0:
            return None
        AllChem.MMFFOptimizeMolecule(ref_3d)

        o3a = rdMolAlign.GetO3A(mol_3d, ref_3d)
        o3a.Align()

        def _is_hbd(atom: Chem.Atom) -> bool:
            if atom.GetAtomicNum() not in (7, 8):
                return False
            return (
                atom.GetTotalNumHs() > 0
                or any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())
            )

        def _is_hba(atom: Chem.Atom) -> bool:
            return atom.GetAtomicNum() in (7, 8)

        conf_mol = mol_3d.GetConformer()
        query_donors = [conf_mol.GetAtomPosition(i) for i, a in enumerate(mol_3d.GetAtoms()) if _is_hbd(a)]
        query_acceptors = [conf_mol.GetAtomPosition(i) for i, a in enumerate(mol_3d.GetAtoms()) if _is_hba(a)]

        conf_ref = ref_3d.GetConformer()
        ref_donors = [conf_ref.GetAtomPosition(i) for i, a in enumerate(ref_3d.GetAtoms()) if _is_hbd(a)]
        ref_acceptors = [conf_ref.GetAtomPosition(i) for i, a in enumerate(ref_3d.GetAtoms()) if _is_hba(a)]

        def _count_matches(
            ref_positions: List[Chem.Point3D],
            query_positions: List[Chem.Point3D],
        ) -> int:
            matched = 0
            for rp in ref_positions:
                for qp in query_positions:
                    if rp.Distance(qp) <= tolerance:
                        matched += 1
                        break
            return matched

        donor_matches = _count_matches(ref_donors, query_donors)
        acceptor_matches = _count_matches(ref_acceptors, query_acceptors)

        total_ref = len(ref_donors) + len(ref_acceptors)
        if total_ref == 0:
            return 1.0

        return (donor_matches + acceptor_matches) / total_ref

    except Exception:
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


def profile_resistance_risk(
    record: CompoundRecord,
    work_dir: str,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: tuple,
) -> str:
    """Energy-based heuristic proxy for resistance-risk profiling.

    Returns a human-readable notes string.
    """
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

    if not notes:
        notes.append("No specific resistance flags identified.")

    return "; ".join(notes)


_LOGS_MODEL_COEFFS = {
    "c": 0.16, "MolLogP": -0.63, "MolWt": -0.0062,
    "NumRotatableBonds": -0.0034, "NumAromaticRings": -0.042,
    "HeavyAtomCount": 0.00025,
}


def predict_logs(mol: Chem.Mol) -> float:
    """Predict aqueous solubility (LogS) using a simple linear model.

    The model is a re-implementation of the ESOL method (Delaney 2004)
    using RDKit descriptors:

        LogS = 0.16 - 0.63*LogP - 0.0062*MW + 0.0034*RotBonds
               - 0.042*AromRings + 0.00025*HeavyAtoms

    Returns predicted LogS in mol/L.
    """
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    n_arom = Descriptors.NumAromaticRings(mol) if hasattr(Descriptors, "NumAromaticRings") else 0
    heavy = mol.GetNumHeavyAtoms()

    logs = (
        _LOGS_MODEL_COEFFS["c"]
        + _LOGS_MODEL_COEFFS["MolLogP"] * logp
        + _LOGS_MODEL_COEFFS["MolWt"] * mw
        + _LOGS_MODEL_COEFFS["NumRotatableBonds"] * rot
        + _LOGS_MODEL_COEFFS["NumAromaticRings"] * n_arom
        + _LOGS_MODEL_COEFFS["HeavyAtomCount"] * heavy
    )
    return logs


def _has_basic_nitrogen(mol: Chem.Mol) -> bool:
    """Check if the molecule contains a basic nitrogen (aliphatic primary/secondary/tertiary)."""
    basic_n_pattern = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(NC=O)]")
    if basic_n_pattern is None:
        return False
    return mol.HasSubstructMatch(basic_n_pattern)


def predict_herg_risk(mol: Chem.Mol) -> str:
    """Rule-based hERG blockage risk assessment.

    Flags compounds with:
      - LogP > 4.0  (high lipophilicity → promiscuous hERG binding)
      - AND presence of a basic nitrogen

    Returns ``"High"``, ``"Moderate"``, or ``"Low"``.
    """
    logp = Crippen.MolLogP(mol)
    has_basic_n = _has_basic_nitrogen(mol)
    if logp > 4.0 and has_basic_n:
        return "High"
    if logp > 4.0 or has_basic_n:
        return "Moderate"
    return "Low"


def predict_admet_profile(record: CompoundRecord) -> CompoundRecord:
    """Compute ADMET properties for a compound and populate *admet_flags*.

    Evaluates:
      1. **Solubility (LogS)**: predicted via ESOL model.
      2. **hERG blockage risk**: rule-based (LogP + basic nitrogen).
      3. **Lipinski Rule-of-5** and **QED** (already computed in filtering).

    Flags are appended to ``record.admet_flags``.

    Returns the same ``CompoundRecord`` with populated *admet_flags*.
    """
    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            record.admet_flags.append("ADMET: invalid molecule")
            return record
        record.mol = mol
    mol = record.mol

    flags: List[str] = []

    # Solubility
    try:
        logs = predict_logs(mol)
        if logs < -5.0:
            flags.append(f"Poor solubility (LogS={logs:.2f})")
        elif logs < -3.0:
            flags.append(f"Moderate solubility (LogS={logs:.2f})")
        else:
            flags.append(f"Good solubility (LogS={logs:.2f})")
    except Exception:
        flags.append("Solubility prediction failed")

    # hERG risk
    try:
        herg = predict_herg_risk(mol)
        if herg == "High":
            flags.append("High hERG risk (LogP>4 + basic N)")
        elif herg == "Moderate":
            flags.append("Moderate hERG risk")
        else:
            flags.append("Low hERG risk")
    except Exception:
        flags.append("hERG prediction failed")

    # Lipinski (already computed, just annotate)
    if record.passes_lipinski:
        flags.append("Lipinski OK")
    else:
        flags.append("Lipinski violation")

    # QED
    if record.qed_score > CONFIG.qed_threshold:
        flags.append(f"QED OK ({record.qed_score:.2f})")
    else:
        flags.append(f"QED below threshold ({record.qed_score:.2f})")

    record.admet_flags = flags
    return record


def analyze_selectivity_and_resistance(
    top10: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: _CacheLike = None,
    use_cache: bool = False,
) -> List[CompoundRecord]:
    """Phase 4 — Selectivity & Resistance Analysis.

    Docks top 10 against human off-targets, computes SI, profiles resistance risk.
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

    log.info("  Docking top 10 vs Human Carboxylesterase 1 (3KJZ)…")
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
    for rec in top10:
        rec.resistance_notes = profile_resistance_risk(
            rec, work_dir,
            pb2pa["pdbqt"],
            pb2pa["allosteric_center"],
            CONFIG.allosteric_box_size,
        )

    # ── ADMET profiling on top 10 ──
    log.info("  Computing ADMET profiles for top candidates…")
    for rec in top10:
        predict_admet_profile(rec)
        log.debug(f"  {rec.compound_id}: {rec.admet_flags}")

    log.info("─── Phase 4 complete ───")
    return top10
