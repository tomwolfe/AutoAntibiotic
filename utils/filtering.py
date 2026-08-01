"""
Filtering utilities
===================

Phase 2.2 of the discovery pipeline: structural, similarity, ADMET and PAINS
filtering of candidate compounds.

The filtering constants (β-lactam SMARTS, reference antibiotics, similarity
thresholds, diversity floor) live in ``config.constants`` and are imported at
module top level. This keeps the ``utils`` package free of a circular import
with ``discovery_pipeline``.
"""

from __future__ import annotations

import logging

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from rdkit.DataStructs import TanimotoSimilarity

from config.constants import (
    SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD_RELAXED,
    DIVERSITY_MIN_COUNT,
    REFERENCE_ANTIBIOTICS,
    BETA_LACTAM_SMARTS,
)

# Shared logger: same name as the one configured in discovery_pipeline.
log = logging.getLogger("AutoAntibiotic")


# ── hERG / CYP450 ADMET Filters ──────────────────────────────────────────────
# Simple rule-based filters to flag pharmacological off-target liabilities.
# These are NOT predictive models — they flag structural features associated
# with hERG blockade and CYP450 inhibition using published rules-of-thumb.

HERG_RISK_SMARTS = [
    "[N;H0;$(N(-[C])-[C])]",        # tertiary amine (common hERG motif)
    "[nH]1ccc2ccccc12",             # indole
    "[#7]1([#6])[#6][#6][#6][#6]1",  # N-alkyl piperidine
    "[#7]1[#6][#6][#6][#6][#6]1",    # piperidine
    "c1ccc2[nH]c3ccccc3c2c1",        # carbazole-like
]

CYP450_INHIBITOR_SMARTS = [
    "[NX3;H2;$(N~C(~O))]",          # aniline / aromatic amine (CYP2E1)
    "[NX2]=[CX3]([NX3])[NX3]",      # guanidine / amidine (CYP2D6)
    "c1ccc(O)c(OC)c1",              # guaiacol (CYP1A2/2E1)
    "[SX2]c1ccccc1",                # thiophenol
    "c1ccc2c(c1)OCO2",              # methylenedioxyphenyl (CYP2D6/3A4)
    "c1ccc2c(c1)OCCO2",             # ethylenedioxyphenyl
    "c1ccc2oc(=O)nc2c1",            # coumarin-like
    "[CX3](=O)[OX2][CX3](=O)",      # anhydride
    "[NX3;H2]c1ccccc1",             # primary aromatic amine
    "[CX3](=O)[CX3]([F,Cl,Br,I])",  # alpha-halo ketone
]


# ── MD Stability Filter ───────────────────────────────────────────────────────
# Post-docking filter: drop candidates whose OpenMM explicit-solvent MD
# simulation indicates unstable binding (high ligand RMSD or zero H-bond
# occupancy for catalytic residues).

MD_STABILITY_MAX_RMSD = 3.0  # mean ligand RMSD threshold (Å)
MD_STABILITY_MIN_HBOND = 1.0  # minimum H-bond contacts for any catalytic residue

# 10 ns multi-replica binding-stability criteria (paper §4.x / D3):
#   "Validated"   — mean ligand RMSD < 3.0 Å over the last 5 ns in >= 2/3
#                   replicas AND >= 50% H-bond occupancy with Ser403 OG.
#   "Metastable"  — RMSD 3–5 Å and >= 25% H-bond retention (Ser403 OG).
#   "Dissociated" — RMSD > 5 Å or zero H-bonds.
VALIDATED_RMSD_MAX = 3.0          # Å, mean over last 5 ns
METASTABLE_RMSD_MAX = 5.0         # Å
VALIDATED_SER403_HBOND_OCC = 0.50  # fraction of frames with Ser403 OG contact
METASTABLE_SER403_HBOND_OCC = 0.25
MIN_REPLICAS_FOR_VALIDATED = 2    # out of 3 replicas
REPLICA_COUNT = 3


def classify_md_stability(
    replicas: list, ligand_rmsd_key: str = "ligand_rmsd_mean_last5ns_A",
) -> str:
    """Classify a candidate's binding stability from per-replica MD metrics.

    Each entry of *replicas* is a dict with at least:
        - ligand_rmsd_mean_last5ns_A (float): mean ligand RMSD over the final
          5 ns of production, relative to the energy-minimised pose.
        - ser403_og_hbond_occupancy (float): fraction of production frames in
          which the ligand stays within H-bond distance of Ser403 OG.
    Missing replicas or missing metrics are ignored when counting consensus
    (never count an absent replica as evidence of stability).

    Returns one of:
        "Validated"   — >= 2/3 replicas with RMSD < 3.0 Å AND Ser403 occupancy
                        >= 0.50.
        "Metastable"  — not Validated, but >= 1 replica with RMSD < 5.0 Å AND
                        Ser403 occupancy >= 0.25.
        "Dissociated" — all remaining cases (RMSD >= 5.0 Å in >= 2/3 replicas
                        or zero retained H-bonds).
    """
    qualifying = []
    for rep in replicas or []:
        if not isinstance(rep, dict):
            continue
        rmsd = rep.get(ligand_rmsd_key)
        occ = rep.get("ser403_og_hbond_occupancy")
        if rmsd is None or occ is None:
            continue
        qualifying.append(rep)
    n = len(qualifying)
    if n == 0:
        return "Dissociated"

    n_validated = sum(
        1 for rep in qualifying
        if rep[ligand_rmsd_key] < VALIDATED_RMSD_MAX
        and rep["ser403_og_hbond_occupancy"] >= VALIDATED_SER403_HBOND_OCC
    )
    if n_validated >= MIN_REPLICAS_FOR_VALIDATED:
        return "Validated"

    n_metastable = sum(
        1 for rep in qualifying
        if rep[ligand_rmsd_key] < METASTABLE_RMSD_MAX
        and rep["ser403_og_hbond_occupancy"] >= METASTABLE_SER403_HBOND_OCC
    )
    if n_metastable >= 1:
        return "Metastable"
    return "Dissociated"


def filter_by_md_stability(record, md_results=None) -> dict:
    """Check if a candidate passes the MD-stability gate.

    Reads from *md_results* (a dict keyed by compound_id, as stored in
    ``output/openmm_minimization_results.json``) and applies two criteria:

        1. Mean ligand RMSD during the simulation <= MD_STABILITY_MAX_RMSD (3.0 Å).
        2. At least one H-bond contact observed for any catalytic residue
           (Ser403, Lys406, Tyr446) during the simulation.

    When *md_results* is None or the compound has no MD data, the function
    returns an {"unstable": True, "reason": "no MD data"} — the candidate
    cannot be validated without simulation.

    Returns:
        {"unstable": bool, "reason": str}
    """
    if md_results is None:
        return {"unstable": True, "reason": "no MD data"}
    entry = md_results.get(record.compound_id)
    if entry is None:
        return {"unstable": True, "reason": f"no MD data for {record.compound_id}"}
    if not entry.get("success", False):
        return {"unstable": True, "reason": "MD simulation failed"}
    md = entry.get("md", {})
    if not md.get("success", False):
        return {"unstable": True, "reason": "MD production failed"}
    mean_rmsd = md.get("ligand_rmsd_mean_A", float("inf"))
    hbond = md.get("hbond_occupancy", {})
    max_contacts = max(
        (hbond.get(r, {}).get("n_contacts", 0) for r in ("SER403_OG", "LYS406_NZ", "TYR446_OH")),
        default=0,
    )
    if mean_rmsd > MD_STABILITY_MAX_RMSD:
        return {
            "unstable": True,
            "reason": f"ligand RMSD {mean_rmsd:.2f} Å > {MD_STABILITY_MAX_RMSD} Å threshold",
        }
    if max_contacts < MD_STABILITY_MIN_HBOND:
        return {
            "unstable": True,
            "reason": f"no stable H-bond contacts (max={max_contacts}) for catalytic residues",
        }
    return {"unstable": False, "reason": f"stable: RMSD={mean_rmsd:.2f} Å, H-bond contacts={int(max_contacts)}"}


def _has_smarts_substruct(mol: Chem.Mol, smarts_list) -> bool:
    """Check if *mol* matches any SMARTS pattern in *smarts_list*."""
    for smarts in smarts_list:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            return True
    return False


def predict_herg_risk(mol: Chem.Mol) -> dict:
    """Predict hERG blockade risk using structural alerts + physicochemical rules.

    Flags a compound as HIGH risk when:
      - It matches a known hERG-phoric SMARTS pattern, OR
      - MW > 350 AND logP > 3.5 AND it contains a basic nitrogen.

    Returns:
        {"risk": "High"|"Moderate"|"Low", "flags": [list of reasons]}
    """
    flags = []
    if _has_smarts_substruct(mol, HERG_RISK_SMARTS):
        flags.append("hERG-phoric substructure detected")
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        n_basic = sum(1 for a in mol.GetAtoms()
                      if a.GetAtomicNum() == 7 and a.GetDegree() <= 3
                      and not a.IsInRing())
        if mw > 350 and logp > 3.5 and n_basic >= 1:
            flags.append(f"high MW ({mw:.0f}), high logP ({logp:.1f}), basic N={n_basic}")
    except Exception:
        pass
    if len(flags) >= 2:
        return {"risk": "High", "flags": flags}
    elif len(flags) == 1:
        return {"risk": "Moderate", "flags": flags}
    return {"risk": "Low", "flags": []}


def predict_cyp_inhibition(mol: Chem.Mol) -> dict:
    """Predict CYP450 inhibition liability using structural alerts.

    Returns:
        {"risk": "High"|"Moderate"|"Low", "flags": [list of CYP isoforms]}
    """
    if _has_smarts_substruct(mol, CYP450_INHIBITOR_SMARTS):
        return {"risk": "Moderate", "flags": ["CYP alert substructure detected"]}
    return {"risk": "Low", "flags": []}


def apply_filters(
    records: "List[CompoundRecord]",
    similarity_threshold: Optional[float] = None,
    return_counts: bool = False,
) -> "List[CompoundRecord]":
    """
    Phase2.2 — Apply structural, similarity, ADMET, and PAINS filters.

    Filter chain:
        1. Structural exclusion (β-lactam SMARTS).
        2. Similarity filter vs reference antibiotics (Morgan FP, Tc < threshold).
        3. ADMET: Lipinski Rule of 5 + QED > 0.5.
        4. PAINS alerts via RDKit FilterCatalog.
        5. Diversity check: if < 100 pass, relax similarity to 0.5.

    Args:
        records: Input compound records.
        similarity_threshold: Initial Tanimoto cutoff (default 0.3).
        return_counts: When True, records are returned with an attribute
            ``_funnel_counts`` containing the intermediate filter counts.

    Returns:
        Filtered list of CompoundRecord (with computed ADMET/similarity fields).
    """
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD
    qed_floor = 0.5

    log.info("─── Phase 2: Filtering ───")

    # ── Precompute reference fingerprints ──
    ref_mols = {}
    for name, smi in REFERENCE_ANTIBIOTICS.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ref_mols[name] = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048,
            )

    # β-lactam SMARTS matcher
    lactam_pattern = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)

    # PAINS filter catalog
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains_catalog = FilterCatalog(pains_params)

    # Brenk alerts filter catalog
    brenk_params = FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_catalog = FilterCatalog(brenk_params)

    def _filter_pass(threshold: float, qed_gate: float) -> tuple:
        """Run the similarity + ADMET + PAINS filter chain on the original records.
        Returns (passed_list, counts_dict)."""
        passed = []
        skipped_structural = 0
        skipped_similarity = 0
        skipped_admet = 0
        skipped_pains = 0
        skipped_brenk = 0

        for record in records:
            if record.mol is None:
                mol = Chem.MolFromSmiles(record.smiles)
                if mol is None:
                    continue
                record.mol = mol
            mol = record.mol

            # 1. Structural — reject β-lactams
            if mol.HasSubstructMatch(lactam_pattern):
                skipped_structural += 1
                continue

            # 2. Similarity — max Tc vs reference antibiotics
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            max_sim = 0.0
            for ref_fp in ref_mols.values():
                sim = TanimotoSimilarity(fp, ref_fp)
                max_sim = max(max_sim, sim)
            record.max_similarity = max_sim

            if max_sim >= threshold:
                skipped_similarity += 1
                continue

            # 3. ADMET — Lipinski + QED
            try:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                lipinski_ok = (mw <= 500) and (logp <= 5.0) and (hbd <= 5) and (hba <= 10)
                qed = QED.qed(mol)
            except Exception:
                continue

            record.passes_lipinski = lipinski_ok
            record.qed_score = qed

            # Antibiotics often have MW > 500 and many HBA; relax for this target
            if not lipinski_ok and not (mw <= 650 and hba <= 12 and qed > 0.4):
                skipped_admet += 1
                continue
            if qed is not None and qed <= 0.3:
                skipped_admet += 1
                continue

            if qed is not None and qed <= qed_gate:
                skipped_admet += 1
                continue

            # 4. PAINS
            pains_match = pains_catalog.HasMatch(mol)
            record.passes_pains = not pains_match
            if pains_match:
                skipped_pains += 1
                continue

            # 5. Brenk alerts
            brenk_match = brenk_catalog.HasMatch(mol)
            if brenk_match:
                skipped_brenk += 1
                continue

            passed.append(record)

        log.info(f"  Structural exclusion (β-lactam): {skipped_structural} removed.")
        log.info(f"  Similarity filter (Tc < {threshold}): {skipped_similarity} removed.")
        log.info(f"  ADMET filter (Lipinski + QED > {qed_gate}): {skipped_admet} removed.")
        log.info(f"  PAINS filter: {skipped_pains} removed.")
        log.info(f"  Brenk alerts: {skipped_brenk} removed.")
        log.info(f"  Passed filters: {len(passed)} compounds.")
        counts = {
            "total_input": len(records),
            "skipped_structural": skipped_structural,
            "skipped_similarity": skipped_similarity,
            "skipped_admet": skipped_admet,
            "skipped_pains": skipped_pains,
            "skipped_brenk": skipped_brenk,
            "passed": len(passed),
        }
        return passed, counts

    passed, counts_strict = _filter_pass(similarity_threshold, qed_floor)

    if len(passed) < DIVERSITY_MIN_COUNT:
        log.info(
            f"  Only {len(passed)} compounds passed filters (< {DIVERSITY_MIN_COUNT}). "
            f"Relaxing similarity threshold to {SIMILARITY_THRESHOLD_RELAXED}."
        )
        passed, counts_relaxed = _filter_pass(SIMILARITY_THRESHOLD_RELAXED, 0.5)
        counts_strict = counts_relaxed
        counts_strict["relaxed"] = True

    if return_counts:
        for r in passed:
            r._funnel_counts = counts_strict

    log.info("─── Phase 2 complete ───")
    return passed

