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
    RECALL_MODE,
    RECALL_QED_FLOOR,
)

# Shared logger: same name as the one configured in discovery_pipeline.
log = logging.getLogger("AutoAntibiotic")


def apply_filters(
    records: "List[CompoundRecord]",
    similarity_threshold: Optional[float] = None,
    recal_mode: bool = RECALL_MODE,
) -> "List[CompoundRecord]":
    """
    Phase2.2 — Apply structural, similarity, ADMET, and PAINS filters.

    Filter chain:
        1. Structural exclusion (β-lactam SMARTS).
        2. Similarity filter vs reference antibiotics (Morgan FP, Tc < threshold).
            3. ADMET: Lipinski Rule of 5 + QED > 0.7 (or > 0.4 in
               recall_mode, see below).
        4. PAINS alerts via RDKit FilterCatalog.
        5. Diversity check: if < 100 pass, relax similarity to 0.5.

    Args:
        records: Input compound records.
        similarity_threshold: Initial Tanimoto cutoff.
        recal_mode: When True, relax the filter chain so known PBP2a
            binders (ceftaroline, meropenem) survive (paper §4.4): the
            similarity threshold falls back to SIMILARITY_THRESHOLD_RELAXED
            and the QED floor is lowered from 0.7 to RECALL_QED_FLOOR (0.4).

    Returns:
        Filtered list of CompoundRecord (with computed ADMET/similarity fields).
    """
    if similarity_threshold is None:
        # In recall mode start from the relaxed similarity threshold so the known
        # binders (which are highly similar to the reference antibiotics by
        # design) are not dropped at step 2.
        similarity_threshold = (
            SIMILARITY_THRESHOLD_RELAXED if recal_mode else SIMILARITY_THRESHOLD
        )
    # The QED floor applied at the ADMET step.
    qed_floor = RECALL_QED_FLOOR if recal_mode else 0.7

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

    def _filter_pass(threshold: float, qed_gate: float) -> "List[CompoundRecord]":
        """Run the similarity + ADMET + PAINS filter chain on the original records."""
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

            # In recall mode the QED floor is relaxed (0.7 → 0.4) so known
            # binders with lower drug-likeness still pass (paper §4.4). The
            # diversity fallback below also lowers this gate when too few
            # compounds survive the default 0.7 cut.
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
        return passed

    passed = _filter_pass(similarity_threshold, qed_floor)

    # Diversity check — if too few passed, relax BOTH the similarity threshold
    # and the QED floor, then re-run the same loop on the original records
    # (simple for-loop, no recursion). The QED gate is the dominant filter for
    # de-novo BRICS libraries, so relaxing similarity alone is insufficient to
    # reach the ≥5 compounds needed for active-site consensus docking.
    if len(passed) < DIVERSITY_MIN_COUNT:
        relaxed_qed = min(qed_floor, RECALL_QED_FLOOR)
        log.info(
            f"  Only {len(passed)} compounds passed filters (< {DIVERSITY_MIN_COUNT}). "
            f"Relaxing similarity threshold to {SIMILARITY_THRESHOLD_RELAXED} and "
            f"QED floor to {relaxed_qed}, then re-filtering."
        )
        passed = _filter_pass(SIMILARITY_THRESHOLD_RELAXED, relaxed_qed)

    log.info("─── Phase 2 complete ───")
    return passed


# ── Phase 3.5: Negative selection (human off-target clash) ───────────────────
#
# Human off-target attribute names whose energies are inspected, plus the
# critical cardiotoxicity / drug-metabolism targets that trigger immediate
# removal when bound tightly. Energies are Vina kcal/mol (negative = binding).
_HUMAN_OFFTARGET_ATTRS = (
    "human_trypsin_energy",
    "human_ces1_energy",
    "human_albumin_energy",
    "human_cyp3a4_energy",
    "human_herg_energy",
    "human_cyp2d6_energy",
)

# Targets whose tight binding is an absolute deal-breaker (regardless of how
# well the compound binds the bacterial target). HERG → cardiotoxicity;
# CYP3A4 → drug–drug metabolism liability.
_HARD_CLASH_TARGETS = {
    "human_herg_energy": "HERG (cardiotoxicity)",
    "human_cyp3a4_energy": "CYP3A4 (drug metabolism)",
}

# Energy below which a compound is considered to *tightly* bind a human
# off-target (kcal/mol). Stronger binding than this ⇒ immediate discard.
_HARD_CLASH_ENERGY = -8.0

# Energy above which the compound is considered to ignore the human off-target
# (essentially no binding). Used only for reporting the max off-target energy.
_WEAK_BIND_ENERGY = -5.0


def filter_by_human_clash(
    records: "List[CompoundRecord]",
    hard_clash_energy: float = _HARD_CLASH_ENERGY,
    hard_clash_targets: Optional[dict] = None,
) -> "List[CompoundRecord]":
    """
    Phase 3.5 — Negative selection against human off-targets.

    The standard Selectivity Index (``|E_bacteria| / |E_human|``) *penalises*
    strong bacterial binders whenever they also bind human proteins weakly.
    Negative selection is stricter: we want compounds that *ignore* human
    proteins, not merely ones that bind bacteria slightly better.

    Rule:
        1. For every human off-target, collect the (valid, binding) energy.
        2. If the compound binds HERG **or** CYP3A4 with energy
           ``< hard_clash_energy`` (default -8.0 kcal/mol), it is DISCARDED
           immediately — no matter how strong its bacterial affinity. It is
           removed from the candidate list, not just flagged.
        3. Otherwise the compound is kept. The maximum (least negative) human
           off-target energy is recorded on ``rec.human_offtarget_max_energy``
           so the report can surface how strongly the compound engages humans.

    Args:
        records: Candidate records (with human off-target energies populated).
        hard_clash_energy: Discard threshold for HERG/CYP3A4 (kcal/mol).
        hard_clash_targets: Mapping ``{attr: label}`` of deal-breaker targets.
            Defaults to HERG and CYP3A4.

    Returns:
        Filtered list with hard-clash compounds removed. A list of discarded
        ``compound_id`` strings is logged for traceability.
    """
    if hard_clash_targets is None:
        hard_clash_targets = _HARD_CLASH_TARGETS

    kept = []
    discarded = []
    for rec in records:
        # Collect all valid human off-target energies (finite + binding).
        energies = [
            e for e in (
                getattr(rec, attr, None) for attr in _HUMAN_OFFTARGET_ATTRS
            )
            if e is not None and np.isfinite(e) and e < 0.0
        ]
        # Record the strongest (most negative) human engagement for reporting.
        rec.human_offtarget_max_energy = max(energies) if energies else None

        # Hard clash: tight binding to a deal-breaker off-target ⇒ discard.
        hard_clash = False
        for attr, label in hard_clash_targets.items():
            e = getattr(rec, attr, None)
            if e is not None and np.isfinite(e) and e < hard_clash_energy:
                hard_clash = True
                log.info(
                    f"  Negative selection: discarding {rec.compound_id} — "
                    f"tight binding to {label} (E = {e:.2f} kcal/mol < "
                    f"{hard_clash_energy:.2f})."
                )
                break

        if hard_clash:
            discarded.append(rec.compound_id)
            continue
        kept.append(rec)

    if discarded:
        log.info(
            f"  Negative selection removed {len(discarded)} compound(s) for "
            f"human off-target clash: {', '.join(discarded)}"
        )
    log.info(
        f"  Negative selection kept {len(kept)} / {len(records)} compound(s)."
    )
    return kept
