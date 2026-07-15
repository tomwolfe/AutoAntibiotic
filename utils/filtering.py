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


def apply_filters(
    records: "List[CompoundRecord]",
    similarity_threshold: Optional[float] = None,
) -> "List[CompoundRecord]":
    """
    Phase 2.2 — Apply structural, similarity, ADMET, and PAINS filters.

    Filter chain:
        1. Structural exclusion (β-lactam SMARTS).
        2. Similarity filter vs reference antibiotics (Morgan FP, Tc < threshold).
            3. ADMET: Lipinski Rule of 5 + QED > 0.7.
        4. PAINS alerts via RDKit FilterCatalog.
        5. Diversity check: if < 100 pass, relax similarity to 0.5.

    Args:
        records: Input compound records.
        similarity_threshold: Initial Tanimoto cutoff.

    Returns:
        Filtered list of CompoundRecord (with computed ADMET/similarity fields).
    """
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD

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

    def _filter_pass(threshold: float) -> "List[CompoundRecord]":
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

            if not lipinski_ok:
                skipped_admet += 1
                continue
            if qed <= 0.7:
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
        log.info(f"  ADMET filter (Lipinski + QED > 0.7): {skipped_admet} removed.")
        log.info(f"  PAINS filter: {skipped_pains} removed.")
        log.info(f"  Brenk alerts: {skipped_brenk} removed.")
        log.info(f"  Passed filters: {len(passed)} compounds.")
        return passed

    passed = _filter_pass(similarity_threshold)

    # Diversity check — if too few passed, relax the similarity threshold and
    # re-run the same loop on the original records (simple for-loop, no recursion).
    if len(passed) < DIVERSITY_MIN_COUNT:
        log.info(
            f"  Only {len(passed)} compounds passed filters (< {DIVERSITY_MIN_COUNT}). "
            f"Relaxing similarity threshold to {SIMILARITY_THRESHOLD_RELAXED} and re-filtering."
        )
        passed = _filter_pass(SIMILARITY_THRESHOLD_RELAXED)

    log.info("─── Phase 2 complete ───")
    return passed
