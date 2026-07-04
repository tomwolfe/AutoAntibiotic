from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    BRICS,
    Crippen,
    Descriptors,
    QED,
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.DataStructs import TanimotoSimilarity

from .config import CONFIG, CompoundRecord
from .io_utils import log

try:
    from tqdm import tqdm as _tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False
    _tqdm = lambda x, **kw: x

try:
    from sascore import compute_sa_score as _compute_sa_score
    _HAVE_SA_SCORE = True
except ImportError:
    _compute_sa_score = None
    _HAVE_SA_SCORE = False


def _count_atoms(mol: Chem.Mol) -> int:
    """Heavy-atom count for a molecule."""
    return mol.GetNumHeavyAtoms()


def _validate_mol(smiles: str) -> Optional[Chem.Mol]:
    """Validate a SMILES string by parsing and sanitising."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol


def _brics_recombination(
    frag_mols: List[Chem.Mol],
    target_count: int,
    seen_smiles: set,
    seed: int = CONFIG.random_seed,
) -> Tuple[List[CompoundRecord], set]:
    """Recombine BRICS fragments using BRICSBuild, then pick a diverse subset via MaxMin."""
    rng = np.random.default_rng(seed)

    pool_mult = CONFIG.diversity_pool_multiplier
    max_products = target_count * pool_mult * 4
    n_produced = 0

    shuffled = list(frag_mols)
    rng.shuffle(shuffled)

    builder = BRICS.BRICSBuild(shuffled)

    pool_records: List[CompoundRecord] = []
    target_pool = target_count * pool_mult
    iterator = _tqdm(
        itertools.islice(builder, max_products),
        desc="  BRICS recombination",
        total=min(max_products, target_pool * 4),
        disable=not _HAVE_TQDM,
    )
    for product in iterator:
        if product is None:
            continue
        try:
            Chem.SanitizeMol(product)
            smi = Chem.MolToSmiles(product)
        except Exception:
            continue

        if smi in seen_smiles:
            continue

        ring_info = product.GetRingInfo()
        if ring_info.NumRings() == 0:
            continue

        seen_smiles.add(smi)

        pool_records.append(CompoundRecord(
            compound_id=f"AA-{n_produced:04d}",
            smiles=smi,
            mol=product,
        ))
        n_produced += 1

        if n_produced >= target_pool:
            break

        if n_produced % 100 == 0 and not _HAVE_TQDM:
            log.info(f"  BRICS pool: {n_produced} / {target_pool}…")

    if not pool_records:
        return [], seen_smiles

    log.info(f"  BRICS pool size: {len(pool_records)}")

    if len(pool_records) <= target_count:
        return pool_records, seen_smiles

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            r.mol, radius=CONFIG.morgan_radius, nBits=CONFIG.morgan_nbits,
        )
        for r in pool_records
    ]

    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
    picker = MaxMinPicker()
    pick_ids = picker.LazyBitVectorPick(
        fps, len(fps), target_count, seed=seed,
    )

    records = [pool_records[i] for i in pick_ids]
    log.info(
        f"  MaxMin selected {len(records)} diverse compounds "
        f"from pool of {len(pool_records)}."
    )
    return records, seen_smiles


def generate_candidate_library(
    target_count: int = CONFIG.library_target_count,
    seed: int = CONFIG.random_seed,
) -> List[CompoundRecord]:
    """Phase 2.1 — Library Generation via BRICS fragment recombination.

    Returns a list of CompoundRecord objects with compound_id, smiles, and mol populated.
    """
    log.info("─── Phase 2: Library Generation ───")

    all_scaffolds: List[str] = CONFIG.natural_product_scaffolds + CONFIG.additional_scaffolds
    scaffold_mols: List[Chem.Mol] = []
    for smi in all_scaffolds:
        mol = _validate_mol(smi)
        if mol is not None:
            scaffold_mols.append(mol)

    log.info(f"  Loaded {len(scaffold_mols)} / {len(all_scaffolds)} valid scaffolds.")

    if not scaffold_mols and not CONFIG.brics_building_blocks:
        log.error("  ✗  No valid scaffolds or building blocks. Aborting library generation.")
        return []

    decomposed_frags: set = set()
    for mol in scaffold_mols:
        try:
            fragments = BRICS.BRICSDecompose(mol, minFragmentSize=CONFIG.brics_min_fragment_size)
            for frag_smi in fragments:
                frag_mol = _validate_mol(frag_smi)
                if frag_mol is not None and _count_atoms(frag_mol) >= CONFIG.brics_min_fragment_size:
                    decomposed_frags.add(frag_smi)
        except Exception:
            continue

    log.info(f"  Decomposed {len(decomposed_frags)} unique BRICS fragments from scaffolds.")

    all_building_blocks: set = set()
    for smi in CONFIG.brics_building_blocks:
        mol = _validate_mol(smi)
        if mol is not None:
            all_building_blocks.add(smi)

    log.info(f"  Loaded {len(all_building_blocks)} pre-built BRICS building blocks.")

    all_frag_smis: set = decomposed_frags | all_building_blocks
    frag_mols: List[Chem.Mol] = []
    for smi in all_frag_smis:
        m = _validate_mol(smi)
        if m is not None:
            frag_mols.append(m)

    log.info(f"  Total BRICS-compatible fragments: {len(frag_mols)}")

    seen_smiles: set = set()
    records: List[CompoundRecord] = []

    for smi in all_scaffolds:
        mol = _validate_mol(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen_smiles:
            continue
        seen_smiles.add(canon)
        records.append(CompoundRecord(
            compound_id=f"SCAFFOLD_{len(records):04d}",
            smiles=canon,
            mol=mol,
        ))

    if len(frag_mols) >= 2:
        recon_records, seen_smiles = _brics_recombination(
            frag_mols, target_count, seen_smiles, seed,
        )
        records.extend(recon_records)
        log.info(f"  BRICS recombination yielded {len(recon_records)} novel compounds.")
    else:
        log.warning(
            f"  Too few fragments ({len(frag_mols)}) for recombination. "
            "Using scaffold enumeration only."
        )

    for name, smi in CONFIG.control_smiles.items():
        mol = _validate_mol(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen_smiles:
            records.append(CompoundRecord(
                compound_id=f"CTRL_{name}",
                smiles=canon,
                mol=mol,
            ))
            seen_smiles.add(canon)

    log.info(f"  Library generation complete: {len(records)} compounds.")
    if len(records) < 300:
        log.warning(
            f"  ⚠  Only {len(records)} compounds generated (target ≥300). "
            "Consider adding more scaffolds or building blocks."
        )

    return records


def apply_filters(
    records: List[CompoundRecord],
    similarity_threshold: float = CONFIG.similarity_threshold,
) -> List[CompoundRecord]:
    """Phase 2.2 — Apply structural, similarity, ADMET, and PAINS filters.

    Filter chain:
        1. Structural exclusion (β-lactam SMARTS).
        2. Similarity filter vs reference antibiotics.
        3. ADMET: Lipinski Rule of 5 + QED > 0.6.
        4. PAINS alerts via RDKit FilterCatalog.
        5. Diversity check: if < 100 pass, relax similarity to 0.5.

    Returns filtered list of CompoundRecord.
    """
    log.info("─── Phase 2: Filtering ───")

    ref_mols: Dict[str, Any] = {}
    for name, smi in CONFIG.reference_antibiotics.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ref_mols[name] = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=CONFIG.morgan_radius, nBits=CONFIG.morgan_nbits,
            )

    lactam_pattern = Chem.MolFromSmarts(CONFIG.beta_lactam_smarts)

    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains_catalog = FilterCatalog(pains_params)

    passed: List[CompoundRecord] = []
    skipped_structural = 0
    skipped_similarity = 0
    skipped_admet = 0
    skipped_pains = 0
    skipped_sa_score = 0

    for record in records:
        if record.mol is None:
            mol = Chem.MolFromSmiles(record.smiles)
            if mol is None:
                continue
            record.mol = mol
        mol = record.mol

        is_control = record.compound_id.startswith("CTRL_")
        if not is_control and mol.HasSubstructMatch(lactam_pattern):
            skipped_structural += 1
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=CONFIG.morgan_radius, nBits=CONFIG.morgan_nbits)
        max_sim = 0.0
        for ref_fp in ref_mols.values():
            sim = TanimotoSimilarity(fp, ref_fp)
            max_sim = max(max_sim, sim)
        record.max_similarity = max_sim

        if max_sim >= similarity_threshold:
            skipped_similarity += 1
            continue

        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            lipinski_ok = (
                mw <= CONFIG.lipinski_mw_max
                and logp <= CONFIG.lipinski_logp_max
                and hbd <= CONFIG.lipinski_hbd_max
                and hba <= CONFIG.lipinski_hba_max
            )
            qed = QED.qed(mol)
        except Exception:
            continue

        record.passes_lipinski = lipinski_ok
        record.qed_score = qed

        if not lipinski_ok or qed <= CONFIG.qed_threshold:
            skipped_admet += 1
            continue

        pains_match = pains_catalog.HasMatch(mol)
        record.passes_pains = not pains_match
        if pains_match:
            skipped_pains += 1
            continue

        if _HAVE_SA_SCORE:
            try:
                sa_score = _compute_sa_score(mol)
                if sa_score > CONFIG.sa_score_threshold:
                    skipped_sa_score += 1
                    continue
            except Exception:
                pass

        passed.append(record)

    log.info(f"  Structural exclusion (β-lactam): {skipped_structural} removed.")
    log.info(f"  Similarity filter (Tc < {similarity_threshold}): {skipped_similarity} removed.")
    log.info(f"  ADMET filter (Lipinski + QED > 0.6): {skipped_admet} removed.")
    log.info(f"  PAINS filter: {skipped_pains} removed.")
    if _HAVE_SA_SCORE:
        log.info(f"  SA Score filter (> {CONFIG.sa_score_threshold}): {skipped_sa_score} removed.")
    else:
        log.info("  SA Score filter: skipped (sascore not installed).")
    log.info(f"  Passed filters: {len(passed)} compounds.")

    if len(passed) < CONFIG.diversity_min_count and similarity_threshold < CONFIG.similarity_threshold_relaxed:
        log.warning(
            f"  Only {len(passed)} compounds passed strict filters (< {CONFIG.diversity_min_count}). "
            f"Relaxing similarity threshold to {CONFIG.similarity_threshold_relaxed} and re-running."
        )
        return apply_filters(records, similarity_threshold=CONFIG.similarity_threshold_relaxed)

    log.info("─── Phase 2 complete ───")
    return passed
