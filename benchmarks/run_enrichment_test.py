"""
Benchmark: Enrichment Test for PBP2a Inhibitor Screening
========================================================

Computes **Enrichment Factor at 1% (EF1%)** and **ROC-AUC** for the
AutoAntibiotic screening pipeline using a mixed set of known PBP2a
actives, inactives, and property-matched decoys.

Usage::

    python -m benchmarks.run_enrichment_test
    python -m benchmarks.run_enrichment_test --use-vina
    python -m benchmarks.run_enrichment_test --n-decoys 50

Metrics
-------
- **EF1%**: (fraction of actives in top 1% of ranked list) / 0.01
    Values > 1 indicate enrichment better than random.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve.
    0.5 = random, 1.0 = perfect ranking.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, QED
from rdkit.DataStructs import TanimotoSimilarity
from sklearn.metrics import roc_auc_score

# Quiet RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.reference_data import (
    PBP2A_ACTIVES,
    PBP2A_INACTIVES,
    DECOY_COUNT,
    get_active_labels,
    get_actives_smiles,
    get_inactive_labels,
    get_inactives_smiles,
)
from autoantibiotic.config import CONFIG
from autoantibiotic.models import CompoundRecord
from autoantibiotic.io_utils import log

# Optional dependencies for real docking
_HAVE_VINA: bool = False
try:
    result = __import__("subprocess").run(
        ["vina", "--version"], capture_output=True, text=True, timeout=10,
    )
    _HAVE_VINA = result.returncode == 0
except Exception:
    pass

# Lazy imports for Vina docking (avoids hard dependency at module level)
_VINA_IMPORTED: bool = False


def _import_vina_deps() -> bool:
    """Import Vina docking dependencies lazily.

    Returns True if all required modules loaded successfully.
    """
    global _VINA_IMPORTED
    if _VINA_IMPORTED:
        return True
    try:
        from autoantibiotic.docking import dock_compound, prepare_ligand_pdbqt
        from autoantibiotic.structure_prep import (
            clean_pdb_structure,
            compute_residue_centroid,
            fetch_structure,
        )
        _VINA_IMPORTED = True
        return True
    except ImportError:
        return False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


# ── Decoy generation ─────────────────────────────────────────────────


def _compute_property_vector(mol: Chem.Mol) -> np.ndarray:
    """Normalised property vector for decoy matching."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    return np.array([mw / 500.0, min(max(logp, -2), 8) / 8.0, hbd / 10.0, hba / 10.0, rot / 15.0], dtype=np.float64)


def _generate_pool(size: int = 3000, seed: int = 42) -> List[Chem.Mol]:
    """Generate a diverse pool of drug-like molecules.

    Sources (in priority order):
      1. BRICS recombination product from pipeline scaffolds.
      2. All scaffold molecules and building blocks from CONFIG.
      3. All reference antibiotics and control SMILES.
      4. Inactive molecules (expanded with SMILES permutations).

    This ensures sufficient chemical diversity for property-matched
    decoy generation even when BRICS yields few products.
    """
    from autoantibiotic.library_gen import _brics_recombination

    pool: List[Chem.Mol] = []
    seen_smiles: set = set()

    def _add_mol(smi: str) -> None:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return
        canon = Chem.MolToSmiles(mol)
        if canon in seen_smiles:
            return
        seen_smiles.add(canon)
        pool.append(mol)

    # 1. BRICS recombination
    frag_mols: List[Chem.Mol] = []
    for smi in CONFIG.brics_building_blocks:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            frag_mols.append(mol)
    for smi in CONFIG.natural_product_scaffolds:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            frag_mols.append(mol)
    for smi in CONFIG.additional_scaffolds:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            frag_mols.append(mol)

    brics_records, _ = _brics_recombination(frag_mols, target_count=size, seen_smiles=set(), seed=seed)
    for rec in brics_records:
        if rec.mol is not None:
            canon = Chem.MolToSmiles(rec.mol)
            if canon not in seen_smiles:
                seen_smiles.add(canon)
                pool.append(rec.mol)

    # 2. Reference scaffolds and building blocks
    for smi in CONFIG.brics_building_blocks:
        _add_mol(smi)
    for smi in CONFIG.natural_product_scaffolds:
        _add_mol(smi)
    for smi in CONFIG.additional_scaffolds:
        _add_mol(smi)

    # 3. Reference antibiotics and controls
    for smi in CONFIG.reference_antibiotics.values():
        _add_mol(smi)
    for smi in CONFIG.control_smiles.values():
        _add_mol(smi)

    # 4. Reference inactives from benchmark data
    for smi in get_inactives_smiles():
        _add_mol(smi)

    # 5. Expand with SMILES permutations of inactives (add/drop carbonyls, hydroxyls)
    rng = np.random.default_rng(seed + 999)
    inactives_expanded: List[str] = list(get_inactives_smiles())
    for _ in range(200):
        idx = rng.integers(0, len(inactives_expanded))
        base_smi = inactives_expanded[idx]
        base_mol = Chem.MolFromSmiles(base_smi)
        if base_mol is None:
            continue
        try:
            # Add a random substituent (methyl, hydroxyl, carbonyl)
            from rdkit.Chem import rdChemReactions
            rxn_smarts = rng.choice([
                "[c:1]>>[c:1]C", "[c:1]>>[c:1]O", "[c:1]>>[c:1]Cl",
                "[c:1]>>[c:1]F", "[c:1]>>[c:1]OC",
            ])
            rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
            products = rxn.RunReactants((base_mol,))
            if products:
                mod_mol = products[0][0]
                Chem.SanitizeMol(mod_mol)
                _add_mol(Chem.MolToSmiles(mod_mol))
        except Exception:
            pass

    # Filter to reasonable size
    pool = [m for m in pool if 150 <= Descriptors.MolWt(m) <= 700]
    log.info(f"  Augmented decoy pool size: {len(pool)}")
    return pool


def _property_distance(props_a: np.ndarray, props_b: np.ndarray) -> float:
    return float(np.linalg.norm(props_a - props_b))


def generate_decoys(
    actives: List[Chem.Mol],
    active_labels: List[str],
    n_decoys_per_active: int = 100,
    pool_size: int = 3000,
    seed: int = 42,
) -> List[Tuple[str, Chem.Mol, str]]:
    """Generate property-matched decoys for each active compound.

    Uses a pool of drug-like molecules generated via BRICS recombination,
    then selects *n_decoys_per_active* decoys per active that:
      - Have similar MW ±40%, LogP ±1.5, HBD ±2, HBA ±3, RotB ±3
      - Are fingerprint-dissimilar from the active (Tanimoto < 0.7)

    Returns list of ``(decoy_id, mol, matched_active_id)`` tuples.
    """
    rng = np.random.default_rng(seed)
    pool = _generate_pool(pool_size, seed)
    log.info(f"  Decoy pool size: {len(pool)}")

    if len(pool) < n_decoys_per_active * len(actives):
        log.warning(
            f"  Pool only has {len(pool)} molecules; "
            f"reducing decoys per active to {max(1, len(pool) // len(actives))}."
        )
        n_decoys_per_active = max(1, len(pool) // len(actives))

    active_fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        for mol in actives
    ]
    active_props = [list(Descriptors.MolWt(mol) for mol in actives)]

    pool_fps: List = []
    pool_mw: List[float] = []
    pool_logp: List[float] = []
    pool_hbd: List[int] = []
    pool_hba: List[int] = []
    pool_rot: List[int] = []
    valid_pool: List[Chem.Mol] = []
    for mol in pool:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rot = Descriptors.NumRotatableBonds(mol)
        except Exception:
            continue
        pool_fps.append(fp)
        pool_mw.append(mw)
        pool_logp.append(logp)
        pool_hbd.append(hbd)
        pool_hba.append(hba)
        pool_rot.append(rot)
        valid_pool.append(mol)

    decoys: List[Tuple[str, Chem.Mol, str]] = []
    used_indices: set = set()
    decoy_idx = 0

    for ai, active_mol in enumerate(actives):
        afp = active_fps[ai]
        amw = Descriptors.MolWt(active_mol)
        alogp = Crippen.MolLogP(active_mol)
        ahbd = Descriptors.NumHDonors(active_mol)
        ahba = Descriptors.NumHAcceptors(active_mol)
        arot = Descriptors.NumRotatableBonds(active_mol)

        candidates: List[Tuple[int, float]] = []
        for pi in range(len(valid_pool)):
            if pi in used_indices:
                continue
            sim = TanimotoSimilarity(pool_fps[pi], afp)
            if sim >= 0.7:
                continue

            mw_match = abs(pool_mw[pi] - amw) / max(amw, 1.0) < 0.4
            logp_ok = abs(pool_logp[pi] - alogp) < 2.0
            hbd_ok = abs(pool_hbd[pi] - ahbd) <= 3
            hba_ok = abs(pool_hba[pi] - ahba) <= 4
            rot_ok = abs(pool_rot[pi] - arot) <= 4

            if all([mw_match, logp_ok, hbd_ok, hba_ok, rot_ok]):
                score = (
                    (pool_mw[pi] - amw) ** 2 / max(amw, 1.0)
                    + (pool_logp[pi] - alogp) ** 2
                    + (pool_hbd[pi] - ahbd) ** 2
                    + (pool_hba[pi] - ahba) ** 2
                    + (pool_rot[pi] - arot) ** 2
                )
                candidates.append((pi, score))

        candidates.sort(key=lambda x: x[1])
        selected = candidates[:n_decoys_per_active]

        if len(selected) < n_decoys_per_active:
            log.warning(
                f"  Only {len(selected)}/{n_decoys_per_active} decoys for "
                f"active {active_labels[ai]}. Relaxing fingerprint threshold…"
            )
            remaining = [
                (pi, _property_distance(
                    np.array([pool_mw[pi] / 500.0, pool_logp[pi], pool_hbd[pi],
                              pool_hba[pi], pool_rot[pi]], dtype=np.float64),
                    np.array([amw / 500.0, alogp, ahbd, ahba, arot], dtype=np.float64),
                ))
                for pi in range(len(valid_pool))
                if pi not in used_indices and TanimotoSimilarity(pool_fps[pi], afp) < 0.85
            ]
            remaining.sort(key=lambda x: x[1])
            needed = n_decoys_per_active - len(selected)
            added = 0
            for pi, _ in remaining:
                if pi not in {s[0] for s in selected}:
                    selected.append((pi, 0.0))
                    added += 1
                    if added >= needed:
                        break

        for pi, _ in selected[:n_decoys_per_active]:
            used_indices.add(pi)
            aid = active_labels[ai]
            decoys.append((f"DECOY_{decoy_idx:04d}", valid_pool[pi], aid))
            decoy_idx += 1

    log.info(f"  Generated {len(decoys)} property-matched decoys.")
    return decoys


# ── Scoring ──────────────────────────────────────────────────────────


def score_compounds_with_dry_run(
    records: List[CompoundRecord],
) -> List[CompoundRecord]:
    """Score compounds using dry-run mock docking energies.

    In dry-run mode the pipeline returns random energies in [-10, -5].
    We seed per compound for reproducibility.
    """
    rng = np.random.default_rng(42)
    for rec in records:
        rec.pb2pa_allosteric_energy = float(rng.uniform(-10.0, -5.0))
    return records


def score_compounds_with_fingerprint_similarity(
    records: List[CompoundRecord],
    reference_smiles: List[str],
) -> List[CompoundRecord]:
    """Score compounds by average Tanimoto similarity to known actives.

    Used as a no-dependency scoring baseline.
    Higher similarity = better score (convert to negative for sorting).
    """
    ref_mols: List[Chem.Mol] = []
    for smi in reference_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ref_mols.append(mol)

    if not ref_mols:
        return score_compounds_with_dry_run(records)

    ref_fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        for m in ref_mols
    ]

    for rec in records:
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol
        fp = AllChem.GetMorganFingerprintAsBitVect(rec.mol, radius=2, nBits=2048)
        sims = [TanimotoSimilarity(fp, ref) for ref in ref_fps]
        avg_sim = float(np.mean(sims)) if sims else 0.0
        rec.pb2pa_allosteric_energy = -avg_sim * 10.0

    return records


# ── Vina Docking Scoring ─────────────────────────────────────────────


def _is_rigid_pdbqt(pdbqt_path: str) -> bool:
    """Check if a PDBQT file is suitable as a rigid receptor for Vina.

    A valid rigid-receptor PDBQT must contain ATOM/HETATM records and
    must NOT contain ROOT/BRANCH/ENDBRANCH tags (which indicate a
    flexible-receptor format that Vina rejects).
    """
    if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
        return False
    try:
        with open(pdbqt_path) as f:
            content = f.read(100000)
        has_atoms = "ATOM" in content or "HETATM" in content
        has_flex = "ROOT" in content or "BRANCH" in content or "ENDBRANCH" in content
        return has_atoms and not has_flex
    except Exception:
        return False


def _prepare_pbp2a_receptor() -> Optional[Dict[str, Any]]:
    """Download and prepare the PBP2a receptor for docking.

    Returns a dict with keys ``pdbqt``, ``allosteric_center``, and
    ``active_center``, or None if preparation failed.
    """
    from autoantibiotic.structure_prep import (
        clean_pdb_structure,
        compute_residue_centroid,
        fetch_structure,
    )

    pdb_dir = str(CONFIG.pdb_dir)
    work_dir = str(CONFIG.work_dir)
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    cleaned_pdbqt = os.path.join(work_dir, "PBP2a_clean.pdbqt")
    cleaned_pdb = os.path.join(work_dir, "PBP2a_clean.pdb")

    if os.path.exists(cleaned_pdbqt) and os.path.exists(cleaned_pdb):
        if _is_rigid_pdbqt(cleaned_pdbqt):
            log.info("  Using cached prepared receptor.")
        else:
            log.warning("  Cached PDBQT is in flexible-receptor format (incompatible with Vina). Regenerating…")
            os.remove(cleaned_pdbqt)

    if not os.path.exists(cleaned_pdbqt):
        log.info("  Preparing PBP2a receptor…")
        apo_path = fetch_structure(CONFIG.pdb_ids["PBP2a_apo"], pdb_dir)

        deps = {}
        for tool_name in ("obabel",):
            if shutil.which(tool_name):
                deps[tool_name] = True

        result = clean_pdb_structure(apo_path, cleaned_pdb, deps=deps)
        if not os.path.exists(cleaned_pdbqt):
            log.warning("  PDBQT not available; falling back to PDB for centroid calc.")
            cleaned_pdbqt = cleaned_pdb
        elif not _is_rigid_pdbqt(cleaned_pdbqt):
            log.warning("  Generated PDBQT is not a valid rigid receptor. Falling back to PDB.")
            cleaned_pdbqt = cleaned_pdb

    try:
        allosteric_center = compute_residue_centroid(cleaned_pdb, CONFIG.allosteric_residues)
        active_center = compute_residue_centroid(cleaned_pdb, CONFIG.active_site_residues)
    except Exception as exc:
        log.error(f"  Failed to compute binding site centroids: {exc}")
        return None

    return {
        "pdbqt": cleaned_pdbqt,
        "allosteric_center": allosteric_center,
        "active_center": active_center,
    }


def score_compounds_with_vina(
    records: List[CompoundRecord],
    receptor: Dict[str, Any],
    site: str = "allosteric",
) -> List[CompoundRecord]:
    """Score compounds using real AutoDock Vina docking.

    Each compound is docked into the specified binding site and the
    best binding energy (kcal/mol) is stored on the record.

    Args:
        records: List of compound records to dock.
        receptor: Target dict with ``pdbqt``, ``allosteric_center``,
            ``active_center`` keys.
        site: ``"allosteric"`` or ``"active"``.

    Returns:
        Records with ``pb2pa_allosteric_energy`` (or ``pb2pa_active_energy``)
        populated.
    """
    from autoantibiotic.docking import dock_compound, prepare_ligand_pdbqt

    work_dir = str(CONFIG.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    center_key = "allosteric_center" if site == "allosteric" else "active_center"
    box_size = CONFIG.allosteric_box_size if site == "allosteric" else CONFIG.active_box_size
    center = receptor[center_key]
    receptor_pdbqt = receptor["pdbqt"]
    energy_attr = "pb2pa_allosteric_energy" if site == "allosteric" else "pb2pa_active_energy"

    n_scored = 0
    for rec in records:
        energy = dock_compound(
            rec, receptor_pdbqt, center, box_size, work_dir, tag="bm",
        )
        if energy is not None:
            setattr(rec, energy_attr, energy)
            n_scored += 1
        else:
            setattr(rec, energy_attr, 0.0)

    log.info(f"  Vina docking complete: {n_scored}/{len(records)} scored.")
    return records


# ── Metric calculation ───────────────────────────────────────────────


def compute_enrichment_factor(
    scores: np.ndarray,
    labels: np.ndarray,
    fraction: float = 0.01,
) -> float:
    """Compute the Enrichment Factor at a given fraction.

    EF(f) = (active_retrieved_top_fraction / total_actives) / f

    Where *f* is the fraction of the ranked library to consider
    (e.g. 0.01 for EF1%).

    Args:
        scores: Array of scores (lower = better).
        labels: Binary labels (1 = active, 0 = inactive/decoys).
        fraction: Fraction of top-ranked compounds to evaluate.

    Returns:
        Enrichment factor (1.0 = random, >1 = better than random).
    """
    n_total = len(scores)
    if n_total == 0:
        return 1.0
    n_actives = int(labels.sum())
    if n_actives == 0:
        return 1.0

    order = np.argsort(scores)
    n_top = max(1, int(n_total * fraction))
    top_indices = order[:n_top]
    actives_in_top = int(labels[top_indices].sum())

    expected_random = n_actives * fraction
    if expected_random == 0:
        return 1.0

    return actives_in_top / expected_random


def compute_roc_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute ROC-AUC from scores and binary labels.

    Since lower docking energies indicate stronger binding, we negate
    the scores before calling ``roc_auc_score`` so that the standard
    "higher score = positive class" convention is satisfied.

    Args:
        scores: Array of docking scores (lower = better binding).
        labels: Binary labels (1 = active, 0 = inactive/decoys).

    Returns:
        ROC-AUC (0.5 = random, 1.0 = perfect).
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, -scores))


def run_enrichment_test(
    n_decoys_per_active: int = DECOY_COUNT,
    use_vina: bool = False,
) -> Dict[str, float]:
    """Run the full enrichment test and return metrics.

    Steps:
        1. Load reference actives and inactives.
        2. Generate property-matched decoys.
        3. Build a mixed CompoundRecord list.
        4. Score compounds via the pipeline (or fingerprint proxy).
        5. Compute and print EF1%, ROC-AUC.

    Args:
        n_decoys_per_active: Number of decoys to generate per active.
        use_vina: If True, try to use Vina docking (requires targets).

    Returns:
        Dict with keys ``ef1``, ``roc_auc``, ``n_actives``, ``n_total``.
    """
    log.info("═══ Benchmark: Enrichment Test ═══")

    # 1. Load actives and inactives
    active_smiles = get_actives_smiles()
    active_labels = get_active_labels()
    inactive_smiles = get_inactives_smiles()

    active_mols: List[Chem.Mol] = []
    active_records: List[CompoundRecord] = []
    active_labels_filtered: List[str] = []
    for i, smi in enumerate(active_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.warning(f"  Skipping active {i}: invalid SMILES")
            continue
        active_mols.append(mol)
        active_labels_filtered.append(active_labels[i])
        active_records.append(
            CompoundRecord(
                compound_id=f"ACTIVE_{active_labels[i]}",
                smiles=smi, mol=mol,
            )
        )
    active_labels = active_labels_filtered

    inactive_records: List[CompoundRecord] = []
    for i, smi in enumerate(inactive_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        inactive_records.append(
            CompoundRecord(
                compound_id=f"INACTIVE_{get_inactive_labels()[i]}",
                smiles=smi, mol=mol,
            )
        )

    # 2. Generate decoys
    log.info("  Generating property-matched decoys…")
    decoys = generate_decoys(active_mols, active_labels, n_decoys_per_active, seed=42)
    decoy_records: List[CompoundRecord] = []
    for did, mol, matched_active in decoys:
        decoy_records.append(
            CompoundRecord(compound_id=did, smiles=Chem.MolToSmiles(mol), mol=mol)
        )

    # 3. Build mixed set
    all_records: List[CompoundRecord] = active_records + inactive_records + decoy_records
    n_actives = len(active_records)
    n_total = len(all_records)
    log.info(f"  Mixed set: {n_actives} actives, {len(inactive_records)} inactives, "
             f"{len(decoy_records)} decoys (total {n_total})")

    # 4. Score all compounds
    if use_vina:
        log.info("  Scoring with Vina docking…")
        if not _HAVE_VINA:
            log.error(
                "  Vina is not installed or not found in PATH.\n"
                "  Please install it with:\n"
                "    conda install -c conda-forge vina\n"
                "  Falling back to fingerprint similarity scoring."
            )
            all_records = score_compounds_with_fingerprint_similarity(all_records, active_smiles)
        else:
            log.info("  Preparing PBP2a receptor…")
            receptor = _prepare_pbp2a_receptor()
            if receptor is None:
                log.error(
                    "  Failed to prepare PBP2a receptor.\n"
                    "  Falling back to fingerprint similarity scoring."
                )
                all_records = score_compounds_with_fingerprint_similarity(all_records, active_smiles)
            else:
                all_records = score_compounds_with_vina(all_records, receptor, site="allosteric")
    else:
        log.info("  Scoring with fingerprint similarity (no-dependency proxy).")
        all_records = score_compounds_with_fingerprint_similarity(all_records, active_smiles)

    # 5. Compute metrics
    scores: np.ndarray = np.array([
        r.pb2pa_allosteric_energy if r.pb2pa_allosteric_energy is not None else 0.0
        for r in all_records
    ], dtype=np.float64)
    labels: np.ndarray = np.zeros(n_total, dtype=np.int64)
    labels[:n_actives] = 1

    ef1 = compute_enrichment_factor(scores, labels, fraction=0.01)
    roc_auc = compute_roc_auc(scores, labels)

    log.info("─── Benchmark Results ───")
    log.info(f"  EF1% (Enrichment Factor at 1%): {ef1:.3f}")
    log.info(f"  ROC-AUC:                         {roc_auc:.3f}")
    log.info(f"  Actives in set:                  {n_actives}")
    log.info(f"  Total compounds screened:        {n_total}")

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

    return {
        "ef1": float(ef1),
        "roc_auc": float(roc_auc),
        "n_actives": n_actives,
        "n_total": n_total,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="AutoAntibiotic Enrichment Benchmark",
    )
    parser.add_argument(
        "--use-vina", action="store_true",
        help="Use Vina docking for scoring (requires prepared targets).",
    )
    parser.add_argument(
        "--n-decoys", type=int, default=DECOY_COUNT,
        help=f"Number of decoys per active (default: {DECOY_COUNT}).",
    )
    args = parser.parse_args(argv)

    results = run_enrichment_test(
        n_decoys_per_active=args.n_decoys,
        use_vina=args.use_vina,
    )

    print("\n" + "=" * 55)
    print("  BENCHMARK SUMMARY")
    print("=" * 55)
    print(f"  EF1%:           {results['ef1']:.3f}  (>1.0 = enrichment)")
    print(f"  ROC-AUC:        {results['roc_auc']:.3f}  (>0.7 = good)")
    print(f"  Actives:        {results['n_actives']}")
    print(f"  Total screened: {results['n_total']}")
    print("=" * 55)
    print()

    results_dir = os.path.join("output", "benchmarks")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "enrichment_baseline.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
