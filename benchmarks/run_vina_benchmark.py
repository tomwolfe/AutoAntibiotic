"""
Benchmark: Real Docking Enrichment via AutoDock Vina
====================================================

Docks known PBP2a actives, inactives, and property-matched decoys
against the PBP2a allosteric site using AutoDock Vina, then
computes **Enrichment Factor at 1% (EF1%)** and **ROC-AUC**.

Usage::

    python -m benchmarks.run_vina_benchmark
    python -m benchmarks.run_vina_benchmark --n-decoys 50
    python -m benchmarks.run_vina_benchmark --site active

Requires ``vina`` in PATH (install via ``conda install -c conda-forge vina``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.reference_data import (
    DECOY_COUNT,
    get_active_labels,
    get_actives_smiles,
    get_inactive_labels,
    get_inactives_smiles,
)
from autoantibiotic.config import CONFIG
from autoantibiotic.io_utils import log
from autoantibiotic.models import CompoundRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

_HAVE_VINA: bool = False
try:
    result = __import__("subprocess").run(
        ["vina", "--version"], capture_output=True, text=True, timeout=10,
    )
    _HAVE_VINA = result.returncode == 0
except Exception:
    pass


def _compute_property_vector(mol: Chem.Mol) -> np.ndarray:
    mw = Chem.Descriptors.MolWt(mol)
    logp = Chem.Crippen.MolLogP(mol)
    hbd = Chem.Descriptors.NumHDonors(mol)
    hba = Chem.Descriptors.NumHAcceptors(mol)
    rot = Chem.Descriptors.NumRotatableBonds(mol)
    return np.array([mw / 500.0, min(max(logp, -2), 8) / 8.0, hbd / 10.0, hba / 10.0, rot / 15.0], dtype=np.float64)


def _generate_pool(size: int = 3000, seed: int = 42) -> List[Chem.Mol]:
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

    for smi in CONFIG.brics_building_blocks:
        _add_mol(smi)
    for smi in CONFIG.natural_product_scaffolds:
        _add_mol(smi)
    for smi in CONFIG.additional_scaffolds:
        _add_mol(smi)
    for smi in CONFIG.reference_antibiotics.values():
        _add_mol(smi)
    for smi in CONFIG.control_smiles.values():
        _add_mol(smi)
    for smi in get_inactives_smiles():
        _add_mol(smi)

    rng = np.random.default_rng(seed + 999)
    inactives_expanded: List[str] = list(get_inactives_smiles())
    for _ in range(200):
        idx = rng.integers(0, len(inactives_expanded))
        base_smi = inactives_expanded[idx]
        base_mol = Chem.MolFromSmiles(base_smi)
        if base_mol is None:
            continue
        try:
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

    pool = [m for m in pool if 150 <= Chem.Descriptors.MolWt(m) <= 700]
    log.info(f"  Decoy pool size: {len(pool)}")
    return pool


def generate_decoys(
    actives: List[Chem.Mol],
    active_labels: List[str],
    n_decoys_per_active: int = 100,
    pool_size: int = 3000,
    seed: int = 42,
) -> List[Tuple[str, Chem.Mol, str]]:
    from rdkit.Chem import AllChem, Crippen, Descriptors
    from rdkit.DataStructs import TanimotoSimilarity

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
            remaining = [
                (pi, np.linalg.norm(
                    np.array([pool_mw[pi] / 500.0, pool_logp[pi], pool_hbd[pi],
                              pool_hba[pi], pool_rot[pi]], dtype=np.float64)
                    - np.array([amw / 500.0, alogp, ahbd, ahba, arot], dtype=np.float64),
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


def _prepare_pbp2a_receptor() -> Optional[Dict[str, Any]]:
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
        log.info("  Using cached prepared receptor.")
    else:
        log.info("  Downloading and preparing PBP2a receptor…")
        apo_path = fetch_structure(CONFIG.pdb_ids["PBP2a_apo"], pdb_dir)
        _ = clean_pdb_structure(apo_path, cleaned_pdb, deps={})
        if not os.path.exists(cleaned_pdbqt):
            log.warning("  PDBQT not available; using PDB.")
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


def _dock_compound(
    record: CompoundRecord,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
) -> Optional[float]:
    from autoantibiotic.docking import dock_compound
    return dock_compound(record, receptor_pdbqt, center, box_size, work_dir, tag="vb")


def compute_enrichment_factor(
    scores: np.ndarray, labels: np.ndarray, fraction: float = 0.01,
) -> float:
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


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, -scores))


def run_vina_benchmark(
    n_decoys_per_active: int = DECOY_COUNT,
    site: str = "allosteric",
    results_dir: str = "output/benchmarks",
) -> Dict[str, Any]:
    """Run the full Vina docking enrichment benchmark.

    Steps:
        1. Check Vina availability.
        2. Prepare the PBP2a receptor.
        3. Load reference actives and inactives.
        4. Generate property-matched decoys.
        5. Dock all compounds with Vina.
        6. Compute EF1%, ROC-AUC.
        7. Save results to JSON.
    """
    log.info("═══ Vina Docking Benchmark ═══")

    if not _HAVE_VINA:
        log.error(
            "  Vina is not installed or not found in PATH.\n"
            "  Install it with:  conda install -c conda-forge vina"
        )
        return {"error": "Vina not installed"}

    log.info("  Preparing PBP2a receptor…")
    receptor = _prepare_pbp2a_receptor()
    if receptor is None:
        log.error("  Failed to prepare receptor. Aborting.")
        return {"error": "Receptor preparation failed"}

    center_key = "allosteric_center" if site == "allosteric" else "active_center"
    box_size = CONFIG.allosteric_box_size if site == "allosteric" else CONFIG.active_box_size
    center = receptor[center_key]
    receptor_pdbqt = receptor["pdbqt"]
    log.info(f"  Binding site: {site}")
    log.info(f"  Center: {center}")
    log.info(f"  Box: {box_size}")

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
            CompoundRecord(compound_id=f"ACTIVE_{active_labels[i]}", smiles=smi, mol=mol)
        )
    active_labels = active_labels_filtered

    inactive_records: List[CompoundRecord] = []
    for i, smi in enumerate(inactive_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        inactive_records.append(
            CompoundRecord(compound_id=f"INACTIVE_{get_inactive_labels()[i]}", smiles=smi, mol=mol)
        )

    log.info("  Generating property-matched decoys…")
    decoys = generate_decoys(active_mols, active_labels, n_decoys_per_active, seed=42)
    decoy_records: List[CompoundRecord] = []
    for did, mol, _matched_active in decoys:
        decoy_records.append(
            CompoundRecord(compound_id=did, smiles=Chem.MolToSmiles(mol), mol=mol)
        )

    all_records: List[CompoundRecord] = active_records + inactive_records + decoy_records
    n_actives = len(active_records)
    n_total = len(all_records)
    log.info(f"  Mixed set: {n_actives} actives, {len(inactive_records)} inactives, "
             f"{len(decoy_records)} decoys (total {n_total})")

    log.info("  Docking all compounds with Vina…")
    work_dir = str(CONFIG.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    energy_attr = "pb2pa_allosteric_energy" if site == "allosteric" else "pb2pa_active_energy"
    n_scored = 0
    for rec in all_records:
        energy = _dock_compound(rec, receptor_pdbqt, center, box_size, work_dir)
        if energy is not None:
            setattr(rec, energy_attr, energy)
            n_scored += 1
        else:
            setattr(rec, energy_attr, 0.0)
    log.info(f"  Docking complete: {n_scored}/{n_total} compounds scored.")

    scores: np.ndarray = np.array([
        getattr(r, energy_attr) if getattr(r, energy_attr) is not None else 0.0
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
        log.info("  ⚠ Enrichment at or below random.")
    if roc_auc > 0.7:
        log.info("  ✓ Good discriminatory power (ROC-AUC > 0.7).")
    elif roc_auc > 0.55:
        log.info("  ✓ Moderate discriminatory power.")
    else:
        log.info("  ⚠ Poor discriminatory power (near random).")

    results: Dict[str, Any] = {
        "ef1": float(ef1),
        "roc_auc": float(roc_auc),
        "n_actives": n_actives,
        "n_total": n_total,
        "n_scored": n_scored,
        "site": site,
        "n_decoys_per_active": n_decoys_per_active,
        "receptor_pdbqt": receptor_pdbqt,
        "center": [float(c) for c in center],
        "box_size": [float(b) for b in box_size],
    }

    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"vina_benchmark_{site}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Results saved to {results_path}")

    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="AutoAntibiotic Vina Docking Benchmark",
    )
    parser.add_argument(
        "--n-decoys", type=int, default=DECOY_COUNT,
        help=f"Number of decoys per active (default: {DECOY_COUNT}).",
    )
    parser.add_argument(
        "--site", choices=["allosteric", "active"], default="allosteric",
        help="Binding site to dock against (default: allosteric).",
    )
    parser.add_argument(
        "--results-dir", default="output/benchmarks",
        help="Directory to save results JSON (default: output/benchmarks).",
    )
    args = parser.parse_args(argv)

    results = run_vina_benchmark(
        n_decoys_per_active=args.n_decoys,
        site=args.site,
        results_dir=args.results_dir,
    )

    print("\n" + "=" * 55)
    print("  VINA BENCHMARK SUMMARY")
    print("=" * 55)
    if "error" in results:
        print(f"  ERROR: {results['error']}")
    else:
        print(f"  EF1%:           {results['ef1']:.3f}  (>1.0 = enrichment)")
        print(f"  ROC-AUC:        {results['roc_auc']:.3f}  (>0.7 = good)")
        print(f"  Actives:        {results['n_actives']}")
        print(f"  Total screened: {results['n_total']}")
        print(f"  Scored:         {results['n_scored']}")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()
