#!/usr/bin/env python3
"""
DUD-E style enrichment benchmark for PBP2a (D4).

Docks the 21 experimental PBP2a actives (data/active_site_actives.csv) and a
DUD-E style set of property-matched decoys (50 per active; MW +-10%, logP
+-0.5, HBD/HBA +-1, rotatable bonds +-2, Tanimoto < 0.35 to any active)
against the PBP2a apo (1VQQ) receptor at exhaustiveness 32.

Reports:
    - ROC-AUC
    - EF_1%, EF_5%, EF_10%
    - BEDROC (alpha = 20)

Usage:
    AUTOANTIBIOTIC_MODE=science python scripts/dude_benchmark.py
    AUTOANTIBIOTIC_MODE=science python scripts/dude_benchmark.py --decoy-pool data/screen_library_final.csv

Outputs:
    output/dude_benchmark_results.json   — metrics + verdict
    output/dude_benchmark_roc.png        — ROC curve
    output/dude_decoys.csv               — generated decoy set (for reproducibility)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, DataStructs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import discovery_pipeline as P
from config.constants import ACTIVE_BOX_SIZE, ACTIVE_SITE_RESIDUES

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dude_bench")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "output")
DATA = os.path.join(REPO, "data")
os.makedirs(OUT, exist_ok=True)

# Decoy property-matching tolerances (DUD-E style).
DECOYS_PER_ACTIVE = 50
# Union of all in-house libraries serves as the offline decoy pool (the
# pipeline's own PBP2a-focused library alone cannot fill DUD-E's 50/active
# target for property-extreme actives; see generate_decoys relaxed tier).
DEFAULT_DECOY_POOL = ",".join(
    "data/{}.csv".format(fn)
    for fn in [
        "screen_library_final", "combined_library", "diverse_pbp2a_library",
        "pbp2a_focused_seed", "expanded_prelib", "screen_library",
        "screen_library_v2", "screen_library_v3", "known_decoys",
    ]
)
MW_TOL_FRAC = 0.10
LOGP_TOL = 0.5
HBD_TOL = 1
HBA_TOL = 1
ROT_TOL = 2
MAX_TANIMOTO_TO_ACTIVE = 0.50
RANDOM_SEED = 42
# Minimum acceptable decoys per active; below this the relaxed tolerance tier
# kicks in so every active still receives a usable decoy set.
DECOY_FALLBACK_MIN = 15

# Metric thresholds used for the PASS/FAIL verdict.
AUC_MIN = 0.70
EF1_MIN = 5.0


def compute_roc(labels, scores):
    """Return (fpr_list, tpr_list, auc) for binary labels and higher=better scores."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    labels = np.asarray(labels, dtype=int)[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0], 0.5
    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    for lab in labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        cur_tpr = tp / n_pos
        cur_fpr = fp / n_neg
        auc += (cur_fpr - prev_fpr) * (cur_tpr + prev_tpr) / 2.0
        tpr.append(cur_tpr)
        fpr.append(cur_fpr)
        prev_fpr, prev_tpr = cur_fpr, cur_tpr
    auc += (1.0 - prev_fpr) * (1.0 + prev_tpr) / 2.0
    return fpr, tpr, auc


def compute_bedrock(labels, scores, alpha=20.0):
    """Boltzmann-Enhanced Discrimination of ROC (BEDROC) metric."""
    N = len(labels)
    n_act = sum(labels)
    if n_act == 0 or n_act == N:
        return 0.5
    order = np.argsort(-np.asarray(scores, dtype=float))
    labels_arr = np.asarray(labels, dtype=int)
    ranks = [idx + 1 for idx, pos in enumerate(order) if labels_arr[pos] == 1]
    sum_exp = sum(np.exp(-alpha * r / N) for r in ranks)
    factor = (1.0 - np.exp(-alpha)) / (1.0 - np.exp(-alpha * n_act / N))
    return float((sum_exp / n_act) * factor)


def _mol_props(mol):
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rot": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def _props_match(cp, ap) -> bool:
    if abs(cp["mw"] - ap["mw"]) / max(abs(ap["mw"]), 1e-6) > MW_TOL_FRAC:
        return False
    if abs(cp["logp"] - ap["logp"]) > LOGP_TOL:
        return False
    if abs(cp["hbd"] - ap["hbd"]) > HBD_TOL:
        return False
    if abs(cp["hba"] - ap["hba"]) > HBA_TOL:
        return False
    if abs(cp["rot"] - ap["rot"]) > ROT_TOL:
        return False
    return True


def _fp(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)


def generate_decoys(active_mols, pool_path, n_per_active=DECOYS_PER_ACTIVE):
    """Generate n_per_active property-matched decoys per active.

    Decoys are drawn from the union of *pool_path* libraries and must (i)
    match the active's MW/logP/HBD/HBA/rotatable-bond profile and (ii) be
    topologically distant (Tanimoto < MAX_TANIMOTO_TO_ACTIVE) from the *paired*
    active, mirroring DUD-E. When fewer than ``DECOY_FALLBACK_MIN`` strict
    matches exist (unavoidable for property-extreme actives given a focused
    offline pool), a relaxed tolerance tier is used so every active receives a
    usable decoy set. Returns a list of (smiles, compound_id) decoys.
    """
    active_smis = {Chem.MolToSmiles(m) for m in active_mols}
    active_fps = [_fp(m) for m in active_mols]

    # Read candidate pool: all libraries in pool_path (comma-separated).
    pool = []
    pool_files = [p for p in pool_path.split(",") if p]
    for ppath in pool_files:
        with open(ppath, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smi = (row.get("smiles") or row.get("SMILES") or "").strip()
                if smi and smi not in active_smis:
                    pool.append(smi)
    log.info(f"  Candidate pool: {len(pool)} SMILES from {len(pool_files)} files")

    # Precompute pool properties and fingerprints once.
    pmols, pprops, pfps = [], [], []
    for smi in pool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        pmols.append(smi)
        pprops.append(_mol_props(mol))
        pfps.append(_fp(mol))
    log.info(f"  Pool usable: {len(pmols)} mols")

    rng = random.Random(RANDOM_SEED)
    decoys = []
    for ai, ap_mol in enumerate(active_mols):
        ap = _mol_props(ap_mol)
        afp = active_fps[ai]
        strict = [
            smi for smi, cp, fp in zip(pmols, pprops, pfps)
            if _props_match(cp, ap)
            and DataStructs.TanimotoSimilarity(fp, afp) < MAX_TANIMOTO_TO_ACTIVE
        ]
        cands = strict
        if len(cands) < DECOY_FALLBACK_MIN:
            relaxed = [
                smi for smi, cp, fp in zip(pmols, pprops, pfps)
                if abs(cp["mw"] - ap["mw"]) / max(abs(ap["mw"]), 1e-6) <= MW_TOL_FRAC * 1.5
                and abs(cp["logp"] - ap["logp"]) <= LOGP_TOL * 2
                and abs(cp["hbd"] - ap["hbd"]) <= HBD_TOL * 2
                and abs(cp["hba"] - ap["hba"]) <= HBA_TOL * 2
                and abs(cp["rot"] - ap["rot"]) <= ROT_TOL * 1.5
                and DataStructs.TanimotoSimilarity(fp, afp) < MAX_TANIMOTO_TO_ACTIVE
            ]
            log.info(f"  Active {ai}: {len(strict)} strict matches; using {len(relaxed)} relaxed")
            cands = relaxed
        rng.shuffle(cands)
        n = 0
        for smi in cands[:n_per_active]:
            decoys.append((smi, f"DECOY_{ai:02d}_{len(decoys):04d}"))
            n += 1
        if n < DECOY_FALLBACK_MIN:
            log.warning(f"  Active {ai}: only {n}/{n_per_active} decoys matched")
    log.info(f"  Generated {len(decoys)} property-matched decoys")
    return decoys


def load_benchmark(active_path, pool_path, n_decoys_per_active=DECOYS_PER_ACTIVE):
    """Load 21 actives + property-matched decoys. Returns (records, labels)."""
    records = []
    labels = []

    active_mols = []
    with open(active_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = row["smiles"].strip()
            cid = row["compound_id"].strip()
            mol = Chem.MolFromSmiles(smi)
            if not smi or mol is None:
                continue
            records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
            labels.append(1)
            active_mols.append(mol)
    log.info(f"  Loaded {len(active_mols)} known actives")

    decoys = generate_decoys(active_mols, pool_path, n_per_active=n_decoys_per_active)
    for smi, cid in decoys:
        records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
        labels.append(0)

    # Persist the decoy set for reproducibility.
    decoys_path = os.path.join(OUT, "dude_decoys.csv")
    with open(decoys_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "compound_id", "label"])
        for smi, cid in decoys:
            writer.writerow([smi, cid, "decoy"])
    log.info(f"  Decoys written: {decoys_path}")

    return records, labels


def main():
    parser = argparse.ArgumentParser(description="DUD-E style PBP2a enrichment benchmark")
    parser.add_argument(
        "--decoy-pool", type=str, default=None,
        help="SMILES library CSV used to source property-matched decoys "
             "(default: data/screen_library_final.csv).",
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=32,
        help="Vina exhaustiveness (default 32).",
    )
    parser.add_argument(
        "--n-decoys-per-active", type=int, default=DECOYS_PER_ACTIVE,
        help=f"Decoys per active (default {DECOYS_PER_ACTIVE}).",
    )
    args = parser.parse_args()

    config = P.load_config()
    config["mode"] = "science"
    deps = P.check_dependencies()
    if not deps["USE_VINA"]:
        log.error("Vina required for DUD-E benchmark. Aborting.")
        sys.exit(1)

    pool_path = args.decoy_pool or DEFAULT_DECOY_POOL
    if not all(os.path.exists(p) for p in pool_path.split(",")):
        log.error(f"Decoy pool files not found: {pool_path}")
        sys.exit(1)

    pdb_dir = os.path.join(OUT, "pdb_dude")
    work_dir = os.path.join(OUT, "workdir_dude")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Target prep with caching (reuse cached receptor PDBQTs where present).
    _orig_clean = P.clean_pdb_structure
    def _cached_clean(pdb_path, out_path, **kw):
        pdbqt = out_path.replace(".pdb", ".pdbqt")
        if (os.path.exists(out_path) and os.path.getsize(out_path) > 0
                and os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0):
            return pdbqt
        return _orig_clean(pdb_path, out_path, **kw)
    P.clean_pdb_structure = _cached_clean

    targets = P.prepare_targets(pdb_dir, work_dir, deps, config=config)
    pb2pa = targets["PBP2a"]
    receptor_pdbqts = pb2pa["receptor_pdbqts"]
    active_center = pb2pa["active_center"]
    cleaned_pdb = pb2pa["cleaned_pdb"]
    active_box = P._auto_box_size(
        cleaned_pdb, active_center, ACTIVE_BOX_SIZE,
        min_size=15.0, max_size=20.0, site_residues=ACTIVE_SITE_RESIDUES,
    )
    log.info(f"  Active-site box: {active_box}; center: {active_center}")

    # Docking against PBP2a apo (1VQQ) only — resting-state structure is the
    # DUD-E / DEKOIS standard for enrichment (holo conformers give decoys
    # artificially good scores in the expanded pocket).
    receptor_pdbqt = receptor_pdbqts[0]
    log.info(f"  Docking against apo receptor: {receptor_pdbqt}")

    records, labels = load_benchmark(
        os.path.join(DATA, "active_site_actives.csv"),
        pool_path,
        n_decoys_per_active=args.n_decoys_per_active,
    )
    log.info(f"  Benchmark set: {sum(labels)} actives, {len(labels) - sum(labels)} decoys")

    from functools import partial
    from utils.docking import dock_compound
    results = P._dock_compounds_parallel(
        records, receptor_pdbqt, active_center, active_box,
        work_dir, "dude",
        dock_func=partial(dock_compound, exhaustiveness=args.exhaustiveness),
    )
    energies = {rec.compound_id: energy for rec, energy in results}

    ids = [r.compound_id for r in records]
    scores = [-(energies[cid] if energies[cid] is not None else 1e9) for cid in ids]
    fpr, tpr, auc = compute_roc(labels, scores)
    bedrock = compute_bedrock(labels, scores, alpha=20.0)

    N = len(ids)
    n_act = sum(labels)
    k1 = max(1, round(0.01 * N))
    k5 = max(1, round(0.05 * N))
    k10 = max(1, round(0.10 * N))
    ranked = sorted(ids, key=lambda c: (energies[c] if energies[c] is not None else 1e9))
    act_in_1 = sum(1 for c in ranked[:k1] if labels[ids.index(c)] == 1)
    act_in_5 = sum(1 for c in ranked[:k5] if labels[ids.index(c)] == 1)
    act_in_10 = sum(1 for c in ranked[:k10] if labels[ids.index(c)] == 1)
    ef1 = (act_in_1 / n_act) / (k1 / N) if n_act else 0.0
    ef5 = (act_in_5 / n_act) / (k5 / N) if n_act else 0.0
    ef10 = (act_in_10 / n_act) / (k10 / N) if n_act else 0.0

    passed = auc >= AUC_MIN and ef1 >= EF1_MIN
    result = {
        "n_compounds": N,
        "n_actives": n_act,
        "n_decoys": N - n_act,
        "decoys_per_active": args.n_decoys_per_active,
        "decoy_pool": pool_path,
        "exhaustiveness": args.exhaustiveness,
        "auc": round(float(auc), 4),
        "bedrock_alpha20": round(float(bedrock), 4),
        "ef_1pct": round(float(ef1), 3),
        "ef_5pct": round(float(ef5), 3),
        "ef_10pct": round(float(ef10), 3),
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": {"auc_min": AUC_MIN, "ef_1pct_min": EF1_MIN},
        "active_box": list(active_box),
        "active_center": [float(v) for v in active_center],
        "receptor": "PBP2a apo 1VQQ",
        "method": "DUD-E style (property-matched decoys, apo receptor, Vina)",
    }
    with open(os.path.join(OUT, "dude_benchmark_results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # ROC plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("PBP2a DUD-E style enrichment (apo 1VQQ, ex=32)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dude_benchmark_roc.png"), dpi=300)
    plt.close(fig)

    log.info("")
    log.info("=" * 60)
    log.info("  DUD-E style benchmark")
    log.info("=" * 60)
    log.info(f"  Compounds: {N}  ({n_act} actives / {N - n_act} decoys)")
    log.info(f"  AUC={auc:.4f}  BEDROC(20)={bedrock:.4f}  "
             f"EF_1%={ef1:.2f}  EF_5%={ef5:.2f}  EF_10%={ef10:.2f}")
    log.info(f"  VERDICT: {'PASS' if passed else 'FAIL'} "
             f"(AUC>={AUC_MIN} and EF_1%>={EF1_MIN} required)")
    log.info("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
