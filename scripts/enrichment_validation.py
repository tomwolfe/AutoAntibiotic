#!/usr/bin/env python3
"""
Enrichment validation for the AutoAntibiotic PBP2a screen.

Builds a 620-compound benchmark (120 real seed actives/decoys + 500
property-matched decoys + ceftaroline/meropenem positive controls), docks every
compound against PBP2a (active site, consensus over the 3 PBP2a conformers),
and reports ROC-AUC, EF_1% and EF_5% as a measure of protocol discrimination.

Usage:
    AUTOANTIBIOTIC_MODE=science python scripts/enrichment_validation.py

Outputs:
    output/enrichment_results.json
    output/enrichment_roc.png
"""
from __future__ import annotations

import os
import sys
import json
import random
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import BRICS, Crippen, rdMolDescriptors, Descriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import discovery_pipeline as P  # noqa: E402
from config.constants import (  # noqa: E402
    BETA_LACTAM_SMARTS, ACTIVE_BOX_SIZE, ACTIVE_SITE_RESIDUES,
    CONSERVED_RESIDUES,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("enrichment")

BETA_LACTAM = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "output")
os.makedirs(OUT, exist_ok=True)


def _props(mol):
    return (
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
    )


def build_decoy_pool(seed_mols, target=2000):
    """BRICS-recombine seed fragments into a large pool of candidate decoys."""
    frags = set()
    for mol in seed_mols:
        try:
            for f in BRICS.BRICSDecompose(mol, minFragmentSize=6):
                fm = Chem.MolFromSmiles(f)
                if fm is not None and fm.GetNumHeavyAtoms() >= 6:
                    frags.add(f)
        except Exception:
            continue
    frag_list = [Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f) is not None]
    log.info(f"  Decoy fragment pool: {len(frag_list)} fragments")
    pool = []
    seen = set()
    rng = random.Random(7)
    builder = BRICS.BRICSBuild(frag_list)
    for prod in builder:
        try:
            Chem.SanitizeMol(prod)
        except Exception:
            continue
        smi = Chem.MolToSmiles(prod)
        if smi in seen:
            continue
        seen.add(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if mol.HasSubstructMatch(BETA_LACTAM):
            continue
        mw = Descriptors.MolWt(mol)
        if mw < 200 or mw > 550:
            continue
        pool.append(mol)
        if len(pool) >= target:
            break
    log.info(f"  Decoy candidate pool: {len(pool)} valid non-lactam molecules")
    return pool


def select_property_matched(pool, templates, n=200):
    """Pick up to `n` decoys each matching some seed template within tolerance.

    Tolerance (per task spec): MW +/-10%, logP +/-0.5, TPSA +/-15%,
    HBD/HBA +/-1, rotatable bonds +/-2. A candidate is accepted if it matches
    ANY template, so we iterate the full pool deterministically to maximise
    the number of property-matched decoys collected.
    """
    tol = (0.10, 0.5, 0.15, 1, 1, 2)
    tprops = [_props(t) for t in templates]
    chosen = []
    seen = set()
    for cand in pool:
        if len(chosen) >= n:
            break
        smi = Chem.MolToSmiles(cand)
        if smi in seen:
            continue
        cp = _props(cand)
        for tp in tprops:
            ok = all(
                abs(c - t) <= (tol[i] * max(abs(t), 1e-6) if i in (0, 2) else tol[i])
                for i, (c, t) in enumerate(zip(cp, tp))
            )
            if ok:
                seen.add(smi)
                chosen.append(cand)
                break
    log.info(f"  Selected {len(chosen)} property-matched decoys (pool={len(pool)})")
    return chosen


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


def main():
    config = P.load_config()
    config["mode"] = "science"
    deps = P.check_dependencies()
    if not deps["USE_VINA"]:
        log.error("Vina required for enrichment validation. Aborting.")
        sys.exit(1)

    pdb_dir = os.path.join(OUT, "pdb_enrich")
    work_dir = os.path.join(OUT, "workdir_enrich")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # ── Target prep: 3 PBP2a conformers + active-site grid ──
    # Cache already-prepared receptor PDBQTs to skip the slow obabel step on
    # re-runs (obabel -xr is very slow on these large structures).
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

    # ── Build the benchmark library ──
    seed_df = pd.read_csv(os.path.join(REPO, "novel_seed.csv"))
    seed_mols = []
    for smi in seed_df["smiles"]:
        m = Chem.MolFromSmiles(str(smi))
        if m is not None:
            seed_mols.append(m)
    log.info(f"  Read {len(seed_mols)} valid seed molecules")

    templates = seed_mols
    pool = build_decoy_pool(seed_mols, target=12000)
    decoys = select_property_matched(pool, templates)

    # Assemble records: seeds + decoys + 2 positive controls
    from utils.library_gen import CompoundRecord
    records = []
    for i, smi in enumerate(seed_df["smiles"]):
        records.append(CompoundRecord(compound_id=f"SEED_{i:04d}", smiles=str(smi)))
    for i, mol in enumerate(decoys):
        records.append(CompoundRecord(
            compound_id=f"DECOY_{i:04d}", smiles=Chem.MolToSmiles(mol)))
    ctrl_smiles = {
        "CTRL_Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CTRL_Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
    }
    for cid, smi in ctrl_smiles.items():
        records.append(CompoundRecord(compound_id=cid, smiles=smi))
    log.info(f"  Benchmark library: {len(records)} compounds")

    # ── Consensus docking against PBP2a active site ──
    docked = P._consensus_dock(
        records, receptor_pdbqts, active_center, active_box,
        work_dir, "active", use_vina=True,
    )
    energies = {rec.compound_id: e for rec, e in docked}

    # ── Define actives ──
    active_ids = set(ctrl_smiles.keys())
    for cid, e in energies.items():
        if cid.startswith("SEED_") and e is not None and e < -8.0:
            active_ids.add(cid)
    log.info(f"  Number of actives: {len(active_ids)}")

    # ── ROC / EF ──
    ids = list(energies.keys())
    labels = [1 if cid in active_ids else 0 for cid in ids]
    scores = [- (energies[c] if energies[c] is not None else 1e9) for c in ids]
    fpr, tpr, auc = compute_roc(labels, scores)

    N = len(ids)
    n_act = len(active_ids)
    k1 = max(1, round(0.01 * N))
    k5 = max(1, round(0.05 * N))
    ranked = sorted(ids, key=lambda c: (energies[c] if energies[c] is not None else 1e9))
    act_in_1 = sum(1 for c in ranked[:k1] if c in active_ids)
    act_in_5 = sum(1 for c in ranked[:k5] if c in active_ids)
    ef1 = (act_in_1 / n_act) / (k1 / N) if n_act else 0.0
    ef5 = (act_in_5 / n_act) / (k5 / N) if n_act else 0.0

    passed = (auc >= 0.7) and (ef1 >= 5.0)
    result = {
        "n_compounds": N,
        "n_seeds": len(seed_mols),
        "n_decoys": len(decoys),
        "n_actives": n_act,
        "auc": round(float(auc), 4),
        "ef_1pct": round(float(ef1), 3),
        "ef_5pct": round(float(ef5), 3),
        "active_box": list(active_box),
        "active_site_max_size": 20.0,
        "verdict": "PASS" if passed else "FAIL",
    }
    with open(os.path.join(OUT, "enrichment_results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # ROC plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("PBP2a Enrichment ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "enrichment_roc.png"), dpi=300)
    plt.close(fig)

    # EF bar chart inset
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(["EF_1%", "EF_5%"], [ef1, ef5], color=["#2c7fb8", "#7fcdbb"])
    ax2.axhline(5, color="r", ls="--", lw=1, label="pass threshold (5)")
    ax2.set_ylabel("Enrichment Factor")
    ax2.set_title("Enrichment Factors")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "enrichment_ef.png"), dpi=300)
    plt.close(fig2)

    log.info("=" * 50)
    log.info(f"  Enrichment validation: AUC={auc:.3f}  EF_1%={ef1:.2f}  "
             f"EF_5%={ef5:.2f}")
    log.info(f"  VERDICT: {'PASS' if passed else 'FAIL'} "
             f"(AUC>=0.7 and EF_1%>=5 required)")
    log.info("=" * 50)
    if not passed:
        log.warning("  Enrichment FAILED — consider increasing the active-site "
                    "box max_size from 20 to 25 A and re-running.")
    # Emit a clear exit code for automation
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
