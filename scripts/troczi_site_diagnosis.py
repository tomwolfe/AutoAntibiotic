#!/usr/bin/env python3
"""
Troczi 2013 site-specific enrichment diagnosis (D1).

The in-house Troczi benchmark (scripts/troczi_benchmark.py) docks the 10
Troczi oxadiazole/quinazolinone actives + 150 decoys against the PBP2a
ACTIVE-site grid (Ser403/Lys406/Tyr446) and reports AUC = 0.297 — far below
the AUC ~0.82 reported by Troczi et al. (JCIM 2013).

Hypothesis tested here: the Troczi oxadiazoles are known ALLOSTERIC PBP2a
inhibitors. Docking them into the catalytic active site is therefore an
off-target measurement, which explains the poor enrichment. This script docks
the same actives/decoys against the ALLOSTERIC grid (Tyr105/Gln199/Glu237)
using the identical protocol and compares the resulting enrichment.

Usage:
    AUTOANTIBIOTIC_MODE=science python scripts/troczi_site_diagnosis.py

Outputs:
    output/troczi_site_diagnosis.json   — AUC/EF at both sites + verdict
    output/troczi_site_comparison.png   — paired bar chart
"""
from __future__ import annotations

import os
import sys
import json
import csv
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import discovery_pipeline as P
from config.constants import (
    ACTIVE_BOX_SIZE, ALLOSTERIC_BOX_SIZE,
    ACTIVE_SITE_RESIDUES, ALLOSTERIC_RESIDUES,
)
from utils.structure_prep import compute_residue_centroid

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("troczi_site")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "output")
DATA = os.path.join(REPO, "data")
os.makedirs(OUT, exist_ok=True)


def load_benchmark(data_dir: str):
    """Load Troczi actives + property-matched decoys. Returns (records, labels)."""
    records = []
    labels = []
    actives_path = os.path.join(data_dir, "troczi_2013_actives.csv")
    with open(actives_path, newline="") as fh:
        for row in csv.DictReader(fh):
            smi = row["smiles"].strip()
            cid = row["compound_id"].strip()
            if smi:
                records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
                labels.append(1)
    decoys_path = os.path.join(data_dir, "known_decoys.csv")
    with open(decoys_path, newline="") as fh:
        for row in csv.DictReader(fh):
            smi = row["smiles"].strip()
            cid = row["compound_id"].strip()
            if smi:
                records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
                labels.append(0)
    return records, labels


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


def dock_and_score(records, labels, receptor_pdbqt, center, box, work_dir, tag):
    """Dock against a single grid and return (energies, auc, ef1, ef5)."""
    results = P._dock_compounds_parallel(
        records, receptor_pdbqt, center, box, work_dir, tag,
    )
    energies = {rec.compound_id: energy for rec, energy in results}

    ids = [r.compound_id for r in records]
    scores = [-(energies[cid] if energies[cid] is not None else 1e9) for cid in ids]
    fpr, tpr, auc = compute_roc(labels, scores)

    N = len(ids)
    n_act = sum(labels)
    k1 = max(1, round(0.01 * N))
    k5 = max(1, round(0.05 * N))
    ranked = sorted(ids, key=lambda c: (energies[c] if energies[c] is not None else 1e9))
    act_in_1 = sum(1 for c in ranked[:k1] if labels[ids.index(c)] == 1)
    act_in_5 = sum(1 for c in ranked[:k5] if labels[ids.index(c)] == 1)
    ef1 = (act_in_1 / n_act) / (k1 / N) if n_act else 0.0
    ef5 = (act_in_5 / n_act) / (k5 / N) if n_act else 0.0
    return energies, auc, ef1, ef5, ranked


def main():
    config = P.load_config()
    config["mode"] = "science"
    deps = P.check_dependencies()
    if not deps["USE_VINA"]:
        log.error("Vina required. Aborting.")
        sys.exit(1)

    pdb_dir = os.path.join(OUT, "pdb_troczi_bench")
    work_dir = os.path.join(OUT, "workdir_troczi_bench")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Target prep with caching (reuses cached PBP2a_clean.pdbqt if present)
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
    receptor_pdbqt = pb2pa["receptor_pdbqts"][0]  # apo only, like main benchmark
    cleaned_pdb = pb2pa["cleaned_pdb"]

    # Grids: active (Ser403/Lys406/Tyr446) and allosteric (Tyr105/Gln199/Glu237)
    active_center = pb2pa["active_center"]
    active_box = P._auto_box_size(
        cleaned_pdb, active_center, ACTIVE_BOX_SIZE,
        min_size=15.0, max_size=20.0, site_residues=ACTIVE_SITE_RESIDUES,
    )
    allosteric_center = pb2pa.get("allosteric_center")
    if allosteric_center is None:
        allosteric_center = compute_residue_centroid(
            cleaned_pdb, ALLOSTERIC_RESIDUES, use_ca=False)
    allosteric_box = P._auto_box_size(
        cleaned_pdb, allosteric_center, ALLOSTERIC_BOX_SIZE,
        min_size=15.0, max_size=18.0, site_residues=ALLOSTERIC_RESIDUES,
    )
    log.info(f"  Active grid: center={active_center}, box={active_box}")
    log.info(f"  Allosteric grid: center={allosteric_center}, box={allosteric_box}")

    records, labels = load_benchmark(DATA)
    log.info(f"  Loaded {sum(labels)} actives + {len(labels) - sum(labels)} decoys")

    log.info("  Docking against ACTIVE site grid (reproduces AUC 0.297)...")
    act_energies, act_auc, act_ef1, act_ef5, _ = dock_and_score(
        records, labels, receptor_pdbqt, active_center, active_box,
        work_dir, "site_diag_active",
    )

    log.info("  Docking against ALLOSTERIC site grid...")
    all_energies, all_auc, all_ef1, all_ef5, all_ranked = dock_and_score(
        records, labels, receptor_pdbqt, allosteric_center, allosteric_box,
        work_dir, "site_diag_allosteric",
    )

    # Verdict: allosteric hypothesis supported if allosteric AUC is meaningfully
    # better than active-site AUC and >= 0.65 (standard enrichment bar).
    delta = all_auc - act_auc
    hypothesis_supported = all_auc >= 0.65 and delta >= 0.2

    result = {
        "n_actives": sum(labels),
        "n_decoys": len(labels) - sum(labels),
        "active_site": {
            "center": [float(v) for v in active_center],
            "box": [float(v) for v in active_box],
            "auc": round(float(act_auc), 4),
            "ef_1pct": round(float(act_ef1), 3),
            "ef_5pct": round(float(act_ef5), 3),
        },
        "allosteric_site": {
            "center": [float(v) for v in allosteric_center],
            "box": [float(v) for v in allosteric_box],
            "auc": round(float(all_auc), 4),
            "ef_1pct": round(float(all_ef1), 3),
            "ef_5pct": round(float(all_ef5), 3),
        },
        "delta_auc_allosteric_minus_active": round(float(delta), 4),
        "hypothesis": (
            "Troczi oxadiazoles are allosteric PBP2a binders; active-site "
            "docking measures an off-target pocket."
        ),
        "hypothesis_supported": bool(hypothesis_supported),
        "allosteric_actives_ranked": [
            {"compound_id": cid,
             "allosteric_energy": round(float(all_energies[cid]), 2)
             if all_energies[cid] is not None else None,
             "rank": i + 1,
             "active_energy": round(float(act_energies[cid]), 2)
             if act_energies[cid] is not None else None}
            for i, cid in enumerate(all_ranked)
            if labels[records.index(next(r for r in records if r.compound_id == cid))] == 1
        ],
    }

    with open(os.path.join(OUT, "troczi_site_diagnosis.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # Paired comparison figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels_bar = ["Active site", "Allosteric site"]
    vals = [act_auc, all_auc]
    colors = ["#d95f02", "#2c7fb8"]
    bars = ax.bar(labels_bar, vals, color=colors, width=0.55)
    ax.axhline(0.7, color="r", ls="--", lw=1, label="enrichment bar (0.70)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=11)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Troczi 2013 oxadiazoles: enrichment by docking site")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "troczi_site_comparison.png"), dpi=300)
    plt.close(fig)

    log.info("")
    log.info("=" * 60)
    log.info("  Troczi site diagnosis")
    log.info("=" * 60)
    log.info(f"  Active-site AUC:    {act_auc:.3f}  (EF_1%={act_ef1:.2f})")
    log.info(f"  Allosteric-site AUC:{all_auc:.3f}  (EF_1%={all_ef1:.2f})")
    log.info(f"  ΔAUC (allo − active): {delta:+.3f}")
    if hypothesis_supported:
        log.info("  ✓ Hypothesis SUPPORTED: Troczi actives enrich against the "
                 "allosteric grid, confirming they are allosteric binders.")
    else:
        log.warning("  ✗ Hypothesis NOT supported: allosteric enrichment is not "
                    "meaningfully better. Investigate protonation/grid/tautomers.")
    log.info("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
