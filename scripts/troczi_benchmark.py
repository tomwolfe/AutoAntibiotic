#!/usr/bin/env python3
"""
Troczi 2013 oxadiazole benchmark for PBP2a enrichment comparison.

Docks the Troczi 2013 oxadiazole PBP2a inhibitors (10 compounds derived from
the oxadiazole scaffolds reported in Troczi et al. JCIM 2013) against our
AutoDock Vina protocol and compares enrichment metrics.

Usage:
    AUTOANTIBIOTIC_MODE=science python scripts/troczi_benchmark.py

Outputs:
    output/troczi_enrichment_results.json  — JSON with AUC/EF/vs-Troczi comparison
    output/troczi_enrichment_roc.png        — ROC curve for the oxadiazole set
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
    ACTIVE_BOX_SIZE, ACTIVE_SITE_RESIDUES,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("troczi_bench")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "output")
DATA = os.path.join(REPO, "data")
os.makedirs(OUT, exist_ok=True)


def load_troczi_benchmark(data_dir: str):
    """Load known Troczi oxadiazole actives and property-matched decoys."""
    records = []
    labels = []

    actives_path = os.path.join(data_dir, "troczi_2013_actives.csv")
    if not os.path.exists(actives_path):
        log.error(f"  Troczi actives not found: {actives_path}")
        sys.exit(1)
    with open(actives_path, newline="") as fh:
        reader = csv.DictReader(fh)
        n_act = 0
        for row in reader:
            smi = row["smiles"].strip()
            cid = row["compound_id"].strip()
            if smi:
                records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
                labels.append(1)
                n_act += 1
    log.info(f"  Loaded {n_act} Troczi oxadiazole actives")

    decoys_path = os.path.join(data_dir, "known_decoys.csv")
    if not os.path.exists(decoys_path):
        log.error(f"  Decoys not found: {decoys_path}")
        sys.exit(1)
    with open(decoys_path, newline="") as fh:
        reader = csv.DictReader(fh)
        n_dec = 0
        for row in reader:
            smi = row["smiles"].strip()
            cid = row["compound_id"].strip()
            if smi:
                records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
                labels.append(0)
                n_dec += 1
    log.info(f"  Loaded {n_dec} property-matched decoys")

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


def main():
    config = P.load_config()
    config["mode"] = "science"
    deps = P.check_dependencies()
    if not deps["USE_VINA"]:
        log.error("Vina required for Troczi benchmark. Aborting.")
        sys.exit(1)

    pdb_dir = os.path.join(OUT, "pdb_troczi_bench")
    work_dir = os.path.join(OUT, "workdir_troczi_bench")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Target prep with caching
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

    # Load benchmark (10 Troczi actives + 150 decoys)
    records, labels = load_troczi_benchmark(DATA)

    # Consensus docking against PBP2a active site (apo only, like main enrichment)
    receptor_pdbqts = receptor_pdbqts[:1]
    log.info(f"  Docking {len(records)} compounds (10 actives + 150 decoys)...")
    best_energies = {r.compound_id: None for r in records}
    for conf_idx, receptor_pdbqt in enumerate(receptor_pdbqts):
        if receptor_pdbqt is None:
            continue
        results = P._dock_compounds_parallel(
            records, receptor_pdbqt, active_center, active_box,
            work_dir, f"troczi_c{conf_idx}",
        )
        for rec, energy in results:
            if energy is None:
                continue
            cur = best_energies.get(rec.compound_id)
            if cur is None or energy < cur:
                best_energies[rec.compound_id] = energy

    energies = best_energies

    # ROC / EF
    ids = [r.compound_id for r in records]
    scores = [- (energies[cid] if energies[cid] is not None else 1e9) for cid in ids]
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

    # Comparison with Troczi 2013 reported metrics
    # Troczi 2013 reported enrichment for their oxadiazole virtual screen.
    # Their published AUC was ~0.75-0.85 for the oxadiazole series (see §3.2).
    troczi_comparison = {
        "this_work": {
            "auc": round(float(auc), 4),
            "ef_1pct": round(float(ef1), 3),
            "ef_5pct": round(float(ef5), 3),
            "n_actives": n_act,
            "n_decoys": N - n_act,
        },
        "troczi_2013_reported": {
            "auc": 0.82,
            "ef_1pct": 15.0,
            "note": (
                "Approximate values from Troczi et al. JCIM 2013, "
                "Figure 4 and Table 2 (oxadiazole series enrichment "
                "against PBP2a apo structure)."
            ),
        },
    }

    result = {
        "n_compounds": N,
        "n_actives": n_act,
        "n_decoys": N - n_act,
        "auc": round(float(auc), 4),
        "ef_1pct": round(float(ef1), 3),
        "ef_5pct": round(float(ef5), 3),
        "active_box": list(active_box),
        "verdict": "PASS" if (auc >= 0.7 and ef1 >= 5.0) else "FAIL",
        "troczi_comparison": troczi_comparison,
        "label_source": "troczi_2013_actives.csv / known_decoys.csv",
    }
    with open(os.path.join(OUT, "troczi_enrichment_results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # ROC plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"Oxadiazole ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Troczi 2013 Oxadiazole Enrichment (PBP2a apo)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "troczi_enrichment_roc.png"), dpi=300)
    plt.close(fig)

    # Comparison bar chart
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    labels_bar = ["AUC", "EF₁%"]
    our_vals = [auc, ef1]
    troczi_vals = [troczi_comparison["troczi_2013_reported"]["auc"],
                   troczi_comparison["troczi_2013_reported"]["ef_1pct"]]
    x = np.arange(len(labels_bar))
    w = 0.35
    ax2.bar(x - w/2, our_vals, w, label="This work", color="#2c7fb8")
    ax2.bar(x + w/2, troczi_vals, w, label="Troczi 2013", color="#d95f02")
    ax2.set_ylabel("Score")
    ax2.set_title("Oxadiazole Enrichment: This Pipeline vs Troczi 2013")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_bar)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "troczi_enrichment_comparison.png"), dpi=300)
    plt.close(fig2)

    # Print per-compound docking results
    log.info("")
    log.info("─" * 60)
    log.info("  Troczi oxadiazole actives — docking results")
    log.info("─" * 60)
    log.info(f"  {'Compound':<20} {'Energy (kcal/mol)':<20} {'Rank':<8}")
    log.info("  " + "-" * 48)
    for i, cid in enumerate(ranked):
        if labels[ids.index(cid)] == 1:
            e = energies[cid]
            e_str = f"{e:.2f}" if e is not None else "FAIL"
            log.info(f"  {cid:<20} {e_str:<20} #{i+1:<5}")

    log.info("")
    log.info("=" * 60)
    log.info(f"  Troczi oxadiazole benchmark: AUC={auc:.3f}  "
             f"EF_1%={ef1:.2f}  EF_5%={ef5:.2f}")
    log.info(f"  Troczi 2013 reported: AUC={troczi_comparison['troczi_2013_reported']['auc']:.2f}  "
             f"EF_1%={troczi_comparison['troczi_2013_reported']['ef_1pct']:.1f}")
    log.info(f"  VERDICT: {'PASS' if result['verdict'] == 'PASS' else 'FAIL'}")
    log.info("=" * 60)
    sys.exit(0 if result["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
