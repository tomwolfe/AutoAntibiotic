#!/usr/bin/env python3
"""Run induced-fit docking (IFD) on the top N candidates and update CSV (D2).

Loads output/top_candidates.csv, builds CompoundRecord objects with the best
active-site docked pose for each candidate, runs utils.ifd.run_ifd_orchestration
(OpenMM pocket minimisation + Vina re-docking, 3 iterations), and writes the
IFD energies back into the IFD_Energy column of the CSV.

Usage:
    python scripts/run_ifd_top20.py [--count 20]
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.docking import find_best_pose_pdbqt
from utils.library_gen import CompoundRecord
from utils.structure_prep import compute_residue_centroid

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("ifd_top20")

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "output" / "top_candidates.csv"
RECEPTOR_PDB = REPO / "output" / "workdir" / "PBP2a_holo_clean.pdb"
WORK_DIR = str(REPO / "output" / "workdir")
ACTIVE_SITE_RESIDUES = ["SER403", "LYS406", "TYR446"]


def main():
    parser = argparse.ArgumentParser(description="Run IFD on top candidates (D2)")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of top candidates to process")
    parser.add_argument("--box", type=float, default=20.0,
                        help="Isotropic grid box half-size (Å)")
    args = parser.parse_args()

    active_center = compute_residue_centroid(
        str(RECEPTOR_PDB), ACTIVE_SITE_RESIDUES, use_ca=False
    )
    active_center = np.asarray(active_center, dtype=float)
    box = (args.box, args.box, args.box)
    log.info(f"  Active center: {active_center}, box: {box}")

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    log.info(f"  Loaded {len(rows)} candidates from CSV; processing top {args.count}")

    records = []
    for row in rows[: args.count]:
        cid = row["Compound_ID"]
        rec = CompoundRecord(compound_id=cid, smiles=row["SMILES"])
        rec.active_docked_pdbqt = find_best_pose_pdbqt(cid, WORK_DIR)
        if rec.active_docked_pdbqt is None:
            log.warning(f"    {cid}: no docked pose found — skipped")
        records.append(rec)

    from utils.ifd import run_ifd_orchestration

    results = run_ifd_orchestration(
        records=records,
        receptor_pdb=str(RECEPTOR_PDB),
        active_center=tuple(active_center),
        active_box=box,
        work_dir=WORK_DIR,
        n_iterations=3,
        output_dir=str(REPO / "output"),
    )

    ifd_map = {}
    for rec in results:
        if getattr(rec, "ifd_energy", None) is not None:
            ifd_map[rec.compound_id] = rec.ifd_energy
            log.info(f"    {rec.compound_id}: IFD energy = {rec.ifd_energy:.2f} kcal/mol")
        else:
            log.warning(f"    {rec.compound_id}: IFD failed (no energy)")

    fieldnames = list(rows[0].keys())
    if "IFD_Energy" not in fieldnames:
        fieldnames.append("IFD_Energy")
    for row in rows:
        row["IFD_Energy"] = (
            f"{ifd_map[row['Compound_ID']]:.2f}"
            if row["Compound_ID"] in ifd_map else "N/A"
        )

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"  Updated {CSV_PATH}: {len(ifd_map)} candidates with IFD energies")
    log.info(f"  Results: {ifd_map}")
    sys.exit(0)


if __name__ == "__main__":
    main()
