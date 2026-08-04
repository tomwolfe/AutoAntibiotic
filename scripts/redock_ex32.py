#!/usr/bin/env python3
"""
Re-dock the top hits at exhaustiveness=32 against all three PBP2a conformers.

The primary screen ran at exhaustiveness~8 (energy noise ~±2 kcal/mol), so the
1.16 kcal/mol spread among the top hits was within the noise window and the hits
were indistinguishable. This script re-docks the top-N candidates at the
recommended exhaustiveness=32 with num_modes=9 against the full PBP2a conformer
ensemble (1VQQ apo, 3ZG0 holo, 4DKI holo), and records the best (most negative)
energy per compound in a new ``PBP2a_Active_Energy_E32`` column of
output/top_candidates.csv.

Usage:
    AUTOANTIBIOTIC_MODE=science python scripts/redock_ex32.py [--count 50]

Outputs:
    Updates output/top_candidates.csv / .json with PBP2a_Active_Energy_E32.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import discovery_pipeline as P
from config.constants import ACTIVE_BOX_SIZE, ACTIVE_SITE_RESIDUES
from utils.library_gen import CompoundRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("redock_ex32")

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "output" / "top_candidates.csv"
WORK_DIR = str(REPO / "output" / "workdir")


def main():
    parser = argparse.ArgumentParser(description="Re-dock top hits at exhaustiveness=32 (Phase 3)")
    parser.add_argument("--count", type=int, default=50, help="number of top candidates")
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--num-modes", type=int, default=9)
    args = parser.parse_args()

    config = P.load_config()
    config["mode"] = "science"
    deps = P.check_dependencies()
    if not deps["USE_VINA"]:
        log.error("Vina required. Aborting.")
        sys.exit(1)

    targets = P.prepare_targets(str(REPO / "output" / "pdb"), WORK_DIR, deps, config=config)
    pb2pa = targets["PBP2a"]
    receptor_pdbqts = [r for r in pb2pa["receptor_pdbqts"] if r]
    active_center = np.asarray(pb2pa["active_center"], dtype=float)
    cleaned_pdb = pb2pa["cleaned_pdb"]
    active_box = P._auto_box_size(cleaned_pdb, active_center, ACTIVE_BOX_SIZE,
                                  min_size=15.0, max_size=20.0,
                                  site_residues=ACTIVE_SITE_RESIDUES)
    log.info(f"  {len(receptor_pdbqts)} PBP2a conformers; box {active_box} center {active_center}")

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    col = "PBP2a_Active_Energy_E32"
    if col not in fieldnames:
        fieldnames.append(col)

    from utils.docking import dock_compound
    from functools import partial

    target = rows[: args.count]
    for i, row in enumerate(target, 1):
        cid = row["Compound_ID"]
        rec = CompoundRecord(compound_id=cid, smiles=row["SMILES"])
        best = None
        for conf_idx, rec_pdbqt in enumerate(receptor_pdbqts):
            e = dock_compound(
                rec, rec_pdbqt, active_center, active_box,
                WORK_DIR, tag=f"e32_c{conf_idx}",
                exhaustiveness=args.exhaustiveness, num_modes=args.num_modes,
            )
            if e is not None and (best is None or e < best):
                best = e
        row[col] = f"{best:.2f}" if best is not None else "N/A"
        log.info(f"    {i}/{len(target)} {cid}: E32 best = {row[col]} kcal/mol")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    json_path = CSV_PATH.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    log.info(f"  Updated {CSV_PATH} with {col} for {len(target)} candidates")
    sys.exit(0)


if __name__ == "__main__":
    main()