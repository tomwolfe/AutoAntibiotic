#!/usr/bin/env python3
"""Build the diverse PBP2a library CSV for the fresh pipeline run."""

import os
import sys
import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.library_gen import (
    generate_candidate_library,
    _HAVE_SA_SCORER, sascorer,
)
from config.constants import BETA_LACTAM_SMARTS

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
log = logging.getLogger("build_diverse_library")

BETA_LACTAM_PATTERN = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)
OUTPUT_CSV = os.path.join("data", "screen_library_v3.csv")


def _valid_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def main():
    os.makedirs("data", exist_ok=True)
    seen_canon = set()
    records = []

    def try_add(smi, cid):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return False
        mw = Descriptors.MolWt(mol)
        if mw < 200 or mw > 550:
            return False
        if BETA_LACTAM_PATTERN and mol.HasSubstructMatch(BETA_LACTAM_PATTERN):
            return False
        canon = Chem.MolToSmiles(mol)
        if canon in seen_canon:
            return False
        seen_canon.add(canon)
        records.append((canon, cid))
        return True

    # 1. novel_seed.csv (if it exists)
    novel_path = "novel_seed.csv"
    if os.path.exists(novel_path):
        for _, row in pd.read_csv(novel_path).iterrows():
            smi = str(row["smiles"]).strip()
            cid = str(row["compound_id"]).strip()
            if smi and smi.lower() not in ("nan", "none"):
                try_add(smi, cid)
        log.info(f"novel_seed: {len(records)}")
    else:
        log.info("novel_seed.csv not found — skipping.")

    # 2. expanded_seed.csv NEW* entries (if it exists)
    expanded_path = "expanded_seed.csv"
    if os.path.exists(expanded_path):
        for _, row in pd.read_csv(expanded_path).iterrows():
            smi = str(row["smiles"]).strip()
            cid = str(row["compound_id"]).strip()
            if smi and cid.startswith("NEW") and smi.lower() not in ("nan", "none"):
                try_add(smi, cid)
        log.info(f"+ NEW entries: {len(records)}")
    else:
        log.info("expanded_seed.csv not found — skipping.")

    # 3. Generate BRICS with relaxed SA (< 5.0) for max diversity, then apply
    #    final SA < 4.5 on output. Also relax MW to 180-600 during gen.
    log.info("Generating BRICS recombinants...")
    brics = generate_candidate_library(target_count=500)
    log.info(f"BRICS generated {len(brics)} raw")
    b_added = 0
    for rec in brics:
        mol = _valid_mol(rec.smiles)
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)
        if mw < 180 or mw > 600:
            continue
        if BETA_LACTAM_PATTERN and mol.HasSubstructMatch(BETA_LACTAM_PATTERN):
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen_canon:
            seen_canon.add(canon)
            records.append((canon, f"DIV-{len(records):04d}"))
            b_added += 1
    log.info(f"+ BRICS: {b_added} (total: {len(records)})")

    # Apply strict SA < 4.5 filter to the final set
    final = []
    for smi, cid in records:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)
        if mw < 200 or mw > 550:
            continue
        if BETA_LACTAM_PATTERN and mol.HasSubstructMatch(BETA_LACTAM_PATTERN):
            continue
        if _HAVE_SA_SCORER and sascorer is not None:
            try:
                sa = float(sascorer.calculateScore(mol))
                if sa >= 4.5:
                    continue
            except Exception:
                pass
        final.append((smi, cid))

    log.info(f"After SA < 4.5 filter: {len(final)}")

    # Write CSV
    df = pd.DataFrame(final, columns=["smiles", "compound_id"])
    df = df.drop_duplicates(subset=["smiles"])
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Wrote {len(df)} compounds to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
