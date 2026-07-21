#!/usr/bin/env python3
"""
Build a focused PBP2a inhibitor library.

Reads the validated novel_seed.csv, filters to SA < 4.5, MW 200-550,
no beta-lactam, and deduplicates. Adds controls.

Output CSV suitable for AUTOANTIBIOTIC_LIB_CSV.

Usage:
    python scripts/build_library.py [--count N] [--output LIBRARY.csv]
"""
from __future__ import annotations

import argparse, csv, logging, sys, os
from rdkit import Chem
from rdkit.Chem import Descriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from rdkit.Chem import RDConfig
    _sys = __import__("sys")
    _sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    _HAVE_SA = True
except Exception:
    sascorer = None
    _HAVE_SA = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_library")

BETA_LACTAM = Chem.MolFromSmarts("[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1")

import pandas as pd

CONTROLS = {
    "CTRL_Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
    "CTRL_Meropenem":   "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
}

def sa_score(mol):
    if _HAVE_SA and sascorer is not None:
        try:
            return float(sascorer.calculateScore(mol))
        except Exception:
            return None
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.output or os.path.join(repo_root, "library.csv")

    # Read novel_seed.csv
    seed_path = os.path.join(repo_root, "novel_seed.csv")
    if not os.path.exists(seed_path):
        log.error(f"novel_seed.csv not found at {seed_path}")
        sys.exit(1)
    df = pd.read_csv(seed_path)
    all_smiles = set()
    for smi in df["smiles"]:
        all_smiles.add(str(smi).strip())

    records = []
    seen = set()
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if mol.HasSubstructMatch(BETA_LACTAM):
            continue
        mw = Descriptors.MolWt(mol)
        if mw < 200 or mw > 550:
            continue
        sa = sa_score(mol)
        if sa is not None and sa >= 4.5:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        seen.add(canon)
        cid = f"AA-{len(records):04d}"
        records.append({"compound_id": cid, "smiles": canon})

    # Add controls
    for cid, smi in CONTROLS.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                records.append({"compound_id": cid, "smiles": canon})
                seen.add(canon)

    # Also add boronic/hydroxamic variants
    extra = [
        ("OB(O)CC1C2CC3CC1CC(C2)C3", "BB01"),
        ("OB(O)CC12CC(CC1)(CC2)", "BB02"),
        ("ONC(=O)CC1C2CC3CC1CC(C2)C3", "HA01"),
        ("ONC(=O)CC12CC(CC1)(CC2)", "HA02"),
    ]
    for smi, suffix in extra:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                mw = Descriptors.MolWt(mol)
                sa = sa_score(mol)
                if 200 <= mw <= 550 and (sa is None or sa < 4.5) and not mol.HasSubstructMatch(BETA_LACTAM):
                    records.append({"compound_id": f"AA-{len(records):04d}", "smiles": canon})
                    seen.add(canon)

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["smiles", "compound_id"])
        writer.writeheader()
        written = 0
        for r in records[:args.count]:
            writer.writerow({"smiles": r["smiles"], "compound_id": r["compound_id"]})
            written += 1
    log.info(f"Wrote {written} compounds to {out_path}")

if __name__ == "__main__":
    main()
