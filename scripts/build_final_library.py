#!/usr/bin/env python3
"""
build_final_library.py — Build the final screening library for PBP2a v5.1.

Reads literature-derived seed libraries and the novel 3D-carboxylic-acid seeds,
deduplicates by canonical SMILES, applies property filters, and writes
data/screen_library_final.csv.

Usage:
    python scripts/build_final_library.py
"""

import os
import sys
import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    _HAVE_SA_SCORER = True
except Exception:
    sascorer = None
    _HAVE_SA_SCORER = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_final_library")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

BETA_LACTAM_SMARTS = "[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1"
lactam_pat = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        log.warning(f"  File not found: {path}")
        return pd.DataFrame(columns=["smiles", "compound_id"])
    df = pd.read_csv(path)
    cols = {c.strip().lower() for c in df.columns}
    if "smiles" not in cols:
        log.warning(f"  Missing 'smiles' column in {path}")
        return pd.DataFrame(columns=["smiles", "compound_id"])
    return df


def canonical_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def passes_filters(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False

    # MW 200-550
    mw = Descriptors.MolWt(mol)
    if mw < 200 or mw > 550:
        return False

    # No beta-lactam
    if lactam_pat and mol.HasSubstructMatch(lactam_pat):
        return False

    # No boron
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 5:
            return False

    # SA score < 4.5
    if _HAVE_SA_SCORER and sascorer is not None:
        try:
            sa = sascorer.calculateScore(mol)
            if sa >= 4.5:
                return False
        except Exception:
            pass

    # QED > 0.3
    try:
        qed = QED.qed(mol)
        if qed <= 0.3:
            return False
    except Exception:
        return False

    return True


def main():
    log.info("Building final screening library…")

    # 1. Read all seed CSVs
    pbp2a_seed = read_csv(os.path.join(DATA_DIR, "pbp2a_focused_seed.csv"))
    allosteric = read_csv(os.path.join(DATA_DIR, "pbp2a_allosteric_library.csv"))
    novel_seed = read_csv(os.path.join(REPO_ROOT, "novel_seed.csv"))

    log.info(f"  pbp2a_focused_seed.csv:       {len(pbp2a_seed)} entries")
    log.info(f"  pbp2a_allosteric_library.csv:  {len(allosteric)} entries")
    log.info(f"  novel_seed.csv:                {len(novel_seed)} entries")

    # 2. Merge and deduplicate by canonical SMILES
    merged = pd.concat([pbp2a_seed, allosteric, novel_seed], ignore_index=True)
    merged["smiles"] = merged["smiles"].astype(str).str.strip()
    merged = merged[merged["smiles"].notna() & (merged["smiles"] != "")]

    # Generate canonical SMILES and deduplicate
    merged["canon"] = merged["smiles"].apply(canonical_smiles)
    merged = merged[merged["canon"] != ""]
    merged = merged.drop_duplicates(subset="canon")
    log.info(f"  After deduplication:           {len(merged)} unique compounds")

    # 3. Apply filters
    merged["passes"] = merged["canon"].apply(passes_filters)
    passed = merged[merged["passes"]].copy()
    log.info(f"  After filters (MW, SA, QED, etc.): {len(passed)} compounds")

    # 4. Generate compound IDs for entries missing them
    def _make_cid(row):
        cid = str(row.get("compound_id", "")).strip()
        if cid and cid.lower() not in ("nan", "none", ""):
            return cid
        return ""
    passed["compound_id"] = passed.apply(_make_cid, axis=1)

    next_id = 1
    out_rows = []
    for _, row in passed.iterrows():
        cid = row["compound_id"]
        if not cid:
            cid = f"FINAL-{next_id:04d}"
            next_id += 1
        out_rows.append({"smiles": row["canon"], "compound_id": cid})

    out_df = pd.DataFrame(out_rows)

    # 5. Write output
    out_path = os.path.join(DATA_DIR, "screen_library_final.csv")
    out_df.to_csv(out_path, index=False)
    log.info(f"  Written: {out_path} ({len(out_df)} compounds)")

    # 6. Summary statistics
    mols = [Chem.MolFromSmiles(s) for s in out_df["smiles"] if Chem.MolFromSmiles(s)]
    if mols:
        mws = [Descriptors.MolWt(m) for m in mols]
        log.info(f"  MW range:     {min(mws):.1f} – {max(mws):.1f} Da")
        log.info(f"  MW mean:      {sum(mws)/len(mws):.1f} Da")
        qeds = [QED.qed(m) for m in mols if QED.qed(m)]
        if qeds:
            log.info(f"  QED range:    {min(qeds):.3f} – {max(qeds):.3f}")
            log.info(f"  QED mean:     {sum(qeds)/len(qeds):.3f}")

    log.info("Done.")


if __name__ == "__main__":
    main()
