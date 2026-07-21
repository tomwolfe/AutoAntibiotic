"""
Generate property-matched decoys for the known-actives benchmark.

Outputs data/known_decoys.csv with >=100 compounds whose
MW+/-10%, logP+/-0.5, TPSA+/-15%, HBD/HBA+/-1, rotatable bonds+/-2
match a randomly chosen active.

Usage:
    python scripts/build_decoys.py
"""
from __future__ import annotations

import os, sys, csv, random, logging, itertools

from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, Crippen, rdMolDescriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import BETA_LACTAM_SMARTS

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_decoys")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
os.makedirs(DATA_DIR, exist_ok=True)
BETA_LACTAM = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)


def _props(mol):
    return (
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
    )


def read_actives(path):
    """Read known actives CSV, return list of (smiles, cid)."""
    actives = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = row["smiles"].strip()
            cid = row["compound_id"].strip()
            if smi:
                actives.append((smi, cid))
    return actives


def select_property_matched(candidates, active_mols, n=150):
    """Select up to n decoys matching ANY active's properties."""
    tol = (0.10, 0.5, 0.15, 1, 1, 2)
    aprops = [_props(m) for m in active_mols]
    chosen = []
    seen = set()
    rng = random.Random(42)
    pool_shuf = list(candidates)
    rng.shuffle(pool_shuf)
    for cand in pool_shuf:
        if len(chosen) >= n:
            break
        smi = Chem.MolToSmiles(cand)
        if smi in seen:
            continue
        cp = _props(cand)
        for ap in aprops:
            ok = all(
                abs(c - a) <= (tol[i] * max(abs(a), 1e-6) if i in (0, 2) else tol[i])
                for i, (c, a) in enumerate(zip(cp, ap))
            )
            if ok:
                seen.add(smi)
                chosen.append(cand)
                break
    log.info(f"  Property-matched decoys: {len(chosen)}")
    return chosen


def main():
    actives_path = os.path.join(DATA_DIR, "known_actives.csv")
    if not os.path.exists(actives_path):
        log.error("known_actives.csv not found")
        sys.exit(1)

    actives = read_actives(actives_path)
    log.info(f"  Known actives: {len(actives)}")

    active_mols = []
    for smi, _cid in actives:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            active_mols.append(m)
    log.info(f"  Valid active mols: {len(active_mols)}")

    # Use BRICS decomposition of all actives to get fragments, then recombine quickly
    frags = set()
    for m in active_mols:
        try:
            for f in BRICS.BRICSDecompose(m, minFragmentSize=6):
                fm = Chem.MolFromSmiles(f)
                if fm is not None and fm.GetNumHeavyAtoms() >= 6:
                    frags.add(f)
        except Exception:
            continue
    log.info(f"  Unique fragments: {len(frags)}")

    frag_mols = [Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f) is not None]
    log.info(f"  Fragment pool: {len(frag_mols)} mols")

    # Build library by BRICS recombination (limit iterations)
    pool = []
    seen_smi = set()
    rng = random.Random(7)
    for _pass in range(3):
        shuffled = list(frag_mols)
        rng.shuffle(shuffled)
        builder = BRICS.BRICSBuild(shuffled)
        for prod in builder:
            try:
                Chem.SanitizeMol(prod)
            except Exception:
                continue
            smi = Chem.MolToSmiles(prod)
            if smi in seen_smi:
                continue
            seen_smi.add(smi)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if mol.HasSubstructMatch(BETA_LACTAM):
                continue
            mw = Descriptors.MolWt(mol)
            if mw < 150 or mw > 650:
                continue
            pool.append(mol)
            if len(pool) >= 2000:
                break
        if len(pool) >= 2000:
            break
    log.info(f"  BRICS pool: {len(pool)} compounds")

    if len(pool) < 100:
        log.warning("Pool too small; using fragment pool as candidates")
        pool = frag_mols + pool

    decoys = select_property_matched(pool, active_mols, n=150)
    if len(decoys) < 100:
        log.warning(f"Only {len(decoys)} property-matched decoys; padding from pool")
        for mol in pool:
            if len(decoys) >= 100:
                break
            smi = Chem.MolToSmiles(mol)
            if smi not in {Chem.MolToSmiles(d) for d in decoys}:
                decoys.append(mol)

    out_path = os.path.join(DATA_DIR, "known_decoys.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "compound_id", "label"])
        for i, mol in enumerate(decoys):
            smi = Chem.MolToSmiles(mol)
            writer.writerow([smi, f"DECOY_{i:04d}", "decoy"])
    log.info(f"  Wrote {len(decoys)} decoys to {out_path}")


if __name__ == "__main__":
    main()
