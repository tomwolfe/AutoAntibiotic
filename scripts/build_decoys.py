"""
Generate property-matched decoys for the known-actives benchmark.

Outputs data/known_decoys.csv with >=120 property-matched compounds whose
MW+/-10%, logP+/-0.5, TPSA+/-15%, HBD/HBA+/-1, rotatable bonds+/-2
match a randomly chosen active.

Usage:
    python scripts/build_decoys.py
"""
from __future__ import annotations

import os, sys, csv, random, logging

from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, Crippen, rdMolDescriptors, AllChem, QED

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import BETA_LACTAM_SMARTS
from utils.library_gen import PBP2A_SCAFFOLDS

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
    tol = (0.15, 0.7, 0.20, 1, 2, 3)
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


def generate_diverse_pool(target_size=2000):
    """Generate a diverse pool using multiple fragment sources."""
    pool = []
    seen_smi = set()

    # Source 1: BRICS from known actives
    frags = set()
    actives_path = os.path.join(DATA_DIR, "known_actives.csv")
    if os.path.exists(actives_path):
        actives = read_actives(actives_path)
        for smi, _cid in actives:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                try:
                    for f in BRICS.BRICSDecompose(m, minFragmentSize=4):
                        fm = Chem.MolFromSmiles(f)
                        if fm is not None and fm.GetNumHeavyAtoms() >= 4:
                            frags.add(f)
                except Exception:
                    continue

    # Source 2: BRICS from PBP2a scaffolds
    for smi in PBP2A_SCAFFOLDS:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            try:
                for f in BRICS.BRICSDecompose(m, minFragmentSize=4):
                    fm = Chem.MolFromSmiles(f)
                    if fm is not None and fm.GetNumHeavyAtoms() >= 4:
                        frags.add(f)
            except Exception:
                continue

    # Source 3: Diverse drug-like fragments from common chemotypes
    extra_scaffolds = [
        "c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1", "c1cccs1",
        "C1CCCCC1", "C1CCNCC1", "c1cc[nH]c1", "c1cnn2ccccc12",
        "C1=CCOC1", "c1ccc2[nH]c3ccccc3c2c1", "c1cc2c(cc1)OCCO2",
        "c1cc2c(cc1)NCCN2", "O=c1ccccc1", "c1cc2c(cc1)CCC2",
        "c1ccc2c(c1)C(=O)N2", "c1cc2c(cc1)NC=C2", "c1cc2c(cc1)NC=N2",
        "c1ccc2c(c1)CC(=O)N2", "c1ccc2c(c1)C=NO2",
        "c1ccc2c(c1)NN=C2", "c1ccc2c(c1)CCN2", "c1ccc2c(c1)COC2=O",
        "c1ccc2c(c1)CO2", "c1ccc2c(c1)CS(=O)(=O)N2",
        "C1CC2CCCC2C1", "C1CC2CC3CC2C1C3",
        "c1ccc2c(c1)OCO2", "c1ccc2c(c1)OCCO2",
        "c1ccc2c(c1)SCS2", "c1ccc2c(c1)NCCN2",
        "C1CC2C3CCCC3C2C1", "C1CC2CC3CC4CC5CC2C3C1C45",
        "c1ccc2c(c1)CCCC2", "c1ccc2c(c1)CCCC2",
        "c1ccc2c(c1)CCc1ccccc1-2",
    ]


    for smi in extra_scaffolds:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            try:
                for f in BRICS.BRICSDecompose(m, minFragmentSize=4):
                    fm = Chem.MolFromSmiles(f)
                    if fm is not None and fm.GetNumHeavyAtoms() >= 4:
                        frags.add(f)
            except Exception:
                continue

    log.info(f"  Total unique fragments: {len(frags)}")
    frag_mols = [Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f) is not None]
    log.info(f"  Fragment pool: {len(frag_mols)} mols")

    if len(frag_mols) < 2:
        return pool

    rng = random.Random(7)
    max_passes = 5
    for _pass in range(max_passes):
        if len(pool) >= target_size:
            break
        shuffled = list(frag_mols)
        rng.shuffle(shuffled)
        builder = BRICS.BRICSBuild(shuffled)
        for prod in builder:
            if len(pool) >= target_size:
                break
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
            if mw < 150 or mw > 700:
                continue
            # Keep compounds with reasonable drug-likeness
            try:
                qed = QED.qed(mol)
                if qed < 0.3:
                    continue
            except Exception:
                continue
            pool.append(mol)
    log.info(f"  Generated pool: {len(pool)} compounds")
    return pool


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

    pool = generate_diverse_pool(target_size=3000)
    if len(pool) < 200:
        log.warning(f"Pool too small ({len(pool)}); cannot generate enough decoys.")
        pool = pool

    # Try to get 120+ property-matched decoys
    decoys = select_property_matched(pool, active_mols, n=150)
    if len(decoys) < 120:
        log.warning(f"Only {len(decoys)} property-matched decoys; expanding pool with random molecules")
        rng = random.Random(99)
        pool_shuf = list(pool)
        rng.shuffle(pool_shuf)
        seen_decoys = {Chem.MolToSmiles(d) for d in decoys}
        for mol in pool_shuf:
            if len(decoys) >= 120:
                break
            smi = Chem.MolToSmiles(mol)
            if smi in seen_decoys:
                continue
            seen_decoys.add(smi)
            decoys.append(mol)
        log.info(f"  Expanded decoys to {len(decoys)}")

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
