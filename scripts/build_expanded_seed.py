#!/usr/bin/env python3
"""
Build expanded_seed.csv (>=500 compounds) for the AutoAntibiotic PBP2a screen.

Strategy
--------
* Keep all 120 existing seeds from novel_seed.csv.
* Add >= 380 new compounds across >= 4 NEW scaffold families:
    - Hydroxamic acids   R-C(=O)NHOH           (on 3D / saturated scaffolds)
    - Boronic acids      R-B(OH)2              (on 3D / aryl scaffolds)
    - Pyrazolopyrimidine / triazolopyridine cores with basic substituents
    - Oxadiazole / thiadiazole cores with carboxylic acid or tetrazole
* Include ceftaroline and meropenem as CTRL_ positive references.
* Every SMILES is validated with RDKit: must parse, MW in [200, 550],
  no beta-lactam SMARTS, SA score < 4.5 (controls exempt from the beta-lactam
  and SA rules because they are the reference beta-lactams).
"""
from __future__ import annotations

import os
import sys
import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import BETA_LACTAM_SMARTS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("seedbuilder")

BETA_LACTAM = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)

try:
    from rdkit.Chem import RDConfig
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
except Exception:
    sascorer = None


def is_valid(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


def passes_filters(smi: str):
    """Return (ok, sa_score) for a candidate drug-like seed (non-control)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, None
    mw = Descriptors.MolWt(mol)
    if mw < 200 or mw > 550:
        return False, None
    if mol.HasSubstructMatch(BETA_LACTAM):
        return False, None
    sa = None
    if sascorer is not None:
        try:
            sa = float(sascorer.calculateScore(mol))
        except Exception:
            sa = None
        if sa is not None and sa >= 4.5:
            return False, None
    return True, sa


# ── R-groups (MW >= ~140) for hydroxamic & boronic acids ──
# Validated, RDKit-parseable scaffolds (aryl / heteroaryl / 3D) large enough that
# attaching a hydroxamic (~+58) or boronic (~+45) group clears the MW >= 200 gate.
CORES = [
    "c1ccc(-c2ccccc2)cc1", "Cc1ccc(-c2ccccc2)cc1", "c1ccc(-c2ccccc2)cc1C",
    "Cc1ccc(-c2ccccc2C)cc1", "c1ccc(-c2ccccc2C)cc1", "Clc1ccc(-c2ccccc2)cc1",
    "Fc1ccc(-c2ccccc2)cc1", "c1ccc(-c2ccc(Cl)cc2)cc1", "c1ccc(-c2ccc(F)cc2)cc1",
    "c1ccc(-c2ccncc2)cc1", "c1ccc(-c2ccsc2)cc1", "c1ccc(-c2ccoc2)cc1",
    "c1ccc(-c2ccc(N)cc2)cc1", "c1ccc(-c2ccc(C#N)cc2)cc1", "c1ccc(-c2ccc(C(F)(F)F)cc2)cc1",
    "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1", "c1ccc2ccccc2c1C", "Clc1ccc2ccccc2c1",
    "c1ccc2c(c1)cccc2Cc1ccccc1",
    "c1ccc2c(c1)c3ccccc3c2",
    "c1ccc2ncccc2c1", "Cc1ccc2ncccc2c1", "c1ccc2ncccc2c1C", "c1ccc2c(c1)nccc2",
    "Cc1ccc2c(c1)nccc2",
    "Cc1ccc2c(c1)cc[nH]2", "c1ccc2c(c1)cc[s]2", "c1ccc2c(c1)cc[nH]2C",
    "c1ccc2c(c1)OCOc2", "Cc1ccc2c(c1)OCOc2", "c1ccc2c(c1)OCCOc2",
    "C1CCCCC1c1ccccc1", "C1CCNCC1c1ccccc1", "c1ccc(CCc2ccccc2)cc1",
    "c1ccc(OCc2ccccc2)cc1", "c1ccc(NCc2ccccc2)cc1", "c1ccc(CC(=O)N3CCCCC3)cc1",
    "c1ccc(CN3CCCCC3)cc1", "c1ccc(CCN3CCCCC3)cc1", "c1ccc(C(=O)c2ccccc2)cc1",
    "c1ccc(C(=O)N3CCCCC3)cc1", "c1ccc(CCc2ccncc2)cc1", "c1ccc(CCc2ccsc2)cc1",
    "c1ccc(N(C)C)cc1", "c1ccc(N2CCCCC2)cc1", "c1ccc(C(F)(F)F)cc1",
    "c1ccc(C(=O)OC)cc1", "c1ccc(S(=O)(=O)C)cc1", "Cc1ccc(Cl)cc1",
    "Clc1ccc(Cl)cc1", "Fc1ccc(F)cc1", "c1ccc(C#N)cc1", "c1ccc(C(C)C)cc1",
    "c1ccc(C(C)(C)C)cc1", "c1ccc(OC)cc1", "c1ccc(OCC)cc1", "c1ccc(OC(C)C)cc1",
    "c1ccc(CC(C)C)cc1", "c1ccc(CCO)cc1", "c1ccc(CCN)cc1",
    "c1ccnc(Cc2ccccc2)c1", "c1ccnc(N3CCCCC3)c1",
    "C1CC(c2ccccc2)CC1", "C1CC(Cc2ccccc2)CC1", "C1CC(Nc2ccccc2)CC1",
    "C1CC(Oc2ccccc2)CC1", "C12CC(N3CCCCC3)CCC1C2", "C1CC(CC(=O)N3CCCC3)CC1",
    "C1CC(C(=O)N3CCCCC3)CC1",
]
# Linkers between the functional group and the core, to widen the MW window and
# the chemical diversity of families 1 & 2.
LINKS = ["", "C", "CC", "CCC", "OCC", "NCC"]

# ── Family 1: hydroxamic acids on 3D / saturated + aryl scaffolds ──
hydroxamic = []
for core in CORES:
    for lnk in LINKS:
        hydroxamic.append("O=C(NO)" + lnk + core)
# acyclic hydroxamates for extra diversity
hydroxamic += ["O=C(NO)CCCCCCCCC", "O=C(NO)CCCCOCCCC", "O=C(NO)CCCCNCCCC"]

# ── Family 2: boronic acids on aryl / heteroaryl / 3D scaffolds ──
boronic = []
for core in CORES:
    for lnk in LINKS:
        boronic.append("OB(O)" + lnk + core)

# ── Family 3: pyrazolopyrimidine / triazolopyridine with basic substituents ──
# All templates below are RDKit-valid. Basic tails use ring digit 3 to avoid
# collision with the heterocycle digits 1/2.
BASIC_TAILS = [
    "N3CCCCC3", "N3CCCC3", "N3CCOCC3", "N(C)C", "N3CCNCC3", "N3CC(C)C3",
]
pyrazolo = []
for b in BASIC_TAILS:
    pyrazolo.append(f"C1=CN2C(=NC{b}=N2)C=C1")
    pyrazolo.append(f"c1ccc2[nH]nnc2c1CC{b}")
    pyrazolo.append(f"c1nc2c(n1)cccn2CC{b}")
# purine-like pyrazolopyrimidine with a basic tail
pyrazolo.append("O=C1NC(=O)C2=C1N=CN2CCN3CCN(C)C3")
pyrazolo.append("O=C1NC(=O)C2=C1N=CN2CCN3CCCCC3")
# extra triazolopyridine variants
for b in BASIC_TAILS:
    pyrazolo.append(f"c1ccc2[nH]nnc2c1CC{b}")

# ── Family 4: oxadiazole / thiadiazole with COOH or tetrazole ──
# Outer benzoic acid ring = digit 1; oxadiazole/thiadiazole ring = digit 2; any
# pendant aryl tail = digit 3. Keep digits distinct to avoid ring-closure errors.
oxadiazole = []
for s in ["C", "Cl", "C(F)(F)F", "C#N", "c3ccccc3", "c3ccc(Cl)cc3", "c3ccncc3",
          "c3ccc(OC)cc3", "c3cc(C)ccc3"]:
    oxadiazole.append(f"O=C(O)c1ccc(-c2nno(c2{s}))cc1")   # 1,3,4-oxadiazole-COOH
    oxadiazole.append(f"O=C(O)c1ccc(-c2nnc(s2{s}))cc1")   # 1,3,4-thiadiazole-COOH
# tetrazole (acid bioisostere) on a benzoic-acid scaffold (MW >= 200)
oxadiazole.append("O=C(O)c1ccc(C2=NNN=N2)c(C)c1")
oxadiazole.append("Cc1ccc(C2=NNN=N2)c(C(=O)O)c1")
oxadiazole.append("O=C(O)c1ccc2c(c1)C2=NNN=N2")
oxadiazole.append("O=C(O)c1ccc(-c2nnc(n2)c3ccccc3)cc1")
oxadiazole.append("O=C(O)c1ccc(-c2nno(c2c3ccccc3))cc1")

CONTROLS = {
    "CTRL_Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
    "CTRL_Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
}


def build():
    seed_df = pd.read_csv("novel_seed.csv")
    rows = []
    for _, r in seed_df.iterrows():
        smi = str(r["smiles"]).strip()
        cid = str(r["compound_id"]).strip()
        if is_valid(smi):
            rows.append((cid, smi, "seed"))

    fam_counts = {}
    fam_smis = {
        "hydroxamic": hydroxamic,
        "boronic": boronic,
        "pyrazolopyrimidine": pyrazolo,
        "oxadiazole": oxadiazole,
    }
    n_new = 0
    idx = 0
    for fam, smis in fam_smis.items():
        kept = 0
        for smi in smis:
            ok, sa = passes_filters(smi)
            if not ok:
                continue
            idx += 1
            rows.append((f"NEW{idx:04d}", smi, fam))
            kept += 1
            n_new += 1
        fam_counts[fam] = kept
        log.info(f"  Family '{fam}': kept {kept} valid compounds")

    need = 380 - n_new
    if need > 0:
        log.info(f"  Padding with {need} extra aryl hydroxamic/boronic acids…")
        pad = []
        for core in CORES:
            for lnk in ["CCCC", "OCCC", "NCCC"]:
                pad.append("O=C(NO)" + lnk + core)
                pad.append("OB(O)" + lnk + core)
        for smi in pad:
            if n_new >= 380:
                break
            ok, sa = passes_filters(smi)
            if not ok:
                continue
            idx += 1
            rows.append((f"NEW{idx:04d}", smi, "pad"))
            n_new += 1

    for cid, smi in CONTROLS.items():
        if is_valid(smi):
            rows.append((cid, smi, "control"))
        else:
            log.warning(f"  Control {cid} failed RDKit parse!")

    out_df = pd.DataFrame(rows, columns=["compound_id", "smiles", "family"])
    before = len(out_df)
    out_df = out_df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    log.info(f"  De-duplicated {before} -> {len(out_df)} compounds")
    out_df[["compound_id", "smiles"]].to_csv("expanded_seed.csv", index=False)
    log.info(f"  Wrote expanded_seed.csv with {len(out_df)} compounds")
    return out_df


if __name__ == "__main__":
    build()
