from __future__ import annotations

import os
import json
import hashlib
import logging
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, rdMolDescriptors, Descriptors, AllChem, DataStructs, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from config.constants import RANDOM_SEED, BETA_LACTAM_SMARTS, SIMILARITY_THRESHOLD

try:
    from rdkit.Chem import RDConfig
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    _HAVE_SA_SCORER = True
except Exception:
    sascorer = None
    _HAVE_SA_SCORER = False

log = logging.getLogger("AutoAntibiotic")


@dataclass
class CompoundRecord:
    CONF_HIGH = "High"
    CONF_LOW = "Low"
    CONF_NONE = "None"

    compound_id: str
    smiles: str
    mol: Optional[Chem.Mol] = None

    pb2pa_active_energy: Optional[float] = None
    pb2pa_allosteric_energy: Optional[float] = None
    pb2pa_best_energy: Optional[float] = None
    binding_site: str = ""
    human_trypsin_energy: Optional[float] = None
    human_ces1_energy: Optional[float] = None
    selectivity_index: Optional[float] = None
    off_target_risk: bool = False
    human_offtarget_max_energy: Optional[float] = None
    max_similarity: float = 0.0
    passes_lipinski: bool = False
    qed_score: float = 0.0
    passes_pains: bool = False
    resistance_notes: str = ""
    selectivity_confidence: str = "None"
    active_docked_pdbqt: Optional[str] = None
    interactions: Optional[dict] = None
    si_vs_ceftaroline: Optional[float] = None
    report_tier: Optional[str] = None
    sa_score: Optional[float] = None
    tpsa: Optional[float] = None
    frac_csp3: Optional[float] = None
    num_rotatable_bonds: Optional[int] = None
    suspect_score: bool = False
    si_provisional: Optional[float] = None


SEED_SCAFFOLDS = [
    # Troczi 2013 oxadiazole cores
    "O=C(Nc1ccc(-c2nc(C3=CC=C(O)C=C3)no2)cc1)c1ccccc1",
    "O=C(O)c1ccc(C2=NOC(=N2)c3ccc(NC(=O)c4ccco4)cc3)cc1",
    "CCCC(=O)Nc1ccc(-c2nc(C3=CC=C(C(=O)O)C=C3)no2)cc1",
    # Quinazolinone scaffold (Bouley 2015)
    "N#Cc1ccc(NC(=O)N2C(c3ccccc3)=NC(c3cccc(C(=O)O)c3)=Nc3ccccc32)cc1",
    # Penicillin-derived non-beta-lactam core
    "CC1(C)SC2C(NC(=O)Cc3ccc(O)cc3)C(=O)N2C1C(=O)O",
    # Diverse heterocyclic cores
    "O=c1[nH]c2ccccc2n1-c1ccccc1",
    "O=c1cc(-c2ccccc2)nc2[nH]ccn12",
    "O=C1C=C(c2ccccc2)n2ccnc21",
    "Cc1ccc(C(=O)Nc2ccc(-c3cc(nn3C)c4ccc(F)cc4)cc2)cc1",
]

CONTROL_SMILES = {
    "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
    "POS_QUIN01": "N#Cc1ccc(NC(=O)N2C(c3ccccc3)=NC(c3cccc(C(=O)O)c3)=Nc3ccccc32)cc1",
    "POS_ODAN01": "CC1(C)SC2C(NC(=O)Cc3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "POS_PYRZ01": "Cc1ccc(C(=O)Nc2ccc(-c3cc(nn3C)c4ccc(F)cc4)cc2)cc1",
    "POS_DIOSM01": "CC1C(C(C(O1)Oc2ccc3c(c2)C(=O)C=C(O3)c4ccc(O)c(OC)c4)O)O",
}

CEFTAROLINE_SMILES = CONTROL_SMILES["Ceftaroline"]


def _enrich_record_properties(rec: CompoundRecord) -> CompoundRecord:
    mol = rec.mol if rec.mol is not None else Chem.MolFromSmiles(rec.smiles)
    if mol is None:
        return rec
    try:
        if _HAVE_SA_SCORER and sascorer is not None:
            rec.sa_score = float(sascorer.calculateScore(mol))
    except Exception:
        pass
    try:
        rec.tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        pass
    try:
        rec.frac_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    except Exception:
        pass
    try:
        rec.num_rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:
        pass
    return rec


def _passes_hard_filters(mol: Chem.Mol) -> bool:
    mw = Descriptors.MolWt(mol)
    if mw < 250 or mw > 500:
        return False

    if _HAVE_SA_SCORER and sascorer is not None:
        try:
            sa = sascorer.calculateScore(mol)
            if sa >= 5.0:
                return False
        except Exception:
            pass

    try:
        qed_val = QED.qed(mol)
        if qed_val <= 0.5:
            return False
    except Exception:
        return False

    lactam_pat = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)
    if lactam_pat and mol.HasSubstructMatch(lactam_pat):
        return False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 5:
            return False

    hbd = Descriptors.NumHDonors(mol)
    if hbd < 1:
        return False

    return True


def _make_fragment_pool() -> List[Chem.Mol]:
    """Create diverse fragment pool from seeds + drug-like fragments."""
    all_smiles = set()

    for smi in SEED_SCAFFOLDS:
        all_smiles.add(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                for f_smi in BRICS.BRICSDecompose(mol, minFragmentSize=4):
                    fm = Chem.MolFromSmiles(f_smi)
                    if fm and fm.GetNumHeavyAtoms() >= 4:
                        all_smiles.add(f_smi)
            except Exception:
                pass

    for smi in CONTROL_SMILES.values():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                for f_smi in BRICS.BRICSDecompose(mol, minFragmentSize=4):
                    fm = Chem.MolFromSmiles(f_smi)
                    if fm and fm.GetNumHeavyAtoms() >= 4:
                        all_smiles.add(f_smi)
            except Exception:
                pass

    diverse_fragments = [
        "c1ccc2ccccc2c1", "c1ccccc1", "c1ccncc1", "c1cccs1",
        "C1CCCCC1", "C1CCNCC1", "c1cc[nH]c1", "c1cnn2ccccc12",
        "c1ccc2[nH]c3ccccc3c2c1", "c1cc2c(cc1)OCCO2",
        "c1ccc2c(c1)C(=O)N2", "c1cc2c(cc1)NC=C2",
        "c1ccc2c(c1)CC(=O)N2", "c1ccc2c(c1)C=NO2",
        "c1ccc2c(c1)NN=C2", "c1ccc2c(c1)CCC2",
        "c1ccc2c(c1)COC2=O", "c1ccc2c(c1)CO2",
        "c1ccc2c(c1)OCO2", "c1ccc2c(c1)OCCO2",
        "C1CC2CCCC2C1", "C1CC2CC3CC2C1C3",
        "c1ccc2c(c1)CCCC2", "CC(=O)O", "C(=O)O",
        "C(=O)N", "CN", "CO",
        "c1ccc(O)cc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1",
        "c1ccc(C(=O)O)cc1", "c1cc(Cl)cc(Cl)c1",
        "CC(=O)Nc1ccccc1", "CN(C)c1ccccc1",
        "c1ccc(S(=O)(=O)N)cc1",
        "c1ccc2c(c1)CCN2", "c1ccc2c(c1)COC2=O",
        "c1cc2c(cc1)CCC2", "C1CC2CC3CC2C1C3",
    ]
    for smi in diverse_fragments:
        all_smiles.add(smi)

    return [Chem.MolFromSmiles(s) for s in all_smiles if Chem.MolFromSmiles(s)]


def generate_candidate_library(
    target_count: int = 500,
    seed: int = RANDOM_SEED,
    input_csv: Optional[str] = None,
    seed_smiles: Optional[List[str]] = None,
) -> List[CompoundRecord]:
    log.info("─── Phase 2: Library Generation ───")

    if input_csv is not None:
        log.info(f"  Loading external compound library from CSV: {input_csv}")
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input library CSV not found: {input_csv}")

        df = pd.read_csv(input_csv)
        df_cols = {str(c).strip().lower() for c in df.columns}
        if not {"smiles", "compound_id"}.issubset(df_cols):
            raise ValueError(
                f"Input CSV must contain 'smiles' and 'compound_id' columns; "
                f"found: {list(df.columns)}"
            )

        records = []
        for _, row in df.iterrows():
            smi = str(row["smiles"]).strip()
            cid = str(row["compound_id"]).strip()
            if not smi or smi.lower() in ("nan", "none"):
                continue
            mol = Chem.MolFromSmiles(smi)
            records.append(CompoundRecord(
                compound_id=cid,
                smiles=smi,
                mol=mol,
            ))
        log.info(f"  Loaded {len(records)} compounds from external CSV.")
        return [_enrich_record_properties(r) for r in records]

    # Build fragment pool from seeds + diverse fragments
    frag_mols = _make_fragment_pool()
    log.info(f"  Fragment pool: {len(frag_mols)} unique fragments.")

    if len(frag_mols) < 2:
        log.warning("  Too few fragments.")
        return []

    # Multi-pass BRICSBuild
    import random as _random
    _random.seed(seed)
    seen_smiles = set()
    records = []

    max_passes = min(10, max(2, target_count // max(1, len(frag_mols) // 2)))
    for _pass in range(max_passes):
        if len(records) >= target_count:
            break
        records_before = len(records)
        shuffled = list(frag_mols)
        _random.shuffle(shuffled)
        builder = BRICS.BRICSBuild(shuffled)
        for product in builder:
            try:
                Chem.SanitizeMol(product)
            except Exception:
                continue
            smi = Chem.MolToSmiles(product)
            if smi in seen_smiles:
                continue
            if not smi or len(smi) < 10:
                continue
            if not _passes_hard_filters(product):
                continue
            seen_smiles.add(smi)
            cid = f"AA-{len(records):04d}"
            records.append(CompoundRecord(
                compound_id=cid, smiles=smi, mol=product,
            ))
            if len(records) >= target_count:
                break
        if len(records) == records_before:
            log.info(f"  Pass {_pass + 1}: exhausted.")
            break
        log.info(f"  Pass {_pass + 1}: {len(records)} unique so far.")
    log.info(f"  BRICS generated: {len(records)} compounds.")

    # Supplement with seed scaffolds that pass filters
    for smi in SEED_SCAFFOLDS:
        if len(records) >= target_count:
            break
        if smi in seen_smiles:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol and _passes_hard_filters(mol):
            seen_smiles.add(smi)
            records.append(CompoundRecord(
                compound_id=f"AA-{len(records):04d}", smiles=smi, mol=mol,
            ))
    log.info(f"  After adding seeds: {len(records)} compounds.")

    # Load known_actives.csv and add filtered entries
    akt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'known_actives.csv')
    if os.path.exists(akt_path) and len(records) < target_count:
        akt_df = pd.read_csv(akt_path)
        for _, row in akt_df.iterrows():
            if len(records) >= target_count:
                break
            smi = str(row.get('smiles', '')).strip()
            cid = str(row.get('compound_id', '')).strip()
            if not smi or smi.lower() in ('nan', 'none'):
                continue
            if smi in seen_smiles:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if not _passes_hard_filters(mol):
                continue
            seen_smiles.add(smi)
            records.append(CompoundRecord(
                compound_id=cid or f"AA-{len(records):04d}", smiles=smi, mol=mol,
            ))
    log.info(f"  After adding known actives: {len(records)} compounds.")

    # Bemis-Murcko framework cap: no framework > 15% of library
    import random as _bm_random
    _bm_random.seed(seed)
    framework_counts = {}
    framework_indices = {}
    for idx, rec in enumerate(records):
        mol = rec.mol if rec.mol is not None else Chem.MolFromSmiles(rec.smiles)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            framework = Chem.MolToSmiles(scaffold) if scaffold else "none"
        except Exception:
            framework = "none"
        framework_counts[framework] = framework_counts.get(framework, 0) + 1
        framework_indices.setdefault(framework, []).append(idx)
    max_frac = 0.15
    cap = max(1, int(len(records) * max_frac))
    dropped = 0
    for framework, indices in framework_indices.items():
        if len(indices) > cap:
            _bm_random.shuffle(indices)
            keep = indices[:cap]
            drop = set(indices[cap:])
            records = [r for i, r in enumerate(records) if i not in drop]
            dropped += len(drop)
    if dropped:
        log.info(f"  Bemis-Murcko framework cap: removed {dropped} compounds ({max_frac*100:.0f}% limit).")

    # Diversity filter: Tanimoto < SIMILARITY_THRESHOLD within library, < 0.7 to ceftaroline
    ceft_fp = None
    ceft_mol = Chem.MolFromSmiles(CEFTAROLINE_SMILES)
    if ceft_mol:
        ceft_fp = AllChem.GetMorganFingerprintAsBitVect(ceft_mol, radius=2, nBits=2048)

    log.info("  Enforcing diversity filters...")
    diversity_records = []
    diversity_fps = []
    for rec in records:
        if rec.mol is None:
            rec.mol = Chem.MolFromSmiles(rec.smiles)
        if rec.mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(rec.mol, radius=2, nBits=2048)
        if ceft_fp:
            sim_to_ceft = DataStructs.TanimotoSimilarity(fp, ceft_fp)
            if sim_to_ceft >= 0.7:
                continue
        is_diverse = all(
            DataStructs.TanimotoSimilarity(fp, existing_fp) < SIMILARITY_THRESHOLD
            for existing_fp in diversity_fps
        )
        if is_diverse:
            diversity_records.append(rec)
            diversity_fps.append(fp)
    records = diversity_records
    log.info(f"  After diversity filters: {len(records)} candidates.")

    # Add control compounds
    for name, smi in CONTROL_SMILES.items():
        if smi not in seen_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                records.append(CompoundRecord(
                    compound_id=f"CTRL_{name}", smiles=smi, mol=mol,
                ))
                seen_smiles.add(smi)

    log.info(f"  Library generation complete: {len(records)} compounds.")
    return [_enrich_record_properties(r) for r in records]
