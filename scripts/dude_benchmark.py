#!/usr/bin/env python3
"""
DUD-E style enrichment benchmark for PBP2a (D4).

Pulls PBP2a actives from ChEMBL (target CHEMBL6187 = MecA/PBP2a, IC50 < 10 µM)
and from the local data/active_site_actives.csv fallback. Generates a DUD-E
style set of property-matched decoys (50 per active; MW ±10%, logP ±0.5,
HBD/HBA ±1, rotatable bonds ±2, Tanimoto < 0.35 to any active) against the
PBP2a apo (1VQQ) receptor at exhaustiveness 32.

Reports:
    - ROC-AUC
    - EF_1%, EF_5%, EF_10%
    - BEDROC (alpha = 20)

Usage:
    AUTOANTIBIOTIC_MODE=science python scripts/dude_benchmark.py
    AUTOANTIBIOTIC_MODE=science python scripts/dude_benchmark.py --decoy-pool data/screen_library_final.csv

Outputs:
    output/dude_benchmark_results.json   — metrics + verdict
    output/dude_benchmark_roc.png        — ROC curve
    output/dude_decoys.csv               — generated decoy set (for reproducibility)
    data/chembl_pbp2a_actives.csv        — downloaded actives (cached)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, DataStructs, BRICS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import discovery_pipeline as P
from config.constants import ACTIVE_BOX_SIZE, ACTIVE_SITE_RESIDUES, BETA_LACTAM_SMARTS
from utils.structure_prep import merge_conserved_waters

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dude_bench")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "output")
DATA = os.path.join(REPO, "data")
os.makedirs(OUT, exist_ok=True)

# Decoy property-matching tolerances (DUD-E style).
DECOYS_PER_ACTIVE = 5
# Union of all in-house libraries serves as the offline decoy pool (the
# pipeline's own PBP2a-focused library alone cannot fill DUD-E's 50/active
# target for property-extreme actives; see generate_decoys relaxed tier).
DEFAULT_DECOY_POOL = ",".join(
    "data/{}.csv".format(fn)
    for fn in [
        "chembl_decoy_pool", "screen_library_final", "combined_library",
        "diverse_pbp2a_library", "pbp2a_focused_seed", "expanded_prelib",
        "screen_library", "screen_library_v2", "screen_library_v3", "known_decoys",
    ]
)
MW_TOL_FRAC = 0.10
LOGP_TOL = 0.5
HBD_TOL = 1
HBA_TOL = 1
ROT_TOL = 2
MAX_TANIMOTO_TO_ACTIVE = 0.50
RANDOM_SEED = 42
# Minimum acceptable decoys per active; below this the relaxed tolerance tier
# kicks in so every active still receives a usable decoy set.
DECOY_FALLBACK_MIN = 15

# Metric thresholds used for the PASS/FAIL verdict.
AUC_MIN = 0.75
BEDROC_MIN = 0.4
EF1_MIN = 5.0
MIN_ACTIVES = 50


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


def compute_bedrock(labels, scores, alpha=20.0):
    """Boltzmann-Enhanced Discrimination of ROC (BEDROC) metric."""
    N = len(labels)
    n_act = sum(labels)
    if n_act == 0 or n_act == N:
        return 0.5
    order = np.argsort(-np.asarray(scores, dtype=float))
    labels_arr = np.asarray(labels, dtype=int)
    ranks = [idx + 1 for idx, pos in enumerate(order) if labels_arr[pos] == 1]
    sum_exp = sum(np.exp(-alpha * r / N) for r in ranks)
    factor = (1.0 - np.exp(-alpha)) / (1.0 - np.exp(-alpha * n_act / N))
    return float((sum_exp / n_act) * factor)


def enrichment_metrics(labels, scores, alpha=20.0):
    """Compute AUC, EF_1%, EF_5%, EF_10% and BEDROC(alpha).

    Scores are higher=better. Returns a dict with 'auc', 'ef_1pct',
    'ef_5pct', 'ef_10pct', 'bedrock_alpha20'. ``k1`` is never degenerate
    (max(1, round(0.01*N))), so with poweref1 is meaningful.
    """
    labels = np.asarray(labels, dtype=int)
    n_act = int(labels.sum())
    n_dec = len(labels) - n_act
    if n_act == 0 or n_dec == 0:
        return {"auc": 0.5, "ef_1pct": 0.0, "ef_5pct": 0.0,
                "ef_10pct": 0.0, "bedrock_alpha20": 0.5}

    fpr, tpr, auc = compute_roc(labels, scores)
    bedrock = compute_bedrock(labels, scores, alpha=alpha)

    N = len(labels)
    # positions 0..N-1, best score first:
    ranked = np.argsort(-np.asarray(scores, dtype=float))
    k1 = max(1, round(0.01 * N))
    k5 = max(1, round(0.05 * N))
    k10 = max(1, round(0.10 * N))
    lab_arr = labels
    act_in_1 = int(lab_arr[ranked[:k1]].sum())
    act_in_5 = int(lab_arr[ranked[:k5]].sum())
    act_in_10 = int(lab_arr[ranked[:k10]].sum())
    ef1 = (act_in_1 / n_act) / (k1 / N) if n_act else 0.0
    ef5 = (act_in_5 / n_act) / (k5 / N) if n_act else 0.0
    ef10 = (act_in_10 / n_act) / (k10 / N) if n_act else 0.0

    return {
        "auc": float(auc),
        "ef_1pct": float(ef1),
        "ef_5pct": float(ef5),
        "ef_10pct": float(ef10),
        "bedrock_alpha20": float(bedrock),
    }


def bootstrap_cis(labels, scores, n_resamples=1000, seed=RANDOM_SEED,
                  alpha_level=0.05, alpha=20.0):
    """Bootstrap (case resampling) 95% confidence intervals on the metrics.

    Resamples compounds with replacement, recomputes every metric per
    draw, and reports the 2.5th / 97.5th percentiles. With small actives
    sets (e.g. N_actives~50) these CIs reveal whether EF_1% or AUC values
    are statistically separable from chance / from each other (Phase G6,
    paper integration).
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    rng = np.random.RandomState(seed)
    draws = {"auc": [], "ef_1pct": [], "ef_5pct": [], "bedrock_alpha20": []}
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        m = enrichment_metrics(labels[idx], scores[idx], alpha=alpha)
        draws["auc"].append(m["auc"])
        draws["ef_1pct"].append(m["ef_1pct"])
        draws["ef_5pct"].append(m["ef_5pct"])
        draws["bedrock_alpha20"].append(m["bedrock_alpha20"])

    lo, hi = alpha_level / 2, 1 - alpha_level / 2
    out = {}
    for key, vals in draws.items():
        vals = np.asarray(vals)
        flag = (vals < np.inf) & (~np.isnan(vals))
        vals = vals[flag]
        if vals.size == 0:
            out[key] = [None, None, None]
            continue
        out[key] = [round(float(vals.mean()), 3),
                    round(float(np.percentile(vals, 100 * lo)), 3),
                    round(float(np.percentile(vals, 100 * hi)), 3)]
    return out


def _mol_props(mol):
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rot": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def _props_match(cp, ap) -> bool:
    if abs(cp["mw"] - ap["mw"]) / max(abs(ap["mw"]), 1e-6) > MW_TOL_FRAC:
        return False
    if abs(cp["logp"] - ap["logp"]) > LOGP_TOL:
        return False
    if abs(cp["hbd"] - ap["hbd"]) > HBD_TOL:
        return False
    if abs(cp["hba"] - ap["hba"]) > HBA_TOL:
        return False
    if abs(cp["rot"] - ap["rot"]) > ROT_TOL:
        return False
    return True


def _fp(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)



# SMIRKS transformations for de novo decoy generation. Each entry is a
# (description, reactant_smarts, product_smarts) tuple that performs a
# conservative structural mutation (substituent swap / add / remove).
_SMIRKS_DECOR = [
    # Methyl → Cl
    ("methyl->Cl", "[CH3:1]", "[Cl:1]"),
    # Methyl → OH
    ("methyl->OH", "[CH3:1]", "[OH:1]"),
    # Methyl → F
    ("methyl->F", "[CH3:1]", "[F:1]"),
    # Methyl → NH2
    ("methyl->NH2", "[CH3:1]", "[NH2:1]"),
    # Cl → F
    ("Cl->F", "[Cl:1]", "[F:1]"),
    # F → Cl
    ("F->Cl", "[F:1]", "[Cl:1]"),
    # OH → NH2
    ("OH->NH2", "[OH:1]", "[NH2:1]"),
    # NH2 → OH
    ("NH2->OH", "[NH2:1]", "[OH:1]"),
    # Demethylation (aromatic methyl → aromatic H)
    ("demethyl_ar", "[cH:1]-[CH3:2]", "[c:1]-[H:2]"),
    # Methylation (aromatic H → aromatic CH3)
    ("methyl_ar", "[c:1]-[H:2]", "[c:1]-[CH3:2]"),
    # CF3 → CH3
    ("CF3->CH3", "[CX3:1](F)(F)F", "[CX3:1]"),
    # CH3 → CF3
    ("CH3->CF3", "[CX3:1]", "[CX3:1](F)(F)F"),
    # Alkene → alkane (reduce double bond)
    ("alkene_sat", "[*:1]-[C;H2,D2]=[C;H2,D2]-[*:2]", "[*:1]-[CH2]-[CH2]-[*:2]"),
    # Ethyl → methyl (chain shortening)
    ("shorten_ethyl", "[CH2:1]-[CH2:2]-[CH3:3]", "[CH2:1]-[CH3:2]"),
]


def _generate_decoy_mutate(active_mol, target_props, all_active_fps,
                           active_smis, rng, max_attempts=300):
    """Generate a property-matched decoy by applying random SMIRKS mutations.

    Picks a random reaction from _SMIRKS_DECOR, applies it to the active, and
    accepts if the product satisfies relaxed property constraints and low
    Tanimoto to all actives. Returns a SMILES string or None.
    """
    mw_t = target_props["mw"]
    logp_t = target_props["logp"]
    hbd_t = target_props["hbd"]
    hba_t = target_props["hba"]
    rot_t = target_props["rot"]

    for _ in range(max_attempts):
        desc, reactant_smarts, product_smarts = rng.choice(_SMIRKS_DECOR)
        try:
            rxn = AllChem.ReactionFromSmarts(f"{reactant_smarts}>>{product_smarts}")
            if rxn is None:
                continue
            mods = rxn.RunReactant(active_mol, 0)
            for mod_mol in mods:
                try:
                    Chem.SanitizeMol(mod_mol)
                except Exception:
                    continue
                smi = Chem.MolToSmiles(mod_mol)
                if smi in active_smis:
                    continue
                mol = mod_mol
                cp = _mol_props(mol)
                if abs(cp["mw"] - mw_t) / max(abs(mw_t), 1e-6) > 0.35:
                    continue
                if abs(cp["logp"] - logp_t) > 2.0:
                    continue
                if abs(cp["hbd"] - hbd_t) > 3:
                    continue
                if abs(cp["hba"] - hba_t) > 4:
                    continue
                if abs(cp["rot"] - rot_t) > 4:
                    continue
                fp = _fp(mol)
                max_sim = max(
                    DataStructs.TanimotoSimilarity(fp, afp)
                    for afp in all_active_fps
                )
                if max_sim >= 0.35:
                    continue
                return smi
        except Exception:
            continue
    return None



def generate_decoys(active_mols, pool_path, n_per_active=DECOYS_PER_ACTIVE):
    """Generate n_per_active property-matched decoys per active.

    Primary strategy: draw decoys from the *pool_path* library (DUD-E style:
    property-matched + Tanimoto < 0.50 to the paired active). When the pool
    cannot supply enough decoys (common for property-extreme ChEMBL actives),
    a secondary SMIRKS-mutation fallback generates new decoy structures de novo
    by applying random substituent swaps to the active.

    Returns a list of (smiles, compound_id) decoys.
    """
    active_smis = {Chem.MolToSmiles(m) for m in active_mols}
    active_fps = [_fp(m) for m in active_mols]
    active_props = [_mol_props(m) for m in active_mols]

    # Read candidate pool: all libraries in pool_path (comma-separated).
    pool = []
    pool_files = [p for p in pool_path.split(",") if p]
    lactam_pat = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)
    for ppath in pool_files:
        if not os.path.exists(ppath):
            continue
        with open(ppath, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smi = (row.get("smiles") or row.get("SMILES") or "").strip()
                if smi and smi not in active_smis:
                    pool.append(smi)
    log.info(f"  Candidate pool: {len(pool)} SMILES from {len(pool_files)} files")

    # Precompute pool properties and fingerprints once.
    # Exclude beta-lactams and multi-fragment molecules from the pool so
    # decoys are structurally distinct from the (beta-lactam) ChEMBL actives.
    pmols, pprops, pfps = [], [], []
    for smi in pool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if mol.HasSubstructMatch(lactam_pat):
            continue
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        if mol.GetNumHeavyAtoms() < 6:
            continue
        pmols.append(smi)
        pprops.append(_mol_props(mol))
        pfps.append(_fp(mol))
    log.info(f"  Pool usable: {len(pmols)} mols (non-beta-lactam, single-fragment)")

    rng = random.Random(RANDOM_SEED)
    decoys = []
    used_pool_smiles = set()

    for ai, ap_mol in enumerate(active_mols):
        ap = active_props[ai]
        afp = active_fps[ai]
        strict = [
            smi for smi, cp, fp in zip(pmols, pprops, pfps)
            if smi not in used_pool_smiles
            and _props_match(cp, ap)
            and DataStructs.TanimotoSimilarity(fp, afp) < MAX_TANIMOTO_TO_ACTIVE
        ]
        cands = strict
        if len(cands) < n_per_active:
            relaxed = [
                smi for smi, cp, fp in zip(pmols, pprops, pfps)
                if smi not in used_pool_smiles
                and abs(cp["mw"] - ap["mw"]) / max(abs(ap["mw"]), 1e-6) <= MW_TOL_FRAC * 2
                and abs(cp["logp"] - ap["logp"]) <= LOGP_TOL * 3
                and abs(cp["hbd"] - ap["hbd"]) <= HBD_TOL * 3
                and abs(cp["hba"] - ap["hba"]) <= HBA_TOL * 3
                and abs(cp["rot"] - ap["rot"]) <= ROT_TOL * 2
                and DataStructs.TanimotoSimilarity(fp, afp) < MAX_TANIMOTO_TO_ACTIVE
            ]
            log.info(f"  Active {ai}: {len(strict)} strict matches; using {len(relaxed)} relaxed")
            cands = relaxed

        rng.shuffle(cands)
        n = 0
        for smi in cands[:n_per_active]:
            decoys.append((smi, f"DECOY_{ai:02d}_{len(decoys):04d}"))
            used_pool_smiles.add(smi)
            n += 1

        # Fallback: generate decoys via SMIRKS mutations on the active.
        if n < n_per_active:
            log.info(f"  Active {ai}: pool only gave {n}/{n_per_active} decoys; generating via SMIRKS")
            needed = n_per_active - n
            attempts = 0
            while n < n_per_active and attempts < needed * 200:
                attempts += 1
                new_smi = _generate_decoy_mutate(
                    ap_mol, ap, active_fps, active_smis,
                    rng, max_attempts=10,
                )
                if new_smi and new_smi not in active_smis:
                    decoys.append((new_smi, f"DECOY_{ai:02d}_{len(decoys):04d}"))
                    active_smis.add(new_smi)
                    n += 1

        if n < n_per_active:
            log.warning(f"  Active {ai}: only {n}/{n_per_active} decoys generated")

    log.info(f"  Generated {len(decoys)} property-matched decoys")
    return decoys




def fetch_chembl_pbp2a_actives() -> List[Tuple[str, str, float]]:
    """Fetch PBP2a (MecA, CHEMBL6187) actives from ChEMBL API.

    Queries ChEMBL for compounds with IC50 ≤ 10 µM against CHEMBL6187 and
    returns (compound_id, smiles, ic50_uM) tuples. Only single-component
    molecules are kept (counter-ions / salt fragments are stripped); multi-
    fragment species are skipped since they cannot be docked as a single ligand.

    Returns [] if the API is unreachable (the caller falls back to local
    active_site_actives.csv).
    """
    import urllib.request
    import json as _json
    from rdkit.Chem import rdmolops

    url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json?"
        "target_chembl_id=CHEMBL6187&"
        "standard_type=IC50&"
        "limit=1000"
    )
    actives: List[Tuple[str, str, float]] = []
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "AutoAntibiotic/7.3.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = _json.loads(response.read().decode())
            activities = data.get("activities", [])
            seen: set = set()
            skipped_fragments = 0
            for a in activities:
                smi = a.get("canonical_smiles", "")
                if not smi:
                    continue
                val = a.get("standard_value")
                try:
                    val_float = float(val)
                except (TypeError, ValueError):
                    continue
                if val_float > 10:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                # Strip counterions / water: keep the largest fragment only
                try:
                    frags = rdmolops.GetMolFrags(mol, asMols=True)
                except Exception:
                    frags = [mol]
                if len(frags) == 0:
                    continue
                if len(frags) > 1:
                    skipped_fragments += 1
                # Take the largest fragment
                best_frag = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                if best_frag.GetNumHeavyAtoms() < 6:
                    # Too small to be a meaningful binder
                    continue
                canonical = Chem.MolToSmiles(best_frag)
                if canonical in seen:
                    continue
                seen.add(canonical)
                cid = a.get("molecule_chembl_id") or f"CHEMBL_{len(actives):04d}"
                actives.append((cid, canonical, val_float))
        log.info(
            f"  Fetched {len(actives)} unique single-component PBP2a actives "
            f"from ChEMBL (IC50 ≤ 10 µM; {skipped_fragments} multi-fragment skipped)"
        )
    except Exception as exc:
        log.warning(f"  ChEMBL API unavailable ({exc}); falling back to local active_site_actives.csv")
    return actives


def _save_chembl_actives(actives: List[Tuple[str, str, float]], path: str) -> None:
    """Save fetched actives to CSV for reproducibility."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["compound_id", "smiles", "ic50_uM"])
        for cid, smi, ic50 in actives:
            writer.writerow([cid, smi, ic50])
    log.info(f"  ChEMBL actives saved: {path}")


def load_benchmark(active_path, pool_path, n_decoys_per_active=DECOYS_PER_ACTIVE):
    """Load PBP2a actives + property-matched decoys. Returns (records, labels).

    Actives are fetched from ChEMBL (CHEMBL6187, IC50 < 10 µM), falling back
    to the local data/active_site_actives.csv. At least 50 actives are required
    for a statistically meaningful enrichment benchmark.
    """
    records = []
    labels = []

    # Try ChEMBL first
    chembl_path = os.path.join(DATA, "chembl_pbp2a_actives.csv")
    chembl_actives = fetch_chembl_pbp2a_actives()

    # Fallback to the cached local actives file if the live ChEMBL query
    # returned too few (e.g. network unreachable). The cache is written by
    # _save_chembl_actives on every successful fetch, so it always reflects
    # the last authoritative pull (reproducibility without live network).
    if len(chembl_actives) < MIN_ACTIVES and os.path.exists(chembl_path):
        log.warning(
            f"  ChEMBL API returned {len(chembl_actives)} actives (< {MIN_ACTIVES}); "
            f"reloading cached local actives: {chembl_path}"
        )
        chembl_actives = []
        with open(chembl_path, newline="") as fh:
            for row in csv.DictReader(fh):
                smi = (row.get("smiles") or "").strip()
                if not smi:
                    continue
                try:
                    ic50 = float(row.get("ic50_uM"))
                except (TypeError, ValueError):
                    ic50 = 10.0
                chembl_actives.append(
                    (row.get("compound_id") or f"CHEMBL_{len(chembl_actives):04d}",
                     smi, ic50)
                )
        log.info(f"  Loaded {len(chembl_actives)} cached ChEMBL actives")

    # Cache ChEMBL actives
    if chembl_actives:
        _save_chembl_actives(chembl_actives, chembl_path)

    active_mols = []
    active_smiles = set()

    if len(chembl_actives) >= MIN_ACTIVES:
        log.info(f"  Using {len(chembl_actives)} ChEMBL actives (≥ {MIN_ACTIVES} required)")
        for cid, smi, ic50 in chembl_actives:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
            labels.append(1)
            active_mols.append(mol)
            active_smiles.add(Chem.MolToSmiles(mol))
    else:
        # Fall back to local actives
        log.info(f"  ChEMBL returned {len(chembl_actives)} actives (< {MIN_ACTIVES} needed); "
                 f"using local active_site_actives.csv")
        if os.path.exists(active_path):
            with open(active_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    smi = row["smiles"].strip()
                    cid = row["compound_id"].strip()
                    mol = Chem.MolFromSmiles(smi)
                    if not smi or mol is None:
                        continue
                    records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
                    labels.append(1)
                    active_mols.append(mol)
                    active_smiles.add(Chem.MolToSmiles(mol))
        log.info(f"  Loaded {len(active_mols)} local actives")

    log.warning(f"  Total actives: {len(active_mols)} (target: ≥{MIN_ACTIVES})")

    decoys = generate_decoys(active_mols, pool_path, n_per_active=n_decoys_per_active)
    for smi, cid in decoys:
        records.append(P.CompoundRecord(compound_id=cid, smiles=smi))
        labels.append(0)

    # Persist the decoy set for reproducibility.
    decoys_path = os.path.join(OUT, "dude_decoys.csv")
    with open(decoys_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "compound_id", "label"])
        for smi, cid in decoys:
            writer.writerow([smi, cid, "decoy"])
    log.info(f"  Decoys written: {decoys_path}")

    return records, labels


def main(argv=None):
    parser = argparse.ArgumentParser(description="DUD-E style PBP2a enrichment benchmark")
    parser.add_argument(
        "--decoy-pool", type=str, default=None,
        help="SMILES library CSV used to source property-matched decoys "
             "(default: data/screen_library_final.csv).",
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=8,
        help="Vina exhaustiveness (default 8 — the pipeline's documented "
             "enrichment-protocol setting; screening runs at 32).",
    )
    parser.add_argument(
        "--n-decoys-per-active", type=int, default=DECOYS_PER_ACTIVE,
        help=f"Decoys per active (default {DECOYS_PER_ACTIVE}).",
    )
    parser.add_argument(
        "--include-conserved-waters", action="store_true",
        help="Merge conserved active-site waters (output/conserved_waters.pdb) "
             "into the receptor before PDBQT conversion and re-run the "
             "water-included benchmark.",
    )
    args = parser.parse_args(argv)

    config = P.load_config()
    config["mode"] = "science"
    deps = P.check_dependencies()
    if not deps["USE_VINA"]:
        log.error("Vina required for DUD-E benchmark. Aborting.")
        sys.exit(1)

    pool_path = args.decoy_pool or DEFAULT_DECOY_POOL
    if not all(os.path.exists(p) for p in pool_path.split(",")):
        log.error(f"Decoy pool files not found: {pool_path}")
        sys.exit(1)

    pdb_dir = os.path.join(OUT, "pdb_dude")
    work_dir = os.path.join(OUT, "workdir_dude")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Target prep with caching (reuse cached receptor PDBQTs where present).
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

    # Docking against PBP2a apo (1VQQ) only — resting-state structure is the
    # DUD-E / DEKOIS standard for enrichment (holo conformers give decoys
    # artificially good scores in the expanded pocket).
    receptor_pdbqt = receptor_pdbqts[0]
    log.info(f"  Docking against apo receptor: {receptor_pdbqt}")

    records, labels = load_benchmark(
        os.path.join(DATA, "active_site_actives.csv"),
        pool_path,
        n_decoys_per_active=args.n_decoys_per_active,
    )
    log.info(f"  Benchmark set: {sum(labels)} actives, {len(labels) - sum(labels)} decoys")

    from functools import partial
    from utils.docking import dock_compound
    results = P._dock_compounds_parallel(
        records, receptor_pdbqt, active_center, active_box,
        work_dir, "dude",
        dock_func=partial(dock_compound, exhaustiveness=args.exhaustiveness),
    )
    energies = {rec.compound_id: energy for rec, energy in results}

    ids = [r.compound_id for r in records]
    scores = [-(energies[cid] if energies[cid] is not None else 1e9) for cid in ids]
    fpr, tpr, auc = compute_roc(labels, scores)
    bedrock = compute_bedrock(labels, scores, alpha=20.0)

    N = len(ids)
    n_act = sum(labels)
    k1 = max(1, round(0.01 * N))
    k5 = max(1, round(0.05 * N))
    k10 = max(1, round(0.10 * N))
    ranked = sorted(ids, key=lambda c: (energies[c] if energies[c] is not None else 1e9))
    act_in_1 = sum(1 for c in ranked[:k1] if labels[ids.index(c)] == 1)
    act_in_5 = sum(1 for c in ranked[:k5] if labels[ids.index(c)] == 1)
    act_in_10 = sum(1 for c in ranked[:k10] if labels[ids.index(c)] == 1)
    ef1 = (act_in_1 / n_act) / (k1 / N) if n_act else 0.0
    ef5 = (act_in_5 / n_act) / (k5 / N) if n_act else 0.0
    ef10 = (act_in_10 / n_act) / (k10 / N) if n_act else 0.0

    passed = auc >= AUC_MIN and ef1 >= EF1_MIN and bedrock >= BEDROC_MIN
    cis = bootstrap_cis(labels, scores, n_resamples=1000, seed=RANDOM_SEED)
    result = {
        "n_compounds": N,
        "n_actives": n_act,
        "n_decoys": N - n_act,
        "decoys_per_active": args.n_decoys_per_active,
        "decoy_pool": pool_path,
        "exhaustiveness": args.exhaustiveness,
        "auc": round(float(auc), 4),
        "bedrock_alpha20": round(float(bedrock), 4),
        "ef_1pct": round(float(ef1), 3),
        "ef_5pct": round(float(ef5), 3),
        "ef_10pct": round(float(ef10), 3),
        "ci_95_bootstrap_1000": cis,
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": {"auc_min": AUC_MIN, "bedroc_min": BEDROC_MIN, "ef_1pct_min": EF1_MIN},
        "active_box": list(active_box),
        "active_center": [float(v) for v in active_center],
        "receptor": "PBP2a apo 1VQQ",
        "method": "DUD-E style (property-matched decoys, apo receptor, Vina)",
    }
    with open(os.path.join(OUT, "dude_benchmark_results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # The DUD-E benchmark is now the single authoritative enrichment
    # validation (≥50 actives, ≥500 property-matched decoys, apo receptor).
    # Mirror the same result into enrichment_results.json so the canonical
    # report / verify_success.py read ONE set of numbers (Phase 2 —
    # reconciliation of the abstract vs Discussion contradiction).
    with open(os.path.join(OUT, "enrichment_results.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # ROC plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"PBP2a DUD-E style enrichment (apo 1VQQ, ex={args.exhaustiveness})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dude_benchmark_roc.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "enrichment_roc.png"), dpi=300)
    plt.close(fig)

    # EF bar chart (mirrored to the canonical enrichment_ef.png)
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(["EF_1%", "EF_5%"], [ef1, ef5], color=["#2c7fb8", "#7fcdbb"])
    ax2.axhline(5, color="r", ls="--", lw=1, label="pass threshold (5)")
    ax2.set_ylabel("Enrichment Factor")
    ax2.set_title(f"Enrichment Factors (apo 1VQQ, ex={args.exhaustiveness})")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "enrichment_ef.png"), dpi=300)
    plt.close(fig2)

    log.info("")
    log.info("=" * 60)
    log.info("  DUD-E style benchmark")
    log.info("=" * 60)
    log.info(f"  Compounds: {N}  ({n_act} actives / {N - n_act} decoys)")
    log.info(f"  AUC={auc:.4f}  BEDROC(20)={bedrock:.4f}  "
             f"EF_1%={ef1:.2f}  EF_5%={ef5:.2f}  EF_10%={ef10:.2f}")
    log.info(f"  VERDICT: {'PASS' if passed else 'FAIL'} "
             f"(AUC>={AUC_MIN}, BEDROC>={BEDROC_MIN}, EF_1%>={EF1_MIN} required)")

    # ── Water-included benchmark (optional) ──────────────────────
    if args.include_conserved_waters:
        water_pdb = os.path.join(OUT, "conserved_waters.pdb")
        if not os.path.exists(water_pdb):
            log.warning(
                "  --include-conserved-waters set but "
                f"{water_pdb} not found; skipping water-included benchmark"
            )
        else:
            log.info("")
            log.info("=" * 60)
            log.info("  Water-Included DUD-E Benchmark")
            log.info("=" * 60)
            try:
                water_receptor_pdb = os.path.join(
                    work_dir, "PBP2a_apo_waters_clean.pdb"
                )
                merge_conserved_waters(
                    cleaned_pdb, water_pdb, water_receptor_pdb,
                )
                water_pdbqt = water_receptor_pdb.replace(".pdb", ".pdbqt")
                if not (os.path.exists(water_pdbqt) and os.path.getsize(water_pdbqt) > 0):
                    P.clean_pdb_structure(water_receptor_pdb, water_receptor_pdb)
                water_box = P._auto_box_size(
                    water_receptor_pdb, active_center, ACTIVE_BOX_SIZE,
                    min_size=15.0, max_size=20.0, site_residues=ACTIVE_SITE_RESIDUES,
                )
                water_results = P._dock_compounds_parallel(
                    records, water_pdbqt, active_center, water_box,
                    work_dir, "dude_water",
                    dock_func=partial(dock_compound, exhaustiveness=args.exhaustiveness),
                )
                water_energies = {rec.compound_id: energy for rec, energy in water_results}
                water_scores = [-(water_energies[cid] if water_energies[cid] is not None else 1e9) for cid in ids]
                water_fpr, water_tpr, water_auc = compute_roc(labels, water_scores)
                water_bedrock = compute_bedrock(labels, water_scores, alpha=20.0)
                water_ranked = sorted(ids, key=lambda c: (water_energies[c] if water_energies[c] is not None else 1e9))
                water_act_in_1 = sum(1 for c in water_ranked[:k1] if labels[ids.index(c)] == 1)
                water_act_in_5 = sum(1 for c in water_ranked[:k5] if labels[ids.index(c)] == 1)
                water_act_in_10 = sum(1 for c in water_ranked[:k10] if labels[ids.index(c)] == 1)
                water_ef1 = (water_act_in_1 / n_act) / (k1 / N) if n_act else 0.0
                water_ef5 = (water_act_in_5 / n_act) / (k5 / N) if n_act else 0.0
                water_ef10 = (water_act_in_10 / n_act) / (k10 / N) if n_act else 0.0
                water_passed = water_auc >= AUC_MIN and water_ef1 >= EF1_MIN and water_bedrock >= BEDROC_MIN
                water_result = {
                    "n_compounds": N,
                    "n_actives": n_act,
                    "n_decoys": N - n_act,
                    "decoys_per_active": args.n_decoys_per_active,
                    "decoy_pool": pool_path,
                    "exhaustiveness": args.exhaustiveness,
                    "auc": round(float(water_auc), 4),
                    "bedrock_alpha20": round(float(water_bedrock), 4),
                    "ef_1pct": round(float(water_ef1), 3),
                    "ef_5pct": round(float(water_ef5), 3),
                    "ef_10pct": round(float(water_ef10), 3),
                    "ci_95_bootstrap_1000": bootstrap_cis(labels, water_scores, n_resamples=1000, seed=RANDOM_SEED),
                    "verdict": "PASS" if water_passed else "FAIL",
                    "thresholds": {"auc_min": AUC_MIN, "bedroc_min": BEDROC_MIN, "ef_1pct_min": EF1_MIN},
                    "active_box": list(water_box),
                    "active_center": [float(v) for v in active_center],
                    "receptor": "PBP2a apo 1VQQ (with conserved waters)",
                    "method": "DUD-E style (water-included receptor, property-matched decoys, Vina)",
                    "water_included": True,
                }
                with open(os.path.join(OUT, "dude_benchmark_water_results.json"), "w") as fh:
                    json.dump(water_result, fh, indent=2)
                log.info(f"  Water-included AUC={water_auc:.4f}  VERDICT={'PASS' if water_passed else 'FAIL'}")
                log.info(f"  Results saved to output/dude_benchmark_water_results.json")
            except Exception as exc:
                log.error(f"  Water-included benchmark failed: {exc}")
                import traceback
                traceback.print_exc()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
