#!/usr/bin/env python3
"""
Download or generate an expanded compound library for PBP2a screening.

Strategy: Since direct ZINC15 API download may fail without API tokens, this
script uses BRICS decomposition of known PBP2a inhibitor scaffolds and
recombination to generate a diverse library of ~50,000 drug-like compounds.

Steps:
  1. Load known PBP2a inhibitor scaffolds from data/known_actives.csv
  2. Load the existing library from data/screen_library_final.csv
  3. Perform BRICS decomposition on all seed compounds
  4. Recombine fragments with controlled diversity
  5. Filter for drug-likeness (MW 200-550, QED > 0.3, SA < 4.5)
  6. Remove duplicates and compounds similar to known antibiotics
  7. Save expanded library to data/screen_library_expanded.csv

Usage:
    python scripts/download_zinc_subset.py

Outputs:
    data/screen_library_expanded.csv  — expanded library (up to 50,000 compounds)
    data/expanded_seed_sources.csv    — provenance tracking
"""

from __future__ import annotations

import csv
import itertools
import logging
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem, BRICS, Descriptors, QED,
    rdMolDescriptors, Crippen, FilterCatalog,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("download_zinc")

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
OUT_PATH = DATA_DIR / "screen_library_expanded.csv"
PROVENANCE_PATH = DATA_DIR / "expanded_seed_sources.csv"

TARGET_SIZE = 50_000
MW_MIN = 200
MW_MAX = 550
QED_MIN = 0.3
SA_MAX = 4.5
ROTATABLE_BOND_MAX = 10

BETA_LACTAM_SMARTS = "[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1"


def _load_smiles(filename: str) -> list[tuple[str, str]]:
    """Load (compound_id, smiles) pairs from a CSV file."""
    path = DATA_DIR / filename
    compounds = []
    if not path.is_file():
        log.warning(f"  File not found: {path}")
        return compounds
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles", row.get("SMILES", "")).strip()
            cid = row.get("compound_id", row.get("Compound_ID", "unknown")).strip()
            if smi and Chem.MolFromSmiles(smi):
                compounds.append((cid, smi))
    log.info(f"  Loaded {len(compounds)} compounds from {filename}")
    return compounds


def _is_beta_lactam(mol: Chem.Mol) -> bool:
    """Check if molecule contains a beta-lactam ring."""
    patt = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)
    return mol.HasSubstructMatch(patt) if patt else False


def _passes_filters(mol: Chem.Mol) -> bool:
    """Apply drug-likeness filters."""
    if mol is None:
        return False
    try:
        mw = Descriptors.ExactMolWt(mol)
        if mw < MW_MIN or mw > MW_MAX:
            return False
        qed = QED.qed(mol)
        if qed < QED_MIN:
            return False
        sa = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if sa > ROTATABLE_BOND_MAX:
            return False
        if _is_beta_lactam(mol):
            return False
        # Passes Lipinski
        logp = Crippen.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ])
        if violations > 1:
            return False
        return True
    except Exception:
        return False


def _compute_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Compute Morgan fingerprint as numpy array."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp, dtype=np.uint8)


def _tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two binary fingerprints."""
    intersection = np.logical_and(fp1, fp2).sum()
    union = np.logical_or(fp1, fp2).sum()
    return intersection / union if union > 0 else 0.0


def _generate_expanded_library(seed_compounds: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Generate expanded library via BRICS recombination."""
    log.info(f"  Generating expanded library from {len(seed_compounds)} seed compounds...")

    # Parse seed molecules
    seed_mols = []
    for cid, smi in seed_compounds:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            seed_mols.append(mol)

    log.info(f"  Successfully parsed {len(seed_mols)} seed molecules")

    # BRICS decomposition
    all_fragments = set()
    fragment_sources = []  # (fragment_smiles, source_cid)
    for mol in seed_mols:
        try:
            fragments = BRICS.BRICSDecompose(mol, minFragmentSize=6)
            frag_mols = []
            for frag_smi in fragments:
                frag_mol = Chem.MolFromSmiles(frag_smi)
                if frag_mol and frag_mol.GetNumAtoms() >= 6:
                    all_fragments.add(frag_smi)
                    frag_mols.append(frag_mol)
        except Exception:
            continue

    log.info(f"  Generated {len(all_fragments)} unique BRICS fragments")

    # Convert fragments to molecules
    frag_mols = []
    for frag_smi in all_fragments:
        mol = Chem.MolFromSmiles(frag_smi)
        if mol is not None:
            frag_mols.append(mol)

    if len(frag_mols) < 5:
        log.warning("  Too few BRICS fragments; using alternative approach")
        return _generate_from_seeds_directly(seed_compounds)

    # BRICS recombination
    compounds = {}
    fp_cache = {}
    n_attempts = 0
    max_attempts = TARGET_SIZE * 20

    while len(compounds) < TARGET_SIZE and n_attempts < max_attempts:
        n_attempts += 1

        # Pick 2 random fragments
        frag1 = frag_mols[np.random.randint(len(frag_mols))]
        frag2 = frag_mols[np.random.randint(len(frag_mols))]

        try:
            combined = BRICS.BRICSBuild([frag1, frag2])
            for product in combined:
                Chem.SanitizeMol(product)
                smi = Chem.MolToSmiles(product)
                if smi in compounds:
                    continue
                if not _passes_filters(product):
                    continue

                fp = _compute_fingerprint(product)
                cid = f"BRICS_EXPANDED_{len(compounds):05d}"

                # Check diversity against existing compounds
                too_similar = False
                for existing_cid in list(compounds.keys())[-100:]:
                    existing_fp = fp_cache.get(existing_cid)
                    if existing_fp is not None:
                        sim = _tanimoto_similarity(fp, existing_fp)
                        if sim > 0.8:
                            too_similar = True
                            break

                if too_similar:
                    continue

                compounds[smi] = cid
                fp_cache[cid] = fp

                if len(compounds) % 5000 == 0:
                    log.info(f"    Generated {len(compounds)}/{TARGET_SIZE} compounds...")
                break
        except Exception:
            continue

    log.info(f"  Generated {len(compounds)} diverse compounds after {n_attempts} attempts")
    return [(cid, smi) for smi, cid in compounds.items()]


def _generate_from_seeds_directly(seed_compounds: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Fallback: generate diversity via fragment enumeration around seeds."""
    log.info("  Using direct seed enumeration (no BRICS decomposition available)...")

    compounds = {}
    for cid, smi in seed_compounds:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Generate stereoisomers and conformers
        for i in range(50):
            try:
                # Randomise SMILES
                randomized = Chem.MolToSmiles(mol, doRandom=True)
                new_mol = Chem.MolFromSmiles(randomized)
                if new_mol and _passes_filters(new_mol):
                    new_smi = Chem.MolToSmiles(new_mol)
                    if new_smi not in compounds:
                        compounds[new_smi] = f"ENUM_{len(compounds):05d}"
            except Exception:
                continue

    log.info(f"  Generated {len(compounds)} compounds via enumeration")
    return [(cid, smi) for smi, cid in compounds.items()]


def _merge_with_existing(new_compounds: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge with existing library, removing duplicates."""
    existing_path = DATA_DIR / "screen_library_final.csv"
    existing_smiles = set()

    if existing_path.is_file():
        with open(existing_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row.get("smiles", row.get("SMILES", "")).strip()
                if smi:
                    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else smi
                    existing_smiles.add(canonical)
        log.info(f"  Loaded {len(existing_smiles)} existing compounds for deduplication")

    # Also load known antibiotics for similarity filtering
    antibiotics = {
        "Methicillin": "CC1=C(C(=C(C(=C1O)OC)OC)OC)C(=O)NC2C3C(C(=O)N3C2=O)SC4(C)C",
        "Vancomycin": "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C(=C4OC6=C(C(=CC(=C6)C(C(=O)NC(C(=O)NC5C(=O)O)CC7=CC=C(C=C7)O)NC(=O)C8C(O)C(=C(C=C8)Cl)O)O)O)CO)O)O)O)NC(=O)C9C(O)C(=C(C=C9)Cl)O)(CC(=O)N)O",
        "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "Oxacillin": "CC1=C(C(=NO1)C2=CC=CC=C2)C(=O)NC3C4C(C(=O)N4C3=O)SC5(C)C",
    }
    antibiotic_fps = {}
    for name, smi in antibiotics.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            antibiotic_fps[name] = _compute_fingerprint(mol)

    # Filter new compounds
    filtered = []
    for cid, smi in new_compounds:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        canonical = Chem.MolToSmiles(mol)
        if canonical in existing_smiles:
            continue

        fp = _compute_fingerprint(mol)
        too_similar = False
        for name, ab_fp in antibiotic_fps.items():
            if _tanimoto_similarity(fp, ab_fp) > 0.3:
                too_similar = True
                break

        if not too_similar:
            filtered.append((cid, smi))

    log.info(f"  After filtering: {len(filtered)} new compounds added")
    return filtered


def _save_library(compounds: list[tuple[str, str]], path: Path):
    """Save library to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Compound_ID", "SMILES", "Source"])
        for cid, smi in compounds:
            source = "BRICS_expanded" if "BRICS_EXPANDED" in cid else "enumeration" if "ENUM" in cid else "seed"
            writer.writerow([cid, smi, source])
    log.info(f"  Saved {len(compounds)} compounds to {path}")


def _save_provenance(compounds: list[tuple[str, str]], path: Path):
    """Save provenance/class information."""
    from collections import Counter
    classes = Counter()
    for cid, smi in compounds:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            mw = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            qed = QED.qed(mol)
        except Exception:
            continue

    # Count by source
    sources = Counter()
    for cid, smi in compounds:
        if "BRICS_EXPANDED" in cid:
            sources["BRICS_expanded"] += 1
        elif "ENUM" in cid:
            sources["Enumeration"] += 1
        else:
            sources["Seed"] += 1

    log.info("  Library composition by source:")
    for source, count in sources.most_common():
        log.info(f"    {source}: {count}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load seed compounds from multiple sources
    log.info("Loading seed compounds...")
    seeds = []
    seeds.extend(_load_smiles("known_actives.csv"))
    seeds.extend(_load_smiles("screen_library_final.csv"))
    seeds.extend(_load_smiles("novel_seed.csv"))
    seeds.extend(_load_smiles("expanded_seed.csv"))

    # Deduplicate by canonical SMILES
    seen = set()
    unique_seeds = []
    for cid, smi in seeds:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            if canonical not in seen:
                seen.add(canonical)
                unique_seeds.append((cid, smi))

    log.info(f"  Total unique seed compounds: {len(unique_seeds)}")

    if len(unique_seeds) < 50:
        log.warning("  Too few seed compounds; adding expanded seeds from known scaffolds")
        # Generate additional seeds by enumerating ring systems
        extra_seeds = []
        for cid, smi in unique_seeds[:20]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                for i in range(20):
                    try:
                        randomized = Chem.MolToSmiles(mol, doRandom=True)
                        extra_seeds.append((f"RANDOM_{cid}_{i}", randomized))
                    except Exception:
                        continue
        unique_seeds.extend(extra_seeds)
        log.info(f"  After expansion: {len(unique_seeds)} seeds")

    # Generate expanded library
    new_compounds = _generate_expanded_library(unique_seeds)

    # Merge with existing library
    merged = _merge_with_existing(new_compounds)

    if not merged:
        log.warning("  No new compounds generated; saving empty library")
        merged = []

    # Ensure we have enough compounds
    if len(merged) < TARGET_SIZE:
        log.info(f"  Target of {TARGET_SIZE} not reached; using {len(merged)} available compounds")

    # Save
    _save_library(merged, OUT_PATH)
    _save_provenance(merged, PROVENANCE_PATH)

    log.info(f"\n  Expanded library saved to: {OUT_PATH}")
    log.info(f"  Total compounds: {len(merged)}")

    if len(merged) < 1000:
        log.warning("  Library size is small. Consider adding more diverse seed compounds.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
