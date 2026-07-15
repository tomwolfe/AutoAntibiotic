#!/usr/bin/env python3
"""
Library generation for the AutoAntibiotic discovery pipeline.

This module owns everything needed to build the candidate compound library:
the :class:`CompoundRecord` data class, the curated natural-product scaffold
list, the positive control SMILES, and the :func:`generate_candidate_library`
entry point (BRICS-based fragment recombination or CSV/SDF input).

It is intentionally self-contained — it depends only on ``config.constants`` and
RDKit — so it can be imported without pulling in the rest of the orchestrator
and without creating a circular import with ``discovery_pipeline``.
"""

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
from rdkit.Chem import BRICS

from config.constants import RANDOM_SEED

# When ``target_count`` is at/above this threshold, decomposed fragments are
# cached to a temp JSON so repeated runs (e.g. re-tuning parameters) avoid
# re-decomposing the same scaffolds. Small runs skip the cache entirely to keep
# things fast and dependency-free.
BRICS_CACHE_MIN_TARGET = 200

# A module-level logger sharing the pipeline's "AutoAntibiotic" logger name so
# that handlers configured in discovery_pipeline capture these messages too.
log = logging.getLogger("AutoAntibiotic")


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPOUND RECORD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompoundRecord:
    """Stores all computed properties for a single candidate."""

    # Selectivity confidence labels
    CONF_HIGH = "High"
    CONF_LOW = "Low"
    CONF_NONE = "None"

    compound_id: str
    smiles: str
    mol: Optional[Chem.Mol] = None

    # Docking scores
    pb2pa_allosteric_energy: Optional[float] = None
    pb2pa_active_energy: Optional[float] = None
    human_trypsin_energy: Optional[float] = None
    human_ces1_energy: Optional[float] = None

    # Selectivity
    selectivity_index: Optional[float] = None

    # Similarity
    max_similarity: float = 0.0

    # ADMET
    passes_lipinski: bool = False
    qed_score: float = 0.0
    passes_pains: bool = False

    # Resistance flags
    resistance_notes: str = ""

    # Selectivity confidence based on how many human off-targets were docked:
    #   "High" if 2 human targets provided valid energies,
    #   "Low"  if 1 human target provided a valid energy,
    #   "None" if 0 human targets provided a valid energy.
    selectivity_confidence: str = "None"

    # Path to the active-site Vina docked pose (PDBQT), populated during
    # screening so that pose-based interaction analysis need not re-dock.
    active_docked_pdbqt: Optional[str] = None

    # Interaction fingerprint (dict returned by analyze_binding_interactions)
    # captured during Phase 4 so reporting can expose per-residue H-bond flags
    # without re-parsing the docked pose.
    interactions: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAFFOLDS & CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

# 15 diverse natural product scaffolds (SMILES)
NATURAL_PRODUCT_SCAFFOLDS = [
    "O=c1c(O)c2c(oc3cc(O)cc(O)c3c2=O)c(O)c1O",                 # Quercetin
    "Oc1ccc(C=Cc2ccc(O)cc2)cc1",                                # Resveratrol
    "COc1ccc(C=CC(=O)CC(=O)C=Cc2ccc(OC)c(O)c2)cc1O",           # Curcumin
    "COc1cc2c(cc1OC)[n+]1ccc3cc4c(cc3c1CC2)OCO4",              # Berberine
    "CC1(C)OC2C3C(=O)OC4C(OO5)C3C5C2C4O1",                     # Artemisinin (approximate)
    "Oc1ccccc1C(=O)O",                                         # Salicylic acid (salicylate)
    "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",                    # 7-Hydroxyflavone (flavonoid core)
    "CC1OCCCC(=O)C1",                                          # Macrolide-like lactone core (no β-lactam)
    "Oc1c(O)c(O)cc(C(=O)O)c1",                                 # Gallic acid (phenolic)
    "CC1=C(C=C(C=C1)O)O",                                      # Hydroquinone
    "COc1cc2c(cc1OC)C(=O)C3=C(O)C=CC(=C3O2)C",                 # Rottlerin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                            # Caffeine
]

# Positive control SMILES (to verify pipeline)
CONTROL_SMILES = {
    "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
    "Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  LIBRARY GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _count_atoms(mol: Chem.Mol) -> int:
    """Heavy-atom count for a molecule."""
    return mol.GetNumHeavyAtoms()


def _brics_cache_path(scaffold_smiles: List[str]) -> str:
    """
    Stable temp-file path for a BRICS fragment cache keyed on the scaffold set.

    The cache is keyed on a hash of the canonical scaffold SMILES list so the
    same input scaffolds reuse the same cache file across runs. Returns the
    path under the system temp dir; callers must handle read/write failures
    gracefully (the cache is strictly an optimisation).
    """
    digest = hashlib.sha256("\n".join(sorted(scaffold_smiles)).encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"autobiotic_brics_{digest}.json")


def _load_brics_cache(cache_path: str) -> Optional[List[str]]:
    """Load cached fragment SMILES from *cache_path*, or None on any failure."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("fragments"), list):
                return [str(s) for s in data["fragments"]]
    except Exception:
        pass
    return None


def _save_brics_cache(cache_path: str, fragments: List[str]) -> None:
    """Persist *fragments* to *cache_path* (best-effort; failures are ignored)."""
    try:
        with open(cache_path, "w") as fh:
            json.dump({"fragments": list(fragments)}, fh)
    except Exception:
        pass


def generate_candidate_library(
    target_count: int = 500,
    seed: int = RANDOM_SEED,
    input_csv: Optional[str] = None,
    input_sdf: Optional[str] = None,
) -> List[CompoundRecord]:
    """
    Phase 2.1 — Generate a diverse library.

    If *input_csv* is provided, the library is read directly from that CSV
    file (expected columns: ``smiles``, ``compound_id``) and the BRICS
    scaffold-generation logic is skipped entirely.

    If *input_sdf* is provided, the library is read directly from that SDF
    file via RDKit's :class:`Chem.SDMolSupplier` and BRICS is skipped entirely.
    Each molecule becomes a record whose ``compound_id`` is taken from its SDF
    ``_Name`` property (falling back to a positional ``SDF-####`` id).

    Otherwise, a library is generated by BRICS decomposition of natural
    product scaffolds, fragment recombination, and expansion.

    Args:
        target_count: Desired number of compounds (~500).
        seed: Random seed for reproducibility.
        input_csv: Optional path to an external compound library CSV.
        input_sdf: Optional path to an external compound library SDF.

    Returns:
        List of CompoundRecord objects (SMILES only, no computed props yet).
    """
    log.info("─── Phase 2: Library Generation ───")

    if input_sdf is not None:
        return _read_records_from_sdf(input_sdf)

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
                log.warning(f"  Skipping row with empty SMILES (compound_id={cid}).")
                continue
            mol = Chem.MolFromSmiles(smi)
            records.append(CompoundRecord(
                compound_id=cid,
                smiles=smi,
                mol=mol,
            ))
        log.info(f"  Loaded {len(records)} compounds from external CSV (BRICS skipped).")
        return records

    all_scaffolds = NATURAL_PRODUCT_SCAFFOLDS
    scaffold_mols = []
    for smi in all_scaffolds:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold_mols.append(mol)

    log.info(f"  Loaded {len(scaffold_mols)} valid scaffolds.")

    # BRICS decompose all scaffolds. For large target libraries the decomposition
    # is cached to a temp JSON (keyed on the scaffold set) so repeated runs skip
    # the expensive BRICSDecompose step. Small runs skip caching entirely.
    cache_enabled = target_count >= BRICS_CACHE_MIN_TARGET
    cache_path = _brics_cache_path(all_scaffolds) if cache_enabled else None
    cached = _load_brics_cache(cache_path) if cache_path else None
    if cached is not None:
        all_fragments = set(cached)
        log.info(f"  Reused {len(all_fragments)} cached BRICS fragments from {cache_path}.")
    else:
        all_fragments = set()
        for mol in scaffold_mols:
            try:
                fragments = BRICS.BRICSDecompose(mol, minFragmentSize=8)
                for frag_smi in fragments:
                    frag_mol = Chem.MolFromSmiles(frag_smi)
                    if frag_mol is not None and _count_atoms(frag_mol) >= 8:
                        all_fragments.add(frag_smi)
            except Exception:
                continue
        if cache_path is not None and all_fragments:
            _save_brics_cache(cache_path, sorted(all_fragments))

    frag_mols = []
    for smi in all_fragments:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            frag_mols.append(m)

    log.info(f"  Generated {len(frag_mols)} unique fragments (>=8 heavy atoms).")

    if len(frag_mols) < 2:
        log.warning(
            "  Too few fragments for recombination (<2). Returning "
            "controls and source scaffolds without novel analogs."
        )
        candidates = []
        for mol in scaffold_mols:
            smi = Chem.MolToSmiles(mol)
            candidates.append(CompoundRecord(
                compound_id=f"SCAFFOLD_{len(candidates)}",
                smiles=smi,
                mol=mol,
            ))
        for name, smi in CONTROL_SMILES.items():
            mol = Chem.MolFromSmiles(smi)
            candidates.append(CompoundRecord(
                compound_id=f"CTRL_{name}",
                smiles=smi,
                mol=mol,
            ))
        return candidates

    # Recombine fragments to create novel analogs via BRICSBuild over all fragments
    seen_smiles = set()
    records = []

    log.info(f"  Building recombinant library via BRICS.BRICSBuild (target ≤ {target_count})…")
    builder = BRICS.BRICSBuild(list(frag_mols))
    for product in builder:
        try:
            Chem.SanitizeMol(product)
        except Exception:
            continue
        smi = Chem.MolToSmiles(product)
        if smi in seen_smiles:
            continue
        seen_smiles.add(smi)

        # Generate unique ID
        cid = f"AA-{len(records):04d}"
        records.append(CompoundRecord(
            compound_id=cid,
            smiles=smi,
            mol=product,
        ))

        if len(records) % 100 == 0:
            log.info(f"  Generated {len(records)} / {target_count} candidates…")

        if len(records) >= target_count:
            break

    # Add controls explicitly (ensures at least controls are always returned)
    for name, smi in CONTROL_SMILES.items():
        if len(records) >= target_count:
            break
        if smi not in seen_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                records.append(CompoundRecord(
                    compound_id=f"CTRL_{name}",
                    smiles=smi,
                    mol=mol,
                ))
                seen_smiles.add(smi)

    log.info(f"  Library generation complete: {len(records)} compounds.")
    return records


def _read_records_from_sdf(sdf_path: str) -> List[CompoundRecord]:
    """
    Read pre-made molecules from an SDF file into ``CompoundRecord`` objects.

    Uses RDKit's :class:`Chem.SDMolSupplier`. Each molecule becomes a record
    with a ``compound_id`` taken from its SDF ``_Name`` property (falling back
    to a positional ``SDF-####`` id) and its canonical SMILES.

    Args:
        sdf_path: Path to the input SDF file.

    Returns:
        List of :class:`CompoundRecord` objects (one per readable molecule).
    """
    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"Input SDF not found: {sdf_path}")

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    records: List[CompoundRecord] = []
    for i, mol in enumerate(supplier):
        if mol is None:
            log.warning(f"  Skipping unreadable entry {i} in SDF.")
            continue
        if mol.HasProp("_Name"):
            cid = mol.GetProp("_Name").strip() or f"SDF-{i:04d}"
        else:
            cid = f"SDF-{i:04d}"
        smiles = Chem.MolToSmiles(mol)
        records.append(CompoundRecord(
            compound_id=cid,
            smiles=smiles,
            mol=mol,
        ))

    if not records:
        log.warning(f"  No valid molecules read from SDF: {sdf_path}")
    else:
        log.info(f"  Loaded {len(records)} molecules from SDF (BRICS skipped).")
    return records
