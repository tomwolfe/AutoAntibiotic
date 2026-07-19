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
    # Selectivity_Index_PanPanel: the OLD pan-panel SI (all human off-targets
    # in the denominator). Preserved as a separate column for transparency so
    # the mechanism-restricted SI can be compared against it (paper §Metrics).
    selectivity_index_panpanel: Optional[float] = None
    # Off-target risk flag (paper §4.1b): True when any *valid* human
    # off-target binds tightly (energy < -8.0 kcal/mol). Kept separate from
    # selectivity_index so the raw SI is never artificially zeroed.
    off_target_risk: bool = False

    # Phase 3.5 — Negative selection: the most negative (strongest) human
    # off-target binding energy observed across the human panel (kcal/mol),
    # populated by utils.filtering.filter_by_human_clash. None when the compound
    # has no valid human off-target energies. Surfaced in the CSV report.
    human_offtarget_max_energy: Optional[float] = None

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

    # Transparency metrics (Task 1). ``si_vs_ceftaroline`` is the supplementary
    # control-indexed metric = |E_PBP2a_best| / CEFTAROLINE_CONTROL_E (no
    # covalent energy bonus is ever applied — Vina cannot model covalent bond
    # formation). It lets the reader gauge each candidate against the clinical
    # reference without a post-hoc energy adjustment.
    warhead_type: Optional[str] = None  # retained for provenance, always "none"
    si_covalent: Optional[float] = None  # deprecated; retained for CSV backward-compat, always None
    si_vs_ceftaroline: Optional[float] = None

    # MM-GBSA-like rerank score (crude MMFF energy of the relaxed active-site
    # pose). Populated by utils.docking.rerank_mmff; None when unavailable.
    mmgbca_score: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAFFOLDS & CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

# 40 diverse scaffolds (natural products, drug-like, antibacterial chemotypes)
# Enriched for BRICS-compatible bonds to generate a larger fragment pool.
NATURAL_PRODUCT_SCAFFOLDS = [
    # ── Flavonoids / polyphenols ──
    "O=c1c(O)c2c(oc3cc(O)cc(O)c3c2=O)c(O)c1O",                    # Quercetin
    "Oc1ccc(C=Cc2ccc(O)cc2)cc1",                                   # Resveratrol
    "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",                       # 7-Hydroxyflavone
    "Oc1ccc2c(c1)OC(C3=CC(=C(C=C3)O)O)C(=O)C2",                   # Eriodictyol
    "COc1ccc2c(c1)CC(=O)C3=C(C=CC(=C3O)OC)O2",                     # 7-O-methylnaringenin
    "COc1cc(OC)c2c(c1)OC(C(C2=O)C3=CC=C(C=C3)O)C4=CC=C(C=C4)O",   # 3,5-dihydroxyflavone derivative

    # ── Curcuminoids / diarylheptanoids ──
    "COc1ccc(C=CC(=O)CC(=O)C=Cc2ccc(OC)c(O)c2)cc1O",              # Curcumin
    "COc1cc(OC)c(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)cc1O",          # Demethoxycurcumin
    "Oc1ccc(C=CC(=O)CCC(=O)C=Cc2ccc(O)cc2)cc1",                   # Bisdemethoxycurcumin

    # ── Alkaloids ──
    "COc1cc2c(cc1OC)[n+]1ccc3cc4c(cc3c1CC2)OCO4",                 # Berberine
    "COc1cc2c(cc1OC)CCN3C2CC(C1=C3COC1=O)C(=O)OC",               # Yohimbine-like core
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                               # Caffeine
    "CN1CCC23C4C5C=CC2(C1)C3=C(C=C5)C(=C4O)OC",                  # Morphine-like core
    "O=C1OC2CC3C4=C(C=CC=C4)CCN3CC2N1C5=CC=CC=C5",               # Aporphine-like core

    # ── Terpenoids / macrolides ──
    "CC1(C)OC2C3C(=O)OC4C(OO5)C3C5C2C4O1",                        # Artemisinin (approximate)
    "CC1OCCCC(=O)C1",                                              # Macrolide-like lactone core
    "CC1(C)CCC2C(C1=O)C3(C)CCC4C5(C)CCC(=O)C(C)(C)C5CCC4C3CC2",  # Limonoid-like core
    "O=C1OC2CC3C4C(C3(C)C)CCC4C2(C)C1",                           # Sesquiterpene lactone core

    # ── Antibacterial / PBP2a-relevant chemotypes ──
    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",                # Penicillin core (6-APA)
    "CC1=C(C(=O)O)CSCC2C(=O)N3C(=O)C=C(C3C2=O)C(=O)O",            # Cephalosporin core (7-ACA)
    "CC1C2C(C(=O)N2C(=C1C(=O)O)C(=O)O)SC3CCNC3=O",               # Carbapenem core (thienamycin-like)
    "CC1(C)SC2C(N)C(=O)N2C1C(=O)O",                               # 6-APA (penicillin nucleus)
    "O=C1NC2C3SC(C)(C)C(NC3C2=O)C(=O)O",                          # Penam ring system

    # ── Phenolic / catechol ──
    "Oc1c(O)c(O)cc(C(=O)O)c1",                                    # Gallic acid
    "Oc1ccccc1C(=O)O",                                            # Salicylic acid
    "CC1=C(C=C(C=C1)O)O",                                         # Hydroquinone
    "Oc1ccc(CCc2ccc(O)cc2)cc1",                                   # 4,4'-biphenol

    # ── Macrocyclic / complex ──
    "COc1cc2c(cc1OC)C(=O)C3=C(O)C=CC(=C3O2)C",                   # Rottlerin
    "COc1cc2c(cc1OC)C3CC4=C(C=C(C=C4)OC)CCN3C2",                 # Papaverine-like
    "CC1(C)Oc2ccccc2C(=O)N1",                                     # Dihydrobenzoxazinone
    "O=C1NC2=CC=CC=C2C(=O)N1",                                     # Isatin
    "CCCCCCCCCCCC(=O)O",                                           # Lauric acid (fatty acid)
    "O=C1CCCC2=C1C=CC=C2",                                        # Tetralone
    "CC1=CC(=O)C2=C(O1)C(=C(C=C2)O)O",                            # Chromone core
    "C1=CC=C2C(=C1)C3=CC=CC=C3C2",                                 # Fluorene
    "O=C1CCN2CC3=CC=CC=C3CC2C1",                                  # Benzazepinone
    "CC1(C)OCC2C3C(C2O1)C(=O)OC3C4=COC=C4",                       # Spiroketal core
    "O=C1OC2=C(C3=C(C=C2)C=CC=C3)C=C1",                            # Coumarin derivative
    "CC(=O)Nc1ccc(O)cc1",                                          # Paracetamol (amide phenol)
    "O=c1[nH]c(=O)n(Cc2ccccc2)cc1C=Cc3ccc(O)cc3",                 # Styrylxanthine
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


# External library CSV env var. When set, the file (columns ``smiles``,
# ``compound_id``) is merged into the generated library *before* filtering,
# reusing the existing input_csv logic. This lets users inject curated/de novo
# compounds without editing source.
EXTERNAL_LIB_CSV_ENV = "AUTOANTIBIOTIC_LIB_CSV"

# Known β-lactam / PBP2a-binding fragments used to bias BRICS recombination:
# their BRICS fragments are allowed into the fragment pool so the recombinant
# library is enriched toward credible PBP2a-binding chemotypes — and, because
# most literature hits are allosteric, a broader seed chemotype set raises the
# chance of an active-site competitor surviving enrichment. Defaults to
# ceftaroline & meropenem (from CONTROL_SMILES) plus a cephalosporin-core
# fragment and two public ceftaroline-class β-lactam cores; overridable via
# ``seed_smiles``. BRICSBuild logic and minFragmentSize (6) are unchanged.
_CEPHALOSPORIN_CORE = "CC1=C(C(=O)O)CSCC2C(=O)N3C(=O)C=C(C3C2=O)C(=O)O"
_PENAM_CORE = "CC1(C)SC2C(N)C(=O)N2C1C(=O)O"
_CARBAPENEM_CORE = "CC1C2C(C(=O)N2C(=C1C(=O)O)C(=O)O)SC3CCNC3=O"

# Non-β-lactam PBP2a-related chemotypes added to the seed pool to broaden the
# recombinant library beyond the β-lactam space. These are real, public SMILES
# representing (a) a macolactin-A-like pyridone fragment, (b) a corbomycin-like
# aglycone, and (c) a public non-β-lactam allosteric PBP2a chemotype. None are
# β-lactams, so they enrich the non-β-lactam novelty axis of the screen.
_MACOLACTIN_FRAGMENT = "CC1=CC(=O)NC(=O)1C2=CC=CC=C2"
_CORBOMYCIN_AGLYCONE = "COc1ccc2c(c1)OC(C)C(=O)N2"
_NON_BETA_LACTAM_PBP2A = "CN1C(=O)C(N=C1C(=O)O)SCC2=CC=CC=C2"

DEFAULT_SEED_SMILES = list(CONTROL_SMILES.values()) + [
    _CEPHALOSPORIN_CORE,
    _PENAM_CORE,
    _CARBAPENEM_CORE,
    _MACOLACTIN_FRAGMENT,
    _CORBOMYCIN_AGLYCONE,
    _NON_BETA_LACTAM_PBP2A,
]


def generate_candidate_library(
    target_count: int = 500,
    seed: int = RANDOM_SEED,
    input_csv: Optional[str] = None,
    input_sdf: Optional[str] = None,
    seed_smiles: Optional[List[str]] = None,
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
        seed_smiles: Optional list of known-binder SMILES whose fragments are
            added to the BRICS fragment pool to bias recombination toward
            credible PBP2a-binding chemotypes. Defaults to ceftaroline &
            meropenem (from ``CONTROL_SMILES``); pass ``[]`` to disable. The
            env var ``AUTOANTIBIOTIC_LIB_CSV``, if set, is merged in as if
            passed via ``input_csv`` (in addition to any explicit argument).

    Returns:
        List of CompoundRecord objects (SMILES only, no computed props yet).
    """
    log.info("─── Phase 2: Library Generation ───")

    # ── External seed CSV (AUTOANTIBIOTIC_LIB_CSV) ──
    # If the env var points to a CSV, read its SMILES as *additional seed
    # molecules* and decompose them into the BRICS fragment pool so BRICS
    # recombination AUGMENTS the seeds (rather than replacing the library).
    # This is distinct from an explicit ``input_csv`` (or ``--library``), which
    # is a full library that skips BRICS entirely. When an explicit
    # ``input_csv`` was given we prefer it and leave the env var untouched so
    # the explicit library still drives the run. Missing/invalid env CSV is
    # skipped gracefully (warning only) so the pipeline keeps running.
    external_csv = os.environ.get(EXTERNAL_LIB_CSV_ENV)
    if external_csv and input_csv is None:
        if os.path.exists(external_csv):
            log.info(
                f"  Reading external seed CSV from env "
                f"{EXTERNAL_LIB_CSV_ENV}: {external_csv} (BRICS augmentation)"
            )
            try:
                _ext_df = pd.read_csv(external_csv)
                _ext_cols = {str(c).strip().lower() for c in _ext_df.columns}
                if "smiles" in _ext_cols:
                    extra_seeds = [
                        str(s).strip() for s in _ext_df["smiles"].tolist()
                    ]
                    extra_seeds = [
                        s for s in extra_seeds
                        if s and s.lower() not in ("nan", "none")
                    ]
                    if extra_seeds:
                        if seed_smiles is None:
                            seed_smiles = list(DEFAULT_SEED_SMILES)
                        seed_smiles = seed_smiles + extra_seeds
                        log.info(
                            f"  Added {len(extra_seeds)} seed SMILES from "
                            f"external CSV to BRICS fragment pool."
                        )
                else:
                    log.warning(
                        f"  {EXTERNAL_LIB_CSV_ENV} CSV has no 'smiles' column; "
                        f"ignoring."
                    )
            except Exception as exc:
                log.warning(
                    f"  Failed to read {EXTERNAL_LIB_CSV_ENV} CSV ({exc}); "
                    f"ignoring."
                )
        else:
            log.warning(
                f"  {EXTERNAL_LIB_CSV_ENV} set but file not found "
                f"({external_csv}); ignoring external merge."
            )

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
                fragments = BRICS.BRICSDecompose(mol, minFragmentSize=6)
                for frag_smi in fragments:
                    frag_mol = Chem.MolFromSmiles(frag_smi)
                    if frag_mol is not None and _count_atoms(frag_mol) >= 6:
                        all_fragments.add(frag_smi)
            except Exception:
                continue

        # ── Known-binder fragment seeding (optional bias) ──
        # Decompose the supplied known binders (default: ceftaroline, meropenem)
        # and allow their BRICS fragments into the pool so recombination is
        # biased toward credible PBP2a-binding chemotypes. Seeds are added to
        # the same fragment set; ``None`` keeps the default seeds, ``[]``
        # disables seeding entirely.
        if seed_smiles is None:
            seed_smiles = DEFAULT_SEED_SMILES
        for smi in seed_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                for frag_smi in BRICS.BRICSDecompose(mol, minFragmentSize=6):
                    frag_mol = Chem.MolFromSmiles(frag_smi)
                    if frag_mol is not None and _count_atoms(frag_mol) >= 6:
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

    log.info(f"  Generated {len(frag_mols)} unique fragments (>=6 heavy atoms).")

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

    # Recombine fragments to create novel analogs via BRICSBuild
    seen_smiles = set()
    records = []
    import random as _random
    _random.seed(seed)

    log.info(f"  Building recombinant library via BRICS.BRICSBuild (target ≤ {target_count})…")

    # Multiple shuffled passes to maximise chemical diversity from the fragment pool.
    # BRICSBuild enumeration order is deterministic; shuffling the fragment list
    # changes the build order, exposing different recombination paths. If the
    # first pass does not reach target_count, subsequent passes resample the
    # remaining unseen combination space with fresh shuffles.
    # Cap passes at 5 to keep runtime bounded even for very large target_counts.
    max_passes = min(5, max(1, target_count // max(1, len(frag_mols) * 2)))
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
            seen_smiles.add(smi)

            cid = f"AA-{len(records):04d}"
            records.append(CompoundRecord(
                compound_id=cid,
                smiles=smi,
                mol=product,
            ))

            if len(records) % 100 == 0:
                log.info(f"  Generated {len(records)} / {target_count} candidates (pass {_pass + 1}/{max_passes})…")

            if len(records) >= target_count:
                break
        if len(records) >= target_count:
            break
        if len(records) == records_before:
            log.info(f"  Pass {_pass + 1}/{max_passes}: no new compounds (combinatorial space exhausted).")
            break
        log.info(f"  Pass {_pass + 1}/{max_passes}: {len(records)} unique compounds so far; reshuffling…")

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
