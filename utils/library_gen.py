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
from rdkit.Chem import BRICS, rdMolDescriptors

from config.constants import RANDOM_SEED

# Synthetic Accessibility (SA) score from the RDKit Contrib collection. Imported
# lazily/guarded so that a missing contrib directory never breaks library
# generation (the SA score is purely descriptive and non-essential downstream).
try:
    from rdkit.Chem import RDConfig
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
    _HAVE_SA_SCORER = True
except Exception:  # pragma: no cover - depends on RDKit contrib availability
    sascorer = None
    _HAVE_SA_SCORER = False

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
    # Mechanism-relevant human off-target energies (docked in the simplified
    # pipeline).
    human_trypsin_energy: Optional[float] = None
    human_ces1_energy: Optional[float] = None
    off_target_risk: bool = False

    # Phase 3.5 — Negative selection: the most negative (strongest) human
    # off-target binding energy observed across the docked human panel
    # (kcal/mol). Surfaced in the CSV report.
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

    # Path to the allosteric-site Vina docked pose (PDBQT), populated during
    # screening so allosteric interaction analysis need not re-dock.
    allosteric_docked_pdbqt: Optional[str] = None

    # Interaction fingerprint (dict returned by analyze_binding_interactions)
    # captured during Phase 4 so reporting can expose per-residue H-bond flags
    # without re-parsing the docked pose.
    interactions: Optional[dict] = None

    # Allosteric interaction fingerprint (dict returned by
    # analyze_allosteric_interactions), captured during Phase 4 so reporting
    # can expose allosteric-contact flags without re-parsing the docked pose.
    allosteric_interactions: Optional[dict] = None
    allosteric_contact: bool = False

    si_vs_ceftaroline: Optional[float] = None

    # Final reported SI tier label override. The report normally derives the
    # tier from selectivity_index, but when a candidate is kept only as a
    # below-threshold filler (SI < SI_PROMISING_THRESHOLD) this is set to
    # "Below gate" so the CSV is unambiguous (paper §A3).
    report_tier: Optional[str] = None

    # Phase C — synthetic accessibility & physicochemical descriptors computed
    # during library generation. ``sa_score`` is the Ertl SA score (lower = easier
    # to synthesise); ``tpsa`` is the topological polar surface area; ``frac_csp3``
    # is the fraction of sp3 carbons; ``num_rotatable_bonds`` is the rotatable-bond
    # count. All default to None when the SMILES cannot be parsed.
    sa_score: Optional[float] = None
    tpsa: Optional[float] = None
    frac_csp3: Optional[float] = None
    num_rotatable_bonds: Optional[int] = None

    # Vina score sanity gate: flagged when docking energy < -11.0 kcal/mol
    suspect_score: bool = False
    si_provisional: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAFFOLDS & CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

# 12 scaffolds derived from KNOWN PBP2a allosteric inhibitors and diverse
# drug-like heterocycles. These replace the earlier natural-product scaffolds
# that led to chemotype bias.
PBP2A_SCAFFOLDS = [
    # ── Tan 2012 oxadiazoles (3 scaffolds) ──
    "O=C(Nc1ccc(-c2nc(C3=CC=C(O)C=C3)no2)cc1)c1ccccc1",            # Tan oxadiazole core 1
    "O=C(O)c1ccc(C2=NOC(=N2)c3ccc(NC(=O)c4ccco4)cc3)cc1",         # Tan oxadiazole core 2
    "CCCC(=O)Nc1ccc(-c2nc(C3=CC=C(C(=O)O)C=C3)no2)cc1",           # Tan oxadiazole core 3

    # ── Troczi 2013 benzothiazole/quinazolinone hits (3 scaffolds) ──
    "O=C(Nc1ccc(-c2nc3ccccc3s2)cc1)c1ccc(O)cc1",                   # Benzothiazole amide
    "O=C1Nc2ccc(-c3ccccc3)cc2C(=O)N1c1ccccc1",                     # Quinazolinone core
    "O=C(Nc1ccc(-c2nc3ccccc3[nH]2)cc1)c1ccccc1",                   # Benzoxazole/benzimidazole

    # ── Simonet 2021 allosteric-site fragments (2 scaffolds) ──
    "O=C(Nc1ccc(O)cc1)C1C2CC3CC(C2)CC1C3",                         # Adamantane carboxamide
    "O=C(Nc1ccccc1)c1ccc(C(=O)Nc2ccccc2)cc1",                      # Terephthalamide

    # ── Diverse drug-like heterocycles (4 scaffolds) ──
    "O=c1[nH]c2ccccc2n1-c1ccccc1",                                  # Indole/benzimidazole
    "O=c1cc(-c2ccccc2)nc2[nH]ccn12",                                 # Triazolopyridine
    "O=C1CSC(=O)N1c1ccccc1",                                        # Thiazolidinone
    "O=C1C=C(c2ccccc2)n2ccnc21",                                    # Pyrazolopyrimidine
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


def _enrich_record_properties(rec: "CompoundRecord") -> "CompoundRecord":
    """
    Compute SA_Score, TPSA, Fraction_CSP3 and Num_Rotatable_Bonds for *rec*.

    Populates ``rec.sa_score``, ``rec.tpsa``, ``rec.frac_csp3`` and
    ``rec.num_rotatable_bonds`` in place (best-effort; any uncomputable value
    stays ``None``). Called for every :class:`CompoundRecord` produced by
    :func:`generate_candidate_library` so the CSV report can surface these
    drug-likeness descriptors (paper §2.6, §3).
    """
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

# ── Pharmacophore pre-filter ──
# Each generated compound must have ≥1 H-bond donor OR acceptor AND
# ≥2 aromatic/hydrophobic rings. Reject otherwise.
def _passes_pharmacophore_filter(mol: Chem.Mol) -> bool:
    from rdkit.Chem import Descriptors, rdMolDescriptors
    try:
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        if hbd + hba < 1:
            return False
        ring_info = rdMolDescriptors.CalcNumAromaticRings(mol)
        # Count all rings (aromatic + aliphatic) as hydrophobic rings
        all_rings = rdMolDescriptors.CalcNumRings(mol)
        if ring_info + (all_rings - ring_info) < 2:
            return False
        return True
    except Exception:
        return False


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
    seed_smiles: Optional[List[str]] = None,
) -> List[CompoundRecord]:
    """
    Phase 2.1 — Generate a diverse library.

    If *input_csv* is provided, the library is read directly from that CSV
    file (expected columns: ``smiles``, ``compound_id``) and the BRICS
    scaffold-generation logic is skipped entirely.

    Otherwise, a library is generated by BRICS decomposition of natural
    product scaffolds, fragment recombination, and expansion.

    Args:
        target_count: Desired number of compounds (~500).
        seed: Random seed for reproducibility.
        input_csv: Optional path to an external compound library CSV.
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
        return [_enrich_record_properties(r) for r in records]

    all_scaffolds = PBP2A_SCAFFOLDS
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
                fragments = BRICS.BRICSDecompose(mol, minFragmentSize=4)
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
                for frag_smi in BRICS.BRICSDecompose(mol, minFragmentSize=4):
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
        return [_enrich_record_properties(r) for r in candidates]

    # Scaffold family definitions for capping (15% per family of target_count)
    SCAFFOLD_FAMILIES = {
        0: list(range(0, 3)),     # Tan 2012 oxadiazoles (scaffolds 0-2)
        1: list(range(3, 6)),     # Troczi 2013 (scaffolds 3-5)
        2: list(range(6, 8)),     # Simonet 2021 (scaffolds 6-7)
        3: [8],                   # Indole/benzimidazole (scaffold 8)
        4: [9],                   # Triazolopyridine (scaffold 9)
        5: [10],                  # Thiazolidinone (scaffold 10)
        6: [11],                  # Pyrazolopyrimidine (scaffold 11)
    }
    FAMILY_MAX = max(1, int(0.15 * target_count))

    # Precompute scaffold fingerprints for family assignment
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    scaffold_fps = []
    for smi in PBP2A_SCAFFOLDS:
        m = Chem.MolFromSmiles(smi)
        if m:
            scaffold_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048))
        else:
            scaffold_fps.append(None)

    def _assign_family(mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        best_sim = -1
        best_idx = 0
        for i, s_fp in enumerate(scaffold_fps):
            if s_fp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fp, s_fp)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        for fam, indices in SCAFFOLD_FAMILIES.items():
            if best_idx in indices:
                return fam
        return 0

    # Recombine fragments to create novel analogs via BRICSBuild
    seen_smiles = set()
    records = []
    family_counts = {f: 0 for f in SCAFFOLD_FAMILIES}
    import random as _random
    _random.seed(seed)

    log.info(f"  Building recombinant library via BRICS.BRICSBuild (target ≤ {target_count})…")
    log.info(f"  Scaffold family cap: {FAMILY_MAX} per family.")

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

            # Pharmacophore pre-filter
            if not _passes_pharmacophore_filter(product):
                continue

            # MW 250-500
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(product)
            if mw < 250 or mw > 500:
                continue

            # SA < 4.0
            if _HAVE_SA_SCORER and sascorer is not None:
                try:
                    sa = sascorer.calculateScore(product)
                    if sa >= 4.0:
                        continue
                except Exception:
                    pass

            # QED > 0.5
            from rdkit.Chem import QED
            try:
                qed_val = QED.qed(product)
                if qed_val <= 0.5:
                    continue
            except Exception:
                continue

            # No beta-lactam
            from config.constants import BETA_LACTAM_SMARTS
            lactam_pat = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)
            if lactam_pat and product.HasSubstructMatch(lactam_pat):
                continue

            # No PAINS
            from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
            pains_params = FilterCatalogParams()
            pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            pains_cat = FilterCatalog(pains_params)
            if pains_cat.HasMatch(product):
                continue

            # Assign to family and cap
            fam = _assign_family(product)
            if family_counts[fam] >= FAMILY_MAX:
                continue
            family_counts[fam] += 1

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

    # Enforce pairwise Morgan Tanimoto ≤ 0.35 across the library
    log.info(f"  Enforcing pairwise Tanimoto ≤ 0.35 across {len(records)} candidates…")
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    tanimoto_threshold = 0.35
    diversity_records = []
    diversity_fps = []
    for rec in records:
        if rec.mol is None:
            rec.mol = Chem.MolFromSmiles(rec.smiles)
        if rec.mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(rec.mol, radius=2, nBits=2048)
        is_diverse = all(
            DataStructs.TanimotoSimilarity(fp, existing_fp) <= tanimoto_threshold
            for existing_fp in diversity_fps
        )
        if is_diverse:
            diversity_records.append(rec)
            diversity_fps.append(fp)
    records = diversity_records
    log.info(f"  After Tanimoto filtering: {len(records)} diverse candidates.")

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
    return [_enrich_record_properties(r) for r in records]



