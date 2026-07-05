from __future__ import annotations

import itertools
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import (
    AllChem,
    BRICS,
    ChemicalFeatures,
    Crippen,
    Descriptors,
    QED,
    rdDistGeom,
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.DataStructs import TanimotoSimilarity

from .config import CONFIG
from .models import CompoundRecord
from .io_utils import log

try:
    from .analysis import _get_ml_admet_predictor, predict_herg_ml
    _HAVE_ML_ADMET = True
except ImportError:
    _HAVE_ML_ADMET = False

try:
    from tqdm import tqdm as _tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False
    _tqdm = lambda x, **kw: x

try:
    from sascore import compute_sa_score as _compute_sa_score
    _HAVE_SA_SCORE = True
except ImportError:
    _compute_sa_score = None
    _HAVE_SA_SCORE = False

_HAVE_TOX_ALERTS = True

_HAVE_PHARMACOPHORE = True
try:
    _fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    _PHARM_FACTORY = ChemicalFeatures.BuildFeatureFactory(_fdef)
except Exception:
    _PHARM_FACTORY = None
    _HAVE_PHARMACOPHORE = False


def _count_atoms(mol: Chem.Mol) -> int:
    """Heavy-atom count for a molecule."""
    return mol.GetNumHeavyAtoms()


def _validate_mol(smiles: str) -> Optional[Chem.Mol]:
    """Validate a SMILES string by parsing and sanitising."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol


def _brics_recombination(
    frag_mols: List[Chem.Mol],
    target_count: int,
    seen_smiles: set,
    seed: int = CONFIG.random_seed,
) -> Tuple[List[CompoundRecord], set]:
    """Recombine BRICS fragments using BRICSBuild, then pick a diverse subset via MaxMin.

    Uses RDKit's BRICSBuild to enumerate recombination products from the
    provided fragment pool.  Duplicate products (by canonical SMILES) and
    acyclic molecules are discarded.  If the resulting pool exceeds the
    requested count, MaxMin diversity picking selects the most diverse
    subset based on Morgan fingerprints.

    Args:
        frag_mols: RDKit Mol objects representing BRICS-compatible fragments.
        target_count: Desired number of compounds to return.
        seen_smiles: Set of SMILES strings already used (e.g. from
            scaffold entries); duplicates are skipped.
        seed: Random seed for fragment shuffling and MaxMin picking.

    Returns:
        Tuple of ``(records, updated_seen_smiles)`` where *records* is
        the list of selected ``CompoundRecord`` objects and
        *updated_seen_smiles* includes the newly generated SMILES.
    """
    rng = np.random.default_rng(seed)

    pool_mult = CONFIG.diversity_pool_multiplier
    max_products = target_count * pool_mult * 4
    target_pool = target_count * pool_mult

    shuffled = list(frag_mols)
    rng.shuffle(shuffled)

    builder = BRICS.BRICSBuild(shuffled)

    # Generator-based pool building — yields records one at a time
    def _product_generator():
        n_produced = 0
        for product in itertools.islice(builder, max_products):
            if product is None:
                continue
            try:
                Chem.SanitizeMol(product)
                smi = Chem.MolToSmiles(product)
            except Exception:
                continue
            if smi in seen_smiles:
                continue
            ring_info = product.GetRingInfo()
            if ring_info.NumRings() == 0:
                continue
            seen_smiles.add(smi)
            rec = CompoundRecord(
                compound_id=f"AA-{n_produced:04d}",
                smiles=smi,
                mol=product,
            )
            n_produced += 1
            yield rec
            if n_produced >= target_pool:
                break

    iterator = _tqdm(
        _product_generator(),
        desc="  BRICS recombination",
        total=target_pool,
        disable=not _HAVE_TQDM,
    )

    pool_records: List[CompoundRecord] = list(iterator)

    if not pool_records:
        return [], seen_smiles

    log.info(f"  BRICS pool size: {len(pool_records)}")

    if len(pool_records) <= target_count:
        return pool_records, seen_smiles

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            r.mol, radius=CONFIG.morgan_radius, nBits=CONFIG.morgan_nbits,
        )
        for r in pool_records
    ]

    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
    picker = MaxMinPicker()
    pick_ids = picker.LazyBitVectorPick(
        fps, len(fps), target_count, seed=seed,
    )

    records = [pool_records[i] for i in pick_ids]
    log.info(
        f"  MaxMin selected {len(records)} diverse compounds "
        f"from pool of {len(pool_records)}."
    )
    return records, seen_smiles


def _generate_records(
    target_count: int,
    seed: int,
) -> Iterator[CompoundRecord]:
    """Generator that yields CompoundRecord objects for the library.

    Used internally by :func:`generate_candidate_library` to allow
    streaming when the target count is large.
    """
    all_scaffolds: List[str] = CONFIG.natural_product_scaffolds + CONFIG.additional_scaffolds
    scaffold_mols: List[Chem.Mol] = []
    for smi in all_scaffolds:
        mol = _validate_mol(smi)
        if mol is not None:
            scaffold_mols.append(mol)

    log.info(f"  Loaded {len(scaffold_mols)} / {len(all_scaffolds)} valid scaffolds.")

    if not scaffold_mols and not CONFIG.brics_building_blocks:
        log.error("  ✗  No valid scaffolds or building blocks. Aborting library generation.")
        return

    decomposed_frags: set = set()
    for mol in scaffold_mols:
        try:
            fragments = BRICS.BRICSDecompose(mol, minFragmentSize=CONFIG.brics_min_fragment_size)
            for frag_smi in fragments:
                frag_mol = _validate_mol(frag_smi)
                if frag_mol is not None and _count_atoms(frag_mol) >= CONFIG.brics_min_fragment_size:
                    decomposed_frags.add(frag_smi)
        except Exception:
            continue

    log.info(f"  Decomposed {len(decomposed_frags)} unique BRICS fragments from scaffolds.")

    all_building_blocks: set = set()
    for smi in CONFIG.brics_building_blocks:
        mol = _validate_mol(smi)
        if mol is not None:
            all_building_blocks.add(smi)

    log.info(f"  Loaded {len(all_building_blocks)} pre-built BRICS building blocks.")

    all_frag_smis: set = decomposed_frags | all_building_blocks
    frag_mols: List[Chem.Mol] = []
    for smi in all_frag_smis:
        m = _validate_mol(smi)
        if m is not None:
            frag_mols.append(m)

    log.info(f"  Total BRICS-compatible fragments: {len(frag_mols)}")

    seen_smiles: set = set()
    scaffold_count = 0

    for smi in all_scaffolds:
        mol = _validate_mol(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen_smiles:
            continue
        seen_smiles.add(canon)
        yield CompoundRecord(
            compound_id=f"SCAFFOLD_{scaffold_count:04d}",
            smiles=canon,
            mol=mol,
        )
        scaffold_count += 1

    if len(frag_mols) >= 2:
        recon_records, seen_smiles = _brics_recombination(
            frag_mols, target_count, seen_smiles, seed,
        )
        for rec in recon_records:
            yield rec
        log.info(f"  BRICS recombination yielded {len(recon_records)} novel compounds.")
    else:
        log.warning(
            f"  Too few fragments ({len(frag_mols)}) for recombination. "
            "Using scaffold enumeration only."
        )

    for name, smi in CONFIG.control_smiles.items():
        mol = _validate_mol(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen_smiles:
            yield CompoundRecord(
                compound_id=f"CTRL_{name}",
                smiles=canon,
                mol=mol,
            )
            seen_smiles.add(canon)


def generate_candidate_library(
    target_count: int = CONFIG.library_target_count,
    seed: int = CONFIG.random_seed,
) -> Union[List[CompoundRecord], Iterator[CompoundRecord]]:
    """Phase 2.1 — Library Generation via BRICS fragment recombination.

    Returns a list of CompoundRecord objects. When *target_count* exceeds
    :attr:`CONFIG.library_generator_threshold` (default 1000), returns a
    generator that yields records lazily to reduce memory pressure.
    """
    log.info("─── Phase 2: Library Generation ───")

    use_generator = target_count > CONFIG.library_generator_threshold
    gen = _generate_records(target_count, seed)

    if use_generator:
        log.info(f"  Streaming generator mode (target > {CONFIG.library_generator_threshold} compounds).")
        return gen

    records = list(gen)
    log.info(f"  Library generation complete: {len(records)} compounds.")
    if len(records) < 300:
        log.warning(
            f"  ⚠  Only {len(records)} compounds generated (target ≥300). "
            "Consider adding more scaffolds or building blocks."
        )
    return records


def _get_pharmacophore_points_3d(
    mol: Chem.Mol,
    conf_id: int = -1,
) -> List[Dict[str, Any]]:
    """Extract 3D pharmacophore feature points from a molecule.

    Uses the RDKit ``ChemicalFeatures`` factory to locate Donor, Acceptor,
    Hydrophobe and Aromatic features, then computes the centroid coordinate
    of each feature's atoms in the given conformer.

    Args:
        mol: Molecule with a 3D conformer.
        conf_id: Conformer ID (default: last conformer).

    Returns:
        List of dicts, each with keys ``type`` (feature family),
        ``pos`` (3-D np.ndarray centroid), and ``atom_ids`` (list of atom
        indices belonging to the feature).
    """
    if _PHARM_FACTORY is None:
        return []
    feats = _PHARM_FACTORY.GetFeaturesForMol(mol)
    points: List[Dict[str, Any]] = []
    conf = mol.GetConformer(conf_id)
    for feat in feats:
        ftype = feat.GetFamily()
        if ftype not in ("Donor", "Acceptor", "Hydrophobe", "Aromatic"):
            continue
        atoms = feat.GetAtomIds()
        if not atoms:
            continue
        pos = np.zeros(3)
        for aid in atoms:
            pt = conf.GetAtomPosition(aid)
            pos += np.array([pt.x, pt.y, pt.z])
        pos /= len(atoms)
        points.append({"type": ftype, "pos": pos, "atom_ids": atoms})
    return points


def _build_allosteric_pharmacophore() -> Optional[Dict[str, Any]]:
    """Build a pharmacophore query model based on PBP2a allosteric pocket features.

    When ``CONFIG.pharmacophore_ref_ligand_smi`` is set, a 3-D pharmacophore
    model is constructed from the reference ligand (generated with ETKDGv3).
    Otherwise the method falls back to the original 2-D feature-counting
    approach based on the three key allosteric residues:
      1. H-bond donor  – TYR159 (phenolic OH)
      2. H-bond acceptor – ALA237 (backbone carbonyl)
      3. Hydrophobic    – MET241 (side chain)

    Returns:
        A dict with mode-specific keys, or ``None`` if the RDKit feature
        factory cannot be loaded.
    """
    if not _HAVE_PHARMACOPHORE or _PHARM_FACTORY is None:
        return None

    # ── 3-D pharmacophore from reference ligand ──
    if CONFIG.pharmacophore_ref_ligand_smi:
        ref_mol = Chem.MolFromSmiles(CONFIG.pharmacophore_ref_ligand_smi)
        if ref_mol is not None:
            ref_mol_3d = Chem.RWMol(ref_mol)
            ref_mol_3d = Chem.AddHs(ref_mol_3d)
            params = Chem.rdDistGeom.ETKDGv3()
            params.randomSeed = CONFIG.random_seed
            if Chem.rdDistGeom.EmbedMolecule(ref_mol_3d, params) >= 0:
                AllChem.MMFFOptimizeMolecule(ref_mol_3d, maxIters=500)
                ref_features = _get_pharmacophore_points_3d(ref_mol_3d)
                if ref_features:
                    return {
                        "ref_mol": ref_mol_3d,
                        "ref_features": ref_features,
                        "feat_types": list({f["type"] for f in ref_features}),
                        "mode": "3d",
                    }
        log.warning("  Could not build 3-D pharmacophore from reference SMILES; "
                     "falling back to 2-D feature check.")

    # ── 2-D feature-based fallback ──
    return {
        "feat_types": ["Donor", "Acceptor", "Hydrophobe"],
        "residue_map": {
            "TYR159": "Donor",
            "ALA237": "Acceptor",
            "MET241": "Hydrophobe",
        },
        "mode": "2d",
    }


def check_pharmacophore_match(
    mol: Chem.Mol,
    query: Optional[Dict[str, Any]] = None,
    min_matches: int = 2,
    tolerance: float = 2.0,
) -> bool:
    """Check whether *mol* satisfies at least *min_matches* pharmacophore features.

    In **3-D mode** (when ``CONFIG.pharmacophore_ref_ligand_smi`` is set):
      1. A 3-D conformer for *mol* is generated with ETKDGv3.
      2. The conformer is aligned to the reference ligand pharmacophore
         using O3A (Open3DAlign) or maximum common substructure matching.
      3. Pharmacophore feature points are extracted from the aligned
         conformer and matched by type to the reference features.
      4. The RMSD of the matched feature pairs is computed; the match
         passes if ``RMSD < CONFIG.pharmacophore_rmsd_threshold``.

    In **2-D mode** (fallback): feature types are counted via the RDKit
    ``ChemicalFeatures`` factory.

    Args:
        mol: The candidate molecule to check.
        query: A pharmacophore model dict from
            :func:`_build_allosteric_pharmacophore`.  If ``None`` the
            allosteric model is built internally.
        min_matches: Minimum number of feature types that must be present
            (used only in 2-D mode).
        tolerance: Distance tolerance for feature matching in 3-D mode
            (reserved for future use; defaults to
            ``CONFIG.pharmacophone_rmsd_threshold`` for 3-D).

    Returns:
        ``True`` if the molecule passes the pharmacophore filter,
        ``False`` otherwise.
    """
    if query is None:
        query = _build_allosteric_pharmacophore()
    if query is None or not _HAVE_PHARMACOPHORE or _PHARM_FACTORY is None:
        return True  # pass-through when pharmacophore is unavailable

    mode = query.get("mode", "2d")

    # ── 3-D pharmacophore matching ──────────────────────────────────
    if mode == "3d":
        ref_mol = query["ref_mol"]
        ref_features = query["ref_features"]
        if not ref_features:
            return False

        # 1. Generate 3-D conformer for the candidate
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = Chem.rdDistGeom.ETKDGv3()
        params.randomSeed = CONFIG.random_seed
        if Chem.rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
            return False
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)

        # 2. Align candidate to reference
        try:
            from rdkit.Chem import rdMolAlign

            # Try MCS-based alignment first
            matches = mol_3d.GetSubstructMatch(ref_mol)
            if matches:
                atom_map = [(matches[i], i) for i in range(len(matches))]
                AllChem.AlignMol(mol_3d, ref_mol, atomMap=atom_map)
            else:
                matches = ref_mol.GetSubstructMatch(mol_3d)
                if matches:
                    atom_map = [(i, matches[i]) for i in range(len(matches))]
                    AllChem.AlignMol(mol_3d, ref_mol, atomMap=atom_map)
                else:
                    o3a = rdMolAlign.GetO3A(mol_3d, ref_mol)
                    o3a.Align()
        except Exception:
            try:
                AllChem.AlignMol(mol_3d, ref_mol)
            except Exception:
                pass

        # 3. Extract pharmacophore features from aligned candidate
        query_features = _get_pharmacophore_points_3d(mol_3d)
        if len(query_features) < min_matches:
            return False

        # 4. Match features by type (nearest-neighbour within each type)
        matched_distances: List[float] = []
        for ref_f in ref_features:
            best_d = float("inf")
            for qf in query_features:
                if qf["type"] == ref_f["type"]:
                    d = float(np.linalg.norm(ref_f["pos"] - qf["pos"]))
                    if d < best_d:
                        best_d = d
            if best_d < float("inf"):
                matched_distances.append(best_d)

        if len(matched_distances) < min_matches:
            return False

        rmsd = float(np.sqrt(np.mean(np.square(matched_distances))))
        return rmsd < CONFIG.pharmacophore_rmsd_threshold

    # ── 2-D feature-counting fallback ───────────────────────────────
    try:
        feats = _PHARM_FACTORY.GetFeaturesForMol(mol)
    except Exception:
        return True

    found: set = set()
    for feat in feats:
        ftype = feat.GetFamily()
        if ftype == "Donor":
            found.add("Donor")
        elif ftype == "Acceptor":
            found.add("Acceptor")
        elif ftype == "Hydrophobe":
            found.add("Hydrophobe")

    return len(found) >= min_matches


def generate_pharmacophore_aware_library(
    target_count: int = CONFIG.library_target_count,
    seed: int = CONFIG.random_seed,
    allosteric_pocket_coords: Optional[np.ndarray] = None,
) -> List[CompoundRecord]:
    """Generate a focused library enriched for PBP2a allosteric-site binding.

    Works by:
      1. Generating BRICS-recombined compounds via the standard pipeline.
      2. Filtering with :func:`check_pharmacophore_match` to retain only
         molecules that satisfy the pharmacophore filter.
      3. Falls back to standard library generation when pharmacophore
         resources are unavailable.

    When ``CONFIG.pharmacophore_ref_ligand_smi`` is set, the filter uses
    3-D alignment-based matching; otherwise the original 2-D feature-counting
    approach is used.

    Args:
        target_count: Desired number of output compounds.
        seed: Random seed for reproducibility.
        allosteric_pocket_coords: Optional (3, 3) array of Cα coordinates
            for the three allosteric residues (TYR159, ALA237, MET241).
            Used for informative logging only; the actual 3-D matching
            uses the reference ligand pharmacophore.

    Returns:
        List of pharmacophore-enriched ``CompoundRecord`` objects.
    """
    log.info("─── Pharmacophore-Aware Library Generation ───")
    query = _build_allosteric_pharmacophore()
    if query is None or not _HAVE_PHARMACOPHORE:
        log.warning("  Pharmacophore factory unavailable; falling back to standard library.")
        return list(generate_candidate_library(target_count, seed))

    pharm_feats = query["feat_types"]
    log.info(f"  Pharmacophore features: {pharm_feats}")
    if allosteric_pocket_coords is not None:
        log.info(f"  3D pocket coords provided ({allosteric_pocket_coords.shape[0]} residues).")

    standard_records = list(generate_candidate_library(target_count, seed))
    log.info(f"  Standard library size: {len(standard_records)}")

    passed: List[CompoundRecord] = []
    for rec in standard_records:
        mol = rec.mol
        if mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol
        if check_pharmacophore_match(
            mol, query,
            min_matches=CONFIG.pharmacophore_min_matches,
            tolerance=CONFIG.pharmacophore_tolerance,
        ):
            passed.append(rec)

    passed = passed[:target_count]
    log.info(f"  Pharmacophore-enriched library: {len(passed)} compounds (≥{CONFIG.pharmacophore_min_matches} feat. matches).")
    return passed


def _setup_toxicity_catalog() -> FilterCatalog:
    """Build an RDKit FilterCatalog for toxicity alerts."""
    tox_params = FilterCatalogParams()
    tox_params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    tox_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    return FilterCatalog(tox_params)


def _setup_reactive_catalog() -> Optional[FilterCatalog]:
    """Build an RDKit FilterCatalog for reactive / unstable group alerts."""
    try:
        rxn_params = FilterCatalogParams()
        rxn_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        return FilterCatalog(rxn_params)
    except Exception:
        return None


def _compute_strain_energy(mol: Chem.Mol) -> Optional[float]:
    """Compute the MMFF94 strain energy (kcal/mol) for a molecule.

    Generates a 3-D conformer with ETKDGv3, optimises with MMFF94, and
    returns the strain energy = initial energy - final energy.
    Returns ``None`` if 3-D embedding or FF setup fails.
    """
    try:
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = CONFIG.random_seed
        if rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
            return None
        props = AllChem.MMFFGetMoleculeProperties(mol_3d)
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol_3d, props, nonBondedThresh=100.0)
        if ff is None:
            return None
        initial = ff.CalcEnergy()
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
        final = ff.CalcEnergy()
        # Use absolute difference as a robust proxy for strain.
        # For most molecules initial > final (optimisation lowers energy).
        # When the force field cannot improve the geometry (e.g. cubane),
        # abs(initial - final) still captures excessive deformation energy.
        strain = abs(initial - final)
        return strain
    except Exception:
        return None


# ── Filter helpers ─────────────────────────────────────────────────


def _setup_filter_catalogs() -> tuple:
    """Build and return (pains_catalog, tox_catalog, reactive_catalog)."""
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains_catalog = FilterCatalog(pains_params)
    tox_catalog = _setup_toxicity_catalog() if _HAVE_TOX_ALERTS else None
    reactive_catalog = _setup_reactive_catalog()
    return pains_catalog, tox_catalog, reactive_catalog


def _build_ref_fingerprints() -> Dict[str, Any]:
    """Build Morgan fingerprints for all reference antibiotics."""
    ref_mols: Dict[str, Any] = {}
    for name, smi in CONFIG.reference_antibiotics.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ref_mols[name] = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=CONFIG.morgan_radius, nBits=CONFIG.morgan_nbits,
            )
    return ref_mols


def _filter_beta_lactam(
    record: CompoundRecord, mol: Chem.Mol, lactam_pattern: Chem.Mol,
) -> tuple[bool, str]:
    """Skip compounds with β-lactam substructure (unless control)."""
    if not record.compound_id.startswith("CTRL_") and mol.HasSubstructMatch(lactam_pattern):
        return False, "structural"
    return True, ""


def _filter_similarity(
    record: CompoundRecord, mol: Chem.Mol, ref_fps: Dict[str, Any],
    threshold: float,
) -> tuple[bool, str]:
    """Skip compounds too similar to known antibiotics."""
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=CONFIG.morgan_radius, nBits=CONFIG.morgan_nbits,
    )
    max_sim = 0.0
    for ref_fp in ref_fps.values():
        sim = TanimotoSimilarity(fp, ref_fp)
        max_sim = max(max_sim, sim)
    record.max_similarity = max_sim

    if max_sim >= threshold:
        return False, "similarity"
    return True, ""


def _filter_lipinski(
    record: CompoundRecord, mol: Chem.Mol,
) -> tuple[bool, str]:
    """Enforce Lipinski Rule-of-5 and QED threshold."""
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        lipinski_ok = (
            mw <= CONFIG.lipinski_mw_max
            and logp <= CONFIG.lipinski_logp_max
            and hbd <= CONFIG.lipinski_hbd_max
            and hba <= CONFIG.lipinski_hba_max
        )
        qed = QED.qed(mol)
    except Exception:
        return False, "admet"

    record.passes_lipinski = lipinski_ok
    record.qed_score = qed

    if not lipinski_ok or qed <= CONFIG.qed_threshold:
        return False, "admet"
    return True, ""


def _filter_pains(
    record: CompoundRecord, mol: Chem.Mol, pains_catalog: FilterCatalog,
) -> tuple[bool, str]:
    """Remove PAINS-alert compounds."""
    pains_match = pains_catalog.HasMatch(mol)
    record.passes_pains = not pains_match
    if pains_match:
        return False, "pains"
    return True, ""


def _filter_sa_score(
    record: CompoundRecord, mol: Chem.Mol,
) -> tuple[bool, str]:
    """Filter by synthetic accessibility score."""
    if _HAVE_SA_SCORE:
        try:
            sa_score = _compute_sa_score(mol)
            if sa_score > CONFIG.sa_score_threshold:
                return False, "sa_score"
        except Exception:
            pass
    return True, ""


def _filter_toxicity(
    record: CompoundRecord, mol: Chem.Mol, tox_catalog: Optional[FilterCatalog],
) -> tuple[bool, str]:
    """Remove compounds flagged by toxicity alerts."""
    if tox_catalog is not None:
        tox_matches = tox_catalog.GetMatches(mol)
        if tox_matches:
            return False, "toxicity"
    return True, ""


def _filter_reactive(
    record: CompoundRecord, mol: Chem.Mol, reactive_catalog: Optional[FilterCatalog],
) -> tuple[bool, str]:
    """Remove compounds with reactive / unstable groups (BRENK)."""
    if reactive_catalog is not None:
        rxn_matches = reactive_catalog.GetMatches(mol)
        if rxn_matches:
            return False, "reactive"
    return True, ""


def _filter_strain(
    record: CompoundRecord, mol: Chem.Mol,
) -> tuple[bool, str]:
    """Remove conformationally strained compounds."""
    strain = _compute_strain_energy(mol)
    if strain is not None and strain > CONFIG.strain_energy_threshold:
        return False, "strain"
    return True, ""


def _filter_ml_admet(
    record: CompoundRecord, mol: Chem.Mol,
) -> tuple[bool, str]:
    """Remove compounds flagged by the ML-ADMET predictor.

    Runs *after* Lipinski/QED but *before* PAINS and docking.
    Uses the same predictor instance as :func:`predict_admet_profile`.
    """
    if not CONFIG.use_ml_admet:
        return True, ""

    predictor = _get_ml_admet_predictor()
    if predictor is None:
        # ML unavailable — pass through (rule-based ADMET is separate)
        return True, ""

    try:
        herg_prob = predictor.predict_herg_probability(mol)
        if herg_prob is not None and herg_prob > CONFIG.ml_admet_herg_threshold:
            return False, "ml_admet"
    except Exception:
        pass

    return True, ""


def apply_filters(
    records: Union[List[CompoundRecord], Iterator[CompoundRecord]],
    similarity_threshold: float = CONFIG.similarity_threshold,
) -> List[CompoundRecord]:
    """Phase 2.2 — Apply structural, similarity, ADMET, PAINS, toxicity, and strain filters.

    Filter chain:
        1. Structural exclusion (β-lactam SMARTS).
        2. Similarity filter vs reference antibiotics.
        3. ADMET: Lipinski Rule of 5 + QED > 0.6.
        4. PAINS alerts via RDKit FilterCatalog.
        5. Synthetic Accessibility (SA Score ≤ 6.0).
        6. Toxicity alerts (mutagenicity / cardiotoxicity if available).
        7. Reactive group filter (BRENK catalog).
        8. 3D strain energy check (MMFF94 via ETKDGv3).
        9. Diversity check: if < 100 pass, relax similarity to 0.5.

    Returns filtered list of CompoundRecord.
    """
    log.info("─── Phase 2: Filtering ───")

    lactam_pattern = Chem.MolFromSmarts(CONFIG.beta_lactam_smarts)
    ref_fps = _build_ref_fingerprints()
    pains_catalog, tox_catalog, reactive_catalog = _setup_filter_catalogs()

    # Build filter pipeline as a list of (name, function) tuples
    filter_pipeline: List[tuple] = [
        ("structural", lambda r, m: _filter_beta_lactam(r, m, lactam_pattern)),
        ("similarity", lambda r, m: _filter_similarity(r, m, ref_fps, similarity_threshold)),
        ("admet", _filter_lipinski),
        ("ml_admet", _filter_ml_admet),
        ("pains", lambda r, m: _filter_pains(r, m, pains_catalog)),
        ("sa_score", _filter_sa_score),
        ("toxicity", lambda r, m: _filter_toxicity(r, m, tox_catalog)),
        ("reactive", lambda r, m: _filter_reactive(r, m, reactive_catalog)),
        ("strain", _filter_strain),
    ]

    skipped: Dict[str, int] = {name: 0 for name, _ in filter_pipeline}
    passed: List[CompoundRecord] = []

    for record in records:
        if record.mol is None:
            mol = Chem.MolFromSmiles(record.smiles)
            if mol is None:
                continue
            record.mol = mol
        mol = record.mol

        for name, filter_fn in filter_pipeline:
            ok, _ = filter_fn(record, mol)
            if not ok:
                skipped[name] += 1
                break
        else:
            passed.append(record)

    log.info(f"  Structural exclusion (β-lactam): {skipped['structural']} removed.")
    log.info(f"  Similarity filter (Tc < {similarity_threshold}): {skipped['similarity']} removed.")
    log.info(f"  ADMET filter (Lipinski + QED > 0.6): {skipped['admet']} removed.")
    if CONFIG.use_ml_admet:
        log.info(f"  ML-ADMET filter (hERG probability > {CONFIG.ml_admet_herg_threshold}): {skipped['ml_admet']} removed.")
    else:
        log.info("  ML-ADMET filter: disabled (use_ml_admet=False).")
    log.info(f"  PAINS filter: {skipped['pains']} removed.")
    if _HAVE_SA_SCORE:
        log.info(f"  SA Score filter (> {CONFIG.sa_score_threshold}): {skipped['sa_score']} removed.")
    else:
        log.info("  SA Score filter: skipped (sascore not installed).")
    if tox_catalog is not None:
        log.info(f"  Toxicity alerts: {skipped['toxicity']} removed.")
    else:
        log.info("  Toxicity alerts: skipped (RDKit Catalogs not available).")
    if reactive_catalog is not None:
        log.info(f"  Reactive group filter: {skipped['reactive']} removed.")
    else:
        log.info("  Reactive group filter: skipped (BRENK catalog unavailable).")
    log.info(f"  Strain energy filter (> {CONFIG.strain_energy_threshold} kcal/mol): {skipped['strain']} removed.")
    log.info(f"  Passed filters: {len(passed)} compounds.")

    if len(passed) < CONFIG.diversity_min_count and similarity_threshold < CONFIG.similarity_threshold_relaxed:
        log.warning(
            f"  Only {len(passed)} compounds passed strict filters (< {CONFIG.diversity_min_count}). "
            f"Relaxing similarity threshold to {CONFIG.similarity_threshold_relaxed} and re-running."
        )
        return apply_filters(records, similarity_threshold=CONFIG.similarity_threshold_relaxed)

    log.info("─── Phase 2 complete ───")
    return passed


def generate_grown_library(
    core_records: List[CompoundRecord],
    building_blocks: Optional[List[str]] = None,
    max_growth_steps: int = 1,
    target_per_core: int = 50,
) -> Iterator[CompoundRecord]:
    """Iteratively grow core fragments by attaching BRICS-compatible
    building blocks to reactive sites on the core.

    For each core record:

    1. Decompose the core into BRICS fragments to expose reactive
       (dummy-atom) sites.
    2. Combine the core fragments with the provided *building_blocks*
       using RDKit's :func:`BRICS.BRICSBuild`.
    3. Filter intermediate products by Lipinski Rule-of-5 and QED
       at each growth step to prevent combinatorial explosion.
    4. Yield each valid product as a ``CompoundRecord``.

    Only products that contain at least one core fragment as a
    substructure are retained, ensuring that the core scaffold is
    preserved.

    Parameters
    ----------
    core_records : list of CompoundRecord
        High-scoring core fragments from a previous pipeline run.
    building_blocks : list of str, optional
        BRICS-compatible building block SMILES (with dummy atoms).
        Defaults to ``CONFIG.brics_building_blocks``.
    max_growth_steps : int
        Number of iterative growth cycles (default 1).
    target_per_core : int
        Maximum number of grown compounds to yield per core (default 50).

    Yields
    ------
    CompoundRecord
        Each valid growth product.
    """
    bbs: List[str] = building_blocks if building_blocks is not None else CONFIG.brics_building_blocks
    bb_mols: List[Chem.Mol] = [
        m for m in (Chem.MolFromSmiles(s) for s in bbs) if m is not None
    ]
    if not bb_mols:
        log.warning("generate_grown_library: no valid building blocks.")
        return

    seen_smiles: set = set()
    compound_counter: int = 0
    rng = np.random.default_rng(CONFIG.random_seed)

    for core_rec in core_records:
        core_mol = core_rec.mol
        if core_mol is None:
            core_mol = Chem.MolFromSmiles(core_rec.smiles)
            if core_mol is None:
                continue
        core_smi = Chem.MolToSmiles(core_mol)
        seen_smiles.add(core_smi)

        # Decompose core to expose BRICS reactive sites
        core_frag_smiles: List[str] = list(
            BRICS.BRICSDecompose(core_mol, minFragmentSize=CONFIG.brics_min_fragment_size)
        )
        if not core_frag_smiles:
            continue

        core_frags: List[Chem.Mol] = [
            m for m in (Chem.MolFromSmiles(s) for s in core_frag_smiles)
            if m is not None
        ]
        if not core_frags:
            continue

        growth_mols: List[Chem.Mol] = [core_mol]

        for step in range(max_growth_steps):
            next_gen: List[Chem.Mol] = []
            for parent in growth_mols:
                parent_frags: List[Chem.Mol] = [
                    m for m in (Chem.MolFromSmiles(s) for s in list(
                        BRICS.BRICSDecompose(parent, minFragmentSize=CONFIG.brics_min_fragment_size)
                    )) if m is not None
                ]
                if not parent_frags:
                    parent_frags = core_frags

                pool = parent_frags + bb_mols
                rng.shuffle(pool)

                try:
                    builder = BRICS.BRICSBuild(pool)
                except Exception:
                    continue

                for product in itertools.islice(
                    builder, target_per_core * 20
                ):
                    if product is None:
                        continue
                    try:
                        Chem.SanitizeMol(product)
                        smi = Chem.MolToSmiles(product)
                    except Exception:
                        continue

                    if smi in seen_smiles:
                        continue

                    ring_info = product.GetRingInfo()
                    if ring_info.NumRings() == 0:
                        continue

                    # Ensure the core scaffold is preserved
                    has_core_fragment = any(
                        product.HasSubstructMatch(f) for f in core_frags
                    )
                    if not has_core_fragment:
                        continue

                    # Lipinski + QED filter
                    try:
                        mw = Descriptors.MolWt(product)
                        logp_val = Crippen.MolLogP(product)
                        hbd = Descriptors.NumHDonors(product)
                        hba = Descriptors.NumHAcceptors(product)
                        qed = QED.qed(product)
                    except Exception:
                        continue

                    lipinski_ok = (
                        mw <= CONFIG.lipinski_mw_max
                        and logp_val <= CONFIG.lipinski_logp_max
                        and hbd <= CONFIG.lipinski_hbd_max
                        and hba <= CONFIG.lipinski_hba_max
                    )
                    if not lipinski_ok or qed < CONFIG.qed_threshold:
                        continue

                    seen_smiles.add(smi)
                    rec = CompoundRecord(
                        compound_id=f"GROWN-{compound_counter:04d}",
                        smiles=smi,
                        mol=product,
                        qed_score=qed,
                        passes_lipinski=lipinski_ok,
                    )
                    compound_counter += 1
                    next_gen.append(product)
                    yield rec

                    if compound_counter >= target_per_core * max(1, len(core_records)):
                        return

            growth_mols = next_gen
            if not growth_mols:
                break

    log.info(
        f"  generate_grown_library: {compound_counter} compounds "
        f"generated from {len(core_records)} cores."
    )
