from __future__ import annotations

import itertools
from typing import Any, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from ..config import CONFIG


def decompose_molecule(mol: Chem.Mol, min_size: int = 8) -> List[str]:
    """Decompose a molecule into BRICS fragment SMILES.

    Uses RDKit's BRICSDecompose to break a molecule at retrosynthetically
    accessible bonds, returning the fragment SMILES strings (which may
    include dummy-atom markers such as ``[1*]``).

    Args:
        mol: The molecule to decompose.
        min_size: Minimum fragment size (heavy-atom count) for retained
            fragments.  Defaults to 8.

    Returns:
        List of fragment SMILES strings.  May be empty if the molecule
        has no BRICS break points or if decomposition fails.
    """
    try:
        fragments = BRICS.BRICSDecompose(mol, minFragmentSize=min_size)
        return [frag for frag in fragments if frag is not None]
    except Exception:
        return []


def recombine_fragments(
    frag_mols: List[Chem.Mol],
    target_count: int,
    seen_smiles: Optional[Set[str]] = None,
    seed: int = CONFIG.random_seed,
    max_products: Optional[int] = None,
) -> Tuple[List[Chem.Mol], Set[str]]:
    """Recombine BRICS fragments using BRICSBuild, then pick a diverse
    subset via MaxMin diversity picking.

    Uses RDKit's BRICSBuild to enumerate recombination products from the
    provided fragment pool.  Duplicate products (by canonical SMILES) and
    acyclic molecules are discarded.  If the resulting pool exceeds the
    requested count, MaxMin diversity picking selects the most diverse
    subset based on Morgan fingerprints.

    Args:
        frag_mols: RDKit Mol objects representing BRICS-compatible
            fragments.
        target_count: Desired number of output molecules.
        seen_smiles: Set of SMILES strings already used; duplicates are
            skipped.  If ``None``, an internal set is created.
        seed: Random seed for shuffling and MaxMin picking.
        max_products: Maximum number of products to generate from the
            BRICSBuild generator.  Defaults to ``target_count * 20``.

    Returns:
        Tuple of ``(product_mols, updated_seen_smiles)`` where
        *product_mols* is the list of diverse, sanitized RDKit Mol
        objects and *updated_seen_smiles* includes the newly generated
        SMILES.
    """
    if seen_smiles is None:
        seen_smiles = set()

    rng = np.random.default_rng(seed)

    pool_mult = CONFIG.diversity_pool_multiplier
    max_products = max_products or target_count * pool_mult * 4
    target_pool = target_count * pool_mult

    shuffled = list(frag_mols)
    rng.shuffle(shuffled)

    builder = BRICS.BRICSBuild(shuffled)

    pool_mols: List[Chem.Mol] = []
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
        pool_mols.append(product)
        n_produced += 1
        if n_produced >= target_pool:
            break

    if not pool_mols:
        return [], seen_smiles

    if len(pool_mols) <= target_count:
        return pool_mols, seen_smiles

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            m, radius=CONFIG.morgan_radius, nBits=CONFIG.morgan_nbits,
        )
        for m in pool_mols
    ]

    picker = MaxMinPicker()
    pick_ids = picker.LazyBitVectorPick(
        fps, len(fps), target_count, seed=seed,
    )

    return [pool_mols[i] for i in pick_ids], seen_smiles
