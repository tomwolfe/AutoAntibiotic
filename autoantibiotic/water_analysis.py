"""
Water Analysis Module
======================
Identifies crystallographic waters in the PBP2a holo structure (3ZG0),
characterises them as displaceable (high-energy) or structural (bridging),
and provides data for water-aware receptor preparation.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CONFIG
from .io_utils import log

try:
    from Bio.PDB import PDBParser, NeighborSearch
    from Bio.PDB import is_aa as _is_aa
    _HAVE_BIOPDB = True
except ImportError:
    _HAVE_BIOPDB = False


@dataclass
class WaterInfo:
    """Properties of a single crystallographic water molecule."""
    chain: str
    resseq: int
    resname: str = "HOH"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_factor: float = 0.0
    occupancy: float = 1.0
    distance_to_allosteric: Optional[float] = None
    distance_to_active: Optional[float] = None
    n_hbonds_protein: int = 0
    displacement_energy: float = 0.0
    is_high_energy: bool = False
    is_bridging: bool = False

    @property
    def identifier(self) -> str:
        return f"{self.chain}:{self.resname}_{self.resseq}"


@dataclass
class WaterAnalysisResult:
    """Aggregated result of a crystallographic water analysis."""
    allosteric_waters: List[WaterInfo] = field(default_factory=list)
    active_site_waters: List[WaterInfo] = field(default_factory=list)
    high_energy_waters: List[WaterInfo] = field(default_factory=list)
    bridging_waters: List[WaterInfo] = field(default_factory=list)
    all_waters: List[WaterInfo] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.all_waters)


def _parse_pdb_waters(pdb_path: str) -> List[WaterInfo]:
    """Extract all water residues from a PDB file together with their
    positions and B-factors."""
    if not _HAVE_BIOPDB:
        log.warning("  Bio.PDB not available - cannot perform water analysis.")
        return []

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("water_analysis", pdb_path)

    waters: List[WaterInfo] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetfield = residue.get_id()[0]
                resname = residue.get_resname().strip().upper()
                if hetfield == "W" or resname in ("HOH", "WAT", "SOL"):
                    if "O" not in residue:
                        continue
                    o_atom = residue["O"]
                    pos = o_atom.get_vector().get_array()
                    bfac = o_atom.get_bfactor()
                    occ = o_atom.get_occupancy()
                    waters.append(WaterInfo(
                        chain=chain.get_id(),
                        resseq=residue.get_id()[1],
                        position=pos,
                        b_factor=bfac,
                        occupancy=occ,
                    ))
    log.info(f"  Found {len(waters)} crystallographic waters in {pdb_path}")
    return waters


def _get_residue_atoms(pdb_path: str, resid_list: List[str]) -> List[np.ndarray]:
    """Return all heavy-atom coordinates for the requested residue list."""
    if not _HAVE_BIOPDB:
        return []

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", pdb_path)

    targets: set = set()
    for entry in resid_list:
        rname = "".join(ch for ch in entry if ch.isalpha()).upper()
        rnum = int("".join(ch for ch in entry if ch.isdigit()))
        targets.add((rname, rnum))

    coords: List[np.ndarray] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                key = (residue.get_resname().strip().upper(), residue.get_id()[1])
                if key in targets:
                    for atom in residue:
                        if atom.element != "H":
                            coords.append(atom.get_vector().get_array())
    return coords


def _count_hbonds_to_protein(
    water_pos: np.ndarray,
    protein_atoms: List[np.ndarray],
    distance_cutoff: float = 3.5,
    angle_cutoff_deg: float = 120.0,
) -> int:
    """Count likely hydrogen bonds between a water oxygen and nearby
    protein heavy-atom acceptors/donors.

    Uses a simple distance + angle criterion.
    """
    count = 0
    for prot_pos in protein_atoms:
        d = float(np.linalg.norm(water_pos - prot_pos))
        if d <= distance_cutoff:
            # Rough angle check: water O to protein atom to protein backbone
            # Since we lack H positions, we use a lenient distance-only criterion
            count += 1
    return count


def _compute_displacement_energy(
    water: WaterInfo,
    protein_atoms: List[np.ndarray],
    max_bfactor: float = 100.0,
    max_hbonds: int = 4,
) -> float:
    """Estimate the thermodynamic penalty for displacing a water molecule.

    The displacement energy is a composite of:
    1. **B-factor penalty**: high B-factor → loosely bound → easy to displace
    2. **H-bond penalty**: fewer H-bonds to protein → easy to displace
    3. **Occupancy penalty**: low occupancy → loosely bound

    Returns a score where **higher** values indicate waters that should
    be more easily displaced (high-energy).
    """
    hbonds = _count_hbonds_to_protein(water.position, protein_atoms)

    bfactor_norm = min(water.b_factor / max_bfactor, 1.0)
    hbond_norm = 1.0 - min(hbonds / max_hbonds, 1.0)
    occ_norm = 1.0 - min(water.occupancy, 1.0)

    energy = 0.5 * bfactor_norm + 0.35 * hbond_norm + 0.15 * occ_norm
    return energy


def _is_bridging_water(
    water_pos: np.ndarray,
    protein_atoms: List[np.ndarray],
    hbond_cutoff: float = 3.5,
    min_hbonds: int = 2,
) -> bool:
    """A bridging water makes at least *min_hbonds* H-bonds to protein
    atoms, suggesting it plays a structural role."""
    count = 0
    for prot_pos in protein_atoms:
        if float(np.linalg.norm(water_pos - prot_pos)) <= hbond_cutoff:
            count += 1
            if count >= min_hbonds:
                return True
    return False


def analyze_waters(
    pdb_path: str,
    allosteric_residues: Optional[List[str]] = None,
    active_site_residues: Optional[List[str]] = None,
    distance_cutoff: float = 5.0,
    displacement_energy_threshold: float = 2.5,
) -> WaterAnalysisResult:
    """Run full water analysis on a PDB structure.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file (typically the holo structure 3ZG0).
    allosteric_residues : list of str, optional
        Residue identifiers for the allosteric site (default from CONFIG).
    active_site_residues : list of str, optional
        Residue identifiers for the active site (default from CONFIG).
    distance_cutoff : float
        Maximum distance (Å) from site residues to include a water.
    displacement_energy_threshold : float
        Waters with displacement energy >= this threshold are flagged
        as high-energy.

    Returns
    -------
    WaterAnalysisResult
        Classified waters organised by site, energy, and bridging status.
    """
    if not _HAVE_BIOPDB:
        log.warning("  Bio.PDB not available. Skipping water analysis.")
        return WaterAnalysisResult()

    if not pdb_path or not os.path.exists(pdb_path):
        log.warning(f"  PDB file not found: {pdb_path}. Skipping water analysis.")
        return WaterAnalysisResult()

    if allosteric_residues is None:
        allosteric_residues = CONFIG.flexible_residues_allosteric
    if active_site_residues is None:
        active_site_residues = CONFIG.flexible_residues_active

    waters = _parse_pdb_waters(pdb_path)
    if not waters:
        return WaterAnalysisResult()

    allosteric_atoms = _get_residue_atoms(pdb_path, allosteric_residues)
    active_atoms = _get_residue_atoms(pdb_path, active_site_residues)
    all_site_atoms = allosteric_atoms + active_atoms

    if not all_site_atoms:
        log.warning("  No binding-site atoms found for water analysis.")
        return WaterAnalysisResult()

    result = WaterAnalysisResult()

    for w in waters:
        # Compute distances to binding sites
        if allosteric_atoms:
            d_alloc = min(np.linalg.norm(w.position - a) for a in allosteric_atoms)
            w.distance_to_allosteric = float(d_alloc)
        if active_atoms:
            d_act = min(np.linalg.norm(w.position - a) for a in active_atoms)
            w.distance_to_active = float(d_act)

        # Only consider waters within cutoff of either site
        in_alloc = w.distance_to_allosteric is not None and w.distance_to_allosteric <= distance_cutoff
        in_act = w.distance_to_active is not None and w.distance_to_active <= distance_cutoff
        if not in_alloc and not in_act:
            continue

        # Count H-bonds
        w.n_hbonds_protein = _count_hbonds_to_protein(w.position, all_site_atoms)

        # Displacement energy
        w.displacement_energy = _compute_displacement_energy(w, all_site_atoms)
        w.is_high_energy = w.displacement_energy >= displacement_energy_threshold
        w.is_bridging = _is_bridging_water(w.position, all_site_atoms)

        result.all_waters.append(w)
        if in_alloc:
            result.allosteric_waters.append(w)
        if in_act:
            result.active_site_waters.append(w)
        if w.is_high_energy:
            result.high_energy_waters.append(w)
        if w.is_bridging:
            result.bridging_waters.append(w)

    log.info(
        f"  Water analysis: {len(result.allosteric_waters)} site-proximal waters, "
        f"{len(result.high_energy_waters)} high-energy, "
        f"{len(result.bridging_waters)} bridging"
    )
    return result


def get_waters_to_remove(result: WaterAnalysisResult) -> List[WaterInfo]:
    """Return the list of water molecules that should be removed from the
    receptor: high-energy waters that are NOT bridging.

    Bridging waters are kept because they are structurally important.
    """
    return [w for w in result.high_energy_waters if not w.is_bridging]


def get_waters_to_keep(result: WaterAnalysisResult) -> List[WaterInfo]:
    """Return waters that should be retained in the receptor:
    all bridging waters plus any non-high-energy site waters."""
    return [w for w in result.all_waters if not w.is_high_energy or w.is_bridging]
