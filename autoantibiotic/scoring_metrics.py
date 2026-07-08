from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolAlign

from .config import CONFIG
from .io_utils import log


_IFP_RESIDUES = ["ASN159", "GLU237", "ARG241", "SER403"]

_RESIDUE_IFP_CLASSES = {
    "ASN": {
        "donor": ["N", "ND2"],
        "acceptor": ["O", "OD1"],
        "hydrophobic": ["CA", "CB"],
        "aromatic": [],
    },
    "GLU": {
        "donor": ["N"],
        "acceptor": ["O", "OE1", "OE2"],
        "hydrophobic": ["CA", "CB", "CG"],
        "aromatic": [],
    },
    "ARG": {
        "donor": ["N", "NE", "NH1", "NH2"],
        "acceptor": ["O"],
        "hydrophobic": ["CA", "CB", "CG", "CD"],
        "aromatic": [],
    },
    "SER": {
        "donor": ["N", "OG"],
        "acceptor": ["O", "OG"],
        "hydrophobic": ["CA", "CB"],
        "aromatic": [],
    },
}

_IFP_HBA_DIST = CONFIG.ifp_hba_dist
_IFP_HBD_DIST = CONFIG.ifp_hbd_dist
_IFP_HYD_DIST = CONFIG.ifp_hyd_dist
_IFP_PI_DIST = CONFIG.ifp_pi_dist


def _parse_pdbqt_ligand_coords(pose_pdbqt: str) -> List[np.ndarray]:
    """Extract ligand heavy-atom coordinates from a docked-pose PDBQT file."""
    coords: List[np.ndarray] = []
    try:
        with open(pose_pdbqt) as f:
            lines = f.readlines()
    except (FileNotFoundError, OSError):
        return coords

    in_ligand = False
    has_root = any(l.startswith("ROOT") for l in lines)

    for line in lines:
        if line.startswith("ROOT"):
            in_ligand = True
            continue
        if line.startswith("ENDROOT") or line.startswith("ENDBRANCH"):
            in_ligand = False
            continue
        if line.startswith("BRANCH"):
            in_ligand = True
            continue
        if has_root and not in_ligand:
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            elem = line[76:78].strip() if len(line) > 76 else ""
            if elem.upper() in ("H", ""):
                continue
            coords.append(np.array([x, y, z]))
        except (ValueError, IndexError):
            continue
    return coords


def _parse_pdb_residue_coords(
    receptor_pdb: str,
    key_residues: List[str],
) -> Dict[str, List[np.ndarray]]:
    """Extract heavy-atom coordinates for a list of residue names from a PDB file."""
    residue_coords: Dict[str, List[np.ndarray]] = {r: [] for r in key_residues}
    with open(receptor_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip()
            resid = line[22:26].strip()
            key = f"{resname}{resid}"
            if key not in residue_coords:
                continue
            elem = line[76:78].strip() if len(line) > 76 else ""
            if elem.upper() in ("H", ""):
                continue
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                residue_coords[key].append(np.array([x, y, z]))
            except (ValueError, IndexError):
                continue
    return residue_coords


def check_key_interactions(
    pose_pdbqt: str,
    receptor_pdb: str,
    key_residues: List[str],
    distance_cutoff: float = 3.5,
) -> bool:
    """Check whether the docked ligand pose contacts any heavy atom in *key_residues*."""
    if not os.path.isfile(pose_pdbqt):
        log.warning(f"  Pose PDBQT not found: {pose_pdbqt}")
        return False
    if not os.path.isfile(receptor_pdb):
        log.warning(f"  Receptor PDB not found: {receptor_pdb}")
        return False

    lig_coords = _parse_pdbqt_ligand_coords(pose_pdbqt)
    if not lig_coords:
        log.warning("  No ligand heavy atoms found in pose PDBQT.")
        return False

    residue_coords = _parse_pdb_residue_coords(receptor_pdb, key_residues)
    found = False
    for key in key_residues:
        res_atoms = residue_coords.get(key, [])
        if not res_atoms:
            continue
        for lc in lig_coords:
            for rc in res_atoms:
                if np.linalg.norm(lc - rc) <= distance_cutoff:
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        log.debug(f"  Key interaction detected with {key_residues}.")
    return found


def compute_pharmacophore_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    tolerance: float = 2.0,
) -> Optional[float]:
    """Compute a pharmacophore feature matching score between two molecules."""
    try:
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = CONFIG.random_seed
        if rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)

        ref_3d = Chem.RWMol(ref_mol)
        ref_3d = Chem.AddHs(ref_3d)
        params_ref = rdDistGeom.ETKDGv3()
        params_ref.randomSeed = CONFIG.random_seed
        if rdDistGeom.EmbedMolecule(ref_3d, params_ref) < 0:
            return None
        AllChem.MMFFOptimizeMolecule(ref_3d)

        o3a = rdMolAlign.GetO3A(mol_3d, ref_3d)
        o3a.Align()

        def _is_hbd(atom: Chem.Atom) -> bool:
            if atom.GetAtomicNum() not in (7, 8):
                return False
            return (
                atom.GetTotalNumHs() > 0
                or any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())
            )

        def _is_hba(atom: Chem.Atom) -> bool:
            return atom.GetAtomicNum() in (7, 8)

        conf_mol = mol_3d.GetConformer()
        query_donors = [conf_mol.GetAtomPosition(i) for i, a in enumerate(mol_3d.GetAtoms()) if _is_hbd(a)]
        query_acceptors = [conf_mol.GetAtomPosition(i) for i, a in enumerate(mol_3d.GetAtoms()) if _is_hba(a)]

        conf_ref = ref_3d.GetConformer()
        ref_donors = [conf_ref.GetAtomPosition(i) for i, a in enumerate(ref_3d.GetAtoms()) if _is_hbd(a)]
        ref_acceptors = [conf_ref.GetAtomPosition(i) for i, a in enumerate(ref_3d.GetAtoms()) if _is_hba(a)]

        def _count_matches(
            ref_positions: List[Chem.Point3D],
            query_positions: List[Chem.Point3D],
        ) -> int:
            matched = 0
            for rp in ref_positions:
                for qp in query_positions:
                    if rp.Distance(qp) <= tolerance:
                        matched += 1
                        break
            return matched

        donor_matches = _count_matches(ref_donors, query_donors)
        acceptor_matches = _count_matches(ref_acceptors, query_acceptors)

        total_ref = len(ref_donors) + len(ref_acceptors)
        if total_ref == 0:
            return 1.0

        return (donor_matches + acceptor_matches) / total_ref

    except Exception:
        return None


def _generate_ifp_conformer(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Generate a 3D conformer for IFP computation."""
    mol_3d = Chem.RWMol(mol)
    mol_3d = Chem.AddHs(mol_3d)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = CONFIG.random_seed
    if rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
    except Exception:
        pass
    return mol_3d


def _parse_ifp_receptor(
    receptor_pdb: str,
    residues: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Parse receptor PDB for IFP key-residue atom names and positions."""
    result: Dict[str, Dict[str, Any]] = {r: {"positions": [], "atom_names": []} for r in residues}
    if not os.path.isfile(receptor_pdb):
        return result
    with open(receptor_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip()
            resid = line[22:26].strip()
            key = f"{resname}{resid}"
            if key not in result:
                continue
            elem = (line[76:78].strip() or line[12:16].strip()).upper()
            if elem in ("H", ""):
                continue
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                result[key]["positions"].append(np.array([x, y, z]))
                result[key]["atom_names"].append(line[12:16].strip())
            except (ValueError, IndexError):
                continue
    return result


def _get_ifp_ring_centroids(res_info: Dict, res_name: str) -> List[np.ndarray]:
    """Get aromatic ring centroids for a residue (if any)."""
    arom_names = _RESIDUE_IFP_CLASSES.get(res_name, {}).get("aromatic", [])
    arom_pts = [
        p for p, n in zip(res_info["positions"], res_info["atom_names"])
        if n in arom_names
    ]
    if not arom_pts:
        return []
    return [np.mean(arom_pts, axis=0)]


def _compute_ifp_bitvector(
    mol_3d: Chem.Mol,
    receptor_data: Dict[str, Dict[str, Any]],
) -> List[bool]:
    """Compute an IFP bit vector for a molecule relative to parsed receptor data."""
    conf = mol_3d.GetConformer()

    lig_hba = [
        i for i, a in enumerate(mol_3d.GetAtoms())
        if a.GetAtomicNum() in (7, 8)
    ]
    lig_hbd = [
        i for i, a in enumerate(mol_3d.GetAtoms())
        if a.GetAtomicNum() in (7, 8) and a.GetTotalNumHs() > 0
    ]
    lig_hydro = [
        i for i, a in enumerate(mol_3d.GetAtoms())
        if a.GetAtomicNum() in (6, 16)
            and not a.GetIsAromatic()
    ]

    ri = mol_3d.GetRingInfo()
    lig_arom_centroids: List[np.ndarray] = []
    for ring in ri.AtomRings():
        if all(mol_3d.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            pts = [conf.GetAtomPosition(i) for i in ring]
            c = np.mean([(p.x, p.y, p.z) for p in pts], axis=0)
            lig_arom_centroids.append(c)

    bits: List[bool] = []
    for res_key, res_info in receptor_data.items():
        res_name = res_key[:3]
        classes = _RESIDUE_IFP_CLASSES.get(res_name, {})

        res_donor = [
            p for p, n in zip(res_info["positions"], res_info["atom_names"])
            if n in classes.get("donor", [])
        ]
        res_acc = [
            p for p, n in zip(res_info["positions"], res_info["atom_names"])
            if n in classes.get("acceptor", [])
        ]
        res_hydro = [
            p for p, n in zip(res_info["positions"], res_info["atom_names"])
            if n in classes.get("hydrophobic", [])
        ]
        res_centroids = _get_ifp_ring_centroids(res_info, res_name)

        hba = False
        for li in lig_hba:
            lp = conf.GetAtomPosition(li)
            lv = np.array([lp.x, lp.y, lp.z])
            if any(np.linalg.norm(lv - rp) <= _IFP_HBA_DIST for rp in res_donor):
                hba = True
                break
        bits.append(hba)

        hbd = False
        for li in lig_hbd:
            lp = conf.GetAtomPosition(li)
            lv = np.array([lp.x, lp.y, lp.z])
            if any(np.linalg.norm(lv - rp) <= _IFP_HBD_DIST for rp in res_acc):
                hbd = True
                break
        bits.append(hbd)

        hydro = False
        for li in lig_hydro:
            lp = conf.GetAtomPosition(li)
            lv = np.array([lp.x, lp.y, lp.z])
            if any(np.linalg.norm(lv - rp) <= _IFP_HYD_DIST for rp in res_hydro):
                hydro = True
                break
        bits.append(hydro)

        if classes.get("aromatic"):
            pi = False
            for lc in lig_arom_centroids:
                if any(np.linalg.norm(lc - rc) <= _IFP_PI_DIST for rc in res_centroids):
                    pi = True
                    break
            bits.append(pi)

    return bits


def _tanimoto_bits(bits_a: List[bool], bits_b: List[bool]) -> float:
    """Tanimoto coefficient between two boolean bit vectors."""
    if len(bits_a) != len(bits_b):
        return 0.0
    n_a = sum(bits_a)
    n_b = sum(bits_b)
    n_and = sum(1 for a, b in zip(bits_a, bits_b) if a and b)
    n_or = n_a + n_b - n_and
    return 1.0 if n_or == 0 else n_and / n_or


def compute_ifp_similarity(
    docked_ligand_mol: Chem.Mol,
    reference_ligand_mol: Chem.Mol,
    receptor_pdb: str,
) -> float:
    """
    Compute the Tanimoto similarity between interaction fingerprints (IFP)
    of a docked candidate and a reference ligand.
    """
    _try_rdmolinteractions = False
    try:
        from rdkit.Chem import rdMolInteractions  # noqa: F401
        _try_rdmolinteractions = True
    except ImportError:
        pass

    docked_3d = _generate_ifp_conformer(docked_ligand_mol)
    ref_3d = _generate_ifp_conformer(reference_ligand_mol)
    if docked_3d is None or ref_3d is None:
        log.warning("  IFP: Failed to generate 3D conformers.")
        return 0.0

    try:
        o3a = rdMolAlign.GetO3A(docked_3d, ref_3d)
        o3a.Align()
    except Exception as exc:
        log.warning(f"  IFP: Alignment failed — {exc}")
        return 0.0

    receptor_data = _parse_ifp_receptor(receptor_pdb, _IFP_RESIDUES)

    if not any(res_info["positions"] for res_info in receptor_data.values()):
        log.warning("  IFP: No receptor atoms found in PDB.")
        return 0.0

    ifp_a = _compute_ifp_bitvector(docked_3d, receptor_data)
    ifp_b = _compute_ifp_bitvector(ref_3d, receptor_data)

    return _tanimoto_bits(ifp_a, ifp_b)
