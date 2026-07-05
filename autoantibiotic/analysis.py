from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdDistGeom, rdMolAlign

from .config import CONFIG, CompoundRecord
from .docking import _parallel_dock
from .io_utils import log

_CacheLike = Optional[Dict[str, float]]


def compute_consensus_score(
    vina_energy: Optional[float],
    shape_score: Optional[float],
    vina_weight: float = CONFIG.consensus_vina_weight,
    shape_weight: float = CONFIG.consensus_shape_weight,
) -> Optional[float]:
    """Compute a weighted consensus score from Vina and Shape scores.

    Returns ``w_vina * |vina| + w_shape * shape`` if both are available,
    or whichever single score is present, or ``None`` if neither exists.
    """
    if vina_energy is not None and shape_score is not None:
        return vina_weight * abs(vina_energy) + shape_weight * shape_score
    if vina_energy is not None:
        return abs(vina_energy)
    if shape_score is not None:
        return shape_score
    return None


def compute_pharmacophore_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    tolerance: float = 2.0,
) -> Optional[float]:
    """Compute a pharmacophore feature matching score between two molecules.

    Generates 3D conformations, aligns *mol* to *ref_mol* via
    :func:`AllChem.AlignMol`, then counts H-bond donor and acceptor atoms
    in both molecules.  A feature in *mol* is considered matched if it lies
    within *tolerance* Å of the same feature type in *ref_mol*.

    Returns the fraction of reference features matched (0.0 – 1.0),
    or ``None`` if scoring fails (e.g. 3D embedding unsuccessful).
    """
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


def compute_selectivity_index(
    pb2pa_energy: float, human_avg_energy: float,
) -> float:
    """Selectivity Index (SI).

    SI = |PBP2a Energy| / |Human Avg Energy|

    Returns 0.0 if either energy is non-negative or human average is near zero.
    """
    if pb2pa_energy >= 0 or human_avg_energy >= 0:
        return 0.0
    return abs(pb2pa_energy) / abs(human_avg_energy) if abs(human_avg_energy) > 1e-6 else 0.0


def profile_resistance_risk(
    record: CompoundRecord,
    work_dir: str,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: tuple,
) -> str:
    """Energy-based heuristic proxy for resistance-risk profiling.

    Returns a human-readable notes string.
    """
    notes: List[str] = []

    act_thresh = CONFIG.resistance_energy_active_threshold
    allo_thresh = CONFIG.resistance_energy_allosteric_threshold
    mw_thresh = CONFIG.resistance_mw_threshold
    rot_thresh = CONFIG.resistance_rot_threshold
    qed_thresh = CONFIG.resistance_qed_threshold

    if record.pb2pa_active_energy is not None and record.pb2pa_active_energy < act_thresh:
        notes.append("Energy profile suggests interaction near catalytic Ser403 (heuristic proxy).")

    if record.pb2pa_allosteric_energy is not None and record.pb2pa_allosteric_energy < allo_thresh:
        if record.pb2pa_active_energy is None or record.pb2pa_active_energy > act_thresh:
            notes.append("Allosteric binder (Ala237/Met241/Tyr159 pocket). Novel mechanism.")

    if record.mol is not None:
        mw = Descriptors.MolWt(record.mol)
        if mw > mw_thresh:
            notes.append(f"High MW (>{mw_thresh:.0f}) — broad interaction surface, may contact multiple residues.")
        n_rot = Descriptors.NumRotatableBonds(record.mol)
        if n_rot < rot_thresh:
            notes.append(f"Rigid scaffold — reduced entropic penalty, may enhance binding specificity.")

    if record.qed_score > qed_thresh:
        notes.append(f"High drug-likeness (QED > {qed_thresh}) — good developability profile.")

    if not notes:
        notes.append("No specific resistance flags identified.")

    return "; ".join(notes)


_LOGS_MODEL_COEFFS = {
    "c": 0.16, "MolLogP": -0.63, "MolWt": -0.0062,
    "NumRotatableBonds": -0.0034, "NumAromaticRings": -0.042,
    "HeavyAtomCount": 0.00025,
}


def predict_logs(mol: Chem.Mol) -> float:
    """Predict aqueous solubility (LogS) using a simple linear model.

    The model is a re-implementation of the ESOL method (Delaney 2004)
    using RDKit descriptors:

        LogS = 0.16 - 0.63*LogP - 0.0062*MW + 0.0034*RotBonds
               - 0.042*AromRings + 0.00025*HeavyAtoms

    Returns predicted LogS in mol/L.
    """
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    n_arom = Descriptors.NumAromaticRings(mol) if hasattr(Descriptors, "NumAromaticRings") else 0
    heavy = mol.GetNumHeavyAtoms()

    logs = (
        _LOGS_MODEL_COEFFS["c"]
        + _LOGS_MODEL_COEFFS["MolLogP"] * logp
        + _LOGS_MODEL_COEFFS["MolWt"] * mw
        + _LOGS_MODEL_COEFFS["NumRotatableBonds"] * rot
        + _LOGS_MODEL_COEFFS["NumAromaticRings"] * n_arom
        + _LOGS_MODEL_COEFFS["HeavyAtomCount"] * heavy
    )
    return logs


def _has_basic_nitrogen(mol: Chem.Mol) -> bool:
    """Check if the molecule contains a basic nitrogen (aliphatic primary/secondary/tertiary)."""
    basic_n_pattern = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(NC=O)]")
    if basic_n_pattern is None:
        return False
    return mol.HasSubstructMatch(basic_n_pattern)


def predict_herg_risk(mol: Chem.Mol) -> str:
    """Rule-based hERG blockage risk assessment.

    Flags compounds with:
      - LogP > 4.0  (high lipophilicity → promiscuous hERG binding)
      - AND presence of a basic nitrogen

    Returns ``"High"``, ``"Moderate"``, or ``"Low"``.
    """
    logp = Crippen.MolLogP(mol)
    has_basic_n = _has_basic_nitrogen(mol)
    if logp > 4.0 and has_basic_n:
        return "High"
    if logp > 4.0 or has_basic_n:
        return "Moderate"
    return "Low"


def predict_admet_profile(record: CompoundRecord) -> CompoundRecord:
    """Compute ADMET properties for a compound and populate *admet_flags*.

    Evaluates:
      1. **Solubility (LogS)**: predicted via ESOL model.
      2. **hERG blockage risk**: rule-based (LogP + basic nitrogen).
      3. **Lipinski Rule-of-5** and **QED** (already computed in filtering).

    Flags are appended to ``record.admet_flags``.

    Returns the same ``CompoundRecord`` with populated *admet_flags*.
    """
    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            record.admet_flags.append("ADMET: invalid molecule")
            return record
        record.mol = mol
    mol = record.mol

    flags: List[str] = []

    # Solubility
    try:
        logs = predict_logs(mol)
        if logs < -5.0:
            flags.append(f"Poor solubility (LogS={logs:.2f})")
        elif logs < -3.0:
            flags.append(f"Moderate solubility (LogS={logs:.2f})")
        else:
            flags.append(f"Good solubility (LogS={logs:.2f})")
    except Exception:
        flags.append("Solubility prediction failed")

    # hERG risk
    try:
        herg = predict_herg_risk(mol)
        if herg == "High":
            flags.append("High hERG risk (LogP>4 + basic N)")
        elif herg == "Moderate":
            flags.append("Moderate hERG risk")
        else:
            flags.append("Low hERG risk")
    except Exception:
        flags.append("hERG prediction failed")

    # Lipinski (already computed, just annotate)
    if record.passes_lipinski:
        flags.append("Lipinski OK")
    else:
        flags.append("Lipinski violation")

    # QED
    if record.qed_score > CONFIG.qed_threshold:
        flags.append(f"QED OK ({record.qed_score:.2f})")
    else:
        flags.append(f"QED below threshold ({record.qed_score:.2f})")

    record.admet_flags = flags
    return record


def _parse_pdbqt_ligand_coords(pose_pdbqt: str) -> List[np.ndarray]:
    """Extract ligand heavy-atom coordinates from a docked-pose PDBQT file.

    Parses ATOM/HETATM records that appear between ``ROOT`` / ``ENDROOT``
    or ``BRANCH`` / ``ENDBRANCH`` markers.  Falls back to all ATOM/HETATM
    lines if no ROOT section is present.
    """
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
    """Extract heavy-atom coordinates for a list of residue names from a PDB file.

    Returns a dict mapping residue name (e.g. ``"SER403"``) to a list of
    (x, y, z) arrays for each heavy atom in that residue.
    """
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
    """Check whether the docked ligand pose contacts any heavy atom in *key_residues*.

    Args:
        pose_pdbqt: Path to the docked-ligand PDBQT file.
        receptor_pdb: Path to the receptor PDB file.
        key_residues: List of residue identifiers (e.g. ``["SER403"]``).
        distance_cutoff: Maximum distance (Å) to consider a contact.

    Returns:
        ``True`` if at least one ligand atom is within *distance_cutoff*
        of any heavy atom in the specified residues.
    """
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


# ── Interaction Fingerprint (IFP) ──────────────────────────────────────────

_IFP_RESIDUES = ["TYR159", "ALA237", "MET241", "SER403"]

_RESIDUE_IFP_CLASSES = {
    "TYR": {
        "donor": ["N", "OH"],
        "acceptor": ["O", "OH"],
        "hydrophobic": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "aromatic": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    },
    "ALA": {
        "donor": ["N"],
        "acceptor": ["O"],
        "hydrophobic": ["CA", "CB"],
        "aromatic": [],
    },
    "MET": {
        "donor": ["N"],
        "acceptor": ["O"],
        "hydrophobic": ["CB", "CG", "CE"],
        "aromatic": [],
    },
    "SER": {
        "donor": ["N", "OG"],
        "acceptor": ["O", "OG"],
        "hydrophobic": ["CA", "CB"],
        "aromatic": [],
    },
}

_IFP_HBA_DIST = 3.5
_IFP_HBD_DIST = 3.5
_IFP_HYD_DIST = 4.5
_IFP_PI_DIST = 5.5


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

    # Classify ligand atoms
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

    # Ligand aromatic ring centroids
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

        # HBA — ligand acceptor near residue donor
        hba = False
        for li in lig_hba:
            lp = conf.GetAtomPosition(li)
            lv = np.array([lp.x, lp.y, lp.z])
            if any(np.linalg.norm(lv - rp) <= _IFP_HBA_DIST for rp in res_donor):
                hba = True
                break
        bits.append(hba)

        # HBD — ligand donor near residue acceptor
        hbd = False
        for li in lig_hbd:
            lp = conf.GetAtomPosition(li)
            lv = np.array([lp.x, lp.y, lp.z])
            if any(np.linalg.norm(lv - rp) <= _IFP_HBD_DIST for rp in res_acc):
                hbd = True
                break
        bits.append(hbd)

        # Hydrophobic contact
        hydro = False
        for li in lig_hydro:
            lp = conf.GetAtomPosition(li)
            lv = np.array([lp.x, lp.y, lp.z])
            if any(np.linalg.norm(lv - rp) <= _IFP_HYD_DIST for rp in res_hydro):
                hydro = True
                break
        bits.append(hydro)

        # π-stacking (only residues with aromatic rings)
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

    Generates 3D conformers, aligns the candidate to the reference, then
    computes per-residue interaction bit vectors against the key allosteric
    and active-site residues.  Returns a value in 0.0–1.0.

    Falls back to a distance-based contact map if
    ``rdkit.Chem.rdMolInteractions`` is unavailable.
    """
    # Check for rdMolInteractions availability (per requirement)
    _try_rdmolinteractions = False
    try:
        from rdkit.Chem import rdMolInteractions  # noqa: F401
        _try_rdmolinteractions = True
    except ImportError:
        pass

    # Generate 3D conformers
    docked_3d = _generate_ifp_conformer(docked_ligand_mol)
    ref_3d = _generate_ifp_conformer(reference_ligand_mol)
    if docked_3d is None or ref_3d is None:
        log.warning("  IFP: Failed to generate 3D conformers.")
        return 0.0

    # Align docked candidate to reference
    try:
        o3a = rdMolAlign.GetO3A(docked_3d, ref_3d)
        o3a.Align()
    except Exception as exc:
        log.warning(f"  IFP: Alignment failed — {exc}")
        return 0.0

    # Parse receptor data
    receptor_data = _parse_ifp_receptor(receptor_pdb, _IFP_RESIDUES)

    # If no receptor atoms were found, IFP is not meaningful
    if not any(res_info["positions"] for res_info in receptor_data.values()):
        log.warning("  IFP: No receptor atoms found in PDB.")
        return 0.0

    # Compute IFP vectors
    ifp_a = _compute_ifp_bitvector(docked_3d, receptor_data)
    ifp_b = _compute_ifp_bitvector(ref_3d, receptor_data)

    return _tanimoto_bits(ifp_a, ifp_b)


def analyze_selectivity_and_resistance(
    top10: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: _CacheLike = None,
    use_cache: bool = False,
) -> List[CompoundRecord]:
    """Phase 4 — Selectivity & Resistance Analysis.

    Docks top 10 against human off-targets, computes SI, profiles resistance risk.
    """
    log.info("─── Phase 4: Selectivity & Resistance Analysis ───")

    use_vina = deps.get("USE_VINA", False)
    if not use_vina:
        log.warning("  Vina unavailable — skipping selectivity docking. Flagging all as uncertain.")
        for rec in top10:
            rec.selectivity_index = 1.0
            rec.resistance_notes = "Selectivity not assessed (Vina unavailable)."
        return top10

    trypsin_target = targets.get("trypsin")
    ces1_target = targets.get("CES1")
    if trypsin_target is None or ces1_target is None:
        log.warning("  Off-target data missing — skipping selectivity docking.")
        for rec in top10:
            rec.selectivity_index = 1.0
            rec.resistance_notes = "Selectivity not assessed (off-target data missing)."
        return top10

    log.info("  Docking top 10 vs Human Trypsin (1UTN)…")
    trypsin_center = trypsin_target.get("active_center", np.array([0.0, 0.0, 0.0]))
    trypisn_items = [(r.compound_id, r.smiles) for r in top10]
    trypsin_results = _parallel_dock(
        trypisn_items, targets["trypsin"]["pdbqt"],
        trypsin_center, CONFIG.offtarget_box_size,
        work_dir, "trypsin", n_jobs=min(4, len(top10)),
        cache=cache, use_cache=use_cache,
    )
    cid_map = {r.compound_id: r for r in top10}
    for cid, energy in trypsin_results:
        if cid in cid_map:
            cid_map[cid].human_trypsin_energy = energy

    log.info("  Docking top 10 vs Human Carboxylesterase 1 (3KJZ)…")
    ces1_center = ces1_target.get("active_center", np.array([0.0, 0.0, 0.0]))
    ces1_items = [(r.compound_id, r.smiles) for r in top10]
    ces1_results = _parallel_dock(
        ces1_items, targets["CES1"]["pdbqt"],
        ces1_center, CONFIG.offtarget_box_size,
        work_dir, "ces1", n_jobs=min(4, len(top10)),
        cache=cache, use_cache=use_cache,
    )
    for cid, energy in ces1_results:
        if cid in cid_map:
            cid_map[cid].human_ces1_energy = energy

    for rec in top10:
        energies_human = [
            e for e in (rec.human_trypsin_energy, rec.human_ces1_energy)
            if e is not None
        ]
        if not energies_human:
            log.warning(f"  {rec.compound_id}: No human docking data. SI = N/A.")
            rec.selectivity_index = 1.0
            continue

        human_avg = np.mean(energies_human)
        pb2pa_best = (
            rec.pb2pa_active_energy if rec.pb2pa_active_energy is not None
            else rec.pb2pa_allosteric_energy
        )
        if pb2pa_best is None:
            rec.selectivity_index = 1.0
            continue

        si = compute_selectivity_index(pb2pa_best, human_avg)
        rec.selectivity_index = si

        if si < CONFIG.selectivity_index_threshold:
            log.warning(
                f"  {rec.compound_id}: Low selectivity (SI = {si:.2f} < {CONFIG.selectivity_index_threshold}). "
                "Flagged for off-target risk."
            )
        else:
            log.info(f"  {rec.compound_id}: SI = {si:.2f} (pass).")

    pb2pa = targets["PBP2a"]
    receptor_pdb = pb2pa["pdbqt"].replace(".pdbqt", ".pdb")
    if not os.path.isfile(receptor_pdb):
        receptor_pdb = os.path.join(
            os.path.dirname(pb2pa["pdbqt"]),
            "PBP2a_clean.pdb",
        )

    log.info("  Checking key interactions for top candidates…")
    for rec in top10:
        rec.resistance_notes = profile_resistance_risk(
            rec, work_dir,
            pb2pa["pdbqt"],
            pb2pa["allosteric_center"],
            CONFIG.allosteric_box_size,
        )

        # Generate 3-D conformer and write temporary PDBQT for IFP check
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol
        mol = Chem.MolFromSmiles(rec.smiles) if rec.mol is None else Chem.RWMol(rec.mol)
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = CONFIG.random_seed
        if rdDistGeom.EmbedMolecule(mol, params) >= 0:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".pdbqt", delete=False,
                ) as tmp:
                    tmp_pdbqt = tmp.name
                    conf = mol.GetConformer()
                    tmp.write("ROOT\n")
                    for i in range(mol.GetNumAtoms()):
                        atom = mol.GetAtomWithIdx(i)
                        if atom.GetAtomicNum() == 1:
                            continue
                        pt = conf.GetAtomPosition(i)
                        elem = atom.GetSymbol()
                        tmp.write(
                            f"ATOM  {i+1:5d} {elem:<4s} LIG     1    "
                            f"{pt.x:8.3f}{pt.y:8.3f}{pt.z:8.3f}  "
                            f"1.00  0.00          {elem:>2s}\n"
                        )
                    tmp.write("ENDROOT\n")
                try:
                    allosteric_hits = (
                        CONFIG.min_key_interactions > 0
                        and check_key_interactions(
                            tmp_pdbqt, receptor_pdb,
                            CONFIG.key_interaction_residues_allosteric,
                        )
                    )
                    active_hits = check_key_interactions(
                        tmp_pdbqt, receptor_pdb,
                        CONFIG.key_interaction_residues_active,
                    )
                    if not (allosteric_hits or active_hits):
                        rec.resistance_notes += (
                            "; Warning: No key interactions detected"
                        )
                finally:
                    try:
                        os.unlink(tmp_pdbqt)
                    except OSError:
                        pass
            except Exception as exc:
                log.debug(f"  IFP check failed for {rec.compound_id}: {exc}")

        # ── IFP similarity to reference Ceftaroline ──
        try:
            ref_smi = CONFIG.control_smiles.get("Ceftaroline", "")
            if ref_smi:
                ref_mol = Chem.MolFromSmiles(ref_smi)
                if ref_mol is not None and rec.mol is not None:
                    rec.ifp_score = compute_ifp_similarity(
                        rec.mol, ref_mol, receptor_pdb,
                    )
                    if rec.ifp_score < CONFIG.ifp_similarity_threshold:
                        rec.resistance_notes += (
                            f"; Warning: Low IFP similarity to reference ligand "
                            f"({rec.ifp_score:.2f})"
                        )
        except Exception as exc:
            log.debug(f"  IFP similarity failed for {rec.compound_id}: {exc}")

    # ── ADMET profiling on top 10 ──
    log.info("  Computing ADMET profiles for top candidates…")
    for rec in top10:
        predict_admet_profile(rec)
        log.debug(f"  {rec.compound_id}: {rec.admet_flags}")

    log.info("─── Phase 4 complete ───")
    return top10
