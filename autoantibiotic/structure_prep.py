from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import CONFIG
from .io_utils import download_with_retry, log, run_tool
from .water_analysis import get_waters_to_remove


def calculate_adaptive_box_size(
    coords: np.ndarray,
    padding: float = 5.0,
    minimum: float = 10.0,
) -> Tuple[float, float, float]:
    """Compute adaptive grid box dimensions from atomic coordinates.

    For each axis the box size is ``(max - min) + 2 * padding``, clamped
    to at least *minimum* Å.

    Args:
        coords: (N, 3) array of atomic coordinates.
        padding: Extra space (Å) added on each side.
        minimum: Minimum allowed box dimension (Å).

    Returns:
        Tuple of (size_x, size_y, size_z).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {coords.shape}")

    if coords.shape[0] < 2:
        log.warning("  Fewer than 2 coordinates provided to adaptive box sizing. Using default minimum.")
        return (minimum, minimum, minimum)

    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    sizes = np.maximum(hi - lo + 2.0 * padding, minimum)
    return (float(sizes[0]), float(sizes[1]), float(sizes[2]))


def _pdb_to_pdbqt_via_rdkit(pdb_path: str, pdbqt_path: str) -> bool:
    """Minimal PDB → PDBQT conversion using RDKit fallback."""
    try:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is None:
            return False
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.ComputeGasteigerCharges(mol)

        _atom_type_map = {
            "C": "C", "c": "C",
            "N": "N", "n": "N",
            "O": "O", "o": "O",
            "S": "S", "s": "S",
            "P": "P", "p": "P",
            "F": "F", "f": "F",
            "Cl": "Cl", "Br": "Br",
            "H": "H",
        }

        conf = mol.GetConformer()
        lines: list = []
        lines.append("ROOT")
        for i, atom in enumerate(mol.GetAtoms()):
            atom_no = i + 1
            elem = atom.GetSymbol()
            pdbx = conf.GetAtomPosition(i)
            gasteiger = atom.GetDoubleProp("_GasteigerCharge")
            ad_type = _atom_type_map.get(elem, "C")

            x, y, z = pdbx.x, pdbx.y, pdbx.z
            atom_name = f"{elem}{atom_no:>3}"[:4]
            line = (
                f"ATOM     {atom_no:>3} {atom_name:>4} PRT X   1    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  "
                f"{gasteiger:>8.3f}     {ad_type:<2s}\n"
            )
            lines.append(line)
        lines.append("ENDROOT")
        lines.append("TORSDOF 0\n")

        with open(pdbqt_path, "w") as f:
            f.writelines(lines)
        return True

    except Exception as exc:
        log.warning(f"  RDKit PDBQT fallback failed: {exc}")
        return False


def fetch_structure(pdb_id: str, out_dir: str) -> str:
    """Download a PDB structure by *pdb_id* (if not already present) into *out_dir*."""
    return download_with_retry(
        pdb_id, out_dir,
        max_attempts=CONFIG.pdb_retry_max_attempts,
        base_delay=CONFIG.pdb_retry_base_delay,
    )


def clean_pdb_structure(
    pdb_path: str, out_path: str,
    remove_waters: bool = True,
    remove_ligands: bool = True,
    add_hydrogens: bool = True,
    deps: Optional[Dict[str, Any]] = None,
    waters_to_remove: Optional[List[str]] = None,
) -> str:
    """Clean a PDB file and convert to PDBQT format.

    Parameters
    ----------
    pdb_path : str
        Path to input PDB file.
    out_path : str
        Path for the cleaned PDB output.
    remove_waters : bool
        Remove ALL water molecules if True (default). Ignored when
        *waters_to_remove* is provided.
    remove_ligands : bool
        Remove hetero-atoms (ligands, ions) if True.
    add_hydrogens : bool
        Add polar hydrogens via RDKit if True.
    deps : dict, optional
        Dependency dictionary (e.g. 'prepare_receptor', 'obabel').
    waters_to_remove : list of str, optional
        Specific water residue identifiers to remove (e.g. ``["A:HOH_123"]``).
        When provided, only these waters are removed, allowing bridging
        waters to be kept.
    """
    if deps is None:
        deps = {}
    try:
        from Bio.PDB import PDBIO, PDBParser, Select

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("target", pdb_path)

        class CleanSelect(Select):
            def accept_residue(self, residue):
                rid = residue.get_id()
                hetfield = rid[0]
                # Specific water removal (selective)
                if waters_to_remove is not None:
                    chain = residue.get_parent().get_id() if residue.get_parent() else "?"
                    resid = f"{chain}:{residue.get_resname()}_{rid[1]}"
                    if resid in waters_to_remove:
                        return False
                    # Keep all other residues including waters not in the list
                    if hetfield == "W":
                        return True
                elif remove_waters and hetfield == "W":
                    return False
                if remove_ligands and hetfield != " " and hetfield != "W":
                    return False
                return True

        io = PDBIO()
        io.set_structure(struct)
        io.save(out_path, CleanSelect())

        if add_hydrogens:
            mol = Chem.MolFromPDBFile(out_path, removeHs=False)
            if mol is not None:
                mol = Chem.AddHs(mol, addCoords=True)
                Chem.MolToPDBFile(mol, out_path)
                log.info(f"  Polar hydrogens added to {out_path}")
            else:
                log.warning("  Could not add hydrogens via RDKit PDB parser.")

        pdbqt_path = out_path.replace(".pdb", ".pdbqt")
        converted = False

        if deps.get("prepare_receptor"):
            try:
                run_tool(
                    ["prepare_receptor", "-r", out_path, "-o", pdbqt_path],
                    timeout=CONFIG.prepare_receptor_timeout,
                )
                if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                    converted = True
                    log.info("  PDBQT via prepare_receptor")
            except RuntimeError:
                pass

        if not converted and deps.get("obabel"):
            try:
                run_tool(
                    ["obabel", out_path, "-O", pdbqt_path, "-h", "--gas"],
                    timeout=CONFIG.obabel_timeout_s,
                )
                if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                    converted = True
                    log.info("  PDBQT via obabel")
            except RuntimeError:
                pass

        if not converted:
            log.warning("  No external PDBQT tool found. Using RDKit fallback.")
            converted = _pdb_to_pdbqt_via_rdkit(out_path, pdbqt_path)
            if converted:
                log.info("  PDBQT via RDKit fallback")
            else:
                log.warning("  ⚠  All PDBQT methods failed. Returning cleaned PDB only.")

        return pdbqt_path if (converted and os.path.exists(pdbqt_path)) else out_path

    except Exception as exc:
        log.error(f"  ✗  Failed to clean {pdb_path}: {exc}")
        raise


def _get_residue_ca_coords(pdb_path: str, resid_list: List[str]) -> np.ndarray:
    """Extract Cα atom coordinates for the given residue identifiers.

    Returns an (N, 3) array of Cα coordinates.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("target", pdb_path)

    target: set = set()
    for entry in resid_list:
        resname = "".join(ch for ch in entry if ch.isalpha()).upper()
        seqnum = int("".join(ch for ch in entry if ch.isdigit()))
        target.add((resname, seqnum))

    ca_coords: list = []
    for model in struct:
        for chain in model:
            for residue in chain:
                rid = residue.get_id()
                if rid[0] != " ":
                    continue
                key = (residue.get_resname().strip().upper(), rid[1])
                if key in target:
                    if "CA" in residue:
                        ca_coords.append(residue["CA"].get_vector().get_array())
                    else:
                        log.warning(
                            f"  ⚠  No Cα found for {key[0]}{key[1]}. "
                            "Using first atom."
                        )
                        atoms = list(residue.get_atoms())
                        if atoms:
                            ca_coords.append(atoms[0].get_vector().get_array())

    if not ca_coords:
        log.error(
            f"  ✗  None of the requested residues {resid_list} were found "
            f"in structure. Available residues: "
            f"{[(r.get_resname(), r.get_id()[1]) for r in struct.get_residues()]}"
        )
        raise ValueError(f"No matching residues found in {pdb_path}")

    return np.asarray(ca_coords)


def compute_residue_centroid(pdb_path: str, resid_list: List[str]) -> np.ndarray:
    """Compute the geometric centroid of Cα atoms for the given list of residue identifiers."""
    ca_coords = _get_residue_ca_coords(pdb_path, resid_list)
    return ca_coords.mean(axis=0)


def prepare_targets(
    pdb_dir: str, work_dir: str, deps: Dict[str, Any],
    water_results: Any = None,
) -> Dict[str, Any]:
    """Phase 1 — Download, clean, and compute grid centres for all targets.

    When ``CONFIG.ensemble_mode`` is enabled and an ensemble directory
    is configured, this function loads multiple receptor PDB structures
    from that directory, prepares each one, and stores them under the
    ``"PBP2a_ensemble"`` key as a list of receptor dicts.

    When *water_results* is provided (from :func:`water_analysis.analyze_waters`),
    high-energy non-bridging waters are selectively removed from the holo
    structure, and bridging waters are retained to produce a water-aware
    receptor for ensemble docking.

    Parameters
    ----------
    pdb_dir : str
        Directory for downloaded PDB files.
    work_dir : str
        Working directory for intermediate files.
    deps : dict
        Dependency dictionary.
    water_results : WaterAnalysisResult or None
        Pre-computed water analysis; triggers selective water handling.

    Returns
        Dict with keys ``holo_pdb``, ``PBP2a``, ``trypsin``, ``CES1``,
        and optionally ``PBP2a_ensemble`` (list of dicts) and
        ``water_results``.
    """
    log.info("─── Phase 1: Target Preparation & Centroid Calculation ───")
    result: Dict[str, Any] = {}

    holo_path = fetch_structure(CONFIG.pdb_ids["PBP2a_holo"], pdb_dir)
    apo_path = fetch_structure(CONFIG.pdb_ids["PBP2a_apo"], pdb_dir)
    trypsin_path = fetch_structure(CONFIG.pdb_ids["trypsin"], pdb_dir)
    ces1_path = fetch_structure(CONFIG.pdb_ids["CES1"], pdb_dir)

    result["holo_pdb"] = holo_path

    if water_results is not None:
        result["water_results"] = water_results
        waters_to_remove = [w.identifier for w in get_waters_to_remove(water_results)]
        if waters_to_remove:
            log.info(f"  Removing {len(waters_to_remove)} high-energy waters from holo structure.")

        log.info("  Cleaning PBP2a (holo, with selected waters)…")
        holo_water_pdbqt = clean_pdb_structure(
            holo_path,
            os.path.join(work_dir, "PBP2a_holo_water.pdb"),
            deps=deps,
            waters_to_remove=waters_to_remove,
        )
        result["PBP2a_holo_water"] = {
            "pdbqt": holo_water_pdbqt,
        }
        log.info(f"  Water-aware holo receptor saved: {holo_water_pdbqt}")
    else:
        log.info("  Cleaning PBP2a (holo, protein-only)…")
        _ = clean_pdb_structure(
            holo_path,
            os.path.join(work_dir, "PBP2a_holo_clean.pdb"),
            deps=deps,
        )

    log.info("  Cleaning PBP2a (apo)…")
    pbp2a_pdbqt = clean_pdb_structure(
        apo_path,
        os.path.join(work_dir, "PBP2a_clean.pdb"),
        deps=deps,
    )

    cleaned_pdb = pbp2a_pdbqt.replace(".pdbqt", ".pdb")
    log.info("  Computing allosteric site centroid (ALA237, MET241, TYR159)…")
    allosteric_center = compute_residue_centroid(cleaned_pdb, CONFIG.allosteric_residues)
    log.info(f"    Allosteric site center: {allosteric_center}")

    log.info("  Computing active site centroid (SER403)…")
    active_center = compute_residue_centroid(cleaned_pdb, CONFIG.active_site_residues)
    log.info(f"    Active site center: {active_center}")

    allosteric_coords = _get_residue_ca_coords(cleaned_pdb, CONFIG.allosteric_residues)
    active_coords = _get_residue_ca_coords(cleaned_pdb, CONFIG.active_site_residues)

    for site_name, coords, box in [
        ("allosteric", allosteric_coords, CONFIG.allosteric_box_size),
        ("active", active_coords, CONFIG.active_box_size),
    ]:
        spread = coords.max(axis=0) - coords.min(axis=0)
        for dim, label in enumerate("XYZ"):
            required = spread[dim] + 4.0
            if box[dim] < required:
                log.warning(
                    f"  ⚠  {site_name.capitalize()} site box size ({box[dim]:.1f} Å "
                    f"along {label}) may be smaller than residue spread "
                    f"({spread[dim]:.1f} Å). Consider increasing to ≥ {required:.1f} Å."
                )

    result["PBP2a"] = {
        "pdbqt": pbp2a_pdbqt,
        "allosteric_center": allosteric_center,
        "active_center": active_center,
    }

    # ── Ensemble mode: load additional receptor structures ──
    if CONFIG.ensemble_mode and CONFIG.ensemble_structures_dir is not None:
        ens_dir = CONFIG.ensemble_structures_dir
        if not os.path.isdir(str(ens_dir)):
            log.warning(f"  ⚠  Ensemble directory '{ens_dir}' not found. Skipping ensemble mode.")
        else:
            pdb_files = sorted(
                [os.path.join(str(ens_dir), f) for f in os.listdir(str(ens_dir))
                 if f.endswith((".pdb", ".pdbqt"))]
            )
            if not pdb_files:
                log.warning(f"  ⚠  No PDB/PDBQT files found in '{ens_dir}'. Skipping ensemble.")
            else:
                log.info(f"  Ensemble mode: loading {len(pdb_files)} structure(s) from '{ens_dir}'.")
                ensemble_targets: List[Dict[str, Any]] = []
                for idx, pdb_path in enumerate(pdb_files):
                    stem = os.path.splitext(os.path.basename(pdb_path))[0]
                    if pdb_path.endswith(".pdbqt"):
                        out_pdbqt = os.path.join(work_dir, f"ens_{stem}.pdbqt")
                        if pdb_path != out_pdbqt:
                            shutil.copy2(pdb_path, out_pdbqt)
                        cleaned_for_centroids = pdb_path.replace(".pdbqt", ".pdb")
                        if not os.path.exists(cleaned_for_centroids):
                            cleaned_for_centroids = cleaned_pdb
                    else:
                        out_pdbqt = clean_pdb_structure(
                            pdb_path,
                            os.path.join(work_dir, f"ens_{stem}.pdb"),
                            deps=deps,
                        )
                        cleaned_for_centroids = out_pdbqt.replace(".pdbqt", ".pdb")

                    try:
                        ens_allocenter = compute_residue_centroid(cleaned_for_centroids, CONFIG.allosteric_residues)
                        ens_actcenter = compute_residue_centroid(cleaned_for_centroids, CONFIG.active_site_residues)
                    except (ValueError, Exception) as exc:
                        log.warning(f"  ⚠  Centroid computation failed for {pdb_path}: {exc}. "
                                    f"Using primary target centers.")
                        ens_allocenter = allosteric_center
                        ens_actcenter = active_center

                    ensemble_targets.append({
                        "pdbqt": out_pdbqt,
                        "allosteric_center": ens_allocenter,
                        "active_center": ens_actcenter,
                    })
                result["PBP2a_ensemble"] = ensemble_targets
                log.info(f"  Ensemble loaded: {len(ensemble_targets)} structures.")

    log.info("  Cleaning Human Trypsin (1UTN)…")
    tryp_pdbqt = clean_pdb_structure(
        trypsin_path,
        os.path.join(work_dir, "trypsin_clean.pdb"),
        deps=deps,
    )
    tryp_center = compute_residue_centroid(
        trypsin_path, CONFIG.trypsin_active_site_residues,
    )
    log.info(f"    Trypsin active site center: {tryp_center}")
    result["trypsin"] = {"pdbqt": tryp_pdbqt, "active_center": tryp_center}

    log.info("  Cleaning Human Carboxylesterase 1 (3KJZ)…")
    ces1_pdbqt = clean_pdb_structure(
        ces1_path,
        os.path.join(work_dir, "CES1_clean.pdb"),
        deps=deps,
    )
    ces1_center = compute_residue_centroid(
        ces1_path, CONFIG.ces1_active_site_residues,
    )
    log.info(f"    CES1 active site center: {ces1_center}")
    result["CES1"] = {"pdbqt": ces1_pdbqt, "active_center": ces1_center}

    grid_dir = os.path.join(work_dir, "grid_configs")
    os.makedirs(grid_dir, exist_ok=True)

    for site_name, center, box in [
        ("allosteric", allosteric_center, CONFIG.allosteric_box_size),
        ("active", active_center, CONFIG.active_box_size),
    ]:
        cfg_path = os.path.join(grid_dir, f"grid_{site_name}.txt")
        with open(cfg_path, "w") as f:
            f.write(f"center_x = {center[0]:.3f}\n")
            f.write(f"center_y = {center[1]:.3f}\n")
            f.write(f"center_z = {center[2]:.3f}\n")
            f.write(f"size_x = {box[0]:.1f}\n")
            f.write(f"size_y = {box[1]:.1f}\n")
            f.write(f"size_z = {box[2]:.1f}\n")
        log.info(f"  Grid config saved: {cfg_path}")

    log.info("─── Phase 1 complete ───")
    return result
