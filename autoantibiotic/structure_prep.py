from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import CONFIG, ConfigurationError
from .io_utils import (
    AutoAntibioticError,
    OpenBabelError,
    VinaError,
    download_with_retry,
    log,
    run_tool,
)
from .water_analysis import get_waters_to_remove

try:
    import pdbfixer  # noqa: F401
    _HAVE_PDBFIXER = True
except ImportError:
    _HAVE_PDBFIXER = False

# Required atoms for critical binding-site residues.
# Backbone atoms are mandatory; side-chain atoms trigger PDBFixer repair if missing.
BACKBONE_ATOMS: Set[str] = {"N", "CA", "C", "O"}

CRITICAL_RESIDUE_ATOMS: Dict[str, Set[str]] = {
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
}


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
    """Minimal PDB → PDBQT conversion using RDKit fallback.

    Produces a rigid-receptor PDBQT (plain ATOM records, no
    ROOT/BRANCH sections) suitable for Vina rigid docking.
    """
    try:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is None:
            return False
        AllChem.ComputeGasteigerCharges(mol)

        _atom_type_map = {
            "C": "C", "c": "C",
            "N": "N", "n": "N",
            "O": "O", "o": "O",
            "S": "S", "s": "S",
            "P": "P", "p": "P",
            "F": "F", "f": "F",
            "Cl": "Cl", "Br": "Br",
            "I": "I",
            "H": "H",
        }

        conf = mol.GetConformer()
        lines: list = []
        for i, atom in enumerate(mol.GetAtoms()):
            atom_no = i + 1
            elem = atom.GetSymbol()
            pdbx = conf.GetAtomPosition(i)
            gasteiger = atom.GetDoubleProp("_GasteigerCharge")
            ad_type = _atom_type_map.get(elem, "C")

            x, y, z = pdbx.x, pdbx.y, pdbx.z
            atom_name = f" {elem:<3s}"[:4]
            line = (
                f"ATOM  {atom_no:>5d} {atom_name} PRT X   1    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  0.00  0.00"
                f"{gasteiger:>10.4f} {ad_type:<2s}\n"
            )
            lines.append(line)

        with open(pdbqt_path, "w") as f:
            f.writelines(lines)
        return True

    except Exception as exc:
        log.warning(f"  RDKit PDBQT fallback failed: {exc}")
        return False


def _is_rigid_pdbqt(pdbqt_path: str) -> bool:
    """Check whether a PDBQT file is suitable as a rigid Vina receptor.

    Returns True if the file contains ATOM/HETATM records and no
    flex-receptor tags (ROOT/BRANCH/ENDBRANCH).
    """
    if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
        return False
    try:
        with open(pdbqt_path) as f:
            head = f.read(50000)
        has_atoms = "ATOM" in head or "HETATM" in head
        has_flex_tags = "ROOT" in head or "BRANCH" in head or "ENDBRANCH" in head
        return has_atoms and not has_flex_tags
    except Exception:
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
                    if _is_rigid_pdbqt(pdbqt_path):
                        converted = True
                        log.info("  PDBQT via prepare_receptor")
                    else:
                        log.debug("  prepare_receptor produced flexible PDBQT; trying other methods.")
                        os.remove(pdbqt_path)
            except (RuntimeError, OpenBabelError, AutoAntibioticError) as exc:
                log.debug(f"  prepare_receptor failed: {exc}")
                pass

        if not converted and deps.get("obabel"):
            try:
                run_tool(
                    ["obabel", out_path, "-O", pdbqt_path, "-h", "--gas"],
                    timeout=CONFIG.obabel_timeout_s,
                )
                if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                    if _is_rigid_pdbqt(pdbqt_path):
                        converted = True
                        log.info("  PDBQT via obabel")
                    else:
                        log.debug("  obabel produced flexible PDBQT.")
                        os.remove(pdbqt_path)
            except (RuntimeError, OpenBabelError, AutoAntibioticError) as exc:
                log.debug(f"  obabel failed: {exc}")
                pass

        if not converted:
            if deps.get("obabel") or deps.get("prepare_receptor"):
                log.warning("  External PDBQT tool(s) failed or timed out. Using RDKit fallback.")
            else:
                log.warning("  No external PDBQT tool available. Using RDKit fallback.")
            converted = _pdb_to_pdbqt_via_rdkit(out_path, pdbqt_path)
            if converted:
                log.info("  PDBQT via RDKit fallback")
                if not _is_rigid_pdbqt(pdbqt_path):
                    log.warning("  RDKit fallback PDBQT is invalid; returning PDB only.")
                    converted = False
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


def _build_ensemble_targets(
    pdb_files: List[str],
    work_dir: str,
    fallback_allocenter: np.ndarray,
    fallback_actcenter: np.ndarray,
    deps: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Prepare a list of PDB files as ensemble docking targets.

    Each PDB is cleaned to PDBQT and its binding-site centroids are
    computed.  If centroid computation fails for a structure the
    primary-target centroids are used as fallback.

    Returns a list of dicts with keys ``pdbqt``, ``allosteric_center``,
    and ``active_center``.
    """
    ensemble_targets: List[Dict[str, Any]] = []
    for idx, pdb_path in enumerate(pdb_files):
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        if pdb_path.endswith(".pdbqt"):
            out_pdbqt = os.path.join(work_dir, f"ens_{stem}.pdbqt")
            if pdb_path != out_pdbqt:
                shutil.copy2(pdb_path, out_pdbqt)
            cleaned_for_centroids = pdb_path.replace(".pdbqt", ".pdb")
            if not os.path.exists(cleaned_for_centroids):
                cleaned_for_centroids = None
        else:
            out_pdbqt = clean_pdb_structure(
                pdb_path,
                os.path.join(work_dir, f"ens_{stem}.pdb"),
                deps=deps,
            )
            cleaned_for_centroids = out_pdbqt.replace(".pdbqt", ".pdb")

        try:
            if cleaned_for_centroids and os.path.exists(cleaned_for_centroids):
                ens_allocenter = compute_residue_centroid(cleaned_for_centroids, CONFIG.allosteric_residues)
                ens_actcenter = compute_residue_centroid(cleaned_for_centroids, CONFIG.active_site_residues)
            else:
                raise ValueError("No cleaned PDB available for centroid computation")
        except (ValueError, Exception) as exc:
            log.warning(f"  ⚠  Centroid computation failed for {pdb_path}: {exc}. "
                        f"Using primary target centers.")
            ens_allocenter = fallback_allocenter
            ens_actcenter = fallback_actcenter

        ensemble_targets.append({
            "pdbqt": out_pdbqt,
            "allosteric_center": ens_allocenter,
            "active_center": ens_actcenter,
        })
    return ensemble_targets


def validate_receptor_integrity(pdb_path: str, work_dir: str, deps: Dict[str, Any]) -> str:
    """Validate and optionally repair a prepared receptor PDB file.

    Checks that every residue listed in ``CONFIG.allosteric_residues`` and
    ``CONFIG.active_site_residues`` is present with complete backbone atoms
    (N, CA, C, O).  If key side-chain atoms are missing and PDBFixer is
    available, the structure is automatically repaired.

    Args:
        pdb_path: Path to the prepared PDB file.
        work_dir: Working directory for intermediate / repaired files.
        deps: Dependency dictionary (used for PDBQT conversion).

    Returns:
        Path to the validated (and potentially repaired) PDB file.

    Raises:
        ConfigurationError:
            - If Bio.PDB is not installed and validation was requested.
            - If any critical residue is entirely missing.
            - If backbone atoms are missing from a critical residue.
            - If PDBFixer repair fails and ``CONFIG.strict_receptor_validation`` is True.
    """
    log.info("  ── Receptor integrity check ──")

    # Combine all critical residue specifiers
    critical_specs: List[str] = list(CONFIG.allosteric_residues) + list(CONFIG.active_site_residues)
    # Build (resname_upper, seqnum) lookup
    target_residues: List[Tuple[str, int]] = []
    for spec in critical_specs:
        resname = "".join(ch for ch in spec if ch.isalpha()).upper()
        seqnum = int("".join(ch for ch in spec if ch.isdigit()))
        target_residues.append((resname, seqnum))

    try:
        from Bio.PDB import PDBParser
    except ImportError:
        log.warning("  Bio.PDB not available — skipping receptor integrity validation.")
        return pdb_path

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("receptor", pdb_path)
    chain_id = None

    # Locate the chain containing critical residues
    for model in struct:
        for chain in model:
            for residue in chain:
                rid = residue.get_id()
                if rid[0] != " ":
                    continue
                key = (residue.get_resname().strip().upper(), rid[1])
                if key in target_residues:
                    chain_id = chain.get_id()
                    break
            if chain_id is not None:
                break
        if chain_id is not None:
            break

    if chain_id is None:
        raise ConfigurationError(
            f"None of the critical residues {critical_specs} were found in {pdb_path}. "
            "Receptor structure is invalid for docking."
        )

    # Scan residues and collect repair info
    residues_found: Dict[Tuple[str, int], Dict[str, List[str]]] = {}
    for model in struct:
        for chain in model:
            if chain.get_id() != chain_id:
                continue
            for residue in chain:
                rid = residue.get_id()
                if rid[0] != " ":
                    continue
                resname = residue.get_resname().strip().upper()
                seqnum = rid[1]
                key = (resname, seqnum)
                if key not in target_residues:
                    continue
                atom_names = {atom.get_id() for atom in residue.get_atoms()}
                expected = CRITICAL_RESIDUE_ATOMS.get(resname, BACKBONE_ATOMS)
                missing = sorted(expected - atom_names)
                residues_found[key] = {"resname": resname, "missing": missing, "present_atoms": atom_names}

    # Check for entirely missing residues
    missing_residues = [r for r in target_residues if r not in residues_found]
    if missing_residues:
        raise ConfigurationError(
            f"Critical residues entirely missing from structure: "
            f"{[f'{r[0]}{r[1]}' for r in missing_residues]}. "
            "Cannot proceed with docking."
        )

    # Categorise issues
    backbone_missing: List[str] = []
    sidechain_missing: List[str] = []
    repair_needed: List[str] = []

    for key, info in residues_found.items():
        resname, seqnum = key
        label = f"{resname}{seqnum}"
        missing = info["missing"]
        expected = CRITICAL_RESIDUE_ATOMS.get(resname, BACKBONE_ATOMS)
        total_expected = len(expected)

        missing_backbone = [a for a in missing if a in BACKBONE_ATOMS]
        missing_sidechain = [a for a in missing if a not in BACKBONE_ATOMS]

        if missing_backbone:
            backbone_missing.append(f"{label} (missing {missing_backbone})")

        if missing_sidechain:
            sidechain_missing.append(label)
            ratio = len(missing_sidechain) / len(expected - BACKBONE_ATOMS)
            if ratio > 0.5:
                repair_needed.append(label)

    # Backbone completeness is mandatory
    if backbone_missing:
        raise ConfigurationError(
            f"Critical backbone atoms are missing from residues: {backbone_missing}. "
            "Receptor structure is not valid for docking."
        )

    # Log warnings for all incomplete residues
    for label in sidechain_missing:
        log.warning(f"  ⚠  {label} has missing side-chain atoms (will be repaired if PDBFixer available).")

    # Attempt repair via PDBFixer
    if repair_needed:
        if not _HAVE_PDBFIXER:
            msg = (
                f"Residues {repair_needed} are missing >50% of side-chain atoms, "
                "but PDBFixer is not installed. Install with:\n"
                "  conda install -c conda-forge pdbfixer"
            )
            if CONFIG.strict_receptor_validation:
                raise ConfigurationError(msg)
            log.warning(f"  ⚠  {msg}")
            log.warning("  Continuing with incomplete structure (strict_receptor_validation=False).")
        else:
            log.info(f"  Repairing residues {repair_needed} with PDBFixer…")
            try:
                from pdbfixer import PDBFixer

                fixer = PDBFixer(filename=pdb_path)
                fixer.findMissingResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(7.0)

                stem = os.path.splitext(os.path.basename(pdb_path))[0]
                repaired_path = os.path.join(work_dir, f"{stem}_validated.pdb")
                with open(repaired_path, "w") as f:
                    from openmm.app import PDBFile
                    PDBFile.writeFile(fixer.topology, fixer.positions, f)

                log.info(f"  Repaired structure saved to {repaired_path}")

                # Re-validate the repaired structure
                # PDBFixer renumbers residues sequentially (starting at 1),
                # so we match by residue name and position order instead of number.
                repaired_struct = parser.get_structure("repaired", repaired_path)
                repair_ok = True
                expected_order = [(r[0], i) for i, r in enumerate(target_residues)]
                found_pos = 0
                for model in repaired_struct:
                    for chain in model:
                        for residue in chain:
                            rid = residue.get_id()
                            if rid[0] != " ":
                                continue
                            resname = residue.get_resname().strip().upper()
                            if found_pos < len(expected_order) and resname == expected_order[found_pos][0]:
                                expected_atoms = CRITICAL_RESIDUE_ATOMS.get(resname, BACKBONE_ATOMS)
                                atom_names = {atom.get_id() for atom in residue.get_atoms()}
                                heavy_only = {a for a in atom_names if not a.startswith("H")}
                                still_missing = expected_atoms - heavy_only
                                if still_missing:
                                    log.warning(f"  ⚠  Repair incomplete for {resname}: still missing {sorted(still_missing)}")
                                    if still_missing & BACKBONE_ATOMS:
                                        raise ConfigurationError(
                                            f"PDBFixer repair failed: backbone atoms still missing from {resname}."
                                        )
                                    repair_ok = False
                                found_pos += 1

                if found_pos < len(expected_order):
                    log.warning(
                        f"  ⚠  PDBFixer repair may not have covered all residues "
                        f"(found {found_pos}/{len(expected_order)})"
                    )
                    repair_ok = False

                if repair_ok:
                    log.info("  Receptor integrity validated after PDBFixer repair.")
                else:
                    log.warning("  Receptor integrity partially restored after repair.")

                pdb_path = repaired_path

            except Exception as exc:
                raise ConfigurationError(
                    f"PDBFixer repair failed for {pdb_path}: {exc}"
                ) from exc
    else:
        log.info("  All critical residues are complete. No repair needed.")

    log.info("  ── Receptor integrity check complete ──")
    return pdb_path


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

    # ── Receptor integrity validation ──
    validated_pdb = validate_receptor_integrity(cleaned_pdb, work_dir, deps)
    if validated_pdb != cleaned_pdb:
        cleaned_pdb = validated_pdb
        # Regenerate PDBQT from the validated structure
        validated_pdbqt = validated_pdb.replace(".pdb", ".pdbqt")
        if not os.path.exists(validated_pdbqt):
            log.info("  Converting validated PDB to PDBQT …")
            try:
                from rdkit import Chem
                mol = Chem.MolFromPDBFile(validated_pdb, removeHs=False)
                if mol is not None:
                    mol = Chem.AddHs(mol, addCoords=True)
                    Chem.MolToPDBFile(mol, validated_pdb)
            except Exception:
                pass
            _pdb_to_pdbqt_via_rdkit(validated_pdb, validated_pdbqt)
            if os.path.exists(validated_pdbqt):
                pbp2a_pdbqt = validated_pdbqt
                log.info(f"  Validated PDBQT: {validated_pdbqt}")

    log.info("  Computing allosteric site centroid (ASN159, GLU237, ARG241)…")
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
    if CONFIG.ensemble_mode:
        if CONFIG.ensemble_structures_dir is not None:
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
                    ensemble_targets = _build_ensemble_targets(pdb_files, work_dir, allosteric_center, active_center, deps)
                    result["PBP2a_ensemble"] = ensemble_targets
                    log.info(f"  Ensemble loaded: {len(ensemble_targets)} structures.")
        else:
            pdb_ids = CONFIG.default_ensemble_pdb_ids
            log.info(f"  Ensemble mode: fetching {len(pdb_ids)} default PDB structures {pdb_ids}…")
            pdb_files: List[str] = []
            for pdb_id in pdb_ids:
                try:
                    pdb_path = fetch_structure(pdb_id, pdb_dir)
                    pdb_files.append(pdb_path)
                except Exception as exc:
                    log.warning(f"  ⚠  Failed to fetch PDB {pdb_id}: {exc}. Skipping.")
            if pdb_files:
                ensemble_targets = _build_ensemble_targets(pdb_files, work_dir, allosteric_center, active_center, deps)
                result["PBP2a_ensemble"] = ensemble_targets
                log.info(f"  Ensemble loaded: {len(ensemble_targets)} structures from fetched PDBs.")
            else:
                log.warning("  ⚠  No ensemble structures could be fetched. Running without ensemble.")

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

    log.info("  Cleaning Human Carboxylesterase 1 (1YA4)…")
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
