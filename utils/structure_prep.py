"""
Structure preparation helpers
==============================

Low-level structural utilities that operate on PDB / PDBQT files and RDKit
molecules: native-ligand extraction, RMSD computation, and centroid helpers.

These functions are self-contained and depend only on the standard library,
RDKit and BioPython (plus :mod:`utils.ligand_prep` for PDBQT preparation).
Keeping them here breaks the former circular import between
``discovery_pipeline`` and the ``utils`` package.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import List, Optional, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from Bio.PDB import PDBParser, PDBIO, Select

from utils.ligand_prep import LigandPreparator

log = logging.getLogger("AutoAntibiotic")

# Hydrogen variant names for OpenMM Modeller.addHydrogens(variants=...)
# Keyed by PROPKA residue type string → (protonated_form, deprotonated_form).
# Only entries whose *non-default* form is a valid OpenMM variant get applied;
# forms equal to the residue's default name (e.g. TYR) are never passed to
# addHydrogens because OpenMM rejects "default name" variants.
_PROPKA_VARIANT_MAP = {
    "ASP": ("ASH", "ASP"),   # ASP: neutral (protonated) / charged
    "GLU": ("GLH", "GLU"),   # GLU: neutral / charged
    "HIS": ("HIP", "HIE"),   # HIS: +charged / neutral (NE2-H)
    "LYS": ("LYS", "LYN"),   # LYS: +charged (3H) / neutral (2H)
    "CYS": ("CYS", "CYX"),   # CYS: neutral / deprotonated or disulfide
    "TYR": ("TYR", "TYM"),   # TYR: neutral (default) / tyrosinate (deprotonated)
    "SER": ("SER", "SER"),   # SER: pKa ~ 13, neutral at pH 7.4 (no variant)
    "THR": ("THR", "THR"),   # THR: neutral (no variant)
    "ARG": ("ARG", "ARG"),   # ARG: +charged (pKa ~ 12) (no variant)
}


def assign_protonation_states(
    pdb_path: str,
    pH: float = 7.4,
) -> Optional[dict]:
    """
    Run PROPKA on *pdb_path* to compute residue pKa values and determine
    the correct OpenMM hydrogen variants at the given pH.

    Returns a dict mapping ``(residue_name, residue_number)`` → variant string
    for residues whose predicted protonation state differs from the default,
    or an empty dict if all residues match default behaviour.
    Returns ``None`` if PROPKA cannot be run.
    """
    try:
        from propka.run import single as propka_single
    except ImportError:
        log.warning("  PROPKA not installed. Install with: pip install propka")
        return None

    if not os.path.exists(pdb_path):
        log.warning(f"  PDB not found for PROPKA: {pdb_path}")
        return None

    try:
        mol = propka_single(pdb_path, write_pka=False)
    except Exception as exc:
        log.warning(f"  PROPKA run failed: {exc}")
        return None

    # Get the first conformation
    conf_keys = list(mol.conformations.keys())
    if not conf_keys:
        log.warning("  PROPKA found no conformations")
        return None

    conf = mol.conformations[conf_keys[0]]
    variants: dict = {}

    for group in conf.groups:
        if not getattr(group, 'titratable', False):
            continue

        res_type = getattr(group, 'residue_type', None)
        if res_type is None:
            continue

        pka = getattr(group, 'pka_value', None)
        if pka is None:
            continue

        res_name = getattr(group.atom, 'res_name', '').strip()
        res_num = getattr(group.atom, 'res_num', None)
        if res_num is None:
            continue

        # Determine correct variant based on pKa vs pH
        variant_map = _PROPKA_VARIANT_MAP.get(res_type)
        if variant_map is None:
            continue

        protonated, deprotonated = variant_map

        if pka > pH:
            desired = protonated
        else:
            desired = deprotonated

        # Default protonation form for this residue at physiological pH:
        # acidic residues default to their deprotonated (charged) form, all
        # others default to their protonated / neutral form.
        is_acidic = res_type in ("ASP", "GLU")
        default_form = deprotonated if is_acidic else protonated

        # Only request an explicit OpenMM hydrogen variant when the PROPKA
        # prediction differs from the residue's default form AND the desired
        # form is a genuine non-default variant name. Passing the default
        # residue name (e.g. "TYR", "SER", "ARG") as a variant is illegal in
        # OpenMM's Modeller.addHydrogens.
        if desired != default_form:
            variants[(res_name, res_num)] = desired
            log.info(
                f"  PROPKA: {res_name}{res_num} ({res_type}) pKa={pka:.2f} "
                f"→ variant {desired} (non-default)"
            )
        elif res_type in ("LYS", "SER") and res_num in (403, 406):
            # Always log key active-site residues for verification
            log.info(
                f"  PROPKA: {res_name}{res_num} ({res_type}) pKa={pka:.2f} "
                f"→ variant {desired} (default)"
            )

    return variants


def build_openmm_variant_list(
    topology,
    propka_variants: dict,
) -> Optional[list]:
    """
    Convert PROPKA variants dict to a list suitable for
    ``Modeller.addHydrogens(variants=...)``.

    The list is indexed by residue index (0-based) matching the OpenMM
    topology's residue order. Each element is either ``None`` (auto) or a
    variant string.
    """
    if propka_variants is None:
        return None

    n_residues = topology.getNumResidues()
    variant_list = [None] * n_residues

    residue_list = list(topology.residues())
    for res_idx in range(n_residues):
        residue = residue_list[res_idx]
        key = (residue.name, int(residue.id))
        if key in propka_variants:
            variant_list[res_idx] = propka_variants[key]

    return variant_list


def _extract_native_ligand_from_holo(
    holo_pdb_path: str,
    output_ligand_smi: str,
    output_ligand_pdbqt: str,
    resname_override: Optional[str] = None,
) -> Optional[str]:
    """
    Parse the holo structure (6TKO), locate the co-crystallised ligand,
    write its SMILES to *output_ligand_smi* and its PDBQT to *output_ligand_pdbqt*.

    Args:
        holo_pdb_path: Path to the holo PDB structure.
        output_ligand_smi: Destination path for the ligand SMILES.
        output_ligand_pdbqt: Destination path for the ligand PDBQT.
        resname_override: Optional explicit ligand residue name (e.g. "CEF").
            When provided, auto-detection is skipped and the residue with this
            name is selected directly. Useful for complex structures where the
            heuristic picks the wrong molecule.

    Returns the SMILES string, or None on failure.
    """
    if resname_override is None:
        log.warning(
            "  ⚠  Native ligand extraction requires an explicit "
            "native_ligand_resname (config.yaml) for science redocking. "
            "Skipping auto-detection — returning None."
        )
        return None

    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("6TKO", holo_pdb_path)

        # ── Explicit resname override (required) ─────────────────────────────
        override = resname_override.strip().upper()
        lig_res = None
        chain_id = None
        for model in struct:
            for chain in model:
                for residue in chain:
                    if residue.get_resname().strip().upper() == override:
                        lig_res = residue
                        chain_id = chain.get_id()
                        break
                if lig_res is not None:
                    break
            if lig_res is not None:
                break
        if lig_res is None:
            log.warning(
                f"  ⚠  resname_override '{resname_override}' not found in "
                f"{holo_pdb_path}."
            )
            return None
        log.info(
            f"  Native ligand (resname override '{resname_override}'): "
            f"chain {chain_id}, residue {lig_res.get_resname()}"
        )

        # Write ligand as a separate PDB file
        pdbio = PDBIO()
        class LigSelect(Select):
            def accept_residue(self, residue):
                return residue is lig_res
        pdbio.set_structure(struct)
        lig_pdb = output_ligand_pdbqt.replace(".pdbqt", ".pdb")
        pdbio.save(lig_pdb, LigSelect())

        # Convert to MOL → SMILES via RDKit's PDB parser (or obabel fallback)
        mol = Chem.MolFromPDBFile(lig_pdb, removeHs=False)
        if mol is None:
            log.warning("  ⚠  RDKit could not read ligand PDB, trying obabel…")
            smi_file = output_ligand_smi
            try:
                subprocess.run(
                    ["obabel", lig_pdb, "-O", smi_file],
                    capture_output=True, timeout=30,
                )
                with open(smi_file) as f:
                    smi = f.readline().strip()
                if smi:
                    return smi
            except Exception:
                pass
            return None

        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)

        with open(output_ligand_smi, "w") as f:
            f.write(smi + "\n")
        log.info(f"  Native ligand SMILES: {smi}")

        # Convert to PDBQT via LigandPreparator
        try:
            mol_pdbqt = Chem.AddHs(mol)
            preparator = LigandPreparator()
            pdbqt_str = preparator.prepare(mol_pdbqt)
            with open(output_ligand_pdbqt, "w") as f:
                f.write(pdbqt_str)
            log.info(f"  Native ligand PDBQT written to {output_ligand_pdbqt}")
        except Exception as exc:
            log.warning(f"  ⚠  LigandPreparator failed for native ligand: {exc}")
            return None

        return smi

    except Exception as exc:
        log.error(f"  ✗  Native ligand extraction failed: {exc}")
        return None


def _compute_rmsd_docked_vs_crystal(
    docked_pdb: str, crystal_pdb: str
) -> Optional[float]:
    """
    Align the docked ligand to the crystal ligand and compute heavy-atom RMSD.

    Parses both PDB files with RDKit, removes hydrogens, and performs
    Kabsch-aligned RMSD on the MCS common substructure.  Also tries RDKit's
    GetBestRMS (Hungarian) as a fast path.

    The crystal PDB (from Bio.PDB extraction) has correct bonding but no
    hydrogens.  The docked PDB (from obabel conversion of the Vina PDBQT) may
    have corrupted bonding; the MCS route is robust to that.
    """
    try:
        # ── Reference: crystal ligand (no hydrogens) ──
        crystal = Chem.MolFromPDBFile(crystal_pdb, removeHs=True)
        if crystal is None:
            log.error("  ✗  Could not parse crystal PDB as an RDKit Mol.")
            return None
        try:
            Chem.SanitizeMol(crystal)
        except Exception:
            pass

        # ── Probe: docked ligand (obabel PDB may have unbound Hs) ──
        docked = Chem.MolFromPDBFile(docked_pdb, removeHs=True)
        if docked is None:
            log.error("  ✗  Could not parse docked PDB as an RDKit Mol.")
            return None

        # Attempt 1: direct GetBestRMS (fast Hungarian path).
        try:
            rms = AllChem.GetBestRMS(docked, crystal)
            if rms is not None and rms >= 0:
                log.info(f"  RMSD (GetBestRMS) = {rms:.3f} Å")
                return float(rms)
        except Exception:
            pass

        # Attempt 2: MCS-based Kabsch alignment.
        from rdkit.Chem import rdFMCS
        mcs = rdFMCS.FindMCS(
            [crystal, docked],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            matchValences=True,
            ringMatchesRingOnly=True,
        )
        if mcs.numAtoms < 4:
            log.warning(
                "  ⚠  Not enough MCS atoms for RMSD alignment "
                f"({mcs.numAtoms} found)."
            )
            return None

        mcs_smarts = Chem.MolFromSmarts(mcs.smartsString)
        ref_match = crystal.GetSubstructMatch(mcs_smarts)
        dock_match = docked.GetSubstructMatch(mcs_smarts)
        if not ref_match or not dock_match:
            log.warning("  ⚠  MCS substructure match failed.")
            return None

        ref_conf = crystal.GetConformer()
        dock_conf = docked.GetConformer()
        ref_pts = np.array([ref_conf.GetAtomPosition(i) for i in ref_match])
        dock_pts = np.array([dock_conf.GetAtomPosition(i) for i in dock_match])

        # Kabsch alignment
        ref_cent = ref_pts.mean(axis=0)
        dock_cent = dock_pts.mean(axis=0)
        ref_pts_c = ref_pts - ref_cent
        dock_pts_c = dock_pts - dock_cent
        H = dock_pts_c.T @ ref_pts_c
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        aligned = dock_pts_c @ R.T
        rmsd = float(np.sqrt(np.mean(np.sum((aligned - ref_pts_c) ** 2, axis=1))))
        log.info(f"  RMSD (Kabsch, {len(ref_match)} atoms) = {rmsd:.3f} Å")
        return rmsd

    except Exception as exc:
        log.error(f"  ✗  RMSD calculation failed: {exc}")
        return None


def _compute_core_rmsd(
    docked_pdb: str, crystal_pdb: str
) -> Optional[float]:
    """
    Heavy-atom RMSD of the *conserved, ring-constrained binding scaffold* only.

    For a flexible co-crystallised ligand (e.g. the cephalosporin ceftaroline /
    PBP2a ligand AI8) the solvent-exposed promoiety tail adopts a
    crystal-packing-dependent conformation that inflates the full-ligand RMSD and
    is irrelevant to binding-mode reproduction. Restricting the RMSD to the
    ring-constrained core (the beta-lactam / thiazolidine fused system that
    anchors Ser403) gives the scientifically meaningful redocking-accuracy
    metric used throughout PBP / beta-lactam docking validation literature.

    Returns the Kabsch-aligned heavy-atom RMSD over the largest common ring
    substructure, or None on failure.
    """
    try:
        from rdkit.Chem import rdFMCS
        crystal = Chem.MolFromPDBFile(crystal_pdb, removeHs=True)
        docked = Chem.MolFromPDBFile(docked_pdb, removeHs=True)
        if crystal is None or docked is None:
            return None
        mcs = rdFMCS.FindMCS(
            [crystal, docked],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            matchValences=True,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        if mcs.numAtoms < 4:
            return None
        smarts = Chem.MolFromSmarts(mcs.smartsString)
        ref_match = crystal.GetSubstructMatch(smarts)
        dock_match = docked.GetSubstructMatch(smarts)
        if not ref_match or not dock_match:
            return None
        ref_conf = crystal.GetConformer()
        dock_conf = docked.GetConformer()
        ref_pts = np.array([ref_conf.GetAtomPosition(i) for i in ref_match])
        dock_pts = np.array([dock_conf.GetAtomPosition(i) for i in dock_match])
        ref_cent = ref_pts.mean(axis=0)
        dock_cent = dock_pts.mean(axis=0)
        ref_pts_c = ref_pts - ref_cent
        dock_pts_c = dock_pts - dock_cent
        H = dock_pts_c.T @ ref_pts_c
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        aligned = dock_pts_c @ R.T
        rmsd = float(np.sqrt(np.mean(np.sum((aligned - ref_pts_c) ** 2, axis=1))))
        log.info(f"  Core RMSD (ring scaffold, {len(ref_match)} atoms) = {rmsd:.3f} Å")
        return rmsd
    except Exception as exc:
        log.warning(f"  Core RMSD calculation failed: {exc}")
        return None


def compute_residue_centroid(pdb_path: str, resid_list: List[str],
                              use_ca: bool = True) -> np.ndarray:
    """
    Compute the geometric centroid of specified atoms for the given list of
    residue identifiers (format: 'TYR105').

    When ``use_ca=True`` (default), the centroid is based on Cα atoms only
    (backwards-compatible behaviour). When ``use_ca=False``, ALL heavy atoms
    of each residue are used — this is appropriate for side-chain-defined
    binding pockets (allosteric site, catalytic triads).

    For homodimers like PBP2a (chains A/B) only the FIRST chain that
    contains any matching residue is used — averaging across chains
    produces a meaningless midpoint grid centre.

    Args:
        pdb_path: Path to PDB structure.
        resid_list: e.g. ["TYR105", "GLN199", "GLU237"].
        use_ca: If True, use Cα atoms only; if False, use all heavy atoms.

    Returns:
        (x, y, z) centroid as numpy array of shape (3,).
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("target", pdb_path)

    # Build set of (resname, seq_num) from input
    target = set()
    is_hetero_target = False
    for entry in resid_list:
        m = re.match(r"^([A-Za-z]{3})(\d+)$", entry)
        if m:
            is_hetero_target = False
            target.add((m.group(1).upper(), int(m.group(2))))
        else:
            is_hetero_target = True
            target.add((entry.strip().upper(), None))

    atom_coords = []
    found_chain = False
    for model in struct:
        if found_chain:
            break
        for chain in model:
            if found_chain:
                break
            for residue in chain:
                rid = residue.get_id()
                if rid[0] != " " and not is_hetero_target:
                    continue
                resname = residue.get_resname().strip().upper()
                if is_hetero_target:
                    if any(t[0] == resname for t in target):
                        atoms = list(residue.get_atoms())
                        if atoms:
                            coords = np.array([a.get_vector().get_array() for a in atoms])
                            atom_coords.append(coords.mean(axis=0))
                            found_chain = True
                            break
                else:
                    key = (resname, rid[1])
                    if key in target:
                        if use_ca:
                            if "CA" in residue:
                                atom_coords.append(residue["CA"].get_vector().get_array())
                                found_chain = True
                            else:
                                atoms = list(residue.get_atoms())
                                if atoms:
                                    coords = np.array([a.get_vector().get_array() for a in atoms])
                                    atom_coords.append(coords.mean(axis=0))
                                    found_chain = True
                        else:
                            # Use all heavy atoms (side-chain + backbone)
                            atoms = [a for a in residue if a.element and a.element.strip().upper() != "H"]
                            if atoms:
                                coords = np.array([a.get_vector().get_array() for a in atoms])
                                atom_coords.append(coords.mean(axis=0))
                                found_chain = True

    if not atom_coords:
        log.error(
            f"  ✗  None of the requested residues {resid_list} were found "
            f"in structure. Available residues: "
            f"{[(r.get_resname(), r.get_id()[1]) for r in struct.get_residues()]}"
        )
        raise ValueError(f"No matching residues found in {pdb_path}")

    centroid = np.mean(atom_coords, axis=0)
    return centroid


def merge_conserved_waters(
    receptor_pdb: str,
    conserved_water_pdb: str,
    output_pdb: str,
) -> str:
    """Merge conserved active-site waters from *conserved_water_pdb* into
    the receptor structure at *receptor_pdb* and write the combined
    structure to *output_pdb*.

    The function parses both PDB files with Bio.PDB, collects all water
    residues (HOH/WAT/SOL) from the conserved-waters file, and appends
    them to the receptor structure before writing the combined PDB.
    This is used to prepare a water-included receptor for docking when
    explicit waters are known to be conserved in the active site.

    Args:
        receptor_pdb: Path to the receptor PDB (protein only, no waters).
        conserved_water_pdb: Path to the conserved waters PDB (output of
            ``scripts/conserved_water_analysis.py``).
        output_pdb: Destination path for the combined receptor + waters PDB.

    Returns:
        The path to *output_pdb*.

    Raises:
        FileNotFoundError: If either input PDB does not exist.
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    if not os.path.exists(receptor_pdb):
        raise FileNotFoundError(f"Receptor PDB not found: {receptor_pdb}")
    if not os.path.exists(conserved_water_pdb):
        raise FileNotFoundError(
            f"Conserved waters PDB not found: {conserved_water_pdb}"
        )

    parser = PDBParser(QUIET=True)

    # Load the receptor structure
    rec_struct = parser.get_structure("receptor", receptor_pdb)

    # Load the conserved waters structure and collect water residues
    water_struct = parser.get_structure("waters", conserved_water_pdb)
    water_residues = []
    for model in water_struct:
        for chain in model:
            for residue in chain:
                rname = residue.get_resname().strip().upper()
                if rname in ("HOH", "WAT", "SOL"):
                    water_residues.append(residue)

    log.info(
        f"  merge_conserved_waters: {len(water_residues)} conserved water "
        f"residues found in {conserved_water_pdb}"
    )

    if not water_residues:
        log.warning(
            "  No conserved water residues found; writing receptor-only PDB."
        )
        io = PDBIO()
        io.set_structure(rec_struct)
        io.save(output_pdb)
        return output_pdb

    # Append water residues to the receptor structure.
    # Bio.PDB does not provide a direct API for adding residues to an
    # existing structure, so we work at the level of the internal
    # child list of the Model/Chain objects.
    # We add waters to the first chain of the first model in the receptor.
    receptor_model = rec_struct[0]
    receptor_chain = None
    for chain in receptor_model:
        receptor_chain = chain
        break

    if receptor_chain is None:
        raise RuntimeError("Receptor structure has no chains")

    # Determine the next residue ID to use (avoid ID collisions)
    existing_ids = {r.id for r in receptor_chain}
    next_id = 1
    for res in water_residues:
        # Use the original residue ID from the water structure but
        # ensure it does not collide with existing receptor residue IDs.
        rid = res.id
        while rid in existing_ids:
            rid = (" ", rid[1] + 1, " ")
        existing_ids.add(rid)
        receptor_chain.add(residue=res)

    io = PDBIO()
    io.set_structure(rec_struct)
    io.save(output_pdb)
    log.info(f"  merge_conserved_waters: wrote {output_pdb} "
             f"({len(water_residues)} waters merged)")
    return output_pdb


