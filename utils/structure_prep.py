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
from typing import List, Optional

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from Bio.PDB import PDBParser, PDBIO, Select

from utils.ligand_prep import LigandPreparator

log = logging.getLogger("AutoAntibiotic")


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


