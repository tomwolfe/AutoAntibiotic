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
            preparator = LigandPreparator()
            pdbqt_str = preparator.prepare(mol)
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

    Uses RDKit's AllChem.GetBestRMS after MCS-based atom-order alignment.
    Returns None if MCS cannot be found or any error occurs.
    """
    try:
        docked_mol = Chem.MolFromPDBFile(docked_pdb, removeHs=False)
        if docked_mol is None:
            log.error("  ✗  Could not parse docked PDB as an RDKit Mol.")
            return None

        crystal_mol = Chem.MolFromPDBFile(crystal_pdb, removeHs=False)
        if crystal_mol is None:
            log.error("  ✗  Could not parse crystal PDB as an RDKit Mol.")
            return None

        rms = AllChem.GetBestRMS(docked_mol, crystal_mol, 0, 0)
        if rms is None:
            log.warning("  ⚠  MCS alignment failed — cannot order atoms consistently.")
            return None

        return rms

    except Exception as exc:
        log.error(f"  ✗  RMSD calculation failed: {exc}")
        return None


def compute_residue_centroid(pdb_path: str, resid_list: List[str]) -> np.ndarray:
    """
    Compute the geometric centroid of Cα atoms for the given list of
    residue identifiers (format: 'ALA237').

    Args:
        pdb_path: Path to PDB structure.
        resid_list: e.g. ["ALA237", "MET241", "TYR159"].

    Returns:
        (x, y, z) centroid as numpy array of shape (3,).
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("target", pdb_path)

    # Build set of (resname, seq_num) from input
    target = set()
    for entry in resid_list:
        # Separate alphabetic resname from numeric seq_id
        resname = "".join(ch for ch in entry if ch.isalpha()).upper()
        seqnum = int("".join(ch for ch in entry if ch.isdigit()))
        target.add((resname, seqnum))

    ca_coords = []
    for model in struct:
        for chain in model:
            for residue in chain:
                rid = residue.get_id()
                # Ignore hetero atoms
                if rid[0] != " ":
                    continue
                key = (residue.get_resname().strip().upper(), rid[1])
                if key in target:
                    if "CA" in residue:
                        ca_coords.append(residue["CA"].get_vector().get_array())
                    else:
                        log.warning(
                            f"  ⚠  No Cα found for {key[0]}{key[1]}. "
                            "Using geometric center of all residue atoms."
                        )
                        atoms = list(residue.get_atoms())
                        if atoms:
                            coords = np.array([a.get_vector().get_array() for a in atoms])
                            ca_coords.append(coords.mean(axis=0))

    if not ca_coords:
        log.error(
            f"  ✗  None of the requested residues {resid_list} were found "
            f"in structure. Available residues: "
            f"{[(r.get_resname(), r.get_id()[1]) for r in struct.get_residues()]}"
        )
        raise ValueError(f"No matching residues found in {pdb_path}")

    centroid = np.mean(ca_coords, axis=0)
    return centroid
