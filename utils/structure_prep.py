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


# Vina atom types are single/short tokens; we map the PDB element column to a
# valid Vina atom type so the receptor PDBQT we write is actually consumable.
_RECEPTOR_PDBQT_ATOM_TYPE = {
    "C": "C",
    "N": "N",
    "O": "O",
    "S": "S",
    "P": "P",
    "F": "F",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "H": "H",
    "NA": "Na",
    "MG": "Mg",
    "ZN": "Zn",
    "FE": "Fe",
    "CA": "Ca",
    "MN": "Mn",
}


def write_receptor_pdbqt(pdb_path: str, pdbqt_path: str) -> bool:
    """
    Write a receptor PDBQT file from a cleaned receptor PDB using RDKit/Bio.PDB.

    This is the OpenBabel-independent fallback used by
    :func:`discovery_pipeline.clean_pdb_structure` when ``obabel`` is not on
    ``PATH``. It produces a *real* PDBQT file (Gasteiger charges + Vina atom
    types + the rigid ``ATOM`` records Vina expects) rather than copying the
    PDB verbatim, which Vina would reject as an invalid PDBQT.

    Returns ``True`` on success, ``False`` if the receptor cannot be parsed.
    """
    # Prefer RDKit so we can assign Gasteiger charges (required by Vina). RDKit
    # parses standard crystallographic PDBs; if it fails (e.g. a minimal mock
    # PDB), fall back to Bio.PDB coordinate/element extraction with charge 0.0.
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    gasteiger = {}
    if mol is not None:
        try:
            from rdkit.Chem import AllChem

            mol = Chem.AddHs(mol, addCoords=True)
            AllChem.ComputeGasteigerCharges(mol)
            for atom in mol.GetAtoms():
                try:
                    c = float(atom.GetProp("_GasteigerCharge"))
                except Exception:
                    c = 0.0
                if not np.isfinite(c):
                    c = 0.0
                gasteiger[atom.GetIdx()] = c
        except Exception as exc:
            log.warning(f"  ⚠  Gasteiger charge assignment failed: {exc}")

    lines = ["REMARK  AutoAntibiotic RDKit/Bio.PDB-generated receptor PDBQT (no obabel)"]
    serial = 1

    if mol is not None and gasteiger:
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            elem = atom.GetSymbol().upper()
            atom_type = _RECEPTOR_PDBQT_ATOM_TYPE.get(elem, elem)
            charge = gasteiger.get(atom.GetIdx(), 0.0)
            pdb_atom = atom.GetPDBResidueInfo()
            if pdb_atom is not None:
                res_name = pdb_atom.GetResidueName().strip() or "RECP"
                chain = pdb_atom.GetChainId() or "A"
                res_seq = pdb_atom.GetResidueNumber() or 1
                atom_name = pdb_atom.GetName().strip() or f"{elem:>2s}"
            else:
                res_name, chain, res_seq, atom_name = "RECP", "A", 1, f"{elem:>2s}"
            line = (
                "ATOM  "
                + f"{serial:5d} "
                + f"{atom_name[:4]:<4s}"
                + f"{res_name[:3]:<3s} "
                + f"{chain:1s}"
                + f"{res_seq:4d}"
                + "    "
                + f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                + f"   {charge:7.4f} "
                + f"{atom_type:<2s}"
            )
            lines.append(line)
            serial += 1
        lines.append("TER")
        lines.append("END")
        with open(pdbqt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        log.info(f"  Receptor PDBQT written via RDKit fallback: {pdbqt_path}")
        return True

    # ── Bio.PDB fallback (handles mock/minimal PDBs RDKit won't parse) ──
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("receptor", pdb_path)
        for model in struct:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        try:
                            coord = atom.get_vector().get_array()
                        except Exception:
                            continue
                        elem = atom.element.strip().upper() if atom.element else ""
                        if not elem:
                            # Infer element from atom name's first alpha char.
                            for ch in atom.get_name():
                                if ch.isalpha():
                                    elem = ch.upper()
                                    break
                        atom_type = _RECEPTOR_PDBQT_ATOM_TYPE.get(elem, elem or "C")
                        res_name = residue.get_resname().strip()[:3] or "RECP"
                        res_seq = residue.get_id()[1] if len(residue.get_id()) > 1 else 1
                        atom_name = atom.get_name().strip() or f"{elem:>2s}"
                        line = (
                            "ATOM  "
                            + f"{serial:5d} "
                            + f"{atom_name[:4]:<4s}"
                            + f"{res_name[:3]:<3s} "
                            + f"{chain.get_id() if hasattr(chain, 'get_id') else 'A':1s}"
                            + f"{res_seq:4d}"
                            + "    "
                            + f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                            + f"   {0.0:7.4f} "
                            + f"{atom_type:<2s}"
                        )
                        lines.append(line)
                        serial += 1
        lines.append("TER")
        lines.append("END")
        with open(pdbqt_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        log.info(f"  Receptor PDBQT written via Bio.PDB fallback: {pdbqt_path}")
        return True
    except Exception as exc:
        log.warning(f"  ⚠  Bio.PDB receptor PDBQT fallback failed: {exc}")
        return False
