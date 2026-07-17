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


def compute_residue_centroid(pdb_path: str, resid_list: List[str]) -> np.ndarray:
    """
    Compute the geometric centroid of Cα atoms for the given list of
    residue identifiers (format: 'ALA237').

    For homodimers like PBP2a (chains A/B) only the FIRST chain that
    contains any matching residue is used — averaging across chains
    produces a meaningless midpoint grid centre.

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
    is_hetero_target = False
    for entry in resid_list:
        # Standard residue format: exactly 3 alpha chars + digits (e.g. "SER403")
        m = re.match(r"^([A-Za-z]{3})(\d+)$", entry)
        if m:
            is_hetero_target = False
            target.add((m.group(1).upper(), int(m.group(2))))
        else:
            # Non-standard residue / ligand (e.g. "AI8"): match only by name
            is_hetero_target = True
            target.add((entry.strip().upper(), None))

    ca_coords = []
    found_chain = False
    for model in struct:
        if found_chain:
            break
        for chain in model:
            if found_chain:
                break
            for residue in chain:
                rid = residue.get_id()
                # Only skip hetero atoms when searching for standard residues
                if rid[0] != " " and not is_hetero_target:
                    continue
                resname = residue.get_resname().strip().upper()
                if is_hetero_target:
                    # For non-standard residues (ligands), match by name only.
                    # Use the FIRST matching residue only.
                    if any(t[0] == resname for t in target):
                        atoms = list(residue.get_atoms())
                        if atoms:
                            coords = np.array([a.get_vector().get_array() for a in atoms])
                            ca_coords.append(coords.mean(axis=0))
                            found_chain = True
                            break
                else:
                    key = (resname, rid[1])
                    if key in target:
                        if "CA" in residue:
                            ca_coords.append(residue["CA"].get_vector().get_array())
                            found_chain = True
                        else:
                            atoms = list(residue.get_atoms())
                            if atoms:
                                coords = np.array([a.get_vector().get_array() for a in atoms])
                                ca_coords.append(coords.mean(axis=0))
                                found_chain = True

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


# ── Side-chain torsion topology for Vina flexible-residue PDBQT ──
#
# Vina's ``--flex`` receptor requires each flexible residue to be written as a
# torsion tree: the rigid backbone lives inside ROOT/ENDROOT and every
# rotatable side-chain bond opens a nested BRANCH…ENDBRANCH block. Emitting
# bare ATOM records inside BEGIN_RES/END_RES (no ROOT/BRANCH) makes Vina abort
# with "Unknown or inappropriate tag found in flex residue".
#
# For each residue type we list the rotatable side-chain bonds in
# root→leaf order as (parent_atom, child_atom); the child atom (and every
# atom that appears after it in the chain, up to the next branch point) moves
# with that torsion. ``chi_atoms`` maps each rotatable bond's child atom to the
# side-chain atoms that belong to (and beyond) that bond, so we can nest the
# BRANCH blocks correctly. The backbone atoms N, CA, C, O always form the ROOT.
_FLEX_BACKBONE = ("N", "CA", "C", "O", "OXT", "H", "HA")

# Ordered rotatable-bond chains (linear side chains). Each entry is the list of
# side-chain atoms from CB outward; consecutive atoms define a rotatable bond
# (CA–CB, CB–CG, …). Terminal H atoms travel with their heavy-atom parent.
_FLEX_SIDECHAIN_CHAIN = {
    "SER": ["CB", "OG"],
    "CYS": ["CB", "SG"],
    "THR": ["CB", "OG1", "CG2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "MET": ["CB", "CG", "SD", "CE"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "VAL": ["CB", "CG1", "CG2"],
    "PRO": ["CB", "CG", "CD"],
}

# Number of rotatable side-chain bonds (chi angles) treated as flexible. Only
# the linear CA–CB, CB–CG(1), CG–CD, CD–CE/… bonds are rotatable; ring closures
# and terminal-atom pairs (e.g. the two OD of ASP) are NOT independent torsions.
_FLEX_ROTATABLE_BONDS = {
    "SER": 1, "CYS": 1, "THR": 1, "LYS": 4, "ARG": 4, "TYR": 2, "PHE": 2,
    "TRP": 2, "HIS": 2, "ASP": 2, "ASN": 2, "GLU": 3, "GLN": 3, "MET": 3,
    "LEU": 2, "ILE": 2, "VAL": 1, "PRO": 0, "ALA": 0, "GLY": 0,
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
                + " "
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
                            + " "
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


def _fmt_flex_atom(serial, atom_name, res_name, chain_id, res_seq,
                   x, y, z, charge, atom_type):
    """
    Format a single PDBQT ATOM record for a flexible-residue block, using the
    canonical AutoDock/Vina fixed-column layout (matching OpenBabel's ``-xr``
    output) so Vina's column-sensitive parser accepts it:

    cols 1-6 ``ATOM``, 7-11 serial, 13-16 atom name, 18-20 res name,
    22 chain, 23-26 res seq, 31-54 xyz, 55-60 occupancy, 61-66 tempfactor,
    69-76 partial charge, 78-79 AutoDock atom type.
    """
    # Atom-name justification: 4-char names start at col 13; shorter names are
    # padded so the element sits in col 14 (PDB convention), which OpenBabel and
    # AutoDockTools both follow.
    name = atom_name.strip()
    if len(name) >= 4:
        name_field = name[:4]
    else:
        name_field = f" {name:<3s}"
    return (
        "ATOM  "
        + f"{serial:5d} "
        + f"{name_field}"
        + " "
        + f"{res_name[:3]:>3s} "
        + f"{(chain_id or 'A'):1s}"
        + f"{res_seq:4d}"
        + "    "
        + f"{x:8.3f}{y:8.3f}{z:8.3f}"
        + f"{1.00:6.2f}{0.00:6.2f}"
        + "    "
        + f"{charge:+6.3f} "
        + f"{atom_type:<2s}"
    )


def _build_flex_res_block(res_name, chain_id, res_seq, atoms, serial_start):
    """
    Build the torsion-tree PDBQT lines for a single flexible residue.

    ``atoms`` maps atom-name → (atom_type, charge, x, y, z). The backbone
    atoms (N, CA, C, O, …) form the ROOT; each rotatable side-chain bond in
    :data:`_FLEX_SIDECHAIN_CHAIN` opens a nested ``BRANCH … ENDBRANCH`` block.
    Non-standard residues (or ones missing side-chain atoms) fall back to a
    single ROOT block with zero rotatable bonds, which Vina accepts.

    Returns ``(lines, next_serial)`` or ``(None, serial_start)`` when the
    residue has no usable atoms.
    """
    res_name = res_name.strip().upper()
    if not atoms:
        return None, serial_start

    serial = serial_start
    name_to_serial = {}
    lines = [f"BEGIN_RES {res_name} {chain_id or 'A'} {res_seq}"]

    # Separate hydrogens from heavy atoms. Each polar H (added by obabel -h) is
    # attached to its nearest heavy-atom parent so it is emitted in the same
    # torsion-tree branch — Vina infers bonds from intra-branch distance.
    def _is_hydrogen(name):
        t = atoms[name][0].strip().upper()
        return t in ("H", "HD", "HS") or name.strip().upper().startswith("H")

    heavy = {n for n in atoms if not _is_hydrogen(n)}
    hydrogens = [n for n in atoms if _is_hydrogen(n)]

    def _nearest_heavy(hname):
        _, _, hx, hy, hz = atoms[hname]
        best, best_d = None, None
        for hv in heavy:
            _, _, x, y, z = atoms[hv]
            d = (x - hx) ** 2 + (y - hy) ** 2 + (z - hz) ** 2
            if best_d is None or d < best_d:
                best, best_d = hv, d
        return best

    h_of = {}
    for h in hydrogens:
        parent = _nearest_heavy(h)
        if parent is not None:
            h_of.setdefault(parent, []).append(h)

    def emit(atom_name):
        nonlocal serial
        atom_type, charge, x, y, z = atoms[atom_name]
        name_to_serial[atom_name] = serial
        lines.append(_fmt_flex_atom(
            serial, atom_name, res_name, chain_id, res_seq,
            x, y, z, charge, atom_type,
        ))
        serial += 1
        # Emit hydrogens bonded to this heavy atom right after it, in the same
        # branch, so Vina links them to the correct parent.
        for h in h_of.get(atom_name, []):
            atom_type, charge, x, y, z = atoms[h]
            name_to_serial[h] = serial
            lines.append(_fmt_flex_atom(
                serial, h, res_name, chain_id, res_seq,
                x, y, z, charge, atom_type,
            ))
            serial += 1

    chain = _FLEX_SIDECHAIN_CHAIN.get(res_name)
    n_bonds = _FLEX_ROTATABLE_BONDS.get(res_name, 0)

    # ── ROOT: rigid backbone atoms present in this residue ──
    lines.append("ROOT")
    root_names = [n for n in _FLEX_BACKBONE if n in heavy]
    # If the side chain is unknown, treat everything except recognised
    # backbone as rigid too (single ROOT, no torsions).
    if chain is None:
        root_names += [n for n in heavy if n not in _FLEX_BACKBONE]
    for n in root_names:
        emit(n)
    lines.append("ENDROOT")

    torsdof = 0
    if chain is not None:
        # Only the first ``n_bonds`` atoms of the chain start rotatable bonds;
        # atoms beyond that (ring / terminal branches) travel with the last
        # rotatable branch as rigid members.
        present_chain = [c for c in chain if c in heavy]
        # Any heavy side-chain atoms not in the linear chain (e.g. CG2 of
        # THR/ILE, ring atoms) are attached as rigid members. Hydrogens travel
        # with their parent heavy atom via emit() and are excluded here.
        extra = [n for n in heavy
                 if n not in _FLEX_BACKBONE and n not in present_chain]

        def close_branches(depth, opened):
            for parent, child in reversed(opened[:depth]):
                lines.append(
                    f"ENDBRANCH {name_to_serial[parent]:3d} "
                    f"{name_to_serial[child]:3d}"
                )

        opened = []
        prev = "CA" if "CA" in atoms else (root_names[0] if root_names else None)
        made = 0
        for child in present_chain:
            if prev is None:
                emit(child)
                continue
            if made < n_bonds:
                # Open a rotatable branch parent→child.
                # Vina requires the parent atom already emitted (it is, in ROOT
                # or a previous branch).
                lines.append(
                    f"BRANCH {name_to_serial.get(prev, name_to_serial.get('CA', 1)):3d} "
                    f"{serial:3d}"
                )
                opened.append((prev if prev in name_to_serial else "CA", child))
                emit(child)
                torsdof += 1
                made += 1
            else:
                # Beyond the last rotatable bond: rigid member of current branch.
                emit(child)
            prev = child

        # Attach leftover side-chain atoms (branches/rings/Hs) as rigid members
        # inside the innermost open branch (or ROOT if none opened).
        for n in extra:
            emit(n)

        # Close all opened branches (innermost first).
        for parent, child in reversed(opened):
            lines.append(
                f"ENDBRANCH {name_to_serial[parent]:3d} "
                f"{name_to_serial[child]:3d}"
            )

    # NOTE: Flexible-residue blocks must NOT carry a ``TORSDOF`` record — that
    # tag is only valid for ligands. Vina's ``--flex`` parser rejects it with
    # "Unknown or inappropriate tag found in flex residue or ligand". (OpenBabel's
    # ``-xs`` flexible-residue output likewise omits TORSDOF.)
    lines.append(f"END_RES {res_name} {chain_id or 'A'} {res_seq}")
    return lines, serial


def write_flex_pdbqt(
    flex_pdb_path: str,
    flex_pdbqt_path: str,
    flex_residues: Optional[List[str]] = None,
) -> bool:
    """
    Write a Vina-valid *flexible-residue* PDBQT (for ``--flex``) from a PDB
    that contains ONLY the flexible residues.

    Vina's ``--flex`` parser requires each residue to be a *torsion tree*: the
    rigid backbone lives inside ``ROOT``/``ENDROOT`` and every rotatable
    side-chain bond opens a nested ``BRANCH … ENDBRANCH`` block, terminated by
    a ``TORSDOF`` count.  Emitting bare ``ATOM`` records inside
    ``BEGIN_RES``/``END_RES`` (as an earlier version did) makes Vina abort with
    *"Unknown or inappropriate tag found in flex residue"*.  This function
    builds a real torsion tree using the per-residue side-chain topology in
    :data:`_FLEX_SIDECHAIN_CHAIN` / :data:`_FLEX_ROTATABLE_BONDS`.

    Atom types and Gasteiger charges are taken from an OpenBabel PDBQT
    conversion when available; otherwise a Bio.PDB fallback assigns Vina atom
    types with zero charges.  Either way the emitted records form a valid
    torsion tree.

    Args:
        flex_pdb_path: PDB containing only the flexible residues.
        flex_pdbqt_path: Destination flex PDBQT path.
        flex_residues: Optional identifiers (e.g. ``["SER403", …]``) for a
            log message. Not required for parsing.

    Returns:
        ``True`` on success, ``False`` if nothing could be written.
    """
    if not os.path.exists(flex_pdb_path) or os.path.getsize(flex_pdb_path) == 0:
        return False

    # ── Gather per-residue atoms with types/charges ──
    # Prefer obabel-derived atom types + Gasteiger charges; fall back to the
    # element→Vina-type map with zero charges when obabel is unavailable.
    obabel_atoms = {}
    try:
        tmp_pdbqt = flex_pdbqt_path + ".obabel.pdbqt"
        res = subprocess.run(
            ["obabel", flex_pdb_path, "-O", tmp_pdbqt, "-h", "-xs"],
            capture_output=True, timeout=60,
        )
        if res.returncode == 0 and os.path.exists(tmp_pdbqt) and os.path.getsize(tmp_pdbqt) > 0:
            with open(tmp_pdbqt) as fh:
                for line in fh:
                    if not line.startswith(("ATOM", "HETATM")):
                        continue
                    try:
                        atom_name = line[12:16].strip()
                        chain = line[21]
                        res_seq = int(line[22:26])
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        charge = float(line[66:76]) if len(line) > 76 else 0.0
                        atom_type = line[77:79].strip() or "C"
                    except (ValueError, IndexError):
                        continue
                    res_atoms = obabel_atoms.setdefault((chain, res_seq), {})
                    # obabel names every added hydrogen "H"; give each a unique
                    # key (H, H2, H3, …) so the torsion-tree builder can attach
                    # them individually to their nearest heavy-atom parent.
                    key = atom_name
                    if key in res_atoms:
                        idx = 2
                        while f"{atom_name}{idx}" in res_atoms:
                            idx += 1
                        key = f"{atom_name}{idx}"
                    res_atoms[key] = (atom_type, charge, x, y, z)
        try:
            os.remove(tmp_pdbqt)
        except OSError:
            pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        obabel_atoms = {}

    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("flex", flex_pdb_path)

        all_lines = []
        serial = 1
        wrote_any = False
        for model in struct:
            for chain in model:
                cid = chain.get_id()
                for residue in chain:
                    if residue.get_id()[0] != " ":
                        continue
                    res_name = residue.get_resname().strip()
                    res_seq = residue.get_id()[1]

                    # Build atom dict for this residue.
                    ob = obabel_atoms.get((cid, res_seq)) or {}
                    atoms = {}
                    for atom in residue:
                        name = atom.get_name().strip()
                        if not name:
                            continue
                        if name in ob:
                            atoms[name] = ob[name]
                            continue
                        try:
                            coord = atom.get_vector().get_array()
                        except Exception:
                            continue
                        elem = atom.element.strip().upper() if atom.element else ""
                        if not elem:
                            for c in name:
                                if c.isalpha():
                                    elem = c.upper()
                                    break
                        atom_type = _RECEPTOR_PDBQT_ATOM_TYPE.get(elem, elem or "C")
                        atoms[name] = (atom_type, 0.0,
                                       float(coord[0]), float(coord[1]), float(coord[2]))

                    # Add obabel-only atoms (the polar hydrogens obabel added via
                    # ``-h``; they are absent from the H-free crystal PDB that
                    # Bio.PDB parsed). Vina's ``--flex`` needs these explicit
                    # polar Hs (HD type) for hydrogen-bond scoring.
                    for oname, oval in ob.items():
                        if oname not in atoms:
                            atoms[oname] = oval

                    block, serial = _build_flex_res_block(
                        res_name, cid, res_seq, atoms, serial,
                    )
                    if block:
                        all_lines.extend(block)
                        wrote_any = True

        if wrote_any:
            with open(flex_pdbqt_path, "w") as fh:
                fh.write("\n".join(all_lines) + "\n")
            src = "obabel-derived" if obabel_atoms else "Bio.PDB"
            log.info(f"  Flex PDBQT written ({src}, torsion-tree): {flex_pdbqt_path}")
            return True
        return False
    except Exception as exc:
        log.warning(f"  ⚠  Flex PDBQT writer failed: {exc}")
        return False


# Vina's ``--flex`` parser accepts only the following block-level tags. Any
# other tag (notably the ligand-only ``TORSDOF``) makes Vina abort with
# "Unknown or inappropriate tag found in flex residue or ligand". We enumerate
# them here so :func:`validate_flex_pdbqt` can reject files that would crash
# Vina before the (expensive) docking step is launched.
_FLEX_VALID_TAGS = {
    "BEGIN_RES", "END_RES", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "ATOM",
    "HETATM", "REMARK", "TER", "END",
}


def validate_flex_pdbqt(flex_pdbqt_path: str) -> bool:
    """
    Validate that a flexible-residue PDBQT is consumable by Vina's ``--flex``.

    A valid flex PDBQT must:
        * contain at least one ``BEGIN_RES`` / ``END_RES`` pair,
        * carry a ROOT block whose atoms are also linked by BRANCH/ENDBRANCH
          tags (a proper torsion tree),
        * contain NO ``TORSDOF`` record (that tag is ligand-only and Vina
          rejects it inside a flex residue), and
        * use only tags recognised by Vina's flex parser.

    Returns ``True`` when the file is a valid flex PDBQT, ``False`` otherwise.
    Used by the pipeline (and the unit tests) to catch malformed flex files
    early, before Vina timeouts on an invalid input.

    Args:
        flex_pdbqt_path: Path to the candidate flex PDBQT.

    Returns:
        ``True`` if the file passes every structural check, else ``False``.
    """
    if not os.path.exists(flex_pdbqt_path) or os.path.getsize(flex_pdbqt_path) == 0:
        return False
    try:
        with open(flex_pdbqt_path) as fh:
            lines = fh.readlines()
    except OSError:
        return False

    # TORSDOF is a ligand-only tag; its presence inside a flex residue makes
    # Vina abort. Reject immediately.
    if any("TORSDOF" in line for line in lines):
        return False

    has_res = False
    has_root = False
    has_branch = False
    for line in lines:
        stripped = line.strip()
        tag = stripped.split()[0] if stripped else ""
        # Unknown block-level tag → invalid for Vina --flex.
        if tag and tag.isupper() and tag not in _FLEX_VALID_TAGS:
            return False
        if tag == "BEGIN_RES":
            has_res = True
        elif tag == "ROOT":
            has_root = True
        elif tag == "BRANCH":
            has_branch = True

    # A flex file with no residues, or with no rigid backbone ROOT, or with no
    # rotatable BRANCH, is not a usable torsion tree.
    if not (has_res and has_root and has_branch):
        return False
    return True

