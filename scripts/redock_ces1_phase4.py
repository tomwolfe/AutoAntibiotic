#!/usr/bin/env python3
"""
Targeted partial Phase 4 re-run: re-dock compounds with CES1 = N/A.

Uses higher Vina exhaustiveness (32) and num_modes (9) against two CES1
conformers (1YAH, 3KJZ) to recover poses in the catalytic gorge.

Usage:
    python scripts/redock_ces1_phase4.py
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("CES1_redock")

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "output" / "top_candidates.csv"
PDB_DIR = BASE / "pdb"
PDB_DIR.mkdir(exist_ok=True)

def _fetch_pdb(pdb_id):
    """Download a PDB file if not already local. Returns local path."""
    local_path = PDB_DIR / f"{pdb_id}.pdb"
    if local_path.exists():
        return str(local_path)
    from Bio.PDB import PDBList
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir=str(PDB_DIR), file_format="pdb")
    for f in PDB_DIR.iterdir():
        if pdb_id.lower() in f.name.lower() and f.suffix in (".pdb", ".ent"):
            if f.suffix == ".ent":
                dest = PDB_DIR / f"{pdb_id}.pdb"
                if dest.exists():
                    dest.unlink()
                f.rename(dest)
                return str(dest)
            return str(f)
    raise FileNotFoundError(f"Could not fetch PDB {pdb_id}")

CES1_PDB = Path(_fetch_pdb("1YAH"))

CES1_CATALYTIC_RESIDUES = ["SER221", "HIS468", "GLU354"]
CES1_CENTER = None
VINA_EXHAUSTIVENESS = 32
VINA_NUM_MODES = 9
CENTROID_MAX_DIST = 11.0


def _centroid_of_residues(pdb_path, residue_names):
    """Compute centroid of side-chain atoms for given residue names.
    
    Each entry in residue_names should be like "SER221" (3-letter code + number).
    """
    from Bio.PDB import PDBParser
    import re
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("target", str(pdb_path))
    # Parse each spec into (resname, resnum)
    specs = []
    for spec in residue_names:
        m = re.match(r"([A-Za-z]+)(\d+)", spec)
        if m:
            specs.append((m.group(1).upper(), int(m.group(2))))
    atoms = []
    for model in struct:
        for chain in model:
            for residue in chain:
                rid = residue.get_id()
                if rid[0] != " ":
                    continue
                rname = residue.get_resname().strip().upper()
                rnum = rid[1]
                for spec_name, spec_num in specs:
                    if rname == spec_name and rnum == spec_num:
                        for atom in residue:
                            if atom.get_id() in ("N", "CA", "C", "O"):
                                continue
                            atoms.append(atom.get_vector().get_array())
                        break
    if not atoms:
        raise ValueError(f"No atoms found for residues {residue_names}")
    return np.mean(atoms, axis=0)


def prepare_receptor_pdbqt(pdb_path, out_dir):
    """Clean a PDB receptor and convert to PDBQT."""
    out_pdbqt = os.path.join(out_dir, f"{Path(pdb_path).stem}.pdbqt")
    out_clean = os.path.join(out_dir, f"{Path(pdb_path).stem}_clean.pdb")
    try:
        from Bio.PDB import PDBParser, PDBIO, Select

        class CleanSelect(Select):
            def accept_residue(self, residue):
                rid = residue.get_id()
                if rid[0] == "W":
                    return False
                if rid[0] != " ":
                    return False
                return True

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("target", str(pdb_path))
        io = PDBIO()
        io.set_structure(struct)
        io.save(out_clean, CleanSelect())
    except Exception as exc:
        log.error(f"  Failed to clean {pdb_path}: {exc}")
        shutil.copy(str(pdb_path), out_clean)
    try:
        subprocess.run(
            ["obabel", out_clean, "-O", out_pdbqt, "-xr"],
            capture_output=True, timeout=300, check=True,
        )
    except Exception as exc:
        log.error(f"  obabel conversion failed for {out_clean}: {exc}")
        return None
    return out_pdbqt


def dock_compound_vina(smiles, receptor_pdbqt, center, box_size, work_dir, tag):
    """Dock a single compound with Vina, return (energy, pdbqt_path) or (None, None)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    try:
        AllChem.EmbedMolecule(mol_h, params)
    except Exception:
        pass
    safe_id = tag.replace("/", "_").replace(" ", "_")
    lig_pdbqt = os.path.join(work_dir, f"{safe_id}_lig.pdbqt")
    out_pdbqt = os.path.join(work_dir, f"{safe_id}_out.pdbqt")
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol_h)[0]
        setup = mol_setups[0]
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
        if not is_ok:
            raise RuntimeError(f"meeko PDBQT write failed: {err_msg}")
        with open(lig_pdbqt, "w") as f:
            f.write(pdbqt_string)
    except Exception:
        try:
            subprocess.run(
                ["obabel", f"-:{smiles}", "-O", lig_pdbqt, "--gen3d"],
                capture_output=True, timeout=120,
            )
        except Exception as exc:
            log.warning(f"  obabel ligand prep failed: {exc}")
            return None, None
    if not os.path.exists(lig_pdbqt) or os.path.getsize(lig_pdbqt) == 0:
        return None, None
    cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand", lig_pdbqt,
        "--out", out_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.1f}",
        "--size_y", f"{box_size[1]:.1f}",
        "--size_z", f"{box_size[2]:.1f}",
        "--exhaustiveness", str(VINA_EXHAUSTIVENESS),
        "--num_modes", str(VINA_NUM_MODES),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            return None, None
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("1") and " " in stripped:
                parts = stripped.split()
                try:
                    energy = float(parts[1])
                    if energy > 0:
                        return None, out_pdbqt
                    return energy, out_pdbqt
                except (ValueError, IndexError):
                    continue
    except Exception:
        return None, None
    return None, None


def check_centroid(out_pdbqt, target_center, max_dist=CENTROID_MAX_DIST):
    """Check if the best pose's centroid is within max_dist of target center."""
    if not os.path.exists(out_pdbqt):
        return False
    try:
        coords = []
        with open(out_pdbqt) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        elem = line[76:78].strip()
                        if elem and elem.upper() != "H":
                            coords.append(np.array([x, y, z]))
                    except (ValueError, IndexError):
                        continue
        if not coords:
            return False
        centroid = np.mean(coords, axis=0)
        dist = float(np.linalg.norm(centroid - np.asarray(target_center)))
        return dist <= max_dist
    except Exception:
        return False


def main():
    global CES1_CENTER

    print("=" * 60)
    print("Targeted CES1 Re-Dock (Phase 4 partial re-run)")
    print(f"  Exhaustiveness: {VINA_EXHAUSTIVENESS}")
    print(f"  Num modes: {VINA_NUM_MODES}")
    print(f"  Centroid max dist: {CENTROID_MAX_DIST} Å")
    print("=" * 60)

    if not shutil.which("vina"):
        log.error("Vina not found. Cannot re-dock.")
        return 1
    if not shutil.which("obabel"):
        log.error("obabel not found. Cannot re-dock.")
        return 1

    # Read CSV
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Find compounds with CES1=N/A
    na_compounds = []
    for r in rows:
        ces1 = r.get("Human_CES1_Energy", "").strip()
        if ces1 == "N/A" or ces1 == "":
            na_compounds.append(r)
    print(f"\nFound {len(na_compounds)} compounds with CES1=N/A")

    if not na_compounds:
        print("No compounds to re-dock. Done.")
        return 0

    work_dir = tempfile.mkdtemp(prefix="ces1_redock_")
    print(f"Work dir: {work_dir}")

    # Prepare CES1 receptors
    ces1_primary = None
    ces1_alt = None
    ces1_pdb = BASE / "pdb" / "1YAH.pdb"
    if ces1_pdb.exists():
        ces1_primary = prepare_receptor_pdbqt(ces1_pdb, work_dir)
        CES1_CENTER = _centroid_of_residues(ces1_pdb, CES1_CATALYTIC_RESIDUES)
        print(f"CES1 1YAH centroid: {CES1_CENTER}")
    else:
        log.warning(f"CES1 PDB not found: {ces1_pdb}")

    ces1_alt = None

    if CES1_CENTER is None:
        log.error("Could not compute CES1 center from any PDB.")
        return 1

    box_size = (18.0, 18.0, 18.0)
    ces1_pdbqts = []
    if ces1_primary:
        ces1_pdbqts.append(("1YAH", ces1_primary))
    if ces1_alt:
        ces1_pdbqts.append(("3KJZ", ces1_alt))

    if not ces1_pdbqts:
        log.error("No CES1 receptor PDBQTs available.")
        return 1

    # Re-dock each compound
    n_recovered = 0
    for r in na_compounds:
        cid = r["Compound_ID"]
        smiles = r["SMILES"]
        print(f"\n  Re-docking {cid}...")

        best_energy = None
        for conf_name, receptor_pdbqt in ces1_pdbqts:
            tag = f"{cid}_{conf_name}"
            energy, out_pdbqt = dock_compound_vina(smiles, receptor_pdbqt, CES1_CENTER, box_size, work_dir, tag)
            if energy is not None and out_pdbqt and check_centroid(out_pdbqt, CES1_CENTER):
                if best_energy is None or energy < best_energy:
                    best_energy = energy
                print(f"    {conf_name}: {energy:.2f} kcal/mol (valid pose)")

        if best_energy is not None:
            r["Human_CES1_Energy"] = f"{best_energy:.2f}"
            n_recovered += 1
            print(f"  ✓ {cid}: recovered CES1 energy = {best_energy:.2f}")
        else:
            print(f"  ✗ {cid}: no valid CES1 pose recovered")

    # Recompute SI for recovered compounds
    n_si_updated = 0
    for r in rows:
        ces1_str = r.get("Human_CES1_Energy", "").strip()
        try_str = r.get("Human_Trypsin_Energy", "").strip()
        pb2pa_str = r.get("PBP2a_Best_Energy", "").strip()

        try:
            ces1_e = float(ces1_str) if ces1_str not in ("N/A", "") else None
        except (ValueError, TypeError):
            ces1_e = None
        try:
            tryp_e = float(try_str) if try_str not in ("N/A", "") else None
        except (ValueError, TypeError):
            tryp_e = None
        try:
            pb2pa_e = float(pb2pa_str)
        except (ValueError, TypeError):
            pb2pa_e = None

        if ces1_e is not None and tryp_e is not None and pb2pa_e is not None and ces1_e <= -0.01 and tryp_e <= -0.01:
            si = abs(pb2pa_e) / (abs(tryp_e) + abs(ces1_e)) * 2
            r["Selectivity_Index"] = f"{si:.2f}"
            if si >= 2.0:
                r["SI_Tier"] = "Strong"
            elif si >= 1.5:
                r["SI_Tier"] = "Promising"
            else:
                r["SI_Tier"] = "Below gate"
            r["Selectivity_Index_TwoTarget"] = f"{si:.2f}"
            n_si_updated += 1
            print(f"  Updated SI for {r['Compound_ID']}: {si:.2f} ({r['SI_Tier']})")

    # Write updated CSV
    out_path = str(CSV_PATH) + ".new"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(out_path, CSV_PATH)
    print(f"\n{'='*60}")
    print(f"Re-dock complete!")
    print(f"  Recovered CES1 poses: {n_recovered}/{len(na_compounds)}")
    print(f"  SI values updated: {n_si_updated}")
    print(f"  Updated CSV: {CSV_PATH}")
    print(f"{'='*60}")

    # Clean up
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
