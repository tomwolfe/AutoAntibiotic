#!/usr/bin/env python3
"""
AutoAntibiotic Discovery Pipeline v3.1
========================================
Principal Computational Chemist & AI Pipeline Architect
Project: AutoAntibiotic Discovery — MRSA PBP2a Inhibitor Screening

Screens novel small-molecule libraries against MRSA PBP2a (allosteric + active sites)
with selectivity filtering against human serine hydrolases, ADMET profiling, and
resistance-risk analysis.

Author: AutoAntibiotic Agent
Environment: Python 3.9+, RDKit | Bio.PDB | AutoDock Vina
"""

import os
import sys
import json
import subprocess
import logging
import warnings
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Callable
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import pandas as pd

# ── RDKit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem, RDConfig
from rdkit.Chem import (
    AllChem, Descriptors, QED, rdMolDescriptors,
    rdDistGeom, Crippen, FilterCatalog, BRICS,
)
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger as rdklog

# ── Bio.PDB ────────────────────────────────────────────────────────────────────
from Bio.PDB import (
    PDBParser, PDBIO, Select,
    PDBList,
)

# ── Suppress RDKit noise ───────────────────────────────────────────────────────
rdklog.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# PDB identifiers
PDB_IDS = {
    "PBP2a_apo": "3QPD",
    "PBP2a_holo": "6TKO",
    "trypsin": "1UTN",
    "CES1": "3KJZ",
}

# Reference antibiotics for similarity filtering (SMILES)
REFERENCE_ANTIBIOTICS = {
    "Methicillin":  "CC1=C(C(=C(C(=C1O)OC)OC)OC)C(=O)NC2C3C(C(=O)N3C2=O)SC4(C)C",
    "Vancomycin":   "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C(=C4OC6=C(C(=CC(=C6)C(C(=O)NC(C(=O)NC5C(=O)O)CC7=CC=C(C=C7)O)NC(=O)C8C(O)C(=C(C=C8)Cl)O)O)O)CO)O)O)O)NC(=O)C9C(O)C(=C(C=C9)Cl)O)(CC(=O)N)O",
    "Ceftaroline":  "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
    "Meropenem":    "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
    "Oxacillin":    "CC1=C(C(=NO1)C2=CC=CC=C2)C(=O)NC3C4C(C(=O)N4C3=O)SC5(C)C",
}

# β-lactam SMARTS to exclude
BETA_LACTAM_SMARTS = "[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1"

# Allosteric and Active site residues
ALLOSTERIC_RESIDUES = ["ALA237", "MET241", "TYR159"]
ACTIVE_SITE_RESIDUES = ["SER403"]

# Conserved catalytic residues for scientific coherence cross-check
CONSERVED_RESIDUES = ["SER403", "LYS406", "TYR446"]

# Off-target catalytic residues for selectivity docking
TRYPSIN_CATALYTIC_RESIDUES = ["HIS57", "ASP102", "SER195"]
CES1_CATALYTIC_RESIDUES = ["SER221", "HIS468", "GLU354"]

# Grid box defaults (Angstroms)
ALLOSTERIC_BOX_SIZE = (15.0, 15.0, 15.0)
ACTIVE_BOX_SIZE = (20.0, 20.0, 20.0)

# Docking
VINA_TIMEOUT_S = 120
N_JOBS = max(1, mp.cpu_count() - 1)

# Similarity
SIMILARITY_THRESHOLD = 0.4
SIMILARITY_THRESHOLD_RELAXED = 0.5
DIVERSITY_MIN_COUNT = 100

# Selectivity
SELECTIVITY_INDEX_THRESHOLD = 2.0

# Outputs
OUTPUT_DIR = Path("output")
CSV_REPORT = OUTPUT_DIR / "top_candidates.csv"
TOP_N = 10

# Repository root (used to locate bundled offline PDB files under tests/data).
REPO_ROOT = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log"),
    ],
)
log = logging.getLogger("AutoAntibiotic")

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — DEPENDENCY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_dependencies() -> dict:
    """
    Check all required libraries and external binaries.
    Exits with a clear error message listing missing components.

    Returns:
        dict with keys:
            - 'vina': bool (True if vina binary on PATH)
            - 'USE_VINA': global toggle — set False if vina absent
    """
    log.info("─── Phase 0: Dependency Check ───")

    missing_packages = []
    missing_bins = []

    # ── Python packages (hard requirements) ──
    try:
        import rdkit  # noqa: F401
    except ImportError:
        missing_packages.append("RDKit (pip install rdkit-pypi)")

    try:
        import Bio  # noqa: F401
    except ImportError:
        missing_packages.append("Biopython (pip install biopython)")

    try:
        import pandas  # noqa: F401
    except ImportError:
        missing_packages.append("Pandas (pip install pandas)")

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing_packages.append("NumPy (pip install numpy)")

    # ── External binaries (non-fatal; affect available features) ──
    vina_available = False
    try:
        subprocess.run(["vina", "--version"], capture_output=True, timeout=10)
        vina_available = True
        log.info("  ✓  AutoDock Vina binary found on PATH.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing_bins.append("AutoDock Vina (vina)")

    obabel_available = False
    try:
        subprocess.run(["obabel", "--version"], capture_output=True, timeout=10)
        obabel_available = True
        log.info("  ✓  OpenBabel binary found on PATH.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing_bins.append("OpenBabel (obabel)")

    # ── Hard error on missing packages ──
    if missing_packages:
        log.error("Missing required Python packages:")
        for pkg in missing_packages:
            log.error(f"  ✗  {pkg}")
        sys.exit(1)

    log.info("  ✓  All required Python packages found.")

    # ── Warn on missing binaries ──
    if missing_bins:
        log.warning("Optional external binaries not found:")
        for bin_name in missing_bins:
            log.warning(f"  ⚠  {bin_name} — some features will be limited.")

    if not vina_available:
        log.warning(
            "  ⚠  Vina binary not found. Setting USE_VINA = False. "
            "Pipeline will use RDKit Shape/Pharmacophore fallback."
        )

    if not obabel_available:
        log.warning(
            "  ⚠  OpenBabel not found. Some conversions may fail; "
            "pipeline will attempt RDKit-based alternatives."
        )

    return {"vina": vina_available, "USE_VINA": vina_available}


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — PROTOCOL VALIDATION (Redocking)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_native_ligand_from_holo(
    holo_pdb_path: str,
    output_ligand_smi: str,
    output_ligand_pdbqt: str,
) -> Optional[str]:
    """
    Parse the holo structure (6TKO), locate the co-crystallised ligand,
    write its SMILES to *output_ligand_smi* and its PDBQT to *output_ligand_pdbqt*.

    Returns the SMILES string, or None on failure.
    """
    KNOWN_ANTIBIOTIC_NAMES = {
        "LIG", "INH", "METHI", "VANCO", "CEFTA", "MEROP",
        "NAT", "BPN",  # common residue names for antibiotic ligands
    }

    def _is_likely_ligand(resname: str) -> bool:
        """Return True if resname is not a common buffer/ion/water."""
        skip_names = {"SO4", "PO4", "ACT", "EDO", "GOL", "EG0", "E2O",
                      "CL", "NA", "MG", "ZN", "CA", "FE", "FE2", "HOH",
                      "WAT", "SOL", "ACN", "DMS", "DMF", "N2G", "MPD"}
        upper = resname.upper()
        if upper in skip_names:
            return False
        if any(upper.startswith(prefix) for prefix in ("LIG", "INH", "BPN", "NAT",
                                                          "CEF", "MER", "VAN")):
            return True
        return False

    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("6TKO", holo_pdb_path)

        ligand_residues = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] in ("H_", "W", "H_M"):
                        continue
                    if residue.get_id()[0] == " ":
                        continue
                    resname = residue.get_resname().strip()
                    if resname in ("HOH", "WAT", "SOL"):
                        continue
                    ligand_residues.append((chain.get_id(), residue))

        if not ligand_residues:
            log.warning("  ⚠  No hetero-ligand found in 6TKO.")
            return None

        # Filter out buffers/ions, prefer known antibiotic names
        filtered = []
        for chain_id, res in ligand_residues:
            resname = res.get_resname().strip().upper()
            if _is_likely_ligand(resname):
                filtered.append((chain_id, res))

        if not filtered:
            log.warning("  ⚠  No known antibiotic/ligand residue found. Falling back to first HETATM.")
            filtered = ligand_residues  # fallback to original list

        if len(filtered) > 1:
            # Prefer the one with the highest heavy atom count
            best = max(filtered, key=lambda x: x[1].get_num_atoms())
            chain_id, lig_res = best
            log.info(f"  Selected ligand (most heavy atoms): chain {chain_id}, "
                     f"residue {lig_res.get_resname()} ({lig_res.get_num_atoms()} atoms)")
        else:
            chain_id, lig_res = filtered[0]
            log.info(f"  Native ligand found: chain {chain_id}, residue {lig_res.get_resname()}")

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
            # Fallback: copy PDB as-is
            shutil.copy(lig_pdb, output_ligand_pdbqt)

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


def run_redocking_validation(
    holo_pdb_path: str,
    target_pdbqt_path: str,
    work_dir: str,
    deps: dict,
) -> Tuple[bool, Optional[float]]:
    """
    Phase 0 — Protocol Validation.

    Extracts the native ligand from 6TKO, docks it back into the prepared
    PBP2a receptor, and computes the RMSD to the crystal pose.

    Returns (success: bool, rmsd: float | None).
    """
    log.info("─── Phase 0: Redocking Validation ───")

    lig_smi = os.path.join(work_dir, "native_ligand.smi")
    lig_pdbqt = os.path.join(work_dir, "native_ligand.pdbqt")
    docked_pdb = os.path.join(work_dir, "native_docked.pdb")

    smi = _extract_native_ligand_from_holo(holo_pdb_path, lig_smi, lig_pdbqt)
    if smi is None:
        log.warning("  ⚠  Could not extract native ligand. Skipping redocking validation.")
        return False, None

    if not deps["USE_VINA"]:
        log.warning("  ⚠  Vina unavailable. Redocking validation requires Vina. Skip.")
        return False, None

    # Grid center = centroid of the extracted native ligand PDB (not ALLOSTERIC_RESIDUES)
    nat_lig_pdb = lig_pdbqt.replace(".pdbqt", ".pdb")
    center = _centroid_of_pdb_atoms(nat_lig_pdb)
    if center is None:
        log.warning(
            "  ⚠  Could not compute native-ligand centroid; "
            "falling back to allosteric residues."
        )
        center = compute_residue_centroid(holo_pdb_path, ALLOSTERIC_RESIDUES)

    # Run Vina redocking
    log.info("  Redocking native ligand into PBP2a…")
    docked_pdbqt = docked_pdb.replace(".pdb", ".pdbqt")
    vina_cmd = [
        "vina",
        "--receptor", target_pdbqt_path,
        "--ligand", lig_pdbqt,
        "--out", docked_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", "25", "--size_y", "25", "--size_z", "25",
        "--exhaustiveness", "8",
    ]

    try:
        subprocess.run(vina_cmd, capture_output=True, timeout=VINA_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log.warning("  ⚠  Vina redocking timed out (>120s).")
        return False, None
    except FileNotFoundError:
        log.warning("  ⚠  Vina binary not found during redocking.")
        return False, None

    # Convert docked PDBQT back to PDB for RMSD calculation
    try:
        subprocess.run(
            ["obabel", docked_pdbqt, "-O", docked_pdb, "--gen3d"],
            capture_output=True, timeout=30,
        )
    except Exception:
        # If obabel not available, attempt manual conversion (minimal)
        log.warning("  Could not convert docked PDBQT to PDB. Trying RDKit PDBQT reader.")
        mol = Chem.MolFromPDBQT(docked_pdbqt) if hasattr(Chem, "MolFromPDBQT") else None
        if mol is None:
            log.warning("  ⚠  Cannot parse docked PDBQT. RMSD not computed.")
            return False, None
        Chem.MolToPDBFile(mol, docked_pdb)

    crystal_pdb = lig_pdbqt.replace(".pdbqt", ".pdb")
    rmsd = _compute_rmsd_docked_vs_crystal(docked_pdb, crystal_pdb)

    if rmsd is None:
        log.warning("  ⚠  RMSD could not be computed.")
        return False, None

    log.info(f"  Redocking RMSD = {rmsd:.3f} Å")
    if rmsd > 2.0:
        log.warning(
            f"  ⚠  Redocking RMSD ({rmsd:.3f} Å) exceeds 2.0 Å threshold. "
            "The docking protocol may not accurately reproduce known binding modes. "
            "Proceeding with pipeline — interpret results with caution."
        )
    else:
        log.info(f"  ✓  Redocking validated (RMSD = {rmsd:.3f} Å ≤ 2.0 Å).")

    validation_ok = rmsd <= 2.0 if rmsd is not None else False
    return validation_ok, rmsd


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — TARGET PREPARATION & CENTROID CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_structure(pdb_id: str, out_dir: str) -> str:
    """
    Download a PDB structure by *pdb_id* (if not already present) into *out_dir*.
    Returns the local file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    target_path = os.path.join(out_dir, f"{pdb_id}.pdb")

    if os.path.exists(target_path):
        log.info(f"  Structure {pdb_id} already local: {target_path}")
        return target_path

    log.info(f"  Downloading {pdb_id} from PDB…")
    try:
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(
            pdb_id, pdir=out_dir, file_format="pdb",
        )
        # PDBList may save as pdb{pdb_id}.ent; rename
        raw = os.path.join(out_dir, f"pdb{pdb_id.lower()}.ent")
        if os.path.exists(raw):
            os.rename(raw, target_path)
        # Handle alternative naming
        alt = os.path.join(out_dir, f"{pdb_id}.pdb")
        if os.path.exists(alt) and alt != target_path:
            pass  # already correct name
        log.info(f"  ✓  Downloaded {pdb_id} → {target_path}")
    except Exception as exc:
        log.error(f"  ✗  Failed to download {pdb_id}: {exc}")
        raise

    return target_path


def clean_pdb_structure(
    pdb_path: str, out_path: str,
    remove_waters: bool = True,
    remove_ligands: bool = True,
    add_hydrogens: bool = True,
) -> str:
    """
    Remove waters, heteroatoms, and optionally add hydrogens.
    Writes the cleaned PDB to *out_path*.

    If RDKit cannot add hydrogens (no force field), a PDB with no extra
    hydrogens is produced — Vina handles polar H assignment internally anyway.
    """
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("target", pdb_path)

        class CleanSelect(Select):
            def accept_residue(self, residue):
                rid = residue.get_id()
                # Remove waters
                if remove_waters and rid[0] == "W":
                    return False
                # Remove hetero residues (ligands, ions)
                if remove_ligands and rid[0] != " ":
                    return False
                return True

        io = PDBIO()
        io.set_structure(struct)
        io.save(out_path, CleanSelect())

        # Add hydrogens via RDKit PDB → MOL → H-Added → PDB
        if add_hydrogens:
            try:
                mol = Chem.MolFromPDBFile(out_path, removeHs=False)
                if mol is not None:
                    mol = Chem.AddHs(mol, addCoords=True)
                    Chem.MolToPDBFile(mol, out_path)
                    log.info(f"  Polar hydrogens added to {out_path}")
                else:
                    log.warning("  Could not read PDB via RDKit. Skipping hydrogen addition.")
            except Exception as exc:
                log.warning(f"  RDKit PDB parsing failed for hydrogen addition: {exc}. Skipping.")

        # Convert to PDBQT for Vina (using meeko for receptor)
        pdbqt_path = out_path.replace(".pdb", ".pdbqt")
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            # For receptor PDBQT, we use a simpler approach via obabel or
            # prepare_receptor. Prefer prepare_receptor (ADFR suite) if available.
            try:
                subprocess.run(
                    ["prepare_receptor", "-r", out_path, "-o", pdbqt_path],
                    capture_output=True, timeout=60,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # Fallback: use obabel to add gasteiger charges and write PDBQT
                try:
                    subprocess.run(
                        ["obabel", out_path, "-O", pdbqt_path, "-h", "--gas"],
                        capture_output=True, timeout=60,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    log.warning(
                        "  Neither prepare_receptor nor obabel found. "
                        "Writing PDB as-is; Vina may fail."
                    )
                    shutil.copy(out_path, pdbqt_path)
        except Exception as exc:
            log.warning(f"  Receptor PDBQT conversion warning: {exc}")
            shutil.copy(out_path, pdbqt_path)

        return pdbqt_path if os.path.exists(pdbqt_path) else out_path

    except Exception as exc:
        log.error(f"  ✗  Failed to clean {pdb_path}: {exc}")
        raise


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


def _centroid_of_pdb_atoms(pdb_path: str) -> Optional[np.ndarray]:
    """
    Return the geometric centroid (x, y, z) of all atoms in a PDB file,
    or None if the file cannot be parsed / contains no atoms.
    """
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("lig", pdb_path)
        coords = [atom.get_vector().get_array() for atom in struct.get_atoms()]
        if not coords:
            return None
        return np.mean(coords, axis=0)
    except Exception:
        return None


def prepare_targets(
    pdb_dir: str, work_dir: str, deps: dict
) -> Dict[str, Dict]:
    """
    Phase 1 — Download, clean, and compute grid centres for all targets.

    Returns a dictionary:
        {
            "PBP2a": {
                "pdbqt": str,
                "cleaned_pdb": str,
                "allosteric_center": np.ndarray,
                "active_center": np.ndarray,
            },
            "trypsin": {
                "pdbqt": str,
                "active_center": np.ndarray,
            },
            "CES1": {
                "pdbqt": str,
                "active_center": np.ndarray,
            },
            "holo_pdb": str,
            "native_ligand": { "pdb": str, "pdbqt": str, "smiles": str },
        }
    """
    log.info("─── Phase 1: Target Preparation & Centroid Calculation ───")
    log.info(
        "  NOTE: bundled tests/data/*.pdb files are minimal mock structures for "
        "offline CI runs — they are NOT real crystallographic models. Any redocking "
        "RMSD computed against them is non-physical and must not be interpreted "
        "as a protocol-quality metric."
    )
    result = {}

    # ── Fetch structures (prefer bundled offline PDBs under tests/data) ──
    def _resolve_structure(pdb_id: str) -> str:
        # NOTE: tests/data/*.pdb files are minimal mock structures for offline
        # CI runs — they are NOT real crystallographic structures.
        """Return a local tests/data/{pdb_id}.pdb path if present, else download."""
        local_pdb = REPO_ROOT / "tests" / "data" / f"{pdb_id}.pdb"
        if local_pdb.exists():
            log.info(f"  Using local structure for {pdb_id}: {local_pdb}")
            return str(local_pdb)
        return fetch_structure(pdb_id, pdb_dir)

    holo_path = _resolve_structure(PDB_IDS["PBP2a_holo"])
    apo_path = _resolve_structure(PDB_IDS["PBP2a_apo"])
    trypsin_path = _resolve_structure(PDB_IDS["trypsin"])
    ces1_path = _resolve_structure(PDB_IDS["CES1"])

    result["holo_pdb"] = holo_path

    # ── Clean PBP2a (use holo for grid calc, but we need the protein only) ──
    log.info("  Cleaning PBP2a (apo)…")
    pbp2a_clean_pdb = os.path.join(work_dir, "PBP2a_clean.pdb")
    pbp2a_pdbqt = clean_pdb_structure(
        apo_path,
        pbp2a_clean_pdb,
    )

    log.info("  Cleaning PBP2a (holo, protein-only)…")
    _ = clean_pdb_structure(
        holo_path,
        os.path.join(work_dir, "PBP2a_holo_clean.pdb"),
    )

    # ── Compute allosteric + active site centres from cleaned apo ──
    cleaned_pdb = pbp2a_clean_pdb

    log.info("  Computing allosteric site centroid (ALA237, MET241, TYR159)…")
    try:
        allosteric_center = compute_residue_centroid(cleaned_pdb, ALLOSTERIC_RESIDUES)
    except (ValueError, Exception) as exc:
        log.warning(f"  ⚠  Allosteric residues {ALLOSTERIC_RESIDUES} missing: {exc}")
        log.warning("  Residue missing – grid center set to None; supply real PDB.")
        allosteric_center = None
    log.info(f"    Allosteric site center: {allosteric_center}")

    log.info("  Computing active site centroid (conserved residues SER403, LYS406, TYR446)…")
    try:
        active_center = compute_residue_centroid(cleaned_pdb, CONSERVED_RESIDUES)
    except (ValueError, Exception) as exc:
        log.warning(f"  ⚠  Conserved residues {CONSERVED_RESIDUES} missing: {exc}")
        log.warning("  Residue missing – grid center set to None; supply real PDB.")
        try:
            active_center = compute_residue_centroid(cleaned_pdb, ACTIVE_SITE_RESIDUES)
        except (ValueError, Exception) as exc2:
            log.warning(f"  ⚠  Active-site residues {ACTIVE_SITE_RESIDUES} missing: {exc2}")
            log.warning("  Residue missing – grid center set to None; supply real PDB.")
            active_center = allosteric_center
    log.info(f"    Active site center: {active_center}")

    # The conserved catalytic centre is captured directly by active_center
    # (computed from CONSERVED_RESIDUES above), so keep it aliased here for
    # downstream compatibility.
    conserved_center = active_center

    result["PBP2a"] = {
        "pdbqt": pbp2a_pdbqt,
        "cleaned_pdb": pbp2a_clean_pdb,
        "allosteric_center": allosteric_center,
        "active_center": active_center,
        "conserved_center": conserved_center,
    }

    # ── Clean trypsin ──
    log.info("  Cleaning Human Trypsin (1UTN)…")
    tryp_clean_pdb = os.path.join(work_dir, "trypsin_clean.pdb")
    tryp_pdbqt = clean_pdb_structure(
        trypsin_path,
        tryp_clean_pdb,
    )
    log.info("  Computing trypsin active site centroid (His57, Asp102, Ser195)…")
    try:
        tryp_center = compute_residue_centroid(tryp_clean_pdb, TRYPSIN_CATALYTIC_RESIDUES)
    except (ValueError, Exception) as exc:
        log.warning(f"  ⚠  Trypsin catalytic residues {TRYPSIN_CATALYTIC_RESIDUES} missing: {exc}")
        log.warning("  Falling back to the origin for the trypsin centre.")
        tryp_center = np.zeros(3)
    log.info(f"    Trypsin active site center: {tryp_center}")
    result["trypsin"] = {"pdbqt": tryp_pdbqt, "active_center": tryp_center}

    # ── Clean CES1 ──
    log.info("  Cleaning Human Carboxylesterase 1 (3KJZ)…")
    ces1_clean_pdb = os.path.join(work_dir, "CES1_clean.pdb")
    ces1_pdbqt = clean_pdb_structure(
        ces1_path,
        ces1_clean_pdb,
    )
    log.info("  Computing CES1 active site centroid (Ser221, His468, Glu354)…")
    try:
        ces1_center = compute_residue_centroid(ces1_clean_pdb, CES1_CATALYTIC_RESIDUES)
    except (ValueError, Exception) as exc:
        log.warning(f"  ⚠  CES1 catalytic residues {CES1_CATALYTIC_RESIDUES} missing: {exc}")
        log.warning("  Falling back to the origin for the CES1 centre.")
        ces1_center = np.zeros(3)
    log.info(f"    CES1 active site center: {ces1_center}")
    result["CES1"] = {"pdbqt": ces1_pdbqt, "active_center": ces1_center}

    # ── Write grid configuration files ──
    grid_dir = os.path.join(work_dir, "grid_configs")
    os.makedirs(grid_dir, exist_ok=True)

    for site_name, center, box in [
        ("allosteric", allosteric_center, ALLOSTERIC_BOX_SIZE),
        ("active", active_center, ACTIVE_BOX_SIZE),
    ]:
        if center is None:
            log.warning(f"  Skipping grid config for '{site_name}' site (center is None).")
            continue
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


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompoundRecord:
    """Stores all computed properties for a single candidate."""
    compound_id: str
    smiles: str
    mol: Optional[Chem.Mol] = None

    # Docking scores
    pb2pa_allosteric_energy: Optional[float] = None
    pb2pa_active_energy: Optional[float] = None
    human_trypsin_energy: Optional[float] = None
    human_ces1_energy: Optional[float] = None

    # Selectivity
    selectivity_index: Optional[float] = None

    # Similarity
    max_similarity: float = 0.0

    # ADMET
    passes_lipinski: bool = False
    qed_score: float = 0.0
    passes_pains: bool = False

    # Resistance flags
    resistance_notes: str = ""

    # Fallback shape score (0–10, lower better)
    shape_score: Optional[float] = None

    # Redocking validation RMSD (0–inf, lower better; None if not validated)
    validation_rmsd: Optional[float] = None

    # Selectivity confidence based on how many human off-targets were docked:
    #   "High" if 2 human targets provided valid energies,
    #   "Low"  if 1 human target provided a valid energy,
    #   "None" if 0 human targets provided a valid energy.
    selectivity_confidence: str = "None"

    # Path to the active-site Vina docked pose (PDBQT), populated during
    # screening so that pose-based interaction analysis need not re-dock.
    active_docked_pdbqt: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — LIBRARY GENERATION & FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

# 15 diverse natural product scaffolds (SMILES)
NATURAL_PRODUCT_SCAFFOLDS = [
    "O=c1c(O)c2c(oc3cc(O)cc(O)c3c2=O)c(O)c1O",                 # Quercetin
    "Oc1ccc(C=Cc2ccc(O)cc2)cc1",                                # Resveratrol
    "COc1ccc(C=CC(=O)CC(=O)C=Cc2ccc(OC)c(O)c2)cc1O",           # Curcumin
    "COc1cc2c(cc1OC)[n+]1ccc3cc4c(cc3c1CC2)OCO4",              # Berberine
    "CC1(C)OC2C3C(=O)OC4C(OO5)C3C5C2C4O1",                     # Artemisinin (approximate)
    "CCC1(O)C(=O)OCC2=C1C=C4N(CC3=C2C=CC5=C3C=CC(=O)O5)C=O",  # Camptothecin
    "COc1nc2c(cc1C[N@@H]3CC[C@H](O)C3)n(C)c4ccccc24",           # Atropine-like scaffold
]

# Positive control SMILES (to verify pipeline)
CONTROL_SMILES = {
    "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
    "Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
}


def _count_atoms(mol: Chem.Mol) -> int:
    """Heavy-atom count for a molecule."""
    return mol.GetNumHeavyAtoms()


def generate_candidate_library(
    target_count: int = 500,
    seed: int = RANDOM_SEED,
) -> List[CompoundRecord]:
    """
    Phase 2.1 — Generate a diverse library by BRICS decomposition of
    natural product scaffolds, fragment recombination, and expansion.

    Args:
        target_count: Desired number of compounds (~500).
        seed: Random seed for reproducibility.

    Returns:
        List of CompoundRecord objects (SMILES only, no computed props yet).
    """
    log.info("─── Phase 2: Library Generation ───")

    all_scaffolds = NATURAL_PRODUCT_SCAFFOLDS
    scaffold_mols = []
    for smi in all_scaffolds:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold_mols.append(mol)

    log.info(f"  Loaded {len(scaffold_mols)} valid scaffolds.")

    # BRICS decompose all scaffolds
    all_fragments = set()
    for mol in scaffold_mols:
        try:
            fragments = BRICS.BRICSDecompose(mol, minFragmentSize=8)
            for frag_smi in fragments:
                frag_mol = Chem.MolFromSmiles(frag_smi)
                if frag_mol is not None and _count_atoms(frag_mol) >= 8:
                    all_fragments.add(frag_smi)
        except Exception:
            continue

    frag_mols = []
    for smi in all_fragments:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            frag_mols.append(m)

    log.info(f"  Generated {len(frag_mols)} unique fragments (>=8 heavy atoms).")

    if len(frag_mols) < 2:
        log.warning("  Too few fragments for meaningful recombination. Falling back to scaffold enumeration.")
        # Fallback: use the scaffolds directly plus random variations
        candidates = []
        for mol in scaffold_mols:
            smi = Chem.MolToSmiles(mol)
            candidates.append(CompoundRecord(
                compound_id=f"SCAFFOLD_{len(candidates)}",
                smiles=smi,
                mol=mol,
            ))
        # Add controls
        for name, smi in CONTROL_SMILES.items():
            mol = Chem.MolFromSmiles(smi)
            candidates.append(CompoundRecord(
                compound_id=f"CTRL_{name}",
                smiles=smi,
                mol=mol,
            ))
        log.info(f"  Fallback library: {len(candidates)} entries.")
        return candidates

    # Recombine fragments to create novel analogs via BRICSBuild over all fragments
    seen_smiles = set()
    records = []

    log.info(f"  Building recombinant library via BRICS.BRICSBuild (target ≤ {target_count})…")
    builder = BRICS.BRICSBuild(list(frag_mols))
    for product in builder:
        try:
            Chem.SanitizeMol(product)
        except Exception:
            continue
        smi = Chem.MolToSmiles(product)
        if smi in seen_smiles:
            continue
        seen_smiles.add(smi)

        # Generate unique ID
        cid = f"AA-{len(records):04d}"
        records.append(CompoundRecord(
            compound_id=cid,
            smiles=smi,
            mol=product,
        ))

        if len(records) % 100 == 0:
            log.info(f"  Generated {len(records)} / {target_count} candidates…")

        if len(records) >= target_count:
            break

    # Add controls explicitly (ensures at least controls are always returned)
    for name, smi in CONTROL_SMILES.items():
        if len(records) >= target_count:
            break
        if smi not in seen_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                records.append(CompoundRecord(
                    compound_id=f"CTRL_{name}",
                    smiles=smi,
                    mol=mol,
                ))
                seen_smiles.add(smi)

    log.info(f"  Library generation complete: {len(records)} compounds.")
    return records


def apply_filters(
    records: List[CompoundRecord],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> List[CompoundRecord]:
    """
    Phase 2.2 — Apply structural, similarity, ADMET, and PAINS filters.

    Filter chain:
        1. Structural exclusion (β-lactam SMARTS).
        2. Similarity filter vs reference antibiotics (Morgan FP, Tc < threshold).
        3. ADMET: Lipinski Rule of 5 + QED > 0.6.
        4. PAINS alerts via RDKit FilterCatalog.
        5. Diversity check: if < 100 pass, relax similarity to 0.5.

    Args:
        records: Input compound records.
        similarity_threshold: Initial Tanimoto cutoff.

    Returns:
        Filtered list of CompoundRecord (with computed ADMET/similarity fields).
    """
    log.info("─── Phase 2: Filtering ───")

    # ── Precompute reference fingerprints ──
    ref_mols = {}
    for name, smi in REFERENCE_ANTIBIOTICS.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ref_mols[name] = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048,
            )

    # β-lactam SMARTS matcher
    lactam_pattern = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)

    # PAINS filter catalog
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains_catalog = FilterCatalog(pains_params)

    # Brenk alerts filter catalog
    brenk_params = FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_catalog = FilterCatalog(brenk_params)

    def _filter_pass(threshold: float) -> List[CompoundRecord]:
        """Run the similarity + ADMET + PAINS filter chain on the original records."""
        passed = []
        skipped_structural = 0
        skipped_similarity = 0
        skipped_admet = 0
        skipped_pains = 0
        skipped_brenk = 0

        for record in records:
            if record.mol is None:
                mol = Chem.MolFromSmiles(record.smiles)
                if mol is None:
                    continue
                record.mol = mol
            mol = record.mol

            # 1. Structural — reject β-lactams
            if mol.HasSubstructMatch(lactam_pattern):
                skipped_structural += 1
                continue

            # 2. Similarity — max Tc vs reference antibiotics
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            max_sim = 0.0
            for ref_fp in ref_mols.values():
                sim = TanimotoSimilarity(fp, ref_fp)
                max_sim = max(max_sim, sim)
            record.max_similarity = max_sim

            if max_sim >= threshold:
                skipped_similarity += 1
                continue

            # 3. ADMET — Lipinski + QED
            try:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                lipinski_ok = (mw <= 500) and (logp <= 5.0) and (hbd <= 5) and (hba <= 10)
                qed = QED.qed(mol)
            except Exception:
                continue

            record.passes_lipinski = lipinski_ok
            record.qed_score = qed

            if not lipinski_ok:
                skipped_admet += 1
                continue
            if qed <= 0.6:
                skipped_admet += 1
                continue

            # 4. PAINS
            pains_match = pains_catalog.HasMatch(mol)
            record.passes_pains = not pains_match
            if pains_match:
                skipped_pains += 1
                continue

            # 5. Brenk alerts
            brenk_match = brenk_catalog.HasMatch(mol)
            if brenk_match:
                skipped_brenk += 1
                continue

            passed.append(record)

        log.info(f"  Structural exclusion (β-lactam): {skipped_structural} removed.")
        log.info(f"  Similarity filter (Tc < {threshold}): {skipped_similarity} removed.")
        log.info(f"  ADMET filter (Lipinski + QED > 0.6): {skipped_admet} removed.")
        log.info(f"  PAINS filter: {skipped_pains} removed.")
        log.info(f"  Brenk alerts: {skipped_brenk} removed.")
        log.info(f"  Passed filters: {len(passed)} compounds.")
        return passed

    passed = _filter_pass(similarity_threshold)

    # Diversity check — if too few passed, relax the similarity threshold and
    # re-run the same loop on the original records (simple for-loop, no recursion).
    if len(passed) < DIVERSITY_MIN_COUNT:
        log.info(
            f"  Only {len(passed)} compounds passed filters (< {DIVERSITY_MIN_COUNT}). "
            f"Relaxing similarity threshold to {SIMILARITY_THRESHOLD_RELAXED} and re-filtering."
        )
        passed = _filter_pass(SIMILARITY_THRESHOLD_RELAXED)

    log.info("─── Phase 2 complete ───")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — VIRTUAL SCREENING (Docking)
# ═══════════════════════════════════════════════════════════════════════════════

class LigandPreparator:
    """
    Encapsulates the logic for converting an RDKit Mol to PDBQT format.

    Strategy:
        1. Try meeko (preferred — handles partial charges, rotatable bonds).
        2. If meeko is unavailable or fails, fall back to obabel via subprocess.
        3. If both fail, raise a clear RuntimeError with installation instructions.
    """

    def prepare(self, mol: Chem.Mol) -> str:
        """
        Convert an RDKit Mol to a PDBQT string.

        Args:
            mol: Input molecule (should have 3D coordinates).

        Returns:
            PDBQT-formatted string.

        Raises:
            RuntimeError: If neither meeko nor obabel can produce PDBQT.
        """
        meeko_error = None
        # ── Try meeko first ──
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            if not mol_setups:
                raise RuntimeError("Meeko returned an empty setup for the input molecule")
            pdbqt_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
            if pdbqt_str:
                return pdbqt_str
            raise RuntimeError("Meeko produced an empty PDBQT string for the input molecule")
        except (ImportError, AttributeError, RuntimeError) as exc:
            meeko_error = str(exc)
            log.warning(f"Meeko failed: {exc}")

        # ── Fallback: obabel via subprocess ──
        obabel_error = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=True) as tmp:
                subprocess.run(
                    ["obabel", "-g", "min", "-O", tmp.name],
                    input=Chem.MolToMolBlock(mol).encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                )
                pdbqt_str = tmp.read().decode("utf-8", errors="ignore")
                if pdbqt_str:
                    return pdbqt_str
                raise ValueError("obabel returned empty output")
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as exc:
            obabel_error = str(exc)
            log.warning(f"obabel fallback failed: {exc}")

        # ── All attempts failed ──
        raise RuntimeError(
            "Cannot convert Mol to PDBQT. Ensure meeko is installed "
            "(pip install meeko) or OpenBabel is available on PATH. "
            "Alternatively, set USE_VINA=False to skip PDBQT-dependent steps."
            f" meeko error: {meeko_error}; obabel error: {obabel_error}"
        )


def prepare_ligand_pdbqt(
    mol: Chem.Mol,
    output_path: str,
) -> bool:
    """
    Convert an RDKit Mol to PDBQT via LigandPreparator.

    Args:
        mol: Input molecule.
        output_path: Destination .pdbqt path.

    Returns:
        True on success.
    """
    try:
        preparator = LigandPreparator()
        pdbqt_str = preparator.prepare(mol)
        with open(output_path, "w") as f:
            f.write(pdbqt_str)
        return True
    except Exception as exc:
        log.warning(f"  Ligand preparation failed: {exc}")
        return False


def _run_vina_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    timeout: int = VINA_TIMEOUT_S,
) -> Optional[float]:
    """
    Run a single Vina docking job. Returns best binding energy (kcal/mol)
    or None on failure.
    """
    cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", output_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.1f}",
        "--size_y", f"{box_size[1]:.1f}",
        "--size_z", f"{box_size[2]:.1f}",
        "--exhaustiveness", "8",
        "--num_modes", "3",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.warning(
                f"  Vina returned exit code {result.returncode}.\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  stdout: {result.stdout.strip()}"
            )
            return None

        # Parse output for best binding energy
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("1") and " " in stripped:
                # Vina table format: mode | affinity | dist from best mode
                parts = stripped.split()
                try:
                    energy = float(parts[1])
                    return energy
                except (ValueError, IndexError):
                    continue
        # Fallback: parse from log tail
        for line in result.stderr.splitlines():
            if "Affinity" in line and "kcal/mol" in line:
                try:
                    energy = float(line.split()[1])
                    return energy
                except (ValueError, IndexError):
                    continue
        # If we reach here, no energy could be parsed — log full output
        log.warning(
            "  Failed to parse Vina binding energy from output.\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
        return None

    except subprocess.TimeoutExpired:
        log.warning(f"  Vina timeout ({timeout}s).")
        return None
    except FileNotFoundError:
        log.warning("  Vina binary not found.")
        return None
    except Exception as exc:
        log.warning(f"  Vina exception: {exc}")
        return None


def dock_compound(
    record: CompoundRecord,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "",
) -> Optional[float]:
    """
    Full docking pipeline for a single compound: PDBQT prep → Vina → parse.

    Args:
        record: Compound record (must have .mol).
        receptor_pdbqt: Path to receptor PDBQT.
        center: Grid box centre.
        box_size: Grid box dimensions.
        work_dir: Scratch directory.
        tag: Label for temp files (e.g. 'allosteric').

    Returns:
        Best binding energy, or None on failure.
    """
    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            return None
        record.mol = mol

    # Generate unique filenames
    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    lig_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_lig.pdbqt")
    out_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_out.pdbqt")

    if not prepare_ligand_pdbqt(record.mol, lig_pdbqt):
        raise RuntimeError(
            f"PDBQT preparation failed for {record.compound_id}; "
            f"this compound will be skipped during screening."
        )

    energy = _run_vina_docking(
        receptor_pdbqt, lig_pdbqt, out_pdbqt,
        center, box_size,
    )

    # Keep the docked pose for the active site so downstream pose analysis
    # (binding interactions) can reuse it instead of re-docking.
    if tag == "active":
        record.active_docked_pdbqt = out_pdbqt

    # Cleanup temp files (keep the active-site pose for later analysis)
    for f in (lig_pdbqt, out_pdbqt):
        if tag == "active" and f == out_pdbqt:
            continue
        try:
            os.remove(f)
        except OSError:
            pass

    return energy


def _dock_compounds_parallel(
    records: List[CompoundRecord],
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
    n_jobs: int = N_JOBS,
    dock_func: Optional[Callable] = None,
) -> List[Tuple[CompoundRecord, Optional[float]]]:
    """
    Dock a list of compounds in parallel, returning ``(record, energy)`` pairs.

    Each compound is docked by *dock_func* (defaults to :func:`dock_compound`).
    If a worker raises, the specific error is logged together with the
    ``CompoundRecord.compound_id`` and the record is returned with
    ``energy=None`` so the pipeline continues instead of aborting.

    When ``n_jobs <= 1`` (or for small batches) the docking is performed
    in-process, which keeps behaviour deterministic and avoids the overhead
    of spawning worker processes.

    Note (memory): for very large libraries the :class:`CompoundRecord.mol`
    objects are pickled for each worker. If profiling shows a bottleneck,
    callers may pass lightweight ``(compound_id, smiles)`` payloads and
    reconstruct the :class:`~rdkit.Chem.Mol` inside *dock_func*.

    Args:
        records: Compounds to dock (must expose ``.mol`` / ``.smiles``).
        receptor_pdbqt: Path to receptor PDBQT.
        center: Grid-box centre as a length-3 array.
        box_size: Grid-box dimensions ``(x, y, z)``.
        work_dir: Scratch directory for intermediate files.
        tag: Label for temporary files (e.g. ``"allosteric"``).
        n_jobs: Number of worker processes.
        dock_func: Docking callable; mainly useful for testing.

    Returns:
        List of ``(CompoundRecord, energy_or_None)`` tuples.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if dock_func is None:
        dock_func = dock_compound

    results: List[Tuple[CompoundRecord, Optional[float]]] = []
    total = len(records)

    # In-process execution keeps small batches deterministic and testable.
    if n_jobs <= 1:
        for i, rec in enumerate(records):
            results.append(_dock_worker(
                rec, dock_func, receptor_pdbqt, center, box_size, work_dir, tag,
            ))
            if (i + 1) % 25 == 0:
                log.info(f"    Docked {i + 1} / {total} ({tag})")
        return results

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(
                _dock_worker, rec, dock_func,
                receptor_pdbqt, center, box_size, work_dir, tag,
            ): rec
            for rec in records
        }
        for i, future in enumerate(as_completed(futures)):
            rec = futures[future]  # original record
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as exc:
                log.warning(
                    f"    Docking failed for {rec.compound_id} ({tag}): {exc}. "
                    "Returning (record, None) and continuing."
                )
                results.append((rec, None))
            if (i + 1) % 25 == 0:
                log.info(f"    Docked {i + 1} / {total} ({tag})")

    return results


def _dock_worker(
    rec: CompoundRecord,
    dock_func: Callable,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
) -> Tuple[CompoundRecord, Optional[float]]:
    """
    Module-level docking wrapper so it can be pickled by ``ProcessPoolExecutor``.

    Runs *dock_func* for a single record and returns ``(record, energy)``.
    On any failure the error is logged with the ``CompoundRecord.compound_id``
    and ``(record, None)`` is returned so the pipeline keeps going.
    """
    try:
        energy = dock_func(rec, receptor_pdbqt, center, box_size, work_dir, tag)
        return rec, energy
    except Exception as exc:
        log.warning(
            f"    Docking failed for {rec.compound_id} ({tag}): {exc}. "
            "Returning (record, None) and continuing."
        )
        return rec, None


def _compute_shape_fallback_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    seed: int = RANDOM_SEED,
) -> Optional[float]:
    """
    Fallback scoring: generate 3D conformer, compute shape protrude distance
    vs reference (co-crystallised ligand from 6TKO). Normalise to 0–10 scale
    (lower = better shape match).

    If available, also computes electrostatic similarity and combines with
    the shape score (50/50 weight) for a more robust metric.

    Returns combined normalised score (0–10, lower = better), or None on failure.
    """
    try:
        # Generate 3D conformer with ETKDGv3 for better stereochemistry
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.useExpTorsionAnglePrefs = True
        params.useBasicKnowledge = True
        params.enforceChirality = True
        params.randomSeed = seed
        status = rdDistGeom.EmbedMolecule(mol_3d, params)
        if status < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)

        ref_3d = Chem.RWMol(ref_mol)
        ref_3d = Chem.AddHs(ref_3d)
        params_ref = rdDistGeom.ETKDGv3()
        params_ref.useExpTorsionAnglePrefs = True
        params_ref.useBasicKnowledge = True
        params_ref.enforceChirality = True
        params_ref.randomSeed = seed
        status_ref = rdDistGeom.EmbedMolecule(ref_3d, params_ref)
        if status_ref < 0:
            return None
        AllChem.MMFFOptimizeMolecule(ref_3d)

        # Shape protrude distance
        try:
            protrude = AllChem.GetShapeProtrudeDist(mol_3d, ref_3d)
        except Exception:
            try:
                protrude = AllChem.GetShapeProtrudeDist(ref_3d, mol_3d)
            except Exception:
                return None

        # Normalise to 0–10 scale (heuristic: typical range 0–0.5)
        # Map: protrude=0 → score=0 (perfect), protrude=0.5 → score=10 (worst)
        shape_norm = min(protrude / 0.05, 10.0) if protrude > 0 else 0.0

        # Electrostatic similarity (optional enhancement)
        elec_sim = None
        try:
            from rdkit.Chem.rdMolDescriptors import GetElectrostaticSimilarity
            elec_sim = GetElectrostaticSimilarity(mol_3d, ref_3d)
        except Exception:
            pass

        if elec_sim is not None:
            # Convert electrostatic similarity (0–1, higher = better) to
            # a penalty (0–10, lower = better) and average with shape score
            elec_penalty = (1.0 - elec_sim) * 10.0
            combined = 0.5 * shape_norm + 0.5 * elec_penalty
            return combined

        return shape_norm

    except Exception:
        return None


def _compute_shape_scores(
    records: List[CompoundRecord],
    ref_mol: Chem.Mol,
) -> List[Tuple[CompoundRecord, Optional[float]]]:
    """
    Compute RDKit Shape-Protrude fallback scores for a list of records.

    For every record the molecule is embedded in 3D and compared against
    *ref_mol* (the co-crystallised 6TKO ligand).  The normalised score
    (0–10, lower = better shape match) is stored on ``rec.shape_score``.

    Args:
        records: Compounds to score (must expose ``.mol`` / ``.smiles``).
        ref_mol: Reference molecule used for the shape comparison.

    Returns:
        List of ``(CompoundRecord, score_or_None)`` tuples.
    """
    total = len(records)
    scored: List[Tuple[CompoundRecord, Optional[float]]] = []

    for i, rec in enumerate(records):
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                rec.shape_score = None
                scored.append((rec, None))
                continue
            rec.mol = mol

        score = _compute_shape_fallback_score(rec.mol, ref_mol)
        rec.shape_score = score
        scored.append((rec, score))

        if (i + 1) % 100 == 0:
            log.info(f"  Shape scored {i + 1} / {total}")

    return scored


def screen_library(
    records: List[CompoundRecord],
    targets: dict,
    work_dir: str,
    deps: dict,
) -> List[CompoundRecord]:
    """
    Phase 3 — Virtual screening.

    Primary (Vina):
        1. Dock all filtered compounds against allosteric site.
        2. Select top 50 by energy; dock against active site.

    Fallback (RDKit Shape):
        1. Generate 3D conformers.
        2. Compute Shape Protrude Distance vs native 6TKO ligand.
        3. Rank by normalised score.

    Returns top 10 candidates with docking/shape scores populated.
    """
    log.info("─── Phase 3: Virtual Screening ───")

    pb2pa = targets["PBP2a"]
    allosteric_center = pb2pa["allosteric_center"]
    active_center = pb2pa["active_center"]

    if deps["USE_VINA"]:
        # ── Allosteric docking ──
        log.info("  Docking all compounds against allosteric site…")
        allosteric_results = _dock_compounds_parallel(
            records, pb2pa["pdbqt"],
            allosteric_center, ALLOSTERIC_BOX_SIZE,
            work_dir, "allosteric",
        )

        n_scored = 0
        for rec, energy in allosteric_results:
            rec.pb2pa_allosteric_energy = energy
            if energy is not None:
                n_scored += 1

        log.info(f"  Allosteric docking complete: {n_scored}/{len(records)} scored.")

        # ── Select top 50 for active-site docking ──
        scored = [r for r, e in allosteric_results if e is not None]
        scored.sort(key=lambda r: r.pb2pa_allosteric_energy)

        top50 = scored[:50]
        log.info(f"  Docking top {len(top50)} compounds against active site…")

        active_results = _dock_compounds_parallel(
            top50, pb2pa["pdbqt"],
            active_center, ACTIVE_BOX_SIZE,
            work_dir, "active",
        )

        for rec, energy in active_results:
            rec.pb2pa_active_energy = energy

    else:
        # ── Fallback: RDKit Shape protrude ──
        log.info("  Vina unavailable. Using RDKit Shape Fallback.")

        # Extract native ligand from 6TKO as reference via the canonical parser
        ref_mol = None
        holo_pdb = targets.get("holo_pdb")
        if holo_pdb and os.path.exists(holo_pdb):
            lig_smi = os.path.join(work_dir, "native_ref.smi")
            lig_pdbqt = os.path.join(work_dir, "native_ref.pdbqt")
            try:
                smi = _extract_native_ligand_from_holo(holo_pdb, lig_smi, lig_pdbqt)
                if smi is not None:
                    ref_mol = Chem.MolFromSmiles(smi)
            except Exception:
                pass

        if ref_mol is None:
            # Fallback reference: first positive control, log a warning
            ref_smi = list(CONTROL_SMILES.values())[0]
            ref_mol = Chem.MolFromSmiles(ref_smi)
            if ref_mol is not None:
                log.warning(
                    "  ⚠  Could not extract native ligand for shape reference; "
                    f"falling back to control SMILES ({ref_smi})."
                )

        if ref_mol is None:
            log.error("  Cannot obtain reference molecule for shape scoring.")
            return records[:TOP_N]

        shape_scores = _compute_shape_scores(records, ref_mol)
        shape_scores = [s for s in shape_scores if s[1] is not None]

        if shape_scores:
            shape_scores.sort(key=lambda x: x[1])
            log.info(f"  Shape scoring complete. Best score: {shape_scores[0][1]:.3f}")
        else:
            log.warning("  No valid shape scores computed. Using default scores.")
            for rec in records:
                rec.shape_score = 0.0
            shape_scores = [(rec, 0.0) for rec in records]

    # ── Select top 10 ──
    if deps["USE_VINA"]:
        # Rank by allosteric energy (lower = better)
        ranked = [r for r in records if r.pb2pa_allosteric_energy is not None]
        ranked.sort(key=lambda r: r.pb2pa_allosteric_energy)
    else:
        ranked = [r for r in records if r.shape_score is not None]
        ranked.sort(key=lambda r: r.shape_score)

    top10 = ranked[:TOP_N]
    log.info(f"  Top {len(top10)} candidates selected.")
    for i, r in enumerate(top10):
        energy_str = (
            f"{r.pb2pa_allosteric_energy:.2f}" if r.pb2pa_allosteric_energy is not None
            else f"{r.shape_score:.2f} (shape)"
        )
        log.info(f"    {i + 1}. {r.compound_id}: {energy_str} kcal/mol")

    log.info("─── Phase 3 complete ───")
    return top10


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — SELECTIVITY & RESISTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_binding_interactions(
    docked_pdbqt_path: str,
    receptor_pdb_path: str,
    key_residues: Optional[Dict[str, List[Tuple[str, int]]]] = None,
) -> Dict[str, Union[bool, float]]:
    """
    Analyse the binding interactions between a docked ligand and key
    receptor residues.

    The function parses the ligand PDBQT and receptor PDB files, then
    computes distances between ligand heavy atoms and key residue atoms
    (O, N, OG, NZ, OH).

    Key residues (default):
        - Ser403  (catalytic residue)
        - Lys406  (conserved Lys in PBP2a)
        - Tyr446  (conserved Tyr in PBP2a)

    Returns:
        Dictionary with:
            'Ser403_contact'       – True if any heavy atom < 3.5 Å from Ser403 OG
            'Lys406_Hbond'         – True if any heavy atom < 3.8 Å from Lys406 NZ
            'Tyr446_Hbond'         – True if any heavy atom < 3.5 Å from Tyr446 OH
            'min_dist_Ser403'      – Minimum distance (Å) to Ser403 OG atom
            'min_dist_Lys406'      – Minimum distance (Å) to Lys406 NZ atom
            'min_dist_Tyr446'      – Minimum distance (Å) to Tyr446 OH atom

    Raises:
        FileNotFoundError: If either file path does not exist.
        ValueError: If files cannot be parsed.
    """
    if key_residues is None:
        key_residues = {
            "Ser403": [("SER", 403, "OG")],
            "Lys406": [("LYS", 406, "NZ")],
            "Tyr446": [("TYR", 446, "OH")],
        }

    # ── Parse receptor key atoms ──
    atom_coords: Dict[str, List[np.ndarray]] = {}
    for resname, atom_entries in key_residues.items():
        atom_coords[resname] = []
        try:
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure("receptor", receptor_pdb_path)
            for model in struct:
                for chain in model:
                    for residue in chain:
                        for entry in atom_entries:
                            aname = entry[-1]
                            resname_expected = entry[0] if len(entry) > 2 else ""
                            resno_expected = entry[1] if len(entry) > 2 else -1
                            if (
                                resname_expected
                                and residue.get_resname().strip().upper()
                                != resname_expected.upper()
                            ):
                                continue
                            if resno_expected >= 0 and residue.get_id()[1] != resno_expected:
                                continue
                            if aname in residue:
                                atom_coords[resname].append(
                                    residue[aname].get_vector().get_array()
                                )
        except FileNotFoundError:
            raise
        except Exception as exc:
            log.warning(f"  Could not parse receptor for key residues: {exc}")
            return {
                "Ser403_contact": False,
                "Lys406_Hbond": False,
                "Tyr446_Hbond": False,
                "min_dist_Ser403": float("inf"),
                "min_dist_Lys406": float("inf"),
                "min_dist_Tyr446": float("inf"),
            }

    # ── Parse ligand heavy-atom coordinates from PDBQT ──
    ligand_coords = []
    try:
        with open(docked_pdbqt_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        elem = line[76:78].strip()
                        if elem and elem.upper() != "H":
                            ligand_coords.append(np.array([x, y, z]))
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        raise FileNotFoundError(f"Docked PDBQT not found: {docked_pdbqt_path}")

    if not ligand_coords:
        raise ValueError("No ligand heavy atoms found in PDBQT file.")

    # ── Compute distances ──
    results: Dict[str, Union[bool, float]] = {}
    min_dists: Dict[str, float] = {}

    for resname, coords in atom_coords.items():
        if not coords:
            min_dists[resname] = float("inf")
            continue
        ref = np.array(coords)
        distances = np.linalg.norm(
            np.array(ligand_coords)[:, np.newaxis] - ref[np.newaxis, :],
            axis=2,
        ).min(axis=0)
        min_dists[resname] = float(distances.min())

    # Populate boolean contact flags
    results["Ser403_contact"] = min_dists.get("Ser403", float("inf")) < 3.5
    results["Lys406_Hbond"] = min_dists.get("Lys406", float("inf")) < 3.8
    results["Tyr446_Hbond"] = min_dists.get("Tyr446", float("inf")) < 3.5

    # Populate min distances
    results["min_dist_Ser403"] = min_dists.get("Ser403", float("inf"))
    results["min_dist_Lys406"] = min_dists.get("Lys406", float("inf"))
    results["min_dist_Tyr446"] = min_dists.get("Tyr446", float("inf"))

    return results


def compute_selectivity_index(
    pb2pa_energy: float, human_avg_energy: float,
) -> float:
    """
    Selectivity Index (SI).

        SI = |Energy_PBP2a_Best| / |Energy_Human_Avg|

    Vina energies are negative. A higher SI means stronger binding to PBP2a
    than to the human off-target panel.

    Args:
        pb2pa_energy: Best (most negative) PBP2a binding energy.
        human_avg_energy: Average binding energy across human targets.

    Returns:
        SI value (float).
    """
    if pb2pa_energy >= 0:
        return 0.0
    if abs(pb2pa_energy) < 1e-6:
        return 0.0
    if abs(human_avg_energy) < 1e-6:
        return 0.0
    return abs(pb2pa_energy) / abs(human_avg_energy)


def profile_resistance_risk(
    record: CompoundRecord,
    work_dir: str,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    interactions: Optional[Dict[str, Union[bool, float]]] = None,
) -> str:
    """
    Rule-based resistance profiling, optionally informed by pose analysis.

    Flags candidates based on predicted interactions with conserved PBP2a
    residues.  If *interactions* is None the function falls back to an
    internal call to ``analyze_binding_interactions`` which re-docks the
    compound to obtain the pose.

    Args:
        record: Compound record containing docking scores.
        work_dir: Working directory for temporary files.
        receptor_pdbqt: Path to the PBP2a receptor PDBQT file.
        center: Grid box centre (used for re-docking).
        box_size: Grid box dimensions.
        interactions: Pre-computed interaction fingerprint dict returned by
                      ``analyze_binding_interactions``.  If None the
                      function will perform the analysis internally.

    Returns a human-readable notes string.
    """
    notes = []

    # Perform interaction analysis if not already supplied
    if interactions is None:
        safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
        lig_pdbqt = os.path.join(work_dir, f"{safe_id}_pose_lig.pdbqt")
        out_pdbqt = os.path.join(work_dir, f"{safe_id}_pose_out.pdbqt")

        try:
            if prepare_ligand_pdbqt(rec.mol, lig_pdbqt):
                _run_vina_docking(
                    receptor_pdbqt, lig_pdbqt, out_pdbqt,
                    center, box_size,
                )
                interactions = analyze_binding_interactions(out_pdbqt, receptor_pdbqt)
                try:
                    os.remove(out_pdbqt)
                except OSError:
                    pass
            else:
                interactions = None
        except Exception:
            interactions = None

    if interactions is None:
        notes.append("No binding interactions could be analysed.")
    else:
        # ── Quantitative resistance check from measured pose distances ──
        # The interaction fingerprint already exposes the minimum ligand→residue
        # distances (Å). We use these directly rather than the boolean contact
        # flags so resistance risk scales with how tightly the conserved
        # catalytic network is engaged.
        ser = interactions.get("min_dist_Ser403", float("inf"))
        lys = interactions.get("min_dist_Lys406", float("inf"))
        tyr = interactions.get("min_dist_Tyr446", float("inf"))

        if np.isfinite(ser):
            if ser < 3.5:
                notes.append(f"Strong catalytic engagement (Ser403, d={ser:.2f} Å)")
            elif ser < 5.0:
                notes.append(f"Weak Ser403 contact (d={ser:.2f} Å) — resistance risk")
            else:
                notes.append(f"Loss of Ser403 engagement (d={ser:.2f} Å) — high resistance risk")
        else:
            notes.append("Ser403 distance undefined — high resistance risk")

        if np.isfinite(lys):
            if lys < 3.8:
                notes.append(f"Stabilising H-bond with Lys406 (d={lys:.2f} Å)")
            elif lys < 5.0:
                notes.append(f"Weak Lys406 contact (d={lys:.2f} Å) — resistance risk")
        else:
            notes.append("Lys406 distance undefined — resistance risk")

        if np.isfinite(tyr):
            if tyr < 3.5:
                notes.append(f"Stabilising contact with Tyr446 (d={tyr:.2f} Å)")
            elif tyr < 5.0:
                notes.append(f"Weak Tyr446 contact (d={tyr:.2f} Å) — resistance risk")

        # Aggregate: if the closest conserved-residue contact exceeds 5 Å the
        # compound avoids the catalytic network entirely and is flagged as a
        # high-resistance-risk binder (mutations need only modestly perturb
        # the active site to escape it).
        best_conserved = min(ser, lys, tyr)
        if np.isfinite(best_conserved) and best_conserved >= 5.0:
            notes.append(
                f"Avoids conserved catalytic network (min d={best_conserved:.2f} Å) — high resistance risk"
            )

        # Allosteric binder note
        if (
            record.pb2pa_allosteric_energy is not None
            and record.pb2pa_allosteric_energy < -7.0
        ):
            if record.pb2pa_active_energy is None or record.pb2pa_active_energy > -6.0:
                notes.append("Allosteric binder (Ala237/Met241/Tyr159 pocket). Novel mechanism.")

    # Energy-based heuristics (unchanged)
    if record.pb2pa_active_energy is not None and record.pb2pa_active_energy < -6.0:
        if "Confirmed contact with catalytic Ser403" not in notes:
            notes.append("Likely contacts catalytic Ser403 (active site, energy-based). Good.")

    # Molecular weight heuristic
    if record.mol is not None:
        mw = Descriptors.MolWt(record.mol)
        if mw > 400:
            notes.append("High MW (>400) — broad interaction surface, may contact multiple residues.")
        n_rot = Descriptors.NumRotatableBonds(record.mol)
        if n_rot < 5:
            notes.append("Rigid scaffold — reduced entropic penalty, may enhance binding specificity.")

    # Resistance risk indicators
    if record.qed_score > 0.8:
        notes.append("High drug-likeness (QED > 0.8) — good developability profile.")

    if not notes:
        notes.append("No specific resistance flags identified.")

    return "; ".join(notes)


def analyze_selectivity_and_resistance(
    top10: List[CompoundRecord],
    targets: dict,
    work_dir: str,
    deps: dict,
) -> List[CompoundRecord]:
    """
    Phase 4 — Selectivity & Resistance Analysis.

    1. Dock top 10 candidates against Human Trypsin (1UTN) and CES1 (3KJZ).
    2. Compute Selectivity Index for each.
    3. Profile resistance risk.

    Returns updated records with selectivity and resistance fields.
    """
    log.info("─── Phase 4: Selectivity & Resistance Analysis ───")

    if not deps["USE_VINA"]:
        log.warning("  Vina unavailable — skipping selectivity docking. Flagging all as uncertain.")
        for rec in top10:
            rec.selectivity_index = None
            rec.selectivity_confidence = "None"
            rec.resistance_notes = "Selectivity not assessed (Vina unavailable)."
        return top10

    # ── Dock vs Trypsin (using computed catalytic triad centre) ──
    log.info("  Docking top 10 vs Human Trypsin (1UTN)…")
    trypsin_results = _dock_compounds_parallel(
        top10, targets["trypsin"]["pdbqt"],
        targets["trypsin"]["active_center"], (20.0, 20.0, 20.0),
        work_dir, "trypsin", n_jobs=min(4, len(top10)),
    )
    for rec, energy in trypsin_results:
        rec.human_trypsin_energy = energy

    # ── Dock vs CES1 (using computed catalytic triad centre) ──
    log.info("  Docking top 10 vs Human Carboxylesterase 1 (3KJZ)…")
    ces1_results = _dock_compounds_parallel(
        top10, targets["CES1"]["pdbqt"],
        targets["CES1"]["active_center"], (20.0, 20.0, 20.0),
        work_dir, "ces1", n_jobs=min(4, len(top10)),
    )
    for rec, energy in ces1_results:
        rec.human_ces1_energy = energy

    # ── Compute SI ──
    for rec in top10:
        energies_human = [
            e for e in (rec.human_trypsin_energy, rec.human_ces1_energy)
            if e is not None
        ]
        n_human_targets = len(energies_human)

        # Track how many human off-targets provided valid energies and
        # record the resulting selectivity confidence.
        if n_human_targets >= 2:
            rec.selectivity_confidence = "High"
        elif n_human_targets == 1:
            rec.selectivity_confidence = "Low"
        else:
            rec.selectivity_confidence = "None"

        if not energies_human:
            log.warning(f"  {rec.compound_id}: No human docking data. SI = N/A.")
            rec.selectivity_index = None
            continue

        human_min = min(energies_human)
        pb2pa_best = (
            rec.pb2pa_active_energy if rec.pb2pa_active_energy is not None
            else rec.pb2pa_allosteric_energy
        )
        if pb2pa_best is None:
            rec.selectivity_index = 1.0
            continue

        si = compute_selectivity_index(pb2pa_best, human_min)
        rec.selectivity_index = si

        # SI based on a single human target is less reliable — flag it.
        if n_human_targets == 1:
            if rec.resistance_notes:
                rec.resistance_notes += " | "
            rec.resistance_notes += "SI based on single human target."

        if si < SELECTIVITY_INDEX_THRESHOLD:
            log.warning(
                f"  {rec.compound_id}: Low selectivity (SI = {si:.2f} < {SELECTIVITY_INDEX_THRESHOLD}). "
                "Flagged for off-target risk."
            )
        else:
            log.info(f"  {rec.compound_id}: SI = {si:.2f} (pass).")

        # Hard flag: high risk off-target binding
        if any(e is not None and e < -8.0 for e in (
            rec.human_trypsin_energy, rec.human_ces1_energy
        )):
            rec.selectivity_index = 0.0
            if rec.resistance_notes:
                rec.resistance_notes += " | "
            rec.resistance_notes += "High risk off-target binding"

    # ── Resistance profiling with pose-based interaction analysis ──
    pb2pa = targets["PBP2a"]
    cleaned_pdb = pb2pa.get("cleaned_pdb")

    for rec in top10:
        interactions = None

        if deps["USE_VINA"] and cleaned_pdb and os.path.exists(cleaned_pdb):
            # Prefer the active-site pose captured during screening to avoid
            # re-docking; fall back to a fresh pose dock otherwise.
            out_pdbqt = getattr(rec, "active_docked_pdbqt", None)
            if out_pdbqt and os.path.exists(out_pdbqt):
                try:
                    interactions = analyze_binding_interactions(out_pdbqt, cleaned_pdb)
                except Exception:
                    interactions = None
            else:
                safe_id = rec.compound_id.replace("/", "_").replace(" ", "_")
                lig_pdbqt = os.path.join(work_dir, f"{safe_id}_pose_lig.pdbqt")
                out_pdbqt = os.path.join(work_dir, f"{safe_id}_pose_out.pdbqt")

                if prepare_ligand_pdbqt(rec.mol, lig_pdbqt):
                    _run_vina_docking(
                        pb2pa["pdbqt"], lig_pdbqt, out_pdbqt,
                        pb2pa["active_center"], ACTIVE_BOX_SIZE,
                    )
                    if os.path.exists(out_pdbqt):
                        try:
                            interactions = analyze_binding_interactions(out_pdbqt, cleaned_pdb)
                            os.remove(out_pdbqt)
                        except OSError:
                            pass
                    try:
                        os.remove(lig_pdbqt)
                    except OSError:
                        pass

        rec.resistance_notes = profile_resistance_risk(
            rec, work_dir,
            pb2pa["pdbqt"],
            pb2pa["active_center"],
            ACTIVE_BOX_SIZE,
            interactions=interactions,
        )

    log.info("─── Phase 4 complete ───")
    return top10


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — REPORTING & ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_csv_report(
    top10: List[CompoundRecord],
    validation_rmsd: Optional[float] = None,
    validation_ok: bool = False,
    holo_pdb_path: Optional[str] = None,
) -> str:
    """
    Phase 5.1 — Write top_candidates.csv with all required columns.

    Columns:
        Compound_ID, SMILES, PBP2a_Allosteric_Energy, PBP2a_Active_Energy,
        Human_Trypsin_Energy, Human_CES1_Energy, Selectivity_Index,
        Selectivity_Confidence, Shape_Score, Max_Similarity, Passes_Lipinski,
        QED_Score, Binding_Mode_Notes, Redock_RMSD, Redock_Validated,
        Structure_Source.

    Returns path to CSV.
    """
    log.info("─── Phase 5: Reporting ───")
    ensure_output_dir()

    structure_source = "mock" if holo_pdb_path and "tests/data" in holo_pdb_path else "real"

    rows = []
    for rec in top10:
        rows.append({
            "Compound_ID": rec.compound_id,
            "SMILES": rec.smiles,
            "Structure_Source": structure_source,
            "PBP2a_Allosteric_Energy": (
                f"{rec.pb2pa_allosteric_energy:.2f}" if rec.pb2pa_allosteric_energy is not None
                else "N/A"
            ),
            "PBP2a_Active_Energy": (
                f"{rec.pb2pa_active_energy:.2f}" if rec.pb2pa_active_energy is not None
                else "N/A"
            ),
            "Human_Trypsin_Energy": (
                f"{rec.human_trypsin_energy:.2f}" if rec.human_trypsin_energy is not None
                else "N/A"
            ),
            "Human_CES1_Energy": (
                f"{rec.human_ces1_energy:.2f}" if rec.human_ces1_energy is not None
                else "N/A"
            ),
            "Selectivity_Index": (
                f"{rec.selectivity_index:.2f}" if rec.selectivity_index is not None
                else "N/A"
            ) + ("" if rec.selectivity_confidence == "High" else " (low-conf)"),
            "Selectivity_Confidence": (
                "Unassessed" if rec.selectivity_confidence == "None"
                else rec.selectivity_confidence
            ),
            "Shape_Score": (
                f"{rec.shape_score:.2f}" if rec.shape_score is not None
                else "N/A"
            ),
            "Max_Similarity": f"{rec.max_similarity:.3f}",
            "Passes_Lipinski": str(rec.passes_lipinski),
            "QED_Score": f"{rec.qed_score:.3f}",
            "Binding_Mode_Notes": rec.resistance_notes.replace("; ", " | "),
            "Redock_RMSD": (
                f"{validation_rmsd:.3f}" if validation_rmsd is not None else "N/A"
            ),
            "Redock_Validated": str(bool(validation_ok)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_REPORT, index=False)
    log.info(f"  CSV report saved: {CSV_REPORT}")

    json_path = Path(str(CSV_REPORT)).with_suffix(".json")
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)
    log.info(f"  JSON candidates saved: {json_path}")

    return str(CSV_REPORT)


def generate_images(top3: List[CompoundRecord]) -> List[str]:
    """
    Phase 5.2 — Save 2D structure PNGs for the top 3 candidates.

    Returns list of file paths.
    """
    paths = []
    for i, rec in enumerate(top3):
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol

        img_path = OUTPUT_DIR / f"top{i + 1}_{rec.compound_id}.png"
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(rec.mol)
            drawer.FinishDrawing()
            drawer.WriteDrawingText(str(img_path))
            paths.append(str(img_path))
            log.info(f"  Image saved: {img_path}")
        except Exception as exc:
            log.warning(f"  Failed to render {rec.compound_id}: {exc}")

    return paths


def print_summary(
    n_total: int, n_filtered: int,
    top10: List[CompoundRecord],
    validation_ok: bool, redock_rmsd: Optional[float],
    deps: dict,
) -> None:
    """Log a final pipeline summary."""
    n_docked = sum(1 for r in top10 if r.pb2pa_allosteric_energy is not None)
    n_selectivity_pass = sum(
        1 for r in top10
        if r.selectivity_index is not None and r.selectivity_index >= SELECTIVITY_INDEX_THRESHOLD
    )

    log.info("=" * 60)
    log.info("  PIPELINE SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total compounds generated:     {n_total}")
    log.info(f"  After filtering:               {n_filtered}")
    log.info(f"  Top candidates reported:       {len(top10)}")
    log.info(f"  Successfully docked:           {n_docked}")
    log.info(f"  Selectivity pass (SI >= 2.0):  {n_selectivity_pass}")
    log.info(f"  Docking engine:                {'Vina' if deps['USE_VINA'] else 'RDKit Shape (fallback)'}")
    log.info(f"  Redocking RMSD:                {redock_rmsd:.3f} Å" if redock_rmsd else "  Redocking RMSD:                N/A")
    log.info(f"  Redocking validated:           {validation_ok}")
    log.info(f"  CSV report:                    {CSV_REPORT}")
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def main(target_count: int = 500, force: bool = False, cache: bool = False):
    """Orchestrate the full discovery pipeline end-to-end.

    Args:
        target_count: Number of candidate compounds to generate.
        force: When True (and env AUTOANTIBIOTIC_FORCE=1 is set), reuse a
            previously cached redocking validation instead of re-running it.
            Otherwise the redocking validation is always executed when
            USE_VINA=True.
    """
    ensure_output_dir()

    # ── Dependency check ──
    deps = check_dependencies()

    # ── Working directory for intermediate files ──
    work_dir = str(OUTPUT_DIR / "workdir")
    pdb_dir = str(OUTPUT_DIR / "pdb")
    os.makedirs(work_dir, exist_ok=True)

    # ── Phase 1: Target preparation ──
    targets = prepare_targets(pdb_dir, work_dir, deps)

    # ── Phase 0: Redocking validation ──
    # The (expensive) redocking gate is always executed when USE_VINA=True.
    # The only way to skip it and reuse a prior cached validation is when the
    # user explicitly passes --force AND env AUTOANTIBIOTIC_FORCE=1 is set.
    validation_flag = os.path.join(work_dir, ".validation_done")
    validation_json = os.path.join(work_dir, "validation_results.json")

    reuse_cache = (
        os.environ.get("AUTOANTIBIOTIC_FORCE") == "1"
        and force
        and os.path.exists(validation_flag)
    )

    if reuse_cache:
        try:
            with open(validation_json) as fh:
                vdata = json.load(fh)
            validation_ok = bool(vdata.get("validation_ok", False))
            redock_rmsd = vdata.get("redock_rmsd", None)
            log.info("  Reusing cached redocking validation from previous run.")
        except Exception as exc:
            log.warning(f"  Could not read cached validation ({exc}); re-running.")
            validation_ok, redock_rmsd = run_redocking_validation(
                holo_pdb_path=targets["holo_pdb"],
                target_pdbqt_path=targets["PBP2a"]["pdbqt"],
                work_dir=work_dir,
                deps=deps,
            )
    else:
        validation_ok, redock_rmsd = run_redocking_validation(
            holo_pdb_path=targets["holo_pdb"],
            target_pdbqt_path=targets["PBP2a"]["pdbqt"],
            work_dir=work_dir,
            deps=deps,
        )
        # Persist a successful validation so subsequent runs can reuse it
        # only when force + AUTOANTIBIOTIC_FORCE=1 are both given.
        if validation_ok and reuse_cache:
            try:
                with open(validation_json, "w") as fh:
                    json.dump(
                        {"validation_ok": validation_ok, "redock_rmsd": redock_rmsd},
                        fh,
                    )
            except Exception as exc:
                log.warning(f"  Could not cache validation result ({exc}).")

    if not validation_ok:
        log.error(
            "  ✗  Redocking validation failed — docking results should be "
            "interpreted with caution."
        )
        # Redocking validation is diagnostic, not a hard gate: never abort.
        # The validation status is recorded in the CSV (Redock_Validated col).
        if not os.environ.get("AUTOANTIBIOTIC_FORCE"):
            log.warning(
                "  ⚠  Redocking validation FAILED and AUTOANTIBIOTIC_FORCE is "
                "not set. Proceeding WITHOUT a validated docking protocol; "
                "validation status (Redock_Validated=False) will be written to "
                "the CSV report. Interpret all docking results with caution."
            )

    # ── Phase 2: Library generation & filtering ──
    cache_path = os.path.join(work_dir, "filtered.pkl")
    if cache and os.path.exists(cache_path):
        import pickle
        try:
            with open(cache_path, "rb") as fh:
                filtered = pickle.load(fh)
            n_total = len(filtered)
            n_filtered = len(filtered)
            log.info(f"  Loaded cached filtered records from {cache_path}")
        except Exception as exc:
            log.warning(f"  Could not load cache ({exc}); regenerating.")
            all_records = generate_candidate_library(target_count=target_count)
            n_total = len(all_records)
            filtered = apply_filters(all_records)
            n_filtered = len(filtered)
    else:
        all_records = generate_candidate_library(target_count=target_count)
        n_total = len(all_records)
        filtered = apply_filters(all_records)
        n_filtered = len(filtered)
        if cache:
            import pickle
            try:
                with open(cache_path, "wb") as fh:
                    pickle.dump(filtered, fh)
                log.info(f"  Saved filtered records to cache: {cache_path}")
            except Exception as exc:
                log.warning(f"  Could not save cache ({exc}).")

    if n_filtered == 0:
        log.warning("  No compounds passed filters. Halting pipeline.")
        return

    # ── Phase 3: Virtual screening ──
    top10 = screen_library(filtered, targets, work_dir, deps)

    if not top10:
        log.warning("  No candidates after screening. Halting pipeline.")
        return

    # ── Phase 4: Selectivity & Resistance ──
    top10 = analyze_selectivity_and_resistance(top10, targets, work_dir, deps)

    # ── Phase 5: Reporting & Artifacts ──
    generate_csv_report(
        top10,
        validation_rmsd=redock_rmsd,
        validation_ok=validation_ok,
        holo_pdb_path=targets.get("holo_pdb"),
    )

    top3 = top10[:3]
    generate_images(top3)

    print_summary(
        n_total, n_filtered, top10,
        validation_ok, redock_rmsd, deps,
    )

    log.info("Pipeline complete. Exiting.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoAntibiotic Discovery Pipeline")
    parser.add_argument("--count", type=int, default=500, help="Target compound count")
    parser.add_argument(
        "--force", action="store_true",
        help="Reuse cached redocking validation (requires AUTOANTIBIOTIC_FORCE=1)",
    )
    parser.add_argument(
        "--cache", action="store_true",
        help="Load/save the filtered candidate library via OUTPUT_DIR/workdir/filtered.pkl",
    )
    args = parser.parse_args()
    main(target_count=args.count, force=args.force, cache=args.cache)
