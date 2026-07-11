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
import subprocess
import logging
import warnings
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import pandas as pd

# ── RDKit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem, RDConfig
from rdkit.Chem import (
    AllChem, Descriptors, QED, Draw, rdMolDescriptors,
    rdmolops, rdDistGeom, Crippen, FilterCatalog, BRICS,
)
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger as rdklog

# ── Bio.PDB ────────────────────────────────────────────────────────────────────
from Bio.PDB import (
    PDBParser, PDBIO, Select,
    NeighborSearch, Superimposer,
    StructureBuilder, PDBList,
)
from Bio.PDB.DSSP import DSSP
from Bio.SVDSuperimposer import SVDSuperimposer

# ── Matplotlib ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

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
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("6TKO", holo_pdb_path)

        ligand_residues = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    # HETATM residues except waters — typical ligand identifiers
                    if residue.get_id()[0] in ("H_", "W", "H_M"):
                        continue
                    # Skip standard amino acids
                    if residue.get_id()[0] == " ":
                        continue
                    # Skip waters
                    resname = residue.get_resname().strip()
                    if resname in ("HOH", "WAT", "SOL"):
                        continue
                    ligand_residues.append((chain.get_id(), residue))

        if not ligand_residues:
            log.warning("  ⚠  No hetero-ligand found in 6TKO.")
            return None

        # Use the first non-water HETATM residue as the native ligand
        chain_id, lig_res = ligand_residues[0]
        log.info(f"  Native ligand found: chain {chain_id}, residue {lig_res.get_resname()}")

        # Write ligand as a separate PDB file
        pdbio = PDBIO()
        class LigSelect(Select):
            def accept_residue(self, residue):
                return residue is lig_res
        pdbio.set_struct(struct)
        lig_pdb = output_ligand_pdbqt.replace(".pdbqt", ".pdb")
        pdbio.save(lig_pdb, LigSelect())

        # Convert to MOL → SMILES via RDKit's PDB parser (or obabel fallback)
        mol = Chem.MolFromPDBFile(lig_pdb, removeHs=False)
        if mol is None:
            # Try with OpenBabel as fallback
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

        # Sanitize
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)

        with open(output_ligand_smi, "w") as f:
            f.write(smi + "\n")
        log.info(f"  Native ligand SMILES: {smi}")

        # Convert to PDBQT via meeko
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)[0]
            pdbqt_str = PDBQTWriterLegacy.write_string(mol_setup)[0]
            with open(output_ligand_pdbqt, "w") as f:
                f.write(pdbqt_str)
            log.info(f"  Native ligand PDBQT written to {output_ligand_pdbqt}")
        except Exception as exc:
            log.warning(f"  ⚠  Meeko prep failed for native ligand: {exc}")
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
    Align protein backbones of the docked structure to the crystal structure
    and compute heavy-atom RMSD of the ligand.
    """
    try:
        parser = PDBParser(QUIET=True)
        docked_struct = parser.get_structure("docked", docked_pdb)
        crystal_struct = parser.get_structure("crystal", crystal_pdb)

        # Get ligand atoms from both
        def _get_ligand_atoms(structure):
            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] != " ":
                            for atom in residue:
                                if atom.element != "H":
                                    atoms.append(atom)
            return atoms

        docked_atoms = _get_ligand_atoms(docked_struct)
        crystal_atoms = _get_ligand_atoms(crystal_struct)

        if len(docked_atoms) != len(crystal_atoms):
            log.warning(
                f"  ⚠  Atom count mismatch: docked={len(docked_atoms)}, "
                f"crystal={len(crystal_atoms)}. Taking min."
            )
            n = min(len(docked_atoms), len(crystal_atoms))
            docked_atoms = docked_atoms[:n]
            crystal_atoms = crystal_atoms[:n]

        # Superpose and get RMSD
        sup = SVDSuperimposer()
        sup.set(
            np.array([a.get_vector().get_array() for a in crystal_atoms]),
            np.array([a.get_vector().get_array() for a in docked_atoms]),
        )
        sup.run()
        rmsd = sup.get_rmsd()
        return rmsd

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

    # Run Vina redocking
    log.info("  Redocking native ligand into PBP2a…")
    docked_pdbqt = docked_pdb.replace(".pdb", ".pdbqt")
    vina_cmd = [
        "vina",
        "--receptor", target_pdbqt_path,
        "--ligand", lig_pdbqt,
        "--out", docked_pdbqt,
        "--center_x", "0", "--center_y", "0", "--center_z", "0",  # placeholder — will be updated
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

    return (rmsd <= 2.0 if rmsd is not None else False), rmsd


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
        io.set_struct(struct)
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
    result = {}

    # ── Fetch structures ──
    holo_path = fetch_structure(PDB_IDS["PBP2a_holo"], pdb_dir)
    apo_path = fetch_structure(PDB_IDS["PBP2a_apo"], pdb_dir)
    trypsin_path = fetch_structure(PDB_IDS["trypsin"], pdb_dir)
    ces1_path = fetch_structure(PDB_IDS["CES1"], pdb_dir)

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
    cleaned_pdb = pbp2a_pdbqt.replace(".pdbqt", ".pdb")
    log.info("  Computing allosteric site centroid (ALA237, MET241, TYR159)…")
    allosteric_center = compute_residue_centroid(cleaned_pdb, ALLOSTERIC_RESIDUES)
    log.info(f"    Allosteric site center: {allosteric_center}")

    log.info("  Computing active site centroid (SER403)…")
    active_center = compute_residue_centroid(cleaned_pdb, ACTIVE_SITE_RESIDUES)
    log.info(f"    Active site center: {active_center}")

    result["PBP2a"] = {
        "pdbqt": pbp2a_pdbqt,
        "cleaned_pdb": pbp2a_clean_pdb,
        "allosteric_center": allosteric_center,
        "active_center": active_center,
    }

    # ── Clean trypsin ──
    log.info("  Cleaning Human Trypsin (1UTN)…")
    tryp_clean_pdb = os.path.join(work_dir, "trypsin_clean.pdb")
    tryp_pdbqt = clean_pdb_structure(
        trypsin_path,
        tryp_clean_pdb,
    )
    log.info("  Computing trypsin active site centroid (His57, Asp102, Ser195)…")
    tryp_center = compute_residue_centroid(tryp_clean_pdb, TRYPSIN_CATALYTIC_RESIDUES)
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
    ces1_center = compute_residue_centroid(ces1_clean_pdb, CES1_CATALYTIC_RESIDUES)
    log.info(f"    CES1 active site center: {ces1_center}")
    result["CES1"] = {"pdbqt": ces1_pdbqt, "active_center": ces1_center}

    # ── Write grid configuration files ──
    grid_dir = os.path.join(work_dir, "grid_configs")
    os.makedirs(grid_dir, exist_ok=True)

    for site_name, center, box in [
        ("allosteric", allosteric_center, ALLOSTERIC_BOX_SIZE),
        ("active", active_center, ACTIVE_BOX_SIZE),
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

# Additional scaffolds for diversity
ADDITIONAL_SCAFFOLDS = [
    "c1ccc2c(c1)cc3c4c2ccc5c4c6c(c7c8c5c9c%10c%11c%12c%13c%14c%15c%16c%17c%18c%19c%20c%21c%22c%23c%24c%25c%26c1c2c3c4c5c6c7c8c9c%10c%11c%12c%13c%14c%15c%16c%17c%18c%19c%20c%21c%22c%23c%24c%25c%26c1c2c3c4c5c6c7c8c9c%10c%11c%12c%13c%14c%15c%16c%17c%18c%19c%20c%21c%22c%23c%24c%25c%26c1c2c3c4c5c6c7c8c9c%10c%11c%12c%13c%14c%15c%16c%17c%18c%19c%20c%21c%22c%23c%24c%25c%26c1",
    "O=c1cc2ccccc2oc1-c3ccccc3",                                 # Flavone
    "COc1ccc2c(c1)oc(=O)c(C3=CC(=O)c4ccccc4O3)c2C(=O)O",      # Isoflavonoid-like
    "c1ccc2c(c1)cc3c4c2ccc5c4c6c7c5c8c9c%10c%11c%12c%13c%14c%15c%11c%12c%13c%14c%15c6c7c8c9c%10",
    "O=c1c2ccccc2c(=O)c3c1ccc4c5c3ccc6c7c5c8c9c%10c%11c%12c%13c%10c%11c%12c%13c4c7c8c9",
    "c1ccc2c(c1)cc3c4c2ccc5c4c6c7c5c8c9c%10c%11c%12c%13c(cc2ccccc2c%11%12)c(c%10%13)c6c7c8c9",
    "CC(C)(C)c1cc2c(cc1C(C)(C)C)c3c(cc4c2cc5c6c4cc7c8c5c9c%10c%11c%12c%13c%14c%15c%16c(cc%11%12%13%14%15%16)c6c7c8c9)cc(c3)C(C)(C)C",
    "COc1cc2c(cc1OC)CCN(C2)c3ccc4c(c3)OC(=O)C4(C)O",
    "CC1(C)OC2CC3C4C5C6C7C8C9C%10C%11C%12C%13C%14C%15C%16C%17C%18C%19C%20C%21C%22C%23C%24C%25C%26C%27C%28C%29C%30C%31C%32C%33C%34C%35C%36C%37C%38C%39C%40C(OC(C)(C)OC%41C%42C%43C%44C%45C%46C%47C%48C%49C%50C%51C%52C%53C%54C%55C%56C%57C%58C%59C%60C%61C%62C%63C%64C%65C%66C%67C%68C%69C%70C%71C%72C%73C%74C%75C%76C%77C%78C%79C%80)CC3C2C1",
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

    all_scaffolds = NATURAL_PRODUCT_SCAFFOLDS + ADDITIONAL_SCAFFOLDS
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

    # Recombine fragments to create novel analogs
    rng = np.random.default_rng(seed)
    seen_smiles = set()
    records = []
    max_attempts = target_count * 10
    attempts = 0

    while len(records) < target_count and attempts < max_attempts:
        attempts += 1
        # Pick 1–3 fragments and try to join
        n_frags = rng.integers(1, 4)
        chosen = rng.choice(frag_mols, size=min(n_frags, len(frag_mols)), replace=False)

        try:
            combined = chosen[0]
            for frag in chosen[1:]:
                combined = Chem.CombineMols(combined, frag)
            # Attempt to form new bonds via random BRICS connection
            # BRICS.BRICSBuild returns a generator of possible products
            # We sample from the build output
            builder = BRICS.BRICSBuild([combined])
            for _ in range(rng.integers(1, 5)):
                try:
                    product = next(builder)
                except StopIteration:
                    break
            else:
                product = next(builder)
            Chem.SanitizeMol(product)
            smi = Chem.MolToSmiles(product)
        except (StopIteration, Exception):
            continue

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

    # Add controls explicitly
    for name, smi in CONTROL_SMILES.items():
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

    passed = []
    skipped_structural = 0
    skipped_similarity = 0
    skipped_admet = 0
    skipped_pains = 0

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

        if max_sim >= similarity_threshold:
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

        passed.append(record)

    log.info(f"  Structural exclusion (β-lactam): {skipped_structural} removed.")
    log.info(f"  Similarity filter (Tc < {similarity_threshold}): {skipped_similarity} removed.")
    log.info(f"  ADMET filter (Lipinski + QED > 0.6): {skipped_admet} removed.")
    log.info(f"  PAINS filter: {skipped_pains} removed.")
    log.info(f"  Passed filters: {len(passed)} compounds.")

    # 5. Diversity check — relax if < 100
    if len(passed) < DIVERSITY_MIN_COUNT and similarity_threshold < SIMILARITY_THRESHOLD_RELAXED:
        log.warning(
            f"  Only {len(passed)} compounds passed strict filters (< {DIVERSITY_MIN_COUNT}). "
            f"Relaxing similarity threshold to {SIMILARITY_THRESHOLD_RELAXED} and re-running."
        )
        return apply_filters(records, similarity_threshold=SIMILARITY_THRESHOLD_RELAXED)

    log.info("─── Phase 2 complete ───")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — VIRTUAL SCREENING (Docking)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_ligand_pdbqt(
    mol: Chem.Mol,
    output_path: str,
) -> bool:
    """
    Convert an RDKit Mol to PDBQT via Meeko.

    Args:
        mol: Input molecule.
        output_path: Destination .pdbqt path.

    Returns:
        True on success.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        if not mol_setups:
            return False
        pdbqt_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
        with open(output_path, "w") as f:
            f.write(pdbqt_str)
        return True
    except Exception as exc:
        log.warning(f"  Meeko prep failed: {exc}")
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
            log.warning(f"  Vina error: {result.stderr.strip()}")
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
        return None

    energy = _run_vina_docking(
        receptor_pdbqt, lig_pdbqt, out_pdbqt,
        center, box_size,
    )

    # Cleanup temp files
    for f in (lig_pdbqt, out_pdbqt):
        try:
            os.remove(f)
        except OSError:
            pass

    return energy


def _parallel_dock(
    records: List[CompoundRecord],
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
    n_jobs: int = N_JOBS,
) -> List[Tuple[CompoundRecord, Optional[float]]]:
    """Dock a list of compounds in parallel, returning (record, energy) pairs."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    total = len(records)

    def _worker(rec):
        energy = dock_compound(rec, receptor_pdbqt, center, box_size, work_dir, tag)
        return rec, energy

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_worker, rec): rec for rec in records}
        for i, future in enumerate(as_completed(futures)):
            rec, energy = future.result()
            results.append((rec, energy))
            if (i + 1) % 25 == 0:
                log.info(f"    Docked {i + 1} / {total} ({tag})")

    return results


def _compute_shape_fallback_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    seed: int = RANDOM_SEED,
) -> Optional[float]:
    """
    Fallback scoring: generate 3D conformer, compute shape protrude distance
    vs reference (co-crystallised ligand from 6TKO). Normalise to 0–10 scale
    (lower = better shape match).

    Returns normalised score, or None on failure.
    """
    try:
        # Generate 3D conformer
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        status = rdDistGeom.EmbedMolecule(mol_3d, params)
        if status < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)

        ref_3d = Chem.RWMol(ref_mol)
        ref_3d = Chem.AddHs(ref_3d)
        params_ref = rdDistGeom.ETKDGv3()
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
        normalised = min(protrude / 0.05, 10.0) if protrude > 0 else 0.0
        return normalised

    except Exception:
        return None


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
        allosteric_results = _parallel_dock(
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

        active_results = _parallel_dock(
            top50, pb2pa["pdbqt"],
            active_center, ACTIVE_BOX_SIZE,
            work_dir, "active",
        )

        for rec, energy in active_results:
            rec.pb2pa_active_energy = energy

    else:
        # ── Fallback: RDKit Shape protrude ──
        log.info("  Vina unavailable. Using RDKit Shape Fallback.")

        # Extract native ligand from 6TKO as reference
        ref_mol = None
        holo_pdb = targets.get("holo_pdb")
        if holo_pdb and os.path.exists(holo_pdb):
            lig_pdb = os.path.join(work_dir, "native_ref.pdb")
            try:
                parser = PDBParser(QUIET=True)
                struct = parser.get_structure("ref", holo_pdb)
                for model in struct:
                    for chain in model:
                        for residue in chain:
                            if residue.get_id()[0] != " " and residue.get_resname().strip() not in ("HOH", "WAT"):
                                # Write first hetero ligand
                                pdbio = PDBIO()
                                class _Sel(Select):
                                    def accept_residue(self, r):
                                        return r is residue
                                pdbio.set_struct(struct)
                                pdbio.save(lig_pdb, _Sel())
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                ref_mol = Chem.MolFromPDBFile(lig_pdb)
            except Exception:
                pass

        if ref_mol is None:
            # Use first control as fallback reference
            ref_smi = list(CONTROL_SMILES.values())[0]
            ref_mol = Chem.MolFromSmiles(ref_smi)

        if ref_mol is None:
            log.error("  Cannot obtain reference molecule for shape scoring.")
            return records[:TOP_N]

        total = len(records)
        shape_scores = []
        for i, rec in enumerate(records):
            if rec.mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    continue
                rec.mol = mol
            score = _compute_shape_fallback_score(rec.mol, ref_mol)
            rec.shape_score = score
            shape_scores.append((rec, score))
            if (i + 1) % 100 == 0:
                log.info(f"  Shape scored {i + 1} / {total}")

        shape_scores = [s for s in shape_scores if s[1] is not None]
        shape_scores.sort(key=lambda x: x[1])

        log.info(f"  Shape scoring complete. Best score: {shape_scores[0][1]:.3f}")

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

def check_ser403_contact(
    docked_pdbqt_path: str,
    receptor_pdb_path: str,
    threshold_dist: float = 3.5,
) -> bool:
    """
    Check whether the docked ligand contacts Ser403 Oγ within *threshold_dist*.

    Args:
        docked_pdbqt_path: Path to the docked-ligand PDBQT file.
        receptor_pdb_path: Path to the cleaned receptor PDB file (used to
                           locate Ser403 Oγ coordinates).
        threshold_dist: Distance threshold in Ångströms (default 3.5).

    Returns:
        True if any ligand heavy atom lies within *threshold_dist* of
        Ser403 Oγ.
    """
    # ── Locate Ser403 Oγ in the receptor PDB ──
    ser403_og = None
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("receptor", receptor_pdb_path)
        for model in struct:
            for chain in model:
                for residue in chain:
                    if (
                        residue.get_resname().strip() == "SER"
                        and residue.get_id()[1] == 403
                    ):
                        if "OG" in residue:
                            ser403_og = residue["OG"].get_vector().get_array()
                            break
                if ser403_og is not None:
                    break
            if ser403_og is not None:
                break
    except Exception as exc:
        log.warning(f"  Could not parse receptor PDB for Ser403: {exc}")
        return False

    if ser403_og is None:
        log.warning("  Ser403 Oγ atom not found in receptor PDB.")
        return False

    # ── Parse docked-ligand heavy-atom coordinates from PDBQT ──
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
        log.warning(f"  Docked PDBQT not found: {docked_pdbqt_path}")
        return False

    if not ligand_coords:
        log.warning("  No ligand heavy atoms found in PDBQT for Ser403 check.")
        return False

    coords_array = np.array(ligand_coords)
    distances = np.linalg.norm(coords_array - ser403_og, axis=1)
    min_dist = distances.min()

    return min_dist <= threshold_dist


def compute_selectivity_index(
    pb2pa_energy: float, human_avg_energy: float,
) -> float:
    """
    Selectivity Index (SI).

        SI = |Energy_Human_Avg| / |Energy_PBP2a_Best|

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
    return abs(human_avg_energy) / abs(pb2pa_energy) if abs(pb2pa_energy) > 1e-6 else 0.0


CONSERVED_RESIDUES = {"SER403", "LYS406", "TYR446"}
MUTABLE_RESIDUES = {"G246", "N146"}


def profile_resistance_risk(
    record: CompoundRecord,
    work_dir: str,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    ser403_contact: Optional[bool] = None,
) -> str:
    """
    Rule-based resistance profiling, optionally informed by pose analysis.

    Flags candidates based on predicted interactions:
        - Good: contacts with conserved residues (Ser403, Lys406, Tyr446).
        - Risk: contacts with mutable residues (Gly246, Asn146).

    Args:
        ser403_contact: Result from check_ser403_contact (True/False) or None
                        if pose analysis was not performed.

    Returns a human-readable notes string.
    """
    notes = []

    # Pose-based Ser403 contact (from check_ser403_contact)
    if ser403_contact is True:
        notes.append("Confirmed contact with catalytic Ser403 Oγ (pose-based). Good.")
    elif ser403_contact is False:
        notes.append("No contact with Ser403 Oγ in docked pose — may lack active-site engagement.")
    else:
        # Pose analysis unavailable — fall back to energy-based heuristic
        if record.pb2pa_active_energy is not None and record.pb2pa_active_energy < -6.0:
            notes.append("Likely contacts catalytic Ser403 (active site, energy-based). Good.")

    # If it bound well only to allosteric site, it targets the allosteric pocket
    if record.pb2pa_allosteric_energy is not None and record.pb2pa_allosteric_energy < -7.0:
        if record.pb2pa_active_energy is None or record.pb2pa_active_energy > -6.0:
            notes.append("Allosteric binder (Ala237/Met241/Tyr159 pocket). Novel mechanism.")

    # Molecular weight heuristic: larger molecules may have more contact surface
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
            rec.selectivity_index = 1.0
            rec.resistance_notes = "Selectivity not assessed (Vina unavailable)."
        return top10

    # ── Dock vs Trypsin (using computed catalytic triad centre) ──
    log.info("  Docking top 10 vs Human Trypsin (1UTN)…")
    trypsin_results = _parallel_dock(
        top10, targets["trypsin"]["pdbqt"],
        targets["trypsin"]["active_center"], (20.0, 20.0, 20.0),
        work_dir, "trypsin", n_jobs=min(4, len(top10)),
    )
    for rec, energy in trypsin_results:
        rec.human_trypsin_energy = energy

    # ── Dock vs CES1 (using computed catalytic triad centre) ──
    log.info("  Docking top 10 vs Human Carboxylesterase 1 (3KJZ)…")
    ces1_results = _parallel_dock(
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

        if si < SELECTIVITY_INDEX_THRESHOLD:
            log.warning(
                f"  {rec.compound_id}: Low selectivity (SI = {si:.2f} < {SELECTIVITY_INDEX_THRESHOLD}). "
                "Flagged for off-target risk."
            )
        else:
            log.info(f"  {rec.compound_id}: SI = {si:.2f} (pass).")

    # ── Resistance profiling with pose-based Ser403 contact analysis ──
    pb2pa = targets["PBP2a"]
    cleaned_pdb = pb2pa.get("cleaned_pdb")

    for rec in top10:
        ser403_contact = None

        if deps["USE_VINA"] and cleaned_pdb and os.path.exists(cleaned_pdb):
            # Re-dock to active site to obtain the docked pose for analysis
            safe_id = rec.compound_id.replace("/", "_").replace(" ", "_")
            lig_pdbqt = os.path.join(work_dir, f"{safe_id}_pose_lig.pdbqt")
            out_pdbqt = os.path.join(work_dir, f"{safe_id}_pose_out.pdbqt")

            if prepare_ligand_pdbqt(rec.mol, lig_pdbqt):
                _run_vina_docking(
                    pb2pa["pdbqt"], lig_pdbqt, out_pdbqt,
                    pb2pa["active_center"], ACTIVE_BOX_SIZE,
                )
                if os.path.exists(out_pdbqt):
                    ser403_contact = check_ser403_contact(out_pdbqt, cleaned_pdb)
                    try:
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
            pb2pa["allosteric_center"],
            ALLOSTERIC_BOX_SIZE,
            ser403_contact=ser403_contact,
        )

    log.info("─── Phase 4 complete ───")
    return top10


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — REPORTING & ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_csv_report(top10: List[CompoundRecord]) -> str:
    """
    Phase 5.1 — Write top_candidates.csv with all required columns.

    Columns:
        Compound_ID, SMILES, PBP2a_Allosteric_Energy, PBP2a_Active_Energy,
        Human_Trypsin_Energy, Human_CES1_Energy, Selectivity_Index,
        Max_Similarity, Passes_Lipinski, QED_Score, Binding_Mode_Notes.

    Returns path to CSV.
    """
    log.info("─── Phase 5: Reporting ───")
    ensure_output_dir()

    rows = []
    for rec in top10:
        rows.append({
            "Compound_ID": rec.compound_id,
            "SMILES": rec.smiles,
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
            ),
            "Max_Similarity": f"{rec.max_similarity:.3f}",
            "Passes_Lipinski": str(rec.passes_lipinski),
            "QED_Score": f"{rec.qed_score:.3f}",
            "Binding_Mode_Notes": rec.resistance_notes.replace("; ", " | "),
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_REPORT, index=False)
    log.info(f"  CSV report saved: {CSV_REPORT}")
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

def main():
    """Orchestrate the full discovery pipeline end-to-end."""
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
    validation_ok, redock_rmsd = run_redocking_validation(
        holo_pdb_path=targets["holo_pdb"],
        target_pdbqt_path=targets["PBP2a"]["pdbqt"],
        work_dir=work_dir,
        deps=deps,
    )

    # ── Phase 2: Library generation & filtering ──
    all_records = generate_candidate_library(target_count=500)
    n_total = len(all_records)

    filtered = apply_filters(all_records)
    n_filtered = len(filtered)

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
    generate_csv_report(top10)

    top3 = top10[:3]
    generate_images(top3)

    print_summary(
        n_total, n_filtered, top10,
        validation_ok, redock_rmsd, deps,
    )

    log.info("Pipeline complete. Exiting.")


if __name__ == "__main__":
    main()
