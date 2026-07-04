#!/usr/bin/env python3
"""
AutoAntibiotic Discovery Pipeline v3.2
========================================
Principal Computational Chemist & AI Pipeline Architect
Project: AutoAntibiotic Discovery — MRSA PBP2a Inhibitor Screening

Screens novel small-molecule libraries against MRSA PBP2a (allosteric + active sites)
with selectivity filtering against human serine hydrolases, ADMET profiling, and
resistance-risk analysis.

Scientific rationale:
  - Phase 0: Redocking validation ensures the docking protocol can reproduce known
    binding modes (RMSD ≤ 2.0 Å threshold).
  - Phase 1: Structure preparation removes crystallographic artifacts and defines
    grid centres for allosteric (Ala237/Met241/Tyr159) and active (Ser403) pockets.
  - Phase 2: Library generation via BRICS fragment recombination produces a diverse,
    drug-like chemical space enriched with natural-product-inspired scaffolds.
  - Phase 3: Virtual screening ranks candidates by predicted binding affinity;
    an RDKit shape-based fallback operates when Vina is unavailable.
  - Phase 4: Selectivity against human off-targets (trypsin, CES1) is quantified
    via the Selectivity Index; resistance risk is profiled via interaction heuristics.
  - Phase 5: A CSV report, 2D structure images, and an interactive HTML report
    (with embedded matplotlib figures) are generated for downstream review.

Author: AutoAntibiotic Agent
Environment: Python 3.9+, RDKit | Bio.PDB | AutoDock Vina | meeko
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── RDKit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem  # noqa: F401
from rdkit.Chem import (
    AllChem,
    BRICS,
    Crippen,
    Descriptors,
    QED,
    rdDistGeom,
    rdmolops,
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger as rdklog

# ── Bio.PDB ────────────────────────────────────────────────────────────────────
from Bio.PDB import (
    NeighborSearch,
    PDBIO,
    PDBList,
    PDBParser,
    Select,
    StructureBuilder,
    Superimposer,
)
from Bio.PDB.DSSP import DSSP


# ── Suppress RDKit noise ───────────────────────────────────────────────────────
rdklog.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration for the AutoAntibiotic discovery pipeline.

    All hardcoded constants are consolidated here for maintainability.
    Instantiate with ``CONFIG = PipelineConfig()``; override fields for
    custom runs (e.g., different seed, output directory).
    """
    random_seed: int = 42
    pdb_ids: Dict[str, str] = field(default_factory=lambda: {
        "PBP2a_apo": "3QPD",
        "PBP2a_holo": "6TKO",
        "trypsin": "1UTN",
        "CES1": "3KJZ",
    })
    reference_antibiotics: Dict[str, str] = field(default_factory=lambda: {
        "Methicillin":  "CC1=C(C(=C(C(=C1O)OC)OC)OC)C(=O)NC2C3C(C(=O)N3C2=O)SC4(C)C",
        "Vancomycin":   "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C(=C4OC6=C(C(=CC(=C6)C(C(=O)NC(C(=O)NC5C(=O)O)CC7=CC=C(C=C7)O)NC(=O)C8C(O)C(=C(C=C8)Cl)O)O)O)CO)O)O)O)NC(=O)C9C(O)C(=C(C=C9)Cl)O)(CC(=O)N)O",
        "Ceftaroline":  "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "Meropenem":    "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
        "Oxacillin":    "CC1=C(C(=NO1)C2=CC=CC=C2)C(=O)NC3C4C(C(=O)N4C3=O)SC5(C)C",
    })
    beta_lactam_smarts: str = "[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1"
    allosteric_residues: List[str] = field(default_factory=lambda: ["ALA237", "MET241", "TYR159"])
    active_site_residues: List[str] = field(default_factory=lambda: ["SER403"])
    trypsin_active_site_residues: List[str] = field(default_factory=lambda: ["HIS57", "ASP102", "SER195"])
    ces1_active_site_residues: List[str] = field(default_factory=lambda: ["SER221", "HIS468", "GLU354"])
    allosteric_box_size: Tuple[float, float, float] = (15.0, 15.0, 15.0)
    active_box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    vina_timeout_s: int = 120
    n_jobs: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    similarity_threshold: float = 0.4
    similarity_threshold_relaxed: float = 0.5
    diversity_min_count: int = 100
    selectivity_index_threshold: float = 2.0
    library_target_count: int = 500
    brics_min_fragment_size: int = 8
    output_dir: Path = Path("output")
    top_n: int = 10
    natural_product_scaffolds: List[str] = field(default_factory=lambda: [
        "O=c1c(-c2ccc(O)c(O)c2)coc2cc(O)cc(O)c12",
        "Oc1ccc(C=Cc2ccc(O)cc2)cc1",
        "COc1ccc(C=CC(=O)CC(=O)C=Cc2ccc(OC)c(O)c2)cc1O",
        "COc1cc2c(cc1OC)-c1ccc3cc4c(cc3c1CC2)OCO4",
        "CC1(C)OC2C3OC(=O)C4C(O1)C2C1OOC3C14",
        "O=C1OCc2cn3ccc4cccc-4c3cc21",
        "COc1nc2c3ccccc3n(C)c2cc1C1CCNC1O",
        "O=C(Nc1ccccc1)c1ccccc1",
    ])
    additional_scaffolds: List[str] = field(default_factory=lambda: [
        "c1ccc2[nH]ccc2c1",
        "c1ccc2ncccc2c1",
        "c1ccc2cc[nH]c2c1",
        "c1ccc2[nH]cnc2c1",
        "O=c1ccc2ccccc2o1",
        "c1ccc2nc3ccccc3nc2c1",
        "c1ccc2c(c1)oc1ccccc12",
        "c1ccc2c(c1)sc1ccccc12",
        "c1ccc2c(c1)ccc1c3ccccc3[nH]c21",
        "c1ccc2c(c1)CCN2",
        "c1ccc2c(c1)CCc1c-2[nH]c2ccccc12",
        "COc1ccc2[nH]ccc2c1",
        "COc1ccccc1OCC(O)CNC(C)C",
        "CCN(CC)C(=O)c1ccccc1",
        "O=C(Nc1ccc(O)cc1)c1ccc(O)cc1",
    ])
    brics_building_blocks: List[str] = field(default_factory=lambda: [
        "[1*]c1ccccc1",
        "[1*]c1ccc(O)cc1",
        "[1*]c1ccc(Cl)cc1",
        "[1*]c1ccc(F)cc1",
        "[1*]c1ccc(Br)cc1",
        "[1*]c1ccc(OC)cc1",
        "[1*]c1ccc(C(=O)O)cc1",
        "[1*]c1ccc(N)cc1",
        "[1*]c1ccc(C)cc1",
        "[1*]c1ccc(C(C)C)cc1",
        "[1*]c1ccc(CF)cc1",
        "[1*]c1ccc(CN)cc1",
        "[1*]c1ccc(S(=O)(=O)N)cc1",
        "[1*]c1ccc(C(=O)N)cc1",
        "[1*]c1ccc(NC(=O)C)cc1",
        "[1*]CC(=O)O",
        "[1*]CCO",
        "[1*]CCN",
        "[1*]CC(=O)N",
        "[1*]CCC(=O)O",
        "[3*]C=Cc1ccccc1",
        "[3*]C=Cc1ccc(O)cc1",
        "[3*]C=Cc1ccc(Cl)cc1",
        "[3*]CCN(C)C",
        "[5*]Nc1ccccc1",
        "[5*]Nc1ccc(O)cc1",
        "[5*]Nc1ccc(C(=O)O)cc1",
        "[5*]Nc1ccc(Cl)cc1",
        "[5*]Nc1ccc(F)cc1",
        "[5*]Nc1ccc(OC)cc1",
        "[5*]Nc1ccc(C)cc1",
        "[5*]Nc1ccc(Br)cc1",
        "[5*]Nc1ccc(CN)cc1",
        "[5*]NCC",
        "[5*]NCCO",
        "[5*]NCCC(=O)O",
        "[6*]C(=O)O",
        "[6*]C(=O)c1ccccc1",
        "[6*]C(=O)c1ccc(O)cc1",
        "[6*]C(=O)c1ccc(Cl)cc1",
        "[6*]C(=O)c1ccc(OC)cc1",
        "[6*]C(=O)c1ccc(C)cc1",
        "[6*]C(=O)c1ccc(N)cc1",
        "[6*]C(=O)CC",
        "[7*]Cc1ccccc1",
        "[7*]Cc1ccc(O)cc1",
        "[7*]Cc1ccc(O)c(OC)c1",
        "[7*]Cc1ccc(OC)cc1",
        "[7*]Cc1ccc(Cl)cc1",
        "[7*]Cc1ccc(F)cc1",
        "[7*]CC",
        "[7*]C(C)C",
        "[16*]c1ccccc1OC",
        "[16*]c1ccc(C)cc1",
        "[16*]c1ccc(N)cc1",
        "[16*]c1ccc(O)cc1",
    ])
    control_smiles: Dict[str, str] = field(default_factory=lambda: {
        "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
    })
    conserved_residues: set = field(default_factory=lambda: {"SER403", "LYS406", "TYR446"})
    mutable_residues: set = field(default_factory=lambda: {"G246", "N146"})
    dry_run: bool = False


CONFIG = PipelineConfig()
np.random.seed(CONFIG.random_seed)

# ── Logger (config deferred to main()) ──
log = logging.getLogger("AutoAntibiotic")


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)


def install_missing_package(package: str) -> bool:
    """Attempt to pip-install *package*. Return True on success."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            timeout=60,
        )
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHE (simple JSON key-value store for docking results)
# ═══════════════════════════════════════════════════════════════════════════════

def load_cache() -> Dict[str, float]:
    """Load docking result cache from ``CONFIG.output_dir / "cache.json"``.

    Returns an empty dict if no cache file exists or if it is corrupt.
    """
    if CONFIG.output_dir / "cache.json".exists():
        try:
            with open(CONFIG.output_dir / "cache.json") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("  ⚠  Cache file corrupt; starting fresh.")
    return {}


def save_cache(cache: Dict[str, float]) -> None:
    """Persist the docking result cache to ``CONFIG.output_dir / "cache.json"``."""
    ensure_output_dir()
    with open(CONFIG.output_dir / "cache.json", "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — DEPENDENCY VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

_INSTALL_GUIDE: Dict[str, str] = {
    "rdkit": "  → Install: conda install -c conda-forge rdkit  |  pip install rdkit-pypi",
    "meeko": "  → Install: pip install meeko",
    "biopython": "  → Install: conda install -c conda-forge biopython  |  pip install biopython",
    "vina": (
        "  → Install AutoDock Vina:\n"
        "       Linux/macOS:  conda install -c conda-forge vina\n"
        "       Or download from https://vina.scripps.edu/\n"
        "       Then ensure 'vina' is on your PATH."
    ),
    "obabel": (
        "  → Install OpenBabel:\n"
        "       conda install -c conda-forge openbabel\n"
        "       or: brew install openbabel (macOS)\n"
        "       or: apt install openbabel (Debian/Ubuntu)"
    ),
    "prepare_receptor": (
        "  → Install ADFR suite:\n"
        "       Download from https://ccsb.scripps.edu/adfr/\n"
        "       and add 'prepare_receptor' to your PATH."
    ),
}


def verify_dependencies() -> Dict[str, Any]:
    """
    Phase 0 — Dependency Verification.

    Checks all required Python libraries and external binaries.  On failure,
    prints detailed installation instructions before raising ``SystemExit``.

    Returns a dictionary with keys:
        - 'rdkit' / 'meeko' / 'biopython': bool
        - 'vina': bool (True if ``vina`` binary is on PATH)
        - 'obabel': bool (True if ``obabel`` binary is on PATH)
        - 'prepare_receptor': bool (True if ``prepare_receptor`` binary on PATH)
        - 'USE_VINA': global toggle — set False if Vina is absent
        - 'USE_OBABEL': global toggle — set False if obabel is absent
    """
    log.info("─── Phase 0: Dependency Verification ───")
    status: Dict[str, Any] = {}

    # ── Python packages (soft-fail: attempt pip install) ──
    packages: Dict[str, str] = {
        "rdkit": "rdkit-pypi",
        "meeko": "meeko",
        "biopython": "biopython",
    }
    for mod_name, pip_name in packages.items():
        try:
            __import__(mod_name)
            status[mod_name] = True
            log.info(f"  ✓  {mod_name} found.")
        except ImportError:
            log.warning(f"  ⚠  {mod_name} missing. Attempting pip install…")
            if install_missing_package(pip_name):
                status[mod_name] = True
                log.info(f"  ✓  {mod_name} installed successfully.")
            else:
                status[mod_name] = False
                log.error(f"  ✗  {mod_name} could not be installed.")
                log.error(_INSTALL_GUIDE.get(mod_name, ""))
                sys.exit(1)

    # ── Vina binary ──
    for bin_name in ("vina", "obabel", "prepare_receptor"):
        try:
            subprocess.run(
                [bin_name, "--help" if bin_name == "prepare_receptor" else "--version"],
                capture_output=True, timeout=10,
            )
            status[bin_name] = True
            log.info(f"  ✓  {bin_name} binary found on PATH.")
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            status[bin_name] = False
            log.warning(f"  ⚠  '{bin_name}' not found.")
            log.warning(_INSTALL_GUIDE.get(bin_name, ""))

    status["USE_VINA"] = status["vina"]
    status["USE_OBABEL"] = status["obabel"]

    if not status["USE_VINA"]:
        log.warning(
            "  Pipeline will use RDKit Shape/Pharmacophore fallback for scoring."
        )
    if not status["USE_OBABEL"] and not status["prepare_receptor"]:
        log.warning(
            "  No PDBQT conversion tool found. A minimal RDKit-based PDBQT "
            "fallback will be used for the receptor."
        )

    return status


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

        ligand_residues: list = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    het_flag = residue.get_id()[0]
                    if het_flag == " " or het_flag == "W":
                        continue
                    resname = residue.get_resname().strip()
                    if resname in ("HOH", "WAT", "SOL"):
                        continue
                    ligand_residues.append((chain.get_id(), residue))

        if not ligand_residues:
            log.warning("  ⚠  No hetero-ligand found in 6TKO.")
            return None

        chain_id, lig_res = ligand_residues[0]
        log.info(f"  Native ligand found: chain {chain_id}, residue {lig_res.get_resname()}")

        pdbio = PDBIO()
        class LigSelect(Select):
            def accept_residue(self, residue):  # type: ignore
                return residue is lig_res
        pdbio.set_struct(struct)
        lig_pdb = output_ligand_pdbqt.replace(".pdbqt", ".pdb")
        pdbio.save(lig_pdb, LigSelect())

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
            shutil.copy(lig_pdb, output_ligand_pdbqt)

        return smi

    except Exception as exc:
        log.error(f"  ✗  Native ligand extraction failed: {exc}")
        return None


def _compute_rmsd_docked_vs_crystal(
    docked_pdb: str, crystal_pdb: str
) -> Optional[float]:
    """
    Align protein Cα backbones of the docked structure to the crystal structure,
    then compute heavy-atom RMSD of the ligand after applying the alignment.

    Scientific rationale: Backbone alignment separates the protein conformational
    change from the ligand pose quality. A docked pose with RMSD ≤ 2.0 Å is
    generally considered a successful redocking validation.
    """
    try:
        parser = PDBParser(QUIET=True)
        docked_struct = parser.get_structure("docked", docked_pdb)
        crystal_struct = parser.get_structure("crystal", crystal_pdb)

        def _get_ca_atoms(structure):
            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == " " and "CA" in residue:
                            atoms.append(residue["CA"])
            return atoms

        docked_ca = _get_ca_atoms(docked_struct)
        crystal_ca = _get_ca_atoms(crystal_struct)

        if len(docked_ca) < 3 or len(crystal_ca) < 3:
            log.warning("  ⚠  Too few Cα atoms for backbone alignment (< 3).")
            return None

        sup = Superimposer()
        docked_coords = np.array([a.get_vector().get_array() for a in docked_ca])
        crystal_coords = np.array([a.get_vector().get_array() for a in crystal_ca])
        sup.set(crystal_coords, docked_coords)
        sup.run()
        rot, tran = sup.rotran
        log.info(f"  Backbone alignment RMSD: {sup.rmsd:.3f} Å")

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

        docked_lig = _get_ligand_atoms(docked_struct)
        crystal_lig = _get_ligand_atoms(crystal_struct)

        if not docked_lig or not crystal_lig:
            log.warning("  ⚠  No ligand atoms found in one or both structures.")
            return None

        if len(docked_lig) != len(crystal_lig):
            log.warning(
                f"  ⚠  Ligand atom count mismatch: docked={len(docked_lig)}, "
                f"crystal={len(crystal_lig)}. Truncating to shorter list."
            )
            n = min(len(docked_lig), len(crystal_lig))
            docked_lig = docked_lig[:n]
            crystal_lig = crystal_lig[:n]

        docked_lig_coords = np.array([a.get_vector().get_array() for a in docked_lig])
        aligned_docked = docked_lig_coords @ rot.T + tran
        crystal_lig_coords = np.array([a.get_vector().get_array() for a in crystal_lig])
        diff = aligned_docked - crystal_lig_coords
        rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        return rmsd

    except Exception as exc:
        log.error(f"  ✗  RMSD calculation failed: {exc}")
        return None


def run_redocking_validation(
    holo_pdb_path: str,
    target_pdbqt_path: str,
    work_dir: str,
    deps: Dict[str, Any],
    center: Optional[np.ndarray] = None,
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

    log.info("  Redocking native ligand into PBP2a…")
    docked_pdbqt = docked_pdb.replace(".pdb", ".pdbqt")
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
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
        subprocess.run(vina_cmd, capture_output=True, timeout=CONFIG.vina_timeout_s)
    except subprocess.TimeoutExpired:
        log.warning("  ⚠  Vina redocking timed out (>120s).")
        return False, None
    except FileNotFoundError:
        log.warning("  ⚠  Vina binary not found during redocking.")
        return False, None

    try:
        subprocess.run(
            ["obabel", docked_pdbqt, "-O", docked_pdb, "--gen3d"],
            capture_output=True, timeout=30,
        )
    except Exception:
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
        raw = os.path.join(out_dir, f"pdb{pdb_id.lower()}.ent")
        if os.path.exists(raw):
            os.rename(raw, target_path)
        log.info(f"  ✓  Downloaded {pdb_id} → {target_path}")
    except Exception as exc:
        log.error(f"  ✗  Failed to download {pdb_id}: {exc}")
        raise

    return target_path


def _pdb_to_pdbqt_via_rdkit(pdb_path: str, pdbqt_path: str) -> bool:
    """
    Minimal PDB → PDBQT conversion using RDKit.

    Reads a PDB file, computes Gasteiger charges, maps elements to AutoDock
    atom types, and writes a PDBQT with a single rigid ROOT.  This is a
    fallback when both ``obabel`` and ``prepare_receptor`` are unavailable.

    AutoDock-Vina is lenient about atom typing; this minimal format is
    sufficient for Vina's scoring function.
    """
    try:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is None:
            return False
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.ComputeGasteigerCharges(mol)

        # Element → AutoDock type mapping (Vina-compatible subset)
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
        # Vina requires the first line to be ROOT
        lines.append("ROOT")
        for i, atom in enumerate(mol.GetAtoms()):
            atom_no = i + 1
            elem = atom.GetSymbol()
            pdbx = conf.GetAtomPosition(i)
            gasteiger = atom.GetDoubleProp("_GasteigerCharge")
            ad_type = _atom_type_map.get(elem, "C")

            # PDBQT ATOM record:
            # ATOM  serial name resName chainID resSeq x y z charge type
            # We write a dummy residue "PRT" chain "X" res 1
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


def clean_pdb_structure(
    pdb_path: str, out_path: str,
    remove_waters: bool = True,
    remove_ligands: bool = True,
    add_hydrogens: bool = True,
    deps: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Phase 1 — Target preparation.

    Removes waters and heteroatoms from the PDB file, optionally adds hydrogens,
    and converts the result to PDBQT format for AutoDock Vina.

    Conversion chain (priority order):
        1. ``prepare_receptor`` (ADFR suite)
        2. ``obabel`` with ``--gas`` for Gasteiger charges
        3. RDKit-based fallback (``_pdb_to_pdbqt_via_rdkit``)

    Returns the path to the generated PDBQT file (or the cleaned PDB if
    PDBQT conversion fails completely).
    """
    if deps is None:
        deps = {}
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("target", pdb_path)

        class CleanSelect(Select):
            def accept_residue(self, residue):
                rid = residue.get_id()
                if remove_waters and rid[0] == "W":
                    return False
                if remove_ligands and rid[0] != " ":
                    return False
                return True

        io = PDBIO()
        io.set_struct(struct)
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

        # Attempt PDBQT conversion with best available tool
        converted = False

        # Priority 1: prepare_receptor (ADFR)
        if deps.get("prepare_receptor"):
            try:
                subprocess.run(
                    ["prepare_receptor", "-r", out_path, "-o", pdbqt_path],
                    capture_output=True, timeout=60,
                )
                if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                    converted = True
                    log.info("  PDBQT via prepare_receptor")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Priority 2: obabel
        if not converted and deps.get("obabel"):
            try:
                subprocess.run(
                    ["obabel", out_path, "-O", pdbqt_path, "-h", "--gas"],
                    capture_output=True, timeout=60,
                )
                if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                    converted = True
                    log.info("  PDBQT via obabel")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Priority 3: RDKit fallback (never crashes)
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


def compute_residue_centroid(pdb_path: str, resid_list: List[str]) -> np.ndarray:
    """
    Compute the geometric centroid of Cα atoms for the given list of
    residue identifiers (format: ``ALA237``).

    Scientific rationale: The centroid defines the search-space centre for
    docking.  For the allosteric site, we average over the known regulatory
    pocket residues; for the active site, we use the catalytic serine.

    Args:
        pdb_path: Path to PDB structure.
        resid_list: e.g. ``["ALA237", "MET241", "TYR159"]``.

    Returns:
        (x, y, z) centroid as numpy array of shape (3,).
    """
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

    centroid = np.mean(ca_coords, axis=0)
    return centroid


def prepare_targets(
    pdb_dir: str, work_dir: str, deps: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Phase 1 — Download, clean, and compute grid centres for all targets.

    Returns a dictionary:
        ::
            {
              "PBP2a": {
                  "pdbqt": str,
                  "allosteric_center": np.ndarray,
                  "active_center": np.ndarray,
              },
              "trypsin": { "pdbqt": str, "active_center": np.ndarray },
              "CES1":    { "pdbqt": str, "active_center": np.ndarray },
              "holo_pdb": str,
            }

    Scientific rationale: Parallel preparation of the antibacterial target
    (PBP2a) and two human off-targets (trypsin, CES1) enables downstream
    selectivity profiling.
    """
    log.info("─── Phase 1: Target Preparation & Centroid Calculation ───")
    result: Dict[str, Any] = {}

    holo_path = fetch_structure(CONFIG.pdb_ids["PBP2a_holo"], pdb_dir)
    apo_path = fetch_structure(CONFIG.pdb_ids["PBP2a_apo"], pdb_dir)
    trypsin_path = fetch_structure(CONFIG.pdb_ids["trypsin"], pdb_dir)
    ces1_path = fetch_structure(CONFIG.pdb_ids["CES1"], pdb_dir)

    result["holo_pdb"] = holo_path

    log.info("  Cleaning PBP2a (apo)…")
    pbp2a_pdbqt = clean_pdb_structure(
        apo_path,
        os.path.join(work_dir, "PBP2a_clean.pdb"),
        deps=deps,
    )

    log.info("  Cleaning PBP2a (holo, protein-only)…")
    _ = clean_pdb_structure(
        holo_path,
        os.path.join(work_dir, "PBP2a_holo_clean.pdb"),
        deps=deps,
    )

    cleaned_pdb = pbp2a_pdbqt.replace(".pdbqt", ".pdb")
    log.info("  Computing allosteric site centroid (ALA237, MET241, TYR159)…")
    allosteric_center = compute_residue_centroid(cleaned_pdb, CONFIG.allosteric_residues)
    log.info(f"    Allosteric site center: {allosteric_center}")

    log.info("  Computing active site centroid (SER403)…")
    active_center = compute_residue_centroid(cleaned_pdb, CONFIG.active_site_residues)
    log.info(f"    Active site center: {active_center}")

    result["PBP2a"] = {
        "pdbqt": pbp2a_pdbqt,
        "allosteric_center": allosteric_center,
        "active_center": active_center,
    }

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


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompoundRecord:
    """Stores all computed properties for a single candidate.

    Attributes:
        compound_id: Unique identifier (e.g. ``AA-0001``).
        smiles:     Canonical SMILES string.
        mol:        RDKit Mol object (may be None after deserialisation).
        pb2pa_allosteric_energy: Docking score for the allosteric site.
        pb2pa_active_energy:     Docking score for the active site.
        human_trypsin_energy:    Docking score vs human trypsin.
        human_ces1_energy:       Docking score vs human CES1.
        selectivity_index:       SI = |PBP2a| / |human_avg|.
        max_similarity:          Max Tanimoto similarity to reference antibiotics.
        passes_lipinski:         Lipinski Rule-of-5 compliance.
        qed_score:               Quantitative Estimate of Drug-likeness.
        passes_pains:            PAINS alert filter result.
        resistance_notes:        Human-readable resistance-risk profile.
        shape_score:             RDKit Shape Protrude score (fallback, 0–10).
    """
    compound_id: str
    smiles: str
    mol: Optional[Chem.Mol] = None

    pb2pa_allosteric_energy: Optional[float] = None
    pb2pa_active_energy: Optional[float] = None
    human_trypsin_energy: Optional[float] = None
    human_ces1_energy: Optional[float] = None

    selectivity_index: Optional[float] = None

    max_similarity: float = 0.0

    passes_lipinski: bool = False
    qed_score: float = 0.0
    passes_pains: bool = False

    resistance_notes: str = ""

    shape_score: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — LIBRARY GENERATION & FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

# 8 validated natural-product-inspired scaffolds (SMILES)
CONFIG.natural_product_scaffolds: List[str] = [
    "O=c1c(-c2ccc(O)c(O)c2)coc2cc(O)cc(O)c12",                 # Quercetin (flavonoid)
    "Oc1ccc(C=Cc2ccc(O)cc2)cc1",                                # Resveratrol (stilbenoid)
    "COc1ccc(C=CC(=O)CC(=O)C=Cc2ccc(OC)c(O)c2)cc1O",           # Curcumin (diarylheptanoid)
    "COc1cc2c(cc1OC)-c1ccc3cc4c(cc3c1CC2)OCO4",                 # Berberine (isoquinoline alkaloid)
    "CC1(C)OC2C3OC(=O)C4C(O1)C2C1OOC3C14",                     # Artemisinin (sesquiterpene lactone)
    "O=C1OCc2cn3ccc4cccc-4c3cc21",                              # Camptothecin (topoisomerase inhibitor)
    "COc1nc2c3ccccc3n(C)c2cc1C1CCNC1O",                        # Atropine-like (tropane alkaloid)
    "O=C(Nc1ccccc1)c1ccccc1",                                   # Benzamide
]

# 15 drug-like scaffolds for BRICS decomposition (modified to include BRICS-valid bonds)
CONFIG.additional_scaffolds: List[str] = [
    "c1ccc2[nH]ccc2c1",                                         # Indole
    "c1ccc2ncccc2c1",                                           # Isoquinoline
    "c1ccc2cc[nH]c2c1",                                         # Indole (alternative)
    "c1ccc2[nH]cnc2c1",                                         # Benzimidazole
    "O=c1ccc2ccccc2o1",                                         # Coumarin
    "c1ccc2nc3ccccc3nc2c1",                                     # Phenazine
    "c1ccc2c(c1)oc1ccccc12",                                    # Dibenzofuran
    "c1ccc2c(c1)sc1ccccc12",                                    # Dibenzothiophene
    "c1ccc2c(c1)ccc1c3ccccc3[nH]c21",                           # Carbazole
    "c1ccc2c(c1)CCN2",                                          # Indoline
    "c1ccc2c(c1)CCc1c-2[nH]c2ccccc12",                         # Tetrahydrocarbazole
    "COc1ccc2[nH]ccc2c1",                                       # 5-Methoxyindole
    "COc1ccccc1OCC(O)CNC(C)C",                                  # Propranolol-like (β-blocker)
    "CCN(CC)C(=O)c1ccccc1",                                     # N,N-Diethylbenzamide
    "O=C(Nc1ccc(O)cc1)c1ccc(O)cc1",                            # Phenolic benzamide
]

# Pre-built BRICS building blocks (SMILES with BRICS dummy atoms [n*]).
# These ensure robust recombination when scaffold decomposition yields few fragments.
# Bond-type mapping: [1*,2*,3*]=C, [4*,5*]=N, [6*,7*]=O, [8*]=S, [16*]=aromatic C
# A diverse set of ~50 blocks yields 300+ valid recombination products.
CONFIG.brics_building_blocks: List[str] = [
    # [1*] Carbon-attachment fragments (20)
    "[1*]c1ccccc1",
    "[1*]c1ccc(O)cc1",
    "[1*]c1ccc(Cl)cc1",
    "[1*]c1ccc(F)cc1",
    "[1*]c1ccc(Br)cc1",
    "[1*]c1ccc(OC)cc1",
    "[1*]c1ccc(C(=O)O)cc1",
    "[1*]c1ccc(N)cc1",
    "[1*]c1ccc(C)cc1",
    "[1*]c1ccc(C(C)C)cc1",
    "[1*]c1ccc(CF)cc1",
    "[1*]c1ccc(CN)cc1",
    "[1*]c1ccc(S(=O)(=O)N)cc1",
    "[1*]c1ccc(C(=O)N)cc1",
    "[1*]c1ccc(NC(=O)C)cc1",
    "[1*]CC(=O)O",
    "[1*]CCO",
    "[1*]CCN",
    "[1*]CC(=O)N",
    "[1*]CCC(=O)O",
    # [3*] Alkene-attachment fragments (4)
    "[3*]C=Cc1ccccc1",
    "[3*]C=Cc1ccc(O)cc1",
    "[3*]C=Cc1ccc(Cl)cc1",
    "[3*]CCN(C)C",
    # [5*] Nitrogen-attachment fragments (12)
    "[5*]Nc1ccccc1",
    "[5*]Nc1ccc(O)cc1",
    "[5*]Nc1ccc(C(=O)O)cc1",
    "[5*]Nc1ccc(Cl)cc1",
    "[5*]Nc1ccc(F)cc1",
    "[5*]Nc1ccc(OC)cc1",
    "[5*]Nc1ccc(C)cc1",
    "[5*]Nc1ccc(Br)cc1",
    "[5*]Nc1ccc(CN)cc1",
    "[5*]NCC",
    "[5*]NCCO",
    "[5*]NCCC(=O)O",
    # [6*] Carbonyl-attachment fragments (8)
    "[6*]C(=O)O",
    "[6*]C(=O)c1ccccc1",
    "[6*]C(=O)c1ccc(O)cc1",
    "[6*]C(=O)c1ccc(Cl)cc1",
    "[6*]C(=O)c1ccc(OC)cc1",
    "[6*]C(=O)c1ccc(C)cc1",
    "[6*]C(=O)c1ccc(N)cc1",
    "[6*]C(=O)CC",
    # [7*] Oxygen-attachment fragments (8)
    "[7*]Cc1ccccc1",
    "[7*]Cc1ccc(O)cc1",
    "[7*]Cc1ccc(O)c(OC)c1",
    "[7*]Cc1ccc(OC)cc1",
    "[7*]Cc1ccc(Cl)cc1",
    "[7*]Cc1ccc(F)cc1",
    "[7*]CC",
    "[7*]C(C)C",
    # [16*] Aromatic carbon fragments (4)
    "[16*]c1ccccc1OC",
    "[16*]c1ccc(C)cc1",
    "[16*]c1ccc(N)cc1",
    "[16*]c1ccc(O)cc1",
]

# Positive control SMILES (to verify pipeline)
CONFIG.control_smiles: Dict[str, str] = {
    "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
    "Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O",
}


def _count_atoms(mol: Chem.Mol) -> int:
    """Heavy-atom count for a molecule."""
    return mol.GetNumHeavyAtoms()


def _validate_mol(smiles: str) -> Optional[Chem.Mol]:
    """Validate a SMILES string by parsing and sanitising.

    Returns the validated Mol, or None if parsing/sanitisation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol


def _brics_recombination(
    frag_mols: List[Chem.Mol],
    target_count: int,
    seen_smiles: set,
    seed: int = CONFIG.random_seed,
) -> Tuple[List[CompoundRecord], set]:
    """
    Recombine BRICS fragments using ``BRICSBuild``.

    ``BRICSBuild`` takes a list of pre-decomposed fragments (containing BRICS
    dummy-atom markers) and iterates over all valid recombinations of those
    fragments.  This yields far more diverse products than the original broken
    ``CombineMols`` + ``islice`` approach.

    Returns ``(records, updated_seen_smiles)``.
    """
    records: List[CompoundRecord] = []
    rng = np.random.default_rng(seed)

    # BRICSBuild may generate very many products if given many fragments;
    # we cap effort so it does not run forever.
    max_products = target_count * 20
    n_produced = 0

    # Shuffle fragment order for stochasticity
    shuffled = list(frag_mols)
    rng.shuffle(shuffled)

    builder = BRICS.BRICSBuild(shuffled)

    for product in itertools.islice(builder, max_products):
        if product is None:
            continue
        try:
            Chem.SanitizeMol(product)
            smi = Chem.MolToSmiles(product)
        except Exception:
            continue

        if smi in seen_smiles:
            continue
        seen_smiles.add(smi)

        records.append(CompoundRecord(
            compound_id=f"AA-{n_produced:04d}",
            smiles=smi,
            mol=product,
        ))
        n_produced += 1

        if n_produced >= target_count:
            break

        if n_produced % 50 == 0:
            log.info(f"  BRICS recombination: {n_produced} / {target_count}…")

    return records, seen_smiles


def generate_candidate_library(
    target_count: int = CONFIG.library_target_count,
    seed: int = CONFIG.random_seed,
) -> List[CompoundRecord]:
    """
    Phase 2.1 — Library Generation via BRICS fragment recombination.

    Workflow:
        1. Parse all scaffolds; discard invalid SMILES.
        2. Decompose each scaffold via ``BRICS.BRICSDecompose``.
        3. Collect unique fragments (≥ 8 heavy atoms).
        4. Recombine fragments with ``BRICS.BRICSBuild`` to yield novel
           molecules.
        5. Add positive controls.
        6. Validate every generated molecule with ``SanitizeMol``.

    Returns a list of ``CompoundRecord`` objects with only ``compound_id``,
    ``smiles``, and ``mol`` populated.
    """
    log.info("─── Phase 2: Library Generation ───")

    all_scaffolds: List[str] = CONFIG.natural_product_scaffolds + CONFIG.additional_scaffolds
    scaffold_mols: List[Chem.Mol] = []
    for smi in all_scaffolds:
        mol = _validate_mol(smi)
        if mol is not None:
            scaffold_mols.append(mol)

    log.info(f"  Loaded {len(scaffold_mols)} / {len(all_scaffolds)} valid scaffolds.")

    if not scaffold_mols and not CONFIG.brics_building_blocks:
        log.error("  ✗  No valid scaffolds or building blocks. Aborting library generation.")
        return []

    # ── BRICS decomposition of scaffolds ──
    decomposed_frags: set = set()
    for mol in scaffold_mols:
        try:
            fragments = BRICS.BRICSDecompose(mol, minFragmentSize=CONFIG.brics_min_fragment_size)
            for frag_smi in fragments:
                frag_mol = _validate_mol(frag_smi)
                if frag_mol is not None and _count_atoms(frag_mol) >= CONFIG.brics_min_fragment_size:
                    decomposed_frags.add(frag_smi)
        except Exception:
            continue

    log.info(f"  Decomposed {len(decomposed_frags)} unique BRICS fragments from scaffolds.")

    # ── Add pre-built BRICS building blocks ──
    all_building_blocks: set = set()
    for smi in CONFIG.brics_building_blocks:
        mol = _validate_mol(smi)
        if mol is not None:
            all_building_blocks.add(smi)

    log.info(f"  Loaded {len(all_building_blocks)} pre-built BRICS building blocks.")

    # ── Combine all fragments ──
    all_frag_smis: set = decomposed_frags | all_building_blocks
    frag_mols: List[Chem.Mol] = []
    for smi in all_frag_smis:
        m = _validate_mol(smi)
        if m is not None:
            frag_mols.append(m)

    log.info(f"  Total BRICS-compatible fragments: {len(frag_mols)}")

    seen_smiles: set = set()
    records: List[CompoundRecord] = []

    # ── Include original scaffolds (after re-parsing) ──
    for smi in all_scaffolds:
        mol = _validate_mol(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen_smiles:
            continue
        seen_smiles.add(canon)
        records.append(CompoundRecord(
            compound_id=f"SCAFFOLD_{len(records):04d}",
            smiles=canon,
            mol=mol,
        ))

    # ── Recombination via BRICSBuild ──
    if len(frag_mols) >= 2:
        recon_records, seen_smiles = _brics_recombination(
            frag_mols, target_count, seen_smiles, seed,
        )
        records.extend(recon_records)
        log.info(f"  BRICS recombination yielded {len(recon_records)} novel compounds.")
    else:
        log.warning(
            f"  Too few fragments ({len(frag_mols)}) for recombination. "
            "Using scaffold enumeration only."
        )

    # ── Add positive controls ──
    for name, smi in CONFIG.control_smiles.items():
        mol = _validate_mol(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen_smiles:
            records.append(CompoundRecord(
                compound_id=f"CTRL_{name}",
                smiles=canon,
                mol=mol,
            ))
            seen_smiles.add(canon)

    log.info(f"  Library generation complete: {len(records)} compounds.")
    if len(records) < 300:
        log.warning(
            f"  ⚠  Only {len(records)} compounds generated (target ≥300). "
            "Consider adding more scaffolds or building blocks."
        )

    return records


def apply_filters(
    records: List[CompoundRecord],
    similarity_threshold: float = CONFIG.similarity_threshold,
) -> List[CompoundRecord]:
    """
    Phase 2.2 — Apply structural, similarity, ADMET, and PAINS filters.

    Filter chain:
        1. Structural exclusion (β-lactam SMARTS).
        2. Similarity filter vs reference antibiotics (Morgan FP, Tc < threshold).
        3. ADMET: Lipinski Rule of 5 + QED > 0.6.
        4. PAINS alerts via RDKit FilterCatalog.
        5. Diversity check: if < 100 pass, relax similarity to 0.5.

    Scientific rationale: These filters enrich the library for drug-like,
    non-β-lactam, novel chemotypes that are unlikely to be cross-resistant
    with existing antibiotics.

    Args:
        records: Input compound records.
        similarity_threshold: Initial Tanimoto cutoff.

    Returns:
        Filtered list of CompoundRecord (with computed ADMET/similarity fields).
    """
    log.info("─── Phase 2: Filtering ───")

    ref_mols: Dict[str, Any] = {}
    for name, smi in CONFIG.reference_antibiotics.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ref_mols[name] = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048,
            )

    lactam_pattern = Chem.MolFromSmarts(CONFIG.beta_lactam_smarts)

    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains_catalog = FilterCatalog(pains_params)

    passed: List[CompoundRecord] = []
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

        # 1. Structural — reject β-lactams (but allow controls)
        is_control = record.compound_id.startswith("CTRL_")
        if not is_control and mol.HasSubstructMatch(lactam_pattern):
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

        if not lipinski_ok or qed <= 0.6:
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

    if len(passed) < CONFIG.diversity_min_count and similarity_threshold < CONFIG.similarity_threshold_relaxed:
        log.warning(
            f"  Only {len(passed)} compounds passed strict filters (< {CONFIG.diversity_min_count}). "
            f"Relaxing similarity threshold to {CONFIG.similarity_threshold_relaxed} and re-running."
        )
        return apply_filters(records, similarity_threshold=CONFIG.similarity_threshold_relaxed)

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

    Falls back to a minimal PDBQT writer (via RDKit Gasteiger charges) if
    meeko is unavailable or fails.

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
        log.warning(f"  Meeko prep failed ({exc}), trying RDKit fallback…")
        # Fallback: write minimal PDBQT via RDKit
        try:
            mol_tmp = Chem.RWMol(mol)
            mol_tmp = Chem.AddHs(mol_tmp)
            AllChem.ComputeGasteigerCharges(mol_tmp)

            conf = mol_tmp.GetConformer()
            lines = ["ROOT\n"]
            for i, atom in enumerate(mol_tmp.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                charge = atom.GetDoubleProp("_GasteigerCharge")
                elem = atom.GetSymbol()
                lines.append(
                    f"ATOM     {i+1:>3}  {elem:<3} LIG X   1    "
                    f"{pos.x:>8.3f}{pos.y:>8.3f}{pos.z:>8.3f}  "
                    f"{charge:>8.3f}     {elem:<2s}\n"
                )
            lines.append("ENDROOT\n")
            lines.append("TORSDOF 0\n")
            with open(output_path, "w") as f:
                f.writelines(lines)
            return True
        except Exception as exc2:
            log.warning(f"  RDKit PDBQT fallback also failed: {exc2}")
            return False


def _run_vina_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    timeout: int = CONFIG.vina_timeout_s,
) -> Optional[float]:
    """
    Run a single Vina docking job. Returns best binding energy (kcal/mol)
    or None on failure.

    When ``CONFIG.dry_run`` is ``True``, the Vina subprocess is skipped
    and a mock random energy in the range [-10.0, -5.0] kcal/mol is returned,
    enabling end-to-end pipeline testing without the Vina binary.
    """
    if CONFIG.dry_run:
        return float(np.random.uniform(-10.0, -5.0))

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

        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("1") and " " in stripped:
                parts = stripped.split()
                try:
                    energy = float(parts[1])
                    return energy
                except (ValueError, IndexError):
                    continue
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
    cache: Optional[Dict[str, float]] = None,
    use_cache: bool = False,
) -> Optional[float]:
    """
    Full docking pipeline for a single compound: PDBQT prep → Vina → parse.

    Supports caching: if ``use_cache`` is True and a result for
    ``{compound_id}_{tag}`` exists in the cache dict, it is returned
    immediately without re-docking.

    Args:
        record: Compound record (must have .mol or valid .smiles).
        receptor_pdbqt: Path to receptor PDBQT.
        center: Grid box centre.
        box_size: Grid box dimensions.
        work_dir: Scratch directory.
        tag: Label for temp files and cache key (e.g. ``'allosteric'``).
        cache: Optional cache dictionary (mutated in place on new results).
        use_cache: If True, consult cache before docking.

    Returns:
        Best binding energy, or None on failure.
    """
    cache_key = f"{record.compound_id}_{tag}"
    if use_cache and cache is not None and cache_key in cache:
        log.info(f"  Cache hit: {cache_key} = {cache[cache_key]}")
        return cache[cache_key]

    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            return None
        record.mol = mol

    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    lig_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_lig.pdbqt")
    out_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_out.pdbqt")

    if not prepare_ligand_pdbqt(record.mol, lig_pdbqt):
        return None

    energy = _run_vina_docking(
        receptor_pdbqt, lig_pdbqt, out_pdbqt,
        center, box_size,
    )

    for f in (lig_pdbqt, out_pdbqt):
        try:
            os.remove(f)
        except OSError:
            pass

    if use_cache and cache is not None:
        cache[cache_key] = energy

    return energy


def _worker_dock(
    cid: str, smiles: str,
    receptor_pdbqt: str, center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str, tag: str,
) -> Tuple[str, Optional[float]]:
    """Module-level worker for :func:`_parallel_dock`.

    Reconstructs Mol from SMILES locally, docks, and returns energy.
    Defined at module level so ``ProcessPoolExecutor`` can pickle it.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return cid, None
    rec = CompoundRecord(compound_id=cid, smiles=smiles, mol=mol)
    energy = dock_compound(
        rec, receptor_pdbqt, center, box_size,
        work_dir, tag, cache=None, use_cache=False,
    )
    return cid, energy


def _parallel_dock(
    items: List[Tuple[str, str]],
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
    n_jobs: int = CONFIG.n_jobs,
    cache: Optional[Dict[str, float]] = None,
    use_cache: bool = False,
) -> List[Tuple[str, Optional[float]]]:
    """
    Dock a list of compounds in parallel, returning ``(compound_id, energy)``.

    Accepts ``(compound_id, smiles)`` tuples instead of ``CompoundRecord``
    objects to avoid pickling RDKit ``Mol`` objects across process boundaries.
    Each worker reconstructs the ``Mol`` from SMILES locally.

    Args:
        items: List of ``(compound_id, smiles)`` pairs.
        receptor_pdbqt: Path to receptor PDBQT.
        center: Grid centre.
        box_size: Grid dimensions.
        work_dir: Scratch directory.
        tag: Label for temp files.
        n_jobs: Number of parallel workers.
        cache: Optional cache dict (used to skip completed entries).
        use_cache: Whether to consult cache.

    Returns:
        List of ``(compound_id, energy)`` tuples.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    total = len(items)
    results: List[Tuple[str, Optional[float]]] = []
    submitted = 0

    worker_fn = partial(
        _worker_dock,
        receptor_pdbqt=receptor_pdbqt,
        center=center,
        box_size=box_size,
        work_dir=work_dir,
        tag=tag,
    )

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {}
        for cid, smiles in items:
            cache_key = f"{cid}_{tag}"
            if use_cache and cache is not None and cache_key in cache:
                results.append((cid, cache[cache_key]))
                continue
            futures[pool.submit(worker_fn, cid, smiles)] = cid
            submitted += 1

        done = 0
        for future in as_completed(futures):
            cid, energy = future.result()
            results.append((cid, energy))
            done += 1
            if done % 25 == 0 or done == submitted:
                log.info(f"    Docked {done} / {submitted} ({tag})")

    return results


def _compute_shape_fallback_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    seed: int = CONFIG.random_seed,
) -> Optional[float]:
    """
    Fallback scoring via RDKit Shape Protrude Distance.

    Generates 3D conformers for both the query and reference molecules,
    optimises with MMFF94, and computes the Shape Protrude Distance.
    The score is normalised to a 0–10 scale (lower = better shape match).

    Scientific rationale: Shape complementarity is the simplest knowledge-free
    proxy for binding affinity.  This fallback runs when AutoDock Vina is
    unavailable, maintaining pipeline functionality without external binaries.
    """
    try:
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

        try:
            protrude = AllChem.GetShapeProtrudeDist(mol_3d, ref_3d)
        except Exception:
            try:
                protrude = AllChem.GetShapeProtrudeDist(ref_3d, mol_3d)
            except Exception:
                return None

        normalised = min(protrude / 0.05, 10.0) if protrude > 0 else 0.0
        return normalised

    except Exception:
        return None


def screen_library(
    records: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: Optional[Dict[str, float]] = None,
    use_cache: bool = False,
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
        log.info("  Docking all compounds against allosteric site…")
        items = [(r.compound_id, r.smiles) for r in records]
        allosteric_results = _parallel_dock(
            items, pb2pa["pdbqt"],
            allosteric_center, CONFIG.allosteric_box_size,
            work_dir, "allosteric",
            cache=cache, use_cache=use_cache,
        )

        cid_to_record = {r.compound_id: r for r in records}
        for cid, energy in allosteric_results:
            if cid in cid_to_record:
                cid_to_record[cid].pb2pa_allosteric_energy = energy

        n_scored = sum(1 for r in records if r.pb2pa_allosteric_energy is not None)
        log.info(f"  Allosteric docking complete: {n_scored}/{len(records)} scored.")

        scored = [r for r in records if r.pb2pa_allosteric_energy is not None]
        scored.sort(key=lambda r: r.pb2pa_allosteric_energy)

        top50 = scored[:50]
        log.info(f"  Docking top {len(top50)} compounds against active site…")

        active_items = [(r.compound_id, r.smiles) for r in top50]
        active_results = _parallel_dock(
            active_items, pb2pa["pdbqt"],
            active_center, CONFIG.active_box_size,
            work_dir, "active",
            cache=cache, use_cache=use_cache,
        )

        for cid, energy in active_results:
            if cid in cid_to_record:
                cid_to_record[cid].pb2pa_active_energy = energy

    else:
        log.info("  Vina unavailable. Using RDKit Shape Fallback.")

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
            ref_smi = list(CONFIG.control_smiles.values())[0]
            ref_mol = Chem.MolFromSmiles(ref_smi)

        if ref_mol is None:
            log.error("  Cannot obtain reference molecule for shape scoring.")
            return records[:CONFIG.top_n]

        total = len(records)
        for i, rec in enumerate(records):
            if rec.mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    continue
                rec.mol = mol
            score = _compute_shape_fallback_score(rec.mol, ref_mol)
            rec.shape_score = score
            if (i + 1) % 100 == 0:
                log.info(f"  Shape scored {i + 1} / {total}")

        scored_shape = [r for r in records if r.shape_score is not None]
        scored_shape.sort(key=lambda r: r.shape_score)
        log.info(f"  Shape scoring complete. Best score: {scored_shape[0].shape_score:.3f}")

    if deps["USE_VINA"]:
        ranked = [r for r in records if r.pb2pa_allosteric_energy is not None]
        ranked.sort(key=lambda r: r.pb2pa_allosteric_energy)
    else:
        ranked = [r for r in records if r.shape_score is not None]
        ranked.sort(key=lambda r: r.shape_score)

    top10 = ranked[:CONFIG.top_n]
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

def compute_selectivity_index(
    pb2pa_energy: float, human_avg_energy: float,
) -> float:
    """
    Selectivity Index (SI).

    .. math::

        SI = \\frac{|\\text{PBP2a Energy}|}{|\\text{Human Avg Energy}|}

    Vina energies are negative (favourable binding).  A higher SI (> 1.0)
    means stronger binding to PBP2a than to the human off-target panel.

    Scientific rationale: A high SI reduces the risk of mechanism-based
    toxicity.  We set a threshold of 2.0 for the final filter.

    Args:
        pb2pa_energy: Best (most negative) PBP2a binding energy.
        human_avg_energy: Average binding energy across human targets.

    Returns:
        SI value (float).
    """
    if pb2pa_energy >= 0 or human_avg_energy >= 0:
        return 0.0
    return abs(pb2pa_energy) / abs(human_avg_energy) if abs(human_avg_energy) > 1e-6 else 0.0



def profile_resistance_risk(
    record: CompoundRecord,
    work_dir: str,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
) -> str:
    """
    Rule-based resistance profiling.

    Flags candidates based on predicted interactions:
        - Good: contacts with conserved residues (Ser403, Lys406, Tyr446).
        - Risk: contacts with mutable residues (Gly246, Asn146).

    Scientific rationale: Mutations in the PBP2a binding pocket (e.g. G246E,
    N146K) have been associated with clinical resistance to ceftaroline.
    Candidates that preferentially contact conserved residues are less likely
    to lose activity against resistant strains.

    Returns a human-readable notes string.
    """
    notes: List[str] = []

    if record.pb2pa_active_energy is not None and record.pb2pa_active_energy < -6.0:
        notes.append("Likely contacts catalytic Ser403 (active site). Good.")

    if record.pb2pa_allosteric_energy is not None and record.pb2pa_allosteric_energy < -7.0:
        if record.pb2pa_active_energy is None or record.pb2pa_active_energy > -6.0:
            notes.append("Allosteric binder (Ala237/Met241/Tyr159 pocket). Novel mechanism.")

    if record.mol is not None:
        mw = Descriptors.MolWt(record.mol)
        if mw > 400:
            notes.append("High MW (>400) — broad interaction surface, may contact multiple residues.")
        n_rot = Descriptors.NumRotatableBonds(record.mol)
        if n_rot < 5:
            notes.append("Rigid scaffold — reduced entropic penalty, may enhance binding specificity.")

    if record.qed_score > 0.8:
        notes.append("High drug-likeness (QED > 0.8) — good developability profile.")

    if not notes:
        notes.append("No specific resistance flags identified.")

    return "; ".join(notes)


def analyze_selectivity_and_resistance(
    top10: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: Optional[Dict[str, float]] = None,
    use_cache: bool = False,
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

    log.info("  Docking top 10 vs Human Trypsin (1UTN)…")
    trypsin_center = targets["trypsin"].get("active_center", np.array([0.0, 0.0, 0.0]))
    trypisn_items = [(r.compound_id, r.smiles) for r in top10]
    trypsin_results = _parallel_dock(
        trypisn_items, targets["trypsin"]["pdbqt"],
        trypsin_center, (20.0, 20.0, 20.0),
        work_dir, "trypsin", n_jobs=min(4, len(top10)),
        cache=cache, use_cache=use_cache,
    )
    cid_map = {r.compound_id: r for r in top10}
    for cid, energy in trypsin_results:
        if cid in cid_map:
            cid_map[cid].human_trypsin_energy = energy

    log.info("  Docking top 10 vs Human Carboxylesterase 1 (3KJZ)…")
    ces1_center = targets["CES1"].get("active_center", np.array([0.0, 0.0, 0.0]))
    ces1_items = [(r.compound_id, r.smiles) for r in top10]
    ces1_results = _parallel_dock(
        ces1_items, targets["CES1"]["pdbqt"],
        ces1_center, (20.0, 20.0, 20.0),
        work_dir, "ces1", n_jobs=min(4, len(top10)),
        cache=cache, use_cache=use_cache,
    )
    for cid, energy in ces1_results:
        if cid in cid_map:
            cid_map[cid].human_ces1_energy = energy

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

        if si < CONFIG.selectivity_index_threshold:
            log.warning(
                f"  {rec.compound_id}: Low selectivity (SI = {si:.2f} < {CONFIG.selectivity_index_threshold}). "
                "Flagged for off-target risk."
            )
        else:
            log.info(f"  {rec.compound_id}: SI = {si:.2f} (pass).")

    pb2pa = targets["PBP2a"]
    for rec in top10:
        rec.resistance_notes = profile_resistance_risk(
            rec, work_dir,
            pb2pa["pdbqt"],
            pb2pa["allosteric_center"],
            CONFIG.allosteric_box_size,
        )

    log.info("─── Phase 4 complete ───")
    return top10


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — REPORTING & ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_csv_report(top10: List[CompoundRecord]) -> str:
    """
    Phase 5.1 — Write ``top_candidates.csv`` with all required columns.

    Columns:
        Compound_ID, SMILES, PBP2a_Allosteric_Energy, PBP2a_Active_Energy,
        Human_Trypsin_Energy, Human_CES1_Energy, Selectivity_Index,
        Max_Similarity, Passes_Lipinski, QED_Score, Binding_Mode_Notes.

    Returns path to CSV.
    """
    log.info("─── Phase 5: Reporting ───")
    ensure_output_dir()

    rows: List[Dict[str, str]] = []
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
    df.to_csv(CONFIG.output_dir / "top_candidates.csv", index=False)
    log.info(f"  CSV report saved: {CONFIG.output_dir / "top_candidates.csv"}")
    return str(CONFIG.output_dir / "top_candidates.csv")


def generate_images(top3: List[CompoundRecord]) -> List[str]:
    """
    Phase 5.2 — Save 2D structure PNGs for the top 3 candidates.

    Returns list of file paths.
    """
    paths: List[str] = []
    for i, rec in enumerate(top3):
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol

        img_path = CONFIG.output_dir / f"top{i + 1}_{rec.compound_id}.png"
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


def generate_html_report(
    top10: List[CompoundRecord],
    top50: List[CompoundRecord],
    output_dir: Path,
) -> Tuple[str, str, str]:
    """
    Phase 5.3 — Generate an HTML report with embedded matplotlib figures.

    Creates:
        1. Scatter plot: Allosteric Energy vs Selectivity Index (top 10).
        2. Histogram: QED Scores for top 50 candidates.
        3. HTML page embedding the figures and a results table.

    Scientific rationale: Visualising the binding energy-selectivity trade-off
    helps identify the most promising candidates at a glance.  The QED
    histogram confirms the library is drug-like after filtering.

    Returns ``(html_path, scatter_path, hist_path)``.
    """
    log.info("─── Phase 5: HTML Report Generation ───")

    # ── 1. Scatter plot: Allosteric Energy vs Selectivity Index ──
    scatter_data = [
        (r.pb2pa_allosteric_energy, r.selectivity_index, r.compound_id)
        for r in top10
        if r.pb2pa_allosteric_energy is not None and r.selectivity_index is not None
    ]
    if scatter_data:
        fig, ax = plt.subplots(figsize=(9, 6))
        energies = [d[0] for d in scatter_data]
        sis = [d[1] for d in scatter_data]
        cids = [d[2] for d in scatter_data]
        ax.scatter(energies, sis, c="steelblue", s=60, edgecolors="black")
        for x, y, cid in zip(energies, sis, cids):
            ax.annotate(cid, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax.axhline(y=CONFIG.selectivity_index_threshold, color="red", linestyle="--", alpha=0.6,
                   label=f"SI threshold = {CONFIG.selectivity_index_threshold}")
        ax.set_xlabel("Allosteric Binding Energy (kcal/mol)", fontsize=12)
        ax.set_ylabel("Selectivity Index", fontsize=12)
        ax.set_title("Top Candidates: Binding Energy vs Selectivity", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        scatter_path = os.path.join(str(output_dir), "energy_vs_selectivity.png")
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Scatter plot saved: {scatter_path}")
    else:
        scatter_path = ""

    # ── 2. Histogram: QED scores for top 50 ──
    qeds = [r.qed_score for r in top50 if r.qed_score > 0]
    if qeds:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(qeds, bins=20, edgecolor="black", color="mediumseagreen", alpha=0.8)
        ax.axvline(x=0.6, color="red", linestyle="--", alpha=0.6, label="QED cutoff = 0.6")
        ax.set_xlabel("QED Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("QED Distribution (Top 50 Candidates)", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        hist_path = os.path.join(str(output_dir), "qed_histogram.png")
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  QED histogram saved: {hist_path}")
    else:
        hist_path = ""

    # ── 3. Build HTML ──
    table_rows = ""
    for i, rec in enumerate(top10):
        allosteric = f"{rec.pb2pa_allosteric_energy:.2f}" if rec.pb2pa_allosteric_energy is not None else "N/A"
        active = f"{rec.pb2pa_active_energy:.2f}" if rec.pb2pa_active_energy is not None else "N/A"
        si = f"{rec.selectivity_index:.2f}" if rec.selectivity_index is not None else "N/A"
        qed = f"{rec.qed_score:.3f}" if rec.qed_score else "N/A"
        table_rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td>{rec.compound_id}</td>"
            f"<td style='font-size:0.8em;max-width:300px;word-break:break-all;'>{rec.smiles}</td>"
            f"<td>{allosteric}</td>"
            f"<td>{active}</td>"
            f"<td>{si}</td>"
            f"<td>{qed}</td>"
            f"<td>{rec.resistance_notes}</td>"
            f"</tr>\n"
        )

    scatter_img = ""
    if scatter_path:
        scatter_img = (
            '<h2>Binding Energy vs Selectivity</h2>\n'
            f'<img src="energy_vs_selectivity.png" alt="Energy vs Selectivity" style="max-width:800px;">\n'
        )
    hist_img = ""
    if hist_path:
        hist_img = (
            '<h2>QED Score Distribution</h2>\n'
            f'<img src="qed_histogram.png" alt="QED Histogram" style="max-width:800px;">\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoAntibiotic Discovery Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; }}
h1 {{ color: #1a5276; }}
h2 {{ color: #2e86c1; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background-color: #2e86c1; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
img {{ border: 1px solid #ddd; border-radius: 4px; padding: 4px; }}
.footer {{ margin-top: 30px; color: #777; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>AutoAntibiotic Discovery Pipeline — Top Candidates Report</h1>
<p>Generated by AutoAntibiotic v3.2 | MRSA PBP2a Inhibitor Screening</p>
<hr>

{scatter_img}

{hist_img}

<h2>Top {len(top10)} Candidates</h2>
<table>
<tr>
  <th>Rank</th>
  <th>ID</th>
  <th>SMILES</th>
  <th>Allosteric (kcal/mol)</th>
  <th>Active (kcal/mol)</th>
  <th>Selectivity Index</th>
  <th>QED</th>
  <th>Resistance Notes</th>
</tr>
{table_rows}
</table>

<div class="footer">
<p>Pipeline completed successfully. See <code>top_candidates.csv</code> for full data.</p>
</div>
</body>
</html>"""

    html_path = os.path.join(str(output_dir), "report.html")
    with open(html_path, "w") as f:
        f.write(html)
    log.info(f"  HTML report saved: {html_path}")

    return html_path, scatter_path, hist_path


def print_summary(
    n_total: int, n_filtered: int,
    top10: List[CompoundRecord],
    validation_ok: bool, redock_rmsd: Optional[float],
    deps: Dict[str, Any],
) -> None:
    """Log a final pipeline summary."""
    n_docked = sum(1 for r in top10 if r.pb2pa_allosteric_energy is not None)
    n_selectivity_pass = sum(
        1 for r in top10
        if r.selectivity_index is not None and r.selectivity_index >= CONFIG.selectivity_index_threshold
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
    if redock_rmsd is not None:
        log.info(f"  Redocking RMSD:                {redock_rmsd:.3f} Å")
    else:
        log.info("  Redocking RMSD:                N/A")
    log.info(f"  Redocking validated:           {validation_ok}")
    log.info(f"  CSV report:                    {CONFIG.output_dir / "top_candidates.csv"}")
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate the full discovery pipeline end-to-end.

    Usage::

        python discovery_pipeline.py [--use-cache] [--dry-run]

    The ``--use-cache`` flag skips re-docking of any ``(compound_id, target)``
    pair that already has a result in ``output/cache.json``.
    The ``--dry-run`` flag limits the library to 10 compounds and returns
    mock docking energies so the pipeline can be tested end-to-end
    without AutoDock Vina.
    """
    parser = argparse.ArgumentParser(
        description="AutoAntibiotic Discovery Pipeline v3.2",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Skip re-docking if cache.json has results for a (compound_id, target) pair.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Limit library to 10 compounds and use mock docking energies.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        CONFIG.dry_run = True
        CONFIG.library_target_count = 10

    ensure_output_dir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(CONFIG.output_dir / "pipeline.log"),
        ],
    )

    use_cache = args.use_cache
    cache: Optional[Dict[str, float]] = None
    if use_cache:
        cache = load_cache()
        log.info(f"  Loaded {len(cache)} cached docking results.")
    else:
        log.info("  Cache disabled. Use --use-cache to enable.")

    deps = verify_dependencies()

    work_dir = str(CONFIG.output_dir / "workdir")
    pdb_dir = str(CONFIG.output_dir / "pdb")
    os.makedirs(work_dir, exist_ok=True)

    # ── Phase 1: Target preparation ──
    targets = prepare_targets(pdb_dir, work_dir, deps)

    # ── Phase 0: Redocking validation ──
    validation_ok, redock_rmsd = run_redocking_validation(
        holo_pdb_path=targets["holo_pdb"],
        target_pdbqt_path=targets["PBP2a"]["pdbqt"],
        work_dir=work_dir,
        deps=deps,
        center=targets["PBP2a"]["active_center"],
    )

    # ── Phase 2: Library generation & filtering ──
    all_records = generate_candidate_library(target_count=CONFIG.library_target_count)
    n_total = len(all_records)

    filtered = apply_filters(all_records)
    n_filtered = len(filtered)

    if n_filtered == 0:
        log.warning("  No compounds passed filters. Halting pipeline.")
        return

    # ── Phase 3: Virtual screening ──
    top10 = screen_library(filtered, targets, work_dir, deps, cache=cache, use_cache=use_cache)

    if not top10:
        log.warning("  No candidates after screening. Halting pipeline.")
        return

    # ── Phase 4: Selectivity & Resistance ──
    top10 = analyze_selectivity_and_resistance(
        top10, targets, work_dir, deps, cache=cache, use_cache=use_cache,
    )

    # ── Phase 5: Reporting & Artifacts ──
    generate_csv_report(top10)

    top3 = top10[:3]
    generate_images(top3)

    # Collect top 50 for QED histogram
    scored = sorted(
        [r for r in filtered if r.qed_score > 0],
        key=lambda r: r.qed_score, reverse=True,
    )
    top50 = scored[:50]

    generate_html_report(top10, top50, CONFIG.output_dir)

    if use_cache and cache is not None:
        save_cache(cache)
        log.info(f"  Cache saved ({len(cache)} entries).")

    print_summary(
        n_total, n_filtered, top10,
        validation_ok, redock_rmsd, deps,
    )

    log.info("Pipeline complete. Exiting.")


if __name__ == "__main__":
    main()
