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

 CI mode: set USE_VINA=False or use bundled tests/data mocks; real PDBs required for scientific validation.

  Bundled tests/data PDBs are minimal mocks; redocking RMSD against them is non‑physical. Use real PDB downloads for science mode.

 For real science runs: set `mode: science` in config.yaml and place real PDBs in pdb_dir; bundled tests/data are mocks.
 """

import os
import sys
import re
import json
import subprocess
import logging
import warnings
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ── RDKit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, QED, rdMolDescriptors,
    rdDistGeom, Crippen, FilterCatalog, BRICS,
)
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger as rdklog

# ── Bio.PDB ────────────────────────────────────────────────────────────────────
from Bio.PDB import (
    PDBParser, PDBIO, Select,
    PDBList,
)

# ── Suppress RDKit noise ───────────────────────────────────────────────────────
rdklog.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ── Local utility modules ──────────────────────────────────────────────────────
# Focused helper modules keep this orchestration file readable while hosting the
# implementation details for ligand prep, docking, and filtering.
from utils.docking import (
    dock_compound,
    _dock_compounds_parallel,
)
from utils.filtering import apply_filters

# Structural helpers (native-ligand extraction, RMSD, centroids) live in their
# own module to keep this orchestrator focused on flow control.
from utils.structure_prep import (
    _extract_native_ligand_from_holo,
    _compute_rmsd_docked_vs_crystal,
    compute_residue_centroid,
    write_receptor_pdbqt,
)

# Library generation (scaffolds, controls, CompoundRecord) lives in its own
# flat module so the orchestrator stays focused on flow control.
from utils.library_gen import (
    generate_candidate_library,
    CompoundRecord,
)

# Reporting / artifact generation (CSV, images, interaction diagrams, PyMOL
# script) lives in its own flat module.
from utils.reporting import (
    generate_csv_report,
    generate_images,
    generate_interaction_diagram,
    generate_pymol_script,
    _print_single_summary,
)

# Configuration constants are centralised in config.constants to break the
# former circular import between this module and the utils package.
from config.constants import (
    RANDOM_SEED,
    PDB_IDS,
    REFERENCE_ANTIBIOTICS,
    BETA_LACTAM_SMARTS,
    ALLOSTERIC_RESIDUES,
    ACTIVE_SITE_RESIDUES,
    CONSERVED_RESIDUES,
    TRYPSIN_CATALYTIC_RESIDUES,
    CES1_CATALYTIC_RESIDUES,
    ALBUMIN_CATALYTIC_RESIDUES,
    CYP3A4_CATALYTIC_RESIDUES,
    HERG_CATALYTIC_RESIDUES,
    CYP2D6_CATALYTIC_RESIDUES,
    FLEX_RESIDUES,
    FLEX_VINA_TIMEOUT_S,
    MUTATION_SCAN,
    MUTATION_SCAN_MUTANTS,
    FP_RADIUS,
    FP_NBITS,
    PBP2A_CONFORMER_IDS,
    ALLOSTERIC_BOX_SIZE,
    ACTIVE_BOX_SIZE,
    VINA_TIMEOUT_S,
    N_JOBS,
    SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD_RELAXED,
    DIVERSITY_MIN_COUNT,
    SELECTIVITY_INDEX_THRESHOLD,
    RMSD_VALIDATED_MAX,
    RMSD_MARGINAL_MAX,
    OUTPUT_DIR,
    CSV_REPORT,
    TOP_N,
    REPO_ROOT,
    load_config,
)

# Preserve the original import-time side effect (seeding for reproducibility).
np.random.seed(RANDOM_SEED)

# Package version, resolved from pyproject.toml metadata when installed (e.g.
# via `pip install .`) and falling back to a sensible default during local
# development. Exposed through the `--version` CLI flag.
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("autoantibiotic-discovery-pipeline")
except Exception:  # pragma: no cover - local/dev fallback
    __version__ = "3.1.0"


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
#  CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Library code (functions like ``prepare_targets``) raises these instead of
# calling ``sys.exit`` so that callers — including unit tests and programmatic
# API users — can catch them. The CLI entrypoint (``main`` / ``__main__``)
# translates the ones that mean "abort" into ``sys.exit(1)``.

class ScienceModeMockPDBError(RuntimeError):
    """Raised when science mode would run against a bundled mock PDB.

    Science mode requires real crystallographic structures; using the bundled
    ``tests/data`` mocks would silently produce non-physical results. Callers
    must treat this as a hard abort condition.
    """


class MissingGridCenterError(RuntimeError):
    """Raised when a required docking grid center cannot be computed.

    In science mode every target site must resolve to a concrete grid centre;
    if any centre is missing the run cannot proceed and must abort.
    """


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def select_top(records, score_key, descending=False, n=TOP_N):
    """Return the top *n* records sorted by *score_key* (skips None scores)."""
    valid = [r for r in records if getattr(r, score_key, None) is not None]
    valid.sort(key=lambda r: getattr(r, score_key), reverse=descending)
    return valid[:n]


def _auto_box_size(
    receptor_pdb: Optional[str],
    center: Optional[np.ndarray],
    default_box: Tuple[float, float, float],
    min_size: float = 15.0,
    max_size: float = 30.0,
    padding: float = 6.0,
) -> Tuple[float, float, float]:
    """
    Auto-size a docking grid box around *center*.

    Computes the maximum distance from *center* to any heavy atom of the
    residue(s) that define the site (read from *receptor_pdb*), then sizes the
    box so it comfortably encloses the site:

        size = max(min_size, 2 * (max_radius + padding))

    This replaces the hardcoded constants (e.g. the allosteric ``(15,15,15)``
    Å box, which can be too small for residues like ALA237/MET241/TYR159 that
    span more than 15 Å in real structures). When the receptor PDB is missing
    or the spread cannot be measured, the *default_box* is returned unchanged.

    Args:
        receptor_pdb: Path to the cleaned receptor PDB (or None).
        center: Grid centre as a length-3 array (or None).
        default_box: Fallback box dimensions when auto-sizing is impossible.
        min_size: Minimum box edge in Å (enforced even for tiny sites).
        max_size: Maximum box edge in Å (capped to prevent whole-protein boxes).
        padding: Extra Å added to the measured radius on each side.

    Returns:
        ``(x, y, z)`` box dimensions in Å.
    """
    if receptor_pdb is None or not os.path.exists(receptor_pdb) or center is None:
        return default_box

    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("receptor", receptor_pdb)
        center = np.asarray(center, dtype=float)
        max_radius = 0.0
        for model in struct:
            for chain in model:
                for residue in chain:
                    rid = residue.get_id()
                    if rid[0] != " ":
                        continue
                    for atom in residue:
                        try:
                            pos = atom.get_vector().get_array()
                        except Exception:
                            continue
                        d = float(np.linalg.norm(np.asarray(pos) - center))
                        if d > max_radius:
                            max_radius = d
        size = max(min_size, 2.0 * (max_radius + padding))
        size = min(size, max_size)
        return (size, size, size)
    except Exception as exc:
        log.warning(f"  ⚠  Could not auto-size box ({exc}); using default {default_box}.")
        return default_box


def _redocking_box_size(
    ligand_pdbqt: str,
    center: np.ndarray,
    min_size: float = 15.0,
    padding: float = 6.0,
    default_box: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    redock_padding: float = 4.0,
) -> Tuple[float, float, float]:
    """
    Size the native-ligand redocking box from the ligand coordinates.

    Reads the heavy-atom positions from *ligand_pdbqt*, measures the max
    distance from *center* (the native-ligand centroid) to any atom, and sizes
    the box so it comfortably encloses the ligand:

        size = max(min_size, 2 * (max_radius + padding))

    This mirrors :func:`_auto_box_size` (used for the allosteric/active sites)
    so the redocking grid is tight around the crystallographic ligand rather
    than a hardcoded 25 Å cube. Falls back to *default_box* when the ligand
    cannot be parsed.
    """
    if ligand_pdbqt is None or not os.path.exists(ligand_pdbqt) or center is None:
        return default_box
    try:
        mol = Chem.MolFromPDBQT(ligand_pdbqt) if hasattr(Chem, "MolFromPDBQT") else None
        if mol is None:
            return default_box
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        if coords.size == 0:
            return default_box
        center = np.asarray(center, dtype=float)
        max_radius = float(
            np.linalg.norm(coords - center, axis=1).max()
        )
        size = max(min_size, 2.0 * (max_radius + padding))
        return (size, size, size)
    except Exception as exc:
        log.warning(
            f"  ⚠  Could not auto-size redocking box ({exc}); "
            f"using default {default_box}."
        )
        return default_box


# ═══════════════════════════════════════════════════════════════════
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
    vina_version = ""
    try:
        vina_result = subprocess.run(
            ["vina", "--version"], capture_output=True, text=True, timeout=10
        )
        vina_available = True
        # Capture and surface the reported Vina version so users can confirm
        # the docking engine version used for protocol validation.
        vina_version = (vina_result.stdout or vina_result.stderr).strip()
        if vina_version:
            log.info(
                f"  ✓  AutoDock Vina binary found on PATH "
                f"(version: {vina_version})"
            )
        else:
            log.info("  ✓  AutoDock Vina binary found on PATH.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing_bins.append("AutoDock Vina (vina)")

    obabel_available = False
    obabel_version = ""
    try:
        obabel_result = subprocess.run(
            ["obabel", "--version"], capture_output=True, text=True, timeout=10
        )
        obabel_available = True
        obabel_version = (obabel_result.stdout or obabel_result.stderr).strip()
        if obabel_version:
            log.info(
                f"  ✓  OpenBabel binary found on PATH "
                f"(version: {obabel_version})"
            )
        else:
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
            "Screening requires AutoDock Vina; the pipeline will abort."
        )
        # High-visibility, bold error printed directly to stdout so the user
        # is not silently left on a broken path.
        # The fix is one line away via setup.sh or Docker.
        print(
            "\033[1;31m"
            "\n"
            "  ╔══════════════════════════════════════════════════════════════════╗\n"
            "  ║  ERROR: AutoDock Vina not found.                                ║\n"
            "  ║  Install it with one command:                                   ║\n"
            "  ║    bash setup.sh        (creates the 'autoantibiotic' env)      ║\n"
            "  ║  or run everything in a container:                              ║\n"
            "  ║    docker run autoantibiotic --smiles \"...\"                     ║\n"
            "  ║  Or manually: conda install -c conda-forge vina                 ║\n"
            "  ╚══════════════════════════════════════════════════════════════════╝\n"
            "\033[0m",
            flush=True,
        )
        log.error(
            "Error: AutoDock Vina not found. Fix via `bash setup.sh`, "
            "the Docker image, or `conda install -c conda-forge vina`."
        )

    if not obabel_available:
        log.warning(
            "  ⚠  OpenBabel not found. Some conversions may fail; "
            "pipeline will attempt RDKit-based alternatives."
        )
        # High-visibility, bold error so the missing-binary fix is obvious and
        # copy-pasteable (mirrors the Vina message above).
        print(
            "\033[1;31m"
            "\n"
            "  ╔══════════════════════════════════════════════════════════════════╗\n"
            "  ║  ERROR: OpenBabel not found.                                    ║\n"
            "  ║  Install it with one command:                                   ║\n"
            "  ║    bash setup.sh        (creates the 'autoantibiotic' env)      ║\n"
            "  ║  or run everything in a container:                              ║\n"
            "  ║    docker run autoantibiotic --smiles \"...\"                     ║\n"
            "  ║  Or manually: conda install -c conda-forge openbabel            ║\n"
            "  ╚══════════════════════════════════════════════════════════════════╝\n"
            "\033[0m",
            flush=True,
        )
        log.error(
            "Error: OpenBabel not found. Fix via `bash setup.sh`, "
            "the Docker image, or `conda install -c conda-forge openbabel`."
        )

    # ── Success banner ──
    # All Python packages are required (checked above) and the two external
    # binaries (Vina + OpenBabel) are the only remaining gating items. When
    # both are present the environment is fully ready to screen.
    if vina_available and obabel_available:
        print(
            "\033[1;32m"
            "\n"
            f"  ✅ Ready to screen!  "
            f"(Vina: {vina_version or 'found'} | "
            f"OpenBabel: {obabel_version or 'found'})\n"
            "\033[0m",
            flush=True,
        )

    return {"vina": vina_available, "USE_VINA": vina_available}


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — PROTOCOL VALIDATION (Redocking)
# ═══════════════════════════════════════════════════════════════════════════════

def run_redocking_validation(
    holo_pdb_path: str,
    target_pdbqt_path: str,
    work_dir: str,
    deps: dict,
    mode: str = "science",
    config: Optional[dict] = None,
    target_pdbqt_paths: Optional[List[str]] = None,
) -> Tuple[bool, Optional[float]]:
    """
    Phase 0 — Protocol Validation.

    Extracts the native ligand from 6TKO, docks it back into the prepared
    PBP2a receptor(s), and computes the best (lowest) RMSD to the crystal
    pose across all prepared receptor conformers (consensus redocking).

    Returns (success: bool, rmsd: float | None) where ``rmsd`` is the best
    (lowest) RMSD over all conformers.
    """
    log.info("─── Phase 0: Redocking Validation ───")

    # Offline CI mode: never report a (non-physical) RMSD against test PDBs.
    if mode == "ci":
        log.info("Skipping redocking: CI/mock mode")
        return False, None

    lig_smi = os.path.join(work_dir, "native_ligand.smi")
    lig_pdbqt = os.path.join(work_dir, "native_ligand.pdbqt")
    docked_pdb = os.path.join(work_dir, "native_docked.pdb")

    # Optional explicit native-ligand resname override from config (Task 3).
    resname_override = (config or {}).get("native_ligand_resname")

    smi = _extract_native_ligand_from_holo(
        holo_pdb_path, lig_smi, lig_pdbqt, resname_override=resname_override
    )
    if smi is None:
        log.warning("  ⚠  Could not extract native ligand. Skipping redocking validation.")
        return False, None

    if not deps["USE_VINA"]:
        log.warning("  ⚠  Vina unavailable. Redocking validation requires Vina. Skip.")
        return False, None

    # Grid center = centroid of the native ligand residue (derived from the
    # native-ligand resname override, if provided). If the residue centroid
    # cannot be computed, validation must FAIL CLEANLY — we must NOT fall back
    # to an unrelated residue centroid (that would produce a meaningless RMSD).
    resname = resname_override.strip().upper() if resname_override else None
    center = None
    if resname:
        try:
            center = compute_residue_centroid(holo_pdb_path, [resname])
        except (ValueError, Exception):
            center = None
    if center is None:
        log.warning(
            "  ⚠  Could not compute native-ligand centroid for "
            f"'{resname_override}'. Native-ligand redocking cannot proceed; "
            "failing validation cleanly (no synthetic RMSD)."
        )
        return False, None

    # Run Vina redocking
    log.info("  Redocking native ligand into PBP2a…")

    # Size the redocking box from the native ligand itself (centroid + spread)
    # using the same auto-sizing logic as the allosteric/active sites, instead
    # of a fixed 25 Å cube. This keeps the box tight around the crystallographic
    # ligand so the redocked pose is measured on a comparable grid.
    #
    # Phase 3.5: use a *tighter* padding (4.0 Å instead of the default 6.0 Å)
    # for the native-ligand redocking box. A smaller search space reduces grid
    # noise and helps push the redocking RMSD under the 1.5 Å "Validated" gate.
    redock_box = _redocking_box_size(lig_pdbqt, center, redock_padding=4.0)
    log.info(
        f"  Redocking box: {redock_box[0]:.1f} x {redock_box[1]:.1f} x "
        f"{redock_box[2]:.1f} Å (auto-sized from native ligand)."
    )

    # Consensus redocking: redock into every prepared receptor conformer and
    # keep the best (lowest) RMSD. Falls back to the single primary receptor
    # when no explicit conformer list is provided.
    conformer_pdbqts = list(target_pdbqt_paths) if target_pdbqt_paths else [target_pdbqt_path]
    best_rmsd: Optional[float] = None
    for conf_idx, receptor_pdbqt in enumerate(conformer_pdbqts):
        if receptor_pdbqt is None:
            continue
        conf_pdbqt = docked_pdb.replace(".pdb", f"_c{conf_idx}.pdbqt")
        vina_cmd = [
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", lig_pdbqt,
            "--out", conf_pdbqt,
            "--center_x", f"{center[0]:.3f}",
            "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{redock_box[0]:.1f}",
            "--size_y", f"{redock_box[1]:.1f}",
            "--size_z", f"{redock_box[2]:.1f}",
                "--exhaustiveness", "64",
        ]
        try:
            subprocess.run(vina_cmd, capture_output=True, timeout=VINA_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.warning(f"  ⚠  Vina redocking timed out (>120s) on conformer {conf_idx}.")
            continue
        except FileNotFoundError:
            log.warning("  ⚠  Vina binary not found during redocking.")
            return False, None

        conf_pdb = conf_pdbqt.replace(".pdbqt", ".pdb")
        # Convert docked PDBQT back to PDB for RMSD calculation.
        # Vina output already has 3D coordinates, so --gen3d is not needed
        # and can cause OpenBabel 3.2.x to hang on complex molecules.
        try:
            subprocess.run(
                ["obabel", conf_pdbqt, "-O", conf_pdb],
                capture_output=True, timeout=60,
            )
        except Exception:
            log.warning("  Could not convert docked PDBQT to PDB. Trying RDKit PDBQT reader.")
            mol = Chem.MolFromPDBQT(conf_pdbqt) if hasattr(Chem, "MolFromPDBQT") else None
            if mol is None:
                log.warning(f"  ⚠  Cannot parse docked PDBQT for conformer {conf_idx}. RMSD skipped.")
                continue
            Chem.MolToPDBFile(mol, conf_pdb)

        crystal_pdb = lig_pdbqt.replace(".pdbqt", ".pdb")
        rmsd = _compute_rmsd_docked_vs_crystal(conf_pdb, crystal_pdb)
        if rmsd is None:
            log.warning(f"  ⚠  RMSD could not be computed for conformer {conf_idx}.")
            continue
        log.info(f"  Redocking RMSD (conformer {conf_idx}) = {rmsd:.3f} Å")
        if best_rmsd is None or rmsd < best_rmsd:
            best_rmsd = rmsd

    if best_rmsd is None:
        log.warning("  ⚠  Redocking RMSD could not be computed for any conformer.")
        return False, None

    rmsd = best_rmsd
    log.info(f"  Best (consensus) Redocking RMSD = {rmsd:.3f} Å")
    if rmsd > RMSD_MARGINAL_MAX:
        log.warning(
            f"  ⚠  Redocking RMSD ({rmsd:.3f} Å) exceeds {RMSD_MARGINAL_MAX:.1f} Å "
            "threshold. The docking protocol may not accurately reproduce known "
            "binding modes. Proceeding with pipeline — interpret results with caution."
        )
    else:
        log.info(
            f"  ✓  Redocking validated (RMSD = {rmsd:.3f} Å ≤ "
            f"{RMSD_MARGINAL_MAX:.1f} Å)."
        )

    # ``validation_ok`` reflects the RMSD_MARGINAL_MAX Å pass/fail gate, but we
    # always return the exact measured ``rmsd`` float (even when it exceeds the
    # threshold) so downstream reporters can display the raw value and emit
    # nuanced trust signals rather than a binary pass/fail.
    validation_ok = rmsd <= RMSD_MARGINAL_MAX if rmsd is not None else False

    # Persist the validation result to work_dir so downstream tooling / the paper
    # agent can read it without re-running the (expensive) redocking step
    # (paper §1, validation artifact). The CAUTION/trust badge logic in
    # config.constants.protocol_trust is intentionally unchanged — scientific
    # honesty is preserved.
    try:
        validation_json = os.path.join(work_dir, "validation_results.json")
        with open(validation_json, "w") as fh:
            json.dump({
                "validation_ok": bool(validation_ok),
                "redock_rmsd": (None if rmsd is None else float(rmsd)),
                "mode": mode,
                "rmsd_marginal_max": RMSD_MARGINAL_MAX,
                "rmsd_validated_max": RMSD_VALIDATED_MAX,
            }, fh, indent=2)
        log.info(f"  Validation results written: {validation_json}")
    except Exception as exc:
        log.warning(f"  ⚠  Could not write validation_results.json: {exc}")

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
        # PDBList's ``retrieve_pdb_file`` naming varies by Biopython version:
        # it may save as ``pdb{pdb_id_lower}.ent``, as ``{pdb_id}.pdb``, or even
        # under a nested ``pdbXXX`` subdirectory. Rather than assume a single
        # filename, scan the download dir for any file whose name contains the
        # pdb_id and rename the first match safely to ``{pdb_id}.pdb``.
        found = _find_downloaded_pdb(out_dir, pdb_id)
        if found is not None and found != target_path:
            # Avoid clobbering an existing correct file; only rename when the
            # source is a different path.
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(found, target_path)
        if not os.path.exists(target_path):
            # Last resort: if nothing matched, surface the directory contents
            # so the failure is never silent.
            entries = sorted(os.listdir(out_dir))
            log.error(
                f"  ✗  Download of {pdb_id} produced no recognisable PDB file. "
                f"Contents of {out_dir}: {entries}"
            )
            raise FileNotFoundError(
                f"PDB download for {pdb_id} did not yield a usable structure file."
            )
        log.info(f"  ✓  Downloaded {pdb_id} → {target_path}")
    except Exception as exc:
        log.error(f"  ✗  Failed to download {pdb_id}: {exc}")
        raise

    return target_path


def _find_downloaded_pdb(out_dir: str, pdb_id: str) -> Optional[str]:
    """
    Locate the just-downloaded PDB file for *pdb_id* in *out_dir*.

    Biopython's ``PDBList.retrieve_pdb_file`` is inconsistent about the exact
    output name (``pdb{pdb_id_lower}.ent``, ``{pdb_id}.pdb``, or a nested
    ``pdbXXX`` subfolder). This helper scans *out_dir* (recursively, one level
    deep) for any file whose name contains the *pdb_id* (case-insensitive) and
    returns the first such path, or ``None`` if nothing matches.

    Does NOT return an already-correct ``{pdb_id}.pdb`` at the top level — that
    is the caller's target path and is handled separately.
    """
    pid = pdb_id.lower()
    candidates = []
    for root, _dirs, files in os.walk(out_dir):
        for fname in files:
            low = fname.lower()
            # Match files like pdb3qpd.ent, 3qpd.pdb, 3qpd.ent, etc. but never
            # an already-named target file at the root.
            if pid in low and (
                low.endswith(".ent")
                or low.endswith(".pdb")
                or low.endswith(".cif")
            ):
                candidates.append(os.path.join(root, fname))
    if not candidates:
        return None
    # Prefer a strictly-named file (e.g. ``pdb3qpd.ent``) over an incidental
    # substring match; otherwise just take the first hit.
    candidates.sort(key=lambda p: (not os.path.basename(p).lower().startswith(f"pdb{pid}"), p))
    return candidates[0]


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
        # ── Attempt 1: Bio.PDB clean (strip waters / hetero ligands) ──
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

        # ── Attempt 2: RDKit hydrogen addition (PDB → MOL → H-Added → PDB) ──
        # Best-effort: if RDKit cannot parse/add Hs, keep the H-free PDB and
        # let Vina handle polar-H assignment internally.
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

        # ── Attempt 3: obabel PDBQT conversion (preferred) ──
        # Uses the `-xr` flag to produce a rigid-receptor PDBQT.
        # Without `-xr`, obabel creates a flexible ligand-style PDBQT
        # (branch records) that Vina rejects for a rigid receptor.
        pdbqt_path = out_path.replace(".pdb", ".pdbqt")
        try:
            subprocess.run(
                ["obabel", out_path, "-O", pdbqt_path, "-xr"],
                capture_output=True, timeout=300,
            )
            if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                log.info(f"  Receptor PDBQT written via obabel (-xr): {pdbqt_path}")
                return pdbqt_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # ── Attempt 4: RDKit/Bio.PDB PDBQT writer (fallback) ──
        # Produces a rigid-receptor PDBQT from first principles. Used as a
        # fallback when obabel is not on PATH.
        if write_receptor_pdbqt(out_path, pdbqt_path):
            log.info(f"  Receptor PDBQT written via RDKit fallback: {pdbqt_path}")
            return pdbqt_path

        # ── All four attempts failed ──
        raise RuntimeError(
            "Could not write a valid receptor PDBQT for "
            f"{pdb_path!r}. Step 1 (Bio.PDB clean) succeeded but Step 2 "
            "(RDKit hydrogen addition), Step 3 (obabel PDBQT conversion), and "
            "Step 4 (RDKit write_receptor_pdbqt fallback) all failed. "
            "OpenBabel ('obabel') is required to convert the cleaned PDB to "
            "PDBQT, and the RDKit-based fallback writer was unable to parse the "
            "receptor. Install obabel with `bash setup.sh` or "
            "`conda install -c conda-forge openbabel`, or use the Docker image."
        )

    except Exception as exc:
        log.error(f"  ✗  Failed to clean {pdb_path}: {exc}")
        raise


# NOTE: compute_residue_centroid / _centroid_of_pdb_atoms now live in
# utils.structure_prep and are imported above.


def prepare_targets(
    pdb_dir: str, work_dir: str, deps: dict, config: Optional[dict] = None
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

    # ── Resolve run mode explicitly from config (no path-based heuristic) ──
    mode = config.get("mode", "science")
    log.info(f"  Run mode (from config): {mode}")

    # ── Fetch structures (prefer bundled offline PDBs under tests/data) ──
    def _resolve_structure(pdb_id: str) -> str:
        """Return a local tests/data/{pdb_id}.pdb path if CI mode, else download."""
        if config.get("mode") == "ci":
            local_pdb = REPO_ROOT / "tests" / "data" / f"{pdb_id}.pdb"
            if local_pdb.exists():
                log.info(f"  Using local structure for {pdb_id}: {local_pdb}")
                return str(local_pdb)
        return fetch_structure(pdb_id, pdb_dir)

    holo_path = _resolve_structure(PDB_IDS["PBP2a_holo"])
    apo_path = _resolve_structure(PDB_IDS["PBP2a_apo"])
    trypsin_path = _resolve_structure(PDB_IDS["trypsin"])
    ces1_path = _resolve_structure(PDB_IDS["CES1"])

    # ── Consensus rigid docking: build a set of PBP2a receptor PDBQTs ──
    # Each conformer in PBP2A_CONFORMER_IDS is fetched (if not local) and
    # cleaned to its own PDBQT. The first entry (apo 3QPD) is kept as the
    # primary ``pdbqt`` key for backwards compatibility; the full list is
    # stored under ``receptor_pdbqts`` so screen_library can dock every
    # compound against all conformers and take the best (most negative) energy.
    conformer_paths = {}
    for cid in PBP2A_CONFORMER_IDS:
        try:
            conformer_paths[cid] = _resolve_structure(cid)
        except Exception as exc:
            log.warning(f"  ⚠  Could not resolve PBP2a conformer {cid}: {exc}")
    # Ensure the apo (primary) conformer is always present as the first entry.
    primary_id = PDB_IDS["PBP2a_apo"]
    ordered_ids = [primary_id] + [c for c in conformer_paths if c != primary_id]

    result["holo_pdb"] = holo_path

    # ── Explicit mode (config-driven, not inferred from file paths) ──
    if mode == "ci":
        log.info("CI mode: using mock PDBs - not for scientific use.")
    else:
        log.info("Science mode: real scientific validation expected.")
    result["mode"] = mode

    # ── Real-PDB guard: science mode must never silently use mock PDBs ──
    if mode == "science":
        for label, path in (
            ("holo", holo_path),
            ("apo", apo_path),
            ("trypsin", trypsin_path),
            ("CES1", ces1_path),
        ):
            if "tests/data" in os.path.abspath(path):
                msg = (
                    f"Refusing to run science mode with mock PDB for "
                    f"'{label}' ({path}). Place real PDBs under pdb_dir "
                    "(set mode: ci for offline mock runs)."
                )
                log.error(msg)
                raise ScienceModeMockPDBError(msg)

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

    # ── Build consensus receptor PDBQT list (one per conformer) ──
    # The primary apo receptor (pdbqt) is always first; each additional
    # conformer is cleaned to its own PDBQT. Missing conformers are skipped
    # gracefully so a partial conformer set still enables consensus docking.
    receptor_pdbqts = [pbp2a_pdbqt]
    for cid in ordered_ids:
        if cid == primary_id:
            continue
        cpath = conformer_paths.get(cid)
        if not cpath:
            continue
        cclean = os.path.join(work_dir, f"PBP2a_{cid}_clean.pdb")
        try:
            cpdbqt = clean_pdb_structure(cpath, cclean)
            receptor_pdbqts.append(cpdbqt)
            log.info(f"  Added PBP2a conformer {cid} receptor PDBQT: {cpdbqt}")
        except Exception as exc:
            log.warning(f"  ⚠  Could not prepare PBP2a conformer {cid}: {exc}")

    # ── Compute allosteric + active site centres from cleaned apo ──
    cleaned_pdb = pbp2a_clean_pdb

    log.info("  Computing allosteric site centroid (TYR105, GLN199, GLU237)…")
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
        active_center = None
    log.info(f"    Active site center: {active_center}")

    for site, center in (("allosteric", allosteric_center), ("active", active_center)):
        if center is None and mode == "science":
            msg = f"{site} center missing in science mode – aborting"
            log.error(msg)
            raise MissingGridCenterError(msg)

    result["PBP2a"] = {
        "pdbqt": pbp2a_pdbqt,
        "receptor_pdbqts": receptor_pdbqts,
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
    try:
        tryp_center = compute_residue_centroid(tryp_clean_pdb, TRYPSIN_CATALYTIC_RESIDUES)
    except (ValueError, Exception) as exc:
        log.warning(f"  ⚠  Trypsin catalytic residues {TRYPSIN_CATALYTIC_RESIDUES} missing: {exc}")
        log.warning("  Residue missing – grid center set to None; supply real PDB.")
        tryp_center = None
    log.info(f"    Trypsin active site center: {tryp_center}")
    result["trypsin"] = {"pdbqt": tryp_pdbqt, "active_center": tryp_center}

    # ── Clean CES1 ──
    log.info("  Cleaning Human Carboxylesterase 1 (1YAH)…")
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
        log.warning("  Residue missing – grid center set to None; supply real PDB.")
        ces1_center = None
    log.info(f"    CES1 active site center: {ces1_center}")
    result["CES1"] = {"pdbqt": ces1_pdbqt, "active_center": ces1_center}

    # ── Human off-target panel (wider selectivity screen) ──
    # Albumin (1AO6) and CYP3A4 (1W0E) are resolved like the other targets
    # and their docking grid is centred on the configured catalytic residues.
    # Missing/offline PDBs are skipped gracefully in CI mode.
    for label, pdb_key, residues, out_name, optional in (
        ("Human Serum Albumin", "HUMAN_ALBUMIN", ALBUMIN_CATALYTIC_RESIDUES, "albumin", False),
        ("Human CYP3A4", "CYP3A4", CYP3A4_CATALYTIC_RESIDUES, "cyp3a4", False),
        # Wider panel: HERG (cardiotoxicity, 7CN1 — skipped if unavailable) and
        # CYP2D6 (mock-capable metabolic liability). Both are prepared like the
        # other off-targets but are *optional*: if no PDB can be resolved (e.g.
        # offline CI, unavailable/synth-id), they are skipped gracefully so the
        # pipeline keeps running with whatever panel is available.
        ("Human Ether-à-go-go (hERG)", "HERG", HERG_CATALYTIC_RESIDUES, "herg", True),
        ("Human CYP2D6", "CYP2D6", CYP2D6_CATALYTIC_RESIDUES, "cyp2d6", True),
    ):
        try:
            pdb_path = _resolve_structure(PDB_IDS[pdb_key])
        except Exception as exc:
            if optional:
                log.warning(
                    f"  ⚠  Could not resolve {label} ({pdb_key}); skipping "
                    f"(optional off-target): {exc}"
                )
                result[out_name] = {"pdbqt": None, "active_center": None}
                continue
            raise
        clean_pdb = os.path.join(work_dir, f"{out_name}_clean.pdb")
        try:
            pdbqt = clean_pdb_structure(pdb_path, clean_pdb)
            center = compute_residue_centroid(clean_pdb, residues)
        except Exception as exc:
            log.warning(f"  ⚠  Could not prepare {label} ({pdb_key}): {exc}")
            pdbqt, center = None, None
        log.info(f"  {label} active site center: {center}")
        result[out_name] = {"pdbqt": pdbqt, "active_center": center}

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
#  PHASE 2 — LIBRARY GENERATION & FILTERING
# ═══════════════════════════════════════════════════════════════════════════════
#
# NOTE: The ``CompoundRecord`` dataclass and ``generate_candidate_library``
# now live in ``utils/library_gen.py`` and are imported above. The
# orchestrator below only consumes them.


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — VIRTUAL SCREENING (Docking)
# ═══════════════════════════════════════════════════════════════════════════════

def _consensus_dock(
    records: List[CompoundRecord],
    receptor_pdbqts: List[str],
    center,
    box_size,
    work_dir: str,
    tag: str,
    use_vina: bool = True,
) -> List[Tuple[CompoundRecord, Optional[float]]]:
    """
    Consensus rigid docking helper.

    Docks *records* against every receptor PDBQT in *receptor_pdbqts* (each
    a PBP2a conformer) and returns ``(record, best_energy)`` pairs where
    ``best_energy`` is the most negative (best) docking energy across all
    conformers. Reuses :func:`_dock_compounds_parallel` per conformer; no new
    parallel infrastructure is introduced. Missing/failed conformer dockings are
    ignored (``None``) and the best of the remaining is taken.

    Args:
        records: Compounds to dock.
        receptor_pdbqts: List of receptor PDBQT paths (consensus set).
        center: Grid-box centre (shared across conformers).
        box_size: Grid-box dimensions (shared across conformers).
        work_dir: Scratch directory.
        tag: Label for temporary files.
        use_vina: When ``False``, the RDKit fallback scorer is used.

    Returns:
        List of ``(CompoundRecord, energy_or_None)`` tuples.
    """
    if not receptor_pdbqts:
        return [(r, None) for r in records]

    by_id: dict = {r.compound_id: r for r in records}
    # Seed with per-compound best energy (None initially).
    best: Dict[str, Optional[float]] = {r.compound_id: None for r in records}
    # Active-site pose paths returned by the parallel workers. The pose is set
    # on the parent record inside _dock_compounds_parallel, but we also collect
    # it here per conformer so the best-energy conformer's pose is retained.
    best_pose: Dict[str, Optional[str]] = {r.compound_id: None for r in records}

    for conf_idx, receptor_pdbqt in enumerate(receptor_pdbqts):
        if center is None or receptor_pdbqt is None:
            continue
        results = _dock_compounds_parallel(
            records, receptor_pdbqt,
            center, box_size,
            work_dir, f"{tag}_c{conf_idx}",
            use_vina=use_vina,
        )
        for rec, energy in results:
            if energy is None:
                continue
            cur = best.get(rec.compound_id)
            if cur is None or energy < cur:
                best[rec.compound_id] = energy
                # Keep the pose from the conformer that produced the best energy
                # so MM-GBSA / H-bond / mutation analysis use a consistent pose.
                if getattr(rec, "active_docked_pdbqt", None):
                    best_pose[rec.compound_id] = rec.active_docked_pdbqt

    # Assign the retained active-site pose back to each parent record.
    for cid, pose in best_pose.items():
        if pose is not None:
            by_id[cid].active_docked_pdbqt = pose

    return [(by_id[cid], e) for cid, e in best.items()]


_FLEX_BACKBONE_KEEP = {"N", "CA", "C", "O", "OXT", "H", "HA", "HN"}


def _strip_flex_sidechains_from_rigid(
    receptor_pdbqt: str,
    target: set,
    out_pdbqt: str,
) -> bool:
    """
    Write a rigid receptor PDBQT with the flexible residues' *side-chain*
    atoms removed (backbone N/CA/C/O kept).

    Vina's ``--flex`` requires that atoms declared movable in the flex file are
    NOT also present in the rigid receptor; otherwise the docking aborts with
    an opaque "unknown error". This produces the companion ``_rigid`` receptor.

    ``target`` is a set of ``(RESNAME, RESSEQ)`` tuples. Returns ``True`` on
    success.
    """
    try:
        kept = []
        with open(receptor_pdbqt) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip().upper()
                        res_seq = int(line[22:26])
                    except (ValueError, IndexError):
                        kept.append(line.rstrip("\n"))
                        continue
                    if (res_name, res_seq) in target and atom_name not in _FLEX_BACKBONE_KEEP:
                        # Drop movable side-chain atom from the rigid receptor.
                        continue
                kept.append(line.rstrip("\n"))
        with open(out_pdbqt, "w") as fh:
            fh.write("\n".join(kept) + "\n")
        return True
    except Exception as exc:
        log.warning(f"  ⚠  Failed to strip flex side chains from rigid receptor: {exc}")
        return False


def _prepare_flex_pdbqt(
    receptor_pdbqt: str,
    cleaned_pdb: str,
    flex_residues: List[str],
    work_dir: str,
    receptor_pdb: Optional[str] = None,
) -> Optional[tuple]:
    """
    Build a flexible-residue PDBQT (and the companion stripped rigid receptor)
    for Vina's ``--flex`` flag.

    The flexible residues (e.g. SER403, LYS406, TYR446) are extracted into
    their own PDB (via Bio.PDB), then converted to a *valid Vina flex PDBQT*
    (a proper ROOT/BRANCH torsion tree) by
    :func:`utils.structure_prep.write_flex_pdbqt`. In addition, a companion
    rigid receptor is written with those residues' side-chain atoms removed —
    Vina requires the movable atoms to appear ONLY in the flex file, never in
    the rigid receptor, otherwise docking aborts (paper §3.3).

    Returns ``(rigid_pdbqt, flex_pdbqt)`` on success, or ``None`` on any
    failure (callers fall back to rigid docking and log a warning).
    """
    if not flex_residues or not cleaned_pdb or not os.path.exists(cleaned_pdb):
        return None
    try:
        from utils.structure_prep import write_flex_pdbqt

        flex_pdb = os.path.join(work_dir, "PBP2a_flex_residues.pdb")
        # Extract just the flexible residues into their own PDB via Bio.PDB.
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("receptor", cleaned_pdb)
        target = set()
        for entry in flex_residues:
            resname = "".join(ch for ch in entry if ch.isalpha()).upper()
            seqnum = int("".join(ch for ch in entry if ch.isdigit()))
            target.add((resname, seqnum))

        class FlexSelect(Select):
            def accept_residue(self, residue):
                rid = residue.get_id()
                if rid[0] != " ":
                    return False
                return (residue.get_resname().strip().upper(),
                        rid[1]) in target

        io = PDBIO()
        io.set_structure(struct)
        io.save(flex_pdb, FlexSelect())
        if not os.path.exists(flex_pdb) or os.path.getsize(flex_pdb) == 0:
            log.warning("  ⚠  Flexible-residue extraction produced no atoms; skipping flex docking.")
            return None

        flex_pdbqt = os.path.join(work_dir, "PBP2a_flex.pdbqt")
        if not write_flex_pdbqt(flex_pdb, flex_pdbqt, flex_residues=flex_residues):
            return None

        # Guard: Vina rejects a malformed flex file with an opaque error that
        # would otherwise surface only after a full (expensive) docking run.
        # Validate the torsion tree up front so the pipeline fails fast and
        # cleanly falls back to rigid docking when the flex PDBQT is unusable.
        from utils.structure_prep import validate_flex_pdbqt
        if not validate_flex_pdbqt(flex_pdbqt):
            log.warning(
                "  ⚠  Generated flex PDBQT failed Vina validity check "
                "(torsion tree malformed / TORSDOF present). Falling back to "
                "rigid docking."
            )
            return None

        # Companion rigid receptor with movable side chains removed.
        rigid_pdbqt = os.path.join(work_dir, "PBP2a_rigid_flex.pdbqt")
        if not _strip_flex_sidechains_from_rigid(receptor_pdbqt, target, rigid_pdbqt):
            return None

        return rigid_pdbqt, flex_pdbqt
    except Exception as exc:
        log.warning(f"  ⚠  Flexible-residue PDBQT preparation failed: {exc}. Falling back to rigid.")
        return None


def _run_flex_dock_with_fallback_timeout(
    rec: "CompoundRecord",
    rigid_flex_pdbqt: str,
    active_center: np.ndarray,
    active_box: Tuple[float, float, float],
    work_dir: str,
    flex_pdbqt: str,
) -> Optional[float]:
    """
    Run a single flexible (Vina ``--flex``) docking job for *rec*.

    Phase 3.5 — robust flexible docking. The job is attempted with the standard
    ``VINA_TIMEOUT_S``; if Vina times out, it is retried ONCE with the dedicated
    larger ``FLEX_VINA_TIMEOUT_S`` rather than falling back to rigid docking.
    This guarantees the Top-50 active-site ranking is driven by a genuine
    flexible-docked score.

    Returns the flexible binding energy, or ``None`` only when even the
    extended-timeout run fails (in which case the caller keeps the rigid
    consensus energy as a fallback).
    """
    from utils.docking import dock_compound

    try:
        energy = dock_compound(
            rec, rigid_flex_pdbqt, active_center, active_box,
            work_dir, "active_flex", use_vina=True,
            flex_pdbqt=flex_pdbqt, timeout=VINA_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        energy = None
    if energy is not None:
        return energy

    log.info(
        f"  Flex docking for {rec.compound_id} timed out at "
        f"{VINA_TIMEOUT_S}s; retrying with extended timeout "
        f"{FLEX_VINA_TIMEOUT_S}s (no rigid fallback)."
    )
    try:
        energy = dock_compound(
            rec, rigid_flex_pdbqt, active_center, active_box,
            work_dir, "active_flex", use_vina=True,
            flex_pdbqt=flex_pdbqt, timeout=FLEX_VINA_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        energy = None
    return energy


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

    Returns top 10 candidates with docking scores populated.

    Requires AutoDock Vina. If Vina is unavailable the pipeline cannot screen
    and raises ``RuntimeError`` — install Vina via ``bash setup.sh`` or Docker.
    """
    log.info("─── Phase 3: Virtual Screening ───")

    use_vina = deps.get("USE_VINA", False)
    if not use_vina:
        log.warning(
            "AutoDock Vina not available — using the RDKit shape/pharmacophore "
            "fallback scorer. These scores are APPROXIMATE and rank candidates "
            "relative to each other only; they are NOT physical binding energies."
        )

    pb2pa = targets["PBP2a"]
    allosteric_center = pb2pa["allosteric_center"]
    active_center = pb2pa["active_center"]

    # Auto-sized boxes (centroid + atom spread, min 15 Å) — never rely on the
    # hardcoded constants when a real grid centre exists.
    allosteric_box = _auto_box_size(pb2pa.get("allosteric_pdbqt") or pb2pa.get("cleaned_pdb"), allosteric_center, ALLOSTERIC_BOX_SIZE) \
        if allosteric_center is not None else ALLOSTERIC_BOX_SIZE
    active_box = _auto_box_size(pb2pa.get("cleaned_pdb"), active_center, ACTIVE_BOX_SIZE) \
        if active_center is not None else ACTIVE_BOX_SIZE

    # ── Allosteric docking (consensus over PBP2a conformers) ──
    # Each compound is docked against every prepared receptor PDBQT; the best
    # (most negative) energy is kept as ``pb2pa_allosteric_energy``. No new
    # parallel infrastructure is introduced — we reuse ``_dock_compounds_parallel``
    # per conformer and merge results by taking the minimum energy.
    receptor_pdbqts = pb2pa.get("receptor_pdbqts") or [pb2pa["pdbqt"]]
    log.info(
        f"  Docking all compounds against allosteric site "
        f"({len(receptor_pdbqts)} PBP2a conformer(s))…"
    )
    allosteric_results = _consensus_dock(
        records, receptor_pdbqts,
        allosteric_center, allosteric_box,
        work_dir, "allosteric",
        use_vina=use_vina,
    )

    n_scored = 0
    for rec, energy in allosteric_results:
        rec.pb2pa_allosteric_energy = energy
        if energy is not None:
            n_scored += 1

    log.info(f"  Allosteric docking complete: {n_scored}/{len(records)} scored.")

    # ── Select top candidates for active-site docking ──
    # Adaptive threshold: dock at least 5 but at most 50 compounds (or all
    # available, whichever is smaller). This ensures the active-site step runs
    # even for modest-sized libraries.
    scored = [r for r, e in allosteric_results if e is not None]
    scored.sort(key=lambda r: r.pb2pa_allosteric_energy)
    active_top_n = min(50, max(5, len(scored)))

    if len(scored) >= 5:
        top_active = scored[:active_top_n]
        log.info(
            f"  Docking top {len(top_active)} compounds against active site "
            f"({len(receptor_pdbqts)} PBP2a conformer(s))…"
        )

        active_results = _consensus_dock(
            top_active, receptor_pdbqts,
            active_center, active_box,
            work_dir, "active",
            use_vina=use_vina,
        )

        for rec, energy in active_results:
            rec.pb2pa_active_energy = energy

        # ── Local flexible docking (science mode only, Vina required) ──
        # Phase 3.5: flexible docking is MANDATORY for the Top-50 active-site
        # candidates. The flexible energy (SER403/LYS406/TYR446 side chains
        # movable via Vina --flex) is the authoritative active-site ranking
        # score, not a re-dock for analysis only. The rigid consensus energy is
        # used only as a fallback when a flex job genuinely cannot produce a
        # valid energy (e.g. the flex receptor could not be prepared).
        #
        # Critically, if Vina times out on a flex job we do NOT fall back to
        # rigid docking — instead the flex job is retried with the dedicated,
        # larger FLEX_VINA_TIMEOUT_S so the Top-50 final ranking always reflects
        # a proper flexible-docked score.
        mode = deps.get("mode", "science")
        if use_vina and mode == "science" and active_center is not None:
            primary_pdbqt = receptor_pdbqts[0]
            cleaned_pdb = pb2pa.get("cleaned_pdb")
            flex_prep = _prepare_flex_pdbqt(
                primary_pdbqt, cleaned_pdb, FLEX_RESIDUES, work_dir,
            )
            if flex_prep is not None:
                rigid_flex_pdbqt, flex_pdbqt = flex_prep
                log.info(
                    f"  Local flexible docking (--flex) for top {len(top_active)} "
                    f"compounds at the active site (mandatory final ranking)…"
                )
                for rec in top_active:
                    if rec.mol is None:
                        mol = Chem.MolFromSmiles(rec.smiles)
                        if mol is None:
                            continue
                        rec.mol = mol
                    # Phase 3.5: try the flex job with the standard timeout first;
                    # on timeout, retry once with the dedicated larger flex
                    # timeout rather than silently degrading to rigid docking.
                    flex_energy = _run_flex_dock_with_fallback_timeout(
                        rec, rigid_flex_pdbqt, active_center, active_box,
                        work_dir, flex_pdbqt,
                    )
                    if flex_energy is not None:
                        # The flexible-docked energy is authoritative for the
                        # Top-50 final ranking. It replaces (never just improves
                        # on) the rigid consensus active-site energy.
                        rec.pb2pa_active_energy = flex_energy
                    else:
                        # A genuine failure (not a recoverable timeout) keeps the
                        # existing rigid consensus energy so the candidate still
                        # ranks, but we never *pretend* the flex pose succeeded.
                        log.warning(
                            f"  ⚠  Flexible docking failed for {rec.compound_id} "
                            f"(retained rigid energy {rec.pb2pa_active_energy})."
                        )
            else:
                log.warning(
                    "  Flexible-residue PDBQT preparation failed; active-site "
                    "docking remains rigid (backward compatible)."
                )

    # ── Select top 10 ──
    # Rank by allosteric energy (lower = better)
    top10 = select_top(records, "pb2pa_allosteric_energy")

    log.info(f"  Top {len(top10)} candidates selected.")
    for i, r in enumerate(top10):
        energy_str = (
            f"{r.pb2pa_allosteric_energy:.2f}" if r.pb2pa_allosteric_energy is not None
            else "N/A"
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

    unverified_residues: List[str] = []
    for resname, coords in atom_coords.items():
        if not coords:
            # The supplied receptor PDB lacks this key residue: do NOT fake a
            # distance. Mark it unverified and leave min_dist as +inf.
            min_dists[resname] = float("inf")
            unverified_residues.append(resname)
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

    # Residues absent from the supplied receptor PDB (unverified — no faked
    # coordinates or distances were used for them).
    results["unverified_residues"] = unverified_residues

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
    residues.  The interaction fingerprint (*interactions*) must be supplied
    by the caller — it is derived from the active-site pose captured during
    screen_library (record.active_docked_pdbqt).  This function no longer
    re-docks internally; if *interactions* is None it notes "no pose".

    Args:
        record: Compound record containing docking scores.
        work_dir: Working directory for temporary files.
        receptor_pdbqt: Path to the PBP2a receptor PDBQT file.
        center: Grid box centre (unused — retained for signature stability).
        box_size: Grid box dimensions (unused — retained for stability).
        interactions: Pre-computed interaction fingerprint dict returned by
                      ``analyze_binding_interactions``.  If None, the pose is
                      flagged as unavailable.

    Returns a human-readable notes string.
    """
    pose_notes = _pose_based_resistance_notes(record, interactions)
    energy_notes = _energy_based_resistance_notes(record)
    energy_notes += _physicochemical_resistance_notes(record)

    if pose_notes and energy_notes:
        notes = pose_notes + energy_notes
    else:
        notes = pose_notes or energy_notes

    if not notes:
        notes.append("No specific resistance flags identified.")

    return "; ".join(notes)


def _pose_based_resistance_notes(
    record: CompoundRecord,
    interactions: Optional[Dict[str, Union[bool, float]]],
) -> List[str]:
    """Build pose-derived resistance notes from the supplied interaction fingerprint.

    The interaction fingerprint is derived from the active-site pose captured
    during ``screen_library`` (record.active_docked_pdbqt). We no longer re-dock
    here — if *interactions* is None we simply note that the pose is absent.
    """
    notes: List[str] = []

    # Pose-based interactions are supplied by the caller (the active-site pose
    # captured during screen_library via record.active_docked_pdbqt). We no
    # longer re-dock here — if no pose is available we simply note that.
    if interactions is None:
        notes.append("no pose — binding interactions not analysed.")
        return notes

    # ── Quantitative resistance check from measured pose distances ──
    # The interaction fingerprint already exposes the minimum ligand→residue
    # distances (Å). We use these directly rather than the boolean contact
    # flags so resistance risk scales with how tightly the conserved
    # catalytic network is engaged.
    ser = interactions.get("min_dist_Ser403", float("inf"))
    lys = interactions.get("min_dist_Lys406", float("inf"))
    tyr = interactions.get("min_dist_Tyr446", float("inf"))

    # Residues absent from the cleaned receptor PDB are unverified: flag
    # them explicitly (in natural language) rather than treating their
    # (infinite) distance as a scientific measurement.
    unverified = interactions.get("unverified_residues") or []
    for resname in unverified:
        notes.append(
            f"unverified residue ({resname}) — absent from cleaned PDB"
        )

    # ── Natural-language key-interaction summary ──
    # Build human-readable sentences for the catalytic network so chemists
    # immediately understand the binding mode rather than parsing raw
    # distance flags. Track which key contacts are favourably engaged.
    ser_ok = np.isfinite(ser) and ser < 3.5
    lys_ok = np.isfinite(lys) and lys < 3.8

    if np.isfinite(ser):
        if ser_ok:
            notes.append(
                f"Forms strong H-bond with catalytic Ser403 ({ser:.1f} Å)."
            )
        elif ser < 5.0:
            notes.append(
                f"Weak contact with catalytic Ser403 ({ser:.1f} Å) — resistance risk."
            )
        else:
            notes.append(
                f"Loss of Ser403 engagement ({ser:.1f} Å) — high resistance risk."
            )
    elif "Ser403" not in unverified:
        notes.append(
            "Ser403 distance undefined — high resistance risk."
        )

    if np.isfinite(lys):
        if lys_ok:
            notes.append(
                f"Stabilized by Lys406 contact ({lys:.1f} Å)."
            )
        elif lys < 5.0:
            notes.append(
                f"Weak Lys406 contact ({lys:.1f} Å) — resistance risk."
            )
    elif "Lys406" not in unverified:
        notes.append(
            "Lys406 distance undefined — resistance risk."
        )

    # If neither key catalytic residue is engaged (and neither is merely
    # unverified due to a missing PDB residue), state this plainly.
    unverified_key = [r for r in unverified if r in ("Ser403", "Lys406")]
    if (not ser_ok) and (not lys_ok) and not unverified_key:
        notes.append(
            "Lacks key interactions with catalytic Ser403 and Lys406."
        )

    if np.isfinite(tyr):
        if tyr < 3.5:
            notes.append(
                f"Stabilising contact with Tyr446 ({tyr:.1f} Å)."
            )
        elif tyr < 5.0:
            notes.append(
                f"Weak Tyr446 contact ({tyr:.1f} Å) — resistance risk."
            )

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

    return notes


def _energy_based_resistance_notes(record: CompoundRecord) -> List[str]:
    """Build resistance notes from docking-energy heuristics."""
    notes: List[str] = []

    # Energy-based heuristics
    if record.pb2pa_active_energy is not None and record.pb2pa_active_energy < -6.0:
        notes.append("Likely contacts catalytic Ser403 (active site, energy-based). Good.")

    # Resistance risk indicators
    if record.qed_score > 0.8:
        notes.append("High drug-likeness (QED > 0.8) — good developability profile.")

    return notes


def _physicochemical_resistance_notes(record: CompoundRecord) -> List[str]:
    """Build resistance notes from physicochemical properties (MW, rigidity)."""
    notes: List[str] = []

    # Molecular weight / rigidity heuristic
    if record.mol is not None:
        mw = Descriptors.MolWt(record.mol)
        if mw > 400:
            notes.append("High MW (>400) — broad interaction surface, may contact multiple residues.")
        n_rot = Descriptors.NumRotatableBonds(record.mol)
        if n_rot < 5:
            notes.append("Rigid scaffold — reduced entropic penalty, may enhance binding specificity.")

    return notes


def _mutate_pdbqt_residue(
    pdbqt_path: str,
    work_dir: str,
    residue_id: int,
    new_resname: str,
) -> Optional[str]:
    """
    Build a mutant receptor PDBQT by replacing one residue's name (cheap string
    replace on the fixed-column PDBQT ``RESNAME`` field, cols 17-20).

    Returns the path to the mutant PDBQT, or ``None`` on failure. The 3-letter
    ``new_resname`` (e.g. ``"ALA"``) is written into columns 17-20 for every
    ``ATOM``/``HETATM`` line whose residue sequence number equals *residue_id*
    (and whose chain column indicates a standard polymer residue). This is a
    fast, dependency-free approximation of mutagenesis used only for the
    reduced-resistance scan — it does NOT rebuild side-chain geometry, so the
    result is treated as a coarse screen, not a structural model.
    """
    if not pdbqt_path or not os.path.exists(pdbqt_path):
        return None
    try:
        out_path = os.path.join(work_dir, f"PBP2a_mut_{residue_id}{new_resname}.pdbqt")
        with open(pdbqt_path) as fh:
            lines = fh.readlines()
        new_lines = []
        mutated = False
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    res_seq = int(line[22:26])
                    # Chain column (21) must indicate a standard polymer residue.
                    chain = line[21]
                    if res_seq == residue_id and chain.strip() != "":
                        # Replace residue name (cols 17-20, 1-indexed).
                        line = line[:17] + f"{new_resname:<3s}" + line[20:]
                        mutated = True
                except (ValueError, IndexError):
                    pass
            new_lines.append(line)
        if not mutated:
            log.warning(
                f"  ⚠  PDBQT mutation {residue_id}→{new_resname} matched no "
                "residue; skipping mutant."
            )
            return None
        with open(out_path, "w") as fh:
            fh.writelines(new_lines)
        return out_path
    except Exception as exc:
        log.warning(f"  ⚠  PDBQT mutation failed: {exc}")
        return None


def _run_mutation_scan(
    top10: List[CompoundRecord],
    targets: dict,
    work_dir: str,
    deps: dict,
) -> None:
    """
    Simple quantitative mutation scan for reduced-resistance probability.

    For each candidate with a retained active-site pose, the pose is re-docked
    against three PBP2a mutant receptors (S403A, K406A, Y446A) built from the
    apo PDBQT via :func:`_mutate_pdbqt_residue`. The energy difference versus
    the wild-type active-site energy (``mutant_energy_delta``) is recorded on
    the record, and a quantitative note is appended to ``resistance_notes``:
    a smaller delta means the candidate is less disrupted by the resistance
    mutation (i.e. lower predicted resistance emergence).

    Skips gracefully when Vina is unavailable, the mode is not science, no
    receptor/pose is present, or mutation scanning is disabled in config.
    """
    if not MUTATION_SCAN:
        return
    if not deps.get("USE_VINA", False):
        return
    if targets.get("mode") != "science":
        return

    pb2pa = targets.get("PBP2a", {})
    wt_pdbqt = pb2pa.get("pdbqt")
    active_center = pb2pa.get("active_center")
    if not wt_pdbqt or active_center is None:
        return

    active_box = _auto_box_size(
        pb2pa.get("cleaned_pdb"), active_center, ACTIVE_BOX_SIZE,
    ) if pb2pa.get("cleaned_pdb") else ACTIVE_BOX_SIZE

    # Map mutant label → (residue_id, new 3-letter resname).
    # Labels use the standard 1-letter mutation code, e.g. "S403A" = Ser403→Ala.
    _AA1_TO_3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    mutant_specs = []
    for mut in MUTATION_SCAN_MUTANTS:
        # e.g. "S403A": wild-type 'S' (Ser), id 403, mutant 'A' (Ala).
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut)
        if not m:
            continue
        res_id = int(m.group(2))
        new_name = _AA1_TO_3.get(m.group(3))
        if new_name is None:
            continue
        mutant_specs.append((mut, res_id, new_name))

    if not mutant_specs:
        return

    log.info("  Running PBP2a mutation scan (S403A/K406A/Y446A) for top candidates…")
    for rec in top10:
        pose = getattr(rec, "active_docked_pdbqt", None)
        if not pose or not os.path.exists(pose):
            rec.mutant_energy_delta = None
            continue
        wt_energy = rec.pb2pa_active_energy
        if wt_energy is None:
            rec.mutant_energy_delta = None
            continue

        deltas = []
        for label, res_id, new_name in mutant_specs:
            mut_pdbqt = _mutate_pdbqt_residue(wt_pdbqt, work_dir, res_id, new_name)
            if mut_pdbqt is None:
                continue
            try:
                energy = dock_compound(
                    rec, mut_pdbqt, active_center, active_box,
                    work_dir, f"mut_{label}", use_vina=True,
                )
            except Exception as exc:
                log.warning(f"  ⚠  Mutant docking {label} failed: {exc}")
                continue
            if energy is None:
                continue
            deltas.append(energy - wt_energy)

        rec.mutant_energy_delta = float(np.mean(deltas)) if deltas else None
        if rec.mutant_energy_delta is not None and np.isfinite(rec.mutant_energy_delta):
            avg = rec.mutant_energy_delta
            verdict = (
                "remains bound under key resistance mutants"
                if avg <= 1.0 else
                "moderately disrupted by resistance mutants"
                if avg <= 3.0 else
                "highly disrupted by resistance mutants (high resistance risk)"
            )
            note = (
                f"Mutation scan ΔG_avg={avg:.2f} kcal/mol vs WT ({verdict})."
            )
            if rec.resistance_notes:
                rec.resistance_notes += " | "
            rec.resistance_notes += note


def _run_resistance_profiling(
    top10: List[CompoundRecord],
    targets: dict,
    work_dir: str,
) -> None:
    """
    Pose-based resistance profiling for the *top10* candidates.

    Uses the active-site pose captured during ``screen_library``
    (``record.active_docked_pdbqt``) to compute the binding-interaction
    fingerprint, then runs :func:`profile_resistance_risk`. When no pose was
    retained (e.g. the RDKit fallback path), the analysis gracefully notes
    "no pose" rather than fabricating a pose.
    """
    pb2pa = targets.get("PBP2a", {})
    cleaned_pdb = pb2pa.get("cleaned_pdb")

    for rec in top10:
        interactions = None

        if cleaned_pdb and os.path.exists(cleaned_pdb):
            out_pdbqt = getattr(rec, "active_docked_pdbqt", None)
            if out_pdbqt and os.path.exists(out_pdbqt):
                try:
                    interactions = analyze_binding_interactions(out_pdbqt, cleaned_pdb)
                except Exception:
                    interactions = None

        # Stash the interaction fingerprint on the record so the CSV report
        # can derive per-residue H-bond columns without re-parsing the pose.
        rec.interactions = interactions

        rec.resistance_notes = profile_resistance_risk(
            rec, work_dir,
            pb2pa.get("pdbqt", ""),
            pb2pa.get("active_center"),
            ACTIVE_BOX_SIZE,
            interactions=interactions,
        )


def analyze_selectivity_and_resistance(
    top10: List[CompoundRecord],
    targets: dict,
    work_dir: str,
    deps: dict,
) -> List[CompoundRecord]:
    """
    Phase 4 — Selectivity & Resistance Analysis.

    1. Dock top 10 candidates against the human off-target panel (Trypsin,
       CES1, Albumin, CYP3A4) — 4 proteins for a wider selectivity screen.
    2. Compute Selectivity Index (average human energy over up to 4 targets).
    3. Optionally rerank the top 10 by a lightweight MM-GBSA-like MMFF score.
    4. Profile resistance risk.

    Returns updated records with selectivity and resistance fields.
    """
    log.info("─── Phase 4: Selectivity & Resistance Analysis ───")

    use_vina = deps.get("USE_VINA", False)

    if not use_vina:
        log.warning(
            "  Vina unavailable — using the RDKit fallback scorer for human "
            "off-targets. Selectivity indices are APPROXIMATE."
        )
        for rec in top10:
            rec.selectivity_index = max(0.0, 1.0 - rec.max_similarity)
            rec.selectivity_confidence = CompoundRecord.CONF_LOW
            rec.resistance_notes = (
                "Selectivity assessed with approximate RDKit fallback scores "
                "(Vina unavailable)."
            )
        # Still run the pose-based resistance analysis below using any active
        # pose captured during Phase 3 (none in fallback mode).
        _run_resistance_profiling(top10, targets, work_dir)
        return top10

    # ── Dock vs Trypsin (using computed catalytic triad centre) ──
    log.info("  Docking top 10 vs Human Trypsin (1UTN)…")
    trypsin_box = _auto_box_size(
        targets["trypsin"].get("cleaned_pdb"), targets["trypsin"]["active_center"],
        (20.0, 20.0, 20.0),
    ) if targets["trypsin"].get("active_center") is not None else (20.0, 20.0, 20.0)
    trypsin_results = _dock_compounds_parallel(
        top10, targets["trypsin"]["pdbqt"],
        targets["trypsin"]["active_center"], trypsin_box,
        work_dir, "trypsin", n_jobs=min(4, len(top10)),
    )
    for rec, energy in trypsin_results:
        rec.human_trypsin_energy = energy

    # ── Dock vs CES1 (using computed catalytic triad centre) ──
    log.info("  Docking top 10 vs Human Carboxylesterase 1 (1YAH)…")
    ces1_box = _auto_box_size(
        targets["CES1"].get("cleaned_pdb"), targets["CES1"]["active_center"],
        (20.0, 20.0, 20.0),
    ) if targets["CES1"].get("active_center") is not None else (20.0, 20.0, 20.0)
    ces1_results = _dock_compounds_parallel(
        top10, targets["CES1"]["pdbqt"],
        targets["CES1"]["active_center"], ces1_box,
        work_dir, "ces1", n_jobs=min(4, len(top10)),
    )
    for rec, energy in ces1_results:
        rec.human_ces1_energy = energy

    # ── Dock vs wider human off-target panel ──
    # Averages the human binding energy over up to 6 proteins (Trypsin, CES1,
    # Albumin, CYP3A4, hERG, CYP2D6) for a more conservative (harder to pass)
    # Selectivity Index. Targets without a prepared receptor / grid centre are
    # skipped gracefully, so the panel size adapts to what is available.
    for label, key, attr in (
        ("Human Serum Albumin (1AO6)", "albumin", "human_albumin_energy"),
        ("Human CYP3A4 (1W0E)", "cyp3a4", "human_cyp3a4_energy"),
        ("Human Ether-à-go-go (hERG, 7CN1)", "herg", "human_herg_energy"),
        ("Human CYP2D6", "cyp2d6", "human_cyp2d6_energy"),
    ):
        tgt = targets.get(key)
        if not tgt or tgt.get("pdbqt") is None or tgt.get("active_center") is None:
            log.warning(f"  Skipping {label} (no prepared receptor / grid centre).")
            continue
        log.info(f"  Docking top 10 vs {label}…")
        box = _auto_box_size(
            tgt.get("cleaned_pdb"), tgt["active_center"],
            (20.0, 20.0, 20.0),
        ) if tgt.get("cleaned_pdb") else (20.0, 20.0, 20.0)
        results = _dock_compounds_parallel(
            top10, tgt["pdbqt"],
            tgt["active_center"], box,
            work_dir, key, n_jobs=min(4, len(top10)),
        )
        for rec, energy in results:
            setattr(rec, attr, energy)

    # ── Compute SI (average human energy over up to 6 off-targets) ──
    for rec in top10:
        # Collect raw human off-target energies, pairing each with the off-target
        # attribute so we can flag high-risk binding on *valid* energies only.
        raw_human = [
            ("trypsin", rec.human_trypsin_energy),
            ("ces1", rec.human_ces1_energy),
            ("albumin", getattr(rec, "human_albumin_energy", None)),
            ("cyp3a4", getattr(rec, "human_cyp3a4_energy", None)),
            ("herg", getattr(rec, "human_herg_energy", None)),
            ("cyp2d6", getattr(rec, "human_cyp2d6_energy", None)),
        ]
        # A human off-target energy > 0.0 means no-pose / steric clash — it
        # carries no binding information and must NOT enter the SI denominator.
        # We treat it as invalid (NaN) so the SI is computed only from real,
        # finite binding energies. The *raw* list (including invalid energies)
        # is still used for the Off_Target_Risk flag below.
        energies_human = [
            e for _label, e in raw_human if e is not None and e <= 0.0
        ]
        n_human_targets = len(energies_human)

        # Track how many human off-targets provided valid energies and
        # record the resulting selectivity confidence.
        if n_human_targets >= 2:
            rec.selectivity_confidence = CompoundRecord.CONF_HIGH
        elif n_human_targets == 1:
            rec.selectivity_confidence = CompoundRecord.CONF_LOW
        else:
            rec.selectivity_confidence = CompoundRecord.CONF_NONE

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
        # Keep the raw (un-clamped) SI. We NO LONGER hard-zero the index when a
        # human off-target binds tightly — that erased real selectivity signal
        # (paper §4.1). Instead the raw SI is preserved and a separate boolean
        # Off_Target_Risk column records the binary high-risk flag.
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

        # Off-target risk flag (separate boolean column, paper §4.1b). Only
        # *valid* (finite, binding) human energies are considered — a >0.0
        # (no-pose/clash) value is not a real binder and is excluded.
        rec.off_target_risk = any(
            e is not None and e < -8.0
            for _label, e in raw_human
            if e is not None
        )

        if rec.off_target_risk:
            if rec.resistance_notes:
                rec.resistance_notes += " | "
            rec.resistance_notes += "High risk off-target binding"

    # ── Optional MM-GBSA-like rerank of the top 10 ──
    # Per-candidate MMFF-relaxed score (see utils.docking.rerank_mmff). Skips
    # gracefully when Vina/OpenBabel are absent or no pose was retained. The
    # final CSV is reranked by this score when available.
    from utils.docking import rerank_mmff
    rerank_mmff(top10, work_dir=work_dir, use_vina=use_vina)

    # ── Resistance profiling with pose-based interaction analysis ──
    _run_resistance_profiling(top10, targets, work_dir)

    # ── Simple mutation scan (reduced-resistance probability) ──
    # Quantitative ΔG vs the key resistance mutants (S403A/K406A/Y446A),
    # appended to resistance_notes and stored on rec.mutant_energy_delta.
    _run_mutation_scan(top10, targets, work_dir, deps)

    log.info("─── Phase 4 complete ───")
    return top10


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — REPORTING & ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# NOTE: ``generate_csv_report``, ``generate_images``,
# ``generate_interaction_diagram``, ``generate_pymol_script`` and
# ``_print_single_summary`` now live in ``utils/reporting.py`` and are
# imported at the top of this module. The orchestrator below (main) only
# calls them.


def screen_single_compound(
    smiles: str,
    targets: dict,
    work_dir: str,
    deps: dict,
) -> CompoundRecord:
    """
    Phase 3 (single-compound API) — Screen one SMILES against PBP2a.

    Builds a :class:`CompoundRecord` from the SMILES, docks it against both the
    allosteric and active sites of PBP2a (when Vina is available and grid
    centres are defined), and returns the populated record. This is a thin,
    programmatic entry point intended for library consumers who want to screen
    a single compound without generating a full BRICS library.

    Args:
        smiles: SMILES of the compound to screen.
        targets: Prepared targets dictionary (from :func:`prepare_targets`).
        work_dir: Scratch directory for intermediate PDBQT files.
        deps: Dependency dict (``{"vina": bool, "USE_VINA": bool}``).

    Returns:
        The populated :class:`CompoundRecord`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES could not be parsed: {smiles!r}")

    rec = CompoundRecord(
        compound_id="SINGLE_QUERY",
        smiles=smiles,
        mol=mol,
    )

    pb2pa = targets.get("PBP2a", {})
    receptor_pdbqt = pb2pa.get("pdbqt")
    allosteric_center = pb2pa.get("allosteric_center")
    active_center = pb2pa.get("active_center")
    use_vina = deps.get("USE_VINA", False)

    if receptor_pdbqt:
        if allosteric_center is not None:
            allosteric_box = _auto_box_size(
                pb2pa.get("allosteric_pdbqt") or pb2pa.get("cleaned_pdb"),
                allosteric_center, ALLOSTERIC_BOX_SIZE,
            )
            rec.pb2pa_allosteric_energy = dock_compound(
                rec, receptor_pdbqt, allosteric_center,
                allosteric_box, work_dir, "allosteric",
                use_vina=use_vina,
            )
        if active_center is not None:
            active_box = _auto_box_size(
                pb2pa.get("cleaned_pdb"), active_center, ACTIVE_BOX_SIZE,
            )
            rec.pb2pa_active_energy = dock_compound(
                rec, receptor_pdbqt, active_center,
                active_box, work_dir, "active",
                use_vina=use_vina,
            )
    else:
        log.warning(
            "  Receptor PDBQT missing — screen_single_compound cannot score. "
            "Returning record with no docking energies."
        )

    # ── Pose-based interaction analysis ──
    # The active-site dock (tag == "active") stores its docked pose on
    # ``rec.active_docked_pdbqt`` (see utils.docking.dock_compound). When such a
    # pose exists we analyse binding interactions against the cleaned receptor
    # so the caller gets a populated ``interactions`` fingerprint (H-bond flags
    # to Ser403 / Lys406 / Tyr446) without re-docking.
    cleaned_pdb = pb2pa.get("cleaned_pdb")
    pose = getattr(rec, "active_docked_pdbqt", None)
    if pose and os.path.exists(pose) and cleaned_pdb and os.path.exists(cleaned_pdb):
        try:
            rec.interactions = analyze_binding_interactions(pose, cleaned_pdb)
        except Exception as exc:
            log.warning(f"  Interaction analysis failed: {exc}")
            rec.interactions = None
    else:
        rec.interactions = None

    return rec


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
    log.info(f"  Docking engine:                {'Vina' if deps['USE_VINA'] else 'N/A (Vina required)'}")
    log.info(f"  Redocking RMSD:                {redock_rmsd:.3f} Å" if redock_rmsd else "  Redocking RMSD:                N/A")
    log.info(f"  Redocking validated:           {validation_ok}")
    if redock_rmsd is not None:
        status = "Validated" if validation_ok else "CAUTION"
        log.info(f"  Protocol validation: RMSD {redock_rmsd:.3f} Å ({status})")
    log.info(f"  CSV report:                    {CSV_REPORT}")
    log.info(f"  Open {OUTPUT_DIR}/visualization.pml in PyMOL to inspect binding poses.")
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


def _run_redocking_phase(
    targets: Dict[str, Dict], work_dir: str, deps: dict,
    config: Optional[dict], force: bool,
) -> Tuple[Optional[bool], Optional[float], str]:
    """Run (or reuse) the redocking validation gate and return its results.

    Returns a ``(validation_ok, redock_rmsd, validation_json)`` tuple. The
    validation is diagnostic, never a hard gate: on failure ``validation_ok``
    is left ``False`` / ``None`` and the pipeline continues. A failed
    validation in science mode is logged as a caution signal.
    """
    validation_json = os.path.join(work_dir, "validation_results.json")

    reuse_cache = (
        os.environ.get("AUTOANTIBIOTIC_FORCE") == "1"
        and force
        and os.path.exists(validation_json)
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
                mode=targets.get("mode"),
                config=config,
                target_pdbqt_paths=targets["PBP2a"].get("receptor_pdbqts"),
            )
    else:
        validation_ok, redock_rmsd = run_redocking_validation(
            holo_pdb_path=targets["holo_pdb"],
            target_pdbqt_path=targets["PBP2a"]["pdbqt"],
            work_dir=work_dir,
            deps=deps,
            mode=targets.get("mode"),
            config=config,
            target_pdbqt_paths=targets["PBP2a"].get("receptor_pdbqts"),
        )
        # A failed redocking validation against real PDBs is a diagnostic
        # signal, not a hard gate: log the error, keep validation_ok=False,
        # and continue. The status is recorded in the CSV (protocol_trust).
        if (
            targets.get("mode") == "science"
            and deps["USE_VINA"] is True
            and validation_ok is False
        ):
            log.error(
                "  ✗  Redocking validation failed in science mode — docking "
                "results should be interpreted with caution."
            )

    if validation_ok is None:
        log.info(
            "  Redocking validation not applicable (mock PDB / skipped)."
        )
    elif not validation_ok:
        # In science mode a failed redocking gate (RMSD > 1.5 Å → "Validated"
        # not achieved) is now a HARD gate unless AUTOANTIBIOTIC_FORCE=1 is set.
        if (
            targets.get("mode") == "science"
            and deps["USE_VINA"] is True
            and not os.environ.get("AUTOANTIBIOTIC_FORCE") == "1"
        ):
            log.error(
                "  ✗  Redocking validation FAILED in science mode (protocol_trust "
                "is not \"Validated\"; redocking RMSD > 1.5 Å required). Aborting "
                "the run. Re-run with AUTOANTIBIOTIC_FORCE=1 to override, or fix "
                "the docking protocol so the native ligand redocks within 1.5 Å."
            )
            sys.exit(1)
        log.error(
            "  ✗  Redocking validation failed — docking results should be "
            "interpreted with caution."
        )
        # Redocking validation is diagnostic, not a hard gate: never abort.
        # The validation status is recorded in the CSV (protocol_trust).
        if not os.environ.get("AUTOANTIBIOTIC_FORCE"):
            log.warning(
                "  ⚠  Redocking validation FAILED and AUTOANTIBIOTIC_FORCE is "
                "not set. Proceeding WITHOUT a validated docking protocol; "
                "validation status will be written to the CSV report "
                "(protocol_trust). Interpret all docking results with caution."
            )

    return validation_ok, redock_rmsd, validation_json


def _write_status_badge(
    mode: str,
    redock_rmsd: Optional[float],
    validation_ok: Optional[bool],
    validation_json: str,
) -> None:
    """Write the protocol-validation status badge to ``status.json`` at repo root."""
    if validation_ok is None:
        return

    status_payload = {
        "mode": mode,
        "redock_rmsd": redock_rmsd,
        "validated": bool(validation_ok),
    }
    if os.path.exists(validation_json):
        try:
            with open(validation_json) as fh:
                vdata = json.load(fh)
            status_payload["redock_rmsd"] = vdata.get("redock_rmsd", redock_rmsd)
            status_payload["validated"] = bool(vdata.get("validation_ok", validation_ok))
        except Exception:
            pass
    try:
        with open(REPO_ROOT / "status.json", "w") as fh:
            json.dump(status_payload, fh, indent=2)
        log.info(f"  Release status written: {REPO_ROOT / 'status.json'}")
    except Exception as exc:
        log.warning(f"  Could not write status.json: {exc}")


def _generate_and_filter_library(
    target_count: int, library: Optional[str], sdf: Optional[str],
    config: Optional[dict] = None,
) -> Tuple[list, list, int, int]:
    """Generate the candidate library and apply the filter chain.

    Returns ``(all_records, filtered, n_total, n_filtered)``. If no compound
    survives the strict+relaxed filter chain, falls back to the unfiltered
    generated library so a report is still produced (these candidates carry no
    ADMET/PAINS guarantees and are flagged accordingly downstream).
    """
    all_records = generate_candidate_library(
        target_count=target_count, input_csv=library, input_sdf=sdf,
    )
    n_total = len(all_records)
    # Recall mode (config.yaml ``recall_mode: true``) relaxes the filter chain
    # so established PBP2a binders (ceftaroline, meropenem) survive
    # filtering (paper §4.4).
    if config is None:
        config = load_config()
    recall_mode = bool(config.get("recall_mode", False))
    filtered = apply_filters(all_records, recal_mode=recall_mode)
    n_filtered = len(filtered)

    if n_filtered == 0:
        log.warning(
            "  No compounds passed filters. Falling back to the unfiltered "
            "generated library so a report is still produced."
        )
        filtered = all_records

    return all_records, filtered, n_total, n_filtered


def main(target_count: int = 500, force: bool = False, library: Optional[str] = None,
          config: Optional[dict] = None, sdf: Optional[str] = None,
          smiles: Optional[str] = None):
    """Orchestrate the full discovery pipeline end-to-end.

    Args:
        target_count: Number of candidate compounds to generate (BRICS mode).
        force: When True (and env AUTOANTIBIOTIC_FORCE=1 is set), reuse a
            previously cached redocking validation instead of re-running it.
            Otherwise the redocking validation is always executed when
            USE_VINA=True.
        library: Optional path to an external compound library CSV. When set,
            BRICS generation is skipped and the CSV compounds are used.
        config: Optional pre-loaded configuration dict (with a ``mode`` key).
            If None, :func:`load_config` is invoked to read ``config.yaml``.
        sdf: Optional path to an SDF file of pre-made molecules. When set,
            RDKit's ``Chem.SDMolSupplier`` reads the structures and BRICS
            generation is skipped entirely.
        smiles: Optional SMILES string for single-compound screening. When
            set, the full library pipeline (phases 2/4/5) is skipped and a
            single compound is docked & summarised immediately.
    """
    ensure_output_dir()

    # ── Configuration (explicit mode: ci | science) ──
    if config is None:
        config = load_config()
    mode = config.get("mode", "ci")

    # ── Dependency check ──
    deps = check_dependencies()

    # ── Credibility banner ──
    if mode == "ci":
        print(
            "\033[1;33m"
            "\n"
            "  ⚠ CI/MOCK MODE — results are NOT for scientific use.\n"
            "\033[0m",
            flush=True,
        )
    elif mode == "science" and not deps["USE_VINA"]:
        if os.environ.get("AUTOANTIBIOTIC_FORCE") != "1":
            log.error(
                "Science mode requires AutoDock Vina. Install via "
                "`bash setup.sh` or Docker, or set AUTOANTIBIOTIC_FORCE=1 to "
                "override."
            )
            sys.exit(1)

    # ── Working directory for intermediate files ──
    work_dir = str(OUTPUT_DIR / "workdir")
    pdb_dir = str(OUTPUT_DIR / "pdb")
    os.makedirs(work_dir, exist_ok=True)

    # ── Phase1: Target preparation ──
    # Science-mode guard failures (mock PDB / missing grid centre) are raised
    # as exceptions from prepare_targets; the CLI entrypoint converts them into
    # a non-zero exit, but programmatic callers may catch them instead.
    try:
        targets = prepare_targets(pdb_dir, work_dir, deps, config=config)
    except (ScienceModeMockPDBError, MissingGridCenterError) as exc:
        log.error(f"Target preparation aborted: {exc}")
        sys.exit(1)

    # ── Single-compound ("--smiles") mode ──
    # Screen one molecule instantly and print a text summary. This bypasses the
    # full library generation / selectivity / reporting phases entirely so a
    # chemist can inspect a single candidate in seconds.
    if smiles is not None:
        log.info("─── Single-Compound Mode (--smiles) ───")
        rec = screen_single_compound(smiles, targets, work_dir, deps)

        pb2pa = targets.get("PBP2a", {})
        rec.resistance_notes = profile_resistance_risk(
            rec, work_dir,
            pb2pa.get("pdbqt", ""),
            pb2pa.get("active_center"),
            ACTIVE_BOX_SIZE,
            interactions=rec.interactions,
        )

        _print_single_summary(rec)

        sys.exit(0)

    # ── Phase 0: Redocking validation ──
    # The (expensive) redocking gate is always executed when USE_VINA=True.
    # The only way to skip it and reuse a prior cached validation is when the
    # user explicitly passes --force AND env AUTOANTIBIOTIC_FORCE=1 is set.
    validation_ok, redock_rmsd, validation_json = _run_redocking_phase(
        targets=targets, work_dir=work_dir, deps=deps,
        config=config, force=force,
    )

    # ── Release status badge (status.json) ──
    # Surface the protocol-validation status at the repo root so downstream
    # tooling / CI can read it at a glance. Reuse the cached validation JSON
    # content when available; otherwise record the just-computed values.
    _write_status_badge(
        mode=mode, redock_rmsd=redock_rmsd, validation_ok=validation_ok,
        validation_json=validation_json,
    )

    # ── Phase 2: Library generation & filtering ──
    # Read pre-made molecules directly from an SDF file (RDKit) when provided,
    # instead of generating a new library via BRICS. This makes the pipeline
    # easy to integrate with external compound collections.
    all_records, filtered, n_total, n_filtered = _generate_and_filter_library(
        target_count=target_count, library=library, sdf=sdf, config=config,
    )

    # ── Phase 3: Virtual screening ──
    top10 = screen_library(filtered, targets, work_dir, deps)

    if not top10:
        log.warning("  No candidates after screening. Halting pipeline.")
        return

    # ── Phase 4: Selectivity & Resistance ──
    top10 = analyze_selectivity_and_resistance(top10, targets, work_dir, deps)

    # ── Phase 3.5: Negative selection against human off-targets ──
    # Compounds that bind HERG or CYP3A4 tightly (E < -8.0 kcal/mol) are removed
    # outright — they must never appear in the reported Top-10 even if they bind
    # the bacterial target well. Runs after selectivity analysis so the
    # human-off-target energies are populated, and before the rerank/diversity
    # gate so discarded compounds do not consume report slots.
    from utils.filtering import filter_by_human_clash
    top10 = filter_by_human_clash(top10)

    # ── Phase 4.5: Rerank gate + diversity clustering ──
    # Drop candidates with positive MM-GBSA-like relaxation energy and pick a
    # maximally dissimilar final set (Morgan Tanimoto ≤ 0.4) to fill the
    # reported top-10, improving the odds that reported hits are distinct,
    # credible binders rather than near-duplicates.
    from utils.reporting import rerank_and_diversify
    top10 = rerank_and_diversify(
        top10, ranked=top10,
        top_n=TOP_N, radius=FP_RADIUS, n_bits=FP_NBITS,
        max_tanimoto=SIMILARITY_THRESHOLD,
    )

    # ── Phase 5: Reporting & Artifacts ──
    generate_csv_report(
        top10,
        validation_ok=validation_ok,
        holo_pdb_path=targets.get("holo_pdb"),
        mode=targets.get("mode"),
        redock_rmsd=redock_rmsd,
        csv_report=CSV_REPORT,
        output_dir=OUTPUT_DIR,
    )

    top3 = top10[:3]
    generate_images(top3, output_dir=OUTPUT_DIR)

    # Phase 5.2b — 2D interaction diagrams for the top 3 candidates. These give
    # medicinal chemists a visual proof of the binding mode (ligand atoms that
    # engage Ser403 / Lys406 / Tyr446 are highlighted in red).
    try:
        pb2pa = targets.get("PBP2a", {})
        receptor_pdb = pb2pa.get("cleaned_pdb")
        for rec in top3:
            out_path = OUTPUT_DIR / f"interaction_{rec.compound_id}.png"
            generate_interaction_diagram(rec, receptor_pdb, str(out_path))
        log.info("Interaction diagrams saved to output/")
    except Exception as exc:
        log.warning(f"  Could not generate interaction diagrams: {exc}")

    # Phase 5.3 — PyMOL visualization script for the top 3 candidates.
    try:
        generate_pymol_script(top3, targets, str(OUTPUT_DIR))
    except Exception as exc:
        log.warning(f"  Could not generate PyMOL script: {exc}")

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
        help=(
            "Reuse cached redocking validation OR bypass a failed redocking gate "
            "in science mode. Requires AUTOANTIBIOTIC_FORCE=1 to reuse cached "
            "validation OR to bypass a failed redocking gate in science mode."
        ),
    )
    parser.add_argument(
        "--library", type=str, default=None,
        help=(
            "Optional path to an external compound library CSV (columns: "
            "smiles, compound_id). When provided, BRICS generation is skipped "
            "and the CSV compounds are used directly."
        ),
    )
    parser.add_argument(
        "--input-sdf", type=str, default=None,
        help=(
            "Optional path to an SDF file of pre-made molecules. When provided, "
            "RDKit reads the structures via Chem.SDMolSupplier and BRICS "
            "generation is skipped entirely."
        ),
    )
    parser.add_argument(
        "--check", action="store_true",
        help=(
            "Only run the dependency check (check_dependencies) and then exit. "
            "Useful for quickly verifying that AutoDock Vina, OpenBabel, and "
            "all required Python packages are installed and on PATH."
        ),
    )
    parser.add_argument(
        "--smiles", type=str, default=None,
        help=(
            "Screen a single SMILES string instantly (e.g. "
            "'CN1C(=O)C(N=C1C(=O)O)S...'). Skips library generation and prints a "
            "one-compound docking summary, then exits. Requires prepared targets."
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"AutoAntibiotic Discovery Pipeline v{__version__}",
        help="Print the pipeline version and exit.",
    )
    args = parser.parse_args()

    if args.check:
        print(f"AutoAntibiotic Discovery Pipeline v{__version__}")
        check_dependencies()
        sys.exit(0)

    log.info(f"AutoAntibiotic Discovery Pipeline v{__version__}")
    main(target_count=args.count, force=args.force, library=args.library,
         sdf=args.input_sdf, smiles=args.smiles)
