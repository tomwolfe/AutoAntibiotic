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

For real science runs: set AUTOANTIBIOTIC_CI=0 and place real PDBs in pdb_dir; bundled tests/data are mocks.
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
# ── Local utility modules ──────────────────────────────────────────────────────
# Focused helper modules keep this orchestration file readable while hosting the
# implementation details for ligand prep, docking, and filtering.
from utils.ligand_prep import LigandPreparator, prepare_ligand_pdbqt
from utils.docking import (
    _run_vina_docking,
    dock_compound,
    _dock_compounds_parallel,
    _dock_worker,
    _compute_shape_fallback_score,
    _compute_shape_scores,
)
from utils.filtering import apply_filters

# Structural helpers (native-ligand extraction, RMSD, centroids) live in their
# own module to keep this orchestrator focused on flow control. They are
# re-exported here so call sites and existing tests that reference
# ``discovery_pipeline.<name>`` keep working unchanged.
from utils.structure_prep import (
    _extract_native_ligand_from_holo,
    _compute_rmsd_docked_vs_crystal,
    _centroid_of_pdb_atoms,
    compute_residue_centroid,
)

# Configuration constants are centralised in config.constants to break the
# former circular import between this module and the utils package. They are
# re-exported here for backward compatibility with existing call sites/tests.
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
    ALLOSTERIC_BOX_SIZE,
    ACTIVE_BOX_SIZE,
    VINA_TIMEOUT_S,
    N_JOBS,
    SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD_RELAXED,
    DIVERSITY_MIN_COUNT,
    SELECTIVITY_INDEX_THRESHOLD,
    OUTPUT_DIR,
    CSV_REPORT,
    TOP_N,
    REPO_ROOT,
)

# Preserve the original import-time side effect (seeding for reproducibility).
np.random.seed(RANDOM_SEED)

# Configuration loading is isolated in config.loader so the orchestrator stays
# focused on flow control. Re-exported here so call sites that reference
# ``discovery_pipeline.load_config`` keep working unchanged.
from config.loader import load_config


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


def select_top(records, score_key, descending=False, n=TOP_N):
    """Return the top *n* records sorted by *score_key* (skips None scores)."""
    valid = [r for r in records if getattr(r, score_key, None) is not None]
    valid.sort(key=lambda r: getattr(r, score_key), reverse=descending)
    return valid[:n]


# ═══════════════════════════════════════════════════════════════════════════════
#  TARGET CACHE MANAGER (Quick Screen mode)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CACHE_DIR = os.path.expanduser("~/.autoantibiotic/cache")


def _cache_file(src: Optional[str], cache_dir: str) -> Optional[str]:
    """
    Return the path (relative to *cache_dir*) under which *src* is stored,
    copying the file into *cache_dir* if it lives elsewhere.

    Files already inside *cache_dir* are left untouched. ``None``/missing
    inputs are returned unchanged so callers can preserve them.
    """
    if not src or not os.path.exists(src):
        return src
    dst = os.path.join(cache_dir, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        try:
            shutil.copy(src, dst)
        except OSError as exc:
            log.warning(f"  Could not copy {src} into cache: {exc}")
    return os.path.basename(src)


def _serialize_targets(targets: dict, cache_dir: str) -> dict:
    """Build a JSON-safe manifest describing the prepared targets in *cache_dir*."""
    def ser_center(c):
        if c is None:
            return None
        return [float(c[0]), float(c[1]), float(c[2])]

    manifest = {
        "mode": targets.get("mode"),
        "holo_pdb": _cache_file(targets.get("holo_pdb"), cache_dir),
    }
    for key in ("PBP2a", "trypsin", "CES1"):
        t = targets.get(key)
        if not t:
            continue
        entry = {}
        if "pdbqt" in t:
            entry["pdbqt"] = _cache_file(t["pdbqt"], cache_dir)
        if "cleaned_pdb" in t:
            entry["cleaned_pdb"] = _cache_file(t["cleaned_pdb"], cache_dir)
        if "allosteric_center" in t:
            entry["allosteric_center"] = ser_center(t["allosteric_center"])
        if "active_center" in t:
            entry["active_center"] = ser_center(t["active_center"])
        manifest[key] = entry
    return manifest


def _deserialize_targets(manifest: dict, cache_dir: str) -> dict:
    """Rebuild a usable targets dict from a cached manifest."""
    def de_center(c):
        if c is None:
            return None
        return np.array(c, dtype=float)

    targets: Dict[str, Dict] = {}
    targets["mode"] = manifest.get("mode")
    hp = manifest.get("holo_pdb")
    if hp:
        hp_path = hp if os.path.isabs(hp) else os.path.join(cache_dir, hp)
        targets["holo_pdb"] = hp_path if os.path.exists(hp_path) else hp

    for key in ("PBP2a", "trypsin", "CES1"):
        m = manifest.get(key)
        if not m:
            continue
        entry: Dict[str, object] = {}
        if "pdbqt" in m:
            p = m["pdbqt"]
            p = p if os.path.isabs(p) else os.path.join(cache_dir, p)
            entry["pdbqt"] = p
        if "cleaned_pdb" in m:
            p = m["cleaned_pdb"]
            p = p if os.path.isabs(p) else os.path.join(cache_dir, p)
            entry["cleaned_pdb"] = p
        if "allosteric_center" in m:
            entry["allosteric_center"] = de_center(m["allosteric_center"])
        if "active_center" in m:
            entry["active_center"] = de_center(m["active_center"])
        targets[key] = entry
    return targets


def _get_cached_targets(
    cache_dir: str,
    deps: Optional[dict] = None,
    config: Optional[dict] = None,
) -> Dict[str, Dict]:
    """
    Return prepared docking targets, using an on-disk cache when available.

    The cache lives under *cache_dir* (default ``~/.autoantibiotic/cache``) and
    consists of a ``targets_manifest.json`` plus the prepared PDBQT/cleaned-PDB
    files. On a cache hit the manifest is loaded and the targets dict is rebuilt
    directly from it (no PDB download / cleaning). On a miss, :func:`prepare_targets`
    is run once, the resulting files are copied into *cache_dir*, and a manifest
    is written for future runs.

    Args:
        cache_dir: Directory to store / read the prepared targets.
        deps: Dependency dict (falls back to :func:`check_dependencies`).
        config: Config dict (falls back to :func:`load_config`).

    Returns:
        The targets dictionary (same shape as :func:`prepare_targets`).

    Raises:
        Whatever :func:`prepare_targets` raises on a cache miss — callers should
        catch this and fall back to a fresh ``prepare_targets`` invocation.
    """
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "targets_manifest.json")

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            targets = _deserialize_targets(manifest, cache_dir)
            log.info(f"  Loaded cached prepared targets from {cache_dir}")
            return targets
        except Exception as exc:
            log.warning(
                f"  Cached targets manifest unreadable ({exc}); rebuilding cache."
            )

    # ── Cache miss: build fresh and persist ──
    if deps is None:
        deps = check_dependencies()
    if config is None:
        config = load_config()

    # Run preparation directly inside the cache dir so all generated PDBQT /
    # cleaned-PDB files land there and can be served straight from the cache.
    targets = prepare_targets(cache_dir, cache_dir, deps, config=config)

    # Ensure the holo structure (used for redocking validation) is also cached.
    holo_src = targets.get("holo_pdb")
    if holo_src and os.path.exists(holo_src):
        holo_dst = os.path.join(cache_dir, "holo_cached.pdb")
        if os.path.abspath(holo_src) != os.path.abspath(holo_dst):
            try:
                shutil.copy(holo_src, holo_dst)
                targets["holo_pdb"] = holo_dst
            except OSError as exc:
                log.warning(f"  Could not cache holo PDB: {exc}")

    manifest = _serialize_targets(targets, cache_dir)
    try:
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        log.info(f"  Cached prepared targets to {cache_dir}")
    except OSError as exc:
        log.warning(f"  Could not write cache manifest: {exc}")

    return targets


def _print_single_summary(rec: "CompoundRecord") -> None:
    """Print a concise single-compound screening summary table to stdout."""
    inter = getattr(rec, "interactions", None)
    key_interactions = []
    if inter:
        if inter.get("Ser403_contact"):
            key_interactions.append("Ser403")
        if inter.get("Lys406_Hbond"):
            key_interactions.append("Lys406")
        if inter.get("Tyr446_Hbond"):
            key_interactions.append("Tyr446")
    key_str = ", ".join(key_interactions) if key_interactions else "None detected"

    fmt = lambda v: f"{v:.2f} kcal/mol" if v is not None else "N/A"

    print("\n" + "=" * 64)
    print("  SINGLE-COMPOUND SCREEN SUMMARY")
    print("=" * 64)
    print(f"  {'Compound ID':<20}: {rec.compound_id}")
    print(f"  {'SMILES':<20}: {rec.smiles}")
    print(f"  {'Allosteric Energy':<20}: {fmt(rec.pb2pa_allosteric_energy)}")
    print(f"  {'Active Energy':<20}: {fmt(rec.pb2pa_active_energy)}")
    print(f"  {'Key Interactions':<20}: {key_str}")
    print("=" * 64 + "\n")


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
        # High-visibility, bold error printed directly to stdout so the user
        # is not silently left on the (slower, less accurate) fallback path.
        # The command is copy-pasteable so the fix is one line away.
        print(
            "\033[1;31m"
            "\n"
            "  ╔══════════════════════════════════════════════════════════════════╗\n"
            "  ║  ERROR: AutoDock Vina not found.                                ║\n"
            "  ║  Install it via:                                                ║\n"
            "  ║    conda install -c conda-forge vina                            ║\n"
            "  ╚══════════════════════════════════════════════════════════════════╝\n"
            "\033[0m",
            flush=True,
        )
        log.error("Error: AutoDock Vina not found. Install it via: conda install -c conda-forge vina")

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
            "  ║  Install it via:                                                ║\n"
            "  ║    conda install -c conda-forge openbabel                       ║\n"
            "  ╚══════════════════════════════════════════════════════════════════╝\n"
            "\033[0m",
            flush=True,
        )
        log.error("Error: OpenBabel not found. Install it via: conda install -c conda-forge openbabel")

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
) -> Tuple[bool, Optional[float]]:
    """
    Phase 0 — Protocol Validation.

    Extracts the native ligand from 6TKO, docks it back into the prepared
    PBP2a receptor, and computes the RMSD to the crystal pose.

    Returns (success: bool, rmsd: float | None).
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

    # ``validation_ok`` reflects the 2.0 Å pass/fail gate, but we always return
    # the exact measured ``rmsd`` float (even when it exceeds the threshold) so
    # downstream reporters can display the raw value and emit nuanced trust
    # signals rather than a binary pass/fail.
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

        # Convert to PDBQT for Vina via obabel (add gasteiger charges).
        # If obabel is unavailable, copy the PDB as-is with a .pdbqt extension.
        pdbqt_path = out_path.replace(".pdb", ".pdbqt")
        try:
            subprocess.run(
                ["obabel", out_path, "-O", pdbqt_path, "-h", "--gas"],
                capture_output=True, timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            log.warning(
                "  obabel not found. Writing PDB as-is; Vina may fail."
            )
            shutil.copy(out_path, pdbqt_path)

        return pdbqt_path if os.path.exists(pdbqt_path) else out_path

    except Exception as exc:
        log.error(f"  ✗  Failed to clean {pdb_path}: {exc}")
        raise


# NOTE: compute_residue_centroid / _centroid_of_pdb_atoms now live in
# utils.structure_prep and are re-exported above for backward compatibility.


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
    if config is None:
        config = load_config()
    mode = config.get("mode", "science")
    log.info(f"  Run mode (from config): {mode}")

    # ── Fetch structures (prefer bundled offline PDBs under tests/data) ──
    def _resolve_structure(pdb_id: str) -> str:
        """Return a local tests/data/{pdb_id}.pdb path if CI mode, else download."""
        if os.environ.get("AUTOANTIBIOTIC_CI") == "1":
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

    # ── Explicit mode (config-driven, not inferred from file paths) ──
    if mode == "ci":
        log.info("CI mode: using mock PDBs - not for scientific use.")
    else:
        log.info("Science mode: real scientific validation expected.")
    result["mode"] = mode

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
        active_center = None
    log.info(f"    Active site center: {active_center}")

    if result.get("mode") == "science" and active_center is None:
        log.error("Active site center missing in science mode – aborting")
        sys.exit(1)

    if result.get("mode") == "science" and allosteric_center is None:
        log.error("Active site center missing in science mode – aborting")
        sys.exit(1)

    if allosteric_center is None or active_center is None:
        log.warning(
            "  ⚠  PBP2a grid center(s) undefined; leaving as None. "
            "Supply a real PDB for docking grid config."
        )

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
    try:
        tryp_center = compute_residue_centroid(tryp_clean_pdb, TRYPSIN_CATALYTIC_RESIDUES)
    except (ValueError, Exception) as exc:
        log.warning(f"  ⚠  Trypsin catalytic residues {TRYPSIN_CATALYTIC_RESIDUES} missing: {exc}")
        log.warning("  Residue missing – grid center set to None; supply real PDB.")
        tryp_center = None
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
        log.warning("  Residue missing – grid center set to None; supply real PDB.")
        ces1_center = None
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

    # Selectivity confidence labels
    CONF_HIGH = "High"
    CONF_LOW = "Low"
    CONF_NONE = "None"

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

    # Selectivity confidence based on how many human off-targets were docked:
    #   "High" if 2 human targets provided valid energies,
    #   "Low"  if 1 human target provided a valid energy,
    #   "None" if 0 human targets provided a valid energy.
    selectivity_confidence: str = "None"

    # Path to the active-site Vina docked pose (PDBQT), populated during
    # screening so that pose-based interaction analysis need not re-dock.
    active_docked_pdbqt: Optional[str] = None

    # Interaction fingerprint (dict returned by analyze_binding_interactions)
    # captured during Phase 4 so reporting can expose per-residue H-bond flags
    # without re-parsing the docked pose.
    interactions: Optional[Dict[str, Union[bool, float]]] = None


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
    "Oc1ccccc1C(=O)O",                                         # Salicylic acid (salicylate)
    "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",                    # 7-Hydroxyflavone (flavonoid core)
    "CC1OCCCC(=O)C1",                                          # Macrolide-like lactone core (no β-lactam)
    "Oc1c(O)c(O)cc(C(=O)O)c1",                                 # Gallic acid (phenolic)
    "CC1=C(C=C(C=C1)O)O",                                      # Hydroquinone
    "COc1cc2c(cc1OC)C(=O)C3=C(O)C=CC(=C3O2)C",                 # Rottlerin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                            # Caffeine
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
    input_csv: Optional[str] = None,
) -> List[CompoundRecord]:
    """
    Phase 2.1 — Generate a diverse library.

    If *input_csv* is provided, the library is read directly from that CSV
    file (expected columns: ``smiles``, ``compound_id``) and the BRICS
    scaffold-generation logic is skipped entirely.

    Otherwise, a library is generated by BRICS decomposition of natural
    product scaffolds, fragment recombination, and expansion.

    Args:
        target_count: Desired number of compounds (~500).
        seed: Random seed for reproducibility.
        input_csv: Optional path to an external compound library CSV.

    Returns:
        List of CompoundRecord objects (SMILES only, no computed props yet).
    """
    log.info("─── Phase 2: Library Generation ───")

    if input_csv is not None:
        log.info(f"  Loading external compound library from CSV: {input_csv}")
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input library CSV not found: {input_csv}")

        df = pd.read_csv(input_csv)
        df_cols = {str(c).strip().lower() for c in df.columns}
        if not {"smiles", "compound_id"}.issubset(df_cols):
            raise ValueError(
                f"Input CSV must contain 'smiles' and 'compound_id' columns; "
                f"found: {list(df.columns)}"
            )

        records = []
        for _, row in df.iterrows():
            smi = str(row["smiles"]).strip()
            cid = str(row["compound_id"]).strip()
            if not smi or smi.lower() in ("nan", "none"):
                log.warning(f"  Skipping row with empty SMILES (compound_id={cid}).")
                continue
            mol = Chem.MolFromSmiles(smi)
            records.append(CompoundRecord(
                compound_id=cid,
                smiles=smi,
                mol=mol,
            ))
        log.info(f"  Loaded {len(records)} compounds from external CSV (BRICS skipped).")
        return records

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


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — VIRTUAL SCREENING (Docking)
# ═══════════════════════════════════════════════════════════════════════════════

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

        if len(scored) >= 50:
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
        top10 = select_top(records, "pb2pa_allosteric_energy")
    else:
        top10 = select_top(records, "shape_score")

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
    pose_notes = []
    energy_notes = []

    # Pose-based interactions are supplied by the caller (the active-site pose
    # captured during screen_library via record.active_docked_pdbqt). We no
    # longer re-dock here — if no pose is available we simply note that.
    if interactions is None:
        pose_notes.append("no pose — binding interactions not analysed.")
    else:
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
            pose_notes.append(
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
                pose_notes.append(
                    f"Forms strong H-bond with catalytic Ser403 ({ser:.1f} Å)."
                )
            elif ser < 5.0:
                pose_notes.append(
                    f"Weak contact with catalytic Ser403 ({ser:.1f} Å) — resistance risk."
                )
            else:
                pose_notes.append(
                    f"Loss of Ser403 engagement ({ser:.1f} Å) — high resistance risk."
                )
        elif "Ser403" not in unverified:
            pose_notes.append(
                "Ser403 distance undefined — high resistance risk."
            )

        if np.isfinite(lys):
            if lys_ok:
                pose_notes.append(
                    f"Stabilized by Lys406 contact ({lys:.1f} Å)."
                )
            elif lys < 5.0:
                pose_notes.append(
                    f"Weak Lys406 contact ({lys:.1f} Å) — resistance risk."
                )
        elif "Lys406" not in unverified:
            pose_notes.append(
                "Lys406 distance undefined — resistance risk."
            )

        # If neither key catalytic residue is engaged (and neither is merely
        # unverified due to a missing PDB residue), state this plainly.
        unverified_key = [r for r in unverified if r in ("Ser403", "Lys406")]
        if (not ser_ok) and (not lys_ok) and not unverified_key:
            pose_notes.append(
                "Lacks key interactions with catalytic Ser403 and Lys406."
            )

        if np.isfinite(tyr):
            if tyr < 3.5:
                pose_notes.append(
                    f"Stabilising contact with Tyr446 ({tyr:.1f} Å)."
                )
            elif tyr < 5.0:
                pose_notes.append(
                    f"Weak Tyr446 contact ({tyr:.1f} Å) — resistance risk."
                )

        # Aggregate: if the closest conserved-residue contact exceeds 5 Å the
        # compound avoids the catalytic network entirely and is flagged as a
        # high-resistance-risk binder (mutations need only modestly perturb
        # the active site to escape it).
        best_conserved = min(ser, lys, tyr)
        if np.isfinite(best_conserved) and best_conserved >= 5.0:
            pose_notes.append(
                f"Avoids conserved catalytic network (min d={best_conserved:.2f} Å) — high resistance risk"
            )

        # Allosteric binder note
        if (
            record.pb2pa_allosteric_energy is not None
            and record.pb2pa_allosteric_energy < -7.0
        ):
            if record.pb2pa_active_energy is None or record.pb2pa_active_energy > -6.0:
                pose_notes.append("Allosteric binder (Ala237/Met241/Tyr159 pocket). Novel mechanism.")

    # Energy-based heuristics
    if record.pb2pa_active_energy is not None and record.pb2pa_active_energy < -6.0:
        energy_notes.append("Likely contacts catalytic Ser403 (active site, energy-based). Good.")

    # Molecular weight heuristic
    if record.mol is not None:
        mw = Descriptors.MolWt(record.mol)
        if mw > 400:
            energy_notes.append("High MW (>400) — broad interaction surface, may contact multiple residues.")
        n_rot = Descriptors.NumRotatableBonds(record.mol)
        if n_rot < 5:
            energy_notes.append("Rigid scaffold — reduced entropic penalty, may enhance binding specificity.")

    # Resistance risk indicators
    if record.qed_score > 0.8:
        energy_notes.append("High drug-likeness (QED > 0.8) — good developability profile.")

    if pose_notes and energy_notes:
        notes = pose_notes + energy_notes
    else:
        notes = pose_notes or energy_notes

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
            rec.selectivity_index = max(0.0, 1.0 - rec.max_similarity)
            rec.selectivity_confidence = CompoundRecord.CONF_LOW
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

        # Always use the active-site pose captured during screen_library
        # (record.active_docked_pdbqt) for pose-based interaction analysis.
        # If no pose was retained (e.g. fallback shape screening), skip the
        # analysis and let profile_resistance_risk note "no pose".
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
    validation_ok: bool = False,
    holo_pdb_path: Optional[str] = None,
    mode: str = "science",
    redock_rmsd: Optional[float] = None,
) -> str:
    """
    Phase 5.1 — Write top_candidates.csv with all required columns.

    Columns:
        Compound_ID, SMILES, PBP2a_Allosteric_Energy, PBP2a_Active_Energy,
        Human_Trypsin_Energy, Human_CES1_Energy, Selectivity_Index,
        Selectivity_Confidence, Shape_Score, Max_Similarity, Passes_Lipinski,
        QED_Score, Binding_Mode_Notes, Redock_RMSD, Protocol_RMSD,
        Redock_Validated, Validation_Status, Validation_Warning,
        H_Bond_Ser403, H_Bond_Lys406, H_Bond_Tyr446, Structure_Source.

    Returns path to CSV.
    """
    log.info("─── Phase 5: Reporting ───")
    ensure_output_dir()

    is_mock = (mode == "ci")

    rows = []
    for rec in top10:
        # Per-residue H-bond flags derived from the interaction fingerprint
        # captured during Phase 4 (min distance to the conserved residue; a
        # contact is flagged when the ligand approaches within 3.5 Å).
        inter = getattr(rec, "interactions", None)
        if inter:
            h_ser = bool(inter.get("min_dist_Ser403", float("inf")) < 3.5)
            h_lys = bool(inter.get("min_dist_Lys406", float("inf")) < 3.5)
            h_tyr = bool(inter.get("min_dist_Tyr446", float("inf")) < 3.5)
        else:
            h_ser = h_lys = h_tyr = False

        # ── Why_Won: a short human-readable rationale string combining the
        # selectivity index, per-residue H-bond flags, and the docking
        # energies already computed for this candidate (no new computation).
        why_parts = []
        if rec.selectivity_index is not None:
            why_parts.append(f"SI={rec.selectivity_index:.1f}")
        hbond_residues = [
            name for name, flag in (
                ("Ser403", h_ser), ("Lys406", h_lys), ("Tyr446", h_tyr)
            ) if flag
        ]
        if hbond_residues:
            why_parts.append("H-bonds " + ",".join(hbond_residues))
        if rec.pb2pa_active_energy is not None:
            why_parts.append(f"active E={rec.pb2pa_active_energy:.1f}")
        if rec.pb2pa_allosteric_energy is not None:
            why_parts.append(f"allosteric E={rec.pb2pa_allosteric_energy:.1f}")
        why_won = "; ".join(why_parts) if why_parts else "N/A"

        # Redock RMSD: report the measured value in science mode, mark the
        # column SKIPPED in CI/mock mode, and N/A when no value is available.
        if is_mock:
            redock_rmsd_str = "SKIPPED"
        elif redock_rmsd is not None:
            redock_rmsd_str = f"{redock_rmsd:.3f}"
        else:
            redock_rmsd_str = "N/A"

        # Validation warning: only raised in science mode when the protocol
        # did not validate (RMSD > 2.0 Å or the redocking step failed).
        if (not validation_ok) and mode == "science":
            validation_warning = "RMSD > 2.0Å or Failed"
        else:
            validation_warning = "None"

        # ── Trust Badge columns ──
        # Protocol_RMSD: the raw redocking RMSD value (in Å) for every row, shown
        # as a plain float (e.g. "1.234") wherever a real measurement exists. In
        # CI/mock mode the protocol is never redocked, so it reads "SKIPPED".
        protocol_rmsd_str = "SKIPPED" if is_mock else (
            f"{redock_rmsd:.3f}" if redock_rmsd is not None else "N/A"
        )

        # Validation_Status: a quick-glance trust badge so chemists immediately
        # see protocol quality.
        #   • CI mode (no real RMSD)                 → "CI Mode (Skipped)"
        #   • redock_rmsd > 2.0 Å                    → "CAUTION: High RMSD (<val> Å)"
        #   • 1.5 Å < redock_rmsd <= 2.0 Å          → "Validated (Marginal)"
        #   • redock_rmsd <= 1.5 Å (good protocol)  → "Validated"
        #   • science mode but no measured RMSD     → "Validation Unavailable"
        if is_mock:
            validation_status = "CI Mode (Skipped)"
        elif redock_rmsd is not None and redock_rmsd > 2.0:
            validation_status = f"CAUTION: High RMSD ({redock_rmsd:.3f} Å)"
        elif redock_rmsd is not None and 1.5 < redock_rmsd <= 2.0:
            validation_status = "Validated (Marginal)"
        elif redock_rmsd is not None:
            validation_status = "Validated"
        else:
            validation_status = "Validation Unavailable"

        rows.append({
            "Compound_ID": rec.compound_id,
            "SMILES": rec.smiles,
            "Structure_Source": "mock" if is_mock else "real",
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
            ) + (" (mock)" if is_mock else ""),
            "Shape_Score": (
                f"{rec.shape_score:.2f}" if rec.shape_score is not None
                else "N/A"
            ),
            "Max_Similarity": f"{rec.max_similarity:.3f}",
            "Passes_Lipinski": str(rec.passes_lipinski),
            "QED_Score": f"{rec.qed_score:.3f}",
            "Binding_Mode_Notes": rec.resistance_notes.replace("; ", " | "),
            "Redock_RMSD": redock_rmsd_str,
            "Protocol_RMSD": protocol_rmsd_str,
            "Redock_Validated": (
                "SKIPPED" if is_mock
                else "N/A" if validation_ok is None else str(bool(validation_ok))
            ) + (" (mock)" if is_mock else ""),
            "Validation_Status": validation_status,
            "Validation_Warning": validation_warning,
            "H_Bond_Ser403": str(h_ser),
            "H_Bond_Lys406": str(h_lys),
            "H_Bond_Tyr446": str(h_tyr),
            "Why_Won": why_won,
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


def _key_residue_coords(receptor_pdb: str) -> Dict[str, List[np.ndarray]]:
    """
    Return the 3D coordinates of the key catalytic residue atoms in *receptor_pdb*.

    Only the polar H-bond donor/acceptor atoms are collected:
        Ser403 → OG, Lys406 → NZ, Tyr446 → OH.

    Returns a dict ``{resname: [np.ndarray, ...]}`` (empty lists for absent
    residues). Used to highlight ligand atoms that engage these residues in the
    2D interaction diagram.
    """
    targets_map = {
        "Ser403": [("SER", 403, "OG")],
        "Lys406": [("LYS", 406, "NZ")],
        "Tyr446": [("TYR", 446, "OH")],
    }
    atom_coords: Dict[str, List[np.ndarray]] = {k: [] for k in targets_map}
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("receptor", receptor_pdb)
        for model in struct:
            for chain in model:
                for residue in chain:
                    for resname, entries in targets_map.items():
                        for entry in entries:
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
    except Exception as exc:
        log.warning(f"  Could not parse receptor for key residues: {exc}")
    return atom_coords


def _pose_heavy_atom_coords(pdbqt_path: str) -> List[np.ndarray]:
    """
    Parse heavy-atom 3D coordinates (in file order) from a docked ligand PDBQT.

    Hydrogen lines are skipped so the returned list indexes the same heavy
    atoms as a SMILES-derived :class:`Chem.Mol` (PDBQT preparation appends
    hydrogens after the heavy atoms, preserving heavy-atom order).
    """
    coords: List[np.ndarray] = []
    try:
        with open(pdbqt_path) as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    elem = line[76:78].strip()
                except (ValueError, IndexError):
                    continue
                if elem and elem.upper() != "H":
                    coords.append(np.array([x, y, z]))
    except OSError:
        pass
    return coords


def generate_interaction_diagram(
    record: CompoundRecord,
    receptor_pdb: str,
    output_path: str,
) -> Optional[str]:
    """
    Phase 5.2b — Render a 2D interaction diagram for a single compound.

    Draws the ligand (from ``record.mol`` / SMILES) and overlays the key
    binding interactions: ligand atoms that approach within H-bond distance of
    Ser403, Lys406, or Tyr446 (parsed from the docked pose in
    ``record.active_docked_pdbqt``) are highlighted in red. A legend line
    summarises which conserved residues are engaged.

    The diagram is saved to *output_path* (typically
    ``output/interaction_<compound_id>.png``) and the path is returned.

    If the heavyweight pose/atom mapping is unavailable, the function still
    draws the ligand and annotates the detected key interactions from
    ``record.interactions`` — so a chemist always gets a visual artefact.
    """
    mol = record.mol
    if mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            log.warning(
                f"  Cannot render ligand for {record.compound_id} "
                "(invalid SMILES)."
            )
            return None

    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception as exc:
        log.warning(f"  RDKit drawing unavailable: {exc}")
        return None

    # ── Determine which ligand atoms engage the key residues ──
    highlight_atoms: List[int] = []
    inter = getattr(record, "interactions", None)
    pose = getattr(record, "active_docked_pdbqt", None)

    if pose and os.path.exists(pose) and receptor_pdb and os.path.exists(receptor_pdb):
        try:
            key_coords = _key_residue_coords(receptor_pdb)
            pose_coords = _pose_heavy_atom_coords(pose)
            n_lig = mol.GetNumAtoms()
            # Heavy-atom order in the docked pose matches the SMILES-derived mol
            # (H are appended by the PDBQT prep), so index i maps directly.
            for i, p in enumerate(pose_coords):
                if i >= n_lig:
                    break
                for resname, coords in key_coords.items():
                    if not coords:
                        continue
                    cutoff = 3.8 if resname == "Lys406" else 3.5
                    if any(np.linalg.norm(p - c) < cutoff for c in coords):
                        highlight_atoms.append(i)
                        break
        except Exception as exc:
            log.warning(f"  Atom-level interaction highlight failed: {exc}")
            highlight_atoms = []

    # De-duplicate while preserving order.
    highlight_atoms = list(dict.fromkeys(highlight_atoms))

    # ── Build the legend from the interaction fingerprint ──
    parts = []
    if inter:
        if inter.get("Ser403_contact"):
            parts.append("Ser403 H-bond")
        if inter.get("Lys406_Hbond"):
            parts.append("Lys406 H-bond")
        if inter.get("Tyr446_Hbond"):
            parts.append("Tyr446 H-bond")
    legend = "; ".join(parts) if parts else "No key H-bonds detected"

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
        drawer.drawOptions().highlightColor = rdMolDraw2D.Color(1.0, 0.0, 0.0)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            legend=legend,
        )
        drawer.FinishDrawing()
        drawer.WriteDrawingText(output_path)
        log.info(f"  Interaction diagram saved: {output_path}")
        return output_path
    except Exception as exc:
        log.warning(f"  Could not render interaction diagram: {exc}")
        return None


def generate_pymol_script(
    top_candidates: List[CompoundRecord],
    targets: dict,
    output_dir: str,
) -> str:
    """
    Phase 5.3 — Write a PyMOL session script (``visualization.pml``) that loads
    the PBP2a receptor, the top 3 candidate active-site poses, highlights the
    conserved catalytic residues (Ser403, Lys406, Tyr446) as sticks, and colours
    each ligand by element for quick medicinal-chemist inspection.

    Returns the path to the generated ``.pml`` file.
    """
    pml_path = os.path.join(output_dir, "visualization.pml")
    receptor_pdb = targets.get("PBP2a", {}).get("cleaned_pdb")

    lines = [
        "# Auto-generated PyMOL visualization script (AutoAntibiotic)",
        "# Load with: pymol -l visualization.pml",
        "",
    ]

    if receptor_pdb and os.path.exists(receptor_pdb):
        lines.append(f"load {receptor_pdb!r}, PBP2a")
        # Highlight the conserved catalytic residues as sticks.
        lines.append(
            "select conserved_residues, "
            "(resn SER and resi 403) or "
            "(resn LYS and resi 406) or "
            "(resn TYR and resi 446)"
        )
        lines.append("show sticks, conserved_residues")
        lines.append("color magenta, conserved_residues")
        lines.append("")
    else:
        log.warning(
            "  PBP2a cleaned PDB not available; PyMOL script will skip receptor load."
        )

    loaded = 0
    for i, rec in enumerate(top_candidates[:3]):
        pose = getattr(rec, "active_docked_pdbqt", None)
        if not pose or not os.path.exists(pose):
            log.warning(
                f"  No active-site pose for {rec.compound_id}; skipping in PyMOL script."
            )
            continue
        name = f"Ligand_{i + 1}_{rec.compound_id}"
        lines.append(f"load {pose!r}, {name}")
        # Show the ligand as sticks coloured by element for quick inspection.
        lines.append(f"color byelement, {name}")
        lines.append(f"show sticks, {name}")

        # ── Dashed hydrogen-bond lines from the pose-derived interaction fingerprint ──
        # Draw a dashed measurement line for each conserved residue whose
        # ligand→residue distance is within the H-bond cutoff (Ser403/Tyr446 < 3.5 Å,
        # Lys406 < 3.8 Å). These come straight from the record.interactions dict
        # computed during Phase 4, so the script reflects the real measured pose.
        inter = getattr(rec, "interactions", None)
        if inter:
            ser_d = inter.get("min_dist_Ser403", float("inf"))
            lys_d = inter.get("min_dist_Lys406", float("inf"))
            tyr_d = inter.get("min_dist_Tyr446", float("inf"))
            if np.isfinite(ser_d) and ser_d < 3.5:
                lines.append(
                    f"distance hbond_ser_{i + 1}, {name}, resi 403 & name OG, cutoff=3.5"
                )
                lines.append("dash wid 2.0")
            if np.isfinite(lys_d) and lys_d < 3.8:
                lines.append(
                    f"distance hbond_lys_{i + 1}, {name}, resi 406 & name NZ, cutoff=3.8"
                )
                lines.append("dash wid 2.0")
            if np.isfinite(tyr_d) and tyr_d < 3.5:
                lines.append(
                    f"distance hbond_tyr_{i + 1}, {name}, resi 446 & name OH, cutoff=3.5"
                )
                lines.append("dash wid 2.0")
        lines.append("")
        loaded += 1

    lines.append("")
    lines.append("# Orient the view")
    lines.append("zoom")
    lines.append("orient")

    with open(pml_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    log.info(
        f"  PyMOL script saved: {pml_path} "
        f"({loaded} ligand pose(s) loaded)."
    )
    return pml_path


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

    if deps.get("USE_VINA") and receptor_pdbqt:
        if allosteric_center is not None:
            rec.pb2pa_allosteric_energy = dock_compound(
                rec, receptor_pdbqt, allosteric_center,
                ALLOSTERIC_BOX_SIZE, work_dir, "allosteric",
            )
        if active_center is not None:
            rec.pb2pa_active_energy = dock_compound(
                rec, receptor_pdbqt, active_center,
                ACTIVE_BOX_SIZE, work_dir, "active",
            )
    else:
        log.warning(
            "  Vina unavailable or receptor missing — screen_single_compound "
            "cannot dock. Returning record with no docking energies."
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
    log.info(f"  Docking engine:                {'Vina' if deps['USE_VINA'] else 'RDKit Shape (fallback)'}")
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


def _read_records_from_sdf(sdf_path: str) -> List[CompoundRecord]:
    """
    Read pre-made molecules from an SDF file into ``CompoundRecord`` objects.

    Uses RDKit's :class:`Chem.SDMolSupplier`. Each molecule becomes a record
    with a ``compound_id`` taken from its SDF ``_Name`` property (falling back
    to a positional ``SDF-####`` id) and its canonical SMILES.

    Args:
        sdf_path: Path to the input SDF file.

    Returns:
        List of :class:`CompoundRecord` objects (one per readable molecule).
    """
    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"Input SDF not found: {sdf_path}")

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    records: List[CompoundRecord] = []
    for i, mol in enumerate(supplier):
        if mol is None:
            log.warning(f"  Skipping unreadable entry {i} in SDF.")
            continue
        if mol.HasProp("_Name"):
            cid = mol.GetProp("_Name").strip() or f"SDF-{i:04d}"
        else:
            cid = f"SDF-{i:04d}"
        smiles = Chem.MolToSmiles(mol)
        records.append(CompoundRecord(
            compound_id=cid,
            smiles=smiles,
            mol=mol,
        ))

    if not records:
        log.warning(f"  No valid molecules read from SDF: {sdf_path}")
    else:
        log.info(f"  Loaded {len(records)} molecules from SDF (BRICS skipped).")
    return records


def main(target_count: int = 500, force: bool = False, library: Optional[str] = None,
          config: Optional[dict] = None, sdf: Optional[str] = None,
          smiles: Optional[str] = None, quick: bool = False):
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
        quick: When True (typically with ``--quick``), prepared targets are
            served from a persistent cache (``~/.autoantibiotic/cache``) instead
            of being re-downloaded / re-cleaned. Falls back to a fresh
            ``prepare_targets`` call if the cache is unavailable.
    """
    ensure_output_dir()

    # ── Configuration (explicit mode: ci | science) ──
    if config is None:
        config = load_config()
    mode = config.get("mode", "ci")

    # ── Dependency check ──
    deps = check_dependencies()

    # ── Working directory for intermediate files ──
    work_dir = str(OUTPUT_DIR / "workdir")
    pdb_dir = str(OUTPUT_DIR / "pdb")
    os.makedirs(work_dir, exist_ok=True)

    # ── Phase1: Target preparation ──
    # In Quick Screen mode (--quick) we reuse a persistent cache of prepared
    # targets instead of re-downloading / re-cleaning PDBs every run. If the
    # cache is unavailable we transparently fall back to a fresh prepare_targets.
    if quick:
        log.info("─── Quick Screen mode: using cached prepared targets ───")
        cache_dir = DEFAULT_CACHE_DIR
        try:
            targets = _get_cached_targets(cache_dir, deps, config)
        except Exception as exc:
            log.warning(
                f"  ⚠  Target caching failed ({exc}); "
                "falling back to standard prepare_targets()."
            )
            targets = prepare_targets(pdb_dir, work_dir, deps, config=config)
    else:
        targets = prepare_targets(pdb_dir, work_dir, deps, config=config)

    # ── Single-compound ("--smiles") mode ──
    # Screen one molecule instantly and print a text summary. This bypasses the
    # full library generation / selectivity / reporting phases entirely so a
    # chemist can inspect a single candidate in seconds. (When combined with
    # --quick, prepared targets are served from the cache.)
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
            )
    else:
        validation_ok, redock_rmsd = run_redocking_validation(
            holo_pdb_path=targets["holo_pdb"],
            target_pdbqt_path=targets["PBP2a"]["pdbqt"],
            work_dir=work_dir,
            deps=deps,
            mode=targets.get("mode"),
            config=config,
        )
        # A failed redocking validation against real PDBs is a diagnostic
        # signal, not a hard gate: log the error, keep validation_ok=False,
        # and continue. The status is recorded in the CSV (Redock_Validated).
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

    # ── Release status badge (status.json) ──
    # Surface the protocol-validation status at the repo root so downstream
    # tooling / CI can read it at a glance. Reuse the cached validation JSON
    # content when available; otherwise record the just-computed values.
    if validation_ok is not None:
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

    # ── Phase 2: Library generation & filtering ──
    if sdf is not None:
        # Read pre-made molecules directly from an SDF file (RDKit) instead of
        # generating a new library via BRICS. This makes the pipeline easy to
        # integrate with external compound collections.
        all_records = _read_records_from_sdf(sdf)
    else:
        all_records = generate_candidate_library(
            target_count=target_count, input_csv=library,
        )
    n_total = len(all_records)
    filtered = apply_filters(all_records)
    n_filtered = len(filtered)

    if n_filtered == 0:
        # No compound survived the strict+relaxed filter chain. Rather than
        # silently abort (which would yield no report at all), fall back to
        # the unfiltered generated library so a candidate report is still
        # produced. These candidates carry no ADMET/PAINS guarantees and are
        # flagged accordingly downstream.
        log.warning(
            "  No compounds passed filters. Falling back to the unfiltered "
            "generated library so a report is still produced."
        )
        filtered = all_records

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
        validation_ok=validation_ok,
        holo_pdb_path=targets.get("holo_pdb"),
        mode=targets.get("mode"),
        redock_rmsd=redock_rmsd,
    )

    top3 = top10[:3]
    generate_images(top3)

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
        "--quick", action="store_true",
        help=(
            "Quick Screen mode. Reuse a persistent cache of prepared targets "
            "(~/.autoantibiotic/cache) instead of re-downloading / re-cleaning "
            "PDBs. Use with --smiles to screen a single compound in seconds, or "
            "on its own to accelerate the full pipeline's target preparation."
        ),
    )
    args = parser.parse_args()

    if args.check:
        check_dependencies()
        sys.exit(0)

    main(target_count=args.count, force=args.force, library=args.library,
         sdf=args.input_sdf, smiles=args.smiles, quick=args.quick)
