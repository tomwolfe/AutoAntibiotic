from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import RDLogger as rdklog

from .config import CONFIG
from .models import ToolResult

rdklog.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger("AutoAntibiotic")


# ── Custom exceptions ──────────────────────────────────────────────


class AutoAntibioticError(Exception):
    """Pipeline-specific error with a clear, actionable message."""


class VinaError(AutoAntibioticError):
    """Error raised when AutoDock Vina or GNINA fails with a recognised
    pattern in its output, providing an actionable message."""


class OpenBabelError(AutoAntibioticError):
    """Error raised when OpenBabel fails with a recognised pattern."""


_TOOL_ERROR_MESSAGES: Dict[str, Dict[str, str]] = {
    "vina": {
        "Could not open": "Receptor or ligand file not found — check PDBQT paths.",
        "Error parsing": "Input file format error — try re-generating PDBQT files.",
        "dimension": "Search box too small for ligand — increase box size or filter by molecular weight.",
        "Fatal Error": "Vina encountered a fatal error — check input file validity.",
        "std::bad_alloc": "Out of memory — reduce exhaustiveness or ligand size.",
        "Error in Read": "PDBQT file corrupted or malformed — re-run structure preparation.",
    },
    "gnina": {
        "Could not open": "Receptor or ligand file not found — check PDBQT paths.",
        "Error parsing": "Input file format error — try re-generating PDBQT files.",
        "CUDA": "CUDA error — check GPU availability or disable GNINA (--use-gnina).",
        "Fatal Error": "GNINA encountered a fatal error — check input file validity.",
        "cudaError": "CUDA runtime error — ensure NVIDIA drivers and CUDA toolkit are compatible.",
    },
    "obabel": {
        "could not open": "Input file not found by OpenBabel — check path.",
        "Cannot convert": "OpenBabel cannot convert this format — check file type.",
        "error": "OpenBabel conversion error — check ligand/receptor structure.",
    },
    "prepare_receptor": {
        "Error": "prepare_receptor failed — check PDB structure for missing atoms or non-standard residues.",
    },
}


def _classify_tool_error(binary_name: str, stderr: str) -> Optional[str]:
    """Check tool *stderr* against known error patterns and return an
    actionable message, or ``None`` if no pattern matches."""
    patterns = _TOOL_ERROR_MESSAGES.get(binary_name, {})
    for pattern, message in patterns.items():
        if pattern.lower() in stderr.lower():
            return message
    return None


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
    "gnina": (
        "  → Install GNINA:\n"
        "       Download from https://github.com/gnina/gnina/releases\n"
        "       or build from source: https://github.com/gnina/gnina\n"
        "       Then ensure 'gnina' is on your PATH."
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


def set_global_seed(seed: int) -> None:
    """Synchronise numpy, random, and RDKit random seeds.

    Call once at the start of ``main()`` to ensure deterministic
    results across all pipeline stages.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        from rdkit import rdBase
        rdBase._RandomGeneratorSeeds(seed)
    except Exception:
        pass
    log.debug(f"Global seed set to {seed}")


def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)


# ── Simple JSON cache ──────────────────────────────────────────────


def make_cache_key(smiles: str, tag: str) -> str:
    """Generate a deterministic cache key from a SMILES string and a tag.

    Uses MD5 hash of the SMILES (fast, not security-sensitive) combined
    with a human-readable tag for debugging.
    """
    md5 = hashlib.md5(smiles.encode("utf-8")).hexdigest()
    return f"{md5}_{tag}"


def load_json_cache(cache_path: Path) -> Dict[str, float]:
    """Load a JSON-formatted cache file from disk.

    Returns an empty dict if the file does not exist or is corrupted.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("  ⚠  Cache file corrupt; starting fresh.")
        return {}


def save_json_cache(cache_path: Path, data: Dict[str, float]) -> None:
    """Persist a cache dict to disk as JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except (OSError, IOError):
        log.warning("  ⚠  Failed to save cache file — pipeline will continue without persistence.")


def run_tool(
    cmd: List[str],
    timeout: int = 120,
    check: bool = True,
    ignore_stderr_warnings: bool = False,
) -> ToolResult:
    """Execute an external binary with timeout and exit-code checking."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        result = ToolResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

        if check and proc.returncode != 0:
            binary_name = os.path.basename(cmd[0])
            error_msg = _classify_tool_error(binary_name, proc.stderr)
            if error_msg:
                if binary_name in ("vina", "gnina"):
                    raise VinaError(
                        f"{binary_name} failed (exit {proc.returncode}): {error_msg}\n"
                        f"  stderr: {proc.stderr.strip()}"
                    )
                elif binary_name in ("obabel", "prepare_receptor"):
                    raise OpenBabelError(
                        f"{binary_name} failed (exit {proc.returncode}): {error_msg}\n"
                        f"  stderr: {proc.stderr.strip()}"
                    )
            raise AutoAntibioticError(
                f"Tool {' '.join(cmd)} failed (exit {proc.returncode}):\n"
                f"  stderr: {proc.stderr.strip()}"
            )

        if ignore_stderr_warnings and proc.returncode == 0 and proc.stderr.strip():
            log.warning(f"  Tool stderr (benign): {proc.stderr.strip()}")

        return result
    except subprocess.TimeoutExpired:
        raise AutoAntibioticError(
            f"Tool {' '.join(cmd)} timed out after {timeout}s. "
            "Consider increasing CONFIG.vina_timeout_s."
        )


def parse_vina_energy(vina_stdout: str) -> Optional[float]:
    """Extract the best (lowest) binding energy from Vina stdout."""
    for line in vina_stdout.splitlines():
        stripped = line.strip()
        m = re.match(r"^\s*1\s+(-?\d+\.?\d*)", stripped)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    for line in vina_stdout.splitlines():
        m = re.search(r"Affinity:\s*(-?\d+\.?\d*)\s*\(?kcal/mol\)?", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def parse_gnina_energy(gnina_stdout: str) -> Optional[float]:
    """Extract the best CNNscore from GNINA stdout.

    GNINA output contains lines like::

        CNNscore    :   0.8567
        CNNaffinity :   7.2345

    Returns CNNscore (0-1, higher = better) or None if parsing fails.
    """
    for line in gnina_stdout.splitlines():
        stripped = line.strip()
        m = re.search(r"CNNscore\s*:\s*(\d+\.?\d*)", stripped)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    for line in gnina_stdout.splitlines():
        stripped = line.strip()
        m = re.search(r"CNNaffinity\s*:\s*(-?\d+\.?\d*)", stripped)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def download_with_retry(
    pdb_id: str,
    out_dir: str,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Download a PDB structure with exponential-backoff retry."""
    os.makedirs(out_dir, exist_ok=True)
    target_path = os.path.join(out_dir, f"{pdb_id}.pdb")

    if os.path.exists(target_path):
        return target_path

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            log.info(f"  Downloading {pdb_id} (attempt {attempt}/{max_attempts})…")
            from Bio.PDB import PDBList
            pdbl = PDBList()
            pdbl.retrieve_pdb_file(
                pdb_id, pdir=out_dir, file_format="pdb",
            )
            raw = os.path.join(out_dir, f"pdb{pdb_id.lower()}.ent")
            if os.path.exists(raw):
                os.rename(raw, target_path)
            if os.path.exists(target_path):
                log.info(f"  ✓  Downloaded {pdb_id} → {target_path}")
                return target_path
        except Exception as exc:
            last_exc = exc
            log.warning(f"  ✗  Attempt {attempt} failed: {exc}")
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                log.info(f"  Retrying in {delay:.0f}s…")
                time.sleep(delay)

    raise RuntimeError(
        f"Failed to download {pdb_id} after {max_attempts} attempts. "
        f"Last error: {last_exc}"
    )


def verify_dependencies() -> Dict[str, Any]:
    """Phase 0 — Dependency Verification.

    Checks all required Python libraries and external binaries.

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

    packages: Dict[str, str] = {
        "rdkit": "rdkit",
        "meeko": "meeko",
        "Bio": "Bio",
    }
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            status[import_name] = True
            log.info(f"  ✓  {import_name} found.")
        except ImportError:
            log.error(f"  ✗  {import_name} not found.")
            log.error(f"  → Run: pip install -r requirements.txt")
            log.error(f"  → Or: pip install {pip_name}")
            raise ImportError(
                f"Required package '{import_name}' is not installed. "
                f"Please run: pip install -r requirements.txt"
            )

    for bin_name in ("vina", "gnina", "obabel", "prepare_receptor"):
        try:
            run_tool(
                [bin_name, "--help" if bin_name == "prepare_receptor" else "--version"],
                timeout=10,
            )
            status[bin_name] = True
            log.info(f"  ✓  {bin_name} binary found on PATH.")
        except (RuntimeError, OSError, AutoAntibioticError):
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
