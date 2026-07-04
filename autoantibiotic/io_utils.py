from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdkit import RDLogger as rdklog

from .config import CONFIG, ToolResult

rdklog.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger("AutoAntibiotic")


_VINA_ERROR_PATTERNS: List[str] = [
    r"(?i)\berror\b",
    r"(?i)\bfatal\b",
    r"(?i)could not open",
    r"(?i)could not read",
    r"(?i)is not a valid",
    r"(?i)segmentation fault",
    r"(?i)exception",
    r"(?i)traceback",
]

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


def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)


def load_cache() -> Dict[str, float]:
    """Load docking result cache from CONFIG.output_dir / \"cache.json\"."""
    if (CONFIG.output_dir / "cache.json").exists():
        try:
            with open(CONFIG.output_dir / "cache.json") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("  ⚠  Cache file corrupt; starting fresh.")
    return {}


def save_cache(cache: Dict[str, float]) -> None:
    """Persist the docking result cache to CONFIG.output_dir / \"cache.json\"."""
    ensure_output_dir()
    try:
        with open(CONFIG.output_dir / "cache.json", "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except (OSError, IOError):
        log.warning("  ⚠  Failed to save cache file — pipeline will continue without persistence.")


def run_tool(
    cmd: List[str],
    timeout: int = 120,
    check: bool = True,
    ignore_stderr_warnings: bool = False,
) -> ToolResult:
    """Execute an external binary with timeout and exit-code checking.

    Args:
        cmd: Command and arguments.
        timeout: Maximum wall-clock seconds.
        check: If True, a non-zero exit code raises ``RuntimeError``.
        ignore_stderr_warnings: When ``True`` and the tool exits with
            return code 0, stderr output is inspected:
              - If it matches ``_VINA_ERROR_PATTERNS``, a ``RuntimeError``
                is still raised.
              - Otherwise it is logged as a warning and execution continues.

    Returns:
        ``ToolResult`` with parsed stdout/stderr.

    Raises:
        RuntimeError: If *check* is True and the process exits non-zero,
            or (when *ignore_stderr_warnings* is ``True``) if stderr
            contains genuine error patterns despite a zero exit code.
    """
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
            raise RuntimeError(
                f"Tool {' '.join(cmd)} failed (code {proc.returncode}):\n"
                f"  stderr: {proc.stderr.strip()}"
            )

        if ignore_stderr_warnings and proc.returncode == 0 and proc.stderr.strip():
            for pattern in _VINA_ERROR_PATTERNS:
                if re.search(pattern, proc.stderr):
                    raise RuntimeError(
                        f"Tool {' '.join(cmd)} produced error-like stderr "
                        f"(return code 0):\n  stderr: {proc.stderr.strip()}"
                    )
            log.warning(f"  Tool stderr (benign): {proc.stderr.strip()}")

        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Tool {' '.join(cmd)} timed out after {timeout}s"
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

    for bin_name in ("vina", "obabel", "prepare_receptor"):
        try:
            run_tool(
                [bin_name, "--help" if bin_name == "prepare_receptor" else "--version"],
                timeout=10,
            )
            status[bin_name] = True
            log.info(f"  ✓  {bin_name} binary found on PATH.")
        except (RuntimeError, OSError):
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
