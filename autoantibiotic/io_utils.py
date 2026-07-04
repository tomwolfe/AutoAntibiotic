from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem, RDLogger as rdklog

from .config import CONFIG, ToolResult

rdklog.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger("AutoAntibiotic")


# ── Custom exceptions ──────────────────────────────────────────────


class AutoAntibioticError(Exception):
    """Base exception for pipeline-specific errors."""


class VinaError(AutoAntibioticError):
    """Raised when AutoDock Vina fails with an actionable message."""


class OpenBabelError(AutoAntibioticError):
    """Raised when OpenBabel fails with an actionable message."""


_VINA_ERROR_PATTERNS: List[str] = [
    r"(?i)\berror\b",
    r"(?i)\bfatal\b",
    r"(?i)could not open",
    r"(?i)could not read",
    r"(?i)is not a valid",
    r"(?i)segmentation fault",
    r"(?i)exception",
    r"(?i)traceback",
    r"(?i)out of memory",
    r"(?i)cannot allocate",
    r"(?i)ligand too large",
    r"(?i)too many atoms",
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


# ── SQLite CacheManager (Phase 2.1) ────────────────────────────────


class CacheManager:
    """Lightweight SQLite-backed cache for docking results.

    Dict-like interface for backward compatibility:
        ``key in cache``, ``cache[key]``, ``cache[key] = value``.

    Automatically migrates existing ``cache.json`` on first use.
    """

    def __init__(self, db_path: str) -> None:
        ensure_output_dir()
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docking_cache (
                cache_key TEXT PRIMARY KEY,
                energy REAL,
                smiles TEXT,
                tag TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()
        self._migrate_from_json_if_present()

    def _migrate_from_json_if_present(self) -> None:
        json_path = CONFIG.output_dir / "cache.json"
        if not json_path.exists():
            return
        try:
            with open(json_path) as f:
                old_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("  ⚠  cache.json corrupt; skipping migration.")
            return

        count = 0
        for key, value in old_cache.items():
            smiles, tag = CacheManager._parse_cache_key(key)
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO docking_cache (cache_key, energy, smiles, tag) VALUES (?, ?, ?, ?)",
                    (key, value, smiles, tag),
                )
                count += 1
            except sqlite3.Error:
                pass
        self._conn.commit()
        json_path.rename(json_path.with_suffix(".json.migrated"))
        log.info(f"  Migrated {count} entries from cache.json → {self._db_path}")

    @staticmethod
    def _parse_cache_key(key: str) -> tuple:
        parts = key.rsplit("_", 1)
        tag = parts[-1] if len(parts) > 1 else "unknown"
        return parts[0] if len(parts) > 1 else key, tag

    def make_key(self, smiles: str, tag: str) -> str:
        md5 = hashlib.md5(smiles.encode("utf-8")).hexdigest()
        return f"{md5}_{tag}"

    def __contains__(self, key: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM docking_cache WHERE cache_key = ?", (key,)
        )
        return cur.fetchone() is not None

    def __getitem__(self, key: str) -> Optional[float]:
        cur = self._conn.execute(
            "SELECT energy FROM docking_cache WHERE cache_key = ?", (key,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def __setitem__(self, key: str, value: Optional[float]) -> None:
        if value is None:
            return
        smiles, tag = CacheManager._parse_cache_key(key)
        self._conn.execute(
            "INSERT OR REPLACE INTO docking_cache (cache_key, energy, smiles, tag) VALUES (?, ?, ?, ?)",
            (key, value, smiles, tag),
        )
        self._conn.commit()

    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM docking_cache")
        return cur.fetchone()[0]

    def items(self) -> List[tuple]:
        cur = self._conn.execute("SELECT cache_key, energy FROM docking_cache")
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        val = self[key]
        return val if val is not None else default


# ── Legacy JSON cache functions (kept for backward compat) ─────────


def load_cache() -> Dict[str, float]:
    """Legacy JSON cache loader — prefer :class:`CacheManager`."""
    json_path = CONFIG.output_dir / "cache.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("  ⚠  Cache file corrupt; starting fresh.")
    return {}


def save_cache(cache: Dict[str, float]) -> None:
    """Legacy JSON cache writer — prefer :class:`CacheManager`."""
    ensure_output_dir()
    try:
        with open(CONFIG.output_dir / "cache.json", "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except (OSError, IOError):
        log.warning("  ⚠  Failed to save cache file — pipeline will continue without persistence.")


_TOOL_ERROR_MESSAGES: Dict[str, str] = {
    "ligand too large": (
        "Ligand too large for Vina search box — increase box size or "
        "filter by molecular weight."
    ),
    "too many atoms": (
        "Ligand exceeds Vina atom limit — reduce molecular complexity "
        "or use a smaller compound."
    ),
    "could not open": (
        "Vina could not open a required file — check file paths and "
        "permissions."
    ),
    "could not read": (
        "Vina could not read a PDBQT file — ensure PDBQT format is "
        "valid and the file is not empty."
    ),
    "out of memory": (
        "Vina ran out of memory — reduce search-box size or use fewer "
        "CPU cores."
    ),
    "cannot allocate": (
        "System cannot allocate memory for Vina — close other "
        "applications or reduce box size."
    ),
}


def _classify_tool_error(cmd: List[str], stderr: str) -> AutoAntibioticError:
    """Classify a tool error into a domain-specific exception."""
    stderr_lower = stderr.lower()
    tool_name = cmd[0] if cmd else ""

    # Vina-specific hints
    if "vina" in tool_name:
        for keyword, msg in _TOOL_ERROR_MESSAGES.items():
            if keyword in stderr_lower:
                return VinaError(
                    f"Vina failed: {msg}\n  stderr: {stderr.strip()}"
                )
        return VinaError(
            f"Vina failed:\n  Command: {' '.join(cmd)}\n  stderr: {stderr.strip()}"
        )

    # OpenBabel-specific hints
    if "obabel" in tool_name:
        return OpenBabelError(
            f"OpenBabel failed — confirm the input format is valid and "
            f"the molecule can be read.\n  stderr: {stderr.strip()}"
        )

    return AutoAntibioticError(
        f"Tool {' '.join(cmd)} failed:\n  stderr: {stderr.strip()}"
    )


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
        check: If True, a non-zero exit code raises an error.
        ignore_stderr_warnings: When ``True`` and the tool exits with
            return code 0, stderr output is inspected:
              - If it matches ``_VINA_ERROR_PATTERNS``, a :class:`VinaError`
                is still raised.
              - Otherwise it is logged as a warning and execution continues.

    Returns:
        ``ToolResult`` with parsed stdout/stderr.

    Raises:
        VinaError: For Vina-specific failures.
        OpenBabelError: For OpenBabel-specific failures.
        AutoAntibioticError: For other tool failures.
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
            raise _classify_tool_error(cmd, proc.stderr)

        if ignore_stderr_warnings and proc.returncode == 0 and proc.stderr.strip():
            for pattern in _VINA_ERROR_PATTERNS:
                if re.search(pattern, proc.stderr):
                    raise _classify_tool_error(cmd, proc.stderr)
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
