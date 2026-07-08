from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    """Error raised when AutoDock Vina fails with a recognised
    pattern in its output, providing an actionable message."""


class GninaError(AutoAntibioticError):
    """Error raised when GNINA fails with a recognised
    pattern in its output, providing an actionable message."""


class OpenBabelError(AutoAntibioticError):
    """Error raised when OpenBabel fails with a recognised pattern."""


class DockingParseError(AutoAntibioticError):
    """Error raised when docking output cannot be parsed or validation fails."""


class PipelineHealthError(AutoAntibioticError):
    """Error raised when the dropout rate in a pipeline phase exceeds
    :attr:`~autoantibiotic.config.PipelineConfig.max_dropout_rate`."""


class PipelineAudit:
    """Tracks compound dropout reasons and enforces health thresholds
    between pipeline phases.

    Each compound that fails a filter, docking run, or other pipeline
    step is recorded with a human-readable reason string.  After a
    phase completes, :meth:`check_health` compares the dropout rate
    against :attr:`CONFIG.max_dropout_rate` and raises
    :class:`PipelineHealthError` if the threshold is exceeded.

    Thread-safe via :class:`threading.Lock`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.dropouts: Dict[str, List[str]] = {}
        self.total_processed: int = 0
        self.total_dropped: int = 0

    def record_dropout(self, compound_id: str, reason: str) -> None:
        """Record that *compound_id* was dropped for *reason*.

        Multiple reasons for the same compound (e.g. a filter fail
        *and* a docking fail on a different path) are accumulated.
        """
        with self._lock:
            if compound_id not in self.dropouts:
                self.dropouts[compound_id] = []
            self.dropouts[compound_id].append(reason)
            self.total_dropped += 1

    def get_summary(self) -> Dict[str, Any]:
        """Return a snapshot dict of audit statistics.

        Keys include ``total_processed``, ``total_dropped``,
        ``dropout_rate``, ``n_unique_compounds_dropped``, and
        ``top_reasons`` (list of ``(reason, count)`` sorted descending).
        """
        from collections import Counter
        reason_counts: Counter = Counter()
        for reasons in self.dropouts.values():
            for r in reasons:
                reason_counts[r] += 1
        top = reason_counts.most_common()
        total = self.total_processed
        rate = (self.total_dropped / total) if total > 0 else 0.0
        return {
            "total_processed": self.total_processed,
            "total_dropped": self.total_dropped,
            "dropout_rate": round(rate, 4),
            "n_unique_compounds_dropped": len(self.dropouts),
            "top_reasons": [{"reason": r, "count": c} for r, c in top],
        }

    def check_health(self, total_input: int, phase_name: str) -> None:
        """Compare the current dropout tally against *total_input*.

        If ``(total_dropped / total_input) > CONFIG.max_dropout_rate``,
        raises :class:`PipelineHealthError` with an informative message.

        Also updates ``self.total_processed``.
        """
        from .config import CONFIG
        self.total_processed = total_input
        rate = self.total_dropped / total_input if total_input > 0 else 0.0
        threshold = CONFIG.max_dropout_rate
        if rate > threshold:
            summary = self.get_summary()
            raise PipelineHealthError(
                f"Pipeline health check FAILED in phase '{phase_name}': "
                f"dropout rate {rate:.1%} exceeds threshold {threshold:.0%}. "
                f"({self.total_dropped} dropped / {total_input} input). "
                f"Top reasons: {summary['top_reasons'][:3]}"
            )

    def set_total_processed(self, n: int) -> None:
        """Explicitly set the total number of compounds entering the phase."""
        self.total_processed = n

    def reset(self) -> None:
        """Clear all accumulated state (for testing or re-use)."""
        with self._lock:
            self.dropouts.clear()
            self.total_processed = 0
            self.total_dropped = 0


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
        "       # Conda (Linux/macOS):\n"
        "       conda install -c conda-forge vina\n"
        "       # Or download from https://vina.scripps.edu/\n"
        "       # Verify:\n"
        "       vina --version"
    ),
    "gnina": (
        "  → Install GNINA:\n"
        "       # Option A — download pre-compiled binary:\n"
        "       wget https://github.com/gnina/gnina/releases/latest/download/gnina \\\n"
        "         -O /usr/local/bin/gnina && chmod +x /usr/local/bin/gnina\n"
        "       # Option B — Conda (Linux only):\n"
        "       conda install -c conda-forge gnina\n"
        "       # Verify:\n"
        "       gnina --help"
    ),
    "obabel": (
        "  → Install OpenBabel:\n"
        "       # Conda (cross-platform):\n"
        "       conda install -c conda-forge openbabel\n"
        "       # macOS:\n"
        "       brew install openbabel\n"
        "       # Debian/Ubuntu:\n"
        "       sudo apt install openbabel\n"
        "       # Verify:\n"
        "       obabel --version"
    ),
    "prepare_receptor": (
        "  → Install ADFR suite (prepare_receptor):\n"
        "       # Download from https://ccsb.scripps.edu/adfr/\n"
        "       # Example:\n"
        "       wget https://ccsb.scripps.edu/adfr/downloads/adfr-1.0rc1-Linux-64bit.tar.gz \\\n"
        "         -O /tmp/adfr.tar.gz && \\\n"
        "       tar xzf /tmp/adfr.tar.gz -C /opt/adfr --strip-components=1 && \\\n"
        "       ln -s /opt/adfr/bin/prepare_receptor /usr/local/bin/prepare_receptor\n"
        "       # Verify:\n"
        "       prepare_receptor --help"
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
                if binary_name == "vina":
                    raise VinaError(
                        f"{binary_name} failed (exit {proc.returncode}): {error_msg}\n"
                        f"  stderr: {proc.stderr.strip()}"
                    )
                elif binary_name == "gnina":
                    raise GninaError(
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
        tool_name = os.path.basename(cmd[0])
        raise AutoAntibioticError(
            f"Tool {tool_name} timed out after {timeout}s. "
            f"The tool may be too slow for this input, or the timeout "
            f"needs to be increased."
        )


def parse_vina_energy(vina_stdout: str) -> Optional[float]:
    """Extract the best (lowest) binding energy from Vina stdout.

    Delegates to :class:`DockingResultValidator` for parsing.
    """
    return DockingResultValidator().parse_vina(vina_stdout)


def parse_gnina_energy(gnina_stdout: str) -> Optional[float]:
    """Extract the best CNNscore from GNINA stdout.

    Delegates to :class:`DockingResultValidator` for parsing.
    """
    return DockingResultValidator().parse_gnina(gnina_stdout)


# ── DockingResultValidator ──────────────────────────────────────────


class DockingResultValidator:
    """Structured validator for docking tool output.

    Parses and validates raw stdout/stderr from Vina and GNINA,
    handling version-agnostic output formats and known error patterns.
    """

    _VINA_ERROR_KEYWORDS = frozenset({
        "Fatal Error", "Segmentation fault", "std::bad_alloc",
        "Could not open", "Error parsing",
    })
    _GNINA_ERROR_KEYWORDS = frozenset({
        "Fatal Error", "Segmentation fault", "CUDA", "cudaError",
        "Could not open", "Error parsing",
    })
    _VINA_TABLE_HEADER_RE = re.compile(
        r"mode\s*\|?\s*affinity", re.IGNORECASE,
    )
    _VINA_MODE_LINE_RE = re.compile(
        r"^\s*(?P<mode>\d+)\s+(?P<affinity>-?\d+\.?\d*)",
    )
    _VINA_AFFINITY_LINE_RE = re.compile(
        r"Affinity:\s*(?P<affinity>-?\d+\.?\d*)",
    )
    _GNINA_CNNSCORE_RE = re.compile(
        r"CNNscore\s*:\s*(?P<score>\d+\.?\d*)",
    )
    _GNINA_CNNAFFINITY_RE = re.compile(
        r"CNNaffinity\s*:\s*(?P<affinity>-?\d+\.?\d*)",
    )

    def parse_vina(self, stdout: str) -> Optional[float]:
        """Parse best (lowest) Vina binding energy from *stdout*.

        Handles both tabular output (``mode | affinity ...``) and
        single-line ``Affinity: X`` format.  Returns ``None`` if no
        valid energy is found or if known error keywords are present.
        """
        if not stdout or not stdout.strip():
            log.debug("Vina stdout is empty")
            return None

        if self._contains_error(stdout, self._VINA_ERROR_KEYWORDS):
            log.warning("Vina output contains known error keywords")
            return None

        # Tabular output — look for the header row, then mode lines
        mode_values: List[float] = []
        found_header = False
        for line in stdout.splitlines():
            stripped = line.strip()
            if not found_header and self._VINA_TABLE_HEADER_RE.search(stripped):
                found_header = True
                continue
            if found_header:
                m = self._VINA_MODE_LINE_RE.match(stripped)
                if m:
                    try:
                        val = float(m.group("affinity"))
                        mode_values.append(val)
                    except ValueError:
                        continue

        if mode_values:
            if len(mode_values) > 1:
                log.warning(
                    f"Multiple docking modes found ({len(mode_values)}); "
                    f"using best (lowest) energy: {mode_values[0]:.3f}"
                )
            return mode_values[0]

        # No header found — try to parse mode lines directly (headerless output)
        for line in stdout.splitlines():
            stripped = line.strip()
            m = self._VINA_MODE_LINE_RE.match(stripped)
            if m:
                try:
                    val = float(m.group("affinity"))
                    mode_values.append(val)
                except ValueError:
                    continue

        if mode_values:
            return mode_values[0]

        # Fallback: single-line "Affinity: X" output
        for line in stdout.splitlines():
            m = self._VINA_AFFINITY_LINE_RE.search(line)
            if m:
                try:
                    return float(m.group("affinity"))
                except ValueError:
                    continue

        log.debug("No Vina energy value could be parsed from output")
        return None

    def parse_gnina(self, stdout: str) -> Optional[float]:
        """Parse best CNNscore from GNINA *stdout*.

        Prioritises ``CNNscore`` over ``CNNaffinity``.  Returns
        ``None`` if CUDA errors, parsing failures, or known error
        keywords are detected.
        """
        if not stdout or not stdout.strip():
            log.debug("GNINA stdout is empty")
            return None

        if self._contains_error(stdout, self._GNINA_ERROR_KEYWORDS):
            log.warning("GNINA output contains known error keywords")
            return None

        # Prioritise CNNscore (first occurrence = best mode)
        for line in stdout.splitlines():
            m = self._GNINA_CNNSCORE_RE.search(line)
            if m:
                try:
                    score = float(m.group("score"))
                    if not math.isfinite(score):
                        log.warning("GNINA CNNscore is NaN or infinite")
                        continue
                    return score
                except ValueError:
                    continue

        # Fallback to CNNaffinity
        for line in stdout.splitlines():
            m = self._GNINA_CNNAFFINITY_RE.search(line)
            if m:
                try:
                    affinity = float(m.group("affinity"))
                    if not math.isfinite(affinity):
                        log.warning("GNINA CNNaffinity is NaN or infinite")
                        continue
                    return affinity
                except ValueError:
                    continue

        log.debug("No GNINA score could be parsed from output")
        return None

    def validate_binary_health(
        self, tool_name: str, version_output: str,
    ) -> bool:
        """Check the version string matches expected patterns.

        Returns ``True`` for known-good version strings, ``False``
        for unknown or incompatible versions.
        """
        if not version_output or not version_output.strip():
            log.warning(f"No version output for {tool_name}")
            return False

        lower = version_output.lower()

        if tool_name == "vina":
            # Vina 1.2.x
            if re.search(r"vina\s+1\.2\.\d", lower):
                return True
            # AutoDock Vina 1.x
            if re.search(r"autodock\s+vina\s+1\.\d", lower):
                return True
            log.warning(f"Unknown Vina version string: {version_output.strip()}")
            return False

        if tool_name == "gnina":
            # GNINA 1.x or higher
            if re.search(r"gnina\s+[1-9]\.\d", lower):
                return True
            # gnina (with version number >= 1.0, not 0.x)
            if re.search(r"gnina.*\b[1-9]\.\d+", lower):
                return True
            log.warning(f"Unknown GNINA version string: {version_output.strip()}")
            return False

        log.warning(f"Unknown tool name: {tool_name}")
        return False

    @staticmethod
    def _contains_error(text: str, keywords: frozenset) -> bool:
        """Check if *text* contains any of the given error *keywords*."""
        lower = text.lower()
        for kw in keywords:
            if kw.lower() in lower:
                return True
        return False


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


# ── BinaryManager ────────────────────────────────────────────────


class BinaryManager:
    """Manages discovery, version-checking, and validation of external
    binaries required by the AutoAntibiotic pipeline."""

    _BINARIES: Dict[str, str] = {
        "vina": "vina",
        "gnina": "gnina",
        "obabel": "obabel",
        "prepare_receptor": "prepare_receptor",
    }
    """Mapping from logical name to expected binary name on PATH."""

    def __init__(self) -> None:
        self._cache: Dict[str, bool] = {}

    def check_binary(self, name: str, path: Optional[str] = None) -> bool:
        """Check whether a binary exists and is executable.

        Args:
            name: Logical binary name (e.g. ``"vina"``, ``"obabel"``).
            path: Optional explicit path to the binary.  If not provided,
                  the binary is looked up on ``PATH``.

        Returns:
            ``True`` if the binary is found and executable.
        """
        import shutil

        binary_path = path or name
        resolved = shutil.which(binary_path)
        if resolved is None:
            return False
        if not os.access(resolved, os.X_OK):
            return False
        return True

    def get_version(self, name: str) -> str:
        """Return the version string for a binary by running ``--version``.

        Args:
            name: Logical binary name.

        Returns:
            Version string (stripped), or ``"unknown"`` if the binary
            cannot be executed or returns no output.
        """
        if name not in self._BINARIES:
            return "unknown"
        binary = self._BINARIES[name]
        try:
            result = run_tool(
                [binary, "--help" if name == "prepare_receptor" else "--version"],
                timeout=10,
                check=False,
            )
            version = (result.stdout or result.stderr or "").strip()
            return version if version else "unknown"
        except (RuntimeError, OSError, AutoAntibioticError):
            return "unknown"

    def validate_all(self) -> Dict[str, bool]:
        """Check all registered binaries and return their availability.

        Returns:
            Dict mapping logical binary name to ``True`` if the binary
            is found and executable.
        """
        results: Dict[str, bool] = {}
        for name in self._BINARIES:
            available = self.check_binary(name)
            results[name] = available
            self._cache[name] = available
        return results


# ── Safe subprocess runner with retry ─────────────────────────────


def safe_run_tool(
    cmd: List[str],
    timeout: int = 120,
    check: bool = True,
    ignore_stderr_warnings: bool = False,
) -> ToolResult:
    """Execute an external binary with a single retry on failure.

    Wraps :func:`run_tool` with one automatic retry.  If the first
    attempt fails (raises :class:`AutoAntibioticError` or returns a
    non-zero exit code when *check* is True), a detailed warning
    including the full stderr is logged and the command is retried
    once.  If the retry also fails, the original exception is
    re-raised.

    All parameters match :func:`run_tool`.

    Returns:
        :class:`ToolResult` from the successful run.
    """
    binary_name = os.path.basename(cmd[0])
    try:
        return run_tool(cmd, timeout=timeout, check=check,
                        ignore_stderr_warnings=ignore_stderr_warnings)
    except (AutoAntibioticError, RuntimeError, OSError) as exc:
        stderr_detail = ""
        if hasattr(exc, "args") and exc.args:
            stderr_detail = str(exc.args[0])
        log.warning(
            "  Tool '%s' failed on first attempt — retrying once.\n"
            "  Command: %s\n"
            "  Error: %s",
            binary_name, " ".join(cmd), stderr_detail,
        )
        try:
            return run_tool(cmd, timeout=timeout, check=check,
                            ignore_stderr_warnings=ignore_stderr_warnings)
        except (AutoAntibioticError, RuntimeError, OSError) as retry_exc:
            log.error(
                "  Tool '%s' failed again on retry.\n"
                "  Command: %s\n"
                "  Error: %s",
                binary_name, " ".join(cmd),
                str(retry_exc.args[0]) if retry_exc.args else str(retry_exc),
            )
            raise


# ── Pipeline input validation ────────────────────────────────────


def validate_pipeline_inputs(config: PipelineConfig) -> Dict[str, List[str]]:
    """Validate all pipeline inputs and return a report of issues.

    Checks:
    - *output_dir* is writable.
    - All SMILES in *reference_antibiotics* and *brics_building_blocks*
      parse correctly with RDKit.
    - *ensemble_structures_dir* exists if set.
    - All registered binaries via :meth:`BinaryManager.validate_all`.

    Args:
        config: The pipeline configuration to validate.

    Returns:
        Dict with keys ``"errors"`` and ``"warnings"``, each containing
        a list of human-readable issue descriptions.  An empty list
        means no issues were found.
    """
    from rdkit import Chem as _Chem

    issues: Dict[str, List[str]] = {"errors": [], "warnings": []}

    # ── Output directory writability ──
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        test_file = config.output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (OSError, IOError) as exc:
        issues["errors"].append(
            f"Output directory '{config.output_dir}' is not writable: {exc}"
        )

    # ── Validate SMILES in reference_antibiotics ──
    for name, smi in config.reference_antibiotics.items():
        if not smi or not smi.strip():
            issues["warnings"].append(
                f"Reference antibiotic '{name}' has an empty SMILES string."
            )
            continue
        mol = _Chem.MolFromSmiles(smi)
        if mol is None:
            issues["errors"].append(
                f"Reference antibiotic '{name}' has an invalid SMILES: '{smi}'."
            )

    # ── Validate SMILES in brics_building_blocks ──
    for smi in config.brics_building_blocks:
        if not smi or not smi.strip():
            issues["warnings"].append("Empty SMILES in brics_building_blocks.")
            continue
        mol = _Chem.MolFromSmiles(smi)
        if mol is None:
            issues["errors"].append(
                f"Invalid BRICS building block SMILES: '{smi}'."
            )

    # ── Check ensemble_structures_dir ──
    if config.ensemble_structures_dir is not None:
        if not config.ensemble_structures_dir.exists():
            issues["warnings"].append(
                f"Ensemble structures directory "
                f"'{config.ensemble_structures_dir}' does not exist."
            )
        elif not config.ensemble_structures_dir.is_dir():
            issues["errors"].append(
                f"Ensemble structures path "
                f"'{config.ensemble_structures_dir}' is not a directory."
            )

    # ── Check binaries ──
    bm = BinaryManager()
    binary_status = bm.validate_all()
    for name, available in binary_status.items():
        if not available:
            guide = _INSTALL_GUIDE.get(name, "")
            issues["errors"].append(
                f"Required binary '{name}' not found or not executable.\n{guide}"
            )

    return issues


def verify_dependencies() -> Dict[str, Any]:
    """Phase 0 — Dependency Verification.

    Checks all required Python libraries and external binaries using
    :class:`BinaryManager`.

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

    bm = BinaryManager()
    binary_status = bm.validate_all()
    for bin_name in ("vina", "gnina", "obabel", "prepare_receptor"):
        available = binary_status.get(bin_name, False)
        status[bin_name] = available
        if available:
            log.info(f"  ✓  {bin_name} binary found on PATH.")
        else:
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
