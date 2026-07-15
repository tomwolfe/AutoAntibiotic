"""
Pipeline configuration constants
=================================

Centralised, dependency-free configuration defaults for the AutoAntibiotic
discovery pipeline.

Everything in this module is pure data (no I/O, no scientific computation) so
that it can be imported by any other module — including ``discovery_pipeline``,
``utils.filtering`` and ``utils.docking`` — without creating a circular import.
"""

import multiprocessing as mp
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("AutoAntibiotic")

# ═══════════════════════════════════════════════════════════════════════════════
#  RANDOM SEED
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# PDB identifiers
# PBP2a conformer set used for consensus rigid docking (science mode). The
# holo (6TKO) and apo (3QPD) structures are already fetched by the pipeline;
# 1ZOO is an additional public PBP2a PDB trivially addable via this list
# (no new download infrastructure). The first entry is treated as the primary
# receptor for backwards compatibility.
PBP2A_CONFORMER_IDS = ["3QPD", "6TKO", "1ZOO"]
PDB_IDS = {
    "PBP2a_apo": "3QPD",
    "PBP2a_holo": "6TKO",
    "PBP2a_conformer_1ZOO": "1ZOO",
    "trypsin": "1UTN",
    "CES1": "3KJZ",
    # Wider human off-target panel (Task 3). 1AO6 = human serum albumin,
    # 1W0E = human CYP3A4. Both are added to the fetch list so they download
    # alongside the other targets (no mock-free static files required).
    "HUMAN_ALBUMIN": "1AO6",
    "CYP3A4": "1W0E",
    # Additional (mock-capable / optional) off-target panel entries. HERG is
    # included behind config; CYP2D6 is mock-capable. Both are merged into the
    # human selectivity panel when a prepared receptor + grid centre exist, so
    # they gracefully skip when no real PDB is available (CI/offline).
    "HERG": "7CN1",
    "CYP2D6": "MOCK_CYP2D6",
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
#
# These residue lists are target-specific and are loaded from
# ``config/targets.yaml`` at runtime (see :func:`_load_target_residues`).
# The hardcoded values below are kept as defaults so that the pipeline still
# works if ``targets.yaml`` is missing, unreadable, or pyyaml is unavailable.
_ALLOSTERIC_RESIDUES_DEFAULT = ["ALA237", "MET241", "TYR159"]
_ACTIVE_SITE_RESIDUES_DEFAULT = ["SER403"]
_CONSERVED_RESIDUES_DEFAULT = ["SER403", "LYS406", "TYR446"]
_TRYPSIN_CATALYTIC_RESIDUES_DEFAULT = ["HIS57", "ASP102", "SER195"]
_CES1_CATALYTIC_RESIDUES_DEFAULT = ["SER221", "HIS468", "GLU354"]
_ALBUMIN_CATALYTIC_RESIDUES_DEFAULT = ["HIS242", "LYS199"]
_CYP3A4_CATALYTIC_RESIDUES_DEFAULT = ["HIS368", "ARG105"]
_HERG_CATALYTIC_RESIDUES_DEFAULT = ["SER624", "TYR652", "PHE656"]
_CYP2D6_CATALYTIC_RESIDUES_DEFAULT = ["ASP301", "GLU216", "SER304"]

# Names of the target residue lists that can be overridden via targets.yaml.
_TARGET_RESIDUE_KEYS = (
    "ALLOSTERIC_RESIDUES",
    "ACTIVE_SITE_RESIDUES",
    "CONSERVED_RESIDUES",
    "TRYPSIN_CATALYTIC_RESIDUES",
    "CES1_CATALYTIC_RESIDUES",
    "ALBUMIN_CATALYTIC_RESIDUES",
    "CYP3A4_CATALYTIC_RESIDUES",
    "HERG_CATALYTIC_RESIDUES",
    "CYP2D6_CATALYTIC_RESIDUES",
)

TARGETS_FILE = Path(__file__).resolve().parent / "targets.yaml"


# Sane default RMSD cutoffs (Angstrom) for the protocol-trust logic. These are
# overridden by the ``thresholds:`` block in ``config/targets.yaml`` when present.
_RMSD_VALIDATED_MAX_DEFAULT = 1.5
_RMSD_MARGINAL_MAX_DEFAULT = 2.0


def _load_thresholds() -> Dict[str, float]:
    """
    Load protocol-trust RMSD cutoffs from ``config/targets.yaml``.

    Returns ``{"rmsd_validated_max": ..., "rmsd_marginal_max": ...}``, falling
    back to the hardcoded ``*_DEFAULT`` values whenever the YAML file is
    missing, unreadable, pyyaml is unavailable, or the ``thresholds`` block is
    absent. Only finite positive floats are accepted; anything else keeps the
    default so the contract (and the trust badge strings) remain stable.
    """
    defaults = {
        "rmsd_validated_max": _RMSD_VALIDATED_MAX_DEFAULT,
        "rmsd_marginal_max": _RMSD_MARGINAL_MAX_DEFAULT,
    }
    try:
        import yaml

        if TARGETS_FILE.exists():
            with open(TARGETS_FILE) as fh:
                data = yaml.safe_load(fh) or {}
            thr = data.get("thresholds", {}) if isinstance(data, dict) else {}
            for key in ("rmsd_validated_max", "rmsd_marginal_max"):
                val = thr.get(key)
                if isinstance(val, (int, float)) and float(val) > 0:
                    defaults[key] = float(val)
    except Exception:
        pass
    return defaults


_loaded_thresholds = _load_thresholds()
RMSD_VALIDATED_MAX = _loaded_thresholds["rmsd_validated_max"]
RMSD_MARGINAL_MAX = _loaded_thresholds["rmsd_marginal_max"]


def _load_target_residues() -> Dict[str, List[str]]:
    """
    Load target residue lists from ``config/targets.yaml``.

    Returns the five residue lists, falling back to the hardcoded
    ``*_DEFAULT`` values whenever the YAML file is missing, unreadable, or
    pyyaml is not installed. Any subset of the keys may be overridden.
    """
    defaults: Dict[str, List[str]] = {
        "ALLOSTERIC_RESIDUES": _ALLOSTERIC_RESIDUES_DEFAULT,
        "ACTIVE_SITE_RESIDUES": _ACTIVE_SITE_RESIDUES_DEFAULT,
        "CONSERVED_RESIDUES": _CONSERVED_RESIDUES_DEFAULT,
        "TRYPSIN_CATALYTIC_RESIDUES": _TRYPSIN_CATALYTIC_RESIDUES_DEFAULT,
        "CES1_CATALYTIC_RESIDUES": _CES1_CATALYTIC_RESIDUES_DEFAULT,
        "ALBUMIN_CATALYTIC_RESIDUES": _ALBUMIN_CATALYTIC_RESIDUES_DEFAULT,
        "CYP3A4_CATALYTIC_RESIDUES": _CYP3A4_CATALYTIC_RESIDUES_DEFAULT,
        "HERG_CATALYTIC_RESIDUES": _HERG_CATALYTIC_RESIDUES_DEFAULT,
        "CYP2D6_CATALYTIC_RESIDUES": _CYP2D6_CATALYTIC_RESIDUES_DEFAULT,
    }
    try:
        import yaml

        if TARGETS_FILE.exists():
            with open(TARGETS_FILE) as fh:
                data = yaml.safe_load(fh) or {}
            targets = data.get("targets", {}) if isinstance(data, dict) else {}
            for key in _TARGET_RESIDUE_KEYS:
                if key in targets and targets[key]:
                    defaults[key] = list(targets[key])
    except Exception:
        # Any failure (missing file, missing pyyaml, bad YAML) → use defaults.
        pass
    return defaults


_loaded_target_residues = _load_target_residues()

ALLOSTERIC_RESIDUES = _loaded_target_residues["ALLOSTERIC_RESIDUES"]
ACTIVE_SITE_RESIDUES = _loaded_target_residues["ACTIVE_SITE_RESIDUES"]
CONSERVED_RESIDUES = _loaded_target_residues["CONSERVED_RESIDUES"]
TRYPSIN_CATALYTIC_RESIDUES = _loaded_target_residues["TRYPSIN_CATALYTIC_RESIDUES"]
CES1_CATALYTIC_RESIDUES = _loaded_target_residues["CES1_CATALYTIC_RESIDUES"]
ALBUMIN_CATALYTIC_RESIDUES = _loaded_target_residues["ALBUMIN_CATALYTIC_RESIDUES"]
CYP3A4_CATALYTIC_RESIDUES = _loaded_target_residues["CYP3A4_CATALYTIC_RESIDUES"]
HERG_CATALYTIC_RESIDUES = _loaded_target_residues["HERG_CATALYTIC_RESIDUES"]
CYP2D6_CATALYTIC_RESIDUES = _loaded_target_residues["CYP2D6_CATALYTIC_RESIDUES"]

# Grid box defaults (Angstroms)
ALLOSTERIC_BOX_SIZE = (15.0, 15.0, 15.0)
ACTIVE_BOX_SIZE = (20.0, 20.0, 20.0)

# Docking
VINA_TIMEOUT_S = 120
N_JOBS = max(1, mp.cpu_count() - 1)

# Similarity
SIMILARITY_THRESHOLD = 0.3
SIMILARITY_THRESHOLD_RELAXED = 0.5
DIVERSITY_MIN_COUNT = 100

# Selectivity
SELECTIVITY_INDEX_THRESHOLD = 2.0

# Outputs
OUTPUT_DIR = Path("output")
CSV_REPORT = OUTPUT_DIR / "top_candidates.csv"
TOP_N = 10

# Local flexible docking: residues treated as flexible (Vina --flex) during the
# active-site step of science mode. These are the conserved catalytic residues
# whose side-chain repositioning matters most for binder pose discrimination.
FLEX_RESIDUES = ["SER403", "LYS406", "TYR446"]

# Simple PBP2a mutation scan (consensus / reduced-resistance probability). For
# the top-N candidates, dock the active-site pose against mutant receptors built
# by mutating the apo PDBQT (cheap string replace). Toggle via config; defaults
# on so experimental-validation odds are improved. Missing/mock PDBs skip.
MUTATION_SCAN = True

# Mutants scanned and their single-residue substitutions of the conserved
# catalytic network.
MUTATION_SCAN_MUTANTS = ["S403A", "K406A", "Y446A", "N146K", "G262S"]

# Morgan fingerprint parameters used for clustering the pre-top-N pool.
FP_RADIUS = 2
FP_NBITS = 2048

# Repository root (used to locate bundled offline PDB files under tests/data).
REPO_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
#  PROTOCOL TRUST MAP
# ═══════════════════════════════════════════════════════════════════════════════

def protocol_trust(mode: str, redock_rmsd: Optional[float]) -> str:
    """
    Return the trust badge string for the docking protocol given the run *mode*
    and the measured *redock_rmsd*.

    The exact output strings are the canonical contract consumed by
    ``utils.reporting.generate_csv_report``:

        - CI mode (no real RMSD)                 → "CI Mode (Skipped)"
        - redock_rmsd > RMSD_MARGINAL_MAX Å      → "CAUTION: High RMSD (<val> Å)"
        - RMSD_VALIDATED_MAX < redock_rmsd <= RMSD_MARGINAL_MAX → "Validated (Marginal)"
        - redock_rmsd <= RMSD_VALIDATED_MAX Å    → "Validated"
        - science mode but no measured RMSD     → "Validation Unavailable"

    The cutoffs (``RMSD_VALIDATED_MAX``, ``RMSD_MARGINAL_MAX``) are loaded from
    ``config/targets.yaml`` (``thresholds:``) with sane defaults so the badge
    strings and their contract are unaffected by YAML absence.
    """
    if mode == "ci":
        return "CI Mode (Skipped)"
    if redock_rmsd is not None and redock_rmsd > RMSD_MARGINAL_MAX:
        return f"CAUTION: High RMSD ({redock_rmsd:.3f} Å)"
    if redock_rmsd is not None and RMSD_VALIDATED_MAX < redock_rmsd <= RMSD_MARGINAL_MAX:
        return "Validated (Marginal)"
    if redock_rmsd is not None:
        return "Validated"
    return "Validation Unavailable"


# Contract assertion: these exact badge strings are cited verbatim in
# README.md (the "Outputs / protocol_trust" column description) and SCIENCE.md
# ("protocol_trust" rules). They are a published contract consumed by the CSV
# report and must not be reworded without updating both docs. Guard the
# canonical set so any accidental drift in this module fails loudly.
_EXPECTED_PROTOCOL_TRUST_STRINGS = {
    "CI Mode (Skipped)",
    "CAUTION: High RMSD",          # prefix; full string interpolates the Å value
    "Validated (Marginal)",
    "Validated",
    "Validation Unavailable",
}
def _assert_protocol_trust_contract() -> None:
    """Assert that protocol_trust can only emit the documented badge strings."""
    samples = [
        protocol_trust("ci", None),
        protocol_trust("science", 2.5),
        protocol_trust("science", 1.8),
        protocol_trust("science", 1.0),
        protocol_trust("science", None),
    ]
    for s in samples:
        # "CAUTION: High RMSD" is the only interpolated (prefix) badge.
        assert s in _EXPECTED_PROTOCOL_TRUST_STRINGS or s.startswith(
            "CAUTION: High RMSD"
        ), f"protocol_trust returned an undocumented badge: {s!r}"


_assert_protocol_trust_contract()


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load pipeline configuration from *config_path* (YAML) or environment.

    The configuration exposes a ``mode`` key, either ``"ci"`` (CI/mock runs,
    no physical redocking) or ``"science"`` (real scientific validation).

    Resolution order (first match wins):
        1. ``AUTOANTIBIOTIC_MODE`` environment variable — explicit override,
            takes precedence over everything on disk. Accepted values:
            ``"ci"`` or ``"science"``.
        2. ``config.yaml`` on disk — the preferred, version-controlled source
            of truth. The ``mode`` key must be exactly ``"ci"`` or
            ``"science"``; anything else is ignored and the warning path runs.
        3. Default fallback — if no file exists (or it is unreadable / missing
            a valid ``mode``), the pipeline defaults to ``mode: ci`` (a fast,
            offline run that proves the install works) and emits a warning.

    To perform heavy scientific computations, create a ``config.yaml`` with
    ``mode: science`` (or set ``AUTOANTIBIOTIC_MODE=science``).

    Returns:
        dict with at least a ``mode`` key.
    """
    cfg: Dict[str, str] = {"mode": "ci"}

    # ── 1: Environment override (explicit is preferred over implicit) ──
    env_mode = os.environ.get("AUTOANTIBIOTIC_MODE")
    if env_mode in ("ci", "science"):
        cfg["mode"] = env_mode
        return cfg

    # ── 2: config.yaml on disk ──
    config_file = Path(config_path)
    if config_file.exists():
        try:
            import yaml

            with open(config_file) as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict) and data.get("mode") in ("ci", "science"):
                cfg["mode"] = data["mode"]
            else:
                log.warning(
                    f"  ⚠  {config_path} missing a valid 'mode' (ci/science); "
                    "defaulting to mode='ci'."
                )
        except ImportError:
            log.warning(
                "  ⚠  pyyaml is not installed; cannot parse config.yaml. "
                "Defaulting to mode='ci'. Install pyyaml for config support."
            )
        except Exception as exc:
            log.warning(
                f"  ⚠  Failed to read {config_path} ({exc}); "
                "defaulting to mode='ci'."
            )
    else:
        log.warning(
            f"  ⚠  {config_path} not found; defaulting to mode='ci'. "
            "Create a config.yaml (mode: ci|science) to set the run mode explicitly."
        )

    # ── 4: default fallback already set above (mode: ci) ──
    return cfg
