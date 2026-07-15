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
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  RANDOM SEED
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

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
REPO_ROOT = Path(__file__).resolve().parent.parent
