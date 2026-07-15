#!/usr/bin/env python3
"""Screen a single compound against PBP2a via the Python API.

This shows how to drive AutoAntibiotic programmatically without the CLI:
prepare the targets, then dock one SMILES against the allosteric and active
sites and print the resulting binding energies.

Run with:
    python examples/single_compound_api.py

Uses CI mode (config={"mode": "ci"}) so it runs offline against the bundled
mock PDBs without downloading real structures or requiring AutoDock Vina.
For real screening, set mode to "science" and install Vina via `bash setup.sh`.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discovery_pipeline import prepare_targets, screen_single_compound

# Use a dedicated working directory under output/ so the example does not
# scatter files across the repository root.
WORK_DIR = Path("output/workdir")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Meropenem — a known carbapenem, used here as a demonstration query.
SMILES = "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O"

# Disable Vina so the example runs anywhere (no docking binary required).
deps = {"vina": False, "USE_VINA": False}

# CI mode uses the bundled mock PDBs (no network / no Vina needed).
targets = prepare_targets("output/pdb", str(WORK_DIR), deps, config={"mode": "ci"})

rec = screen_single_compound(SMILES, targets, str(WORK_DIR), deps)

print(f"Compound          : {rec.compound_id}")
print(f"SMILES            : {rec.smiles}")
print(f"Allosteric Energy : {rec.pb2pa_allosteric_energy}")
print(f"Active Energy     : {rec.pb2pa_active_energy}")
