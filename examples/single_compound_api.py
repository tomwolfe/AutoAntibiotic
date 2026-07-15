#!/usr/bin/env python3
"""Screen a single compound against PBP2a via the Python API.

This shows how to drive AutoAntibiotic programmatically without the CLI:
prepare the targets, then dock one SMILES against the allosteric and active
sites and print the resulting binding energies.

Run with:
    AUTOANTIBIOTIC_CI=1 python examples/single_compound_api.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discovery_pipeline import prepare_targets, screen_single_compound

# Meropenem — a known carbapenem, used here as a demonstration query.
SMILES = "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O"

# Disable Vina so the example runs anywhere (no docking binary required).
deps = {"vina": False, "USE_VINA": False}

targets = prepare_targets("output/pdb", "output/workdir", deps)

rec = screen_single_compound(SMILES, targets, ".", deps)

print(f"Compound          : {rec.compound_id}")
print(f"SMILES            : {rec.smiles}")
print(f"Allosteric Energy : {rec.pb2pa_allosteric_energy}")
print(f"Active Energy     : {rec.pb2pa_active_energy}")
