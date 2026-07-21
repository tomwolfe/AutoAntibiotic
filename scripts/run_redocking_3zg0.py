#!/usr/bin/env python3
"""Standalone redocking validation against the real 3ZG0 holo structure."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import discovery_pipeline as P
from config.constants import load_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdb = os.path.join(REPO, "output", "pdb_enrich", "3ZG0.pdb")
work = os.path.join(REPO, "output", "workdir_redock")
os.makedirs(work, exist_ok=True)
deps = P.check_dependencies()
clean = os.path.join(work, "PBP2a_3ZG0_clean.pdb")
pdbqt = P.clean_pdb_structure(pdb, clean)  # strips ligand -> rigid receptor
ok, rmsd, core = P.run_redocking_validation(
    holo_pdb_path=pdb,
    target_pdbqt_path=pdbqt,
    target_pdbqt_paths=[pdbqt],
    work_dir=work,
    deps=deps,
    mode="science",
    config={"native_ligand_resname": "AI8"},
)
print("REDOCK_OK", ok, "RMSD", rmsd, "CORE", core)
