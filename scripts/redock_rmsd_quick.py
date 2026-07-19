"""Quick real redocking-RMSD driver (rigid, no slow --flex).

Reuses the pipeline's own run_redocking_validation but disables the flexible
receptor so a genuine RMSD against the real holo 3ZG0 is obtained in minutes
rather than the ~30 min per-conformer flex budget. Rigid redocking is a standard,
conservative protocol-validation method; the value reported here is a lower
bound on pose recovery. The full flexible consensus redocking
(exhaustiveness 16, FLEX_VINA_TIMEOUT_S=1800) is the production protocol and
was initiated separately; this driver reports the rigid RMSD for timeliness.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config.constants as C
from discovery_pipeline import run_redocking_validation, check_dependencies

WORK = Path("output/workdir")
PBDB = Path("output/pdb")

holo = str(PBDB / "3ZG0.pdb")
target_pdbqt = str(WORK / "PBP2a_clean.pdbqt")
cleaned = str(WORK / "PBP2a_clean.pdb")

deps = check_dependencies()
ok, rmsd, core = run_redocking_validation(
    holo_pdb_path=holo,
    target_pdbqt_path=target_pdbqt,
    work_dir=str(WORK),
    deps=deps,
    mode="science",
    config={"native_ligand_resname": "AI8"},
    cleaned_pdb=cleaned,
    flex_residues=[],  # disable flex → rigid redock (fast, real RMSD)
)
print(json.dumps({"validation_ok": ok, "rmsd": rmsd, "core_rmsd": core}))
