# Science Mode — What's Real and What's Not

This pipeline screens small-molecule candidates against MRSA PBP2a. Two run
modes exist, selected via `config.yaml` (`mode: ci|science`) or the
`AUTOANTIBIOTIC_MODE` environment variable.

## CI / Mock mode (default)

`mode: ci` is for offline smoke tests only. It uses the **minimal mock PDBs**
bundled under `tests/data/`. These are **not** crystallographic models — any
redocking RMSD computed against them is **non-physical** and must never be
interpreted as a protocol-quality metric. Outputs from CI mode are **not for
scientific use**. The pipeline prints a `⚠ CI/MOCK MODE` banner to make this
unmistakable.

## Science mode (real work)

For genuine scientific screening you must:

1. **Use real PDB structures.** Place real downloads (e.g. `6TKO`) in the PDB
   directory or let the pipeline fetch them. Never feed the bundled mock PDBs
   to science mode — they are non-physical and will produce meaningless
   docking/redocking results.
2. **Install AutoDock Vina.** Docking and native-ligand redocking validation
   require Vina. Install it with one command via `bash setup.sh` (creates the
   `autoantibiotic` conda env) or run everything inside the Docker image, which
   bundles Vina and OpenBabel. If Vina is missing, science-mode runs hard-fail
   (override with `AUTOANTIBIOTIC_FORCE=1` only if you accept invalid results).

## Trust signal

The candidate CSV (`output/top_candidates.csv`) carries a single
`protocol_trust` column:

- `CI Mode (Skipped)` — mock run, not scientifically valid.
- `Validated` — redocking RMSD ≤ 1.5 Å.
- `Validated (Marginal)` — 1.5 Å < RMSD ≤ 2.0 Å.
- `CAUTION: High RMSD` — RMSD > 2.0 Å; interpret with care.
- `Validation Unavailable` — science mode but no RMSD was measured.

Treat any result whose `protocol_trust` is not `Validated` as preliminary.

> The canonical logic for these exact trust strings lives in
> `config/constants.py` (`protocol_trust`).
