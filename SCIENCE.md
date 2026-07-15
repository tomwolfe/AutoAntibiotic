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

   > **Offline CI / `--smiles` runs without Vina:** when `USE_VINA=False` the
   > pipeline no longer aborts on `screen_library` / `screen_single_compound`.
   > Instead it produces *approximate* scores with a built-in RDKit
   > shape/pharmacophore scoring fallback (lower accuracy — do **not** treat
   > these as real binding energies; they only rank candidates relative to each
   > other, and are reported with a `"(fallback)"` prefix / warning). Redocking
   > validation still requires Vina and is skipped otherwise.

 3. **Set `native_ligand_resname`.** Provide the exact co-crystallised ligand
    residue name (e.g. `native_ligand_resname: CEF`). Without it, redocking
    validation cannot run and the protocol reports `Validation Unavailable` —
    i.e. the science run produces *no* physical RMSD and should be interpreted
    with caution.

## Trust signal

The candidate CSV (`output/top_candidates.csv`) carries a single
`protocol_trust` column:

- `CI Mode (Skipped)` — mock run, not scientifically valid.
- `Validated` — redocking RMSD ≤ 1.5 Å (`RMSD_VALIDATED_MAX`).
- `Validated (Marginal)` — 1.5 Å < RMSD ≤ 2.0 Å (`RMSD_MARGINAL_MAX`).
- `CAUTION: High RMSD` — RMSD > 2.0 Å; interpret with care.
- `Validation Unavailable` — science mode but no RMSD was measured.

> The 1.5 Å / 2.0 Å cutoffs are configurable in `config/targets.yaml`
> (`thresholds:`), loaded by `config/constants.py` with the defaults above.

Treat any result whose `protocol_trust` is not `Validated` as preliminary.

> The canonical logic for these exact trust strings lives in
> `config/constants.py` (`protocol_trust`).

## Improving experimental-validation odds

To raise the likelihood that the top-ranked candidates are genuine PBP2a
inhibitors — without adding deep learning, FEP, or new external services —
the pipeline uses three low-complexity, RDKit/Vina-only measures. **Consensus
rigid docking** docks every compound against a set of PBP2a conformer PDBQTs
(apo 3QPD, holo 6TKO, 1ZOO) and keeps the best (most negative) energy;
redocking validation reports the lowest RMSD across those conformers, so a
single fortuitous crystal pose cannot inflate confidence. **MM-GBSA-like
rerank** relaxes each top-10 active-site pose with `MMFFOptimizeMolecule`
and stores the MMFF energy as `MMGBSA_Score` (a cheap physics-based tie-breaker,
not a full GBSA solver); the reported CSV is reranked by it when present.
**Wider selectivity panel** averages the human off-target docking energy over
four proteins (Trypsin, CES1, Serum Albumin 1AO6, CYP3A4 1W0E), making the
Selectivity Index (threshold still SI ≥ 2.0) more conservative and harder to
pass on an artefact. All residue lists and PDB IDs are configurable in
`config/targets.yaml` / `config/constants.py`.
