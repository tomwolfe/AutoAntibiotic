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

 1. **Use real PDB structures.** Place real downloads (e.g. `3ZG0`)
    in the PDB directory or let the pipeline fetch them. Never feed the
    bundled mock PDBs to science mode — they are non-physical and will
    produce meaningless docking/redocking results.
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
    residue name (e.g. `native_ligand_resname: AI8` for 3ZG0). Without it, redocking
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

## Improving experimental-validation odds (v4.0 — simplified)

To raise the likelihood that the top-ranked candidates are genuine PBP2a
inhibitors — without adding deep learning, FEP, or new external services —
the pipeline keeps only the measures that change *which* molecules are reported.
**Consensus rigid docking** docks every compound against a set of PBP2a
conformer PDBQTs (apo 1VQQ, holo 3ZG0, 4DKI) and keeps the best (most negative)
energy; redocking validation reports the lowest RMSD across those conformers,
so a single fortuitous crystal pose cannot inflate confidence. **Mechanism-
restricted selectivity** docks only the two mechanism-relevant serine
hydrolases (trypsin 1UTN, CES1 1YAH); the promiscuous liability panel
(albumin, CYP3A4, hERG, CYP2D6) is no longer docked. The primary Selectivity
Index uses only trypsin/CES1 and is tiered (Strong ≥ 2.0, Promising 1.5–2.0,
Weak < 1.5). **Off-target box correction:** `_auto_box_size` now measures the
grid radius from *only* the catalytic-site residues (trypsin His57/Asp102/Ser195;
CES1 Ser221/His468/Glu354), capped at 15 Å, instead of from every receptor atom,
so ligands can no longer dock on distant surface patches and inflate off-target
scores. PBP2a active/allosteric boxes are similarly anchored and capped at
20/18 Å. This is the second half of the selectivity fix. **Diversity clustering**
keeps a maximally dissimilar final set (Morgan Tanimoto ≤ 0.4). All residue lists
and PDB IDs are configurable in `config/targets.yaml` / `config/constants.py`.

## Removed features (v4.0)

The following were removed because they did not change which molecules appear
in the report, and they slowed the run:
- **Flexible (`--flex`) docking** of the active-site step — rigid consensus
  docking is now the authoritative active-site ranking.
- **MM-GBSA-like rerank** (`MMGBSA_Score`, `rerank_mmff`) — final ranking is by
  PBP2a active-site consensus energy.
- **Mutation scan** (`Mutant_Energy_Delta`) — resistance is now reported from
  pose-based interaction analysis only.
- **Liability-panel docking** (albumin/CYP3A4/hERG/CYP2D6) and the **negative
  selection filter** (`filter_by_human_clash`) — off-target risk is reported
  (from trypsin/CES1) but no longer discards candidates.
- **Pan-panel SI** (`Selectivity_Index_PanPanel`) — replaced by the tiered,
  mechanism-restricted `Selectivity_Index` / `Selectivity_Index_TwoTarget`.

## Known defects fixed in this revision

The following pipeline defects (reported in the prior `paper_draft.md`) have
been corrected. These were *engineering* bugs that suppressed real signal —
no scientific-validity logic (e.g. the `protocol_trust` CAUTION badge) was
weakened.

1. **Pose loss across parallel workers (§4.3).** `_dock_worker` returns the
   active-site pose path alongside `(record, energy)`; `_dock_compounds_parallel`
   and `_consensus_dock` propagate `active_docked_pdbqt` back to the parent
   `CompoundRecord`. This lets the `H_Bond_*` flags populate from the real
   docked pose.

2. **Selectivity Index hard-zero (§4.1).** The override that set
   `rec.selectivity_index = 0.0` when any human off-target bound tightly
   (energy < -8.0) has been removed. The raw SI is preserved. A separate
   boolean `Off_Target_Risk` column records the high-risk flag. Before the SI
   denominator is computed, any human off-target energy `> 0.0` (no-pose /
   steric clash) is treated as invalid (excluded), so the SI is computed only
   from real binding energies.

3. **Filter relaxation for known binders (§4.4).** `config.yaml` gains
   `recall_mode: false`. When set `true`, `apply_filters` uses
   `SIMILARITY_THRESHOLD_RELAXED` and a QED floor of `0.4` (not `0.7`) so
   ceftaroline / meropenem survive filtering.

4. **Validation artifact (§1).** `run_redocking_validation` writes
   `validation_results.json` to `work_dir`. The honest `protocol_trust` CAUTION
   badge is unchanged.

> Structure note: the repo screens **3ZG0** (holo, ceftaroline ligand
> **AI8**) and **1VQQ** (apo), *not* 6TKO/CEF as earlier docs claimed.
