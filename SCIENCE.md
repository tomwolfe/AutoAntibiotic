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

## Features and caveats (v7.0.0)

### Docking pipeline
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
20/18 Å. **Diversity clustering** keeps a maximally dissimilar final set
(Morgan Tanimoto ≤ 0.4). All residue lists and PDB IDs are configurable in
`config/targets.yaml` / `config/constants.py`.

### Exhaustiveness caveat
The primary screen uses Vina exhaustiveness=8 (default for `dock_compound`).
This is lower than the VS-recommended 32+ and introduces ~±2 kcal/mol noise.
The energy spread among top hits (typically 1--2 kcal/mol) is within this noise
window; treat the top cluster as equipotent rather than strictly ranked.

### MMFF94 strain-interaction rescoring
`rescore_mmff94_strain()` in `utils/docking.py` computes an MMFF94 strain
penalty + distance-dependent dielectric interaction + TPSA solvation correction
in arbitrary units. **This is NOT an MM-GBSA score.** It is an approximate
rescoring filter that flags ligands with excessive conformational strain
(e.g., BRICS_0022 at 738 a.u. vs ~366 a.u. for ALL_QU04).

### Flexible side-chain docking (v7.0.0)
`dock_compound_flexible()` in `utils/docking.py` enables Vina `--flex` docking
for catalytic residues (Tyr446, Ser403). This is not used in the primary screen
(which uses rigid consensus docking for throughput) but is available for
follow-up refinement of top candidates.

### MD stability filter (v7.0.0)
`filter_by_md_stability()` in `utils/filtering.py` flags candidates with
mean ligand RMSD > 3.0 Å or zero H-bond contacts to catalytic residues during
explicit-solvent MD. The threshold is strict: in the current study, all top
candidates would be flagged by the RMSD criterion alone. This filter is
available as an automated triage gate for future runs.

### OpenMM MD results — honest assessment
Gas-phase minimisation (1000-step L-BFGS) converges all top candidates. With
the ligand correctly placed in the *docked* pose, the 100 ps explicit-solvent
runs showed all three tested candidates bound (ligand RMSD 1.86–3.49 Å).
The 1~ns extensions (OpenCL + HMR, ~13.7 ns/day) reveal a more nuanced
picture: SEED\_01150 is **Metastable** (RMSD~4.64 Å, Ser403
H-bond occupancy~0.91), but BRICS\_0022 **dissociates** (RMSD~6.01 Å,
Ser403 occupancy~0.24) and ALL\_QU04 loses its catalytic contact
(Ser403 occupancy~0.04). The 100~ns $\times$ 3 replica campaign is **in
progress** on the M5 Pro. Do not treat the 100~ps–1~ns single-replica runs
as proof of binding — they are preliminary. Genuine binding-mode validation
requires the 100~ns multi-replica campaign followed by trajectory-ensemble
MM-GBSA.

### OpenMM MD infrastructure (GPU acceleration + checkpointing)
The production driver auto-selects the best OpenMM platform
(`Metal → CUDA → OpenCL → CPU`, `utils/openmm_platform.py`); on Apple Silicon
the default is the Metal-backed **OpenCL** runtime, which runs the real
419,607-atom solvated PBP2a system at **~7–9 ns/day vs ~1.06 ns/day on CPU**
(~7–8.5×). Position restraints use `periodicdistance(...)` (the naive
`(x-x0)^2+...` form produced NaN energies on OpenCL for the periodic system).
NPT production writes OpenMM checkpoints plus a rolling positions file, so
interrupted runs continue with `--resume` from the last saved step instead of
restarting; measured `ns/day` is logged per replica and stored under
`production.performance`. See `docs/metal_acceleration.md`.

### DUD-E enrichment — a validated negative result
The standardised DUD-E-style benchmark (76 ChEMBL actives, 704
property-matched decoys, apo 1VQQ, raw Vina affinities, ex=8) reports
**AUC = 0.134, EF₁% = EF₅% = EF₁₀% = 0, BEDROC₂₀ = 0 — FAIL**. Labels were
assigned independently of docking energies, so this is **not circular**; the
potency-scaled / active-informative rescoring that produced inflated
self-consistency in earlier versions was removed and must not be reintroduced.
The failure is reported as a *key finding on PBP2a druggability and
rigid-docking discrimination*, not hidden. `scripts/enrichment_saturation_analysis.py`
sweeps exhaustiveness 8→64 (cached/resumable) to determine whether the failure
is undersampling or fundamental: if AUC plateaus below 0.7, report it as a
definitive method–target limitation. **Never tune parameters to force a PASS —
the negative result is the contribution.**

### Conserved active-site waters
Ordered waters occupy the active site in the source crystals (1VQQ: 7, 3ZG0: 2,
4DKI: 1 within 5 Å of the catalytic triad; one position conserved across
structures). The docking protocol strips all waters before PDBQT preparation,
so water-mediated contacts and displacement penalties are not modelled.
`scripts/conserved_water_analysis.py` reports the conserved positions
(`output/conserved_waters.json`) and exports them to
`output/conserved_waters.pdb` (reference frame) for optional water-included
receptor preparation and "docking with waters" re-benchmarking.

## Removed and restored features

### Removed (v4.0)
The following were removed because they did not change which molecules appear
in the report, and they slowed the run:
- **Flexible (`--flex`) docking** of the active-site step — rigid consensus
  docking was made the authoritative active-site ranking in v4.0.
- **MMFF94 strain-interaction rescoring** (`MMFF94_Strain_Score`, `rerank_mmff`) — final ranking is by
  PBP2a active-site consensus energy. (Restored in v5.2.0 as a complementary filter; renamed from MM-GBSA in v5.4.0.)
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

### Restored (v7.0.0)
- **Flexible (`--flex`) docking** was restored in v6.0.0 via
  `utils/docking.py:dock_compound_flexible()` and
  `_prepare_flexible_pdbqt()`. It is available for targeted refinement of
  top candidates but is not part of the primary screening pipeline.
- **MMFF94 strain-interaction rescoring** is available as
  `rescore_mmff94_strain()` for complementary ranking.
- **MD stability filtering** (`filter_by_md_stability()`) is new in v7.0.0
  as a post-MD triage gate.
