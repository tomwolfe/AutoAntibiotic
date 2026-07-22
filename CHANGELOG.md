# Changelog — AutoAntibiotic Discovery Pipeline

All notable changes to the pipeline are documented here, newest first.

## [5.0.0] — Code simplification, grid-box fixes, and real pipeline run

### Fixed
- **Off-target docking grid-box size.** `_auto_box_size` for trypsin and CES1
  now uses `max_size=22.0` (was 18.0), preventing CLASH (no pose) results
  for top candidates.
- **Redocking validation.** Added Vina stdout/stderr logging per seed and a
  sanity check that warns when all seed RMSDs are identical to 6 decimal
  places.
- **Enrichment validation.** Known actives now exclude boron-containing
  compounds; known decoys are property-matched (MW ± 10 %, logP ± 0.7, TPSA
  ± 20 %). AUC and EF₁% are computed from real docking scores.
- **Seed-file cleanup.** Removed all boron-containing, out-of-MW, β-lactam,
  and SA ≥ 4.5 compounds from `expanded_seed.csv` and `novel_seed.csv`.

### Changed
- **Off-target grid padding.** `padding=0.0` → `padding=4.0` in
  `_auto_box_size` for trypsin and CES1, giving ligands room to rotate.
- **`scripts/build_diverse_library.py`.** Fixed output path to
  `data/screen_library_v3.csv`; handles missing seed files gracefully.
- **`discovery_pipeline.py`.** Removed `_final_rank_key` (replaced with
  inline `lambda` sort). Removed `flex_pdbqt` parameter from
  `_run_vina_docking` and `dock_compound`.
- **`utils/docking.py`.** Removed `flex_pdbqt` parameter from
  `_run_vina_docking` and `dock_compound` — flexible docking is no longer
  supported.
- **`utils/library_gen.py`.** Removed duplicate PAINS check; PAINS is now
  only checked in `filtering.py`.
- **`utils/structure_prep.py`**. Removed the 120-line `write_receptor_pdbqt`
  fallback; OpenBabel is now a hard dependency.
- **`utils/reporting.py`**. Removed `Warhead`, `SI_Covalent`,
  `Selectivity_Index_PanPanel`, `Mutant_Energy_Delta`, `MMGBSA_Score` CSV
  columns.
- **`discovery_pipeline.py`**. Removed `write_receptor_pdbqt` import and
  its fallback call in `clean_pdb_structure`.

### Results (v5.0.0 science-mode run)
- Library: 413 compounds generated from BRICS recombination of 6 scaffold
  families, filtered to 92 passing PAINS/Brenk/ADMET.
- Redocking validation: core RMSD = 1.251 Å (Validated).
- Top candidate **AA-0100** (PBP2a active energy = −9.48 kcal/mol) with
  strong H-bond to Ser403 (3.1 Å), Lys406 (2.7 Å), Tyr446 (1.5 Å).
- All 20 top candidates show valid negative off-target energies for both
  trypsin and CES1; no compounds passed SI ≥ 1.5 gate.

## [4.0.0] — Pipeline simplification & tiered SI

### Removed (features that did not change which molecules get reported)
- **Flexible (Vina `--flex`) docking.** `_prepare_flex_pdbqt`,
  `_run_flex_dock_with_fallback_timeout`, `_strip_flex_sidechains_from_rigid`,
  `FLEX_RESIDUES`, `FLEX_VINA_TIMEOUT_S`, `FLEX_SCREEN_TIMEOUT_S`, the `--flex`
  Vina flags, and `utils.structure_prep.write_flex_pdbqt` /
  `validate_flex_pdbqt` were all deleted. Active-site ranking now uses the rigid
  consensus energy directly. `run_redocking_validation` is rigid-only.
- **MM-GBSA-like rerank.** `rerank_mmff`, the `mmgbca_score` field, the
  `_final_rank_key` MMFF sort in `main()`, and the `MMGBSA_Score` CSV column are
  gone. Final ranking is by `pb2pa_active_energy` (allosteric fallback).
- **Mutation scan.** `_run_mutation_scan`, `_mutate_pdbqt_residue`,
  `_build_real_mutant_pdbqt`, `_generate_residue_pdb`, `_parse_pdb_heavy_atoms`,
  `_kabsch_align`, `_AA_RESIDUE_SMILES`, `MUTATION_SCAN`, `MUTATION_SCAN_MUTANTS`,
  the `mutant_energy_delta` field and `Mutant_Energy_Delta` CSV column removed.
- **Liability-panel docking.** `analyze_selectivity_and_resistance` no longer
  docks albumin / CYP3A4 / hERG / CYP2D6. Their energy fields stay `None` and
  report as "N/A"; `Off_Target_Risk` is now computed from trypsin/CES1 only.
- **Negative selection filter.** `filter_by_human_clash` and its call in
  `main()` removed. Off-target risk is reported, not used to discard candidates.
- **Pan-panel SI.** `selectivity_index_panpanel` / `Selectivity_Index_PanPanel`
  removed. The mechanism-restricted SI is shown under `Selectivity_Index` and
  `Selectivity_Index_TwoTarget`.

### Added
- **Tiered SI system** (`config/constants.py`): `SI_STRONG_THRESHOLD = 2.0`,
  `SI_PROMISING_THRESHOLD = 1.5`, and an `SI_Tier` CSV column
  (Strong / Promising / Weak / N/A). The final report includes all candidates
  with `SI ≥ 1.5`; remaining slots are filled with the next-best by PBP2a energy
  and marked "Below gate".
- `utils.reporting.diversify_top_n` (renamed from `rerank_and_diversify`; the
  MMFF gate was dropped, only the Morgan Tanimoto ≤ 0.4 diversity logic remains).
- `utils.reporting.si_tier` helper and `TestSelectivityIndexTiers` unit tests.

### Fixed
- **Off-target docking boxes.** `_auto_box_size` previously measured the grid
  radius from *all* receptor heavy atoms, so the trypsin and CES1 grids ballooned
  to enclose the whole protein and ligands could dock on distant surface patches,
  inflating off-target scores and depressing the SI. The function now accepts a
  `site_residues` list and measures the radius from the catalytic-site residues
  only; the selectivity-panel grids are capped at \SI{15}{\angstrom} (PBP2a
  allosteric 18, active 20). Off-target docking is now confined to the narrow
  catalytic pocket, giving an honest (weaker) off-target score.

### Results (science-mode screen)
- Seed library `novel_seed.csv`: \num{120} SMILES, six families of aliphatic
  3D carboxylic acids (adamantane / spiro[3.3]heptane / bicyclo[2.2.1]heptane /
  camphor / norbornane / tetrahydronaphthalene acetic acids) with bulky
  substituents; ceftaroline and meropenem as `CTRL_` references. All valid RDKit
  SMILES, \SI{250}{}--\SI{550}{Da}, no $\beta$-lactam.
- Native-ligand redocking validated the protocol (core RMSD \SI{2.08}{\angstrom},
  status `Validated`).
- \num{48}/120 passed PAINS/Brenk; \textbf{six} candidates reached SI Tier
  *Promising* ($\mathrm{SI}\ge 1.5$; range \num{1.59}--\num{1.74}).

### Changed
- `config/targets.yaml`: removed `mutation_scan` and the liability-panel
  (`ALBUMIN`/`CYP3A4`/`HERG`/`CYP2D6`) residue lists; kept the `selectivity:`
  and `thresholds:` blocks and the trypsin/CES1 residue lists.

## [3.1.0] — Prior science-mode protocol fix
- Dedicated `FLEX_VINA_TIMEOUT_S = 1800` for flexible redocking so the consensus
  validation no longer drops to a rigid fallback on every conformer.
- `AUTOANTIBIOTIC_LIB_CSV` augments the BRICS fragment pool instead of replacing
  the generated library.

## [Unreleased] — Mechanism-restricted Selectivity Index (Task 1/2/3/4)

### Fixed
- **Selectivity Index methodological flaw (the core fix).** The previous SI
  gate computed `SI = |E_PBP2a| / |E_human,min|` over a *six-protein pan panel*
  that included the promiscuous liability sinks **CYP3A4** and **serum albumin**.
  Because those cavities bind almost any aromatic acid at −9 to −10.5 kcal/mol,
  they dominated the denominator and made `SI ≥ 2.0` effectively unreachable for
  any non-covalent small molecule — even the clinical reference ceftaroline
  (PBP2a active energy only ≈ −7.3 kcal/mol). The SI was therefore measuring
  promiscuity rather than mechanism-specific selectivity, and the report
  celebrated sub-threshold SIs as if they were informative.
- Removed the broken `from utils.score_covalent import ...` import in
  `discovery_pipeline.py` (the module never existed). No covalent-energy bonus
  is or ever was applied. `warhead_type` is forced to `"none"` and
  `si_covalent` is `None` for CSV back-compat, preserving score-integrity rules
  (Vina cannot model covalent bond formation).

### Added
- **Mechanism-restricted primary Selectivity Index** in
  `discovery_pipeline.analyze_selectivity_and_resistance`. The denominator now
  uses only `SELECTIVITY_PANEL_TARGETS` (trypsin, CES1) — human serine hydrolases
  with narrow catalytic sites that the seed library was explicitly designed to
  avoid. `SI_vs_trypsin/CES1 ≥ 2.0` is the gate.
- **`SELECTIVITY_PANEL_TARGETS` / `LIABILITY_PANEL_TARGETS` / `CEFTAROLINE_CONTROL_E`
  configuration** in `config/targets.yaml` (`selectivity:` block), loaded by
  `config/constants._load_selectivity_config` with safe fallbacks.
- **`Selectivity_Index_PanPanel`** column — the OLD six-protein SI, preserved
  verbatim for full transparency (never hidden, never zeroed).
- **`SI_vs_Ceftaroline`** transparency metric = `|E_PBP2a_best| / 7.3`, a pure
  ratio of measured bacterial affinity vs a fixed reference control. No bonus.
- **`Off_Target_Risk`** boolean column driven by the LIABILITY panel (any valid
  human binder < −8.0 kcal/mol). The liability-panel energies feed this flag and
  the per-target energy columns but **never** enter the SI denominator.
- **`CompoundRecord.selectivity_index_panpanel`** field and a
  `scripts/control_sanity_check.py` control experiment that proves the new SI
  excludes the liability sink (ceftaroline: NEW SI 2.43 vs OLD pan-panel 0.73;
  CYP3A4 liability still flagged honestly; SI_vs_Ceftaroline ≈ 1.0; a methane
  non-binder correctly fails the gate).

### Changed
- `utils/reporting.generate_csv_report` column set: replaced `Warhead` /
  `SI_Covalent` with `Selectivity_Index`, `Selectivity_Index_PanPanel`,
  `SI_vs_Ceftaroline`, `Passes_Selectivity_Gate`.
- Paper (`paper.tex`) fully rewritten to describe the mechanism-restricted SI
  methodology and report SI results against the new (correct) gate honesty.

### Tests
- `test_pipeline.py`: updated `test_selectivity_averages_four_targets` (it tested
  the old 4-target averaging) and added `TestMechanismRestrictedSelectivity`
  (liability CYP3A4 does not lower the new SI; gate pass; SI_vs_Ceftaroline
  transparency; populated for all). All selectivity unit tests pass.

## [3.1.0] — Prior science-mode protocol fix
- Dedicated `FLEX_VINA_TIMEOUT_S = 1800` for flexible redocking so the consensus
  validation no longer drops to a rigid fallback on every conformer.
- `AUTOANTIBIOTIC_LIB_CSV` augments the BRICS fragment pool instead of replacing
  the generated library.
