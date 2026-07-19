# Changelog — AutoAntibiotic Discovery Pipeline

All notable changes to the pipeline are documented here, newest first.

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
