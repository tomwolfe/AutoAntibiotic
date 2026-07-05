# Changelog — AutoAntibiotic v4.0.0

## Phase 1: Robustness & Reproducibility

### 1.1 Audit Hardcoded Values
- **`config.py`**: Added config fields for all previously hardcoded analysis thresholds, report filenames, batch sizes, and cache DB name.
- **`docking.py`**: `_extract_native_ligand_from_holo` now references `CONFIG.pdb_ids["PBP2a_holo"]` instead of the literal `"6TKO"`.
- **`analysis.py`**: `profile_resistance_risk` uses `CONFIG.resistance_energy_active_threshold`, `resistance_energy_allosteric_threshold`, `resistance_mw_threshold`, `resistance_rot_threshold`, and `resistance_qed_threshold`.
- **`reporting.py`**: All output filenames (CSV, HTML, PNG) now use `CONFIG.csv_report_name`, `CONFIG.html_report_name`, `CONFIG.scatter_plot_name`, `CONFIG.qed_histogram_name`. QED histogram label uses `CONFIG.qed_threshold`.
- **`main.py`**: `top_n_for_images` and `top_n_for_html_report` replaced hardcoded `3` and `50`.
- **`docking.py`**: `top50 = scored[:50]` → `scored[:CONFIG.top_n_for_active]`.

### 1.2 Deterministic Seeding
- **`io_utils.py`**: New `set_global_seed(seed)` function synchronises `numpy.random.seed`, `random.seed`, and (via `rdBase._RandomGeneratorSeeds`) RDKit's internal RNG.
- **`main.py`**: Calls `set_global_seed(CONFIG.random_seed)` at pipeline start.
- **`docking.py`**: `_worker_dock_wrapper` now receives a per-worker seed; dry-run uses `numpy.random.default_rng(seed)` for reproducible mock energies across workers.

### 1.3 Error Specificity
- **`io_utils.py`**: Added `AutoAntibioticError`, `VinaError`, and `OpenBabelError` custom exception classes.
- **`io_utils.py`**: `run_tool` now raises `VinaError` or `OpenBabelError` with actionable advice (e.g., "Ligand too large for Vina box — increase box size or filter by molecular weight.") when known error patterns are detected.
- Implemented `_classify_tool_error` and `_TOOL_ERROR_MESSAGES` mapping for pattern-based error classification.

## Phase 2: Performance & Scaling

### 2.1 SQLite Cache
- **`io_utils.py`**: New `CacheManager` class backed by SQLite (`cache.db`). Provides dict-like interface: `__contains__`, `__getitem__`, `__setitem__`, `__len__`, `items`.
- Automatically migrates existing `cache.json` on first use (renames to `cache.json.migrated` after import).
- `make_key` static method creates consistent MD5-based cache keys matching the old format.
- Legacy `load_cache` / `save_cache` functions retained for backward compatibility.
- **`main.py`**: Uses `CacheManager` instead of `load_cache`/`save_cache` when `--use-cache` is passed.
- **`docking.py`**: `_CacheLike = Optional[Union[CacheManager, Dict[str, float]]]` type alias for backward-compatible cache parameter signatures.

### 2.2 Memory-Efficient Library Gen
- **`library_gen.py`**: `_brics_recombination` now uses an internal generator (`_product_generator`) for BRICS enumeration, reducing peak memory during pool construction.
- **`library_gen.py`**: `generate_candidate_library` returns a generator (via `_generate_records`) when `target_count > CONFIG.library_generator_threshold` (default 1000), preventing full materialisation of large libraries.
- **`library_gen.py`**: `apply_filters` accepts `Union[List[CompoundRecord], Iterator[CompoundRecord]]` for streaming input.
- **`main.py`**: Wraps generator result in `list()` for `len()` usage.

### 2.3 Batched Parallel Docking
- **`docking.py`**: `_parallel_dock` processes compounds in batches of `CONFIG.batch_size_docking` (default 75). After each batch the worker pool is closed and `gc.collect()` is called, preventing memory bloat in long-running screening runs.

## Phase 3: Scientific Enhancement

### 3.1 Consensus Scoring Stub
- **`analysis.py`**: New `compute_consensus_score(vina_energy, shape_score, vina_weight, shape_weight)` function returns a weighted average. Weighted by `CONFIG.consensus_vina_weight` (0.7) and `CONFIG.consensus_shape_weight` (0.3) by default.

### 3.2 Toxicity Alerts
- **`library_gen.py`**: `apply_filters` now checks compounds against an RDKit `FilterCatalog` configured with `NIH` and `PAINS_A` catalogs for mutagenicity/cardiotoxicity alerts. Toxicity-filtered compounds are logged as a separate count.

## Phase 4: UX & Deployment

### 4.1 YAML Config Support
- **`config.py`**: `_merge_yaml_overrides()` function checks for `config.yaml` in the project root. If found and `pyyaml` is installed, its key-value pairs are merged into the global `CONFIG` instance. Unknown keys log a warning but do not halt execution.

### 4.2 Dockerfile
- **`Dockerfile`**: New Docker image based on `ubuntu:22.04` with Miniconda, Python 3.11, RDKit 2023.9.6, OpenBabel 3.1.1, AutoDock Vina 1.2.3, and all Python dependencies pre-installed. Entry point: `python -m autoantibiotic`.

---

## v4.0.0 Major Enhancements

### 1. Enhanced Resistance Risk Profiling
- **`analysis.py`**: New `profile_resistance_mutation_sensitivity` function docks candidates against multiple mutant receptor PDBQTs and computes the standard deviation of binding energies (high std = high resistance risk).
- **`analysis.py`**: `profile_resistance_risk` accepts an optional `mutant_pdbqts` list and stores the result in `record.resistance_stability_score`.
- **`models.py`**: `CompoundRecord` gains `resistance_stability_score: Optional[float]` field.
- **`config.py`**: New flags `use_mutation_sampling` (default `False`) and `mutation_variants`.

### 2. Dynamic Fragment Growth
- **`library_gen.py`**: New `generate_grown_library` function iteratively attaches BRICS building blocks to reactive sites on high-scoring cores, filtering by Lipinski/QED at each growth step to prevent combinatorial explosion. Yields `CompoundRecord` objects.

### 3. Meta-Learner Consensus Scoring
- **`analysis.py`**: New `MetaScorer` class — a stacking regressor (`RandomForestRegressor`) trained on benchmark actives/inactives from `benchmarks/reference_data.py`.
  - Features: Vina Energy, GNINA Score, Shape Score, IFP Score, QED, LogP, MolWt, Rotatable Bonds.
  - Target: Binary activity label (1 = active, 0 = inactive).
  - Model saved/loaded via `joblib` in the `output/` directory.
- **`analysis.py`**: New `predict_meta_score` function replaces `compute_consensus_score` when `use_meta_scoring=True`.
- **`analysis.py`**: `compute_consensus_score` retained as fallback.

### 4. MD Validation Stub
- **`md_validation.py`**: New module with `run_short_md(ligand_mol, receptor_pdb, duration_ns=10)`.
  - Uses OpenMM when available (`_HAVE_OPENMM`); gracefully returns `None` otherwise.
  - Returns a dict with `ligand_rmsd_angstrom` stability metric.
- **`orchestrator.py`**: Integrated as optional final step (`apply_md_validation`).
- **`main.py`**: New `--run-md-validation` CLI flag.

### Configuration Changes
- **`config.py`**: New fields in `PipelineConfig`:
  - `use_mutation_sampling: bool = False`
  - `use_meta_scoring: bool = True`
  - `meta_scorer_model_path: str = "output/meta_scorer.joblib"`
  - `md_validation_duration_ns: int = 10`
  - `mutation_variants: List[str] = ["G246", "N146"]`

### Dependency Changes
- **`requirements.txt`**: Added `joblib>=1.3.0`.
- **`pyproject.toml`**: Added `joblib>=1.3.0`; version bumped to `4.0.0`.
