# AutoAntibiotic Discovery Pipeline

A virtual screening pipeline for discovering novel MRSA PBP2a inhibitors. The pipeline screens small-molecule libraries against *S. aureus* PBP2a (allosteric + active sites) with selectivity filtering against human serine hydrolases, ADMET profiling, resistance-risk analysis, and synthetic accessibility scoring.

## Features

- **Redocking Validation (Phase 0)** — Validates the docking protocol by re-docking the co-crystallised ligand and computing RMSD.
- **Target Preparation (Phase 1)** — Downloads PDB structures, removes crystallographic artifacts, adds hydrogens, and converts to PDBQT. Grid centres are auto-computed for allosteric (Ala237/Met241/Tyr159) and active (Ser403) pockets.
- **Library Generation (Phase 2)** — Generates a diverse, drug-like library via BRICS fragment recombination from natural-product-inspired scaffolds and synthetic building blocks. **Smart Stereochemistry Handling** is always enabled: undefined stereocenters are enumerated and strain-filtered (MMFF94, >10 kcal/mol discarded) before entering the library pool.
- **Dynamic Fragment Growth (Phase 2)** — Iteratively grows high-scoring core fragments by attaching BRICS-compatible building blocks, filtering by Lipinski/QED at each step.
- **Filtering (Phase 2)** — Applies β-lactam exclusion, Tanimoto similarity vs reference antibiotics, Lipinski Rule-of-5, QED ≥ 0.6, PAINS alerts, and **Synthetic Accessibility (SA) Score** (SA ≤ 6.0).
- **Virtual Screening (Phase 3)** — AutoDock Vina or GNINA (deep-learning) docking against allosteric (full library) and active (top 50) sites. Supports ensemble docking against multiple receptor structures with consensus scoring. Falls back to RDKit Shape Protrude scoring when Vina/GNINA is unavailable. Optionally rescore with **explicit-solvent MM-GB/SA** (Phase 4.6) for rigorous ΔG prediction using TIP3P water box (see `--use-explicit-solvent`).
- **Meta-Learner Consensus Scoring (Phase 4.5)** — Trains a stacking regressor on benchmark actives/inactives to predict activity from Vina energy, shape score, IFP, QED, and LogP features. If MD validation was run, **dynamic stability features** (ligand RMSD, pocket Rg stability) are automatically included in the feature vector.
- **Selectivity Profiling (Phase 4)** — Docks top candidates against human trypsin (1UTN) and CES1 (3KJZ) off-targets; computes Selectivity Index and resistance-risk profile.
- **Resistance Mutation Profiling (Phase 4)** — Optionally docks candidates against mutant receptor variants and computes binding-energy standard deviation as a resistance-risk metric.
- **Free Energy Perturbation (FEP) Resistance Profiling (Phase 4.8)** — **Top-hit only.** When enabled (`use_fep_resistance=True`), replaces heuristic resistance scoring with a rigorous Equilibrium FEP calculation using OpenMM and openmmtools. FEP is only triggered for the top *N* candidates (default: 5, configurable via `fep_top_n`) after docking and MM-GB/SA rescoring. Candidates with >50 heavy atoms or SMILES length >100 are automatically skipped. Uses 11 λ-windows, alchemical decoupling of the ligand, and MBAR free energy estimation to compute ΔΔG between wild-type and mutant receptor binding. Requires `openmm` and `openmmtools`.
- **Generative Design (Phase 2)** — When enabled (`generative_mode=True`), replaces BRICS recombination with a **Graph-based Genetic Algorithm** that evolves a population of molecules via BRICS crossover and mutation, optimized for QED and SA Score. Returns valid, sanitized RDKit Mol objects.
- **MD Validation (Phase 4.7)** — Optional explicit-solvent MD simulation (OpenMM) of top candidates to assess ligand stability via RMSD and pocket Rg stability. Results are stored in `CompoundRecord` and consumed by the MetaScorer (Phase 4.5).
- **Reporting (Phase 5)** — Generates a CSV report, 2D structure images (top 3), and an interactive HTML report with embedded matplotlib figures.

## Prerequisites

- **Python 3.9+**
- **Conda** (recommended) or **pip**
- **Optional external binaries**:
  - [AutoDock Vina](https://vina.scripps.edu/) — molecular docking
  - [GNINA](https://github.com/gnina/gnina) — deep-learning CNN-based docking (higher accuracy)
  - [OpenBabel](https://openbabel.org/) — file format conversion
  - [ADFR Suite](https://ccsb.scripps.edu/adfr/) — `prepare_receptor` for PDBQT conversion

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd AutoAntibiotic
```

### 2. Create a Conda environment (recommended)

```bash
conda create -n autoantibiotic python=3.11
conda activate autoantibiotic
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install RDKit (Conda-only, recommended over pip)

```bash
conda install -c conda-forge rdkit
```

> **Note:** The `rdkit-pypi` pip package works but the Conda version is more stable and includes additional functionality.

### 5. Install optional external binaries

#### AutoDock Vina

```bash
conda install -c conda-forge vina
```

Or download from [https://vina.scripps.edu/](https://vina.scripps.edu/) and ensure `vina` is on your `PATH`.

#### OpenBabel

```bash
conda install -c conda-forge openbabel
# or
brew install openbabel  # macOS
# or
apt install openbabel   # Debian/Ubuntu
```

#### GNINA (deep-learning docking)

```bash
# Conda (Linux only)
conda install -c conda-forge gnina
```

Or download the precompiled binary from [https://github.com/gnina/gnina/releases](https://github.com/gnina/gnina/releases) and ensure `gnina` is on your `PATH`.

```bash
# Verify installation
gnina --help
```

> GNINA provides CNN-based scoring (`CNNscore` / `CNNaffinity`) that correlates better with experimental binding affinities than AutoDock Vina. Enable with `--use-gnina`.

#### ADFR Suite (prepare_receptor)

Download from [https://ccsb.scripps.edu/adfr/](https://ccsb.scripps.edu/adfr/) and add the `prepare_receptor` binary to your `PATH`.

#### OpenMM Tools (FEP resistance profiling)

```bash
conda install -c conda-forge openmmtools
```

OpenMM Tools provides the alchemical factory and MBAR/MBAR free energy estimators used by the rigorous FEP resistance profiling module. Install alongside `openmm` and `pdbfixer`:

```bash
conda install -c conda-forge openmm openmmtools pdbfixer
```

Note: FEP resistance profiling (`use_fep_resistance=True`) and explicit-solvent MM-GB/SA (`use_explicit_solvent_mmgbsa=True`) now **require** their respective dependencies. The pipeline will raise a clear `ConfigurationError` at startup if a requested feature's dependencies are missing, rather than silently falling back.

> The pipeline works without these binaries — a fallback RDKit-based PDBQT converter is used when they are absent.

## Usage

### Standard run

```bash
python -m autoantibiotic
```

### Dry run (no external binaries required)

```bash
python -m autoantibiotic --dry-run
```

Generates a small library (10 compounds) and returns mock docking energies for end-to-end testing.

### Using cache

```bash
python -m autoantibiotic --use-cache
```

Re-uses previously computed docking results stored in `output/cache.json` to avoid re-docking identical compound–target pairs.

### Using GNINA (deep-learning docking)

```bash
python -m autoantibiotic --use-gnina
```

Uses GNINA's CNN-based scoring instead of AutoDock Vina. Falls back to Vina if GNINA fails.

### Ensemble docking (multiple receptor structures)

```bash
python -m autoantibiotic --ensemble-dir /path/to/receptor/structures
```

Docks against every receptor in the directory and computes a consensus score (mean by default). The directory should contain PDB or PDBQT files.

### MD validation (requires OpenMM)

```bash
python -m autoantibiotic --run-md-validation
```

Runs a 10 ns explicit-solvent MD simulation (OpenMM) on top candidates and reports ligand RMSD and pocket Rg stability. Skips gracefully if OpenMM is not installed.

### Explicit-solvent MM-GB/SA rescoring (requires OpenMM + PDBFixer)

```bash
python -m autoantibiotic --use-explicit-solvent
```

Replaces the default implicit-solvent (OBC2) MM-GB/SA rescoring with a more rigorous explicit-solvent calculation. The complex is solvated with TIP3P water (10 Å padding), energy-minimised, equilibrated, and ΔG_binding is averaged over multiple frames.

> **Note:** The pipeline now validates that `openmm` and `pdbfixer` are installed before starting. If they are missing, a clear `ConfigurationError` is raised — no silent fallback.

### FEP resistance profiling — Top-Hit Only (requires OpenMM + openmmtools)

Configure `use_fep_resistance=True` in `config.py` or set it programmatically:

```python
from autoantibiotic.config import CONFIG
CONFIG.use_fep_resistance = True
```

**FEP is now a top-hit only feature.** It is only triggered for the top *N* candidates (default: 5, configurable via `CONFIG.fep_top_n`) after initial docking and MM-GB/SA rescoring. This avoids computationally prohibitive simulations on less-promising compounds.

Additionally, FEP is automatically skipped for molecules with:
- **>50 heavy atoms** (excessive simulation time)
- **SMILES length >100 characters** (ligand too large for practical alchemical decoupling)

When triggered, this replaces the heuristic mutation-sampling approach with a rigorous Equilibrium FEP calculation:
1. Builds explicit-solvent systems for WT and mutant receptor–ligand complexes.
2. Creates alchemical systems using `openmmtools`' `AbsoluteAlchemicalFactory`.
3. Runs 11 λ-windows (configurable via `fep_lambda_windows`) to decouple the ligand.
4. Computes ΔG using MBAR (Multistate Bennett Acceptance Ratio).
5. Returns ΔΔG = ΔG_mutant − ΔG_WT.

A `ConfigurationError` is raised if `openmm` or `openmmtools` are not installed.

### Mutation sampling / FEP resistance profiling

```bash
python -m autoantibiotic --use-mutation-sampling
```

Docks top candidates against mutant receptor variants and reports binding-energy variance as a resistance-risk indicator.

For **rigorous Free Energy Perturbation (FEP)** resistance profiling, enable `use_fep_resistance` in the configuration:

```bash
python -c "from autoantibiotic.config import CONFIG; CONFIG.use_fep_resistance = True" && python -m autoantibiotic --use-mutation-sampling
```

This uses OpenMM + openmmtools to compute ΔΔG between wild-type and mutant receptor binding via alchemical free energy methods (see "FEP resistance profiling — Top-Hit Only" above).

### Combining options

```bash
python -m autoantibiotic --dry-run --use-cache
python -m autoantibiotic --use-gnina --ensemble-dir /path/to/structures
python -m autoantibiotic --run-md-validation --use-mutation-sampling
python -m autoantibiotic --use-explicit-solvent --run-md-validation
```

## Output

All artifacts are written to the `output/` directory.

| File | Description |
|---|---|
| `top_candidates.csv` | Full results table with docking energies, selectivity indices, ADMET properties, and resistance notes. |
| `top1_<ID>.png`, `top2_<ID>.png`, `top3_<ID>.png` | 2D structure images for the top 3 candidates. |
| `report.html` | Interactive HTML report with embedded scatter plot (energy vs selectivity) and QED histogram. |
| `energy_vs_selectivity.png` | Scatter plot of allosteric binding energy vs selectivity index. |
| `qed_histogram.png` | Histogram of QED scores for the top 50 candidates. |
| `pipeline.log` | Full pipeline execution log. |
| `cache.json` | Docking result cache (when `--use-cache` is used). |

## Configuration

Key parameters are defined in the `PipelineConfig` dataclass (`autoantibiotic/config.py`). Notable fields:

| Parameter | Default | Description |
|---|---|---|
| `library_target_count` | 500 | Target number of compounds to generate |
| `similarity_threshold` | 0.4 | Max Tanimoto similarity to reference antibiotics |
| `qed_threshold` | 0.6 | Minimum QED score |
| `sa_score_threshold` | 6.0 | Maximum Synthetic Accessibility score (lower = easier to synthesise) |
| `vina_exhaustiveness` | 8 | Vina/GNINA exhaustiveness parameter |
| `n_jobs` | CPU count − 1 | Parallel worker count |
| `top_n` | 10 | Number of final candidates to report |
| `use_gnina` | `False` | Enable GNINA deep-learning docking |
| `gnina_binary_path` | `"gnina"` | Path to the GNINA binary |
| `ensemble_mode` | `False` | Enable ensemble docking against multiple receptor structures |
| `ensemble_structures_dir` | `None` | Directory containing receptor PDB/PDBQT files for ensemble docking |
| `consensus_scoring_method` | `"mean"` | Consensus scoring: `"mean"`, `"median"`, or `"min"` |
| `use_mutation_sampling` | `False` | Enable mutation-sensitivity resistance profiling |
| `use_meta_scoring` | `True` | Enable MetaScorer stacking-regressor consensus |
| `meta_scorer_model_path` | `"output/meta_scorer.joblib"` | Path to saved MetaScorer model |
| `md_validation_duration_ns` | `10` | MD simulation length in nanoseconds |
| `use_explicit_solvent_mmgbsa` | `True` | Enable explicit-solvent (TIP3P) MM-GB/SA rescoring |
| `explicit_solvent_frames` | `10` | Number of trajectory frames for explicit MM-GB/SA averaging |
| `max_stereoisomers` | `8` | Max stereoisomers per undefined-stereo molecule (strain-filtered) |
| `use_fep_resistance` | `False` | Enable rigorous FEP resistance profiling (requires openmm + openmmtools) |
| `fep_lambda_windows` | `11` | Number of λ-windows for the alchemical transformation |
| `fep_time_step_ps` | `0.002` | Time step (ps) for FEP MD simulation |
| `fep_n_steps` | `500` | Number of FEP steps per λ-window |
| `fep_kT_kcal_per_mol` | `0.596` | kT value (kcal/mol) at 298.15 K for FEP |
| `generative_mode` | `False` | Enable Graph-based GA generative design (BRICS-based) |
| `generative_n_samples` | `100` | Number of novel scaffolds to generate per core |
| `generative_temperature` | `0.8` | Sampling temperature (neural backend only) |

## Troubleshooting

### "vina not found"

```text
⚠ 'vina' not found.
```

Install AutoDock Vina (see Installation step 5). Verify with `vina --version`.

### "gnina not found"

```text
GNINA execution failed: Tool gnina ... timed out or returned non-zero.
```

Install GNINA (see Installation step 5). Verify with `gnina --help`. The pipeline falls back to AutoDock Vina automatically.

### "prepare_receptor" not found

```text
⚠ 'prepare_receptor' not found.
```

The pipeline falls back to OpenBabel or an RDKit-based PDBQT converter. For best results, install ADFR Suite or ensure `obabel` is on your `PATH`.

### "obabel not found"

```text
⚠ 'obabel' not found.
```

Install OpenBabel. The pipeline uses an RDKit fallback for PDBQT conversion, but obabel is recommended for robust file format handling.

### "sascore not installed"

```text
⚠ sascore not installed. SA Score filter will be skipped.
```

The SA Score filter is an optional enhancement. Install with `pip install sascore`. The pipeline continues without it.

### "OpenMM is not installed" or "openmmtools is not installed"

```text
Configuration Error: Free Energy Perturbation (FEP) requested but OpenMM is not installed.
```

This error is raised when `use_fep_resistance=True` or `use_explicit_solvent_mmgbsa=True` but the required dependencies are missing. The pipeline no longer silently falls back — install the missing packages:

```bash
conda install -c conda-forge openmm openmmtools pdbfixer
```

If you do not need these features, disable them in the configuration:

```python
from autoantibiotic.config import CONFIG
CONFIG.use_fep_resistance = False
CONFIG.use_explicit_solvent_mmgbsa = False
```

### Pipeline fails during PDB download

The pipeline uses exponential-backoff retry (3 attempts). If downloads consistently fail, check your internet connection or manually download the PDB files and place them in `output/pdb/` before running.

### Docking produces N/A energies in dry-run mode

This is expected. Dry-run mode uses mock random energies. Run without `--dry-run` with Vina installed for real docking scores.

### Tests fail with import errors

Ensure you are running tests from the project root directory:

```bash
cd AutoAntibiotic
python -m pytest tests/
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

See [LICENSE](LICENSE).
