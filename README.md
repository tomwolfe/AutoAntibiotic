# AutoAntibiotic Discovery Pipeline

A virtual screening pipeline for discovering novel MRSA PBP2a inhibitors. The pipeline screens small-molecule libraries against *S. aureus* PBP2a (allosteric + active sites) with selectivity filtering against human serine hydrolases, ADMET profiling, resistance-risk analysis, and synthetic accessibility scoring.

## Features

- **Redocking Validation (Phase 0)** — Validates the docking protocol by re-docking the co-crystallised ligand and computing RMSD.
- **Target Preparation (Phase 1)** — Downloads PDB structures, removes crystallographic artifacts, adds hydrogens, and converts to PDBQT. Grid centres are auto-computed for allosteric (Ala237/Met241/Tyr159) and active (Ser403) pockets.
- **Library Generation (Phase 2)** — Generates a diverse, drug-like library via BRICS fragment recombination from natural-product-inspired scaffolds and synthetic building blocks.
- **Filtering (Phase 2)** — Applies β-lactam exclusion, Tanimoto similarity vs reference antibiotics, Lipinski Rule-of-5, QED ≥ 0.6, PAINS alerts, and **Synthetic Accessibility (SA) Score** (SA ≤ 6.0).
- **Virtual Screening (Phase 3)** — AutoDock Vina docking against allosteric (full library) and active (top 50) sites. Falls back to RDKit Shape Protrude scoring when Vina is unavailable.
- **Selectivity Profiling (Phase 4)** — Docks top candidates against human trypsin (1UTN) and CES1 (3KJZ) off-targets; computes Selectivity Index and resistance-risk profile.
- **Reporting (Phase 5)** — Generates a CSV report, 2D structure images (top 3), and an interactive HTML report with embedded matplotlib figures.

## Prerequisites

- **Python 3.9+**
- **Conda** (recommended) or **pip**
- **Optional external binaries**:
  - [AutoDock Vina](https://vina.scripps.edu/) — molecular docking
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

#### ADFR Suite (prepare_receptor)

Download from [https://ccsb.scripps.edu/adfr/](https://ccsb.scripps.edu/adfr/) and add the `prepare_receptor` binary to your `PATH`.

> The pipeline works without these binaries — a fallback RDKit-based PDBQT converter is used when they are absent.

## Usage

### Standard run

```bash
python discovery_pipeline.py
```

### Dry run (no external binaries required)

```bash
python discovery_pipeline.py --dry-run
```

Generates a small library (10 compounds) and returns mock docking energies for end-to-end testing.

### Using cache

```bash
python discovery_pipeline.py --use-cache
```

Re-uses previously computed docking results stored in `output/cache.json` to avoid re-docking identical compound–target pairs.

### Combining options

```bash
python discovery_pipeline.py --dry-run --use-cache
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

Key parameters are defined in the `PipelineConfig` dataclass (`discovery_pipeline.py`). Notable fields:

| Parameter | Default | Description |
|---|---|---|
| `library_target_count` | 500 | Target number of compounds to generate |
| `similarity_threshold` | 0.4 | Max Tanimoto similarity to reference antibiotics |
| `qed_threshold` | 0.6 | Minimum QED score |
| `sa_score_threshold` | 6.0 | Maximum Synthetic Accessibility score (lower = easier to synthesise) |
| `vina_exhaustiveness` | 8 | Vina exhaustiveness parameter |
| `n_jobs` | CPU count − 1 | Parallel worker count |
| `top_n` | 10 | Number of final candidates to report |

## Troubleshooting

### "vina not found"

```text
⚠ 'vina' not found.
```

Install AutoDock Vina (see Installation step 5). Verify with `vina --version`.

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

### Pipeline fails during PDB download

The pipeline uses exponential-backoff retry (3 attempts). If downloads consistently fail, check your internet connection or manually download the PDB files and place them in `output/pdb/` before running.

### Docking produces N/A energies in dry-run mode

This is expected. Dry-run mode uses mock random energies. Run without `--dry-run` with Vina installed for real docking scores.

### Tests fail with "Cannot import discovery_pipeline"

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
