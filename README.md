# AutoAntibiotic Discovery Pipeline — MRSA PBP2a Inhibitor Screening

An end-to-end, reproducible computational pipeline that screens and ranks
novel inhibitor candidates against the methicillin-resistant *Staphylococcus
aureus* (MRSA) penicillin-binding protein **PBP2a** (PDB `6TKO`). It combines
structure-based virtual screening, protocol validation by native-ligand
redocking, ADMET/PAINS filtering, and selectivity/resistance analysis to
prioritize compounds for experimental follow-up.

> **Protocol validated on 6TKO** (redocking RMSD reported in
> `output/top_candidates.csv` column `Validation_Status`).

---

## Table of Contents

- [Installation](#installation)
  - [Option A — Docker (Zero-Install, Recommended)](#option-a--docker-zero-install-recommended)
  - [Option B — Quick Install Script (Non-Docker)](#option-b--quick-install-script-non-docker)
  - [Option C — Manual Install](#option-c--manual-install)
- [Quick Start (CI Mode)](#quick-start-ci-mode)
- [Python API](#python-api)
- [Real Run (Science Mode)](#real-run-science-mode)
- [Verifying Your Setup](#verifying-your-setup)
- [Output Interpretation](#output-interpretation)
- [Configuration](#configuration)
- [License](#license)

---

## Prerequisites

The pipeline relies on two external binaries that are **not** pip packages:
**AutoDock Vina** (docking + redocking validation) and **OpenBabel** (PDBQT /
structure conversion). AutoDock Vina is **required** for screening — if it is
missing, the pipeline aborts with a clear message. For the best scientific
results, install both (or run via the Docker image, which bundles them).

| Tool           | Purpose                                                            |
| -------------- | ------------------------------------------------------------------ |
| **AutoDock Vina** | Rigid/semi-flexible molecular docking (Phase 3) and native-ligand redocking validation (Phase 0). |
| **OpenBabel**     | PDBQT / ligand format preparation and structure conversion.       |

Python **3.9+** is required.

> **New here?** You do not need to install anything locally. The two options
> below (Docker, or the `setup.sh` script) handle Vina, OpenBabel, and the
> Python package for you.

---

## Installation

There are three ways to install AutoAntibiotic. **Option A (Docker)** is the
zero-install, reproducible path; **Option B (`setup.sh`)** is the recommended
path for local, non-Docker use; **Option C** is for those who want full manual
control.

### Option A — Docker (Zero-Install, Recommended)

A self-contained image (`continuumio/miniconda3`) with Vina and OpenBabel
pre-installed ships everything you need. Build it once, then screen compounds
without touching your host environment:

```bash
# Build the image
docker build -t autoantibiotic .

# Screen a single compound (mount ./output so results land on your host)
docker run -v "$(pwd)/output:/app/output" autoantibiotic --smiles "CN1C(=O)C(N=C1C(=O)O)S..."

# Or run the full pipeline
docker run -v "$(pwd)/output:/app/output" autoantibiotic --count 10
```

The `output/` directory is created inside the container and is mounted so all
reports, images, and the `visualization.pml` script are available on your host.

### Option B — Quick Install Script (Non-Docker)

For local installs, `setup.sh` is the recommended one-command path. It
installs Miniforge if you don't have `conda`/`mamba`, creates a dedicated
`autoantibiotic` environment, installs Vina + OpenBabel from conda-forge, and
installs the Python package:

```bash
bash setup.sh
```

After it finishes, activate the environment and you are ready to screen:

```bash
conda activate autoantibiotic
autoantibiotic --check          # verify Vina + OpenBabel are present
autoantibiotic --count 10       # quick offline smoke test
```

### Option C — Manual Install

Clone the repository, then install the package and its Python dependencies:

```bash
# Install the pipeline and its core Python dependencies
pip install .

# Or, for development (editable install) with the docking extras
pip install -e ".[docking]"
```

The `[docking]` extra pulls in `meeko` and `openbabel-wheel` (PDBQT
preparation helpers). On platforms where the OpenBabel wheel is unavailable,
rely on the `conda` install of `openbabel` shown above.

After installation the `autoantibiotic` command is available on your `PATH`.

---

## Quick Start (CI Mode)

The fastest way to confirm your installation works — **no large PDB files are
downloaded** — is to run an offline, mock CI run:

```bash
autoantibiotic --count 10
```

This runs the full pipeline against small synthetic/mock structures and proves
that all components (RDKit, Biopython, the CLI, reporting) are wired together
correctly. It completes in seconds and produces a candidate report without any
network access or heavy computation. Use it as a smoke test after installing.

> New to the project? The shipped `config.yaml` uses `mode: ci`, and when no
> `config.yaml` is present the pipeline also **defaults to `mode: ci`**, so a
> bare `autoantibiotic --count 10` behaves the same as the CI command above.
> Explicitly create/switch a `config.yaml` to `mode: science` (see
> [Configuration](#configuration)) — and provide `native_ligand_resname` plus
> real PDBs and Vina — to perform real, computationally intensive runs.

To screen a small curated set (e.g. known ligands plus decoys), pass a CSV
library:

```bash
autoantibiotic --library examples/known_ligands.csv --count 4
```

---

## Python API

You can drive the pipeline programmatically instead of via the CLI. A minimal
example (the full version lives in
[`examples/single_compound_api.py`](examples/single_compound_api.py)):

```python
from discovery_pipeline import prepare_targets, screen_single_compound

SMILES = "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O"
deps = {"vina": False, "USE_VINA": False}
targets = prepare_targets("output/pdb", "output/workdir", deps)
rec = screen_single_compound(SMILES, targets, ".", deps)
print(rec.compound_id, rec.pb2pa_allosteric_energy, rec.pb2pa_active_energy)
```

---

## Real Run (Science Mode)

> **Important:** The shipped `config.yaml` uses `mode: ci` so a bare
> `autoantibiotic --count 10` never requires Vina. To do real work you must
> *explicitly* enable science mode and meet its prerequisites (see below).
> This is different from an earlier version of this README that claimed
> `mode: science` was the default — that was wrong and would silently leave
> science runs with "Validation Unavailable".

For genuine scientific screening against the real PBP2a structure, create a
`config.yaml` at the repository root that selects science mode:

```yaml
mode: science
# REQUIRED for redocking validation against the real holo structure:
native_ligand_resname: CEF
```

Science mode prerequisites (all required, otherwise the run aborts or is
untrustworthy):

1. **Real PDB structures** — place/download real `6TKO`, `3QPD`, `1UTN`, `3KJZ`
   (the bundled `tests/data/*.pdb` are minimal mocks and are rejected by
   science mode).
2. **`native_ligand_resname`** set to the exact co-crystallised ligand residue
   name (e.g. `CEF`). Without it, redocking validation cannot run and the
   protocol reports `Validation Unavailable`.
3. **AutoDock Vina** on `PATH` (docking + redocking). Install via `bash
   setup.sh` or the Docker image. If Vina is missing, science-mode runs abort
   (override with `AUTOANTIBIOTIC_FORCE=1` only if you accept invalid results).

Then run the full pipeline (this downloads/uses real PDB structures, performs
native-ligand redocking validation, and runs Vina docking):

```bash
autoantibiotic --count 500
```

Key options:

| Flag         | Description                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| `--count N`  | Number of candidate compounds to generate (BRICS mode). Default `500`.      |
| `--library PATH` | Path to an external compound library CSV (`smiles`, `compound_id`); skips BRICS generation. |
| `--force`    | Reuse a cached redocking validation or bypass a failed redocking gate in science mode (requires `AUTOANTIBIOTIC_FORCE=1`). |
| `--check`    | **Only** run the dependency check, then exit. See below.                    |

---

## Verifying Your Setup

To instantly verify that Vina, OpenBabel, and all required Python packages are
installed and on `PATH` — without running any science — use the `--check`
flag:

```bash
autoantibiotic --check
```

When everything is present, you get a clean green confirmation that includes the
detected Vina and OpenBabel versions:

```
AutoAntibiotic Discovery Pipeline v3.1.0
  ✅ Ready to screen!  (Vina: 1.2.5 (... ) | OpenBabel: Open Babel 3.1.1)
```

If **AutoDock Vina** (or **OpenBabel**) is missing, the pipeline prints a
bold, high-visibility error that points you straight to `setup.sh` or the
Docker image:

```
  ╔══════════════════════════════════════════════════════════════════╗
  ║  ERROR: AutoDock Vina not found.                                ║
  ║  Install it with one command:                                   ║
  ║    bash setup.sh        (creates the 'autoantibiotic' env)      ║
  ║  or run everything in a container:                              ║
  ║    docker run autoantibiotic --smiles "..."                     ║
  ║  Or manually: conda install -c conda-forge vina                 ║
  ╚══════════════════════════════════════════════════════════════════╝
```

The `--check` command exits `0` once the check completes (even when the
optional binaries are missing, so it is safe to use in CI). To print just the
version, use `autoantibiotic --version`.

---

## Output Interpretation

The pipeline writes a ranked candidate report (CSV) to `output/`. Two columns
are particularly important for triaging candidates:

| Column                | Meaning                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **`Selectivity_Index`** | Ratio of the candidate's predicted mammalian-cell toxicity to its anti-PBP2a potency. **Higher is better** — a large value means the compound is predicted to hit the bacterial target while sparing human cells. |
| **`protocol_trust`**     | A single trust badge for the docking protocol: `CI Mode (Skipped)`, `Validated`, `Validated (Marginal)`, `CAUTION: High RMSD`, or `Validation Unavailable`. In CI mode no physical RMSD is computed and results are not for scientific use. The canonical logic for these exact strings lives in `config/constants.py` (`protocol_trust`). |

Additional artifacts (top-candidate images, a `pipeline.log`, and the
validation JSON) are written under `output/` as well.

---

## Configuration

Configuration is resolved in this order:

1. `config.yaml` on disk (preferred) — set `mode: ci` or `mode: science`.
2. The `AUTOANTIBIOTIC_MODE` environment variable (`ci` or `science`).

If no `config.yaml` exists, the pipeline defaults to `mode: ci` so a new user
sees results immediately. A real, heavy computational run requires an explicit
`config.yaml` with `mode: science`.

The allosteric and active-site docking boxes are auto-sized at runtime from the
resolved residue centroids (see `_auto_box_size` in `discovery_pipeline.py`); the
hardcoded `ALLOSTERIC_BOX_SIZE` / `ACTIVE_BOX_SIZE` in `config/constants.py` are
only a fallback used when a site centre cannot be computed.

### Target-specific residue configuration (`config/targets.yaml`)

The residue lists used to build docking grids and for scientific cross-checks —
`ALLOSTERIC_RESIDUES`, `ACTIVE_SITE_RESIDUES`, `CONSERVED_RESIDUES`,
`TRYPSIN_CATALYTIC_RESIDUES`, and `CES1_CATALYTIC_RESIDUES` — live in
`config/targets.yaml` under a top-level `targets:` key:

```yaml
targets:
  ALLOSTERIC_RESIDUES: ["ALA237", "MET241", "TYR159"]
  ACTIVE_SITE_RESIDUES: ["SER403"]
  CONSERVED_RESIDUES: ["SER403", "LYS406", "TYR446"]
  TRYPSIN_CATALYTIC_RESIDUES: ["HIS57", "ASP102", "SER195"]
  CES1_CATALYTIC_RESIDUES: ["SER221", "HIS468", "GLU354"]
```

`config/constants.py` loads these at runtime; you may override any subset. If
`config/targets.yaml` is missing, unreadable, or `pyyaml` is unavailable, the
pipeline falls back to the original hardcoded defaults, so it keeps working.

### Native ligand override (`native_ligand_resname`)

Native-ligand auto-detection has been removed. For science redocking you MUST
provide the exact co-crystallised ligand residue name; it is required:

```yaml
native_ligand_resname: CEF
```

If left absent, native-ligand extraction is skipped and redocking validation
cannot run in science mode. In CI mode this is not needed (redocking is skipped).

---

## License

Released under the license in [LICENSE](LICENSE).
