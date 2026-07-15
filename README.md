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

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start (CI Mode)](#quick-start-ci-mode)
- [Python API](#python-api)
- [Real Run (Science Mode)](#real-run-science-mode)
- [Verifying Your Setup](#verifying-your-setup)
- [Output Interpretation](#output-interpretation)
- [Configuration](#configuration)
- [License](#license)

---

## Prerequisites

The pipeline relies on two external binaries that are **not** pip packages.
Install them with `conda` (recommended):

```bash
conda install -c conda-forge vina openbabel
```

| Tool           | Purpose                                                            |
| -------------- | ------------------------------------------------------------------ |
| **AutoDock Vina** | Rigid/semi-flexible molecular docking (Phase 3) and native-ligand redocking validation (Phase 0). |
| **OpenBabel**     | PDBQT / ligand format preparation and structure conversion.       |

If either binary is missing, the pipeline still runs but falls back to an
RDKit-based Shape/Pharmacophore scoring path and emits a clear warning. For
best scientific results, install both via the `conda` command above.

Python **3.9+** is required.

---

## Installation

Clone the repository, then install the package and its Python dependencies:

```bash
# Install the pipeline and its core Python dependencies
pip install .

# Or, for development (editable install) with the docking extras
pip install -e ".[docking]"
```

The `[docking]` extra pulls in `meeko` and `openbabel-wheel` (PDBQT
preparation helpers). On platforms where the OpenBabel wheel is unavailable,
rely on the `conda` install of `openbabel` shown in [Prerequisites](#prerequisites).

After installation the `autoantibiotic` command is available on your `PATH`.

---

## Quick Start (CI Mode)

The fastest way to confirm your installation works — **no large PDB files are
downloaded** — is to run an offline, mock CI run:

```bash
AUTOANTIBIOTIC_CI=1 autoantibiotic --count 10
```

This runs the full pipeline against small synthetic/mock structures and proves
that all components (RDKit, Biopython, the CLI, reporting) are wired together
correctly. It completes in seconds and produces a candidate report without any
network access or heavy computation. Use it as a smoke test after installing.

> New to the project? When no `config.yaml` is present, the pipeline
> **defaults to `mode: ci`**, so a bare `autoantibiotic --count 10` behaves
> the same as the CI command above. Explicitly create a `config.yaml` with
> `mode: science` (see [Configuration](#configuration)) to perform real,
> computationally intensive runs.

To screen a small curated set (e.g. known ligands plus decoys), pass a CSV
library:

```bash
autoantibiotic --library examples/known_ligands.csv --count 4
```

---

## Python API

You can drive the pipeline programmatically instead of via the CLI. See
[`examples/single_compound_api.py`](examples/single_compound_api.py) for a
complete example that prepares the targets and screens one compound:

```python
from discovery_pipeline import prepare_targets, screen_single_compound

deps = {"vina": False, "USE_VINA": False}
targets = prepare_targets("output/pdb", "output/workdir", deps)
rec = screen_single_compound(
    "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O", targets, ".", deps
)
print(rec.pb2pa_allosteric_energy, rec.pb2pa_active_energy)
```

---

## Real Run (Science Mode)

For genuine scientific screening against the real PBP2a structure, create a
`config.yaml` at the repository root that selects science mode:

```yaml
mode: science
```

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

If **AutoDock Vina** is missing, the pipeline prints a bold, high-visibility
warning directing you to install it:

```
  ╔══════════════════════════════════════════════════════════════════╗
  ║  WARNING: Vina is missing.                                      ║
  ║  For best results, install via:                                 ║
  ║    conda install -c conda-forge vina                            ║
  ╚══════════════════════════════════════════════════════════════════╝
```

The command exits `0` once the check completes.

---

## Output Interpretation

The pipeline writes a ranked candidate report (CSV) to `output/`. Two columns
are particularly important for triaging candidates:

| Column                | Meaning                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **`Selectivity_Index`** | Ratio of the candidate's predicted mammalian-cell toxicity to its anti-PBP2a potency. **Higher is better** — a large value means the compound is predicted to hit the bacterial target while sparing human cells. |
| **`Redock_Validated`**  | `True` when the docking protocol was validated by redocking the native ligand into PBP2a within an RMSD threshold in science mode. `False` (or absent in CI mode) means docking results should be interpreted with caution. |

Additional artifacts (top-candidate images, a `pipeline.log`, and the
validation JSON) are written under `output/` as well.

---

## Configuration

Configuration is resolved in this order:

1. `config.yaml` on disk (preferred) — set `mode: ci` or `mode: science`.
2. The `AUTOANTIBIOTIC_MODE` environment variable (`ci` or `science`).
3. `AUTOANTIBIOTIC_CI=1` → `ci` (legacy offline escape hatch).

If no `config.yaml` exists, the pipeline defaults to `mode: ci` so a new user
sees results immediately. A real, heavy computational run requires an explicit
`config.yaml` with `mode: science`.

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

During redocking validation the pipeline auto-detects the co-crystallised
ligand in the holo PDB. In complex structures this can select the wrong
molecule. Force the correct residue by name:

```yaml
native_ligand_resname: CEF
```

When set, auto-detection is skipped and the residue with that name (e.g. `CEF`)
is selected directly. Leave it absent for automatic detection.

---

## License

Released under the license in [LICENSE](LICENSE).
