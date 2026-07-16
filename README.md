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

1. **Real PDB structures** — place/download real `6TKO`, `3QPD`, `1UTN`, `1YAH`
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
| **`protocol_trust`**     | A single trust badge for the docking protocol: `CI Mode (Skipped)`, `Validated`, `Validated (Marginal)`, `CAUTION: High RMSD`, or `Validation Unavailable`. In CI mode no physical RMSD is computed and results are not for scientific use. The canonical logic for these exact strings lives in `config/constants.py` (`protocol_trust`). The RMSD cutoffs (1.5 Å / 2.0 Å) are configurable in `config/targets.yaml` (`thresholds:`). |

> **Offline RDKit fallback scores:** when Vina is unavailable (`USE_VINA=False`),
> docking returns heuristic RDKit shape/pharmacophore scores rather than Vina
> kcal/mol binding energies. These are **qualitative only** and are labelled with
> a `"(fallback)"` prefix (e.g. `(fallback) -3.21 (not kcal/mol)`) wherever they
> are reported, and a warning is always logged when the fallback is used. Do not
> interpret them as physical binding energies.

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

### Improving experimental-validation odds (consensus + rerank + wider panel)

To raise the chance that top candidates are real, selective, low-resistance
binders that experimentally validate, the pipeline layers several
low-complexity boosts (all RDKit/AutoDock-Vina only, no deep learning, FEP, or
external services):
(1) **Consensus rigid docking** — every compound is docked against a small set
of PBP2a conformer PDBQTs (apo 3QPD, holo 6TKO, plus 1ZOO) and the most
negative energy is kept as `pb2pa_allosteric_energy` / `pb2pa_active_energy`;
redocking validation likewise reports the best (lowest) RMSD across conformers.
(2) **Local flexible docking** — in `science` mode the active-site step is also
run with Vina `--flex` on a flexible-residue PDBQT built from SER403/LYS406/TYR446
(see `FLEX_RESIDUES` in `config/constants.py`), giving a more realistic pose for
interaction analysis and the MM-GBSA-like rerank; it falls back to rigid docking
if flex prep fails. (3) **MM-GBSA-like rerank + diversity gate** — after Vina
active-site docking of the top 10, each retained pose is relaxed with
`MMFFOptimizeMolecule` and a crude MMFF energy stored as `MMGBSA_Score` (no full
GBSA solver). Candidates with a *positive* relaxation energy are dropped, then
the survivors are clustered by Morgan fingerprint (radius 2, 2048 bits) and a
maximally dissimilar set (pairwise Tanimoto ≤ 0.4) fills the final top-10 so
reported hits are distinct rather than near-duplicates. (4) **Wider selectivity
panel** — the human off-target screen now averages docking energy over up to 6
proteins (Trypsin, CES1, Serum Albumin [1AO6], CYP3A4 [1W0E], hERG [7CN1], and
CYP2D6); new residue lists live in `config/targets.yaml`
(`ALBUMIN_CATALYTIC_RESIDUES`, `CYP3A4_CATALYTIC_RESIDUES`,
`HERG_CATALYTIC_RESIDUES`, `CYP2D6_CATALYTIC_RESIDUES`) with sane defaults in
`config/constants.py`, and the SI threshold stays at 2.0. (5) **Quantitative
mutation scan** — toggle `mutation_scan: true` in `config/targets.yaml` (default
on); for the top candidates the active-site pose is re-docked against three PBP2a
mutants (S403A, K406A, Y446A) built by mutating the apo PDBQT, and the mean
energy delta vs wild-type (`Mutant_Energy_Delta`) is recorded and folded into the
resistance notes so low-resistance-emergence candidates are favoured. (6)
**Library diversification** — BRICS recombination is biased by decomposing known
binders (ceftaroline, meropenem; override via `seed_smiles`), and an external
library CSV pointed to by `AUTOANTIBIOTIC_LIB_CSV` (columns `smiles,compound_id`)
is merged in before filtering, reusing the existing CSV loader. Set
`AUTOANTIBIOTIC_LIB_CSV` to a CSV containing known non-β-lactam PBP2a binders
(e.g. ceftaroline analogs, glycopeptides) to bias BRICS generation via
`seed_smiles`. (7) **Protocol-trust gate** — in `science` mode the run now
*aborts* unless redocking RMSD ≤ 1.5 Å (`protocol_trust == "Validated"`); set
`AUTOANTIBIOTIC_FORCE=1` to override. (8) **Tighter filters** — the ADMET gate
requires `QED > 0.7` (was 0.6) and the strict similarity cutoff is `0.3` (was
0.4; relaxed stays 0.5). (9) **More clinical mutants** — the mutation scan now
covers `["S403A","K406A","Y446A","N146K","G262S"]`. (10) **More thorough docking**
— Vina `--exhaustiveness` is `16` (was 8) and the `--flex` residues
`["SER403","LYS406","TYR446"]` (FLEX_RESIDUES) stay active. (11) **Post-report
key-H-bond filter** — any final candidate lacking both a Ser403 and Lys406
catalytic H-bond is dropped (unless fewer than `TOP_N` would remain).


The allosteric and active-site docking boxes are auto-sized at runtime from the
resolved residue centroids (see `_auto_box_size` in `discovery_pipeline.py`); the
hardcoded `ALLOSTERIC_BOX_SIZE` / `ACTIVE_BOX_SIZE` in `config/constants.py` are
only a fallback used when a site centre cannot be computed. The science-mode
native-ligand redocking box is likewise auto-sized from the native ligand's
centroid + atomic spread via `_redocking_box_size` (instead of a fixed 25 Å cube),
falling back to 25 Å only if the ligand cannot be parsed.

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

The same file also holds the protocol-trust RMSD cutoffs under a top-level
`thresholds:` key (defaults `rmsd_validated_max: 1.5` and
`rmsd_marginal_max: 2.0` Å). Override them to retune the `protocol_trust` gate
without touching source code; sane defaults keep the contract stable if the file
is absent.

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
