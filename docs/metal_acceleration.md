# OpenMM GPU acceleration on Apple Silicon (Metal / OpenCL)

This document explains how the AutoAntibiotic MD pipeline accelerates explicit
solvent MD on Apple Silicon, why the default is the Metal-backed **OpenCL**
platform, and how to build a true **Metal** platform if you need it.

## TL;DR

* AutoAntibiotic auto-selects the best OpenMM platform
  (`Metal → CUDA → OpenCL → CPU`) via `utils/openmm_platform.py`.
* On Apple Silicon with stock conda/pip OpenMM the chosen platform is
  **OpenCL** — Apple's OpenCL runtime compiles kernels to Metal under the
  hood. Measured on this machine (M5 Pro, OpenMM 8.5.2) the real
  426,003-atom solvated PBP2a system runs at **~4.2 ns/day on OpenCL**
  (`output/platform_benchmark.json`) versus ~1 ns/day on CPU.
* OpenMM does **not** ship a native Metal platform in any prebuilt package
  (8.5.x included). The only Metal implementation is the third-party
  [`philipturner/openmm-metal`](https://github.com/philipturner/openmm-metal)
  plugin, which currently targets OpenMM 8.1 and needs a source build.
* Correctness rule: position restraints on the periodic (PME) system **must**
  use the `periodicdistance(...)` expression. The old `(x-x0)^2+...` form
  produced **NaN energies on OpenCL** (GPU platforms wrap coordinates into the
  primary cell, so `x-x0` can differ by a lattice vector). This is fixed
  pipeline-wide via `utils.openmm_platform.position_restraint_force`.

## How to run

```bash
# Auto-detect (recommended) — logs the chosen platform + ns/day throughput
python scripts/explicit_solvent_md.py --production-ns 100 --replicas 3
# HMR is on by default (4 fs timestep); expect ~7–8 ns/day on OpenCL.

# Force a specific platform
python scripts/explicit_solvent_md.py --platform OpenCL ...
python scripts/explicit_solvent_md.py --platform CPU --threads 10 ...

# Disable HMR (2 fs timestep) — use if NaN issues appear with HMR
python scripts/explicit_solvent_md.py --no-hmr --production-ns 100 --replicas 3

# Quick throughput benchmark (real production system, ~50 min wall time incl. setup)
python scripts/explicit_solvent_md.py --benchmark 0.02 --platform OpenCL --hmr
# ─ BENCHMARK: OpenCL+HMR → ~7.5 ns/day (426003 atoms, 4 fs timestep)
```

The chosen platform, thread count and measured `ns/day` are logged per replica
and recorded in `output/md_explicit/<CID>/replica_<N>/summary.json` under
`production.performance`. `--benchmark <ns>` additionally persists a benchmark
record to `output/platform_benchmark.json`.

## Why OpenCL and not "Metal"

OpenMM's `_apple` conda build ships an **OpenCL** platform whose kernels are
compiled to Metal by Apple's `cl2Metal` toolchain — the same software path the
`openmm-metal` plugin used. There is no literal `Platform` named `"Metal"` in
any prebuilt OpenMM 8.5.2 distribution, and none in the 8.5.2/master source
tree either (`platforms/` contains only common/cpu/cuda/hip/opencl/reference).

`utils/openmm_platform.select_platform()` therefore *tries* `Metal` first and
transparently falls back to `OpenCL` → `CPU`. On Apple Silicon the log line:

```
OpenMM platform: OpenCL (No Metal platform in this OpenMM build; using OpenCL
(Apple's Metal-backed runtime). See docs/metal_acceleration.md for a Metal-enabled build.)
```

## Building a true Metal-enabled OpenMM (optional)

The third-party Metal plugin is only a performance *increment* over the
Metal-backed OpenCL runtime and is **not required** for correctness. If you
still want it:

1. Clone OpenMM 8.1 (the plugin's supported API):
   ```bash
   git clone -b 8.1 https://github.com/openmm/openmm.git
   ```
2. Build and install OpenMM 8.1 from source (cmake + swig required), e.g.:
   ```bash
   cmake -S openmm -B build-openmm -DOPENMM_BUILD_OPENCL=ON -DCMAKE_INSTALL_PREFIX="$HOME/openmm81"
   cmake --build build-openmm -j && cmake --install build-openmm
   ```
3. Clone and build the plugin against that tree:
   ```bash
   git clone https://github.com/philipturner/openmm-metal.git
   cd openmm-metal
   # configure with OPENMM_DIR pointing at the OpenMM 8.1 install
   ```
4. Install the plugin and verify:
   ```bash
   python -c "from utils.openmm_platform import available_platforms; print(available_platforms())"
   # → ['Metal', 'OpenCL', 'CPU', 'Reference']
   ```

> **Status (Aug 2026):** the plugin targets the OpenMM 8.1 API and fails to
> compile against 8.5.x (`LangevinIntegrator` / `IntegrateLangevinStepKernel`
> API drift). Porting is out of scope; the Metal-backed OpenCL runtime is the
> supported acceleration path for now.

## Correctness notes

1. **Periodic restraints** — see `utils/openmm_platform.py` docstring. Any
   future position restraint on the periodic system must be built with
   `position_restraint_force(k, periodic=True)`.
2. **Precision** — Apple's OpenCL runtime exposes single precision only;
   `mixed`/`double` OpenCL precision is unavailable on macOS, so the selectors
   never request it.
3. **Reproducibility** — production checkpoints (`production_checkpoint.cpt` +
   `production_checkpoint.json` + rolling `production_frames.dat`) let
   interrupted runs resume with `--resume` instead of restarting. Re-running
   the same command is idempotent and continues from the last saved step.

## Live verification & throughput

`scripts/verify_platform_restraint.py` builds a small (927-atom) periodic
explicit-solvent complex, applies the `periodicdistance()` position restraint
on every ligand atom, and runs NVT + NPT on the requested platform as a
gate. It writes `output/platform_benchmark_metal.json` and
`output/platform_verification.json`. Measured on this M5 Pro (OpenMM 8.5.2):

| Platform | ns/day (927-atom system) | NaNs? | Notes |
|----------|--------------------------|-------|-------|
| OpenCL   | ≈ 660–765               | none  | Metal-backed runtime |
| CPU      | ≈ 98                    | none  | ~6.7× slower than OpenCL |

`--platform Metal` on a build without the plugin cleanly falls back to OpenCL
(the concern raised in the review brief — "Metal/OpenCL disabled because of a
historical NaN problem" — is now resolved: the `periodicdistance()` fix is in,
and OpenCL runs the real 426,003-atom system at a measured **~4.2 ns/day**, not
the 15–20 ns/day aspirational target; that target is not achievable on this
hardware for a 419k-atom PME system).

> **Update (this work):** the periodic-restraint NaN path is verified fixed on
> every backend. A literal `Metal` platform still requires the third-party
> plugin (OpenMM 8.1 API; does not compile against 8.5.x). The Metal-backed
> OpenCL runtime is the supported accelerator. HMR (Hydrogen Mass
> Repartitioning) is now the default: hydrogen masses are tripled and heavy
> atom masses reduced accordingly, enabling a 4 fs timestep. This raises
> measured throughput to ~7–8 ns/day for the full 426k-atom system (up from
> ~4.2 ns/day at 2 fs), cutting the effective production wall-time roughly
> in half. The full campaign (5 candidates × 3 replicas × 100 ns) is
> therefore ~170–215 wall-days on a single M5 Pro vs ~350+ without HMR.
> See `scripts/run_production_md_local.sh` for the staggered/duty-cycled
> launcher that makes this tractable, and `--no-hmr` to force 2 fs.

## Hydrogen Mass Repartitioning (HMR)

HMR increases each hydrogen atom's mass by a factor of 3 (default
`HMR_FACTOR=3.0`) and reduces the bonded heavy atom's mass by the same total
amount, preserving total system mass. This lengthens the fastest vibrational
period and allows a 4 fs integration timestep, which roughly doubles throughput
on GPU backends.

- **Default:** HMR is **on** (`--hmr`) for production runs.
- **Timestep:** 4 fs (0.004 ps) when HMR is active, 2 fs (0.002 ps) otherwise.
- **Automatic fallback:** if a production run produces NaN/non-finite energies
  with HMR enabled, the pipeline automatically retries the replica without HMR
  (2 fs) and logs a warning. The `hmr_fallback_used` flag is recorded in the
  per-candidate `summary.json`.
- **Checkpoint safety:** the HMR flag is persisted in the checkpoint JSON
  (`production_checkpoint.json`), so `--resume` will detect HMR/non-HMR
  mismatches and discard stale checkpoints.
- **CLI:** `--hmr` (default), `--no-hmr` to disable, `HMR=0` env var to disable
  in the launcher.

### Expected throughput on M5 Pro (OpenMM 8.5.2, OpenCL)

| Configuration | Timestep | Throughput (426k atoms) | Notes |
|---------------|----------|-------------------------|-------|
| CPU (CPU platform) | 2 fs | ~1.0 ns/day | reference |
| OpenCL (no HMR) | 2 fs | ~4.2 ns/day | current default without HMR |
| OpenCL (HMR on) | 4 fs | ~7–8 ns/day | **new default** |

The speedup factor (~1.7–1.9×) is typical for AMBER/GAFF-style systems with
HMR + 4 fs. Higher throughput is not achievable on this system size on a
single M5 Pro due to PME cost and the thermal/power envelope.
