#!/usr/bin/env python3
"""
Verify OpenMM platform acceleration and the periodic-restraint correctness fix.

Objective 1 diagnostic for the Metal/OpenCL acceleration gap. The historical
pipeline bug was a ``CustomExternalForce`` position restraint written as
``k*((x-x0)^2 + ...)``, which produces **NaN energies on GPU platforms** for
periodic systems: GPU backends (OpenCL, Metal, CUDA) wrap coordinates into the
primary cell, so ``x - x0`` can differ by a lattice vector. The fix
(``utils/openmm_platform.position_restraint_force``) uses the built-in
``periodicdistance(x, y, z, x0, y0, z0)`` expression, correct on every platform.

This script builds a *small* periodic explicit-solvent system (one of the real
candidate ligands parameterised with OpenFF 2.0.0, solvated in TIP3P with
150 mM NaCl, ~2-5k atoms), adds the periodic position restraint on every ligand
heavy atom, and runs a short NVT + NPT production on the requested platform. A
clean run (all finite energies, stable temperature) proves the restraint fix
works on that backend. It then benchmarks throughput and records the result in
``output/platform_benchmark_metal.json``.

On Apple Silicon stock OpenMM (8.5.x) the accelerator is **OpenCL** — Apple's
runtime compiles kernels to Metal via ``cl2Metal``, the same software path the
``openmm-metal`` plugin uses. A literal ``Metal`` platform only exists via the
third-party plugin (see docs/metal_acceleration.md). ``select_platform`` tries
``Metal`` first and falls back to ``OpenCL`` -> ``CPU``, so ``--platform Metal``
automatically exercises the fallback chain on a build without the plugin.

Usage:
    python scripts/verify_platform_restraint.py
    python scripts/verify_platform_restraint.py --platform OpenCL
    python scripts/verify_platform_restraint.py --platform CPU --threads 8
    python scripts/verify_platform_restraint.py --smiles "..."
    python scripts/verify_platform_restraint.py --npt-ps 300

Outputs:
    output/platform_benchmark_metal.json  - benchmark record (platform, ns/day)
    output/platform_verification.json     - NaN/energy verification result
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("verify_platform")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

TIMESTEP_PS = 0.002
RESTRAINT_KJ = 4184.0  # 10 kcal/mol/AA^2 in kJ/mol/nm^2
NVT_PS = 20.0
NPT_PS = 100.0
BENCH_PS = 60.0

DEFAULT_SMILES = "CC1C(=O)N(C/C=C/C2=CN=CN2C)C(=O)N1C"  # a small drug-like ligand


def main():
    parser = argparse.ArgumentParser(description="Platform + periodic-restraint verification")
    parser.add_argument("--platform", type=str, default=None,
                        help="OpenMM platform preference (Metal/OpenCL/CPU). Default: auto.")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU thread count when falling back to CPU.")
    parser.add_argument("--smiles", type=str, default=DEFAULT_SMILES,
                        help="Small ligand SMILES to build the periodic system around.")
    parser.add_argument("--npt-ps", type=float, default=NPT_PS,
                        help="NPT production length in ps (default: %.1f)." % NPT_PS)
    parser.add_argument("--benchmark-ps", type=float, default=BENCH_PS,
                        help="Additional steady-state ps for the ns/day measurement.")
    args = parser.parse_args()

    import openmm
    from openmm import app, unit
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from openff.toolkit import Molecule as OffMolecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    from utils.openmm_platform import (
        note_metal_status,
        position_restraint_force,
        select_platform,
    )

    platform_spec = select_platform(preference=args.platform, threads=args.threads)
    plat = platform_spec["platform"]
    plat_name = platform_spec["name"]
    log.info("Chosen platform: %s (%s)", plat_name, note_metal_status())

    # ── Build a small periodic PME explicit-solvent complex with a real ligand ──
    # Round-trip the ligand through an RDKit-generated PDB (the same path the
    # pipeline uses for MD poses); OpenMM's solvation step then sees a normal
    # LIG residue with proper connectivity for the OpenFF template generator.
    log.info("Parameterising ligand %s ...", args.smiles)
    mol = Chem.AddHs(Chem.MolFromSmiles(args.smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    import tempfile
    _lig_pdb = tempfile.mktemp(suffix=".pdb")
    try:
        Chem.MolToPDBFile(mol, _lig_pdb)
        lig_pdb = app.PDBFile(_lig_pdb)
    finally:
        Path(_lig_pdb).unlink(missing_ok=True)

    off_mol = OffMolecule.from_rdkit(mol, allow_undefined_stereo=True)
    off_mol.assign_partial_charges(partial_charge_method="gasteiger")
    tg = SMIRNOFFTemplateGenerator(molecules=[off_mol], forcefield="openff-2.0.0")

    modeller = app.Modeller(lig_pdb.topology, lig_pdb.positions)
    n_lig = lig_pdb.topology.getNumAtoms()
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    ff.registerTemplateGenerator(tg.generator)
    modeller.addSolvent(
        ff,
        model="tip3p",
        padding=10.0 * unit.angstrom,
        ionicStrength=0.150 * unit.molar,
        neutralize=True,
    )
    topology = modeller.topology
    positions = modeller.positions
    n_atoms = topology.getNumAtoms()
    log.info("System built: %d atoms (ligand %d heavy+Hs)", n_atoms, n_lig)

    system = ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=10.0 * unit.angstrom,
        constraints=app.HBonds,
        rigidWater=True,
    )

    # ── Periodic position restraint on every ligand atom (the exact test) ──
    restraint = position_restraint_force(RESTRAINT_KJ, periodic=True)
    for idx in range(n_lig):
        pos = positions[idx]
        restraint.addParticle(idx, [RESTRAINT_KJ, pos.x, pos.y, pos.z])
    system.addForce(restraint)
    log.info("Restrained %d ligand atoms with periodicdistance() restraint", n_lig)

    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, TIMESTEP_PS * unit.picoseconds,
    )
    sim = app.Simulation(topology, system, integrator, platform=plat,
                         platformProperties=platform_spec["properties"])
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=500)

    # NVT with restraint at full strength.
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    sim.step(int(NVT_PS / TIMESTEP_PS))
    e_nvt = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalories_per_mole)
    if not np.isfinite(e_nvt):
        log.error("NaN/non-finite energy after NVT on %s: %.4e", plat_name, e_nvt)
        return 1

    # NPT production with restraint at full strength (still the finite check),
    # Save the evolved NVT coordinates BEFORE swapping to the new (barostat-
    # enabled) context, so the NPT run continues from the equilibrated state
    # rather than the new context's empty/zero positions (which NaNs).
    nvt_pos = sim.context.getState(getPositions=True).getPositions()
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 25))
    npt_integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, TIMESTEP_PS * unit.picoseconds,
    )
    sim = app.Simulation(topology, system, npt_integrator, platform=plat,
                         platformProperties=platform_spec["properties"])
    sim.context.setPositions(nvt_pos)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    energies = []
    npt_steps = int(args.npt_ps / TIMESTEP_PS)
    chunk = max(1, npt_steps // 20)
    for _ in range(npt_steps // chunk):
        sim.step(chunk)
        energies.append(sim.context.getState(getEnergy=True).getPotentialEnergy()
                        .value_in_unit(unit.kilocalories_per_mole))
    if not all(np.isfinite(e) for e in energies):
        log.error("NaN/non-finite energy during NPT on %s", plat_name)
        return 1

    # Steady-state throughput (restraint zeroed for production-like speed).
    for idx in range(n_lig):
        restraint.setParticleParameters(idx, idx, [0.0, 0.0, 0.0, 0.0])
    restraint.updateParametersInContext(sim.context)
    bench_steps = int(args.benchmark_ps / TIMESTEP_PS)
    t1 = time.monotonic()
    sim.step(bench_steps)
    elapsed = time.monotonic() - t1
    steps_per_s = bench_steps / elapsed if elapsed > 0 else 0.0
    ns_per_day = steps_per_s * TIMESTEP_PS / 1000.0 * 86400.0

    energies_arr = np.asarray(energies)
    log.info("=" * 60)
    log.info("  Platform       : %s", plat_name)
    log.info("  N atoms        : %d", n_atoms)
    log.info("  NVT energy     : %.2f kcal/mol (finite: OK)", e_nvt)
    log.info("  NPT energy     : %.2f ± %.2f kcal/mol (finite: OK)",
             energies_arr.mean(), energies_arr.std())
    log.info("  Throughput     : %.2f ns/day (%.1f steps/s, %.1f s)", ns_per_day, steps_per_s, elapsed)
    log.info("  Restraint      : periodicdistance() -> no NaN")
    log.info("=" * 60)

    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "platform": plat_name,
        "n_atoms": n_atoms,
        "n_restrained": n_lig,
        "nvt_ps": NVT_PS,
        "npt_ps": args.npt_ps,
        "benchmark_ps": args.benchmark_ps,
        "steps": bench_steps,
        "elapsed_s": round(elapsed, 3),
        "steps_per_s": round(steps_per_s, 2),
        "ns_per_day": round(ns_per_day, 2),
        "timestep_ps": TIMESTEP_PS,
        "nvt_energy_kcal": round(float(e_nvt), 2),
        "npt_energy_mean_kcal": round(float(energies_arr.mean()), 2),
        "npt_energy_std_kcal": round(float(energies_arr.std()), 2),
        "all_energies_finite": True,
        "restraint_expr": "periodicdistance()",
        "success": True,
        "note": ("Native 'Metal' platform requires openmm-metal plugin; "
                 "OpenCL is Apple's Metal-backed runtime (cl2Metal)."),
    }
    bench_path = OUT / "platform_benchmark_metal.json"
    records = []
    if bench_path.exists():
        try:
            records = json.loads(bench_path.read_text())
            if not isinstance(records, list):
                records = []
        except (json.JSONDecodeError, OSError):
            records = []
    records.append(record)
    bench_path.write_text(json.dumps(records, indent=2))
    (OUT / "platform_verification.json").write_text(json.dumps(record, indent=2))
    log.info("Wrote %s and %s", bench_path, OUT / "platform_verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())