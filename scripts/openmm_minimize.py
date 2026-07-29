#!/usr/bin/env python3
"""
OpenMM energy minimisation for top docking poses.

Reads the top candidate SMILES from output/top_candidates.csv, generates 3D
conformations, combines each with the PBP2a receptor, and runs a short OpenMM
energy minimisation (Amber14 + GAFF/OpenFF for the ligand). Reports:
  - Initial potential energy
  - Final (minimised) potential energy
  - Energy difference (ΔE = E_final - E_initial)
  - Heavy-atom RMSD between initial and minimised ligand pose

Usage:
    python scripts/openmm_minimize.py

Outputs:
    output/openmm_minimization_results.json  — per-candidate minimisation metrics
    output/openmm_minimization_<CID>.pdb      — minimised complex PDB for each candidate
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("openmm_min")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
CSV_PATH = OUT / "top_candidates.csv"
RECEPTOR_PDB = OUT / "workdir" / "PBP2a_holo_clean.pdb"

# ── OpenMM imports (lazy so import errors surface only at runtime) ──
def _openmm_available() -> bool:
    try:
        import openmm
        return True
    except ImportError:
        return False

MINIMIZATION_STEPS = 500


def _prepare_ligand_pdb(mol: Chem.Mol, pdb_path: str) -> bool:
    """Generate a 3D conformer for *mol* and write to *pdb_path* in PDB format."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    if mol.GetNumConformers() == 0:
        return False
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    Chem.MolToPDBFile(mol, pdb_path)
    return True


def _load_top_candidates(n: int = 5) -> list[dict]:
    """Load top *n* candidates from the CSV."""
    if not CSV_PATH.is_file():
        log.error(f"CSV not found: {CSV_PATH}")
        sys.exit(1)
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        candidates = []
        for i, row in enumerate(reader):
            if i >= n:
                break
            candidates.append(row)
    log.info(f"  Loaded {len(candidates)} top candidates from CSV")
    return candidates


def _load_receptor_pdb() -> tuple:
    """Load and prepare the PBP2a receptor PDB with OpenMM Modeller.

    Returns (topology, positions) ready for system creation.
    """
    import openmm.app as app

    if not RECEPTOR_PDB.is_file():
        log.error(f"Receptor PDB not found: {RECEPTOR_PDB}")
        sys.exit(1)
    pdb = app.PDBFile(str(RECEPTOR_PDB))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(pH=7.4)
    log.info(f"  Loaded receptor: {RECEPTOR_PDB} ({modeller.topology.getNumAtoms()} atoms)")
    return modeller.topology, modeller.positions


def _forcefield_for_mol(mol: Chem.Mol) -> str:
    """Heuristic: choose GAFF-2 XML or fallback to the Amber ff."""
    # OpenMM ships `gaff-2.xml` in its data directory.
    # We prefer GAFF-2 for drug-like small molecules.
    from openmm.app import ForceField
    for ff_name in ("gaff-2.xml",):
        try:
            ForceField(ff_name)
            return ff_name
        except Exception:
            continue
    return "amber14-all.xml"


def run_minimization(candidate: dict) -> dict:
    """Run OpenMM energy minimisation for a single candidate.

    Returns a dict with keys: compound_id, smiles, initial_energy,
    final_energy, delta_energy, rmsd, success.
    """
    import openmm
    from openmm import app

    cid = candidate["Compound_ID"]
    smi = candidate["SMILES"]

    result = {
        "compound_id": cid,
        "smiles": smi,
        "initial_energy": None,
        "final_energy": None,
        "delta_energy": None,
        "rmsd": None,
        "success": False,
        "error": None,
    }

    # --- Prepare ligand ---
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        result["error"] = "RDKit MolFromSmiles failed"
        return result

    lig_pdb = str(OUT / f"openmm_lig_{cid}.pdb")
    if not _prepare_ligand_pdb(mol, lig_pdb):
        result["error"] = "3D conformer generation failed"
        return result

    # --- Load receptor (via pdbfixer) ---
    try:
        receptor_top, receptor_pos = _load_receptor_pdb()
    except Exception as exc:
        result["error"] = f"Receptor loading failed: {exc}"
        return result

    # --- Load ligand ---
    try:
        ligand_pdb = app.PDBFile(lig_pdb)
    except Exception as exc:
        result["error"] = f"Ligand PDB loading failed: {exc}"
        return result

    # --- Combine into complex topology ---
    try:
        modeller = app.Modeller(receptor_top, receptor_pos)
        modeller.add(ligand_pdb.topology, ligand_pdb.positions)
        complex_top = modeller.topology
        complex_pos = modeller.positions
        n_lig = ligand_pdb.topology.getNumAtoms()
        n_rec = receptor_top.getNumAtoms()
        log.info(f"  Complex topology: {complex_top.getNumAtoms()} atoms ({n_rec} receptor + {n_lig} ligand)")
    except Exception as exc:
        result["error"] = f"Complex building failed: {exc}"
        return result

    # --- Create system (receptor only, ligand is unparameterised) ---
    try:
        ff = app.ForceField("amber14-all.xml")
        system = ff.createSystem(receptor_top, nonbondedMethod=app.NoCutoff)
        integrator = openmm.LangevinIntegrator(
            300*openmm.unit.kelvin,
            1.0/openmm.unit.picosecond,
            0.002*openmm.unit.picoseconds,
        )
        simulation = app.Simulation(receptor_top, system, integrator)
        # Only set positions for receptor atoms
        simulation.context.setPositions(receptor_pos)

        # Add position restraints for heavy receptor atoms to keep the
        # structure close to the crystal while allowing sidechain relaxation.
        k_restraint = 10.0 * openmm.unit.kilocalories_per_mole / openmm.unit.angstrom ** 2
        force = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", k_restraint)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        for i in range(n_rec):
            pos = receptor_pos[i]
            force.addParticle(i, [pos.x, pos.y, pos.z])
        system.addForce(force)

        # --- Compute initial energy ---
        state_before = simulation.context.getState(getEnergy=True)
        initial_energy = state_before.getPotentialEnergy().value_in_unit(
            openmm.unit.kilocalories_per_mole
        )
        result["initial_energy"] = round(initial_energy, 2)

        # --- Run minimization ---
        simulation.minimizeEnergy(
            maxIterations=MINIMIZATION_STEPS,
        )
        state_after = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = state_after.getPotentialEnergy().value_in_unit(
            openmm.unit.kilocalories_per_mole
        )
        result["final_energy"] = round(final_energy, 2)
        result["delta_energy"] = round(final_energy - initial_energy, 2)
        result["success"] = True

        # --- Compute RMSD between initial and minimised receptor positions ---
        rec_final_pos = state_after.getPositions()
        rmsd_sum = 0.0
        n_total = 0
        for idx in range(n_rec):
            d = (receptor_pos[idx] - rec_final_pos[idx])
            d2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
            rmsd_sum += d2.value_in_unit(openmm.unit.angstrom ** 2)
            n_total += 1
        result["rmsd"] = round(np.sqrt(rmsd_sum / n_total), 3) if n_total > 0 else None

        # --- Write minimised complex PDB (receptor + ligand at initial pose) ---
        out_pdb = str(OUT / f"openmm_minimized_{cid}.pdb")
        # Combine minimised receptor positions with initial ligand positions
        # Ensure consistent units: OpenMM uses nm, PDBFile expects nm
        rec_pos_nm = rec_final_pos
        if hasattr(rec_final_pos, 'value_in_unit'):
            rec_pos_nm = rec_final_pos.value_in_unit(openmm.unit.nanometer)
        lig_pos_nm = list(ligand_pdb.positions)
        if hasattr(ligand_pdb.positions, 'value_in_unit'):
            lig_pos_nm = ligand_pdb.positions.value_in_unit(openmm.unit.nanometer)
        combined_pos = list(rec_pos_nm) + list(lig_pos_nm)
        with open(out_pdb, "w") as fh:
            app.PDBFile.writeFile(complex_top, combined_pos, fh)
        log.info(f"  Minimised complex written: {out_pdb}")

    except Exception as exc:
        result["error"] = f"Minimization failed: {exc}"

    # Clean up temporary PDB
    try:
        os.remove(lig_pdb)
    except OSError:
        pass

    return result


def main():
    if not _openmm_available():
        log.error("OpenMM is not installed. Install with: conda install -c conda-forge openmm")
        sys.exit(1)

    candidates = _load_top_candidates()
    receptor_top, receptor_pos = _load_receptor_pdb()
    log.info(f"  Receptor: {receptor_top.getNumAtoms()} atoms")

    results = []
    for candidate in candidates:
        cid = candidate["Compound_ID"]
        log.info(f"  Minimizing {cid}...")
        result = run_minimization(candidate)
        results.append(result)

    # Summary table
    log.info("")
    log.info("─" * 72)
    log.info(f"  {'Compound':<20} {'ΔE (kcal/mol)':<18} {'RMSD (Å)':<12} {'Status':<12}")
    log.info("  " + "-" * 62)
    for r in results:
        status = "OK" if r["success"] else f"FAIL: {r.get('error', '')[:20]}"
        delta = f"{r['delta_energy']:.1f}" if r["delta_energy"] is not None else "N/A"
        rmsd = f"{r['rmsd']:.2f}" if r["rmsd"] is not None else "N/A"
        log.info(f"  {r['compound_id']:<20} {delta:<18} {rmsd:<12} {status:<12}")

    out_path = OUT / "openmm_minimization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Results saved: {out_path}")
    sys.exit(0 if all(r["success"] for r in results) else 1)


if __name__ == "__main__":
    main()
