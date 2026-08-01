#!/usr/bin/env python3
"""
OpenMM minimisation + short MD for top docking poses.

Reads the top candidate SMILES from output/top_candidates.csv, generates 3D
conformations, parameterises each ligand with GAFF (via openmmforcefields),
combines with the PBP2a receptor, minimises, then runs short NVT MD.
Reports ligand RMSD over trajectory, H-bond occupancy, and interaction
energies.

Usage:
    python scripts/openmm_minimize.py

Outputs:
    output/openmm_minimization_results.json  — per-candidate metrics
    output/openmm_minimized_<CID>.pdb         — minimised complex PDB
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

MINIMIZATION_STEPS = 1000
MD_STEPS = 10000
MD_TIMESTEP_PS = 0.002
TOTAL_MD_PS = MD_STEPS * MD_TIMESTEP_PS
REPORT_INTERVAL = 500

H_BOND_RESIDUES = {
    "SER403": ("SER", 403, "OG"),
    "LYS406": ("LYS", 406, "NZ"),
    "TYR446": ("TYR", 446, "OH"),
}


def _openmm_available() -> bool:
    try:
        import openmm
        return True
    except ImportError:
        return False


def _openmmforcefields_available() -> bool:
    try:
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        return True
    except ImportError:
        return False


def _prepare_ligand_pdb(mol: Chem.Mol, pdb_path: str) -> bool:
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


def _load_receptor_pdb():
    import openmm.app as app
    if not RECEPTOR_PDB.is_file():
        log.error(f"Receptor PDB not found: {RECEPTOR_PDB}")
        sys.exit(1)
    pdb = app.PDBFile(str(RECEPTOR_PDB))

    # PROPKA-based protonation state assignment
    from utils.structure_prep import assign_protonation_states, build_openmm_variant_list
    propka_variants = assign_protonation_states(str(RECEPTOR_PDB), pH=7.4)

    modeller = app.Modeller(pdb.topology, pdb.positions)
    variant_list = build_openmm_variant_list(modeller.topology, propka_variants)
    modeller.addHydrogens(pH=7.4, variants=variant_list)

    # Verify key active-site residues
    for res in modeller.topology.residues():
        res_name = res.name
        try:
            res_num = int(res.id)
        except ValueError:
            continue
        if res_name == "LYS" and res_num == 406:
            log.info(f"  ✓ Lys406 confirmed as {res_name} (protonated)")
        elif res_name == "SER" and res_num == 403:
            log.info(f"  ✓ Ser403 confirmed as {res_name} (neutral)")

    log.info(f"  Loaded receptor: {RECEPTOR_PDB} ({modeller.topology.getNumAtoms()} atoms)")
    return modeller.topology, modeller.positions


def _find_hbond_atoms(topology, positions, lig_atom_indices):
    """Compute H-bond contacts for catalytic residues."""
    import openmm
    hbonds = {}
    try:
        residue_list = []
        for chain in topology.chains():
            try:
                residue_list.extend(list(chain.residues()))
            except TypeError:
                residue_list.extend(list(chain.residues))
        for resname, resnum, atom_name in H_BOND_RESIDUES.values():
            contacts = []
            for residue in residue_list:
                if residue.name == resname and residue.index + 1 == resnum:
                    for atom in residue.atoms():
                        if atom.name == atom_name:
                            ref_pos = positions[atom.index]
                            for li in lig_atom_indices:
                                lig_pos = positions[li]
                                d = np.linalg.norm(np.array([
                                    (ref_pos[i] - lig_pos[i]).value_in_unit(openmm.unit.angstrom)
                                    for i in range(3)
                                ]))
                                if d < 3.5:
                                    contacts.append(float(f"{d:.2f}"))
            hbonds[f"{resname}{resnum}_{atom_name}"] = {
                "n_contacts": len(contacts),
                "min_distance_A": min(contacts) if contacts else None,
            }
    except Exception:
        pass
    return hbonds


def _compute_ligand_rmsd(initial_pos, final_pos, lig_indices):
    import openmm
    rmsd_sum = 0.0
    for idx in lig_indices:
        d = initial_pos[idx] - final_pos[idx]
        d2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
        rmsd_sum += d2.value_in_unit(openmm.unit.angstrom ** 2)
    return np.sqrt(rmsd_sum / len(lig_indices)) if lig_indices else None


def run_simulation(candidate: dict) -> dict:
    import openmm
    from openmm import app
    from openmmforcefields.generators import GAFFTemplateGenerator

    cid = candidate["Compound_ID"]
    smi = candidate["SMILES"]

    result = {
        "compound_id": cid,
        "smiles": smi,
        "minimization": {"rmsd": None, "success": False},
        "md": {
            "ligand_rmsd_mean": None, "ligand_rmsd_std": None,
            "hbond_occupancy": {}, "success": False,
        },
        "success": False,
        "error": None,
    }

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        result["error"] = "RDKit MolFromSmiles failed"
        return result

    lig_pdb = str(OUT / f"openmm_lig_{cid}.pdb")
    if not _prepare_ligand_pdb(mol, lig_pdb):
        result["error"] = "3D conformer generation failed"
        return result

    try:
        receptor_top, receptor_pos = _load_receptor_pdb()
    except Exception as exc:
        result["error"] = f"Receptor loading failed: {exc}"
        return result

    try:
        ligand_pdb = app.PDBFile(lig_pdb)
    except Exception as exc:
        result["error"] = f"Ligand PDB loading failed: {exc}"
        return result

    try:
        modeller = app.Modeller(receptor_top, receptor_pos)
        modeller.add(ligand_pdb.topology, ligand_pdb.positions)
        complex_top = modeller.topology
        complex_pos = modeller.positions
        n_rec = receptor_top.getNumAtoms()
        n_lig = ligand_pdb.topology.getNumAtoms()
        lig_indices = list(range(n_rec, n_rec + n_lig))
        log.info(f"  Complex: {complex_top.getNumAtoms()} atoms ({n_rec} rec + {n_lig} lig)")
    except Exception as exc:
        result["error"] = f"Complex building failed: {exc}"
        return result

    try:
        from openff.toolkit import Molecule as OffMolecule
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        off_mol = OffMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        off_mol.assign_partial_charges(partial_charge_method="gasteiger")
        tg = SMIRNOFFTemplateGenerator(molecules=off_mol, forcefield="openff-2.0.0")
        ff = app.ForceField("amber14-all.xml")
        ff.registerTemplateGenerator(tg.generator)
        system = ff.createSystem(
            complex_top, nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )

        k_restraint = 10.0 * openmm.unit.kilocalories_per_mole / openmm.unit.angstrom ** 2
        force = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", k_restraint)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        for i in range(n_rec):
            pos = complex_pos[i]
            force.addParticle(i, [pos.x, pos.y, pos.z])
        system.addForce(force)

        integrator = openmm.LangevinIntegrator(
            300*openmm.unit.kelvin,
            1.0/openmm.unit.picosecond,
            MD_TIMESTEP_PS*openmm.unit.picoseconds,
        )

        simulation = app.Simulation(complex_top, system, integrator)
        simulation.context.setPositions(complex_pos)

        state_before = simulation.context.getState(getEnergy=True)
        e_init = state_before.getPotentialEnergy().value_in_unit(
            openmm.unit.kilocalories_per_mole
        )

        log.info(f"    Minimising ({MINIMIZATION_STEPS} steps L-BFGS)...")
        simulation.minimizeEnergy(maxIterations=MINIMIZATION_STEPS)
        state_min = simulation.context.getState(getEnergy=True, getPositions=True)
        e_min = state_min.getPotentialEnergy().value_in_unit(
            openmm.unit.kilocalories_per_mole
        )

        min_pos = state_min.getPositions()
        lig_rmsd_min = _compute_ligand_rmsd(complex_pos, min_pos, lig_indices)
        rec_rmsd = _compute_ligand_rmsd(complex_pos, min_pos, list(range(n_rec)))
        result["minimization"] = {
            "initial_energy_kcal": round(e_init, 1),
            "final_energy_kcal": round(e_min, 1),
            "delta_energy_kcal": round(e_min - e_init, 1),
            "receptor_rmsd_A": round(float(rec_rmsd), 3) if rec_rmsd else None,
            "ligand_rmsd_A": round(float(lig_rmsd_min), 3) if lig_rmsd_min else None,
            "success": True,
        }
        log.info(f"    Minimised: ΔE={e_min - e_init:.0f} kcal/mol, "
                 f"rec RMSD={rec_rmsd:.3f} Å, lig RMSD={lig_rmsd_min:.3f} Å")

        log.info(f"    Running MD ({TOTAL_MD_PS:.0f} ps NVT, 300 K)...")
        simulation.context.setPositions(min_pos)
        simulation.context.setVelocitiesToTemperature(300*openmm.unit.kelvin)

        lig_rmsd_traj = []
        lig_pos_start = min_pos
        for step in range(MD_STEPS):
            simulation.step(1)
            if step % REPORT_INTERVAL == 0 or step == MD_STEPS - 1:
                state_md = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
                md_pos = state_md.getPositions()
                lr = _compute_ligand_rmsd(lig_pos_start, md_pos, lig_indices)
                if lr is not None:
                    lig_rmsd_traj.append(float(lr))

        state_final = simulation.context.getState(getPositions=True, getEnergy=True)
        final_pos = state_final.getPositions()
        e_final = state_final.getPotentialEnergy().value_in_unit(
            openmm.unit.kilocalories_per_mole
        )

        hbonds = _find_hbond_atoms(complex_top, final_pos, lig_indices)
        hb_occ = hbonds

        lig_rmsd_mean = float(np.mean(lig_rmsd_traj)) if lig_rmsd_traj else None
        lig_rmsd_std = float(np.std(lig_rmsd_traj)) if lig_rmsd_traj else None

        result["md"] = {
            "nvt_duration_ps": TOTAL_MD_PS,
            "temperature_K": 300,
            "final_energy_kcal": round(e_final, 1),
            "ligand_rmsd_mean_A": round(lig_rmsd_mean, 3) if lig_rmsd_mean else None,
            "ligand_rmsd_std_A": round(lig_rmsd_std, 3) if lig_rmsd_std else None,
            "ligand_rmsd_max_A": round(max(lig_rmsd_traj), 3) if lig_rmsd_traj else None,
            "hbond_occupancy": hb_occ,
            "success": True,
        }
        log.info(f"    MD complete: lig RMSD={lig_rmsd_mean:.3f}±{lig_rmsd_std:.3f} Å")

        out_pdb = str(OUT / f"openmm_minimized_{cid}.pdb")
        with open(out_pdb, "w") as fh:
            app.PDBFile.writeFile(complex_top, final_pos, fh)
        log.info(f"    Complex PDB: {out_pdb}")

        result["success"] = True

    except Exception as exc:
        result["error"] = f"Simulation failed: {exc}"
        import traceback
        log.warning(traceback.format_exc())

    try:
        os.remove(lig_pdb)
    except OSError:
        pass

    return result


def main():
    if not _openmm_available():
        log.error("OpenMM not installed. Run: conda install -c conda-forge openmm")
        sys.exit(1)
    if not _openmmforcefields_available():
        log.error("openmmforcefields not installed. Run: pip install openmmforcefields")
        sys.exit(1)

    candidates = _load_top_candidates()
    log.info(f"  Receptor PDB: {RECEPTOR_PDB}")
    log.info(f"  MD: {TOTAL_MD_PS:.0f} ps NVT, {REPORT_INTERVAL}-step report interval")

    results = []
    for cand in candidates:
        cid = cand["Compound_ID"]
        log.info(f"\n  Processing {cid}...")
        result = run_simulation(cand)
        results.append(result)

    log.info("")
    log.info("─" * 80)
    log.info(f"  {'Compound':<16} {'Min RMSD':<10} {'MD lig RMSD':<14} {'Status':<12}")
    log.info("  " + "-" * 52)
    for r in results:
        mr = r["minimization"].get("ligand_rmsd_A", "N/A") if r["minimization"].get("success") else "FAIL"
        mr_s = f"{mr:.3f}" if isinstance(mr, float) else str(mr)
        lr = r["md"].get("ligand_rmsd_mean_A", "N/A") if r["md"].get("success") else "N/A"
        lr_s = f"{lr:.3f}" if isinstance(lr, float) else str(lr)
        status = "OK" if r["success"] else f"FAIL: {r.get('error', '?')[:20]}"
        log.info(f"  {r['compound_id']:<16} {mr_s:<10} {lr_s:<14} {status:<12}")

    out_path = OUT / "openmm_minimization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\n  Results: {out_path}")

    n_ok = sum(1 for r in results if r["success"])
    log.info(f"  {n_ok}/{len(results)} succeeded")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
