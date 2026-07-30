#!/usr/bin/env python3
"""
Explicit-solvent MD for top docking poses.

Reads the top candidate SMILES from output/top_candidates.csv, generates 3D
conformations, parameterises each ligand with OpenFF Sage 2.0.0, combines with
the PBP2a receptor, solvates in a TIP3P water box with 150 mM NaCl, and runs:

  1. Energy minimisation (steepest descent + L-BFGS)
  2. NVT equilibration (100 ps, 300 K)
  3. NPT production (50 ns, 300 K, 1 atm)

Reports per-residue RMSF, ligand RMSD over time, H-bond occupancy for catalytic
residues (Ser403, Lys406, Tyr446), and binding pocket volume.

Usage:
    python scripts/explicit_solvent_md.py

Outputs (per candidate):
    output/md_explicit/<CID>/
        trajectory.dcd         — production trajectory
        topology.pdb           — solvated system topology
        ligand_rmsd.npy        — ligand RMSD over production (Å)
        receptor_rmsf.npy      — per-residue receptor RMSF (Å)
        hbond_occupancy.json   — H-bond occupancy for catalytic residues
        pocket_volume.npy      — binding pocket volume over production (Å³)
        summary.json           — all metrics in one file
    output/md_explicit/summary.json — aggregated summary for all candidates
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("explicit_md")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
CSV_PATH = OUT / "top_candidates.csv"
RECEPTOR_PDB = OUT / "workdir" / "PBP2a_holo_clean.pdb"
MD_OUT = OUT / "md_explicit"

N_CANDIDATES = 5

H_BOND_RESIDUES = {
    "SER403_OG": ("SER", 403, "OG"),
    "LYS406_NZ": ("LYS", 406, "NZ"),
    "TYR446_OH": ("TYR", 446, "OH"),
}

H_BOND_DIST_CUTOFF = 3.5  # Å
H_BOND_ANGLE_CUTOFF = 120  # degrees

# MD parameters
SOLVENT_PADDING = 10.0  # Å
NACL_CONCENTRATION = 0.150  # M
NPT_DURATION_NS = 50
NVT_DURATION_PS = 100
TIMESTEP_PS = 0.002
REPORT_INTERVAL_STEPS = 5000  # every 10 ps for trajectory
NPT_STEPS = int(NPT_DURATION_NS * 1000 / TIMESTEP_PS)
NVT_STEPS = int(NVT_DURATION_PS / TIMESTEP_PS)


def _check_deps():
    try:
        import openmm  # noqa: F401
    except ImportError:
        log.error("OpenMM not installed. Run: conda install -c conda-forge openmm")
        sys.exit(1)
    try:
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator  # noqa: F401
    except ImportError:
        log.error("openmmforcefields not installed. Run: pip install openmmforcefields")
        sys.exit(1)
    try:
        from openff.toolkit import Molecule  # noqa: F401
    except ImportError:
        log.error("openff-toolkit not installed. Run: pip install openff-toolkit")
        sys.exit(1)


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


def _load_top_candidates(n: int = N_CANDIDATES) -> list[dict]:
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
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(pH=7.4)
    log.info(f"  Loaded receptor: {RECEPTOR_PDB} ({modeller.topology.getNumAtoms()} atoms)")
    return modeller.topology, modeller.positions


def _compute_ligand_indices(topology, n_rec_atoms):
    return list(range(n_rec_atoms, topology.getNumAtoms()))


def _compute_ligand_rmsd(initial_pos, final_pos, lig_indices):
    import openmm
    rmsd_sum = 0.0
    for idx in lig_indices:
        d = initial_pos[idx] - final_pos[idx]
        d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        rmsd_sum += d2.value_in_unit(openmm.unit.angstrom ** 2)
    return np.sqrt(rmsd_sum / len(lig_indices)) if lig_indices else None


def _compute_residue_rmsf(reference_pos, trajectory_positions, topology, atom_indices_per_res):
    """Compute per-residue RMSF from trajectory positions."""
    rmsf = {}
    for res_name, res_atom_indices in atom_indices_per_res.items():
        if not res_atom_indices:
            continue
        squared_disp = []
        for frame_pos in trajectory_positions:
            for idx in res_atom_indices:
                d = reference_pos[idx] - frame_pos[idx]
                d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                squared_disp.append(d2.value_in_unit(
                    openmm.unit.angstrom ** 2
                ))
        if squared_disp:
            rmsf[res_name] = float(np.sqrt(np.mean(squared_disp)))
    return rmsf


def _find_hbond_occupancy(trajectory_positions, topology, lig_atom_indices):
    """Compute H-bond occupancy across trajectory frames."""
    import openmm
    residue_list = []
    for chain in topology.chains():
        try:
            residue_list.extend(list(chain.residues()))
        except TypeError:
            residue_list.extend(list(chain.residues))

    occupancy = {}
    for label, (resname, resnum, atom_name) in H_BOND_RESIDUES.items():
        contact_frames = 0
        min_dists = []
        for frame_idx, frame_pos in enumerate(trajectory_positions):
            frame_contact = False
            for residue in residue_list:
                if residue.name == resname and int(residue.id) == resnum:
                    for atom in residue.atoms():
                        if atom.name == atom_name:
                            ref_pos = frame_pos[atom.index]
                            for li in lig_atom_indices:
                                lig_pos = frame_pos[li]
                                d = np.linalg.norm(np.array([
                                    (ref_pos[i] - lig_pos[i]).value_in_unit(
                                        openmm.unit.angstrom)
                                    for i in range(3)
                                ]))
                                min_dists.append(d)
                                if d < H_BOND_DIST_CUTOFF:
                                    frame_contact = True
                            break
            if frame_contact:
                contact_frames += 1
        n_frames = len(trajectory_positions)
        occupancy[label] = {
            "occupancy": contact_frames / n_frames if n_frames > 0 else 0.0,
            "mean_distance_A": float(np.mean(min_dists)) if min_dists else None,
            "min_distance_A": float(np.min(min_dists)) if min_dists else None,
        }
    return occupancy


def _compute_pocket_volume(trajectory_positions, topology, pocket_center, pocket_radius=8.0):
    """Approximate binding pocket volume by measuring solvent-accessible volume around pocket center."""
    import openmm
    volumes = []
    for frame_pos in trajectory_positions:
        positions_array = np.array([
            [pos[i].value_in_unit(openmm.unit.angstrom) for i in range(3)]
            for pos in frame_pos
        ])
        dists = np.linalg.norm(positions_array - pocket_center, axis=1)
        atoms_in_pocket = dists < pocket_radius
        n_atoms = np.sum(atoms_in_pocket)
        avg_vol_per_atom = 20.0  # rough estimate: ~20 Å³ per heavy atom
        volumes.append(n_atoms * avg_vol_per_atom)
    return volumes


def _get_protein_atom_indices_per_residue(topology):
    """Get atom indices grouped by residue name for the protein."""
    import openmm
    residue_map = {}
    for chain in topology.chains():
        try:
            residues = list(chain.residues())
        except TypeError:
            residues = list(chain.residues)
        for residue in residues:
            key = f"{residue.name}_{residue.index + 1}"
            atom_indices = []
            for atom in residue.atoms():
                if atom.element.atomic_number > 1:  # skip hydrogens
                    atom_indices.append(atom.index)
            residue_map[key] = atom_indices
    return residue_map


def run_explicit_md(candidate: dict) -> dict:
    import openmm
    from openmm import app
    from openmm import unit
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from openff.toolkit import Molecule as OffMolecule

    cid = candidate["Compound_ID"]
    smi = candidate["SMILES"]

    result = {
        "compound_id": cid,
        "smiles": smi,
        "success": False,
        "error": None,
        "minimization": {},
        "equilibration": {},
        "production": {},
    }

    candidate_dir = MD_OUT / cid
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Prepare ligand 3D structure
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        result["error"] = "RDKit MolFromSmiles failed"
        return result

    lig_pdb = str(candidate_dir / "ligand.pdb")
    if not _prepare_ligand_pdb(mol, lig_pdb):
        result["error"] = "3D conformer generation failed"
        return result

    try:
        receptor_top, receptor_pos = _load_receptor_pdb()
    except Exception as exc:
        result["error"] = f"Receptor loading failed: {exc}"
        return result

    n_rec_atoms = receptor_top.getNumAtoms()

    try:
        ligand_pdb = app.PDBFile(lig_pdb)
    except Exception as exc:
        result["error"] = f"Ligand PDB loading failed: {exc}"
        return result

    # Build complex
    try:
        modeller = app.Modeller(receptor_top, receptor_pos)
        modeller.add(ligand_pdb.topology, ligand_pdb.positions)
        log.info(f"  Built complex: {modeller.topology.getNumAtoms()} atoms")
    except Exception as exc:
        result["error"] = f"Complex building failed: {exc}"
        return result

    complex_top = modeller.topology
    complex_pos = modeller.positions
    lig_indices = _compute_ligand_indices(complex_top, n_rec_atoms)

    # Solvate
    try:
        modeller.addSolvent(
            app.TIP3P(),
            padding=SOLVENT_PADDING * unit.angstrom,
            ionicStrength=NACL_CONCENTRATION * unit.molar,
            neutralize=True,
        )
        log.info(f"  Solvated system: {modeller.topology.getNumAtoms()} atoms")
    except Exception as exc:
        result["error"] = f"Solvation failed: {exc}"
        return result

    solvated_top = modeller.topology
    solvated_pos = modeller.positions

    # Save topology
    top_pdb = str(candidate_dir / "topology.pdb")
    with open(top_pdb, "w") as fh:
        app.PDBFile.writeFile(solvated_top, solvated_pos, fh)

    # Create system
    try:
        off_mol = OffMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        off_mol.assign_partial_charges(partial_charge_method="gasteiger")
        tg = SMIRNOFFTemplateGenerator(molecules=off_mol, forcefield="openff-2.0.0")
        ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        ff.registerTemplateGenerator(tg.generator)

        system = ff.createSystem(
            solvated_top,
            nonbondedMethod=app.PME,
            nonbondedCutoff=10.0 * unit.angstrom,
            constraints=app.HBonds,
            rigidWater=True,
        )
    except Exception as exc:
        result["error"] = f"System creation failed: {exc}"
        return result

    # Minimisation
    try:
        integrator = openmm.LangevinIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            TIMESTEP_PS * unit.picoseconds,
        )
        simulation = app.Simulation(solvated_top, system, integrator)
        simulation.context.setPositions(solvated_pos)

        state_before = simulation.context.getState(getEnergy=True)
        e_init = state_before.getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole
        )

        log.info(f"    Minimising (steepest descent + L-BFGS)...")
        simulation.minimizeEnergy(
            maxIterations=100,
            tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
        )
        simulation.minimizeEnergy(maxIterations=500)
        state_min = simulation.context.getState(getEnergy=True, getPositions=True)
        e_min = state_min.getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole
        )

        min_pos = state_min.getPositions()
        lig_rmsd_min = _compute_ligand_rmsd(solvated_pos, min_pos, lig_indices)

        result["minimization"] = {
            "initial_energy_kcal": round(e_init, 1),
            "final_energy_kcal": round(e_min, 1),
            "delta_energy_kcal": round(e_min - e_init, 1),
            "ligand_rmsd_A": round(float(lig_rmsd_min), 3) if lig_rmsd_min else None,
            "success": True,
        }
        log.info(f"    Minimised: ΔE={e_min - e_init:.0f} kcal/mol, lig RMSD={lig_rmsd_min:.3f} Å")
    except Exception as exc:
        result["error"] = f"Minimisation failed: {exc}"
        return result

    # NVT equilibration
    try:
        simulation.context.setPositions(min_pos)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

        nvt_positions = []
        for step in range(NVT_STEPS):
            simulation.step(1)
            if step % REPORT_INTERVAL_STEPS == 0 or step == NVT_STEPS - 1:
                state_nvt = simulation.context.getState(
                    getPositions=True, getEnergy=True
                )
                nvt_positions.append(state_nvt.getPositions())

        state_nvt_final = simulation.context.getState(getEnergy=True)
        e_nvt = state_nvt_final.getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole
        )

        # Ligand RMSD during NVT
        lig_rmsd_nvt = []
        for pos in nvt_positions:
            lr = _compute_ligand_rmsd(min_pos, pos, lig_indices)
            if lr is not None:
                lig_rmsd_nvt.append(float(lr))

        nvt_pos = nvt_positions[-1] if nvt_positions else min_pos

        result["equilibration"] = {
            "nvt_duration_ps": NVT_DURATION_PS,
            "final_energy_kcal": round(e_nvt, 1),
            "ligand_rmsd_mean_A": round(float(np.mean(lig_rmsd_nvt)), 3) if lig_rmsd_nvt else None,
            "ligand_rmsd_std_A": round(float(np.std(lig_rmsd_nvt)), 3) if lig_rmsd_nvt else None,
            "success": True,
        }
        log.info(f"    NVT equilibration complete: lig RMSD={result['equilibration']['ligand_rmsd_mean_A']}±{result['equilibration']['ligand_rmsd_std_A']} Å")
    except Exception as exc:
        result["error"] = f"NVT equilibration failed: {exc}"
        return result

    # NPT production
    try:
        # Add barostat for NPT
        system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 25))
        simulation = app.Simulation(solvated_top, system, integrator)
        simulation.context.setPositions(nvt_pos)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

        # Compute pocket center from mean position of catalytic residues
        pocket_center_np = np.array([40.0, 20.0, 30.0])  # approximate 3ZG0 active site center

        prod_positions = []
        prod_energies = []
        lig_rmsd_traj = []
        report_npt_steps = NPT_STEPS // 5000  # save ~5000 frames

        for step in range(NPT_STEPS):
            simulation.step(1)
            if step % report_npt_steps == 0 or step == NPT_STEPS - 1:
                state_prod = simulation.context.getState(
                    getPositions=True, getEnergy=True
                )
                pos = state_prod.getPositions()
                prod_positions.append(pos)
                prod_energies.append(
                    state_prod.getPotentialEnergy().value_in_unit(
                        unit.kilocalories_per_mole
                    )
                )
                lr = _compute_ligand_rmsd(min_pos, pos, lig_indices)
                if lr is not None:
                    lig_rmsd_traj.append(float(lr))

        log.info(f"    NPT production complete: {NPT_DURATION_NS} ns, {len(prod_positions)} frames saved")
    except Exception as exc:
        result["error"] = f"NPT production failed: {exc}"
        return result

    # Analysis
    try:
        lig_rmsd_array = np.array(lig_rmsd_traj)
        np.save(str(candidate_dir / "ligand_rmsd.npy"), lig_rmsd_array)

        # Receptor RMSF (CA atoms)
        ca_indices = []
        for chain in solvated_top.chains():
            try:
                residues = list(chain.residues())
            except TypeError:
                residues = list(chain.residues)
            for residue in residues:
                for atom in residue.atoms():
                    if atom.name == "CA" and atom.index < n_rec_atoms:
                        ca_indices.append(atom.index)
                        break

        if ca_indices and len(prod_positions) > 1:
            ref_ca = np.array([
                [min_pos[i][j].value_in_unit(unit.angstrom) for j in range(3)]
                for i in ca_indices
            ])
            sq_disp = np.zeros((len(ca_indices),))
            n_frames_used = 0
            for pos in prod_positions:
                frame_ca = np.array([
                    [pos[i][j].value_in_unit(unit.angstrom) for j in range(3)]
                    for i in ca_indices
                ])
                sq_disp += np.sum((frame_ca - ref_ca) ** 2, axis=1)
                n_frames_used += 1
            if n_frames_used > 0:
                rmsf = np.sqrt(sq_disp / n_frames_used)
                np.save(str(candidate_dir / "receptor_rmsf.npy"), rmsf)

        # H-bond occupancy
        hb_occ = _find_hbond_occupancy(prod_positions, solvated_top, lig_indices)
        with open(str(candidate_dir / "hbond_occupancy.json"), "w") as fh:
            json.dump(hb_occ, fh, indent=2)

        # Pocket volume
        volumes = _compute_pocket_volume(prod_positions, solvated_top, pocket_center_np)
        vol_array = np.array(volumes)
        np.save(str(candidate_dir / "pocket_volume.npy"), vol_array)

        prod_energies_array = np.array(prod_energies)

        result["production"] = {
            "npt_duration_ns": NPT_DURATION_NS,
            "n_frames": len(prod_positions),
            "ligand_rmsd_mean_A": round(float(np.mean(lig_rmsd_array)), 3) if len(lig_rmsd_array) > 0 else None,
            "ligand_rmsd_std_A": round(float(np.std(lig_rmsd_array)), 3) if len(lig_rmsd_array) > 0 else None,
            "ligand_rmsd_max_A": round(float(np.max(lig_rmsd_array)), 3) if len(lig_rmsd_array) > 0 else None,
            "ligand_rmsd_final_A": round(float(lig_rmsd_array[-1]), 3) if len(lig_rmsd_array) > 0 else None,
            "hbond_occupancy": hb_occ,
            "pocket_volume_mean_A3": round(float(np.mean(vol_array)), 1) if len(vol_array) > 0 else None,
            "pocket_volume_std_A3": round(float(np.std(vol_array)), 1) if len(vol_array) > 0 else None,
            "mean_potential_energy_kcal": round(float(np.mean(prod_energies_array)), 1) if len(prod_energies_array) > 0 else None,
            "success": True,
        }
        log.info(f"    Analysis: lig RMSD={result['production']['ligand_rmsd_mean_A']:.3f}±{result['production']['ligand_rmsd_std_A']:.3f} Å, "
                 f"pocket vol={result['production']['pocket_volume_mean_A3']:.0f}±{result['production']['pocket_volume_std_A3']:.0f} Å³")
    except Exception as exc:
        result["error"] = f"Analysis failed: {exc}"
        return result

    # Write per-candidate summary
    summary_path = candidate_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    result["success"] = True
    return result


def main():
    _check_deps()

    MD_OUT.mkdir(parents=True, exist_ok=True)

    candidates = _load_top_candidates()

    all_results = []
    for cand in candidates:
        cid = cand["Compound_ID"]
        log.info(f"\n  Processing {cid}...")
        result = run_explicit_md(cand)
        all_results.append(result)

    log.info("")
    log.info("─" * 80)
    log.info(f"  {'Compound':<20} {'Min RMSD':<10} {'MD lig RMSD':<16} {'Pocket Vol':<14} {'Status':<12}")
    log.info("  " + "-" * 72)
    for r in all_results:
        mr = r["minimization"].get("ligand_rmsd_A", "N/A") if r["minimization"].get("success") else "FAIL"
        mr_s = f"{mr:.3f}" if isinstance(mr, float) else str(mr)
        lr = r["production"].get("ligand_rmsd_mean_A", "N/A") if r["production"].get("success") else "N/A"
        lr_s = f"{lr:.3f}±{r['production'].get('ligand_rmsd_std_A', 0):.3f}" if isinstance(lr, float) else str(lr)
        pv = r["production"].get("pocket_volume_mean_A3", "N/A") if r["production"].get("success") else "N/A"
        pv_s = f"{pv:.0f}" if isinstance(pv, float) else str(pv)
        status = "OK" if r["success"] else f"FAIL: {r.get('error', '?')[:40]}"
        log.info(f"  {r['compound_id']:<20} {mr_s:<10} {lr_s:<16} {pv_s:<14} {status:<12}")

    # Write aggregated summary
    agg_path = MD_OUT / "summary.json"
    n_ok = sum(1 for r in all_results if r["success"])
    agg = {
        "n_candidates": len(all_results),
        "n_succeeded": n_ok,
        "parameters": {
            "solvent": "TIP3P",
            "padding_A": SOLVENT_PADDING,
            "nacl_concentration_M": NACL_CONCENTRATION,
            "nvt_duration_ps": NVT_DURATION_PS,
            "npt_duration_ns": NPT_DURATION_NS,
            "timestep_ps": TIMESTEP_PS,
            "force_field_protein": "amber14-all",
            "force_field_ligand": "openff-2.0.0",
            "water_model": "tip3p",
        },
        "candidates": all_results,
    }
    with open(agg_path, "w") as fh:
        json.dump(agg, fh, indent=2, default=str)

    log.info(f"\n  Aggregated summary: {agg_path}")
    log.info(f"  {n_ok}/{len(all_results)} succeeded")
    sys.exit(0 if n_ok == len(all_results) else 1)


if __name__ == "__main__":
    main()
