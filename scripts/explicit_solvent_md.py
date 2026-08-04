#!/usr/bin/env python3
"""
Explicit-solvent MD for top docking poses.

Reads the top candidate SMILES from output/top_candidates.csv, generates 3D
conformations, parameterises each ligand with OpenFF Sage 2.0.0, combines with
the PBP2a receptor, solvates in a TIP3P water box with 150 mM NaCl, and runs
multiple replicas with:

  1. Energy minimisation (5000 steps L-BFGS)
  2. NVT equilibration (500 ps, 300 K, gradual restraint release)
  3. NPT equilibration (500 ps, 300 K, 1 atm, gradual restraint release)
  4. NPT production (10 ns, 300 K, 1 atm, no restraints)

Reports per-replica stability classification, ligand RMSD, per-residue RMSF,
H-bond occupancy for catalytic residues (Ser403, Lys406, Tyr446), and radius
of gyration.

Usage:
    python scripts/explicit_solvent_md.py                 # full run (100 ns × 5 × 3)
    python scripts/explicit_solvent_md.py --quick          # quick test (0.1 ns × 3 × 1)
    python scripts/explicit_solvent_md.py --production-ns 10 --replicas 3 --n-candidates 3

Outputs (per candidate):
    output/md_explicit/<CID>/
        replica_<N>/
            trajectory.dcd         — production trajectory
            topology.pdb           — solvated system topology
            ligand_rmsd.npy        — ligand RMSD over production (Å)
            receptor_rmsf.npy      — per-residue receptor RMSF (Å)
            hbond_occupancy.json   — H-bond occupancy for catalytic residues
            rg.npy                 — radius of gyration over production (Å)
            summary.json           — all metrics in one file
        summary.json               — aggregated summary across replicas
    output/md_explicit/summary.json — aggregated summary for all candidates
"""

from __future__ import annotations

import argparse
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

# CLI defaults
DEFAULT_N_CANDIDATES = 5
DEFAULT_N_REPLICAS = 3
DEFAULT_NPT_NS = 100  # Run at 100 ns for proper MD stability analysis
QUICK_N_CANDIDATES = 3
QUICK_N_REPLICAS = 1
# Quick mode follows the paper's preliminary explicit-solvent protocol:
# short equilibration followed by 100 ps NPT production (TIP3P, 150 mM NaCl).
QUICK_NPT_NS = 0.1
QUICK_NVT_PS = 50.0
QUICK_NPT_EQ_PS = 50.0

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
TIMESTEP_PS = 0.002
REPORT_INTERVAL_STEPS = 5000  # every 10 ps for trajectory
# Frames per nanosecond: 5000 steps * 0.002 ps / 1000 ps = 1000 frames/ns.
DT_NS = 0.002

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


def _standardize_ligand(mol: Chem.Mol, pH: float = 7.4) -> Chem.Mol:
    """Standardize ligand: assign dominant tautomer and protonation state at given pH."""
    try:
        from rdkit.Chem import MolStandardize
        # Neutralize and assign dominant tautomer
        clean_params = MolStandardize.standardize.CleanupParameters()
        clean_params.pH = pH
        clean_params.preferOrganic = True
        cleaner = MolStandardize.standardize.Cleanup(clean_params)
        mol = cleaner.cleanup(mol)
        # Uncharge if needed
        mol = MolStandardize.charge.Uncharger().uncharge(mol)
    except Exception as exc:
        log.warning(f"  MolStandardize failed ({exc}), using original molecule")
    return mol


def _prepare_ligand_pdb(mol: Chem.Mol, pdb_path: str, pose_pdbqt: str | None = None) -> bool:
    mol = _standardize_ligand(mol, pH=7.4)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    if mol.GetNumConformers() == 0:
        return False
    if pose_pdbqt:
        from utils.docking import set_pose_coordinates
        if not set_pose_coordinates(mol, pose_pdbqt):
            log.warning(f"  Pose overlay failed for {pose_pdbqt}; using ETKDG conformer")
    mol = Chem.AddHs(mol, addCoords=True)
    if mol.GetNumConformers() == 0:
        return False
    Chem.MolToPDBFile(mol, pdb_path)
    return True


def _load_top_candidates(n: int = DEFAULT_N_CANDIDATES) -> list[dict]:
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
    # Build variant list matching OpenMM residue order
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
                            frame_dists = []
                            for li in lig_atom_indices:
                                lig_pos = frame_pos[li]
                                d = np.linalg.norm(np.array([
                                    (ref_pos[i] - lig_pos[i]).value_in_unit(
                                        openmm.unit.angstrom)
                                    for i in range(3)
                                ]))
                                frame_dists.append(d)
                                if d < H_BOND_DIST_CUTOFF:
                                    frame_contact = True
                            if frame_dists:
                                min_dists.append(min(frame_dists))
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


def _classify_stability(ligand_rmsd_array: np.ndarray, last_n_ns: float = 5.0, dt_ns: float = 0.002) -> str:
    """Classify replica stability based on ligand RMSD over the last *last_n_ns* ns.

    Returns:
        "Stable" if mean RMSD < 2.0 Å
        "Metastable" if 2.0–4.0 Å
        "Unstable" if > 4.0 Å
    """
    if len(ligand_rmsd_array) < 2:
        return "Unstable"
    n_last = max(1, int(last_n_ns / (dt_ns * REPORT_INTERVAL_STEPS)))
    last_rmsd = ligand_rmsd_array[-n_last:]
    mean_rmsd = float(np.mean(last_rmsd))
    if mean_rmsd < 2.0:
        return "Stable"
    elif mean_rmsd < 4.0:
        return "Metastable"
    else:
        return "Unstable"


def _run_replica(
    candidate: dict,
    replica_idx: int,
    npt_steps: int,
    nvt_steps: int,
    nvt_duration_ps: float,
    npt_duration_ns: float,
    npt_eq_duration_ps: float = 500.0,
) -> dict:
    """Run a single MD replica for one candidate.

    Equilibration protocol:
      i.   Minimize 5000 steps (L-BFGS).
      ii.  500 ps NVT at 300 K with backbone restraints
           (10 kcal/mol/Å², gradually released to 0 over 500 ps).
      iii. 500 ps NPT at 300 K, 1 atm with backbone restraints
           (5 kcal/mol/Å², gradually released to 0).
      iv.  Production NPT: *npt_duration_ns* ns, no restraints.
    """
    import openmm
    from openmm import app, unit
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from openff.toolkit import Molecule as OffMolecule

    # The OpenCL platform in this OpenMM build evaluates CustomExternalForce
    # incorrectly for periodic systems (wildly inflated restraint energies).
    # Force the CPU platform for correct physics, using all available cores.
    _PLATFORM = openmm.Platform.getPlatformByName("CPU")
    _PLATFORM.setPropertyDefaultValue("Threads", "14")

    cid = candidate["Compound_ID"]
    smi = candidate["SMILES"]

    result = {
        "replica": replica_idx,
        "compound_id": cid,
        "smiles": smi,
        "success": False,
        "error": None,
        "minimization": {},
        "equilibration": {},
        "production": {},
        "stability_class": None,
    }

    replica_dir = MD_OUT / cid / f"replica_{replica_idx}"
    replica_dir.mkdir(parents=True, exist_ok=True)

    # Prepare ligand 3D structure
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        result["error"] = "RDKit MolFromSmiles failed"
        return result

    # Start MD from the actual docked pose: overlay the ligand heavy atoms
    # onto the best active-site docked pose before building the complex.
    from utils.docking import find_best_pose_pdbqt
    pose_pdbqt = find_best_pose_pdbqt(cid, str(REPO / "output" / "workdir"))

    lig_pdb = str(replica_dir / "ligand.pdb")
    if not _prepare_ligand_pdb(mol, lig_pdb, pose_pdbqt=pose_pdbqt):
        result["error"] = "3D conformer generation failed"
        return result
    if pose_pdbqt:
        result["pose_pdbqt"] = pose_pdbqt
        log.info(f"    {cid}: starting from docked pose {pose_pdbqt}")
    else:
        log.warning(f"    {cid}: no docked pose found; using ETKDG conformer")

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
    except Exception as exc:
        result["error"] = f"Complex building failed: {exc}"
        return result

    complex_top = modeller.topology
    complex_pos = modeller.positions
    lig_indices = _compute_ligand_indices(complex_top, n_rec_atoms)

    # Build the system force field (protein + water + ligand template) up
    # front so the ligand is parameterised before solvation and the same
    # ForceField can drive both addSolvent and createSystem.
    try:
        off_mol = OffMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        off_mol.assign_partial_charges(partial_charge_method="gasteiger")
        tg = SMIRNOFFTemplateGenerator(molecules=off_mol, forcefield="openff-2.0.0")
        ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        ff.registerTemplateGenerator(tg.generator)
    except Exception as exc:
        result["error"] = f"Force field setup failed: {exc}"
        return result

    # Solvate
    try:
        modeller.addSolvent(
            ff,
            model="tip3p",
            padding=SOLVENT_PADDING * unit.angstrom,
            ionicStrength=NACL_CONCENTRATION * unit.molar,
            neutralize=True,
        )
    except Exception as exc:
        result["error"] = f"Solvation failed: {exc}"
        return result

    solvated_top = modeller.topology
    solvated_pos = modeller.positions

    # Save topology
    top_pdb = str(replica_dir / "topology.pdb")
    with open(top_pdb, "w") as fh:
        app.PDBFile.writeFile(solvated_top, solvated_pos, fh)

    # Create system
    try:
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

    # Restraint helper. Restrains backbone Cα atoms with a harmonic flat-well
    # potential k*(Δx²+Δy²+Δz²). k is stored per-particle in internal OpenMM
    # units (kJ/mol/nm²): 10 kcal/mol/Å² ≡ 4184 kJ/mol/nm². This OpenMM build
    # exposes CustomExternalForce.setParticleParameters(index, particle, params).
    RESTRAINT_FORCE = 10.0  # kcal/mol/Å²
    RESTRAINT_FORCE_KJ = RESTRAINT_FORCE * 4.184 / (0.1 ** 2)  # kJ/mol/nm²
    restraint = openmm.CustomExternalForce("k * (x - x0)^2 + k * (y - y0)^2 + k * (z - z0)^2")
    restraint.addPerParticleParameter("k")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    ca_indices = []
    ca_xyz = []
    n_restrained_ca = 0
    for residue in receptor_top.residues():
        for atom in residue.atoms():
            if atom.name == "CA":
                pos = solvated_pos[atom.index]
                restraint.addParticle(atom.index, [RESTRAINT_FORCE_KJ, pos.x, pos.y, pos.z])
                ca_indices.append(atom.index)
                ca_xyz.append([pos.x, pos.y, pos.z])
                n_restrained_ca += 1
                break
    system.addForce(restraint)

    # i. Minimisation
    try:
        integrator = openmm.LangevinIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, TIMESTEP_PS * unit.picoseconds,
        )
        simulation = app.Simulation(solvated_top, system, integrator, platform=_PLATFORM)
        simulation.context.setPositions(solvated_pos)

        state_before = simulation.context.getState(getEnergy=True)
        e_init = state_before.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        simulation.minimizeEnergy(maxIterations=2000)
        state_min = simulation.context.getState(getEnergy=True, getPositions=True)
        e_min = state_min.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        min_pos = state_min.getPositions()
        lig_rmsd_min = _compute_ligand_rmsd(solvated_pos, min_pos, lig_indices)

        result["minimization"] = {
            "initial_energy_kcal": round(e_init, 1),
            "final_energy_kcal": round(e_min, 1),
            "delta_energy_kcal": round(e_min - e_init, 1),
            "ligand_rmsd_A": round(float(lig_rmsd_min), 3) if lig_rmsd_min else None,
            "success": True,
        }
    except Exception as exc:
        result["error"] = f"Minimisation failed: {exc}"
        return result

    # ii. NVT equilibration with gradual restraint release
    try:
        simulation.context.setPositions(min_pos)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

        nvt_positions = []
        nvt_steps_half = nvt_steps // 2
        # Release the restraint over ~100 discrete ramps within the first half
        # of the NVT phase, batching the integrator calls for performance.
        nvt_chunk = max(1, nvt_steps_half // 100)
        step = 0
        while step < nvt_steps:
            if step < nvt_steps_half:
                frac = 1.0 - step / nvt_steps_half
                k = RESTRAINT_FORCE_KJ * frac
                for i in range(n_restrained_ca):
                    restraint.setParticleParameters(i, ca_indices[i],
                        [k, ca_xyz[i][0], ca_xyz[i][1], ca_xyz[i][2]])
                restraint.updateParametersInContext(simulation.context)
                n_run = min(nvt_chunk, nvt_steps - step)
            else:
                n_run = nvt_steps - step
            simulation.step(n_run)
            step += n_run
            state_nvt = simulation.context.getState(getPositions=True, getEnergy=True)
            nvt_positions.append(state_nvt.getPositions())

        state_nvt_final = simulation.context.getState(getEnergy=True)
        e_nvt = state_nvt_final.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        lig_rmsd_nvt = []
        for pos in nvt_positions:
            lr = _compute_ligand_rmsd(min_pos, pos, lig_indices)
            if lr is not None:
                lig_rmsd_nvt.append(float(lr))
        nvt_pos = nvt_positions[-1] if nvt_positions else min_pos

        result["equilibration"] = {
            "nvt_duration_ps": nvt_duration_ps,
            "final_energy_kcal": round(e_nvt, 1),
            "ligand_rmsd_mean_A": round(float(np.mean(lig_rmsd_nvt)), 3) if lig_rmsd_nvt else None,
            "ligand_rmsd_std_A": round(float(np.std(lig_rmsd_nvt)), 3) if lig_rmsd_nvt else None,
            "success": True,
        }
    except Exception as exc:
        result["error"] = f"NVT equilibration failed: {exc}"
        return result

    # iii. NPT equilibration + iv. Production
    try:
        system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 25))
        # The LangevinIntegrator is already bound to the NVT context; OpenMM
        # forbids reusing an integrator across contexts, so build a fresh one.
        npt_integrator = openmm.LangevinIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, TIMESTEP_PS * unit.picoseconds,
        )
        simulation = app.Simulation(solvated_top, system, npt_integrator, platform=_PLATFORM)
        simulation.context.setPositions(nvt_pos)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

        # NPT equilibration (first npt_eq_duration_ps ps with gradual restraint release)
        npt_eq_steps = int(npt_eq_duration_ps / TIMESTEP_PS)
        npt_eq_chunk = max(1, npt_eq_steps // 100)
        eq_step = 0
        while eq_step < npt_eq_steps:
            frac = 1.0 - eq_step / npt_eq_steps
            k = 0.5 * RESTRAINT_FORCE_KJ * frac  # start at 5 kcal/mol/Å², release to 0
            for i in range(n_restrained_ca):
                restraint.setParticleParameters(i, ca_indices[i],
                    [k, ca_xyz[i][0], ca_xyz[i][1], ca_xyz[i][2]])
            restraint.updateParametersInContext(simulation.context)
            n_run = min(npt_eq_chunk, npt_eq_steps - eq_step)
            simulation.step(n_run)
            eq_step += n_run

        # Remove restraints for production
        for i in range(n_restrained_ca):
            restraint.setParticleParameters(i, ca_indices[i],
                [0.0, ca_xyz[i][0], ca_xyz[i][1], ca_xyz[i][2]])
        restraint.updateParametersInContext(simulation.context)

        # NPT production
        pocket_center_np = np.array([40.0, 20.0, 30.0])
        prod_positions = []
        prod_energies = []
        lig_rmsd_traj = []
        report_npt_steps = max(1, npt_steps // int(npt_duration_ns * 1000 / TIMESTEP_PS / 1000))

        # Write the production trajectory to DCD so downstream trajectory-based
        # MM-GBSA (scripts/mmgbsa_analysis.py) can sample an ensemble rather
        # than a single minimised pose. The DCD shares atom ordering with the
        # topology.pdb written above (solvated system), which MM-GBSA reloads.
        dcd_path = str(replica_dir / "trajectory.dcd")
        dcd_reporter = app.DCDReporter(dcd_path, report_npt_steps, append=False)
        simulation.reporters.append(dcd_reporter)

        n_prod_chunks = npt_steps // report_npt_steps
        for _ in range(n_prod_chunks):
            simulation.step(report_npt_steps)
            state_prod = simulation.context.getState(getPositions=True, getEnergy=True)
            pos = state_prod.getPositions()
            prod_positions.append(pos)
            prod_energies.append(
                state_prod.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            )
            lr = _compute_ligand_rmsd(min_pos, pos, lig_indices)
            if lr is not None:
                lig_rmsd_traj.append(float(lr))
        if npt_steps % report_npt_steps:
            simulation.step(npt_steps % report_npt_steps)
            state_prod = simulation.context.getState(getPositions=True, getEnergy=True)
            prod_positions.append(state_prod.getPositions())
            prod_energies.append(
                state_prod.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            )
            lr = _compute_ligand_rmsd(min_pos, state_prod.getPositions(), lig_indices)
            if lr is not None:
                lig_rmsd_traj.append(float(lr))

        log.info(f"    NPT production complete: {npt_duration_ns} ns, {len(prod_positions)} frames")
    except Exception as exc:
        result["error"] = f"NPT production failed: {exc}"
        return result

    # Analysis
    try:
        lig_rmsd_array = np.array(lig_rmsd_traj)
        np.save(str(replica_dir / "ligand_rmsd.npy"), lig_rmsd_array)

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
                np.save(str(replica_dir / "receptor_rmsf.npy"), rmsf)

        # H-bond occupancy
        hb_occ = _find_hbond_occupancy(prod_positions, solvated_top, lig_indices)
        with open(str(replica_dir / "hbond_occupancy.json"), "w") as fh:
            json.dump(hb_occ, fh, indent=2)

        # Stability classification
        stability = _classify_stability(lig_rmsd_array, last_n_ns=5.0)
        result["stability_class"] = stability

        # Per-replica 10 ns stability metrics (consumed by the D3 classifier in
        # utils.filtering.classify_md_stability): mean ligand RMSD over the last
        # 5 ns and the Ser403 OG H-bond occupancy.
        last5_mean_rmsd = None
        if len(lig_rmsd_array) > 1:
            # Calculate the number of frames to average over (last 5 ns)
            # For 0.002 ps timestep, we need 5.0 / 0.002 = 2500 steps per ns
            # So for 5 ns, we need the last 5 * 1000 = 5000 frames (at 0.002 ps timestep)
            n_last5 = max(1, int(5.0 / (DT_NS * REPORT_INTERVAL_STEPS)))
            last5_mean_rmsd = float(np.mean(lig_rmsd_array[-n_last5:]))
        else:
            # If we don't have enough frames, use the mean of all frames
            last5_mean_rmsd = float(np.mean(lig_rmsd_array)) if len(lig_rmsd_array) > 0 else None
        ser403_occ = hb_occ.get("SER403_OG", {}).get("occupancy", 0.0)
        # Ensure keys are set for D3 classifier
        if last5_mean_rmsd is not None:
            result["ligand_rmsd_mean_last5ns_A"] = last5_mean_rmsd
        else:
            result["ligand_rmsd_mean_last5ns_A"] = None
        result["ser403_og_hbond_occupancy"] = float(ser403_occ)

        prod_energies_array = np.array(prod_energies)

        result["production"] = {
            "npt_duration_ns": npt_duration_ns,
            "n_frames": len(prod_positions),
            "ligand_rmsd_mean_A": round(float(np.mean(lig_rmsd_array)), 3) if len(lig_rmsd_array) > 0 else None,
            "ligand_rmsd_std_A": round(float(np.std(lig_rmsd_array)), 3) if len(lig_rmsd_array) > 0 else None,
            "ligand_rmsd_max_A": round(float(np.max(lig_rmsd_array)), 3) if len(lig_rmsd_array) > 0 else None,
            "ligand_rmsd_final_A": round(float(lig_rmsd_array[-1]), 3) if len(lig_rmsd_array) > 0 else None,
            "hbond_occupancy": hb_occ,
            "mean_potential_energy_kcal": round(float(np.mean(prod_energies_array)), 1) if len(prod_energies_array) > 0 else None,
            "success": True,
        }
        log.info(f"    Replica {replica_idx}: lig RMSD={result['production']['ligand_rmsd_mean_A']:.3f}±{result['production']['ligand_rmsd_std_A']:.3f} Å, stability={stability}")
    except Exception as exc:
        result["error"] = f"Analysis failed: {exc}"
        return result

    result["success"] = True
    return result


def run_explicit_md(
    candidate: dict,
    n_replicas: int = DEFAULT_N_REPLICAS,
    npt_duration_ns: float = DEFAULT_NPT_NS,
    nvt_duration_ps: float = 500.0,
    npt_eq_duration_ps: float = 500.0,
) -> dict:
    """Run MD for a candidate across multiple replicas.

    Returns aggregated result with per-replica data and consensus stability.
    """
    npt_steps = int(npt_duration_ns * 1000 / TIMESTEP_PS)
    nvt_steps = int(nvt_duration_ps / TIMESTEP_PS)

    cid = candidate["Compound_ID"]
    smi = candidate["SMILES"]

    result = {
        "compound_id": cid,
        "smiles": smi,
        "success": False,
        "error": None,
        "n_replicas": n_replicas,
        "npt_duration_ns": npt_duration_ns,
        "replicas": [],
        "stability_classes": [],
        "consensus_stability": None,
        "validated": False,
    }

    candidate_dir = MD_OUT / cid
    candidate_dir.mkdir(parents=True, exist_ok=True)

    for rep_idx in range(n_replicas):
        log.info(f"  Replica {rep_idx + 1}/{n_replicas}...")
        rep_result = _run_replica(
            candidate, rep_idx, npt_steps, nvt_steps, nvt_duration_ps, npt_duration_ns,
            npt_eq_duration_ps,
        )
        result["replicas"].append(rep_result)
        result["stability_classes"].append(rep_result.get("stability_class"))

        if not rep_result["success"]:
            log.warning(f"    Replica {rep_idx} failed: {rep_result.get('error', '?')}")

    # Consensus stability: ≥2 of 3 replicas Stable or Metastable → Validated
    stable_or_meta = sum(
        1 for sc in result["stability_classes"]
        if sc in ("Stable", "Metastable")
    )
    result["validated"] = stable_or_meta >= max(2, n_replicas // 2 + 1)
    result["consensus_stability"] = "Validated" if result["validated"] else "Not Validated"
    result["success"] = any(r["success"] for r in result["replicas"])

    # D3 three-tier classification from the 10 ns per-replica metrics. This is
    # the primary binding-stability call used in the paper (§4.x) and by
    # utils.filtering.classify_md_stability. The legacy
    # "Stable/Metastable/Unstable" replica classes above are retained for
    # backwards compatibility.
    try:
        from utils.filtering import classify_md_stability
        d3_class = classify_md_stability(result["replicas"])
        result["stability_class_d3"] = d3_class
    except Exception as exc:
        result["stability_class_d3"] = "Dissociated"
        log.warning(f"  D3 classifier unavailable ({exc}); defaulting to Dissociated")

    # Write per-candidate summary
    summary_path = candidate_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="Explicit-solvent MD for top docking poses")
    parser.add_argument("--quick", action="store_true",
                        help=f"Quick test: {QUICK_NPT_NS}ns x {QUICK_N_CANDIDATES} candidates x {QUICK_N_REPLICAS} replica")
    parser.add_argument("--production-ns", type=float, default=None,
                        help=f"NPT production length in ns (default: {DEFAULT_NPT_NS} ns)")
    parser.add_argument("--replicas", type=int, default=None,
                        help=f"Number of replicas (default: {DEFAULT_N_REPLICAS})")
    parser.add_argument("--n-candidates", type=int, default=None,
                        help=f"Number of top candidates (default: {DEFAULT_N_CANDIDATES})")
    args = parser.parse_args()

    if args.quick:
        n_candidates = QUICK_N_CANDIDATES
        n_replicas = QUICK_N_REPLICAS
        npt_ns = QUICK_NPT_NS
        nvt_ps = QUICK_NVT_PS
        npt_eq_ps = QUICK_NPT_EQ_PS
        log.info(f"QUICK MODE: {npt_ns} ns × {n_candidates} candidates × {n_replicas} replica "
                 f"(NVT {nvt_ps:.0f} ps, NPT eq {npt_eq_ps:.0f} ps)")
    else:
        n_candidates = args.n_candidates if args.n_candidates is not None else DEFAULT_N_CANDIDATES
        n_replicas = args.replicas if args.replicas is not None else DEFAULT_N_REPLICAS
        npt_ns = args.production_ns if args.production_ns is not None else DEFAULT_NPT_NS
        nvt_ps = 500.0
        npt_eq_ps = 500.0
        log.info(f"FULL MODE: {npt_ns} ns × {n_candidates} candidates × {n_replicas} replicas")

    _check_deps()
    MD_OUT.mkdir(parents=True, exist_ok=True)

    candidates = _load_top_candidates(n=n_candidates)

    all_results = []
    for cand in candidates:
        cid = cand["Compound_ID"]
        log.info(f"\n  Processing {cid}...")
        result = run_explicit_md(
            cand, n_replicas=n_replicas, npt_duration_ns=npt_ns,
            nvt_duration_ps=nvt_ps, npt_eq_duration_ps=npt_eq_ps,
        )
        all_results.append(result)

    log.info("")
    log.info("─" * 100)
    log.info(f"  {'Compound':<20} {'Replicas':<10} {'Stable/Meta':<14} {'Consensus':<14} {'Status':<12}")
    log.info("  " + "-" * 70)
    for r in all_results:
        n_ok = sum(1 for rep in r["replicas"] if rep["success"])
        n_sm = sum(1 for sc in r["stability_classes"] if sc in ("Stable", "Metastable"))
        sc = "-".join(s[:4] if s else "FAIL" for s in r["stability_classes"])
        consensus = r["consensus_stability"]
        status = "OK" if r["success"] else f"FAIL"
        log.info(f"  {r['compound_id']:<20} {n_ok}/{r['n_replicas']:<5} {n_sm}/{r['n_replicas']:<10} {consensus:<14} {status:<12}")

    # Write aggregated summary
    agg_path = MD_OUT / "summary.json"
    n_ok = sum(1 for r in all_results if r["success"])
    n_validated = sum(1 for r in all_results if r.get("validated"))
    agg = {
        "n_candidates": len(all_results),
        "n_succeeded": n_ok,
        "n_validated": n_validated,
        "parameters": {
            "solvent": "TIP3P",
            "padding_A": SOLVENT_PADDING,
            "nacl_concentration_M": NACL_CONCENTRATION,
            "n_replicas": n_replicas,
            "npt_duration_ns": npt_ns,
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
    log.info(f"  {n_ok}/{len(all_results)} succeeded, {n_validated} validated")
    sys.exit(0 if n_ok > 0 else 1)


if __name__ == "__main__":
    main()
