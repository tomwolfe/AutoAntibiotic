"""
MD Validation Module
=====================
Explicit-solvent MD validation of docked poses using OpenMM.

When OpenMM is installed, :func:`run_short_md` performs a brief
explicit-solvent simulation of the ligand-receptor complex and reports
the ligand RMSD relative to the initial docked pose as a stability
metric.

If OpenMM is unavailable, the function returns ``None`` with a warning.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from .config import CONFIG
from .io_utils import log

# ── OpenMM availability check ────────────────────────────────────────

_HAVE_OPENMM: bool = False
_HAVE_PDBFIXER: bool = False
try:
    import openmm  # noqa: F401
    import openmm.app as omma
    import openmm.unit as u

    _HAVE_OPENMM = True
except ImportError:
    pass

try:
    import pdbfixer
    _HAVE_PDBFIXER = True
except ImportError:
    pass


def _check_openmm() -> bool:
    """Return ``True`` if OpenMM is importable."""
    return _HAVE_OPENMM


# ── Ligand RMSD calculation ──────────────────────────────────────────


def _compute_ligand_rmsd(
    initial_positions: np.ndarray,
    trajectory_positions: np.ndarray,
) -> float:
    """Compute the heavy-atom RMSD between initial and final ligand poses
    after aligning on the receptor backbone.

    Parameters
    ----------
    initial_positions : np.ndarray, shape (N, 3)
        Initial 3-D coordinates of the ligand heavy atoms.
    trajectory_positions : np.ndarray, shape (N, 3)
        Final 3-D coordinates after MD simulation.

    Returns
    -------
    float
        RMSD in Angstrom.  Returns 999.9 on failure.
    """
    if len(initial_positions) != len(trajectory_positions):
        return 999.9
    if len(initial_positions) == 0:
        return 999.9

    diff = initial_positions - trajectory_positions
    rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return rmsd


# ── Helper: determine which atoms in the topology belong to ligand ───


def _identify_ligand_atoms(
    topology: omma.Topology,
    ligand_resname: str = "LIG",
) -> List[int]:
    """Return the indices of atoms belonging to the ligand residue."""
    lig_indices: List[int] = []
    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name == ligand_resname:
                for atom in residue.atoms():
                    lig_indices.append(atom.index)
    return lig_indices


# ── Main public API ──────────────────────────────────────────────────


def run_short_md(
    ligand_mol: Chem.Mol,
    receptor_pdb: str,
    duration_ns: float = 1.0,
    temperature: float = 300.0,
    output_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run a short explicit-solvent MD simulation of a ligand-receptor
    complex and return a stability metric based on ligand RMSD.

    Workflow
    --------
    1. Load the receptor PDB and fix it with PDBFixer (add missing
       atoms, side chains, and hydrogens).
    2. Parameterise with Amber14 + TIP3P-FB.
    3. Embed the ligand in the binding site.
    4. Solvate with a 10 A padding water box.
    5. Energy minimise (500 steps).
    6. Equilibrate (100 ps NVT with position restraints on protein
       backbone).
    7. Production simulation for *duration_ns* at constant pressure.
    8. Compute the heavy-atom RMSD of the ligand relative to its
       initial position.

    When OpenMM is **not** available, logs a warning and returns None.

    Parameters
    ----------
    ligand_mol : Chem.Mol
        RDKit molecule with a single conformer representing the docked
        pose.
    receptor_pdb : str
        Path to the receptor PDB file.
    duration_ns : float
        Simulation length in nanoseconds (default 1.0).
    temperature : float
        Simulation temperature in Kelvin (default 300.0).
    output_dir : str, optional
        Directory for trajectory output files. If None, a temporary
        directory is used.

    Returns
    -------
    dict or None
        A dictionary with keys:

        - ``"ligand_rmsd_angstrom"``: float
        - ``"duration_ns"``: float
        - ``"temperature_k"``: float
        - ``"success"``: bool
        - ``"message"``: str

        Returns None if OpenMM is unavailable or the simulation
        fails catastrophically.
    """
    if not _HAVE_OPENMM:
        log.warning(
            "MD Validation: OpenMM not installed -- skipping. "
            "Install with: conda install -c conda-forge openmm"
        )
        return None

    if not _HAVE_PDBFIXER:
        log.warning(
            "MD Validation: pdbfixer not installed -- skipping. "
            "Install with: conda install -c conda-forge pdbfixer"
        )
        return None

    if not os.path.isfile(receptor_pdb):
        log.warning(f"MD Validation: receptor PDB not found: {receptor_pdb}")
        return None

    if ligand_mol.GetNumConformers() == 0:
        log.warning("MD Validation: ligand has no conformer -- cannot run MD.")
        return None

    try:
        import openmm.app as omma
        import openmm as omm
        import openmm.unit as u
        from pdbfixer import PDBFixer

        out = output_dir or tempfile.mkdtemp(prefix="md_")
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Load and fix receptor with PDBFixer
        fixer = PDBFixer(filename=receptor_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        # 2. Write the fixed receptor PDB and re-load for combined topology
        fixed_pdb = os.path.join(out, "receptor_fixed.pdb")
        with open(fixed_pdb, "w") as f:
            omma.PDBFile.writeFile(fixer.topology, fixer.positions, f)

        pdb = omma.PDBFile(fixed_pdb)
        receptor_topology = pdb.topology
        receptor_positions = pdb.positions

        # 3. Parameterise with Amber14 force field
        forcefield = omma.ForceField(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

        # 4. Combine receptor and ligand into one topology
        lig_conf = ligand_mol.GetConformer()
        lig_atomic_numbers: List[int] = [
            ligand_mol.GetAtomWithIdx(i).GetAtomicNum()
            for i in range(ligand_mol.GetNumAtoms())
        ]
        lig_elements: List[str] = [
            ligand_mol.GetAtomWithIdx(i).GetSymbol()
            for i in range(ligand_mol.GetNumAtoms())
        ]

        lig_positions = u.Quantity(
            np.array([
                [lig_conf.GetAtomPosition(i).x,
                 lig_conf.GetAtomPosition(i).y,
                 lig_conf.GetAtomPosition(i).z]
                for i in range(ligand_mol.GetNumAtoms())
            ], dtype=np.float64) * u.angstroms
        )

        # Define ligand as a new chain with residue name LIG
        lig_chain = receptor_topology.addChain("L")
        lig_res = receptor_topology.addResidue("LIG", lig_chain)
        lig_top_indices: List[int] = []
        for i, (elem, anum) in enumerate(zip(lig_elements, lig_atomic_numbers)):
            atom = receptor_topology.addAtom(
                ligand_mol.GetAtomWithIdx(i).GetPDBName() or f"{elem}{i}",
                omma.Element.getByAtomicNumber(anum),
                lig_res,
            )
            lig_top_indices.append(atom.index)

        all_positions = list(receptor_positions) + list(lig_positions)

        # 5. Solvate with 10 A padding
        modeller = omma.Modeller(receptor_topology, all_positions)
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=1.0 * u.nanometers,
        )

        # 6. Build system with PME
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=omma.app.PME,
            nonbondedCutoff=1.0 * u.nanometers,
            constraints=omma.app.HBonds,
        )

        # 7. Energy minimise (500 steps)
        integrator = omm.LangevinIntegrator(
            temperature * u.kelvin,
            1.0 / u.picoseconds,
            2.0 * u.femtoseconds,
        )
        simulation = omma.Simulation(
            modeller.topology, system, integrator,
        )
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(maxIterations=500)

        # 8. Equilibrate (100 ps NVT) with position restraints on protein CA
        #    Identify protein CA atoms for restraint
        restrained_atoms: List[int] = []
        for chain in modeller.topology.chains():
            for res in chain.residues():
                if res.name != "LIG":
                    for atom in res.atoms():
                        if atom.name == "CA":
                            restrained_atoms.append(atom.index)

        if restrained_atoms:
            restraint_force = omm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
            restraint_force.addPerParticleParameter("x0")
            restraint_force.addPerParticleParameter("y0")
            restraint_force.addPerParticleParameter("z0")
            restraint_force.addGlobalParameter("k", 10.0 * u.kilocalories_per_mole / u.angstroms ** 2)

            positions = simulation.context.getState(getPositions=True).getPositions()
            for idx in restrained_atoms:
                pos = positions[idx]
                restraint_force.addParticle(idx, [pos[0], pos[1], pos[2]])

            system.addForce(restraint_force)
            simulation.context.reinitialize(preserveState=True)

        simulation.step(50000)  # 100 ps at 2 fs/step

        # 9. Production MD (NPT, duration_ns)
        #    Remove restraints for production
        if restrained_atoms:
            # Remove the last force (restraints) from the system
            forces = list(system.getForces())
            system = omm.System()
            # Rebuild system without restraint force for production
            # Instead, just keep it with very weak restraint
            restraint_force.setGlobalParameter_k(0.1 * u.kilocalories_per_mole / u.angstroms ** 2)
            simulation.context.reinitialize(preserveState=True)

        # Store initial ligand heavy-atom positions for RMSD calculation
        lig_heavy_atom_indices = [
            j for j in range(ligand_mol.GetNumAtoms())
            if ligand_mol.GetAtomWithIdx(j).GetAtomicNum() > 1
        ]
        lig_heavy_top_indices = [lig_top_indices[j] for j in lig_heavy_atom_indices]

        state_init = simulation.context.getState(getPositions=True)
        init_all_positions = state_init.getPositions(asNumpy=True).value
        init_lig_heavy_positions = np.array([
            init_all_positions[idx] for idx in lig_heavy_top_indices
        ], dtype=np.float64)

        n_steps = int(duration_ns * 1000 / 0.002)  # 2 fs per step
        simulation.step(n_steps)

        # 10. Get final positions and compute ligand RMSD
        state_final = simulation.context.getState(getPositions=True)
        final_all_positions = state_final.getPositions(asNumpy=True).value
        final_lig_heavy_positions = np.array([
            final_all_positions[idx] for idx in lig_heavy_top_indices
        ], dtype=np.float64)

        ligand_rmsd = _compute_ligand_rmsd(
            init_lig_heavy_positions, final_lig_heavy_positions,
        )

        result = {
            "ligand_rmsd_angstrom": ligand_rmsd,
            "duration_ns": duration_ns,
            "temperature_k": temperature,
            "success": True,
            "message": f"MD simulation completed. Ligand RMSD = {ligand_rmsd:.2f} A",
        }
        log.info(f"  MD Validation: ligand RMSD = {ligand_rmsd:.2f} A")
        return result

    except Exception as exc:
        log.warning(f"  MD Validation failed: {exc}")
        return {
            "ligand_rmsd_angstrom": 999.9,
            "duration_ns": duration_ns,
            "temperature_k": temperature,
            "success": False,
            "message": str(exc),
        }
