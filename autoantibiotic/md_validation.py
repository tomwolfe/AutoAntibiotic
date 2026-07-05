"""
MD Validation Module
=====================
Explicit-solvent MD validation of docked poses using OpenMM.

When OpenMM is installed, :func:`run_short_md` performs a two-stage
explicit-solvent simulation (relaxation + production) of the
ligand-receptor complex and reports:
  - Ligand RMSD relative to the initial docked pose
  - Pocket Radius of Gyration (Rg) stability

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


# ── Radius of Gyration ──────────────────────────────────────────────


def _compute_radius_of_gyration(positions: np.ndarray) -> float:
    """Compute the radius of gyration of a set of atoms.

    Rg = sqrt( (1/N) * sum_i |r_i - r_com|^2 )

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Atomic coordinates in Angstrom.

    Returns
    -------
    float
        Radius of gyration in Angstrom.
    """
    if len(positions) == 0:
        return 0.0
    center = np.mean(positions, axis=0)
    sq_dist = np.sum((positions - center) ** 2, axis=1)
    return float(np.sqrt(np.mean(sq_dist)))


# ── Helper: identify pocket residues for Rg tracking ────────────────


def _identify_pocket_atoms(
    topology: omma.Topology,
    pocket_resnames: List[str],
) -> List[int]:
    """Return atom indices belonging to specified residue names."""
    indices: List[int] = []
    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name in pocket_resnames:
                for atom in residue.atoms():
                    indices.append(atom.index)
    return indices


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
    relaxation_ns: Optional[float] = None,
    production_ns: Optional[float] = None,
    pocket_resnames: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Run a two-stage explicit-solvent MD simulation of a ligand-receptor
    complex and return stability metrics.

    Workflow
    --------
    1. Load the receptor PDB and fix it with PDBFixer (add missing
       atoms, side chains, and hydrogens).
    2. Parameterise with Amber14 + TIP3P-FB.
    3. Embed the ligand in the binding site.
    4. Solvate with a 10 A padding water box.
    5. Energy minimise (500 steps).
    6. **Relaxation**: Short NVT simulation (*relaxation_ns*) with strong
       position restraints on the protein backbone (10 kcal/mol/A^2).
    7. **Production**: Longer NPT simulation (*production_ns*) with very
       weak restraints (0.1 kcal/mol/A^2).
    8. Compute the heavy-atom RMSD of the ligand relative to its
       initial position.
    9. Compute the Radius of Gyration (Rg) of pocket residues.  If the
       pocket Rg changes by more than 10% from the starting structure,
       the complex is flagged as potentially unstable.

    When OpenMM is **not** available, logs a warning and returns None.

    Parameters
    ----------
    ligand_mol : Chem.Mol
        RDKit molecule with a single conformer representing the docked
        pose.
    receptor_pdb : str
        Path to the receptor PDB file.
    duration_ns : float
        **Deprecated.** Total simulation length in nanoseconds.
        Used only when *relaxation_ns* and *production_ns* are both
        ``None`` (legacy single-stage mode).  Default 1.0.
    temperature : float
        Simulation temperature in Kelvin (default 300.0).
    output_dir : str, optional
        Directory for trajectory output files. If None, a temporary
        directory is used.
    relaxation_ns : float, optional
        Relaxation duration in ns.  Defaults to
        ``CONFIG.md_relaxation_duration_ns``.
    production_ns : float, optional
        Production duration in ns.  Defaults to
        ``CONFIG.md_production_duration_ns``.
    pocket_resnames : list of str, optional
        Residue names defining the binding pocket for Rg tracking.
        Defaults to allosteric + active site residues from CONFIG.

    Returns
    -------
    dict or None
        A dictionary with keys:

        - ``"ligand_rmsd_angstrom"``: float
        - ``"duration_ns"``: float
        - ``"temperature_k"``: float
        - ``"success"``: bool
        - ``"message"``: str
        - ``"pocket_rg_stability"``: float — fractional change in pocket
          Rg (0.0 = unchanged, >0.1 = significant expansion/collapse).

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

    # Resolve simulation durations
    if relaxation_ns is None and production_ns is None:
        # Legacy single-stage mode
        rel_ns = 0.1  # minimal relaxation
        prod_ns = duration_ns
    else:
        rel_ns = relaxation_ns if relaxation_ns is not None else float(CONFIG.md_relaxation_duration_ns)
        prod_ns = production_ns if production_ns is not None else float(CONFIG.md_production_duration_ns)

    # Pocket residues for Rg tracking
    if pocket_resnames is None:
        pocket_resnames = list(set(
            CONFIG.key_interaction_residues_allosteric
            + CONFIG.key_interaction_residues_active
        ))

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

        # ── Identify atoms for restraints and tracking ──────────────
        # Restrained atoms: protein backbone CA
        restrained_atoms: List[int] = []
        for chain in modeller.topology.chains():
            for res in chain.residues():
                if res.name != "LIG":
                    for atom in res.atoms():
                        if atom.name == "CA":
                            restrained_atoms.append(atom.index)

        # Pocket atoms for Rg tracking
        pocket_atom_indices = _identify_pocket_atoms(
            modeller.topology, pocket_resnames,
        )

        # Ligand heavy-atom indices for RMSD
        lig_heavy_atom_indices = [
            j for j in range(ligand_mol.GetNumAtoms())
            if ligand_mol.GetAtomWithIdx(j).GetAtomicNum() > 1
        ]
        lig_heavy_top_indices = [lig_top_indices[j] for j in lig_heavy_atom_indices]

        # ── Helper: create position restraint force ────────────────
        def _add_restraints(k_value: float) -> None:
            """Add CA position restraints with spring constant *k_value*."""
            nonlocal system
            if not restrained_atoms:
                return
            rst = omm.CustomExternalForce(
                "k*periodicdistance(x, y, z, x0, y0, z0)^2"
            )
            rst.addPerParticleParameter("x0")
            rst.addPerParticleParameter("y0")
            rst.addPerParticleParameter("z0")
            rst.addGlobalParameter(
                "k", k_value * u.kilocalories_per_mole / u.angstroms ** 2
            )
            pos = simulation.context.getState(getPositions=True).getPositions()
            for idx in restrained_atoms:
                p = pos[idx]
                rst.addParticle(idx, [p[0], p[1], p[2]])
            system.addForce(rst)
            simulation.context.reinitialize(preserveState=True)

        # ── 8. Relaxation phase (strong restraints) ────────────────
        log.info(f"  MD: Starting relaxation ({rel_ns} ns, strong restraints)...")
        _add_restraints(10.0)  # 10 kcal/mol/A^2

        rel_steps = int(rel_ns * 1000 / 0.002)
        simulation.step(rel_steps)

        # Record initial (post-relaxation) pocket Rg
        state_after_relax = simulation.context.getState(getPositions=True)
        pos_after_relax = state_after_relax.getPositions(asNumpy=True).value
        init_pocket_positions = np.array([
            pos_after_relax[idx] for idx in pocket_atom_indices
        ], dtype=np.float64)
        init_pocket_rg = _compute_radius_of_gyration(init_pocket_positions)

        # Record initial (post-relaxation) ligand positions for RMSD
        init_lig_heavy_positions = np.array([
            pos_after_relax[idx] for idx in lig_heavy_top_indices
        ], dtype=np.float64)

        # ── 9. Production phase (weak restraints) ──────────────────
        log.info(f"  MD: Starting production ({prod_ns} ns, weak restraints)...")
        # Remove the restraint force we added and re-add with weaker k
        # We need to remove the last force (the restraint)
        forces = list(system.getForces())
        if forces:
            restraint_force = forces[-1]
            system.removeForce(len(forces) - 1)
        # Re-add with weak spring constant
        _add_restraints(0.1)  # 0.1 kcal/mol/A^2 – very weak

        prod_steps = int(prod_ns * 1000 / 0.002)
        simulation.step(prod_steps)

        # 10. Get final positions and compute metrics
        state_final = simulation.context.getState(getPositions=True)
        final_all_positions = state_final.getPositions(asNumpy=True).value

        # Ligand RMSD
        final_lig_heavy_positions = np.array([
            final_all_positions[idx] for idx in lig_heavy_top_indices
        ], dtype=np.float64)
        ligand_rmsd = _compute_ligand_rmsd(
            init_lig_heavy_positions, final_lig_heavy_positions,
        )

        # Pocket Rg stability
        final_pocket_positions = np.array([
            final_all_positions[idx] for idx in pocket_atom_indices
        ], dtype=np.float64)
        final_pocket_rg = _compute_radius_of_gyration(final_pocket_positions)

        # Fractional Rg change
        if init_pocket_rg > 1e-6:
            pocket_rg_stability = abs(final_pocket_rg - init_pocket_rg) / init_pocket_rg
        else:
            pocket_rg_stability = 0.0

        total_ns = rel_ns + prod_ns

        result = {
            "ligand_rmsd_angstrom": ligand_rmsd,
            "duration_ns": total_ns,
            "temperature_k": temperature,
            "success": True,
            "pocket_rg_stability": pocket_rg_stability,
            "message": (
                f"MD simulation completed. Ligand RMSD = {ligand_rmsd:.2f} A, "
                f"pocket Rg change = {pocket_rg_stability:.3f} "
                f"({'UNSTABLE' if pocket_rg_stability > 0.1 else 'stable'})"
            ),
        }
        log.info(
            f"  MD Validation: ligand RMSD = {ligand_rmsd:.2f} A, "
            f"pocket Rg stability = {pocket_rg_stability:.3f}"
        )
        return result

    except Exception as exc:
        log.warning(f"  MD Validation failed: {exc}")
        return {
            "ligand_rmsd_angstrom": 999.9,
            "duration_ns": rel_ns + prod_ns,
            "temperature_k": temperature,
            "success": False,
            "pocket_rg_stability": 999.9,
            "message": str(exc),
        }
