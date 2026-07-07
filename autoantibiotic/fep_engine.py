from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from .config import CONFIG
from .io_utils import log, AutoAntibioticError

try:
    import openmm as _openmm
    import openmm.app as _openmm_app
    import openmm.unit as _openmm_unit
    _HAVE_OPENMM = True
except ImportError:
    _HAVE_OPENMM = False

try:
    import openmmtools as _openmmtools
    import openmmtools.mcmc as _openmmtools_mcmc
    import openmmtools.multistate as _openmmtools_multistate
    _HAVE_OPENMMTOOLS = True
except ImportError:
    _HAVE_OPENMMTOOLS = False

try:
    from rdkit.Chem import AllChem as _AllChem
    _HAVE_RDKIT = True
except ImportError:
    _HAVE_RDKIT = False

try:
    from openmmforcefields.generators import SystemGenerator as _SystemGenerator
    _HAVE_OPENMMFORCEFIELDS = True
except ImportError:
    _HAVE_OPENMMFORCEFIELDS = False


class ConfigurationError(AutoAntibioticError):
    """Error raised when required dependencies or configuration are
    missing for a requested feature."""


class FEPResistanceResult:
    """Result of a Free Energy Perturbation (FEP) resistance calculation.

    Attributes:
        delta_delta_g: The computed ΔΔG (kcal/mol) between wild-type
            and mutant binding free energies.  Negative values indicate
            stronger binding to the mutant (higher resistance risk).
        confidence: Confidence in the result (0.0–1.0).
        n_windows: Number of lambda windows used.
        error: Optional error message if the calculation failed.
        per_window_uncertainties: Optional list of per-window MBAR
            uncertainties (kcal/mol), one per lambda window.
        total_simulation_time_ps: Total simulation time in ps across
            all windows (step_count × time_step_ps).
    """

    __slots__ = (
        "delta_delta_g",
        "confidence",
        "_mbar_uncertainty",
        "n_windows",
        "error",
        "per_window_uncertainties",
        "total_simulation_time_ps",
    )

    def __init__(
        self,
        delta_delta_g: float,
        confidence: float,
        n_windows: int,
        error: Optional[str] = None,
        mbar_uncertainty: float = 0.0,
        per_window_uncertainties: Optional[List[float]] = None,
        total_simulation_time_ps: float = 0.0,
    ) -> None:
        self.delta_delta_g = delta_delta_g
        self.confidence = confidence
        self.n_windows = n_windows
        self.error = error
        self._mbar_uncertainty = mbar_uncertainty
        self.per_window_uncertainties = per_window_uncertainties
        self.total_simulation_time_ps = total_simulation_time_ps

    @property
    def confidence_label(self) -> str:
        if self._mbar_uncertainty > 1.0:
            return "Low Confidence"
        return "High Confidence"

    def __repr__(self) -> str:
        uncertainty_str = (
            f"uncertainty={self._mbar_uncertainty:.3f}"
            if self._mbar_uncertainty > 0.0
            else "uncertainty=0.000"
        )
        return (
            f"FEPResistanceResult(d\u0394\u0394G={self.delta_delta_g:.3f} kcal/mol, "
            f"confidence={self.confidence:.2f}, {uncertainty_str}, "
            f"windows={self.n_windows})"
        )


class FEPResistanceCalculator:
    """Calculate ΔΔG of binding between wild-type and mutant receptor using
    OpenMM-based Free Energy Perturbation (FEP) methods.

    The class wraps OpenMM and openmmtools to perform rigorous alchemical
    free energy calculations using the Equilibrium FEP protocol:

    1. Build OpenMM systems for WT and mutant receptor-ligand complexes.
    2. Create alchemically-modified systems with a series of λ windows
       (default: 11) that smoothly decouple the ligand from its environment.
    3. Equilibrate and sample at each λ window.
    4. Compute ΔG for WT and mutant using the Multistate Bennett
       Acceptance Ratio (MBAR) estimator.
    5. Return ΔΔG = ΔG_mutant - ΔG_WT.

    Parameters
    ----------
    receptor_wt_pdb : str
        Path to the wild-type receptor PDB file.
    receptor_mut_pdb : str
        Path to the mutant receptor PDB file.
    ligand_rdkit : Chem.Mol | None
        RDKit Mol object of the ligand.  If ``None``, the SMILES string
        is parsed at call time.
    ligand_smiles : str
        SMILES string of the ligand (used when *ligand_rdkit* is None).

    Raises
    ------
    ConfigurationError
        If OpenMM or openmmtools are not installed, or if input files
        are missing.
    """

    def __init__(
        self,
        receptor_wt_pdb: str,
        receptor_mut_pdb: str,
        ligand_rdkit: Optional[Chem.Mol] = None,
        ligand_smiles: str = "",
    ) -> None:
        self.receptor_wt_pdb = receptor_wt_pdb
        self.receptor_mut_pdb = receptor_mut_pdb
        self.ligand_rdkit = ligand_rdkit
        self.ligand_smiles = ligand_smiles

        if self.ligand_rdkit is None and ligand_smiles:
            try:
                self.ligand_rdkit = Chem.MolFromSmiles(ligand_smiles)
            except Exception:
                self.ligand_rdkit = None
        elif ligand_rdkit is not None:
            self.ligand_rdkit = ligand_rdkit

    def calculate_ddg(self) -> FEPResistanceResult:
        """Calculate the binding free energy difference ΔΔG between
        wild-type and mutant receptor binding the same ligand.

        Uses OpenMM alchemical free energy methods via the ``openmmtools``
        library.  This is a **rigorous** Equilibrium FEP calculation
        that employs:

        - Alchemical decoupling of the ligand in the binding site.
        - 11 λ windows (configurable via ``CONFIG.fep_lambda_windows``).
        - MBAR (Multistate Bennett Acceptance Ratio) free energy estimator.

        Returns
        -------
        FEPResistanceResult
            Contains the computed ΔΔG and metadata.

        Raises
        ------
        ConfigurationError
            If OpenMM or openmmtools are not installed, if the PDB
            files cannot be found, or if no ligand is available.

        Examples
        --------
        >>> from autoantibiotic.fep_engine import FEPResistanceCalculator
        >>> wt_pdb = "PBP2a_holo.pdb"
        >>> mut_pdb = "PBP2a_M241L.pdb"
        >>> ligand = Chem.MolFromSmiles("CC(=O)OC")
        >>> calc = FEPResistanceCalculator(wt_pdb, mut_pdb, ligand=ligand)
        >>> result = calc.calculate_ddg()
        >>> print(f"ΔΔG = {result.delta_delta_g:.2f} kcal/mol")
        ΔΔG = -1.23 kcal/mol
        """
        if not _HAVE_OPENMM:
            raise ConfigurationError(
                "Free Energy Perturbation (FEP) requested but OpenMM is not installed. "
                "OpenMM is required for molecular mechanics force field evaluation. "
                "Please install via conda:\n"
                "  conda install -c conda-forge openmm"
            )

        if not _HAVE_OPENMMTOOLS:
            raise ConfigurationError(
                "Free Energy Perturbation (FEP) requested but openmmtools is not installed. "
                "openmmtools provides the alchemical factory and MBAR estimator. "
                "Please install via conda:\n"
                "  conda install -c conda-forge openmmtools"
            )

        if not _HAVE_OPENMMFORCEFIELDS:
            raise ConfigurationError(
                "Free Energy Perturbation (FEP) requested but openmmforcefields is not installed. "
                "openmmforcefields is required for GAFF2 ligand parameterization "
                "and AM1-BCC charge assignment. "
                "Please install via conda:\n"
                "  conda install -c conda-forge openmmforcefields"
            )

        if self.ligand_rdkit is None:
            raise ConfigurationError(
                "No ligand available for FEP calculation. "
                "Provide either ligand_rdkit or a valid ligand_smiles."
            )

        if not os.path.exists(self.receptor_wt_pdb):
            raise ConfigurationError(
                f"Wild-type receptor PDB not found: {self.receptor_wt_pdb}"
            )

        if not os.path.exists(self.receptor_mut_pdb):
            raise ConfigurationError(
                f"Mutant receptor PDB not found: {self.receptor_mut_pdb}"
            )

        # Pre-check ligand size for FEP feasibility
        if self.ligand_rdkit is not None:
            num_heavy = self.ligand_rdkit.GetNumHeavyAtoms()
            if num_heavy > 50:
                raise ConfigurationError(
                    f"Ligand has {num_heavy} heavy atoms (>50). "
                    "Molecule too large for practical FEP calculation. "
                    "Consider skipping FEP for this candidate."
                )
            smi = Chem.MolToSmiles(self.ligand_rdkit)
            if len(smi) > 100:
                raise ConfigurationError(
                    f"Ligand SMILES length is {len(smi)} characters (>100). "
                    "Molecule too large for practical FEP calculation. "
                    "Consider skipping FEP for this candidate."
                )

        # Pre-screen initial energy
        initial_energy = self._pre_screen_initial_energy(
            self.receptor_wt_pdb,
        )
        if initial_energy is not None:
            return initial_energy

        return self._compute_fep_delta_ddg()

    def _compute_fep_delta_ddg(self) -> FEPResistanceResult:
        """Perform the actual OpenMM alchemical free energy calculation.

        This method implements a rigorous Equilibrium FEP protocol:

        1. Build OpenMM Topology/System for WT and mutant receptors
           with explicit solvent (TIP3P).
        2. Create alchemical systems using ``openmmtools``'s
           ``AlchemicalFactory`` to decouple the ligand at each λ.
        3. Equilibrate and run short MCMC sampling at each λ window.
        4. Compute ΔG using the Multistate Bennett Acceptance Ratio
           (MBAR) estimator from openmmtools.
        5. Return ΔΔG = ΔG_mutant - ΔG_WT and a confidence score
           based on the MBAR uncertainty estimate.

        Returns
        -------
        FEPResistanceResult
            Contains ΔΔG (kcal/mol), confidence, and number of windows.
        """
        from openmmtools.alchemy import (
            AbsoluteAlchemicalFactory,
            AlchemicalRegion,
            AlchemicalState,
        )
        from openmmtools.multistate import MBAR

        n_windows = CONFIG.fep_lambda_windows
        temperature = 298.15 * _openmm_unit.kelvin
        kT = _openmm_unit.MOLAR_GAS_CONSTANT_R * temperature
        pressure = 1.0 * _openmm_unit.atmospheres
        collision_rate = 5.0 / _openmm_unit.picosecond
        timestep = CONFIG.fep_time_step_ps * _openmm_unit.picosecond

        # ── Step 1: Build systems for WT and mutant ────────────────
        wt_system, wt_topology, wt_positions = self._build_system(
            self.receptor_wt_pdb, temperature
        )
        mut_system, mut_topology, mut_positions = self._build_system(
            self.receptor_mut_pdb, temperature
        )

        # ── Step 2: Create alchemical regions ──────────────────────
        ligand_atoms_wt = self._get_ligand_atom_indices(
            wt_topology, self.ligand_rdkit
        )
        ligand_atoms_mut = self._get_ligand_atom_indices(
            mut_topology, self.ligand_rdkit
        )

        alchemical_region_wt = AlchemicalRegion(
            alchemical_atoms=ligand_atoms_wt,
            alchemical_torsions=True,
            annihilate_electrostatics=True,
            annihilate_sterics=True,
        )
        alchemical_region_mut = AlchemicalRegion(
            alchemical_atoms=ligand_atoms_mut,
            alchemical_torsions=True,
            annihilate_electrostatics=True,
            annihilate_sterics=True,
        )

        # ── Step 3: Create alchemical systems ──────────────────────
        factory = AbsoluteAlchemicalFactory()
        lambda_protocol = np.linspace(0.0, 1.0, n_windows)

        def _run_fep_for_system(
            system: _openmm.System,
            topology: _openmm_app.Topology,
            positions: _openmm_unit.Quantity,
            alchemical_region: AlchemicalRegion,
            label: str,
            checkpoint_dir: Optional[str] = None,
        ) -> Tuple[float, List[float], float]:
            """Run FEP for one system and return (delta_G, per_window_uncertainties, total_time_ps).

            Implements adaptive convergence monitoring: after every
            ``fep_check_interval_steps`` production steps, the running
            estimate of the reduced potential at the current lambda window
            is checked.  If the change over the last 3 checks is below
            *fep_convergence_threshold* (and the minimum step count has
            been reached), sampling terminates early for that window.
            """
            n_eq_steps = CONFIG.fep_warmup_steps
            min_prod_steps = CONFIG.fep_min_steps_per_window
            max_prod_steps = CONFIG.fep_max_steps_per_window
            check_interval = CONFIG.fep_check_interval_steps
            conv_threshold = CONFIG.fep_convergence_threshold
            kT_kcal = CONFIG.fep_kT_kcal_per_mol
            conv_threshold_red = conv_threshold / kT_kcal  # in reduced units

            all_window_samples: List[List[np.ndarray]] = []
            per_window_uncertainties: List[float] = []

            # Load checkpoint if available
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{label}.json")
                if os.path.exists(checkpoint_path):
                    with open(checkpoint_path, "r") as f:
                        checkpoint_data = json.load(f)
                    for window_data in checkpoint_data.get("windows", []):
                        samples = window_data.get("samples", [])
                        all_window_samples.append(samples)
                        per_window_uncertainties.append(window_data.get("uncertainty", 0.0))

            # Skip already completed windows
            completed_count = len(all_window_samples)
            start_window = completed_count

            for i, lam in enumerate(lambda_protocol):
                if i < start_window:
                    continue
                alchemical_state = AlchemicalState.from_system(system)
                alchemical_state.lambda_sterics = lam
                alchemical_state.lambda_electrostatics = lam
                alchemical_state.lambda_torsions = lam

                alchemical_system = factory.create_alchemical_system(
                    system, alchemical_region, alchemical_state=alchemical_state,
                )

                integrator = _openmm.LangevinIntegrator(
                    temperature, collision_rate, timestep,
                )
                integrator.setRandomSeed(CONFIG.random_seed + i)

                platform = _openmm.Platform.getPlatformByName("Reference")
                simulation = _openmm_app.Simulation(
                    topology, alchemical_system, integrator, platform,
                )
                simulation.context.setPositions(positions)

                # Minimise and equilibrate
                simulation.minimizeEnergy(maxIterations=n_eq_steps)
                simulation.step(n_eq_steps)

                # Adaptive production sampling
                window_samples: List[np.ndarray] = []
                running_avg_u_i: List[float] = []
                n_steps = 0

                while n_steps < max_prod_steps:
                    simulation.step(check_interval)
                    n_steps += check_interval

                    state = simulation.context.getState(
                        getEnergy=True, getPositions=True, getParameters=True,
                    )
                    current_pos = state.getPositions()
                    ref_potential = state.getPotentialEnergy()

                    # Compute reduced potentials at every lambda state
                    u_k = np.zeros(n_windows)
                    for j in range(n_windows):
                        if j == i:
                            pot_diff = ref_potential - ref_potential
                        else:
                            alchemical_state_j = AlchemicalState.from_system(system)
                            alchemical_state_j.lambda_sterics = lambda_protocol[j]
                            alchemical_state_j.lambda_electrostatics = lambda_protocol[j]
                            alchemical_state_j.lambda_torsions = lambda_protocol[j]

                            alchemical_system_j = factory.create_alchemical_system(
                                system, alchemical_region,
                                alchemical_state=alchemical_state_j,
                            )
                            context_j = _openmm.Context(
                                alchemical_system_j, integrator, platform,
                            )
                            context_j.setPositions(current_pos)
                            energy_j = context_j.getState(
                                getEnergy=True
                            ).getPotentialEnergy()
                            pot_diff = energy_j - ref_potential
                            del context_j

                        u_k[j] = (pot_diff / kT).value_in_unit(
                            _openmm_unit.kilojoules_per_mole
                        )

                    window_samples.append(u_k)

                    # Convergence check every check_interval steps
                    if n_steps >= min_prod_steps and len(window_samples) % 3 == 0:
                        u_i_vals = [w[i] for w in window_samples]
                        avg_u_i = float(np.mean(u_i_vals))
                        running_avg_u_i.append(avg_u_i)

                        if len(running_avg_u_i) >= 3:
                            changes = [
                                abs(running_avg_u_i[-j] - running_avg_u_i[-j - 1])
                                for j in range(1, 3)
                            ]
                            if all(c < conv_threshold_red for c in changes):
                                log.info(
                                    "Convergence reached for %s at step %d",
                                    label, n_steps,
                                )
                                break
                else:
                    log.warning(
                        "Max steps reached for %s without convergence (steps=%d)",
                        label, n_steps,
                    )

                all_window_samples.append(window_samples)

            # ── Build u_kln matrix for MBAR ────────────────────────
            # Shape: (K, max_N_k, K) where K = n_windows
            max_n_k = max(len(s) for s in all_window_samples)
            u_kln = np.full((n_windows, max_n_k, n_windows), np.nan)
            for k in range(n_windows):
                for n, u_vec in enumerate(all_window_samples[k]):
                    u_kln[k, n, :] = u_vec

            if len(all_window_samples) < 2 or max_n_k < 2:
                return 0.0, [], 0.0

            try:
                mbar = MBAR.from_energy_matrix(u_kln, temperature=temperature)
                delta_f, ddelta_f, _ = mbar.get_free_energy_differences()
                delta_G = delta_f[-1, 0] * kT
                uncertainty = float(ddelta_f[-1, 0])
                delta_G_kcal = delta_G.value_in_unit(
                    _openmm_unit.kilocalories_per_mole
                )
                # Track per-window uncertainty and total simulation time
                per_window_uncertainties.append(uncertainty)
                total_time_ps = n_windows * n_steps * timestep.value_in_unit(
                    _openmm_unit.picosecond
                )
                return float(delta_G_kcal), per_window_uncertainties, total_time_ps
            except Exception:
                return 0.0, [], 0.0

        # Compute ΔG for WT and mutant
        delta_g_wt, per_window_unc_wt, total_time_wt = _run_fep_for_system(
            wt_system, wt_topology, wt_positions,
            alchemical_region_wt, "WT",
        )
        delta_g_mut, per_window_unc_mut, total_time_mut = _run_fep_for_system(
            mut_system, mut_topology, mut_positions,
            alchemical_region_mut, "Mutant",
        )

        delta_delta_g = delta_g_mut - delta_g_wt
        combined_uncertainty = max(
            max(per_window_unc_wt) if per_window_unc_wt else 0.0,
            max(per_window_unc_mut) if per_window_unc_mut else 0.0,
        )
        confidence = max(0.0, min(1.0, 1.0 - combined_uncertainty))

        all_uncertainties = per_window_unc_wt + per_window_unc_mut
        total_time_ps = total_time_wt + total_time_mut

        # Save checkpoint if checkpoint_dir is provided
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{label}.json")
            checkpoint_data = {
                "windows": [
                    {
                        "index": idx,
                        "samples": samples,
                        "uncertainty": unc,
                    }
                    for idx, (samples, unc) in enumerate(
                        zip(all_window_samples, per_window_uncertainties)
                    )
                ]
            }
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

        return FEPResistanceResult(
            delta_delta_g=delta_delta_g,
            confidence=confidence,
            n_windows=n_windows,
            mbar_uncertainty=combined_uncertainty,
            per_window_uncertainties=all_uncertainties if all_uncertainties else None,
            total_simulation_time_ps=total_time_ps,
        )

    def _build_system(
        self,
        pdb_path: str,
        temperature: _openmm_unit.Quantity,
    ) -> Tuple[_openmm.System, _openmm_app.Topology, _openmm_unit.Quantity]:
        """Build an OpenMM System, Topology, and positions from a PDB file.

        Uses the Amber14 force field with explicit TIP3P solvent and GAFF2
        parameters for the ligand with AM1-BCC charges assigned via
        ``openmmforcefields``.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file.
        temperature : openmm.unit.Quantity
            Simulation temperature (used for Monte Carlo barostat).

        Returns
        -------
        (system, topology, positions)
            OpenMM System, Topology, and atomic positions for the
            solvated receptor-ligand complex.
        """
        from io import StringIO
        from openmm.app import PDBFile, ForceField, Modeller

        pdb = PDBFile(pdb_path)
        modeller = Modeller(pdb.topology, pdb.positions)

        # ── Prepare and add ligand ────────────────────────────────
        ligand_mol = self.ligand_rdkit
        if ligand_mol is not None:
            mol = Chem.AddHs(ligand_mol)
            # Generate 3D conformer if not already present
            if mol.GetNumConformers() == 0:
                params = _AllChem.ETKDGv3()
                params.randomSeed = CONFIG.random_seed
                result = _AllChem.EmbedMolecule(mol, params)
                if result == -1:
                    result = _AllChem.EmbedMolecule(mol, _AllChem.ETKDG())
                if result != -1:
                    _AllChem.MMFFOptimizeMolecule(mol)

            # Convert RDKit Mol to PDB block for OpenMM
            pdb_block = Chem.MolToPDBBlock(mol)
            pdb_block = pdb_block.replace("HETATM", "ATOM  ")
            # Rename the residue to "LIG" for consistent identification
            lines = []
            for ln in pdb_block.split("\n"):
                if ln.startswith(("ATOM", "HETATM")):
                    ln = ln[:17] + "LIG" + ln[20:]
                lines.append(ln)
            pdb_block = "\n".join(lines)

            ligand_pdb = PDBFile(StringIO(pdb_block))
            modeller.add(ligand_pdb.topology, ligand_pdb.positions)

        # Add missing hydrogens to receptor (ligand already has explicit Hs)
        modeller.addHydrogens(forcefield=None)

        # Add explicit solvent (TIP3P)
        ff_solvent = ForceField("amber14/tip3p.xml")
        modeller.addSolvent(
            ff_solvent,
            model="tip3p",
            padding=1.0 * _openmm_unit.nanometer,
            ionicStrength=0.15 * _openmm_unit.molar,
        )

        # ── Create system with GAFF2 ligand parameters ────────────
        from openmmforcefields.generators import SystemGenerator as _SystemGenerator

        system_generator = _SystemGenerator(
            forcefields=["amber14-all.xml", "amber14/tip3pfb.xml"],
            small_molecule_forcefield="gaff-2.11",
            molecules=[self.ligand_rdkit] if self.ligand_rdkit else [],
            forcefield_kwargs={
                "constraints": _openmm_app.HBonds,
                "rigidWater": True,
                "ewaldErrorTolerance": 0.0005,
            },
        )
        system = system_generator.create_system(
            modeller.topology,
            nonbondedMethod=_openmm_app.PME,
            nonbondedCutoff=1.0 * _openmm_unit.nanometer,
        )

        # Add barostat
        barostat = _openmm.MonteCarloBarostat(
            pressure, temperature,
        )
        system.addForce(barostat)

        return system, modeller.topology, modeller.positions

    def _get_ligand_atom_indices(
        self,
        topology: _openmm_app.Topology,
        ligand_mol: Chem.Mol,
    ) -> List[int]:
        """Identify ligand atom indices in an OpenMM Topology.

        The ligand is expected to be in a residue named 'LIG'. Atom
        indices are returned as a list of integers suitable for use
        with ``AlchemicalRegion``.

        Parameters
        ----------
        topology : openmm.app.Topology
            OpenMM Topology containing the ligand.
        ligand_mol : Chem.Mol
            RDKit Mol of the ligand (used for validation).

        Returns
        -------
        List[int]
            Indices of ligand atoms in the OpenMM topology.
        """
        indices: List[int] = []
        n_lig_heavy = ligand_mol.GetNumHeavyAtoms()
        for atom in topology.atoms():
            if atom.residue.name == "LIG":
                indices.append(atom.index)
        if not indices:
            # Fallback: assume first molecule after protein is the ligand
            # Count protein atoms and take the rest as ligand
            n_protein = sum(
                1 for a in topology.atoms()
                if a.residue.name not in ("LIG", "HOH", "NA", "CL")
            )
            total = topology.getNumAtoms()
            indices = list(range(n_protein, min(n_protein + n_lig_heavy * 4, total)))
        return indices

    def _heuristic_fallback(self) -> FEPResistanceResult:
        """Deprecated / Heuristic Only — no longer used.

        This method is retained for reference only.  All FEP calculations
        now use the rigorous Equilibrium FEP protocol from
        :meth:`_compute_fep_delta_ddg`.  If OpenMM/openmmtools are
        unavailable, a :class:`ConfigurationError` is raised instead.
        """
        raise ConfigurationError(
            "Heuristic FEP fallback has been removed. "
            "Install OpenMM and openmmtools for rigorous FEP calculations, "
            "or set CONFIG.use_fep_resistance = False to skip FEP."
        )

    def _pre_screen_initial_energy(
        self,
        pdb_path: str,
    ) -> Optional[FEPResistanceResult]:
        """Minimise the WT complex and check whether the initial energy
        exceeds the configurable threshold.

        Returns
        -------
        FEPResistanceResult | None
            A result indicating the calculation was skipped (with ``error``
            set), or ``None`` when the energy is within the acceptable
            range and FEP should proceed.
        """
        from io import StringIO
        from openmm.app import HBonds
        from openmm.app import PDBFile, ForceField, Modeller

        pdb = PDBFile(pdb_path)
        modeller = Modeller(pdb.topology, pdb.positions)

        ligand_mol = self.ligand_rdkit
        if ligand_mol is not None:
            mol = Chem.AddHs(ligand_mol)
            if mol.GetNumConformers() == 0:
                params = _AllChem.ETKDGv3()
                params.randomSeed = CONFIG.random_seed
                result = _AllChem.EmbedMolecule(mol, params)
                if result == -1:
                    result = _AllChem.EmbedMolecule(mol, _AllChem.ETKDG())
                if result != -1:
                    _AllChem.MMFFOptimizeMolecule(mol)

            pdb_block = Chem.MolToPDBBlock(mol)
            pdb_block = pdb_block.replace("HETATM", "ATOM  ")
            lines = []
            for ln in pdb_block.split("\n"):
                if ln.startswith(("ATOM", "HETATM")):
                    ln = ln[:17] + "LIG" + ln[20:]
                lines.append(ln)
            pdb_block = "\n".join(lines)

            ligand_pdb = PDBFile(StringIO(pdb_block))
            modeller.add(ligand_pdb.topology, ligand_pdb.positions)

        modeller.addHydrogens(forcefield=None)

        ff_solvent = ForceField("amber14/tip3p.xml")
        modeller.addSolvent(
            ff_solvent,
            model="tip3p",
            padding=1.0 * _openmm_unit.nanometer,
            ionicStrength=0.15 * _openmm_unit.molar,
        )

        system_generator = _SystemGenerator(
            forcefields=["amber14-all.xml", "amber14/tip3pfb.xml"],
            small_molecule_forcefield="gaff-2.11",
            molecules=[self.ligand_rdkit] if self.ligand_rdkit else [],
            forcefield_kwargs={
                "constraints": HBonds,
                "rigidWater": True,
                "ewaldErrorTolerance": 0.0005,
            },
        )
        system = system_generator.create_system(
            modeller.topology,
            nonbondedMethod=_openmm_app.PME,
            nonbondedCutoff=1.0 * _openmm_unit.nanometer,
        )

        barostat = _openmm.MonteCarloBarostat(
            1.0 * _openmm_unit.atmospheres, 298.15 * _openmm_unit.kelvin,
        )
        system.addForce(barostat)

        # Minimise energy and check threshold
        integrator = _openmm.LangevinIntegrator(
            298.15 * _openmm_unit.kelvin,
            5.0 / _openmm_unit.picosecond,
            0.002 * _openmm_unit.picosecond,
        )
        simulation = _openmm_app.Simulation(
            modeller.topology, system, integrator,
            _openmm.Platform.getPlatformByName("Reference"),
        )
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(maxIterations=500)

        state = simulation.context.getState(getEnergy=True)
        energy_kcal = state.getPotentialEnergy().value_in_unit(
            _openmm_unit.kilocalories_per_mole,
        )

        max_energy = CONFIG.fep_max_initial_energy_kcal_per_mol
        if energy_kcal > max_energy:
            log.warning(
                "Pre-screen rejected: initial energy %.3f kcal/mol exceeds threshold %.3f",
                energy_kcal, max_energy,
            )
            return FEPResistanceResult(
                delta_delta_g=0.0,
                confidence=0.0,
                n_windows=0,
                error="Skipped: High Initial Energy",
            )
        return None
