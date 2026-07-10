from __future__ import annotations

import json
import logging
import math
import os
import hashlib
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


class FEPConvergenceError(AutoAntibioticError):
    """Error raised when the FEP calculation fails to converge within
    the allowed number of lambda windows or steps."""


class FETopologyError(AutoAntibioticError):
    """Error raised when the FEP calculation encounters an invalid
    molecular topology (e.g. missing atom types, ligand parameterisation
    failure, or inconsistent residue naming)."""


class FEResourceError(AutoAntibioticError):
    """Error raised when the FEP calculation exhausts computational
    resources (e.g. out-of-memory, disk full, GPU OOM)."""


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

    _system_cache: Dict[str, Tuple[_openmm_app.Topology, _openmm_unit.Quantity]] = {}

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

    @staticmethod
    def _make_cache_key(pdb_path: str) -> str:
        abs_path = os.path.abspath(pdb_path)
        ff_id = (
            "amber14/tip3p.xml:"
            "tip3p:1.0:0.15:"
            "amber14-all.xml,amber14/tip3pfb.xml"
        )
        raw = f"{abs_path}:{ff_id}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _build_solvated_receptor(
        self,
        pdb_path: str,
    ) -> Tuple[_openmm_app.Topology, _openmm_unit.Quantity]:
        cache_key = self._make_cache_key(pdb_path)
        if cache_key in self._system_cache:
            try:
                return self._system_cache[cache_key]
            except Exception as exc:
                log.warning(
                    "System cache retrieval failed for %s: %s. Rebuilding.",
                    pdb_path, exc,
                )

        from openmm.app import PDBFile, ForceField, Modeller

        pdb = PDBFile(pdb_path)
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield=None)

        ff_solvent = ForceField("amber14/tip3p.xml")
        modeller.addSolvent(
            ff_solvent,
            model="tip3p",
            padding=CONFIG.fep_solvent_padding_nm * _openmm_unit.nanometer,
            ionicStrength=CONFIG.fep_ionic_strength_molar * _openmm_unit.molar,
        )

        self._system_cache[cache_key] = (modeller.topology, modeller.positions)
        return modeller.topology, modeller.positions

    def pre_screen_ligand(self) -> None:
        """Pre-screen the ligand for FEP feasibility.

        Checks:
        - Heavy atom count <= 50.
        - SMILES length <= 100.

        Raises
        ------
        ConfigurationError
            If the ligand exceeds any of the pre-screen thresholds.
        """
        if self.ligand_rdkit is None:
            return
        num_heavy = self.ligand_rdkit.GetNumHeavyAtoms()
        if num_heavy > CONFIG.fep_max_heavy_atoms:
            raise ConfigurationError(
                f"Ligand has {num_heavy} heavy atoms (>CONFIG.fep_max_heavy_atoms). "
                "Molecule too large for practical FEP calculation. "
                "Consider skipping FEP for this candidate."
            )
        smi = Chem.MolToSmiles(self.ligand_rdkit)
        if len(smi) > CONFIG.fep_max_smiles_length:
            raise ConfigurationError(
                f"Ligand SMILES length is {len(smi)} characters (>CONFIG.fep_max_smiles_length). "
                "Molecule too large for practical FEP calculation. "
                "Consider skipping FEP for this candidate."
            )

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
        FEPConvergenceError
            If the MBAR estimator fails to converge.
        FETopologyError
            If the molecular topology is invalid for FEP.
        FEResourceError
            If computational resources are exhausted.

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
        self.pre_screen_ligand()

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

        # Pre-screen initial energy
        initial_energy = self._pre_screen_initial_energy(
            self.receptor_wt_pdb,
        )
        if initial_energy is not None:
            return initial_energy

        try:
            return self._compute_fep_delta_ddg()
        except FEPConvergenceError:
            raise
        except FETopologyError:
            raise
        except FEResourceError:
            raise
        except Exception as exc:
            exc_msg = str(exc)
            exc_type = type(exc).__name__

            # Topology / parameterisation errors
            if any(kw in exc_msg.lower() for kw in (
                "residue", "topology", "parameter", "gaff", "atom type",
                "ligand", "forcefield", "nonbonded",
            )):
                raise FETopologyError(
                    f"FEP topology error for ligand {self.ligand_smiles}: "
                    f"{exc_type}: {exc_msg}"
                ) from exc

            # Resource errors
            if any(kw in exc_msg.lower() for kw in (
                "memory", "cuda", "out of memory", "oom", "disk",
                "resource", "cudart",
            )):
                raise FEResourceError(
                    f"FEP resource error: {exc_type}: {exc_msg}"
                ) from exc

            # Default to convergence error for other OpenMM/MBAR failures
            raise FEPConvergenceError(
                f"FEP calculation failed: {exc_type}: {exc_msg}"
            ) from exc

    def retry_with_increased_windows(self, extra_windows: int = 4) -> FEPResistanceResult:
        """Retry the FEP calculation with an increased number of lambda
        windows.  This is useful when a :class:`FEPConvergenceError`
        indicates that the original lambda schedule had insufficient
        phase-space overlap.

        Parameters
        ----------
        extra_windows : int
            Number of additional lambda windows to add (default 4).

        Returns
        -------
        FEPResistanceResult
            The result of the retried calculation.

        Raises
        ------
        FEPConvergenceError
            If the retry also fails to converge.
        """
        original_windows = CONFIG.fep_lambda_windows
        CONFIG.fep_lambda_windows = original_windows + extra_windows
        try:
            log.info(
                "Retrying FEP with %d lambda windows (was %d).",
                CONFIG.fep_lambda_windows, original_windows,
            )
            return self.calculate_ddg()
        finally:
            CONFIG.fep_lambda_windows = original_windows

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
        pressure = CONFIG.fep_pressure_atm * _openmm_unit.atmospheres
        collision_rate = CONFIG.fep_collision_rate_per_ps / _openmm_unit.picosecond
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

        # ── Adaptive lambda refinement (Stages 1 & 2) ────────────
        if CONFIG.fep_adaptive_lambda_insertion and CONFIG.fep_initial_short_steps > 0:
            try:
                log.info(
                    "Adaptive lambda: running diagnostic with %d steps "
                    "on %d windows", CONFIG.fep_initial_short_steps,
                    len(lambda_protocol),
                )
                u_kln_diag = self._run_diagnostic_u_kln(
                    wt_system, wt_topology, wt_positions,
                    alchemical_region_wt, lambda_protocol,
                )
                mbar = MBAR.from_energy_matrix(
                    u_kln_diag, temperature=temperature,
                )
                poor_indices = self._check_overlap_matrix(
                    mbar, lambda_protocol,
                )
                if poor_indices:
                    overlap_matrix = mbar.getOverlapMatrix()
                    log.info(
                        "Adaptive lambda: %d poor-overlap pairs found, "
                        "refining schedule.", len(poor_indices),
                    )
                    log.info(
                        "Initial lambda schedule: %s",
                        np.array_str(lambda_protocol, precision=4),
                    )
                    log.info(
                        "Overlap matrix:\n%s",
                        np.array_str(overlap_matrix, precision=4),
                    )
                    lambda_protocol = self._refine_lambda_schedule(
                        lambda_protocol, poor_indices,
                    )
                    log.info(
                        "Refined lambda schedule: %s",
                        np.array_str(lambda_protocol, precision=4),
                    )
                else:
                    log.info(
                        "Adaptive lambda: all overlaps above threshold "
                        "(%.3f), using initial schedule.",
                        CONFIG.fep_overlap_threshold,
                    )
            except Exception as exc:
                log.warning(
                    "Adaptive lambda refinement failed: %s. "
                    "Falling back to initial %d-window schedule.",
                    exc, len(lambda_protocol),
                )

        checkpoint_path_wt: Optional[str] = None
        checkpoint_path_mut: Optional[str] = None
        if CONFIG.fep_enable_checkpointing:
            ckpt_dir = str(CONFIG.output_dir / "fep_checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_path_wt = os.path.join(ckpt_dir, "checkpoint_WT.json")
            checkpoint_path_mut = os.path.join(ckpt_dir, "checkpoint_Mutant.json")

        def _run_fep_for_system(
            system: _openmm.System,
            topology: _openmm_app.Topology,
            positions: _openmm_unit.Quantity,
            alchemical_region: AlchemicalRegion,
            label: str,
            checkpoint_path: Optional[str] = None,
        ) -> Tuple[float, List[float], float]:
            n_eq_steps = CONFIG.fep_warmup_steps
            min_prod_steps = CONFIG.fep_min_steps_per_window
            max_prod_steps = CONFIG.fep_max_steps_per_window
            check_interval = CONFIG.fep_check_interval_steps
            conv_threshold = CONFIG.fep_convergence_threshold_kcal_per_mol
            unc_threshold = CONFIG.fep_uncertainty_threshold
            min_samples_mbar = CONFIG.fep_min_samples_mbar

            all_window_samples: List[List[np.ndarray]] = []
            per_window_uncertainties: List[float] = []

            # ── Load checkpoint if available ───────────────────────
            start_window = 0
            if checkpoint_path is not None and os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, "r") as f:
                        checkpoint_data = json.load(f)
                    for window_data in checkpoint_data.get("windows", []):
                        samples_list = window_data.get("samples", [])
                        converted = [np.array(s) for s in samples_list]
                        all_window_samples.append(converted)
                        per_window_uncertainties.append(
                            window_data.get("uncertainty", 0.0)
                        )
                    start_window = len(all_window_samples)
                    log.info(
                        "Loaded checkpoint for %s: %d windows completed",
                        label, start_window,
                    )
                except Exception as e:
                    log.warning(
                        "Failed to load checkpoint for %s: %s. "
                        "Restarting from scratch.", label, e,
                    )
                    all_window_samples = []
                    per_window_uncertainties = []

            running_delta_g: List[float] = []

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

                # Adaptive production sampling with MBAR convergence
                window_samples: List[np.ndarray] = []
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

                    # ── MBAR-based convergence check ─────────────
                    if n_steps >= min_prod_steps:
                        current_all = list(all_window_samples) + [window_samples]
                        # Pad with empty lists for windows not yet started
                        while len(current_all) < n_windows:
                            current_all.append([])
                        total_frames = sum(len(s) for s in current_all)
                        n_valid = sum(1 for s in current_all if len(s) > 0)

                        if total_frames >= min_samples_mbar and n_valid >= 2:
                            max_n = max(len(s) for s in current_all)
                            u_kln = np.full((n_windows, max_n, n_windows), np.nan)
                            for k in range(n_windows):
                                for n, u_vec in enumerate(current_all[k]):
                                    u_kln[k, n, :] = u_vec

                            try:
                                mbar = MBAR.from_energy_matrix(
                                    u_kln, temperature=temperature,
                                )
                                delta_f, ddelta_f, _ = mbar.get_free_energy_differences()
                                delta_G = delta_f[-1, 0] * kT
                                uncertainty = float(ddelta_f[-1, 0])
                                delta_G_kcal = delta_G.value_in_unit(
                                    _openmm_unit.kilocalories_per_mole
                                )
                                running_delta_g.append(delta_G_kcal)

                                log.info(
                                    "%s window %d: ΔG = %.3f kcal/mol, "
                                    "uncertainty = %.3f kcal/mol "
                                    "(step %d/%d)",
                                    label, i, delta_G_kcal, uncertainty,
                                    n_steps, max_prod_steps,
                                )

                                if len(running_delta_g) >= 3:
                                    changes = [
                                        abs(running_delta_g[-j]
                                            - running_delta_g[-j - 1])
                                        for j in range(1, 3)
                                    ]
                                    if (all(c < conv_threshold for c in changes)
                                            and uncertainty < unc_threshold):
                                        log.info(
                                            "Convergence reached for %s "
                                            "window %d at step %d "
                                            "(ΔG=%.3f, unc=%.3f)",
                                            label, i, n_steps,
                                            delta_G_kcal, uncertainty,
                                        )
                                        break
                            except Exception:
                                pass
                else:
                    log.warning(
                        "Max steps reached for %s window %d "
                        "without convergence (steps=%d)",
                        label, i, n_steps,
                    )

                all_window_samples.append(window_samples)

                # ── Save checkpoint after each window ─────────────
                if checkpoint_path is not None:
                    try:
                        ckpt_data: Dict[str, Any] = {
                            "label": label,
                            "windows": [
                                {
                                    "index": idx,
                                    "samples": [s.tolist() for s in w_samples],
                                }
                                for idx, w_samples in enumerate(all_window_samples)
                            ],
                            "per_window_uncertainties": per_window_uncertainties,
                        }
                        with open(checkpoint_path, "w") as f:
                            json.dump(ckpt_data, f, indent=2)
                    except Exception as e:
                        log.warning(
                            "Failed to save checkpoint for %s: %s", label, e,
                        )

            # ── Final MBAR estimate ────────────────────────────────
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
                per_window_uncertainties.append(uncertainty)
                total_time_ps = 0.0
                for w_samples in all_window_samples:
                    total_time_ps += (
                        len(w_samples)
                        * check_interval
                        * timestep.value_in_unit(_openmm_unit.picosecond)
                    )
                return float(delta_G_kcal), per_window_uncertainties, total_time_ps
            except Exception:
                return 0.0, [], 0.0

        # Compute ΔG for WT and mutant
        delta_g_wt, per_window_unc_wt, total_time_wt = _run_fep_for_system(
            wt_system, wt_topology, wt_positions,
            alchemical_region_wt, "WT", checkpoint_path_wt,
        )
        delta_g_mut, per_window_unc_mut, total_time_mut = _run_fep_for_system(
            mut_system, mut_topology, mut_positions,
            alchemical_region_mut, "Mutant", checkpoint_path_mut,
        )

        delta_delta_g = delta_g_mut - delta_g_wt
        combined_uncertainty = max(
            max(per_window_unc_wt) if per_window_unc_wt else 0.0,
            max(per_window_unc_mut) if per_window_unc_mut else 0.0,
        )
        if combined_uncertainty > 1.0:
            log.warning(
                "MBAR uncertainty %.3f kcal/mol exceeds 1.0 kcal/mol threshold. "
                "Result marked as Low Confidence.",
                combined_uncertainty,
            )

        confidence = max(0.0, min(1.0, 1.0 - combined_uncertainty))

        all_uncertainties = per_window_unc_wt + per_window_unc_mut
        total_time_ps = total_time_wt + total_time_mut

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

        try:
            receptor_topology, receptor_positions = self._build_solvated_receptor(pdb_path)
        except Exception as exc:
            raise FETopologyError(
                f"Failed to build solvated receptor from {pdb_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        modeller = Modeller(receptor_topology, receptor_positions)

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
            try:
                pdb_block = Chem.MolToPDBBlock(mol)
            except Exception as exc:
                raise FETopologyError(
                    f"Failed to convert ligand to PDB block: {type(exc).__name__}: {exc}"
                ) from exc
            pdb_block = pdb_block.replace("HETATM", "ATOM  ")
            # Rename the residue to "LIG" for consistent identification
            lines = []
            for ln in pdb_block.split("\n"):
                if ln.startswith(("ATOM", "HETATM")):
                    ln = ln[:17] + "LIG" + ln[20:]
                lines.append(ln)
            pdb_block = "\n".join(lines)

            try:
                ligand_pdb = PDBFile(StringIO(pdb_block))
                modeller.add(ligand_pdb.topology, ligand_pdb.positions)
            except Exception as exc:
                raise FETopologyError(
                    f"Failed to add ligand to receptor model: {type(exc).__name__}: {exc}"
                ) from exc

        # ── Create system with GAFF2 ligand parameters ────────────
        from openmmforcefields.generators import SystemGenerator as _SystemGenerator

        try:
            system_generator = _SystemGenerator(
                forcefields=["amber14-all.xml", "amber14/tip3pfb.xml"],
                small_molecule_forcefield="gaff-2.11",
                molecules=[self.ligand_rdkit] if self.ligand_rdkit else [],
                forcefield_kwargs={
                    "constraints": _openmm_app.HBonds,
                    "rigidWater": True,
                    "ewaldErrorTolerance": CONFIG.fep_ewald_error_tolerance,
                },
            )
            system = system_generator.create_system(
                modeller.topology,
                nonbondedMethod=_openmm_app.PME,
                nonbondedCutoff=CONFIG.fep_nonbonded_cutoff_nm * _openmm_unit.nanometer,
            )
        except Exception as exc:
            exc_msg = str(exc).lower()
            if any(kw in exc_msg for kw in ("memory", "cuda", "oom")):
                raise FEResourceError(
                    f"FEP resource error during system creation: {type(exc).__name__}: {exc}"
                ) from exc
            raise FETopologyError(
                f"FEP topology error during system creation: {type(exc).__name__}: {exc}"
            ) from exc

        # Add barostat
        pressure = CONFIG.fep_pressure_atm * _openmm_unit.atmospheres
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
        for atom in topology.atoms():
            if atom.residue.name == "LIG":
                indices.append(atom.index)
        if not indices:
            n_lig_heavy = ligand_mol.GetNumHeavyAtoms()
            # Fallback: assume first molecule after protein is the ligand
            n_protein = sum(
                1 for a in topology.atoms()
                if a.residue.name not in ("LIG", "HOH", "NA", "CL")
            )
            total = topology.getNumAtoms()
            fallback_indices = list(range(n_protein, min(n_protein + n_lig_heavy * 4, total)))
            if fallback_indices:
                log.warning(
                    "No LIG residue found in topology — using fallback atom index "
                    "heuristic (%d atoms). Ligand SMILES: %s",
                    len(fallback_indices),
                    Chem.MolToSmiles(ligand_mol) if ligand_mol else "unknown",
                )
                return fallback_indices
            raise FETopologyError(
                f"No LIG residue found in topology and fallback heuristic failed. "
                f"The ligand may not have been properly added to the system. "
                f"Ligand SMILES: "
                f"{Chem.MolToSmiles(ligand_mol) if ligand_mol else 'unknown'}"
            )
        return indices

    @staticmethod
    def _check_overlap_matrix(
        mbar_instance: Any,
        lambda_schedule: np.ndarray,
    ) -> List[int]:
        """Check the MBAR overlap matrix and return indices of adjacent
        lambda windows where the overlap integral is below the threshold.

        Parameters
        ----------
        mbar_instance
            A fitted ``openmmtools.multistate.MBAR`` instance whose
            ``getOverlapMatrix()`` returns an ``(n_windows, n_windows)``
            overlap matrix.
        lambda_schedule : np.ndarray
            Array of lambda values for each window, used only to
            determine the number of windows.

        Returns
        -------
        List[int]
            Indices *i* such that the overlap between window *i* and
            *i+1* is below ``CONFIG.fep_overlap_threshold``.
        """
        overlap_matrix = mbar_instance.getOverlapMatrix()
        n_windows = len(lambda_schedule)
        poor: List[int] = []
        for i in range(n_windows - 1):
            overlap = overlap_matrix[i, i + 1]
            if overlap < CONFIG.fep_overlap_threshold:
                poor.append(i)
        return poor

    @staticmethod
    def _refine_lambda_schedule(
        lambda_schedule: np.ndarray,
        poor_indices: List[int],
    ) -> np.ndarray:
        """Insert intermediate lambda windows at poor-overlap pairs.

        Parameters
        ----------
        lambda_schedule : np.ndarray
            Current lambda schedule (sorted, values in [0, 1]).
        poor_indices : List[int]
            Indices of windows where the overlap with the next window
            is insufficient (from ``_check_overlap_matrix``).

        Returns
        -------
        np.ndarray
            Refined lambda schedule with intermediate windows inserted,
            capped at ``CONFIG.fep_max_lambda_windows``.
        """
        new_schedule = list(float(v) for v in lambda_schedule)
        inserted = 0
        max_windows = CONFIG.fep_max_lambda_windows

        for idx in sorted(poor_indices, reverse=True):
            if len(new_schedule) >= max_windows:
                log.warning(
                    "Adaptive lambda: max windows (%d) reached, "
                    "cannot insert more.", max_windows,
                )
                break
            mid = (new_schedule[idx] + new_schedule[idx + 1]) * 0.5
            new_schedule.insert(idx + 1, mid)
            inserted += 1
            log.info(
                "Inserting intermediate lambda window at λ=%.4f "
                "between windows %d (λ=%.4f) and %d (λ=%.4f)",
                mid, idx, new_schedule[idx], idx + 2, new_schedule[idx + 2],
            )

        if inserted:
            log.info(
                "Refined lambda schedule: %d windows → %d windows",
                len(lambda_schedule), len(new_schedule),
            )
        return np.array(sorted(new_schedule))

    def _run_diagnostic_u_kln(
        self,
        system: _openmm.System,
        topology: _openmm_app.Topology,
        positions: _openmm_unit.Quantity,
        alchemical_region: Any,
        lambda_schedule: np.ndarray,
    ) -> np.ndarray:
        """Run a short diagnostic simulation and return the u_kln matrix.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System to simulate.
        topology : openmm.app.Topology
            The corresponding Topology.
        positions : openmm.unit.Quantity
            Initial atomic positions.
        alchemical_region : AlchemicalRegion
            Region defining which atoms are alchemically modified.
        lambda_schedule : np.ndarray
            Lambda values for each window.

        Returns
        -------
        np.ndarray
            The ``(n_windows, max_n_frames, n_windows)`` reduced-potential
            matrix suitable for ``MBAR.from_energy_matrix``.
        """
        from openmmtools.alchemy import (
            AbsoluteAlchemicalFactory,
            AlchemicalState,
        )

        n_windows = len(lambda_schedule)
        temperature = 298.15 * _openmm_unit.kelvin
        kT = _openmm_unit.MOLAR_GAS_CONSTANT_R * temperature
        n_steps = CONFIG.fep_initial_short_steps
        timestep = CONFIG.fep_time_step_ps * _openmm_unit.picosecond
        collision_rate = CONFIG.fep_collision_rate_per_ps / _openmm_unit.picosecond

        factory = AbsoluteAlchemicalFactory()
        platform = _openmm.Platform.getPlatformByName("Reference")
        all_samples: List[List[np.ndarray]] = [[] for _ in range(n_windows)]

        for i, lam in enumerate(lambda_schedule):
            alchemical_state = AlchemicalState.from_system(system)
            alchemical_state.lambda_sterics = lam
            alchemical_state.lambda_electrostatics = lam
            alchemical_state.lambda_torsions = lam

            alchemical_system = factory.create_alchemical_system(
                system, alchemical_region,
                alchemical_state=alchemical_state,
            )

            integrator = _openmm.LangevinIntegrator(
                temperature, collision_rate, timestep,
            )
            integrator.setRandomSeed(CONFIG.random_seed + i)

            simulation = _openmm_app.Simulation(
                topology, alchemical_system, integrator, platform,
            )
            simulation.context.setPositions(positions)
            simulation.minimizeEnergy(maxIterations=n_steps)
            simulation.step(n_steps)

            state = simulation.context.getState(
                getEnergy=True, getPositions=True, getParameters=True,
            )
            current_pos = state.getPositions()
            ref_potential = state.getPotentialEnergy()

            u_k = np.zeros(n_windows)
            for j in range(n_windows):
                if j == i:
                    pot_diff = ref_potential - ref_potential
                else:
                    alchemical_state_j = AlchemicalState.from_system(system)
                    alchemical_state_j.lambda_sterics = lambda_schedule[j]
                    alchemical_state_j.lambda_electrostatics = lambda_schedule[j]
                    alchemical_state_j.lambda_torsions = lambda_schedule[j]

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
                    _openmm_unit.kilojoules_per_mole,
                )

            all_samples[i].append(u_k)

        max_n = max(len(s) for s in all_samples)
        u_kln = np.full((n_windows, max_n, n_windows), np.nan)
        for k in range(n_windows):
            for n, u_vec in enumerate(all_samples[k]):
                u_kln[k, n, :] = u_vec

        return u_kln

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
            padding=CONFIG.fep_solvent_padding_nm * _openmm_unit.nanometer,
            ionicStrength=CONFIG.fep_ionic_strength_molar * _openmm_unit.molar,
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
            CONFIG.fep_pressure_atm * _openmm_unit.atmospheres,
            298.15 * _openmm_unit.kelvin,
        )
        system.addForce(barostat)

        # Record initial positions of ligand heavy atoms for RMSD check
        ligand_heavy_indices: List[int] = []
        for atom in modeller.topology.atoms():
            if atom.residue.name == "LIG" and atom.element is not None and atom.element.symbol != "H":
                ligand_heavy_indices.append(atom.index)

        initial_lig_positions: List[np.ndarray] = []
        for idx in ligand_heavy_indices:
            pos = modeller.positions[idx]
            initial_lig_positions.append(np.array([
                pos[0].value_in_unit(_openmm_unit.angstroms),
                pos[1].value_in_unit(_openmm_unit.angstroms),
                pos[2].value_in_unit(_openmm_unit.angstroms),
            ]))

        # Minimise energy and check threshold
        integrator = _openmm.LangevinIntegrator(
            298.15 * _openmm_unit.kelvin,
            CONFIG.fep_collision_rate_per_ps / _openmm_unit.picosecond,
            CONFIG.fep_time_step_ps * _openmm_unit.picosecond,
        )
        simulation = _openmm_app.Simulation(
            modeller.topology, system, integrator,
            _openmm.Platform.getPlatformByName("Reference"),
        )
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(maxIterations=CONFIG.fep_minimization_iterations)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy_kcal = state.getPotentialEnergy().value_in_unit(
            _openmm_unit.kilocalories_per_mole,
        )

        # Check ligand RMSD after minimisation
        if ligand_heavy_indices:
            min_positions = state.getPositions()
            rmsd_sq_sum = 0.0
            for i, idx in enumerate(ligand_heavy_indices):
                pos = min_positions[idx]
                min_p = np.array([
                    pos[0].value_in_unit(_openmm_unit.angstroms),
                    pos[1].value_in_unit(_openmm_unit.angstroms),
                    pos[2].value_in_unit(_openmm_unit.angstroms),
                ])
                rmsd_sq_sum += float(np.sum((initial_lig_positions[i] - min_p) ** 2))
            rmsd = math.sqrt(rmsd_sq_sum / len(ligand_heavy_indices))
            if rmsd > 2.0:
                log.warning(
                    "Pre-screen rejected: ligand RMSD %.3f Å exceeds 2.0 Å threshold",
                    rmsd,
                )
                return FEPResistanceResult(
                    delta_delta_g=0.0,
                    confidence=0.0,
                    n_windows=0,
                    error="Skipped: High Ligand RMSD",
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
