from __future__ import annotations

import logging
import multiprocessing as mp
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ConfigurationError(Exception):
    """Error raised when the pipeline configuration is invalid or
    required dependencies for a requested feature are missing.

    The error message is always actionable, telling the user what
    is missing and how to resolve it.
    """


@dataclass
class PipelineConfig:
    """Top-level configuration container for the AutoAntibiotic pipeline."""

    # ── Pipeline-level fields ──
    random_seed: int = 42
    output_dir: Path = Path("output")
    dry_run: bool = False
    library_target_count: int = 500
    library_generator_threshold: int = 1000
    brics_min_fragment_size: int = 8
    morgan_radius: int = 2
    morgan_nbits: int = 2048
    pdb_retry_max_attempts: int = 3
    pdb_retry_base_delay: float = 2.0
    n_jobs: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))

    selectivity_index_threshold: float = 2.0
    shape_score_norm_factor: float = 0.05
    diversity_pool_multiplier: int = 5
    redocking_rmsd_cutoff: float = 2.0

    # Resistance / analysis thresholds
    resistance_energy_active_threshold: float = -6.0
    resistance_energy_allosteric_threshold: float = -7.0
    resistance_mw_threshold: float = 400.0
    resistance_rot_threshold: int = 5
    resistance_qed_threshold: float = 0.8

    # Consensus scoring weights
    consensus_vina_weight: float = 0.7
    consensus_shape_weight: float = 0.3

    # Mutation sampling / resistance profiling
    use_mutation_sampling: bool = False
    mutation_variants: List[str] = field(default_factory=lambda: [
        "G246", "N146", "E150", "H351", "E644",
        "A601", "F241", "N104", "G298", "S403",
        "N159", "R241",
    ])

    # Meta-learner consensus scoring
    use_meta_scoring: bool = True
    meta_scorer_model_path: str = "output/meta_scorer.joblib"

    # GNN rescoring (replaces/ augments meta-scorer when enabled)
    use_gnn_rescoring: bool = False
    gnn_model_path: str = "output/gnn_model.pt"

    # MD validation
    md_validation_duration_ns: int = 10
    md_production_duration_ns: int = 50
    md_relaxation_duration_ns: int = 1
    md_convergence_check_interval_ns: float = 5.0
    """Interval (ns) for convergence checking during MD production.
    After each chunk, RMSD is evaluated; if stable (std < 0.1 Å over
    last *window_size* frames), the simulation stops early."""
    md_max_duration_ns: int = 100
    """Maximum MD production duration (ns) for adaptive sampling.
    If convergence is not reached within ``md_production_duration_ns``,
    the simulation continues up to this hard cap.  Default 100 ns."""
    md_convergence_window_chunks: int = 3
    """Number of recent chunks used to assess convergence during
    adaptive MD sampling.  Default 3."""
    md_rmsd_convergence_threshold: float = 0.1
    """RMSD standard deviation threshold (Å) for declaring convergence
    during adaptive MD sampling.  Default 0.1 Å."""

    # MetaScorer training constraints
    min_training_samples: int = 20
    """Minimum number of actives + inactives required to train the
    MetaScorer.  If fewer samples are available, a ConfigurationError
    is raised to prevent fitting on noise."""

    # Force MD for meta-scoring
    force_md_for_meta_scoring: bool = False
    """When True, raise ConfigurationError if MD validation fails for
    top candidates before meta-scoring.  This ensures that the MetaScorer
    always receives MD-derived dynamic features for accurate predictions."""

    # MM-GB/SA rescoring solvent model
    mmgbsa_solvent_model: str = "implicit"
    """Solvent model for MM-GB/SA rescoring.  Options: ``"implicit"``
    (OBC2, default) or ``"explicit"`` (TIP3P with pose relaxation).
    When set to ``"explicit"``, ``pdbfixer`` must be installed."""
    use_explicit_solvent_mmgbsa: bool = True
    """When True, use explicit-solvent (TIP3P) MM-GB/SA for rescoring top
    candidates instead of the implicit-solvent (OBC2) heuristic."""
    explicit_solvent_frames: int = 10
    """Number of trajectory snapshots to average for explicit-solvent
    MM-GB/SA rescoring (default 10)."""
    use_strict_scoring: bool = False
    """When True, tightens the volume-overlap clash detection in water
    displacement penalty scoring, lowering the low-overlap threshold from
    10 % to 5 % for more aggressive penalty application.  When used in
    combination with ``--strict-scoring`` CLI flag, ``explicit_solvent_frames``
    is increased to 20 for higher precision."""
    """Relaxation MD duration (ns) with strong position restraints. Default 1 ns."""

    # Benchmark
    benchmark_mode: bool = False
    reference_actives_path: Optional[Path] = None
    reference_inactives_path: Optional[Path] = None
    benchmark_n_decoys: int = 100

    # Library data
    beta_lactam_smarts: str = "[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1"
    allosteric_residues: List[str] = field(default_factory=lambda: ["ASN159", "GLU237", "ARG241"])
    active_site_residues: List[str] = field(default_factory=lambda: ["SER403"])
    trypsin_active_site_residues: List[str] = field(default_factory=lambda: ["HIS57", "ASP102", "SER195"])
    ces1_active_site_residues: List[str] = field(default_factory=lambda: ["SER221", "HIS468", "GLU354"])
    conserved_residues: set = field(default_factory=lambda: {"SER403", "KYS406", "TYR446"})
    mutable_residues: set = field(default_factory=lambda: {"G246", "N146", "E150", "H351", "E644", "A601", "F241", "N104", "G298", "S403", "N159", "R241"})
    use_pharmacophore_filter: bool = True
    strict_receptor_validation: bool = True
    """When True, raised ConfigurationError if PDBFixer repair fails or
    critical backbone atoms are missing during receptor integrity check.
    Set to False to continue with incomplete structures (logged as warnings)."""

    pdb_ids: Dict[str, str] = field(default_factory=lambda: {
        "PBP2a_apo": "1VQQ",
        "PBP2a_holo": "3ZG0",
        "trypsin": "1UTN",
        "CES1": "1YA4",
    })
    reference_antibiotics: Dict[str, str] = field(default_factory=lambda: {
        "Methicillin":  "CC1(C(N2C(S1)C(C2=O)NC(=O)C3=C(C(=C(C=C3)OC)OC)OC)C(=O)O)C",
        "Vancomycin":   "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C(=C4OC6=C(C(=CC(=C6)C(C(=O)NC(C(=O)NC5C(=O)O)CC7=CC=C(C=C7)O)NC(=O)C8C(O)C(=C(C=C8)Cl)O)O)O)CO)O)O)O)NC(=O)C9C(O)C(=C(C=C9)Cl)O)(CC(=O)N)O",
        "Ceftaroline":  "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "Meropenem":    "CC1C2C(C(=O)N2C(=C1SC3CC(NCC3)C(=O)O)C(=O)O)(C)O",
        "Oxacillin":    "CC1=C(C(=NO1)C2=CC=CC=C2)C(=O)NC3C4C(C(=O)N4C3=O)SC5(C)C",
    })
    control_smiles: Dict[str, str] = field(default_factory=lambda: {
        "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NCC3)C(=O)O)C(=O)O)(C)O",
    })
    natural_product_scaffolds: List[str] = field(default_factory=lambda: [
        "O=c1c(-c2ccc(O)c(O)c2)coc2cc(O)cc(O)c12",
        "Oc1ccc(C=Cc2ccc(O)cc2)cc1",
        "COc1ccc(C=CC(=O)CC(=O)C=Cc2ccc(OC)c(O)c2)cc1O",
        "COc1cc2c(cc1OC)-c1ccc3cc4c(cc3c1CC2)OCO4",
        "CC1(C)OC2C3OC(=O)C4C(O1)C2C1OOC3C14",
        "O=C1OCc2cn3ccc4cccc-4c3cc21",
        "COc1nc2c3ccccc3n(C)c2cc1C1CCNC1O",
        "O=C(Nc1ccccc1)c1ccccc1",
    ])
    additional_scaffolds: List[str] = field(default_factory=lambda: [
        "c1ccc2[nH]ccc2c1",
        "c1ccc2ncccc2c1",
        "c1ccc2cc[nH]c2c1",
        "c1ccc2[nH]cnc2c1",
        "O=c1ccc2ccccc2o1",
        "c1ccc2nc3ccccc3nc2c1",
        "c1ccc2c(c1)oc1ccccc12",
        "c1ccc2c(c1)sc1ccccc12",
        "c1ccc2c(c1)ccc1c3ccccc3[nH]c21",
        "c1ccc2c(c1)CCN2",
        "c1ccc2c(c1)CCc1c-2[nH]c2ccccc12",
        "COc1ccc2[nH]ccc2c1",
        "COc1ccccc1OCC(O)CNC(C)C",
        "CCN(CC)C(=O)c1ccccc1",
        "O=C(Nc1ccc(O)cc1)c1ccc(O)cc1",
    ])
    brics_building_blocks: List[str] = field(default_factory=lambda: [
        "[1*]c1ccccc1", "[1*]c1ccc(O)cc1", "[1*]c1ccc(Cl)cc1",
        "[1*]c1ccc(F)cc1", "[1*]c1ccc(Br)cc1", "[1*]c1ccc(OC)cc1",
        "[1*]c1ccc(C(=O)O)cc1", "[1*]c1ccc(N)cc1", "[1*]c1ccc(C)cc1",
        "[1*]c1ccc(C(C)C)cc1", "[1*]c1ccc(CF)cc1", "[1*]c1ccc(CN)cc1",
        "[1*]c1ccc(S(=O)(=O)N)cc1", "[1*]c1ccc(C(=O)N)cc1",
        "[1*]c1ccc(NC(=O)C)cc1", "[1*]CC(=O)O", "[1*]CCO", "[1*]CCN",
        "[1*]CC(=O)N", "[1*]CCC(=O)O", "[3*]C=Cc1ccccc1",
        "[3*]C=Cc1ccc(O)cc1", "[3*]C=Cc1ccc(Cl)cc1", "[3*]CCN(C)C",
        "[5*]Nc1ccccc1", "[5*]Nc1ccc(O)cc1", "[5*]Nc1ccc(C(=O)O)cc1",
        "[5*]Nc1ccc(Cl)cc1", "[5*]Nc1ccc(F)cc1", "[5*]Nc1ccc(OC)cc1",
        "[5*]Nc1ccc(C)cc1", "[5*]Nc1ccc(Br)cc1", "[5*]Nc1ccc(CN)cc1",
        "[5*]NCC", "[5*]NCCO", "[5*]NCCC(=O)O", "[6*]C(=O)O",
        "[6*]C(=O)c1ccccc1", "[6*]C(=O)c1ccc(O)cc1",
        "[6*]C(=O)c1ccc(Cl)cc1", "[6*]C(=O)c1ccc(OC)cc1",
        "[6*]C(=O)c1ccc(C)cc1", "[6*]C(=O)c1ccc(N)cc1",
        "[6*]C(=O)CC", "[7*]Cc1ccccc1", "[7*]Cc1ccc(O)cc1",
        "[7*]Cc1ccc(O)c(OC)c1", "[7*]Cc1ccc(OC)cc1",
        "[7*]Cc1ccc(Cl)cc1", "[7*]Cc1ccc(F)cc1", "[7*]CC",
        "[7*]C(C)C", "[16*]c1ccccc1OC", "[16*]c1ccc(C)cc1",
        "[16*]c1ccc(N)cc1", "[16*]c1ccc(O)cc1",
    ])

    # ── Docking parameters ──
    use_dynamic_box_sizing: bool = True
    """When True, calculate grid box dimensions dynamically based on the
    ligand's maximum heavy-atom distance.  Falls back to fixed box sizes
    (``allosteric_box_size`` / ``active_box_size``) when False."""
    dynamic_box_padding: float = 5.0
    """Padding (Å) added to the ligand's half-maximum dimension when
    computing the dynamic grid box size."""
    validate_docking_binaries_on_startup: bool = True
    """When True, validate docking binary versions and health on startup."""
    vina_exhaustiveness: int = 8
    vina_num_modes: int = 3
    vina_timeout_s: int = 120
    job_timeout_s: int = 180
    allosteric_box_size: Tuple[float, float, float] = (15.0, 15.0, 15.0)
    active_box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    offtarget_box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    redocking_box_size: Tuple[float, float, float] = (25.0, 25.0, 25.0)
    batch_size_docking: int = 75
    prepare_receptor_timeout: int = 60
    obabel_timeout_s: int = 60
    use_gnina: bool = False
    gnina_binary_path: str = "gnina"
    ensemble_mode: bool = True
    ensemble_structures_dir: Optional[Path] = None
    default_ensemble_pdb_ids: List[str] = field(default_factory=lambda: ["1VQQ", "3ZG0", "4CJN"])
    consensus_scoring_method: str = "rank"
    flexible_docking: bool = False
    flexible_residues_allosteric: List[str] = field(default_factory=lambda: ["GLU237", "ARG241", "ASN159"])
    flexible_residues_active: List[str] = field(default_factory=lambda: ["SER403"])
    max_flexible_conformers: int = 9
    use_ml_rescoring: bool = True
    use_mm_gbsa: bool = False
    use_mm_gbsa_rescoring: bool = False
    mm_gbsa_top_n: int = 50
    key_interaction_residues_allosteric: List[str] = field(default_factory=lambda: ["ASN159", "GLU237", "ARG241"])
    key_interaction_residues_active: List[str] = field(default_factory=lambda: ["SER403"])
    min_key_interactions: int = 1
    require_key_interactions_for_rescoring: bool = True
    """When True, filter out docked poses that lack key interactions
    (H-bond / hydrophobic contacts with critical residues) before
    expensive MM-GB/SA rescoring.  Fail-safe: if the interaction check
    fails, the compound is kept."""

    # ── Filtering parameters ──
    similarity_threshold: float = 0.4
    similarity_threshold_relaxed: float = 0.5
    diversity_min_count: int = 100
    qed_threshold: float = 0.6
    lipinski_mw_max: float = 500.0
    lipinski_logp_max: float = 5.0
    lipinski_hbd_max: int = 5
    lipinski_hba_max: int = 10
    sa_score_threshold: float = 6.0
    strain_energy_threshold: float = 10.0
    ifp_similarity_threshold: float = 0.6
    pharmacophore_min_matches: int = 2
    pharmacophore_tolerance: float = 2.0
    pharmacophore_rmsd_threshold: float = 1.5
    pharmacophore_ref_ligand_smi: str = ""

    # ── Expensive ML feature toggle ──
    # Stereochemistry enumeration is ALWAYS enabled (removed from this gate)
    # and uses a Smart Filter (MMFF94 strain > 10 kcal/mol → discard).
    use_expensive_ml_features: bool = False
    """Enable expensive features: ensemble MM-GB/SA, and similar.
    Set to ``True`` for production runs; keep ``False`` for dry-run / quick-test.
    NOTE: stereoisomer enumeration is now always on (not gated by this flag)."""

    mmgbsa_n_conformers: int = 10
    """Number of ligand-receptor conformers for ensemble MM-GB/SA averaging."""

    max_stereoisomers: int = 8
    """Maximum stereoisomers to enumerate per undefined-stereo molecule.
    Each isomer is strain-filtered via MMFF94 before entering the library pool."""

    # ── ML-ADMET parameters ──
    use_ml_admet: bool = True
    ml_admet_herg_threshold: float = 0.5
    ml_admet_solubility_threshold: float = -4.0
    ml_admet_model_type: str = "chemberta_rf"
    """Model type for ML-ADMET: ``"rule_based"`` (no ML), ``"rf_legacy"``
    (fingerprint + RandomForest), or ``"chemberta_rf"`` (ChemBERTa
    embeddings + RandomForest -- falls back to ``rf_legacy`` if
    transformers/torch are unavailable)."""

    chemberta_model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    """HuggingFace model name for ChemBERTa embeddings."""

    # ── Water analysis parameters ──
    use_water_analysis: bool = True
    water_distance_cutoff: float = 5.0
    water_displacement_energy_threshold: float = 2.5

    # ── IFP distance thresholds (moved from scoring_metrics.py) ──
    ifp_hba_dist: float = 3.5
    """Distance cutoff (Å) for H-bond acceptor interactions in IFP."""
    ifp_hbd_dist: float = 3.5
    """Distance cutoff (Å) for H-bond donor interactions in IFP."""
    ifp_hyd_dist: float = 4.5
    """Distance cutoff (Å) for hydrophobic interactions in IFP."""
    ifp_pi_dist: float = 5.5
    """Distance cutoff (Å) for pi-stacking interactions in IFP."""

    # ── FEP physical constants (moved from fep_engine.py) ──
    fep_collision_rate_per_ps: float = 5.0
    """Collision rate (1/ps) for the Langevin integrator in FEP."""
    fep_nonbonded_cutoff_nm: float = 1.0
    """Non-bonded cutoff (nm) for FEP simulations."""
    fep_solvent_padding_nm: float = 1.0
    """Solvent padding (nm) for FEP solvation box."""
    fep_ionic_strength_molar: float = 0.15
    """Ionic strength (M) for FEP solvation."""
    fep_pressure_atm: float = 1.0
    """Pressure (atm) for FEP barostat."""
    fep_min_samples_mbar: int = 100
    """Minimum number of total samples required for MBAR estimation."""
    fep_max_heavy_atoms: int = 50
    """Maximum heavy-atom count for FEP pre-screening."""
    fep_max_smiles_length: int = 100
    """Maximum SMILES length for FEP pre-screening."""
    fep_minimization_iterations: int = 500
    """Number of minimization iterations for FEP energy pre-screening."""
    fep_ewald_error_tolerance: float = 0.0005
    """Ewald error tolerance for FEP simulations."""

    # ── ADMET reference data path ──
    admet_reference_csv: str = "data/admet_reference_curated.csv"
    """Path to the curated ADMET reference CSV used as fallback when ChEMBL
    API is unavailable."""

    # ── FEP / resistance profiling ──
    use_fep_resistance: bool = True
    """When True, use OpenMM-based Free Energy Perturbation (FEP) for
    resistance profiling instead of the heuristic standard-deviation
    approach.  Falls back to the heuristic when OpenMM/alchemical tools
    are unavailable."""
    use_heuristic_resistance_fallback: bool = True
    """When True and FEP fails for a specific compound, fall back to
    heuristic docking-based SD resistance profiling instead of crashing
    or skipping the compound entirely."""
    fep_lambda_windows: int = 11
    """Number of lambda windows for the FEP alchemical transformation."""
    fep_stages: str = "van_der_waals_and_electrostatics"
    """FEP stage combination: ``'van_der_waals_and_electrostatics'``,
    ``'van_der_waals'``, or ``'electrostatics'``."""
    fep_n_steps: int = 5000
    """Number of FEP steps per lambda window (equilibration + production)."""
    fep_production_steps: int = 4500
    """Number of production FEP steps per lambda window, distinct from
    equilibration / warm-up steps.  Default 4500."""
    fep_convergence_threshold: float = 0.1
    """Convergence threshold (kcal/mol) for adaptive FEP sampling.  When
    the cumulative ΔG estimate changes by less than this value over the
    last 3 checks, the lambda window terminates early.  Default 0.1."""
    fep_convergence_threshold_kcal_per_mol: float = 0.5
    """Convergence threshold (kcal/mol) for adaptive FEP sampling.
    When the change in cumulative ΔG over the last 3 checks is below
    this value and uncertainty is below ``fep_uncertainty_threshold``,
    the lambda window terminates early.  Default 0.5."""
    fep_uncertainty_threshold: float = 1.0
    """MBAR uncertainty threshold (kcal/mol) for adaptive FEP sampling.
    When the cumulative uncertainty drops below this value and the ΔG
    change is below ``fep_convergence_threshold_kcal_per_mol``, the
    lambda window terminates early.  Default 1.0."""
    fep_check_interval_steps: int = 500
    """Number of production steps between convergence checks.
    Default 500."""
    fep_min_steps_per_window: int = 1000
    """Minimum number of production steps per lambda window, regardless of
    convergence.  Default 1000."""
    fep_max_steps_per_window: int = 10000
    """Maximum number of production steps per lambda window (hard cap).
    Default 10000."""
    fep_kT_kcal_per_mol: float = 0.596
    """kT value (kcal/mol) at 298.15 K for FEP free-energy calculation."""
    fep_time_step_ps: float = 0.002
    """Time step (ps) for FEP MD simulation."""
    fep_warmup_steps: int = 500
    """Number of warm-up steps before production FEP sampling."""
    fep_warmup_min_iterations: int = 500
    """Minimum energy minimisation iterations before FEP production."""
    fep_max_initial_energy_kcal_per_mol: float = 1000.0
    """Maximum potential energy (kcal/mol) for pre-screening acceptance.
    If the initial energy minimisation on the WT complex exceeds this
    threshold, the FEP calculation is skipped and a "Skipped: High
    Initial Energy" result is returned."""
    fep_enable_checkpointing: bool = True
    """When True, save and load checkpoint files during FEP calculations
    so that interrupted simulations can be resumed from the last
    completed window.  When False (or when checkpoint_dir is None),
    no checkpointing is performed."""
    fep_adaptive_lambda_insertion: bool = True
    """When True, use adaptive lambda window refinement: run a short
    diagnostic simulation, compute the MBAR overlap matrix, and insert
    intermediate lambda windows where adjacent-window overlap is below
    ``fep_overlap_threshold``.  Set to False to use the original fixed
    lambda schedule."""
    fep_overlap_threshold: float = 0.03
    """Minimum acceptable MBAR overlap integral between adjacent lambda
    windows.  If any adjacent pair has overlap below this threshold, an
    intermediate lambda window is inserted at the midpoint.  Only used
    when ``fep_adaptive_lambda_insertion`` is True."""
    fep_max_lambda_windows: int = 21
    """Hard cap on the number of lambda windows after adaptive insertion.
    Prevents unbounded window proliferation."""
    fep_initial_short_steps: int = 100
    """Number of steps for the initial diagnostic run used to assess
    phase-space overlap between adjacent lambda windows.  Only used
    when ``fep_adaptive_lambda_insertion`` is True."""
    fep_top_n: int = 20
    """Maximum number of top candidates to run rigorous FEP on.
    FEP is only triggered if the candidate is within this many top
    compounds after docking and MM-GB/SA rescoring.  Default 20."""
    fep_pre_screen_pool_size: int = 20
    """Number of top candidates to consider for IFP pre-screening
    before running FEP.  Candidates are first expanded to this pool,
    then filtered by IFP similarity before FEP.  Default 20."""
    fep_ifp_threshold: float = 0.5
    """Minimum IFP Tanimoto similarity to the reference ligand
    (Ceftaroline) required for a candidate to proceed to FEP after
    pre-screening.  Default 0.5."""
    fep_top_n_strict: int = 5
    """Number of top candidates to run FEP on after strict pre-screening
    (IFP >= 0.7 and allosteric energy < -8.0 kcal/mol).  Default 5."""

    # ── Entropy estimation ──
    include_entropy: bool = False
    """When True, include entropy estimation (Normal Mode Analysis) in MM-GB/SA rescoring."""
    entropy_nma_frames: int = 10
    """Number of trajectory frames for quasi-harmonic entropy estimation."""
    entropy_min_rmsd: float = 0.5
    """Minimum RMSD threshold for including frames in entropy calculation."""

    # ── Generative design parameters ──
    generative_mode: bool = False
    """When True, use a JT-VAE / graph-based generative model to produce
    novel scaffold analogs instead of rigid BRICS recombination."""
    generative_n_samples: int = 100
    """Number of novel scaffolds to generate per core scaffold."""
    generative_temperature: float = 0.8
    """Sampling temperature for the JT-VAE latent-space decode."""
    generative_max_length: int = 40
    """Maximum SMILES length for generated molecules."""
    generative_min_length: int = 8
    """Minimum SMILES length for generated molecules."""
    generative_n_workers: int = 4
    """Number of parallel workers for latent-space decoding."""

    # ── Synthesis planning parameters ──
    strict_synthesis_check: bool = False
    """When True, apply hard synthesis-filter based on retrosynthesis API
    results (e.g. IBM RXN or ASKCOS)."""
    synthesis_api_url: str = "https://rxn.rxnchemistry.com/rxnchem"
    """Base URL for the retrosynthesis API (IBM RXN default)."""
    synthesis_api_timeout_s: int = 30
    """Timeout (seconds) for retrosynthesis API requests."""
    synthesis_api_min_confidence: float = 0.5
    """Minimum confidence score to accept a synthetic route."""
    synthesis_api_max_routes: int = 3
    """Maximum number of synthesis routes to evaluate per compound."""

    # ── Active learning parameters ──
    uncertainty_threshold: float = 0.1
    """Standard-deviation threshold for prediction uncertainty.
    When the ensemble prediction std exceeds this value, the compound
    is flagged for manual review."""
    retrain_model_path: Optional[str] = None
    """Path to a CSV file ({smiles, ic50}) for active-learning retraining.
    When set, the pipeline will retrain the MetaScorer with the new data."""

    # ── Reporting parameters ──
    # ── Audit / observability ──
    audit_enabled: bool = True
    """When True, track compound dropout reasons and enforce health
    thresholds between pipeline phases."""
    max_dropout_rate: float = 0.5
    """Maximum allowed fraction of compounds lost in a single phase
    before :class:`~autoantibiotic.io_utils.PipelineHealthError` is
    raised.  Default 0.5 (50 %)."""
    audit_output_name: str = "pipeline_audit.json"
    """Filename for the audit summary JSON written during
    :meth:`~autoantibiotic.orchestrator.PipelineOrchestrator._finalize`."""

    csv_report_name: str = "top_candidates.csv"
    html_report_name: str = "report.html"
    pipeline_log_name: str = "pipeline.log"
    scatter_plot_name: str = "energy_vs_selectivity.png"
    qed_histogram_name: str = "qed_histogram.png"
    cache_name: str = "cache.json"
    top_n: int = 10
    top_n_for_active: int = 50
    top_n_for_images: int = 3
    top_n_for_html_report: int = 50

    # ── Derived properties ──
    @property
    def work_dir(self) -> Path:
        return self.output_dir / "workdir"

    @property
    def pdb_dir(self) -> Path:
        return self.output_dir / "pdb"

    def validate_config(self) -> None:
        """Validate the configuration for logical consistency.

        Checks that:
        - All IFP distance thresholds are positive.
        - ML-ADMET thresholds are in [0, 1].
        - All FEP physical parameters are within physically reasonable
          bounds.
        - If ``use_explicit_solvent_mmgbsa`` is True, OpenMM and pdbfixer
          are importable.
        - If ``use_fep_resistance`` is True, OpenMM and openmmtools are
          importable.
        - If ``generative_mode`` is True, the required generative backend
          (at minimum RDKit for the GA backend) is available.

        When ``dry_run`` is True, dependency checks are skipped since
        those features will not actually be executed.

        Raises
        ------
        ConfigurationError
            If a required dependency for an enabled feature is not
            installed, or if a config value is outside physically
            reasonable bounds.
        """
        if self.dry_run:
            return

        # ── Sanity-check IFP distance thresholds ────────────────
        for name, val in [
            ("ifp_hba_dist", self.ifp_hba_dist),
            ("ifp_hbd_dist", self.ifp_hbd_dist),
            ("ifp_hyd_dist", self.ifp_hyd_dist),
            ("ifp_pi_dist", self.ifp_pi_dist),
        ]:
            if val <= 0.0:
                raise ConfigurationError(
                    f"{name} must be > 0, got {val}. "
                    "Distance thresholds must be positive."
                )

        # ── Sanity-check ML-ADMET thresholds ────────────────────
        for name, val in [
            ("ml_admet_herg_threshold", self.ml_admet_herg_threshold),
            ("max_dropout_rate", self.max_dropout_rate),
            ("consensus_vina_weight", self.consensus_vina_weight),
            ("consensus_shape_weight", self.consensus_shape_weight),
            ("ifp_similarity_threshold", self.ifp_similarity_threshold),
            ("fep_ifp_threshold", self.fep_ifp_threshold),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ConfigurationError(
                    f"{name} must be in [0, 1], got {val}."
                )

        # ── Sanity-check FEP physical parameters ────────────────
        if self.fep_nonbonded_cutoff_nm <= 0.0:
            raise ConfigurationError(
                f"fep_nonbonded_cutoff_nm must be > 0, got {self.fep_nonbonded_cutoff_nm}."
            )
        if self.fep_solvent_padding_nm <= 0.0:
            raise ConfigurationError(
                f"fep_solvent_padding_nm must be > 0, got {self.fep_solvent_padding_nm}."
            )
        if self.fep_collision_rate_per_ps <= 0.0:
            raise ConfigurationError(
                f"fep_collision_rate_per_ps must be > 0, got {self.fep_collision_rate_per_ps}."
            )
        if self.fep_pressure_atm <= 0.0:
            raise ConfigurationError(
                f"fep_pressure_atm must be > 0, got {self.fep_pressure_atm}."
            )
        if self.fep_ionic_strength_molar < 0.0:
            raise ConfigurationError(
                f"fep_ionic_strength_molar must be >= 0, got {self.fep_ionic_strength_molar}."
            )
        if self.fep_min_samples_mbar < 10:
            raise ConfigurationError(
                f"fep_min_samples_mbar must be >= 10, got {self.fep_min_samples_mbar}."
            )
        if self.fep_max_heavy_atoms < 1:
            raise ConfigurationError(
                f"fep_max_heavy_atoms must be >= 1, got {self.fep_max_heavy_atoms}."
            )
        if self.fep_max_smiles_length < 1:
            raise ConfigurationError(
                f"fep_max_smiles_length must be >= 1, got {self.fep_max_smiles_length}."
            )
        if self.fep_top_n_strict < 1:
            raise ConfigurationError(
                f"fep_top_n_strict must be >= 1, got {self.fep_top_n_strict}."
            )
        if self.fep_ewald_error_tolerance <= 0.0:
            raise ConfigurationError(
                f"fep_ewald_error_tolerance must be > 0, got {self.fep_ewald_error_tolerance}."
            )

        # ── Sanity-check box sizes are positive ────────────────
        for name, box in [
            ("allosteric_box_size", self.allosteric_box_size),
            ("active_box_size", self.active_box_size),
            ("offtarget_box_size", self.offtarget_box_size),
            ("redocking_box_size", self.redocking_box_size),
        ]:
            if any(d <= 0 for d in box):
                raise ConfigurationError(
                    f"All dimensions of {name} must be > 0, got {box}."
                )

        # ── Sanity-check dynamic box padding ────────────────────
        if self.dynamic_box_padding <= 0.0:
            raise ConfigurationError(
                f"dynamic_box_padding must be > 0, got {self.dynamic_box_padding}."
            )

        if self.use_explicit_solvent_mmgbsa and self.mmgbsa_solvent_model != "explicit":
            logging.getLogger("AutoAntibiotic").warning(
                "DEPRECATED: use_explicit_solvent_mmgbsa=True is deprecated. "
                "Set mmgbsa_solvent_model='explicit' to use the explicit-solvent "
                "path. Currently mmgbsa_solvent_model='%s', so the implicit OBC2 "
                "path will be used.",
                self.mmgbsa_solvent_model,
            )

        if self.mmgbsa_solvent_model == "explicit" or self.use_explicit_solvent_mmgbsa:
            try:
                import openmm  # noqa: F401
            except ImportError:
                raise ConfigurationError(
                    "Explicit solvent MM-GB/SA requested (mmgbsa_solvent_model='explicit') "
                    "but OpenMM is not installed. Please install via conda:\n"
                    "  conda install -c conda-forge openmm"
                )
            try:
                import pdbfixer  # noqa: F401
            except ImportError:
                self.use_explicit_solvent_mmgbsa = False
                logging.getLogger("AutoAntibiotic").warning(
                    "Explicit-solvent MM-GB/SA disabled: pdbfixer not installed. "
                    "Falling back to implicit-solvent (OBC2) heuristic. "
                    "Install via: conda install -c conda-forge pdbfixer"
                )

        if self.use_fep_resistance:
            try:
                import openmm  # noqa: F401
            except ImportError:
                raise ConfigurationError(
                    "FEP resistance profiling requested (use_fep_resistance=True) "
                    "but OpenMM is not installed. OpenMM is required for molecular "
                    "mechanics force field evaluation. Please install via conda:\n"
                    "  conda install -c conda-forge openmm"
                )
            try:
                import openmmtools  # noqa: F401
            except ImportError:
                self.use_fep_resistance = False
                logging.getLogger("AutoAntibiotic").warning(
                    "FEP disabled: openmmtools not found. "
                    "Falling back to heuristic resistance profiling. "
                    "Install via: conda install -c conda-forge openmmtools"
                )
            try:
                import openmmforcefields  # noqa: F401
            except ImportError:
                self.use_fep_resistance = False
                logging.getLogger("AutoAntibiotic").warning(
                    "FEP disabled: openmmforcefields not found. "
                    "Falling back to heuristic resistance profiling. "
                    "Install via: conda install -c conda-forge openmmforcefields"
                )

        if self.generative_mode:
            try:
                from rdkit import Chem  # noqa: F401
            except ImportError:
                raise ConfigurationError(
                    "Generative mode requested (generative_mode=True) "
                    "but RDKit is not installed. Please install via conda:\n"
                    "  conda install -c conda-forge rdkit"
                )
            # Also check for the GA backend dependencies
            try:
                from rdkit.Chem import BRICS  # noqa: F401
            except ImportError:
                raise ConfigurationError(
                    "Generative mode requested (generative_mode=True) "
                    "but RDKit BRICS module is not available. "
                    "Please upgrade RDKit:\n"
                    "  conda install -c conda-forge rdkit"
                )


CONFIG = PipelineConfig()

np.random.seed(CONFIG.random_seed)
random.seed(CONFIG.random_seed)
