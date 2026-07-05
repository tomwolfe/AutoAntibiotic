from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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

    # Benchmark
    benchmark_mode: bool = False
    reference_actives_path: Optional[Path] = None
    reference_inactives_path: Optional[Path] = None
    benchmark_n_decoys: int = 100

    # Library data
    beta_lactam_smarts: str = "[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1"
    allosteric_residues: List[str] = field(default_factory=lambda: ["ALA237", "MET241", "TYR159"])
    active_site_residues: List[str] = field(default_factory=lambda: ["SER403"])
    trypsin_active_site_residues: List[str] = field(default_factory=lambda: ["HIS57", "ASP102", "SER195"])
    ces1_active_site_residues: List[str] = field(default_factory=lambda: ["SER221", "HIS468", "GLU354"])
    conserved_residues: set = field(default_factory=lambda: {"SER403", "KYS406", "TYR446"})
    mutable_residues: set = field(default_factory=lambda: {"G246", "N146"})
    use_pharmacophore_filter: bool = True

    pdb_ids: Dict[str, str] = field(default_factory=lambda: {
        "PBP2a_apo": "3QPD",
        "PBP2a_holo": "6TKO",
        "trypsin": "1UTN",
        "CES1": "3KJZ",
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
    default_ensemble_pdb_ids: List[str] = field(default_factory=lambda: ["3QPD", "6TKO", "4CJN"])
    consensus_scoring_method: str = "rank"
    flexible_docking: bool = False
    flexible_residues_allosteric: List[str] = field(default_factory=lambda: ["ALA237", "MET241", "TYR159"])
    flexible_residues_active: List[str] = field(default_factory=lambda: ["SER403"])
    max_flexible_conformers: int = 9
    use_ml_rescoring: bool = True
    use_mm_gbsa: bool = False
    use_mm_gbsa_rescoring: bool = False
    mm_gbsa_top_n: int = 50
    key_interaction_residues_allosteric: List[str] = field(default_factory=lambda: ["TYR159", "ALA237", "MET241"])
    key_interaction_residues_active: List[str] = field(default_factory=lambda: ["SER403"])
    min_key_interactions: int = 1

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

    # ── Water analysis parameters ──
    use_water_analysis: bool = True
    water_distance_cutoff: float = 5.0
    water_displacement_energy_threshold: float = 2.5

    # ── Reporting parameters ──
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


CONFIG = PipelineConfig()

np.random.seed(CONFIG.random_seed)
random.seed(CONFIG.random_seed)
