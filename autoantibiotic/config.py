from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem


@dataclass
class ToolResult:
    """Result from an external tool execution."""
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class PipelineConfig:
    """Configuration for the AutoAntibiotic discovery pipeline."""
    random_seed: int = 42
    pdb_ids: Dict[str, str] = field(default_factory=lambda: {
        "PBP2a_apo": "3QPD",
        "PBP2a_holo": "6TKO",
        "trypsin": "1UTN",
        "CES1": "3KJZ",
    })
    reference_antibiotics: Dict[str, str] = field(default_factory=lambda: {
        "Methicillin":  "CC1=C(C(=C(C(=C1O)OC)OC)OC)C(=O)NC2C3C(C(=O)N3C2=O)SC4(C)C",
        "Vancomycin":   "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C(=C4OC6=C(C(=CC(=C6)C(C(=O)NC(C(=O)NC5C(=O)O)CC7=CC=C(C=C7)O)NC(=O)C8C(O)C(=C(C=C8)Cl)O)O)O)CO)O)O)O)NC(=O)C9C(O)C(=C(C=C9)Cl)O)(CC(=O)N)O",
        "Ceftaroline":  "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "Meropenem":    "CC1C2C(C(=O)N2C(=C1SC3CC(NCC3)C(=O)O)C(=O)O)(C)O",
        "Oxacillin":    "CC1=C(C(=NO1)C2=CC=CC=C2)C(=O)NC3C4C(C(=O)N4C3=O)SC5(C)C",
    })
    beta_lactam_smarts: str = "[C;H1,D3]1[C;H0,D3](=[O;D1])[N;H1,D2][C;H1,D3]1"
    allosteric_residues: List[str] = field(default_factory=lambda: ["ALA237", "MET241", "TYR159"])
    active_site_residues: List[str] = field(default_factory=lambda: ["SER403"])
    trypsin_active_site_residues: List[str] = field(default_factory=lambda: ["HIS57", "ASP102", "SER195"])
    ces1_active_site_residues: List[str] = field(default_factory=lambda: ["SER221", "HIS468", "GLU354"])
    allosteric_box_size: Tuple[float, float, float] = (15.0, 15.0, 15.0)
    active_box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    offtarget_box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    redocking_box_size: Tuple[float, float, float] = (25.0, 25.0, 25.0)
    vina_exhaustiveness: int = 8
    vina_num_modes: int = 3
    vina_timeout_s: int = 120
    job_timeout_s: int = 180
    prepare_receptor_timeout: int = 60
    n_jobs: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    similarity_threshold: float = 0.4
    similarity_threshold_relaxed: float = 0.5
    diversity_min_count: int = 100
    selectivity_index_threshold: float = 2.0
    library_target_count: int = 500
    brics_min_fragment_size: int = 8
    output_dir: Path = Path("output")
    top_n: int = 10
    qed_threshold: float = 0.6
    lipinski_mw_max: float = 500.0
    lipinski_logp_max: float = 5.0
    lipinski_hbd_max: int = 5
    lipinski_hba_max: int = 10
    redocking_rmsd_cutoff: float = 2.0
    sa_score_threshold: float = 6.0
    shape_score_norm_factor: float = 0.05
    diversity_pool_multiplier: int = 5
    top_n_for_active: int = 50
    top_n_for_images: int = 3
    top_n_for_html_report: int = 50
    batch_size_docking: int = 75
    library_generator_threshold: int = 1000
    morgan_radius: int = 2
    morgan_nbits: int = 2048
    pdb_retry_max_attempts: int = 3
    pdb_retry_base_delay: float = 2.0
    obabel_timeout_s: int = 60
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
        "[1*]c1ccccc1",
        "[1*]c1ccc(O)cc1",
        "[1*]c1ccc(Cl)cc1",
        "[1*]c1ccc(F)cc1",
        "[1*]c1ccc(Br)cc1",
        "[1*]c1ccc(OC)cc1",
        "[1*]c1ccc(C(=O)O)cc1",
        "[1*]c1ccc(N)cc1",
        "[1*]c1ccc(C)cc1",
        "[1*]c1ccc(C(C)C)cc1",
        "[1*]c1ccc(CF)cc1",
        "[1*]c1ccc(CN)cc1",
        "[1*]c1ccc(S(=O)(=O)N)cc1",
        "[1*]c1ccc(C(=O)N)cc1",
        "[1*]c1ccc(NC(=O)C)cc1",
        "[1*]CC(=O)O",
        "[1*]CCO",
        "[1*]CCN",
        "[1*]CC(=O)N",
        "[1*]CCC(=O)O",
        "[3*]C=Cc1ccccc1",
        "[3*]C=Cc1ccc(O)cc1",
        "[3*]C=Cc1ccc(Cl)cc1",
        "[3*]CCN(C)C",
        "[5*]Nc1ccccc1",
        "[5*]Nc1ccc(O)cc1",
        "[5*]Nc1ccc(C(=O)O)cc1",
        "[5*]Nc1ccc(Cl)cc1",
        "[5*]Nc1ccc(F)cc1",
        "[5*]Nc1ccc(OC)cc1",
        "[5*]Nc1ccc(C)cc1",
        "[5*]Nc1ccc(Br)cc1",
        "[5*]Nc1ccc(CN)cc1",
        "[5*]NCC",
        "[5*]NCCO",
        "[5*]NCCC(=O)O",
        "[6*]C(=O)O",
        "[6*]C(=O)c1ccccc1",
        "[6*]C(=O)c1ccc(O)cc1",
        "[6*]C(=O)c1ccc(Cl)cc1",
        "[6*]C(=O)c1ccc(OC)cc1",
        "[6*]C(=O)c1ccc(C)cc1",
        "[6*]C(=O)c1ccc(N)cc1",
        "[6*]C(=O)CC",
        "[7*]Cc1ccccc1",
        "[7*]Cc1ccc(O)cc1",
        "[7*]Cc1ccc(O)c(OC)c1",
        "[7*]Cc1ccc(OC)cc1",
        "[7*]Cc1ccc(Cl)cc1",
        "[7*]Cc1ccc(F)cc1",
        "[7*]CC",
        "[7*]C(C)C",
        "[16*]c1ccccc1OC",
        "[16*]c1ccc(C)cc1",
        "[16*]c1ccc(N)cc1",
        "[16*]c1ccc(O)cc1",
    ])
    control_smiles: Dict[str, str] = field(default_factory=lambda: {
        "Ceftaroline": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "Meropenem": "CC1C2C(C(=O)N2C(=C1SC3CC(NCC3)C(=O)O)C(=O)O)(C)O",
    })
    conserved_residues: set = field(default_factory=lambda: {"SER403", "KYS406", "TYR446"})
    mutable_residues: set = field(default_factory=lambda: {"G246", "N146"})
    dry_run: bool = False

    # Analysis thresholds (Phase 4)
    resistance_energy_active_threshold: float = -6.0
    resistance_energy_allosteric_threshold: float = -7.0
    resistance_mw_threshold: float = 400.0
    resistance_rot_threshold: int = 5
    resistance_qed_threshold: float = 0.8

    # Consensus scoring (Phase 3)
    consensus_vina_weight: float = 0.7
    consensus_shape_weight: float = 0.3

    # Report filenames
    csv_report_name: str = "top_candidates.csv"
    html_report_name: str = "report.html"
    pipeline_log_name: str = "pipeline.log"
    scatter_plot_name: str = "energy_vs_selectivity.png"
    qed_histogram_name: str = "qed_histogram.png"

    # Cache
    cache_db_name: str = "cache.db"

    @property
    def work_dir(self) -> Path:
        return self.output_dir / "workdir"

    @property
    def pdb_dir(self) -> Path:
        return self.output_dir / "pdb"


CONFIG = PipelineConfig()


def _merge_yaml_overrides() -> None:
    """Load overrides from ``config.yaml`` in the output directory root.

    If a ``config.yaml`` file exists alongside the ``output/`` directory,
    its values are merged into the global ``CONFIG`` instance.  Only
    top-level dataclass fields are supported.
    """
    yaml_path = Path("config.yaml")
    if not yaml_path.exists():
        yaml_path = CONFIG.output_dir.parent / "config.yaml"
    if not yaml_path.exists():
        return

    try:
        import yaml
    except ImportError:
        log = logging.getLogger("AutoAntibiotic")
        log.warning("  ⚠  config.yaml found but PyYAML is not installed.  Run: pip install pyyaml")
        return

    try:
        with open(yaml_path) as f:
            overrides = yaml.safe_load(f)
    except Exception as exc:
        log = logging.getLogger("AutoAntibiotic")
        log.warning(f"  ⚠  Failed to parse config.yaml: {exc}")
        return

    if not isinstance(overrides, dict):
        return

    log = logging.getLogger("AutoAntibiotic")
    applied = 0
    for key, value in overrides.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
            applied += 1
            log.info(f"  Config override: {key} = {value}")
        else:
            log.warning(f"  ⚠  Unknown config key '{key}' in config.yaml — skipped.")

    if applied > 0:
        log.info(f"  Applied {applied} config override(s) from {yaml_path}")


_merge_yaml_overrides()

np.random.seed(CONFIG.random_seed)
random.seed(CONFIG.random_seed)


@dataclass
class CompoundRecord:
    """Stores all computed properties for a single candidate."""
    compound_id: str
    smiles: str
    mol: Optional[Chem.Mol] = None

    pb2pa_allosteric_energy: Optional[float] = None
    pb2pa_active_energy: Optional[float] = None
    human_trypsin_energy: Optional[float] = None
    human_ces1_energy: Optional[float] = None

    selectivity_index: Optional[float] = None

    max_similarity: float = 0.0

    passes_lipinski: bool = False
    qed_score: float = 0.0
    passes_pains: bool = False

    resistance_notes: str = ""

    shape_score: Optional[float] = None
