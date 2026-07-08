from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
import statistics
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

import rdkit
if not hasattr(rdkit, "six"):
    import sys as _sys
    import io as _io
    _six_mod = type(_sys)("rdkit.six")
    _six_mod.StringIO = _io.StringIO
    _sys.modules["rdkit.six"] = _six_mod
    rdkit.six = _six_mod
    del _sys, _io, _six_mod

from .config import CONFIG, ConfigurationError
from .models import CompoundRecord
from .io_utils import (
    AutoAntibioticError,
    DockingParseError,
    DockingResultValidator,
    GninaError,
    PipelineAudit,
    VinaError,
    OpenBabelError,
    log,
    make_cache_key,
    parse_gnina_energy,
    parse_vina_energy,
    run_tool,
    safe_run_tool,
)

try:
    from .water_analysis import WaterAnalysisResult
except ImportError:
    WaterAnalysisResult = None  # type: ignore

try:
    from Bio.PDB import PDBIO, PDBParser
    from Bio.PDB import is_aa as _is_aa_pdb
    _HAVE_BIOPDB = True
except ImportError:
    _HAVE_BIOPDB = False

try:
    from .structure_prep import clean_pdb_structure, calculate_adaptive_box_size, get_ligand_max_dimension
    _HAVE_CLEAN = True
except ImportError:
    _HAVE_CLEAN = False

try:
    from .ml_scoring import rescore_with_ml as _rescore_with_ml
    _HAVE_ML_SCORING = True
except ImportError:
    _HAVE_ML_SCORING = False
    _rescore_with_ml = None

_CacheLike = Optional[Dict[str, float]]

try:
    from tqdm import tqdm as _tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False
    _tqdm = lambda x, **kw: x

_DOCKING_BINARY_VALIDATED: bool = False
"""Tracking sentinel so binary health check runs at most once per session."""


def _extract_native_ligand_from_holo(
    holo_pdb_path: str,
    output_ligand_smi: str,
    output_ligand_pdbqt: str,
) -> Optional[str]:
    """Parse the holo structure, locate the co-crystallised ligand,
    write its SMILES to *output_ligand_smi* and its PDBQT to *output_ligand_pdbqt*."""
    holo_pdb_id = CONFIG.pdb_ids["PBP2a_holo"]
    try:
        from Bio.PDB import PDBIO, PDBParser, Select

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(holo_pdb_id, holo_pdb_path)

        ligand_residues: list = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    het_flag = residue.get_id()[0]
                    if het_flag == " " or het_flag == "W":
                        continue
                    resname = residue.get_resname().strip()
                    if resname in ("HOH", "WAT", "SOL"):
                        continue
                    ligand_residues.append((chain.get_id(), residue))

        if not ligand_residues:
            log.warning(f"  ⚠  No hetero-ligand found in {holo_pdb_id}.")
            return None

        chain_id, lig_res = ligand_residues[0]
        log.info(f"  Native ligand found: chain {chain_id}, residue {lig_res.get_resname()}")

        pdbio = PDBIO()

        class LigSelect(Select):
            def accept_residue(self, residue):
                return residue is lig_res

        pdbio.set_structure(struct)
        lig_pdb = output_ligand_pdbqt.replace(".pdbqt", ".pdb")
        pdbio.save(lig_pdb, LigSelect())

        mol = Chem.MolFromPDBFile(lig_pdb, removeHs=False)
        if mol is None:
            log.warning("  ⚠  RDKit could not read ligand PDB, trying obabel…")
            smi_file = output_ligand_smi
            try:
                safe_run_tool(["obabel", lig_pdb, "-O", smi_file], timeout=CONFIG.obabel_timeout_s)
                with open(smi_file) as f:
                    smi = f.readline().strip()
                if smi:
                    return smi
            except (RuntimeError, OSError, AutoAntibioticError):
                pass
            return None

        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)

        with open(output_ligand_smi, "w") as f:
            f.write(smi + "\n")
        log.info(f"  Native ligand SMILES: {smi}")

        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)[0]
            pdbqt_str = PDBQTWriterLegacy.write_string(mol_setup)[0]
            with open(output_ligand_pdbqt, "w") as f:
                f.write(pdbqt_str)
            log.info(f"  Native ligand PDBQT written to {output_ligand_pdbqt}")
        except Exception as exc:
            log.warning(f"  ⚠  Meeko prep failed for native ligand: {exc}")
            shutil.copy(lig_pdb, output_ligand_pdbqt)

        return smi

    except Exception as exc:
        log.error(f"  ✗  Native ligand extraction failed: {exc}")
        return None


def _compute_rmsd_docked_vs_crystal(
    docked_pdb: str, crystal_pdb: str
) -> Optional[float]:
    """Align protein Cα backbones and compute heavy-atom RMSD of the ligand."""
    try:
        from Bio.PDB import PDBParser, Superimposer

        parser = PDBParser(QUIET=True)
        docked_struct = parser.get_structure("docked", docked_pdb)
        crystal_struct = parser.get_structure("crystal", crystal_pdb)

        def _get_ca_atoms(structure):
            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == " " and "CA" in residue:
                            atoms.append(residue["CA"])
            return atoms

        docked_ca = _get_ca_atoms(docked_struct)
        crystal_ca = _get_ca_atoms(crystal_struct)

        if len(docked_ca) < 3 or len(crystal_ca) < 3:
            log.warning("  ⚠  Too few Cα atoms for backbone alignment (< 3).")
            return None

        sup = Superimposer()
        docked_coords = np.array([a.get_vector().get_array() for a in docked_ca])
        crystal_coords = np.array([a.get_vector().get_array() for a in crystal_ca])
        sup.set(crystal_coords, docked_coords)
        sup.run()
        rot, tran = sup.rotran
        log.info(f"  Backbone alignment RMSD: {sup.rmsd:.3f} Å")

        def _get_ligand_atoms(structure):
            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] != " ":
                            for atom in residue:
                                if atom.element != "H":
                                    atoms.append(atom)
            return atoms

        docked_lig = _get_ligand_atoms(docked_struct)
        crystal_lig = _get_ligand_atoms(crystal_struct)

        if not docked_lig or not crystal_lig:
            log.warning("  ⚠  No ligand atoms found in one or both structures.")
            return None

        if len(docked_lig) != len(crystal_lig):
            log.warning(
                f"  ⚠  Ligand atom count mismatch: docked={len(docked_lig)}, "
                f"crystal={len(crystal_lig)}. Truncating to shorter list."
            )
            n = min(len(docked_lig), len(crystal_lig))
            docked_lig = docked_lig[:n]
            crystal_lig = crystal_lig[:n]

        docked_lig_coords = np.array([a.get_vector().get_array() for a in docked_lig])
        aligned_docked = docked_lig_coords @ rot.T + tran
        crystal_lig_coords = np.array([a.get_vector().get_array() for a in crystal_lig])
        diff = aligned_docked - crystal_lig_coords
        rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        return rmsd

    except Exception as exc:
        log.error(f"  ✗  RMSD calculation failed: {exc}")
        return None


def run_redocking_validation(
    holo_pdb_path: str,
    target_pdbqt_path: str,
    work_dir: str,
    deps: Dict[str, Any],
    center: Optional[np.ndarray] = None,
) -> Tuple[bool, Optional[float]]:
    """Phase 0 — Protocol Validation.

    Extracts the native ligand from 3ZG0, docks it back into the prepared
    PBP2a receptor, and computes the RMSD to the crystal pose.

    Returns (success: bool, rmsd: float | None).
    """
    log.info("─── Phase 0: Redocking Validation ───")

    lig_smi = os.path.join(work_dir, "native_ligand.smi")
    lig_pdbqt = os.path.join(work_dir, "native_ligand.pdbqt")
    docked_pdb = os.path.join(work_dir, "native_docked.pdb")

    smi = _extract_native_ligand_from_holo(holo_pdb_path, lig_smi, lig_pdbqt)
    if smi is None:
        log.warning("  ⚠  Could not extract native ligand. Skipping redocking validation.")
        return False, None

    if not deps.get("USE_VINA", False):
        log.warning("  ⚠  Vina unavailable. Redocking validation requires Vina. Skip.")
        return False, None

    log.info("  Redocking native ligand into PBP2a…")
    docked_pdbqt = docked_pdb.replace(".pdb", ".pdbqt")
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    bx, by, bz = CONFIG.redocking_box_size
    vina_cmd = [
        "vina",
        "--receptor", target_pdbqt_path,
        "--ligand", lig_pdbqt,
        "--out", docked_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{bx:.1f}", "--size_y", f"{by:.1f}", "--size_z", f"{bz:.1f}",
        "--exhaustiveness", str(CONFIG.vina_exhaustiveness),
    ]

    try:
        safe_run_tool(vina_cmd, timeout=CONFIG.vina_timeout_s, ignore_stderr_warnings=True)
    except (RuntimeError, VinaError, GninaError, AutoAntibioticError) as exc:
        log.warning(f"  Vina redocking failed: {exc}")
        return False, None

    try:
        safe_run_tool(
            ["obabel", docked_pdbqt, "-O", docked_pdb, "--gen3d"],
            timeout=CONFIG.obabel_timeout_s,
        )
    except (RuntimeError, OpenBabelError, AutoAntibioticError):
        log.warning("  Could not convert docked PDBQT to PDB. Trying RDKit PDBQT reader.")
        mol = Chem.MolFromPDBQT(docked_pdbqt) if hasattr(Chem, "MolFromPDBQT") else None
        if mol is None:
            log.warning("  ⚠  Cannot parse docked PDBQT. RMSD not computed.")
            return False, None
        Chem.MolToPDBFile(mol, docked_pdb)

    crystal_pdb = lig_pdbqt.replace(".pdbqt", ".pdb")
    rmsd = _compute_rmsd_docked_vs_crystal(docked_pdb, crystal_pdb)

    if rmsd is None:
        log.warning("  ⚠  RMSD could not be computed.")
        return False, None

    log.info(f"  Redocking RMSD = {rmsd:.3f} Å")
    cutoff = CONFIG.redocking_rmsd_cutoff
    if rmsd > cutoff:
        log.warning(
            f"  ⚠  Redocking RMSD ({rmsd:.3f} Å) exceeds {cutoff} Å threshold. "
            "The docking protocol may not accurately reproduce known binding modes. "
            "Proceeding with pipeline — interpret results with caution."
        )
    else:
        log.info(f"  ✓  Redocking validated (RMSD = {rmsd:.3f} Å ≤ {cutoff} Å).")

    return (rmsd <= cutoff if rmsd is not None else False), rmsd


def prepare_ligand_pdbqt(
    mol: Chem.Mol,
    output_path: str,
) -> bool:
    """Convert an RDKit Mol to PDBQT via Meeko.

    Attempts conversion using Meeko's MoleculePreparation and
    PDBQTWriterLegacy.  If Meeko fails, falls back to a minimal PDBQT
    writer that assigns Gasteiger charges and writes a rigid (TORSDOF 0)
    PDBQT entry.

    Args:
        mol: RDKit molecule with at least one conformer.
        output_path: Path for the output PDBQT file.

    Returns:
        True on success, False if all conversion methods failed.
    """
    try:
        if mol.GetNumAtoms() > 150 or mol.GetNumHeavyAtoms() > 100:
            log.debug("Molecule too large for docking")
            return False

        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            mol_3d = mol
            if mol_3d.GetNumConformers() == 0:
                mol_3d = Chem.RWMol(mol)
                mol_3d = Chem.AddHs(mol_3d)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)

            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol_3d)
            if not mol_setups:
                return False
            pdbqt_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
            with open(output_path, "w") as f:
                f.write(pdbqt_str)
            return True
        except Exception as exc:
            log.warning(f"  Meeko prep failed ({exc}), trying RDKit fallback…")
            try:
                mol_tmp = Chem.RWMol(mol)
                mol_tmp = Chem.AddHs(mol_tmp, addCoords=True)
                if mol_tmp.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(mol_tmp, randomSeed=42)
                AllChem.ComputeGasteigerCharges(mol_tmp)

                _ad_type_map = {
                    "C": "C", "c": "C",
                    "N": "N", "n": "N",
                    "O": "O", "o": "O",
                    "S": "S", "s": "S",
                    "P": "P", "p": "P",
                    "F": "F", "f": "F",
                    "Cl": "Cl", "Br": "Br",
                    "I": "I",
                    "H": "H",
                }

                conf = mol_tmp.GetConformer()
                lines = ["ROOT\n"]
                for i, atom in enumerate(mol_tmp.GetAtoms()):
                    pos = conf.GetAtomPosition(i)
                    charge = atom.GetDoubleProp("_GasteigerCharge")
                    elem = atom.GetSymbol()
                    ad_type = _ad_type_map.get(elem, "C")
                    atom_name = f" {elem:<3s}"[:4]
                    lines.append(
                        f"ATOM  {i+1:>5d} {atom_name} LIG X   1    "
                        f"{pos.x:>8.3f}{pos.y:>8.3f}{pos.z:>8.3f}  0.00  0.00"
                        f"{charge:>10.4f} {ad_type:<2s}\n"
                    )
                lines.append("ENDROOT\n")
                lines.append("TORSDOF 0\n")
                with open(output_path, "w") as f:
                    f.writelines(lines)
                return True
            except Exception as exc2:
                log.warning(f"  RDKit PDBQT fallback also failed: {exc2}")
                return False
    except Exception as exc3:
        log.warning(f"  Ligand preparation failed unexpectedly: {exc3}")
        return False


def _run_docking_tool(
    tool_name: str,
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    timeout: int = CONFIG.vina_timeout_s,
) -> Optional[float]:
    """Run a single docking job via the external tool wrapper.

    Builds the command-line invocation for the given docking tool
    (*vina* or *gnina*) with the specified receptor, ligand, search-box
    centre and dimensions, then parses the score from the tool's output.

    Args:
        tool_name: ``"vina"`` or ``"gnina"``.
        receptor_pdbqt: Path to the receptor PDBQT file.
        ligand_pdbqt: Path to the ligand PDBQT file.
        output_pdbqt: Path to write the docked-pose PDBQT file.
        center: 3-element array of (x, y, z) box centre coordinates.
        box_size: Tuple of (x, y, z) box dimensions in Ångström.
        timeout: Maximum wall-clock seconds for the subprocess.

    Returns:
        Best binding energy (kcal/mol) for vina, CNNscore (0–1) for
        gnina, or None if docking failed or timed out.
    """
    if CONFIG.dry_run:
        if tool_name == "gnina":
            return float(np.random.uniform(0.5, 0.95))
        return float(np.random.uniform(-10.0, -5.0))

    binary = CONFIG.gnina_binary_path if tool_name == "gnina" else "vina"

    cmd = [
        binary,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", output_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.1f}",
        "--size_y", f"{box_size[1]:.1f}",
        "--size_z", f"{box_size[2]:.1f}",
        "--exhaustiveness", str(CONFIG.vina_exhaustiveness),
        "--num_modes", str(CONFIG.vina_num_modes),
    ]

    validator = DockingResultValidator()

    # One-time binary health check at startup
    global _DOCKING_BINARY_VALIDATED
    if (
        CONFIG.validate_docking_binaries_on_startup
        and not _DOCKING_BINARY_VALIDATED
    ):
        try:
            version_result = safe_run_tool(
                [binary, "--version"], timeout=10, check=False,
            )
            version_out = version_result.stdout or version_result.stderr
            if not validator.validate_binary_health(tool_name, version_out):
                raise ConfigurationError(
                    f"{binary} version check failed. "
                    f"Expected Vina 1.2.x or GNINA 1.x, got: "
                    f"{version_out.strip()!r}"
                )
            log.info(f"  ✓  {binary} binary health validated.")
        except (RuntimeError, OSError, AutoAntibioticError) as exc:
            raise ConfigurationError(
                f"Cannot run {binary} for version check: {exc}"
            )
        _DOCKING_BINARY_VALIDATED = True

    try:
        result = safe_run_tool(cmd, timeout=timeout, check=False, ignore_stderr_warnings=True)
        if result.returncode != 0:
            log.warning(f"  {binary} error: {result.stderr.strip()}")
            raise DockingParseError(
                f"{binary} returned non-zero exit code {result.returncode}. "
                f"stderr: {result.stderr.strip()}"
            )
        if tool_name == "gnina":
            score = validator.parse_gnina(result.stdout)
            if score is not None:
                return score
            score = validator.parse_gnina(result.stderr)
            if score is not None:
                return score
            raise DockingParseError(
                f"{binary} output did not contain a valid CNNscore/CNNaffinity."
            )
        else:
            energy = validator.parse_vina(result.stdout)
            if energy is not None:
                return energy
            energy = validator.parse_vina(result.stderr)
            if energy is not None:
                return energy
            raise DockingParseError(
                f"{binary} output did not contain a valid binding energy."
            )
    except (RuntimeError, VinaError, GninaError, AutoAntibioticError) as exc:
        log.warning(f"  {binary} execution failed: {exc}")
        if isinstance(exc, DockingParseError):
            raise
        if isinstance(exc, (VinaError, GninaError)):
            raise DockingParseError(str(exc)) from exc
        return None


def dock_compound(
    record: CompoundRecord,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "",
    cache: _CacheLike = None,
    use_cache: bool = False,
) -> Tuple[Optional[float], str]:
    """Full docking pipeline for a single compound: PDBQT prep → dock → parse.

    Returns ``(energy, method)`` where *method* is ``"GNINA"``, ``"Vina"``,
    ``"None"``, or ``"Unknown"`` (cache hit).
    """
    tool_name = "gnina" if CONFIG.use_gnina else "vina"
    cache_key = make_cache_key(record.smiles, tool_name)
    if use_cache and cache is not None and cache_key in cache:
        return cache[cache_key], "Unknown"

    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            return None, "None"
        record.mol = mol

    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    lig_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_lig.pdbqt")
    out_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_out.pdbqt")

    if not prepare_ligand_pdbqt(record.mol, lig_pdbqt):
        return None, "PrepFailure"

    method = "None"
    try:
        energy = _run_docking_tool(tool_name, receptor_pdbqt, lig_pdbqt, out_pdbqt, center, box_size)
        method = "GNINA" if tool_name == "gnina" else "Vina"
    except DockingParseError:
        energy = None

    if energy is None and CONFIG.use_gnina:
        log.warning("  GNINA docking failed, falling back to Vina.")
        try:
            energy = _run_docking_tool("vina", receptor_pdbqt, lig_pdbqt, out_pdbqt, center, box_size)
            method = "Vina"
        except DockingParseError:
            energy = None

    # Keep out_pdbqt on disk for downstream IFP analysis.
    for f in (lig_pdbqt,):
        try:
            os.remove(f)
        except OSError:
            pass

    if use_cache and cache is not None:
        cache[cache_key] = energy

    return energy, method


def _compute_rank_consensus(energies_list: List[List[float]]) -> List[float]:
    """Rank-based consensus scoring across multiple receptors.

    For each receptor, compounds are ranked by binding energy (lower =
    better, rank 1 = best).  The ranks are then averaged across all
    receptors.

    Args:
        energies_list: ``[receptor_idx][compound_idx]`` matrix of
            docking energies.  All inner lists must have the same length
            and contain no ``None`` values.

    Returns:
        Average rank for each compound (lower is better).
    """
    if not energies_list or not energies_list[0]:
        return []

    n_compounds = len(energies_list[0])
    rank_sums = [0.0] * n_compounds

    for receptor_energies in energies_list:
        indexed = list(enumerate(receptor_energies))
        indexed.sort(key=lambda x: x[1])
        for rank, (orig_idx, _) in enumerate(indexed, 1):
            rank_sums[orig_idx] += rank

    n_receptors = len(energies_list)
    return [s / n_receptors for s in rank_sums]


def dock_compound_ensemble(
    record: CompoundRecord,
    receptor_pdbqt_list: List[str],
    center_list: List[np.ndarray],
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "",
) -> Tuple[Optional[float], str]:
    """Dock a compound against an ensemble of receptor structures.

    Each receptor structure is docked independently.  The final score
    is aggregated via ``CONFIG.consensus_scoring_method`` ("mean",
    "median", "min", or "rank").

    .. note::
       The ``"rank"`` method requires energies across *all* compounds
       to compute per-receptor rankings.  When called for a single
       compound, rank consensus cannot be computed and ``mean`` is
       returned as a fallback.  Use :func:`_compute_rank_consensus`
       in batch mode for proper rank-based consensus.

    Returns ``(consensus_score, method)``.
    """
    energies: List[float] = []
    method = "GNINA" if CONFIG.use_gnina else "Vina"
    for i, (rec_pdbqt, ctr) in enumerate(zip(receptor_pdbqt_list, center_list)):
        e, _ = dock_compound(
            record, rec_pdbqt, ctr, box_size,
            work_dir, f"{tag}_ens{i}",
        )
        if e is not None:
            energies.append(e)

    if not energies:
        return None, "None"

    consensus = CONFIG.consensus_scoring_method
    if consensus == "min":
        return min(energies), method
    elif consensus == "median":
        return statistics.median(energies), method
    elif consensus == "rank":
        return statistics.mean(energies), method
    else:
        return statistics.mean(energies), method


# ── Flexible residue docking helpers ─────────────────────────────

_CHI_DEFS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "VAL": [("N", "CA", "CB", "CG1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
}


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for a rotation of *angle_rad* around
    *axis* (unit vector) via Rodrigues' formula."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def _collect_descendant_atoms(residue, atom_name: str, excluded: set) -> List[str]:
    """Collect all atom names in *residue* that are topologically
    'beyond' *atom_name* (i.e. reachable without going through
    *excluded*)."""
    if atom_name not in residue:
        return []
    from Bio.PDB import NeighborSearch as _NS

    atom = residue[atom_name]
    visited = set(excluded)
    descendants: List[str] = []

    def _dfs(current, parent_name):
        if current.get_name() in visited:
            return
        visited.add(current.get_name())
        if current.get_name() != parent_name:
            descendants.append(current.get_name())
        for bonded in current.neighbors:
            bonded_name = bonded.get_name()
            if bonded_name not in visited and bonded_name in residue:
                _dfs(bonded, current.get_name())

    _dfs(atom, atom_name)
    return descendants


def _rotate_sidechain_atoms(
    residue,
    chi_atoms: Tuple[str, str, str, str],
    angle_deg: float,
) -> None:
    """Rotate the side-chain atoms that lie beyond the third atom of the
    dihedral defined by *chi_atoms* by *angle_deg* degrees."""
    a1, a2, a3, a4 = chi_atoms
    if a3 not in residue or a2 not in residue or a4 not in residue:
        return

    p2 = residue[a2].get_vector().get_array()
    p3 = residue[a3].get_vector().get_array()

    axis_vec = p3 - p2
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len < 1e-8:
        return
    axis = axis_vec / axis_len

    angle_rad = math.radians(angle_deg)
    rot_mat = _rotation_matrix(axis, angle_rad)

    descendants = _collect_descendant_atoms(residue, a4, {a2, a3})
    for dname in descendants:
        if dname not in residue:
            continue
        vec = residue[dname].get_vector().get_array()
        rel = vec - p2
        rotated = rot_mat @ rel + p2
        residue[dname].set_coord(rotated)


def _generate_flexible_conformers(
    receptor_pdb: str,
    flexible_residues: List[str],
    max_conformers: int = 9,
) -> List[str]:
    """Generate multiple receptor PDB conformers by rotating side-chain
    dihedral angles of the given *flexible_residues*.

    For each residue with known chi-angle definitions (see ``_CHI_DEFS``),
    the function generates systematic rotamers:

        chi1: {current, current ± 60°}
        chi2: {current, current ± 30°}  (if applicable)

    The Cartesian product of all residue states is built and randomly
    sampled to respect *max_conformers*.

    Returns
    -------
    list of str
        Paths to the generated conformer PDB files (may be empty).
    """
    if not _HAVE_BIOPDB:
        log.warning("  Bio.PDB not available for flexible docking.")
        return []

    out_dir = os.path.join(os.path.dirname(receptor_pdb), "flex_conformers")
    os.makedirs(out_dir, exist_ok=True)

    target_list: List[Tuple[str, int]] = []
    for entry in flexible_residues:
        rname = "".join(ch for ch in entry if ch.isalpha()).upper()
        rnum = int("".join(ch for ch in entry if ch.isdigit()))
        target_list.append((rname, rnum))

    parser = PDBParser(QUIET=True)
    base_struct = parser.get_structure("flex_ref", receptor_pdb)

    residue_states: List[Tuple[int, List[Dict[int, float]]]] = []

    for model in base_struct:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                key = (residue.get_resname().strip().upper(), residue.get_id()[1])
                if key not in target_list:
                    continue
                resname = key[0]
                if resname not in _CHI_DEFS:
                    continue
                chi_defs = _CHI_DEFS[resname]
                states: List[Dict[int, float]] = [{}]
                for chi_idx in range(len(chi_defs)):
                    if chi_idx == 0:
                        angles = [0.0, 60.0, -60.0]
                    else:
                        angles = [0.0, 30.0, -30.0]
                    new_states = []
                    for s in states:
                        for a in angles:
                            ns = dict(s)
                            ns[chi_idx] = a
                            new_states.append(ns)
                    states = new_states
                residue_states.append((residue.get_id()[1], states))

    if not residue_states:
        log.warning("  No flexible residues with known chi definitions found.")
        return []

    from itertools import product as _product

    state_combos = list(_product(*(rs for _, rs in residue_states)))
    if len(state_combos) > max_conformers:
        rng = np.random.default_rng(CONFIG.random_seed)
        indices = rng.choice(len(state_combos), size=max_conformers, replace=False)
        state_combos = [state_combos[i] for i in sorted(indices)]

    conformer_pdbs_out: List[str] = []
    for combo_idx, combo in enumerate(state_combos):
        if all(all(v == 0.0 for v in state.values()) for state in combo):
            continue

        struct = parser.get_structure("flex", receptor_pdb)
        flex_idx = 0
        for model in struct:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] != " ":
                        continue
                    key = (residue.get_resname().strip().upper(), residue.get_id()[1])
                    if key not in target_list:
                        continue
                    resname = key[0]
                    if resname not in _CHI_DEFS:
                        continue
                    chi_defs = _CHI_DEFS[resname]
                    state = combo[flex_idx]
                    flex_idx += 1

                    for chi_idx, chi_tuple in enumerate(chi_defs):
                        angle = state.get(chi_idx, 0.0)
                        if abs(angle) > 0.1:
                            _rotate_sidechain_atoms(residue, chi_tuple, angle)

        out_pdb = os.path.join(out_dir, f"flex_{combo_idx:03d}.pdb")
        io = PDBIO()
        io.set_structure(struct)
        io.save(out_pdb)
        conformer_pdbs_out.append(out_pdb)

    log.info(f"  Generated {len(conformer_pdbs_out)} flexible-receptor conformers.")
    return conformer_pdbs_out


def _prepare_flexible_receptors(
    receptor_pdb: str,
    receptor_pdbqt: str,
    flexible_residues: List[str],
    max_conformers: int,
    deps: Dict[str, Any],
) -> List[Tuple[str, np.ndarray]]:
    """Generate flexible receptor conformers and convert them to PDBQT.

    Returns a list of ``(pdbqt_path, center)`` tuples suitable for
    ensemble docking.  The center is the same for all conformers
    (placeholder — caller should overwrite)."""
    conformer_pdbs = _generate_flexible_conformers(
        receptor_pdb, flexible_residues, max_conformers,
    )
    result: List[Tuple[str, np.ndarray]] = []
    for pdb_path in conformer_pdbs:
        pdbqt_path = pdb_path.replace(".pdb", ".pdbqt")
        try:
            cleaned = clean_pdb_structure(
                pdb_path, pdb_path,
                remove_waters=False,
                remove_ligands=False,
                add_hydrogens=True,
                deps=deps,
            )
            # cleaned returns the pdbqt path
            if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                result.append((pdbqt_path, np.zeros(3)))
        except Exception:
            continue
    return result


def _worker_dock_wrapper(
    args: Tuple[str, str, str, np.ndarray, Tuple[float, float, float], str, str, bool, int],
) -> Tuple[str, Optional[float], Optional[str], str]:
    """Module-level worker for :func:`_parallel_dock` (pool.map compatible).

    Returns ``(cid, energy, error_reason, method)`` where *error_reason* is
    ``"PrepFailure"`` when ligand preparation failed, ``"DockingFailure"``
    when the docking tool failed, and ``None`` otherwise.  *method* is the
    docking engine used.

    Callers use the error reason to populate
    :class:`~autoantibiotic.io_utils.PipelineAudit`.

    The per-job wall-clock timeout is enforced by :func:`run_tool` via
    ``subprocess.run(timeout=...)``, so no additional alarm mechanism is
    needed here.

    Accepts a seed value so that dry-run mode produces deterministic
    results across workers.
    """
    cid, smiles, receptor_pdbqt, center, box_size, work_dir, tag, dry_run, seed = args

    rng = np.random.default_rng(seed)
    if dry_run:
        method = "GNINA" if CONFIG.use_gnina else "Vina"
        return cid, float(rng.uniform(-10.0, -5.0)), None, method
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return cid, None, "DockingFailure", "None"
    rec = CompoundRecord(compound_id=cid, smiles=smiles, mol=mol)
    energy, method = dock_compound(
        rec, receptor_pdbqt, center, box_size,
        work_dir, tag, cache=None, use_cache=False,
    )
    if energy is None:
        error_reason = "PrepFailure" if method == "PrepFailure" else "DockingFailure"
    else:
        error_reason = None
    return cid, energy, error_reason, method


_Item = Tuple[str, str, Optional[np.ndarray], Optional[Tuple[float, float, float]]]
"""Extended item format: ``(compound_id, smiles, per_center, per_box_size)``.

When *per_center* and *per_box_size* are not ``None``, they override the
default *center* / *box_size* arguments of :func:`_parallel_dock` for
that specific compound.  This enables dynamic (ligand-adaptive) grid
box sizing without pickling RDKit Mol objects across process boundaries.
"""


def _parallel_dock(
    items: List[_Item],
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
    n_jobs: int = CONFIG.n_jobs,
    cache: _CacheLike = None,
    use_cache: bool = False,
    dry_run: bool = CONFIG.dry_run,
) -> List[Tuple[str, Optional[float], Optional[str], str]]:
    """Dock a list of compounds in parallel with batched processing.

    *items* may be 2-tuples ``(compound_id, smiles)`` for backward
    compatibility, or 4-tuples ``(compound_id, smiles, per_center,
    per_box_size)`` where *per_center* / *per_box_size* override the
    default grid parameters for that specific compound.

    Compounds are processed in batches (:attr:`CONFIG.batch_size_docking`,
    default 75) to allow periodic garbage collection and prevent memory
    bloat in worker processes.

    Returns list of ``(compound_id, energy, error_reason, method)`` tuples
    where *error_reason* is ``"DockingFailure"`` when the compound could not
    be docked and ``None`` otherwise.  *method* is the docking engine used.
    """
    results: List[Tuple[str, Optional[float], Optional[str], str]] = []
    to_dock: List[Tuple[str, str, str, Optional[np.ndarray], Optional[Tuple[float, float, float]]]] = []

    tool_name = "gnina" if CONFIG.use_gnina else "vina"
    for item in items:
        if len(item) == 4:
            cid, smiles, per_center, per_box = item
        else:
            cid, smiles = item  # type: ignore[misc]
            per_center, per_box = None, None
        cache_key = make_cache_key(smiles, tool_name)
        if use_cache and cache is not None and cache_key in cache:
            cached_val = cache[cache_key]
            results.append((cid, cached_val, None, "Unknown"))
            log.debug(f"    Cache hit: {cid} ({tag})")
        else:
            to_dock.append((cid, smiles, cache_key, per_center, per_box))

    if not to_dock:
        return results

    n_jobs_eff = min(n_jobs, len(to_dock))
    batch_size = CONFIG.batch_size_docking

    for batch_start in range(0, len(to_dock), batch_size):
        batch = to_dock[batch_start:batch_start + batch_size]

        worker_seed = CONFIG.random_seed + batch_start
        work_items: List[Tuple[str, str, str, np.ndarray, Tuple[float, float, float], str, str, bool, int]] = [
            (
                cid, smiles, receptor_pdbqt,
                per_center if per_center is not None else center,
                per_box if per_box is not None else box_size,
                work_dir, tag, dry_run, worker_seed,
            )
            for cid, smiles, _, per_center, per_box in batch
        ]

        chunksize_val = max(1, len(work_items) // (n_jobs_eff * 4))
        with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
            mapped = list(
                _tqdm(
                    pool.map(_worker_dock_wrapper, work_items, chunksize=chunksize_val),
                    total=len(work_items),
                    desc=f"  Docking {tag} batch {batch_start // batch_size + 1}",
                    disable=not _HAVE_TQDM,
                )
            )

        for (cid, _, cache_key, _, _), (_, energy, err, method) in zip(batch, mapped):
            results.append((cid, energy, err, method))
            if use_cache and cache is not None:
                cache[cache_key] = energy

        gc.collect()

    return results


def _compute_shape_fallback_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    seed: int = CONFIG.random_seed,
) -> Optional[float]:
    """Fallback scoring via RDKit Shape Protrude Distance and Pharmacophore matching.

    Combines shape and pharmacophore scores with a 0.5 / 0.5 weight when
    both are available.  Falls back to shape score alone if the
    pharmacophore calculation fails.
    """
    try:
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        status = rdDistGeom.EmbedMolecule(mol_3d, params)
        if status < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)

        ref_3d = Chem.RWMol(ref_mol)
        ref_3d = Chem.AddHs(ref_3d)
        params_ref = rdDistGeom.ETKDGv3()
        params_ref.randomSeed = seed
        status_ref = rdDistGeom.EmbedMolecule(ref_3d, params_ref)
        if status_ref < 0:
            return None
        AllChem.MMFFOptimizeMolecule(ref_3d)

        try:
            protrude = AllChem.GetShapeProtrudeDist(mol_3d, ref_3d)
        except Exception:
            try:
                protrude = AllChem.GetShapeProtrudeDist(ref_3d, mol_3d)
            except Exception:
                return None

        shape_score = min(protrude / CONFIG.shape_score_norm_factor, 10.0) if protrude > 0 else 0.0

        from .scoring_metrics import compute_pharmacophore_score

        pharm_score = compute_pharmacophore_score(mol, ref_mol)
        if pharm_score is not None:
            return 0.5 * shape_score + 0.5 * (1.0 - pharm_score)

        return shape_score

    except Exception:
        return None


def _parallel_dock_ensemble(
    items: List[Tuple[str, str]],
    receptor_pdbqt_list: List[str],
    center_list: List[np.ndarray],
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
    n_jobs: int = CONFIG.n_jobs,
    cache: _CacheLike = None,
    use_cache: bool = False,
    dry_run: bool = CONFIG.dry_run,
) -> List[Tuple[str, Optional[float], str]]:
    """Dock a list of compounds against an ensemble of receptors.

    Each compound is docked independently against every receptor,
    then a consensus score is computed via ``CONFIG.consensus_scoring_method``.

    Returns list of ``(compound_id, consensus_energy, method)`` tuples.
    """
    results: List[Tuple[str, Optional[float], str]] = []

    rng = np.random.default_rng(CONFIG.random_seed)
    for cid, smiles in items:
        if dry_run:
            method = "GNINA" if CONFIG.use_gnina else "Vina"
            results.append((cid, float(rng.uniform(-10.0, -5.0)), method))
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append((cid, None, "None"))
            continue
        rec = CompoundRecord(compound_id=cid, smiles=smiles, mol=mol)
        energy, method = dock_compound_ensemble(
            rec, receptor_pdbqt_list, center_list,
            box_size, work_dir, tag,
        )
        results.append((cid, energy, method))

    return results


def _build_flexible_ensemble(
    receptor_pdb: str,
    receptor_pdbqt: str,
    flexible_residues: List[str],
    max_conformers: int,
    deps: Dict[str, Any],
) -> Tuple[List[str], List[np.ndarray]]:
    """Generate flexible-receptor PDBQT conformers and return
    ``(pdbqt_list, center_list)`` suitable for ensemble docking.

    The caller is responsible for overriding the centers with the
    appropriate binding-site coordinates.
    """
    if not _HAVE_BIOPDB or not _HAVE_CLEAN:
        return [receptor_pdbqt], [np.zeros(3)]

    pdb_dir = os.path.dirname(receptor_pdb)
    conformer_pdbs = _generate_flexible_conformers(
        receptor_pdb, flexible_residues, max_conformers,
    )
    flex_list: List[str] = [receptor_pdbqt]
    for pdb_path in conformer_pdbs:
        pdbqt_path = pdb_path.replace(".pdb", ".pdbqt")
        try:
            cleaned = clean_pdb_structure(
                pdb_path, pdb_path,
                remove_waters=False,
                remove_ligands=False,
                add_hydrogens=True,
                deps=deps,
            )
            if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
                flex_list.append(pdbqt_path)
        except Exception:
            continue
    return flex_list, [np.zeros(3)] * len(flex_list)


def _apply_consensus_scoring(
    records: List[CompoundRecord],
    receptor_energies: List[List[Optional[float]]],
    attr_name: str = "pb2pa_allosteric_energy",
) -> None:
    """Compute consensus score per compound from per-receptor energies.

    Supports methods ``"mean"``, ``"median"``, ``"min"``, and ``"rank"``.

    *receptor_energies* is ``[receptor_idx][compound_idx]``, matching
    the order of *records*.
    """
    n_rec = len(receptor_energies)
    if n_rec == 0:
        return

    method = CONFIG.consensus_scoring_method
    n_compounds = len(records)

    if method == "rank":
        valid_indices: List[int] = []
        valid_by_receptor: List[List[float]] = [[] for _ in range(n_rec)]
        for j in range(n_compounds):
            energies = [receptor_energies[i][j] for i in range(n_rec)]
            if all(e is not None for e in energies):
                valid_indices.append(j)
                for i, e in enumerate(energies):
                    valid_by_receptor[i].append(e)

        if valid_by_receptor and valid_by_receptor[0]:
            avg_ranks = _compute_rank_consensus(valid_by_receptor)
            for orig_idx, avg_rank in zip(valid_indices, avg_ranks):
                setattr(records[orig_idx], attr_name, avg_rank)
            for j in range(n_compounds):
                if j not in valid_indices:
                    setattr(records[j], attr_name, None)
        else:
            for record in records:
                setattr(record, attr_name, None)
        return

    # mean / median / min
    for j, record in enumerate(records):
        energies = [receptor_energies[i][j] for i in range(n_rec)]
        valid = [e for e in energies if e is not None]
        if not valid:
            setattr(record, attr_name, None)
            continue
        if method == "min":
            setattr(record, attr_name, min(valid))
        elif method == "median":
            setattr(record, attr_name, statistics.median(valid))
        else:
            setattr(record, attr_name, statistics.mean(valid))


def _screen_ensemble(
    records: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    cache: _CacheLike = None,
    use_cache: bool = False,
    dry_run: bool = CONFIG.dry_run,
    audit: Optional[PipelineAudit] = None,
) -> List[CompoundRecord]:
    """Ensemble docking against multiple receptor structures with consensus scoring.

    Docks all compounds against each receptor independently, then
    computes a consensus score across receptors using the configured
    ``CONFIG.consensus_scoring_method`` ("mean", "min", "median", or
    "rank").
    """
    ensemble_targets = targets["PBP2a_ensemble"]
    receptor_pdbqt_list = [t["pdbqt"] for t in ensemble_targets]
    allosteric_center_list = [t["allosteric_center"] for t in ensemble_targets]
    active_center_list = [t["active_center"] for t in ensemble_targets]
    n_rec = len(receptor_pdbqt_list)

    log.info(f"  Ensemble docking all compounds against {n_rec} structures (allosteric site)…")

    # Pre-compute per-compound box parameters (same for all receptors)
    per_compound_box: Dict[str, Tuple[float, float, float]] = {}
    if CONFIG.use_dynamic_box_sizing:
        for r in records:
            _, per_box = _compute_dynamic_box_params(
                r, allosteric_center_list[0], CONFIG.allosteric_box_size,
            )
            per_compound_box[r.compound_id] = per_box

    # Phase 1: dock all compounds against each receptor separately
    receptor_energies: List[List[Optional[float]]] = []
    for i, (rec_pdbqt, center) in enumerate(zip(receptor_pdbqt_list, allosteric_center_list)):
        if per_compound_box:
            items: List[_Item] = [
                (r.compound_id, r.smiles, center, per_compound_box.get(r.compound_id))
                for r in records
            ]
        else:
            items = [(r.compound_id, r.smiles, None, None) for r in records]
        results = _parallel_dock(
            items, rec_pdbqt, center, CONFIG.allosteric_box_size,
            work_dir, f"ens_alloc_{i}",
            cache=cache, use_cache=use_cache, dry_run=dry_run,
        )
        receptor_energies.append([e for _, e, _, _ in results])

    # Phase 2: consensus scoring across receptors
    _apply_consensus_scoring(records, receptor_energies)

    n_scored = sum(1 for r in records if r.pb2pa_allosteric_energy is not None)
    log.info(f"  Ensemble allosteric docking complete: {n_scored}/{len(records)} scored.")

    # Record compounds that failed on ALL receptors
    if audit is not None:
        for r in records:
            if r.pb2pa_allosteric_energy is None:
                audit.record_dropout(r.compound_id, "DockingFailure")

    scored = [r for r in records if r.pb2pa_allosteric_energy is not None]
    primary_tool = "GNINA" if CONFIG.use_gnina else "Vina"
    for r in scored:
        r.docking_method = primary_tool
    scored.sort(key=lambda r: r.pb2pa_allosteric_energy)
    top50 = scored[:CONFIG.top_n_for_active]

    if top50:
        log.info(f"  Ensemble docking top {len(top50)} against active site ({n_rec} structures)…")
        active_receptor_energies: List[List[Optional[float]]] = []
        for i, (rec_pdbqt, center) in enumerate(zip(receptor_pdbqt_list, active_center_list)):
            if per_compound_box:
                active_items: List[_Item] = [
                    (r.compound_id, r.smiles, center, per_compound_box.get(r.compound_id))
                    for r in top50
                ]
            else:
                active_items = [(r.compound_id, r.smiles, None, None) for r in top50]
            results = _parallel_dock(
                active_items, rec_pdbqt, center, CONFIG.active_box_size,
                work_dir, f"ens_act_{i}",
                cache=cache, use_cache=use_cache, dry_run=dry_run,
            )
            active_receptor_energies.append([e for _, e, _, _ in results])
        _apply_consensus_scoring(top50, active_receptor_energies, "pb2pa_active_energy")

    return scored[:CONFIG.top_n]


def _screen_flexible(
    records: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: _CacheLike = None,
    use_cache: bool = False,
    dry_run: bool = CONFIG.dry_run,
) -> List[CompoundRecord]:
    """Flexible docking with side-chain rotamer conformers (min-score consensus)."""
    pb2pa = targets["PBP2a"]
    allosteric_center = pb2pa["allosteric_center"]
    active_center = pb2pa["active_center"]

    receptor_pdb = pb2pa["pdbqt"].replace(".pdbqt", ".pdb")
    if not os.path.exists(receptor_pdb):
        receptor_pdb = targets.get("holo_pdb", "")

    if not os.path.exists(receptor_pdb):
        log.warning("  Receptor PDB not found for flexible docking; falling back to standard.")
        return _screen_standard(records, targets, work_dir, cache=cache, use_cache=use_cache, dry_run=dry_run)

    log.info("  Flexible-docking mode enabled — generating side-chain conformers…")
    flex_alloc_pdbqt_list, flex_alloc_center_list = _build_flexible_ensemble(
        receptor_pdb, pb2pa["pdbqt"],
        CONFIG.flexible_residues_allosteric,
        CONFIG.max_flexible_conformers, deps,
    )
    flex_alloc_center_list = [allosteric_center] * len(flex_alloc_pdbqt_list)
    log.info(f"  Flexible allosteric docking using {len(flex_alloc_pdbqt_list)} conformers (min-score consensus)…")

    saved_method = CONFIG.consensus_scoring_method
    CONFIG.consensus_scoring_method = "min"

    allosteric_results = _parallel_dock_ensemble(
        [(r.compound_id, r.smiles) for r in records],
        flex_alloc_pdbqt_list, flex_alloc_center_list,
        CONFIG.allosteric_box_size, work_dir, "flex_alloc",
        cache=cache, use_cache=use_cache, dry_run=dry_run,
    )
    CONFIG.consensus_scoring_method = saved_method

    cid_to_record = {r.compound_id: r for r in records}
    primary_tool = "GNINA" if CONFIG.use_gnina else "Vina"
    for cid, energy, method in allosteric_results:
        if cid in cid_to_record:
            cid_to_record[cid].pb2pa_allosteric_energy = energy
            cid_to_record[cid].docking_method = method or primary_tool

    n_scored = sum(1 for r in records if r.pb2pa_allosteric_energy is not None)
    log.info(f"  Flexible allosteric docking complete: {n_scored}/{len(records)} scored.")

    scored = [r for r in records if r.pb2pa_allosteric_energy is not None]
    scored.sort(key=lambda r: r.pb2pa_allosteric_energy)
    top50 = scored[:CONFIG.top_n_for_active]

    flex_act_pdbqt_list, flex_act_center_list = _build_flexible_ensemble(
        receptor_pdb, pb2pa["pdbqt"],
        CONFIG.flexible_residues_active,
        max(3, CONFIG.max_flexible_conformers // 2), deps,
    )
    flex_act_center_list = [active_center] * len(flex_act_pdbqt_list)
    log.info(f"  Flexible active-site docking top {len(top50)} using {len(flex_act_pdbqt_list)} conformers…")

    saved_method = CONFIG.consensus_scoring_method
    CONFIG.consensus_scoring_method = "min"

    active_results = _parallel_dock_ensemble(
        [(r.compound_id, r.smiles) for r in top50],
        flex_act_pdbqt_list, flex_act_center_list,
        CONFIG.active_box_size, work_dir, "flex_act",
        cache=cache, use_cache=use_cache, dry_run=dry_run,
    )
    CONFIG.consensus_scoring_method = saved_method

    for cid, energy, method in active_results:
        if cid in cid_to_record:
            cid_to_record[cid].pb2pa_active_energy = energy
            cid_to_record[cid].docking_method = method or primary_tool

    return scored[:CONFIG.top_n]


def _compute_dynamic_box_params(
    record: CompoundRecord,
    center: np.ndarray,
    base_box_size: Tuple[float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Compute per-compound dynamic box parameters.

    When ``CONFIG.use_dynamic_box_sizing`` is True and the record has a
    valid ``mol``, the box size is expanded to accommodate the ligand's
    maximum dimension.  The center remains unchanged.

    Args:
        record: Compound record (uses ``.mol`` if available).
        center: Default binding-site centroid.
        base_box_size: Default box dimensions (e.g. ``CONFIG.allosteric_box_size``).

    Returns:
        ``(center, box_size)`` tuple — either the defaults or an expanded box.
    """
    if not CONFIG.use_dynamic_box_sizing or record.mol is None:
        return center, base_box_size

    try:
        ligand_max_dim = get_ligand_max_dimension(record.mol)
    except Exception:
        return center, base_box_size

    half_buffer = ligand_max_dim / 2.0 + CONFIG.dynamic_box_padding
    base = np.array(base_box_size, dtype=float)
    dyn = np.maximum(base, half_buffer)
    return center, (float(dyn[0]), float(dyn[1]), float(dyn[2]))


def _screen_standard(
    records: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    cache: _CacheLike = None,
    use_cache: bool = False,
    dry_run: bool = CONFIG.dry_run,
    audit: Optional[PipelineAudit] = None,
) -> List[CompoundRecord]:
    """Standard Vina/GNINA docking: allosteric site → top 50 to active site."""
    pb2pa = targets["PBP2a"]
    allosteric_center = pb2pa["allosteric_center"]
    active_center = pb2pa["active_center"]

    log.info("  Docking all compounds against allosteric site…")
    items: List[_Item] = []
    for r in records:
        per_center, per_box = _compute_dynamic_box_params(
            r, allosteric_center, CONFIG.allosteric_box_size,
        )
        items.append((r.compound_id, r.smiles, per_center, per_box))
    allosteric_results = _parallel_dock(
        items, pb2pa["pdbqt"],
        allosteric_center, CONFIG.allosteric_box_size,
        work_dir, "allosteric",
        cache=cache, use_cache=use_cache, dry_run=dry_run,
    )

    cid_to_record = {r.compound_id: r for r in records}
    for cid, energy, err, method in allosteric_results:
        if cid in cid_to_record:
            cid_to_record[cid].pb2pa_allosteric_energy = energy
            cid_to_record[cid].docking_method = method
        if err is not None and audit is not None:
            audit.record_dropout(cid, err)

    n_scored = sum(1 for r in records if r.pb2pa_allosteric_energy is not None)
    log.info(f"  Allosteric docking complete: {n_scored}/{len(records)} scored.")

    scored = [r for r in records if r.pb2pa_allosteric_energy is not None]
    scored.sort(key=lambda r: r.pb2pa_allosteric_energy)

    top50 = scored[:CONFIG.top_n_for_active]
    log.info(f"  Docking top {len(top50)} compounds against active site…")

    active_items: List[_Item] = []
    for r in top50:
        per_center, per_box = _compute_dynamic_box_params(
            r, active_center, CONFIG.active_box_size,
        )
        active_items.append((r.compound_id, r.smiles, per_center, per_box))
    active_results = _parallel_dock(
        active_items, pb2pa["pdbqt"],
        active_center, CONFIG.active_box_size,
        work_dir, "active",
        cache=cache, use_cache=use_cache, dry_run=dry_run,
    )

    for cid, energy, err, method in active_results:
        if cid in cid_to_record:
            cid_to_record[cid].pb2pa_active_energy = energy
            cid_to_record[cid].docking_method = method
        if err is not None and audit is not None:
            audit.record_dropout(cid, err)

    return scored[:CONFIG.top_n]


def _screen_shape_fallback(
    records: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
) -> List[CompoundRecord]:
    """Fallback scoring using RDKit Shape Protrude Distance + Pharmacophore."""
    log.info("  Vina unavailable. Using RDKit Shape Fallback.")

    ref_mol = None
    holo_pdb = targets.get("holo_pdb")
    if holo_pdb and os.path.exists(holo_pdb):
        lig_pdb = os.path.join(work_dir, "native_ref.pdb")
        try:
            from Bio.PDB import PDBIO, PDBParser, Select

            parser = PDBParser(QUIET=True)
            struct = parser.get_structure("ref", holo_pdb)
            for model in struct:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] != " " and residue.get_resname().strip() not in ("HOH", "WAT"):
                            pdbio = PDBIO()

                            class _Sel(Select):
                                def accept_residue(self, r):
                                    return r is residue

                            pdbio.set_structure(struct)
                            pdbio.save(lig_pdb, _Sel())
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            ref_mol = Chem.MolFromPDBFile(lig_pdb)
        except Exception:
            pass

    if ref_mol is None:
        ref_smi = list(CONFIG.control_smiles.values())[0]
        ref_mol = Chem.MolFromSmiles(ref_smi)

    if ref_mol is None:
        log.error("  Cannot obtain reference molecule for shape scoring.")
        for rec in records[:CONFIG.top_n]:
            rec.docking_method = "Failed"
        return records[:CONFIG.top_n]

    total = len(records)
    shape_iter = _tqdm(
        enumerate(records), total=total,
        desc="  Shape scoring", disable=not _HAVE_TQDM,
    )
    for i, rec in shape_iter:
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol
        score = _compute_shape_fallback_score(rec.mol, ref_mol)
        rec.shape_score = score
        rec.docking_method = "ShapeFallback"
        if (i + 1) % 100 == 0 and not _HAVE_TQDM:
            log.info(f"  Shape scored {i + 1} / {total}")

    scored_shape = [r for r in records if r.shape_score is not None]
    scored_shape.sort(key=lambda r: r.shape_score)
    if scored_shape:
        log.info(f"  Shape scoring complete. Best score: {scored_shape[0].shape_score:.3f}")
    else:
        log.warning("  No shape scores computed.")

    return scored_shape[:CONFIG.top_n]


def _apply_ml_rescoring(
    scored: List[CompoundRecord],
    pb2pa_pdbqt: str,
    work_dir: str,
    water_results: Optional[WaterAnalysisResult] = None,
) -> None:
    """Apply ML rescoring to top N Vina hits (in-place)."""
    if not (CONFIG.use_ml_rescoring and _HAVE_ML_SCORING and scored):
        return
    n_rescore = min(len(scored), CONFIG.mm_gbsa_top_n) if (CONFIG.use_mm_gbsa or CONFIG.use_mm_gbsa_rescoring) else 50
    log.info(f"  Applying ML rescoring to top {n_rescore} Vina hits…")
    try:
        top_to_rescore = scored[:n_rescore]
        _rescore_with_ml(top_to_rescore, pb2pa_pdbqt, work_dir, water_results=water_results)
    except Exception as exc:
        log.warning(f"  ML rescoring failed: {exc}")


def screen_library(
    records: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: _CacheLike = None,
    use_cache: bool = False,
    water_results: Optional[WaterAnalysisResult] = None,
    dry_run: bool = CONFIG.dry_run,
    audit: Optional[PipelineAudit] = None,
) -> List[CompoundRecord]:
    """Phase 3 — Virtual screening.

    Dispatches to the appropriate docking strategy based on config flags:

    * **Ensemble mode** → :func:`_screen_ensemble`
    * **Flexible docking** → :func:`_screen_flexible`
    * **Standard Vina/GNINA** → :func:`_screen_standard`
    * **No Vina available** → :func:`_screen_shape_fallback`

    After docking, ML rescoring is applied to the top hits when enabled.

    When *water_results* is provided, it is forwarded to the ML rescoring
    stage for water displacement correction in MM-GB/SA.

    When *audit* is provided, docking failures are recorded as dropouts.

    Returns top 10 candidates.
    """
    log.info("─── Phase 3: Virtual Screening ───")

    pb2pa = targets["PBP2a"]
    use_vina = deps.get("USE_VINA", False)

    if use_vina:
        ensemble_targets = targets.get("PBP2a_ensemble")
        ensemble_active = ensemble_targets is not None and len(ensemble_targets) > 0

        if ensemble_active:
            if CONFIG.flexible_docking:
                log.info("  Both ensemble and flexible-docking enabled; ensemble takes priority.")
            log.info(f"  Ensemble docking (method={CONFIG.consensus_scoring_method}) against {len(ensemble_targets)} structures…")
            top_candidates = _screen_ensemble(
                records, targets, work_dir, cache=cache, use_cache=use_cache, dry_run=dry_run,
                audit=audit,
            )
        elif CONFIG.flexible_docking and _HAVE_BIOPDB:
            top_candidates = _screen_flexible(
                records, targets, work_dir, deps, cache=cache, use_cache=use_cache, dry_run=dry_run,
            )
        else:
            top_candidates = _screen_standard(
                records, targets, work_dir, cache=cache, use_cache=use_cache, dry_run=dry_run,
                audit=audit,
            )

        # ML rescoring on top N Vina hits
        scored = [r for r in records if r.pb2pa_allosteric_energy is not None]
        scored.sort(key=lambda r: r.pb2pa_allosteric_energy)
        _apply_ml_rescoring(scored, pb2pa["pdbqt"], work_dir, water_results)

        ranked = scored
    else:
        rank = _screen_shape_fallback(records, targets, work_dir)
        ranked = rank

    ranked.sort(key=lambda r: r.pb2pa_allosteric_energy if r.pb2pa_allosteric_energy is not None else float("inf"))
    top10 = ranked[:CONFIG.top_n]

    # Assign docked pose PDBQT paths for top candidates (used in IFP filtering)
    tags_to_try = ["active", "allosteric", "flex_act", "flex_alloc",
                   "ens_act_0", "ens_alloc_0"]
    for r in top10:
        safe_id = r.compound_id.replace("/", "_").replace(" ", "_")
        for tag in tags_to_try:
            expected = os.path.join(work_dir, f"{safe_id}_{tag}_out.pdbqt")
            if os.path.exists(expected):
                r.docked_pose_path = expected
                break

    log.info(f"  Top {len(top10)} candidates selected.")
    for i, r in enumerate(top10):
        energy_str = (
            f"{r.pb2pa_allosteric_energy:.2f}" if r.pb2pa_allosteric_energy is not None
            else f"{r.shape_score:.2f} (shape)"
        )
        log.info(f"    {i + 1}. {r.compound_id}: {energy_str} kcal/mol")

    log.info("─── Phase 3 complete ───")
    return top10
