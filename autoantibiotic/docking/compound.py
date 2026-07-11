"""Compound-level docking functions (single-compound operations)."""

from __future__ import annotations

import math
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..config import CONFIG, PipelineConfig, ConfigurationError
from ..models import CompoundRecord
from ..io_utils import (
    AutoAntibioticError,
    log,
    make_cache_key,
)
from .base import DockingEngine

_CacheLike = Optional[Dict[str, float]]


# ── Ligand Preparation ─────────────────────────────────────────────


def prepare_ligand_pdbqt(
    mol: Chem.Mol,
    output_path: str,
    config: Optional[PipelineConfig] = None,
) -> bool:
    """Convert an RDKit Mol to PDBQT via Meeko.

    Attempts conversion using Meeko's MoleculePreparation and
    PDBQTWriterLegacy.  If Meeko fails, falls back to a minimal PDBQT
    writer that assigns Gasteiger charges and writes a rigid (TORSDOF 0)
    PDBQT entry.

    Args:
        mol: RDKit molecule with at least one conformer.
        output_path: Path for the output PDBQT file.
        config: Optional pipeline config.

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

# ── Single-compound docking ────────────────────────────────────────


def dock_compound(
    record: CompoundRecord,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "",
    cache: _CacheLike = None,
    use_cache: bool = False,
    config: Optional[PipelineConfig] = None,
    engine: Optional[DockingEngine] = None,
) -> Tuple[Optional[float], str]:
    """Full docking pipeline for a single compound: PDBQT prep / dock / parse.

    Returns ``(energy, method)`` where *method* is ``"GNINA"``, ``"Vina"``,
    ``"None"``, or ``"Unknown"`` (cache hit).
    """
    cfg = config or CONFIG
    tool_name = "gnina" if cfg.use_gnina else "vina"
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

    if not prepare_ligand_pdbqt(record.mol, lig_pdbqt, config=cfg):
        return None, "PrepFailure"

    method = "None"
    if engine is None:
        from . import get_engine
        engine = get_engine(tool_name, cfg)
    energy = engine.dock(lig_pdbqt, receptor_pdbqt, center, box_size)
    method = "GNINA" if tool_name == "gnina" else "Vina"

    if energy is None and cfg.use_gnina:
        log.warning("  GNINA docking failed, falling back to Vina.")
        from . import get_engine as _get_engine
        fallback_engine = _get_engine("vina", cfg)
        energy = fallback_engine.dock(lig_pdbqt, receptor_pdbqt, center, box_size)
        method = "Vina"

    # Keep out_pdbqt on disk for downstream IFP analysis.
    for f in (lig_pdbqt,):
        try:
            os.remove(f)
        except OSError:
            pass

    if use_cache and cache is not None:
        cache[cache_key] = energy

    return energy, method


def dock_compound_ensemble(
    record: CompoundRecord,
    receptor_pdbqt_list: List[str],
    center_list: List[np.ndarray],
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "",
    config: Optional[PipelineConfig] = None,
) -> Tuple[Optional[float], str]:
    """Dock a compound against an ensemble of receptor structures.

    Each receptor structure is docked independently.  The final score
    is aggregated via ``config.consensus_scoring_method`` ("mean",
    "median", "min", or "rank").

    .. note::
       The ``"rank"`` method requires energies across *all* compounds
       to compute per-receptor rankings.  When called for a single
       compound, rank consensus cannot be computed and ``mean`` is
       returned as a fallback.  Use :func:`_compute_rank_consensus`
       in batch mode for proper rank-based consensus.

    Returns ``(consensus_score, method)``.
    """
    cfg = config or CONFIG
    energies: List[float] = []
    method = "GNINA" if cfg.use_gnina else "Vina"
    for i, (rec_pdbqt, ctr) in enumerate(zip(receptor_pdbqt_list, center_list)):
        e, _ = dock_compound(
            record, rec_pdbqt, ctr, box_size,
            work_dir, f"{tag}_ens{i}", config=cfg,
        )
        if e is not None:
            energies.append(e)

    if not energies:
        return None, "None"

    consensus = cfg.consensus_scoring_method
    if consensus == "min":
        return min(energies), method
    elif consensus == "median":
        return statistics.median(energies), method
    elif consensus == "rank":
        return statistics.mean(energies), method
    else:
        return statistics.mean(energies), method
