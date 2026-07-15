"""
Docking utilities
==================

Virtual screening helpers: AutoDock Vina invocation, single/multi-compound
docking orchestration, and the RDKit Shape-Protrude fallback scorer used when
Vina is unavailable.

Docking-related constants (``VINA_TIMEOUT_S``, ``N_JOBS``, ``RANDOM_SEED``)
live in ``config.constants`` and are imported at module top level, which keeps
the ``utils`` package free of a circular import with ``discovery_pipeline``.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Tuple, Optional, Callable

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from .ligand_prep import prepare_ligand_pdbqt
from config.constants import VINA_TIMEOUT_S, N_JOBS, RANDOM_SEED

# Shared logger: same name as the one configured in discovery_pipeline.
log = logging.getLogger("AutoAntibiotic")


def _run_vina_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    timeout: Optional[int] = None,
) -> Optional[float]:
    """
    Run a single Vina docking job. Returns best binding energy (kcal/mol)
    or None on failure.
    """
    if timeout is None:
        timeout = VINA_TIMEOUT_S

    cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", output_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.1f}",
        "--size_y", f"{box_size[1]:.1f}",
        "--size_z", f"{box_size[2]:.1f}",
        "--exhaustiveness", "8",
        "--num_modes", "3",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.warning(
                f"  Vina returned exit code {result.returncode}.\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  stdout: {result.stdout.strip()}"
            )
            return None

        # Parse output for best binding energy
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("1") and " " in stripped:
                # Vina table format: mode | affinity | dist from best mode
                parts = stripped.split()
                try:
                    energy = float(parts[1])
                    return energy
                except (ValueError, IndexError):
                    continue
        # Fallback: parse from log tail
        for line in result.stderr.splitlines():
            if "Affinity" in line and "kcal/mol" in line:
                try:
                    energy = float(line.split()[1])
                    return energy
                except (ValueError, IndexError):
                    continue
        # If we reach here, no energy could be parsed — log full output
        log.warning(
            "  Failed to parse Vina binding energy from output.\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
        return None

    except subprocess.TimeoutExpired:
        log.warning(f"  Vina timeout ({timeout}s).")
        return None
    except FileNotFoundError:
        log.warning("  Vina binary not found.")
        return None
    except Exception as exc:
        log.warning(f"  Vina exception: {exc}")
        return None


def dock_compound(
    record: "CompoundRecord",
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "",
) -> Optional[float]:
    """
    Full docking pipeline for a single compound: PDBQT prep → Vina → parse.

    Args:
        record: Compound record (must have .mol).
        receptor_pdbqt: Path to receptor PDBQT.
        center: Grid box centre.
        box_size: Grid box dimensions.
        work_dir: Scratch directory.
        tag: Label for temp files (e.g. 'allosteric').

    Returns:
        Best binding energy, or None on failure.
    """
    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            return None
        record.mol = mol

    # Generate unique filenames
    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    lig_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_lig.pdbqt")
    out_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_out.pdbqt")

    if not prepare_ligand_pdbqt(record.mol, lig_pdbqt):
        raise RuntimeError(
            f"PDBQT preparation failed for {record.compound_id}; "
            f"this compound will be skipped during screening."
        )

    energy = _run_vina_docking(
        receptor_pdbqt, lig_pdbqt, out_pdbqt,
        center, box_size,
    )

    # Keep the docked pose for the active site so downstream pose analysis
    # (binding interactions) can reuse it instead of re-docking.
    if tag == "active":
        record.active_docked_pdbqt = out_pdbqt

    # Cleanup temp files (keep the active-site pose for later analysis)
    for f in (lig_pdbqt, out_pdbqt):
        if tag == "active" and f == out_pdbqt:
            continue
        try:
            os.remove(f)
        except OSError:
            pass

    return energy


def _dock_compounds_parallel(
    records: "List[CompoundRecord]",
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
    n_jobs: Optional[int] = None,
    dock_func: Optional[Callable] = None,
) -> List[Tuple["CompoundRecord", Optional[float]]]:
    """
    Dock a list of compounds in parallel, returning ``(record, energy)`` pairs.

    Each compound is docked by *dock_func* (defaults to :func:`dock_compound`).
    If a worker raises, the specific error is logged together with the
    ``CompoundRecord.compound_id`` and the record is returned with
    ``energy=None`` so the pipeline continues instead of aborting.

    When ``n_jobs <= 1`` (or for small batches) the docking is performed
    in-process, which keeps behaviour deterministic and avoids the overhead
    of spawning worker processes.

    Note (memory): for very large libraries the :class:`CompoundRecord.mol`
    objects are pickled for each worker. If profiling shows a bottleneck,
    callers may pass lightweight ``(compound_id, smiles)`` payloads and
    reconstruct the :class:`~rdkit.Chem.Mol` inside *dock_func*.

    Args:
        records: Compounds to dock (must expose ``.mol`` / ``.smiles``).
        receptor_pdbqt: Path to receptor PDBQT.
        center: Grid-box centre as a length-3 array.
        box_size: Grid-box dimensions ``(x, y, z)``.
        work_dir: Scratch directory for intermediate files.
        tag: Label for temporary files (e.g. ``"allosteric"``).
        n_jobs: Number of worker processes.
        dock_func: Docking callable; mainly useful for testing.

    Returns:
        List of ``(CompoundRecord, energy_or_None)`` tuples.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if n_jobs is None:
        n_jobs = N_JOBS
    if dock_func is None:
        dock_func = dock_compound

    results: List[Tuple["CompoundRecord", Optional[float]]] = []
    total = len(records)

    # In-process execution keeps small batches deterministic and testable.
    if n_jobs <= 1:
        for i, rec in enumerate(records):
            results.append(_dock_worker(
                rec, dock_func, receptor_pdbqt, center, box_size, work_dir, tag,
            ))
            if (i + 1) % 25 == 0:
                log.info(f"    Docked {i + 1} / {total} ({tag})")
        return results

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(
                _dock_worker, rec, dock_func,
                receptor_pdbqt, center, box_size, work_dir, tag,
            ): rec
            for rec in records
        }
        for i, future in enumerate(as_completed(futures)):
            rec = futures[future]  # original record
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as exc:
                log.warning(
                    f"    Docking failed for {rec.compound_id} ({tag}): {exc}. "
                    "Returning (record, None) and continuing."
                )
                results.append((rec, None))
            if (i + 1) % 25 == 0:
                log.info(f"    Docked {i + 1} / {total} ({tag})")

    return results


def _dock_worker(
    rec: "CompoundRecord",
    dock_func: Callable,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
) -> Tuple["CompoundRecord", Optional[float]]:
    """
    Module-level docking wrapper so it can be pickled by ``ProcessPoolExecutor``.

    Runs *dock_func* for a single record and returns ``(record, energy)``.
    On any failure the error is logged with the ``CompoundRecord.compound_id``
    and ``(record, None)`` is returned so the pipeline keeps going.
    """
    try:
        energy = dock_func(rec, receptor_pdbqt, center, box_size, work_dir, tag)
        return rec, energy
    except Exception as exc:
        log.warning(
            f"    Docking failed for {rec.compound_id} ({tag}): {exc}. "
            "Returning (record, None) and continuing."
        )
        return rec, None


def _compute_shape_fallback_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    seed: Optional[int] = None,
) -> Optional[float]:
    """
    Fallback scoring: generate 3D conformer, compute shape protrude distance
    vs reference (co-crystallised ligand from 6TKO). Normalise to 0–10 scale
    (lower = better shape match).

    If available, also computes electrostatic similarity and combines with
    the shape score (50/50 weight) for a more robust metric.

    Returns combined normalised score (0–10, lower = better), or None on failure.
    """
    if seed is None:
        seed = RANDOM_SEED

    try:
        # Generate 3D conformer with ETKDGv3 for better stereochemistry
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.useExpTorsionAnglePrefs = True
        params.useBasicKnowledge = True
        params.enforceChirality = True
        params.randomSeed = seed
        status = rdDistGeom.EmbedMolecule(mol_3d, params)
        if status < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)

        ref_3d = Chem.RWMol(ref_mol)
        ref_3d = Chem.AddHs(ref_3d)
        params_ref = rdDistGeom.ETKDGv3()
        params_ref.useExpTorsionAnglePrefs = True
        params_ref.useBasicKnowledge = True
        params_ref.enforceChirality = True
        params_ref.randomSeed = seed
        status_ref = rdDistGeom.EmbedMolecule(ref_3d, params_ref)
        if status_ref < 0:
            return None
        AllChem.MMFFOptimizeMolecule(ref_3d)

        # Shape protrude distance
        try:
            protrude = AllChem.GetShapeProtrudeDist(mol_3d, ref_3d)
        except Exception:
            try:
                protrude = AllChem.GetShapeProtrudeDist(ref_3d, mol_3d)
            except Exception:
                return None

        # Normalise to 0–10 scale (heuristic: typical range 0–0.5)
        # Map: protrude=0 → score=0 (perfect), protrude=0.5 → score=10 (worst)
        shape_norm = min(protrude / 0.05, 10.0) if protrude > 0 else 0.0

        # Electrostatic similarity (optional enhancement)
        elec_sim = None
        try:
            from rdkit.Chem.rdMolDescriptors import GetElectrostaticSimilarity
            elec_sim = GetElectrostaticSimilarity(mol_3d, ref_3d)
        except Exception:
            pass

        if elec_sim is not None:
            # Convert electrostatic similarity (0–1, higher = better) to
            # a penalty (0–10, lower = better) and average with shape score
            elec_penalty = (1.0 - elec_sim) * 10.0
            combined = 0.5 * shape_norm + 0.5 * elec_penalty
            return combined

        return shape_norm

    except Exception:
        return None


def _compute_shape_scores(
    records: "List[CompoundRecord]",
    ref_mol: Chem.Mol,
) -> List[Tuple["CompoundRecord", Optional[float]]]:
    """
    Compute RDKit Shape-Protrude fallback scores for a list of records.

    For every record the molecule is embedded in 3D and compared against
    *ref_mol* (the co-crystallised 6TKO ligand).  The normalised score
    (0–10, lower = better shape match) is stored on ``rec.shape_score``.

    Args:
        records: Compounds to score (must expose ``.mol`` / ``.smiles``).
        ref_mol: Reference molecule used for the shape comparison.

    Returns:
        List of ``(CompoundRecord, score_or_None)`` tuples.
    """
    total = len(records)
    scored: List[Tuple["CompoundRecord", Optional[float]]] = []

    for i, rec in enumerate(records):
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                rec.shape_score = None
                scored.append((rec, None))
                continue
            rec.mol = mol

        score = _compute_shape_fallback_score(rec.mol, ref_mol)
        rec.shape_score = score
        scored.append((rec, score))

        if (i + 1) % 100 == 0:
            log.info(f"  Shape scored {i + 1} / {total}")

    return scored
