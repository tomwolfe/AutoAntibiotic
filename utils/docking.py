"""
Docking utilities
==================

Virtual screening helpers: AutoDock Vina invocation and single/multi-compound
docking orchestration.

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
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
try:  # RDKit exposes a fast 3D alignment used by the offline fallback.
    from rdkit.Chem import rdShapeHelpers
except Exception:  # pragma: no cover - very old RDKit builds
    rdShapeHelpers = None

from .ligand_prep import prepare_ligand_pdbqt
from .library_gen import CompoundRecord
from config.constants import VINA_TIMEOUT_S, N_JOBS, RANDOM_SEED

# Pharmacophore feature weights for the lightweight RDKit fallback scorer.
# These approximate the broad physicochemical tendencies that drive binding
# (H-bond donors/acceptors, hydrophobic/aromatic bulk, positive charge). They
# are *qualitative* and must NOT be read as kcal/mol.
_PHARM_WEIGHTS = {
    "Donor": 0.6,
    "Acceptor": 0.6,
    "PosIonizable": 0.5,
    "NegIonizable": 0.4,
    "Aromatic": 0.3,
    "Hydrophobe": 0.2,
}

# Shared logger: same name as the one configured in discovery_pipeline.
log = logging.getLogger("AutoAntibiotic")


def _run_vina_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    timeout: Optional[int] = None,
    flex_pdbqt: Optional[str] = None,
) -> Optional[float]:
    """
    Run a single Vina docking job. Returns best binding energy (kcal/mol)
    or None on failure.

    When *flex_pdbqt* is provided it is passed as Vina's ``--flex`` argument so
    the listed receptor residues are treated as flexible during docking (local
    flexible docking). *flex_pdbqt* must already be a valid flexible-residue
    PDBQT produced by the caller (e.g. via ``write_receptor_pdbqt`` logic or
    ``obabel``); failures are handled upstream by falling back to rigid docking.
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
    if flex_pdbqt is not None:
        cmd += ["--flex", flex_pdbqt]

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


def _rdkit_fallback_score(
    mol: Chem.Mol,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
) -> Optional[float]:
    """
    Lightweight RDKit shape/pharmacophore scoring used when AutoDock Vina is
    unavailable (e.g. offline CI runs, ``--smiles`` without Vina).

    This is a *heuristic* proxy, not a real docking energy. It estimates how
    well a ligand fits a cubic grid centred on ``center`` with half-extents
    ``box_size / 2`` and adds a crude pharmacophore term (H-bond /
    hydrophobic / charge features). Lower (more negative) values rank better,
    mirroring Vina's energy ordering, but the absolute numbers are meaningless
    and must never be reported as binding energies.

    Returns ``(score, None)`` where ``score`` is a synthetic negative number,
    or ``None`` if the molecule cannot be embedded/scored.

    Because these synthetic numbers can look like binding energies, a warning
    is always logged whenever the fallback is used so that downstream callers
    (and anyone reading logs) are never misled into treating them as Vina
    kcal/mol. Reports should additionally prefix any fallback score with
    ``"(fallback)"`` — see :func:`format_fallback_score`.
    """
    if mol is None:
        log.warning(
            "  ⚠  RDKit fallback scorer used (Vina unavailable). Scores are "
            "HEURISTIC ONLY and must NOT be read as kcal/mol binding energies."
        )
        return None
    try:
        log.warning(
            "  ⚠  Using RDKit shape/pharmacophore FALLBACK score (Vina "
            "unavailable). This is a heuristic proxy, NOT a kcal/mol binding "
            "energy. Prefix any reported value with \"(fallback)\"."
        )
        m = Chem.AddHs(mol)
        # Embed into a 3D conformation; if embedding fails, fall back to a
        # single-shot 2D->3D attempt with random coords.
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        if AllChem.EmbedMolecule(m, params) != 0:
            if AllChem.EmbedMolecule(m) != 0:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(m)
        except Exception:
            pass

        conf = m.GetConformer()
        coords = conf.GetPositions()
        if coords.size == 0:
            return None

        center = np.asarray(center, dtype=float)
        half = np.asarray(box_size, dtype=float) / 2.0

        # Distance from the ligand centroid to the grid centre (closer is
        # better). Penalise compounds whose atoms spill well outside the box.
        centroid = coords.mean(axis=0)
        dist_to_center = float(np.linalg.norm(centroid - center))

        # Spread (max distance from ligand centroid) — large ligands that
        # cannot fit the box are penalised.
        spread = float(np.linalg.norm(coords - centroid, axis=1).max())
        max_half = float(half.max())
        overflow = max(0.0, spread - max_half)

        # Pharmacophore term: a cheap element/property heuristic instead of the
        # full feature factory, to avoid extra RDKit factory construction cost.
        n_hbd = rdMolDescriptors.CalcNumHBD(m)
        n_hba = rdMolDescriptors.CalcNumHBA(m)
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        n_aro = rdMolDescriptors.CalcNumAromaticRings(m)
        n_rings = rdMolDescriptors.CalcNumRings(m)
        pharm = (
            _PHARM_WEIGHTS["Donor"] * n_hbd
            + _PHARM_WEIGHTS["Acceptor"] * n_hba
            + _PHARM_WEIGHTS["Aromatic"] * n_aro
            + _PHARM_WEIGHTS["Hydrophobe"] * n_rings
            + _PHARM_WEIGHTS["PosIonizable"] * (1 if Descriptors.NumHeteroatoms(m) else 0)
        )

        # Synthetic "energy": more negative for compact, centred, feature-rich
        # ligands. Purely relative — not a physical kcal/mol value.
        score = -(pharm - 0.15 * (dist_to_center + overflow + 0.2 * n_rot))
        return float(score)
    except Exception as exc:
        log.warning(f"  RDKit fallback scoring failed: {exc}")
        return None


def format_fallback_score(score: Optional[float]) -> str:
    """
    Format a fallback score for human-readable reports.

    Returns a string prefixed with ``"(fallback)"`` (and a clear "not kcal/mol"
    note) so that synthetic heuristic scores are never mistaken for Vina
    binding energies. When *score* is ``None`` (scoring failed) an explicit
    placeholder is returned.

    Args:
        score: The synthetic fallback score (negative float) or ``None``.

    Returns:
        Human-readable string such as ``"(fallback) -3.21 (not kcal/mol)"``.
    """
    if score is None:
        return "(fallback) N/A (not kcal/mol)"
    return f"(fallback) {score:.3f} (not kcal/mol)"


def dock_compound(
    record: "CompoundRecord",
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "",
    use_vina: bool = True,
    flex_pdbqt: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Optional[float]:
    """
    Full docking pipeline for a single compound: PDBQT prep → Vina → parse.

    When ``use_vina`` is ``False`` (AutoDock Vina unavailable), the function
    does NOT raise. Instead it returns an approximate score from the built-in
    RDKit shape/pharmacophore fallback (see :func:`_rdkit_fallback_score`).
    These fallback scores are *qualitative only* — they rank candidates
    relative to each other but are not physical binding energies.

    Args:
        record: Compound record (must have .mol).
        receptor_pdbqt: Path to receptor PDBQT.
        center: Grid box centre.
        box_size: Grid box dimensions.
        work_dir: Scratch directory.
        tag: Label for temp files (e.g. 'allosteric').
        use_vina: When ``False``, skip Vina and use the RDKit fallback scorer.
        flex_pdbqt: Optional flexible-residue PDBQT passed to Vina's ``--flex``
            for local flexible docking (active-site step). Ignored when Vina is
            unavailable.
        timeout: Optional per-call Vina timeout override (seconds). Used to give
            flexible (``--flex``) docking jobs a larger timeout than rigid jobs
            so they do not fall back to rigid docking on a transient timeout
            (Phase 3.5: robust flexible docking).

    Returns:
        Best binding energy (Vina) or fallback score, or None on failure.
    """
    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            return None
        record.mol = mol

    # Offline fallback path — no PDBQT prep, no Vina invocation.
    if not use_vina:
        log.info(
            f"  Vina unavailable — using RDKit shape/pharmacophore fallback "
            f"for {record.compound_id} ({tag})."
        )
        score = _rdkit_fallback_score(record.mol, center, box_size)
        if tag == "active" and score is not None:
            # No real pose file in fallback mode; clear any stale pose so pose
            # analysis is skipped downstream rather than mis-attributed.
            record.active_docked_pdbqt = None
        return score

    # Generate unique filenames
    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    lig_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_lig.pdbqt")
    out_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_out.pdbqt")

    # Ensure explicit hydrogens and 3D coordinates (required by meeko)
    mol_for_prep = Chem.AddHs(record.mol)
    if not mol_for_prep.GetNumConformers():
        from rdkit.Chem import AllChem
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        try:
            AllChem.EmbedMolecule(mol_for_prep, params)
        except Exception:
            pass
    if not prepare_ligand_pdbqt(mol_for_prep, lig_pdbqt):
        raise RuntimeError(
            f"PDBQT preparation failed for {record.compound_id}; "
            f"this compound will be skipped during screening."
        )

    energy = _run_vina_docking(
        receptor_pdbqt, lig_pdbqt, out_pdbqt,
        center, box_size,
        timeout=timeout,
        flex_pdbqt=flex_pdbqt,
    )

    # Keep the docked pose for the active site so downstream pose analysis
    # (MM-GBSA rerank, H-bond flags, mutation scan) can reuse it instead of
    # re-docking. Consensus docking uses per-conformer tags ("active_c0"…) and
    # flexible docking uses "active_flex"; all of these are active-site poses
    # and must be retained. Mutant scans use "mut_*" and are NOT retained.
    #
    # IMPORTANT: only record the pose when the dock actually SUCCEEDED (energy
    # is not None AND the output file was written). Otherwise a failed dock —
    # e.g. a flexible ("active_flex") re-dock that Vina rejects — would clobber
    # a previously-retained good rigid pose with a path to a non-existent file,
    # silently breaking MM-GBSA / H-bond / mutation analysis downstream.
    is_active_pose = tag == "active" or tag.startswith("active_")
    dock_succeeded = (
        energy is not None
        and os.path.exists(out_pdbqt)
        and os.path.getsize(out_pdbqt) > 0
    )
    keep_out = is_active_pose and dock_succeeded
    if keep_out:
        record.active_docked_pdbqt = out_pdbqt

    # Cleanup temp files (keep the active-site pose for later analysis)
    for f in (lig_pdbqt, out_pdbqt):
        if keep_out and f == out_pdbqt:
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
    use_vina: bool = True,
) -> List[Tuple["CompoundRecord", Optional[float]]]:
    """
    Dock a list of compounds in parallel, returning ``(record, energy)`` pairs.

    Only ``(compound_id, smiles)`` is pickled for each worker, so the heavy
    :class:`~rdkit.Chem.Mol` objects stored on the records are never shipped to
    the worker processes — this keeps memory bounded for large libraries. The
    Mol is reconstructed inside the worker via ``Chem.MolFromSmiles`` and the
    result is mapped back to the original :class:`CompoundRecord` by id.

    Each compound is docked by *dock_func* (defaults to :func:`dock_compound`).
    If a worker raises, the specific error is logged together with the
    ``CompoundRecord.compound_id`` and the record is returned with
    ``energy=None`` so the pipeline continues instead of aborting.

    When ``n_jobs <= 1`` (or for small batches) the docking is performed
    in-process, which keeps behaviour deterministic and avoids the overhead
    of spawning worker processes.

    Args:
        records: Compounds to dock (must expose ``.compound_id`` / ``.smiles``).
        receptor_pdbqt: Path to receptor PDBQT.
        center: Grid-box centre as a length-3 array.
        box_size: Grid-box dimensions ``(x, y, z)``.
        work_dir: Scratch directory for intermediate files.
        tag: Label for temporary files (e.g. ``"allosteric"``).
        n_jobs: Number of worker processes.
        dock_func: Docking callable; mainly useful for testing.
        use_vina: When ``False``, the default *dock_func* uses the RDKit
            shape/pharmacophore fallback instead of invoking Vina.

    Returns:
        List of ``(CompoundRecord, energy_or_None)`` tuples.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if n_jobs is None:
        n_jobs = N_JOBS
    if dock_func is None:
        dock_func = dock_compound

    # Lightweight payloads: pickling only (id, smiles) avoids shipping the Mol.
    payloads = [(rec.compound_id, rec.smiles) for rec in records]
    by_id = {rec.compound_id: rec for rec in records}

    results: List[Tuple["CompoundRecord", Optional[float]]] = []
    total = len(records)

    # In-process execution keeps small batches deterministic and testable.
    if n_jobs <= 1:
        for i, payload in enumerate(payloads):
            rec, energy, pose = _dock_worker(
                payload, dock_func, receptor_pdbqt, center, box_size, work_dir, tag,
                use_vina,
            )
            parent = by_id[rec.compound_id]
            results.append((parent, energy))
            # Propagate the active-site pose path back to the parent record so
            # downstream pose analysis (MM-GBSA, H-bond flags, mutation scan)
            # can use it. Only the "active" tag produces a retained pose.
            if pose is not None:
                parent.active_docked_pdbqt = pose
            if (i + 1) % 25 == 0:
                log.info(f"    Docked {i + 1} / {total} ({tag})")
        return results

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(
                _dock_worker, payload, dock_func,
                receptor_pdbqt, center, box_size, work_dir, tag, use_vina,
            ): payload[0]
            for payload in payloads
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()   # worker returns (rec, energy_or_None, pose)
            rec, energy, pose = result
            parent = by_id[rec.compound_id]
            results.append((parent, energy))
            # Propagate the active-site pose path back to the parent record; the
            # consensus dock only keeps the best energy, but the retained pose is
            # needed for pose-based analysis. Keep the best (most recent valid)
            # pose — consensus docking uses the same grid, so any conformer's
            # active pose is usable for interaction analysis.
            if pose is not None:
                parent.active_docked_pdbqt = pose
            if (i + 1) % 25 == 0:
                log.info(f"    Docked {i + 1} / {total} ({tag})")

    return results


def _dock_worker(
    payload: Tuple[str, str],
    dock_func: Callable,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
    use_vina: bool = True,
) -> Tuple["CompoundRecord", Optional[float]]:
    """
    Module-level docking wrapper so it can be pickled by ``ProcessPoolExecutor``.

    *payload* is ``(compound_id, smiles)``; the Mol is reconstructed here from
    SMILES. A fresh :class:`CompoundRecord` is built, docked by *dock_func*,
    and ``(record, energy)`` is returned. On any failure the error is logged
    with the ``compound_id`` and ``(record, None)`` is returned so the
    pipeline keeps going.
    """
    compound_id, smiles = payload
    rec = CompoundRecord(compound_id=compound_id, smiles=smiles)
    try:
        energy = dock_func(
            rec, receptor_pdbqt, center, box_size, work_dir, tag, use_vina,
        )
        # The active-site pose (record.active_docked_pdbqt) is set inside the
        # worker process (dock_compound, tag == "active"). Because the worker
        # runs in a separate ProcessPool, the path must be returned explicitly
        # (the parent record's attribute is not mutated across processes).
        return rec, energy, rec.active_docked_pdbqt
    except Exception as exc:
        log.warning(
            f"    Docking failed for {compound_id} ({tag}): {exc}. "
            "Returning (record, None) and continuing."
        )
        return rec, None, None

