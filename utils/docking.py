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
from rdkit.Chem import AllChem

from .ligand_prep import prepare_ligand_pdbqt
from .library_gen import CompoundRecord
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
    exhaustiveness: int = 8,
    num_modes: int = 3,
) -> Optional[float]:
    """
    Run a single Vina docking job. Returns best binding energy (kcal/mol)
    or None on failure.

    Args:
        exhaustiveness: Vina exhaustiveness (default 8, use 32 for CES1).
        num_modes: Number of output modes (default 3, use 9 for CES1).
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
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
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
                    if energy > 0:
                        return None
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
    timeout: Optional[int] = None,
    exhaustiveness: int = 8,
    num_modes: int = 3,
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
        timeout: Optional per-call Vina timeout override (seconds).
        exhaustiveness: Vina exhaustiveness (default 8).
        num_modes: Number of output modes (default 3).

    Returns:
        Best binding energy (Vina) or None on failure.
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
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
    )

    is_active_pose = tag == "active" or tag.startswith("active_")
    dock_succeeded = (
        energy is not None
        and os.path.exists(out_pdbqt)
        and os.path.getsize(out_pdbqt) > 0
    )
    if is_active_pose and dock_succeeded:
        record.active_docked_pdbqt = out_pdbqt

    for f in (lig_pdbqt, out_pdbqt):
        if is_active_pose and dock_succeeded and f == out_pdbqt:
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
                receptor_pdbqt, center, box_size, work_dir, tag,
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


def rescore_mmffsa(
    record: "CompoundRecord",
    receptor_pdb: Optional[str] = None,
) -> Optional[float]:
    """
    Compute a relative MMFF94 strain-aware rescoring score.

    The score combines three physically motivated terms:
        score = e_strain + e_receptor_interaction + e_np

    where:
      - e_strain is the ligand strain energy (MMFF94 energy of the
        docked pose minus MMFF94 energy of the relaxed conformation).
      - e_receptor_interaction is an electrostatic proxy computed via
        distance-dependent dielectric interaction between Gasteiger
        ligand charges and a uniform -0.2 receptor atom charge proxy.
      - e_np is a non-polar solvation term estimated from TPSA.

    When the docked pose is unavailable, returns None.

    A more negative score suggests a more favourable binding pose.
    The absolute value should not be interpreted as a true binding
    free energy; it is a relative ranking score for comparing candidates.

    Args:
        record: Compound record with a valid SMILES.
        receptor_pdb: Path to receptor PDB for protein-ligand
            interaction calculation.

    Returns:
        Approximate protein-ligand MM-GBSA score (kcal/mol) or None.
    """
    mol = record.mol if record.mol is not None else Chem.MolFromSmiles(record.smiles)
    if mol is None:
        return None

    from rdkit.Chem import AllChem, rdMolDescriptors
    from io import StringIO

    try:
        # --- Compute ligand energy in its docked (bound) conformation ---
        pose_path = getattr(record, "active_docked_pdbqt", None) if record is not None else None
        docked_coords = None

        if pose_path and os.path.exists(pose_path):
            try:
                docked_coords = _parse_pdbqt_heavy_coords(pose_path)
            except Exception:
                docked_coords = None

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol_h, params)
        if status != 0:
            return None

        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, "MMFF94")
            if mmff_props is None:
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, "MMFF94s")
            if mmff_props is None:
                return None
        except Exception:
            return None

        # Compute ligand energy in its docked (bound) conformation
        e_lig_bound = None
        if docked_coords is not None:
            n_heavy_mol = mol_h.GetNumHeavyAtoms()
            if len(docked_coords) == n_heavy_mol:
                try:
                    conf = mol_h.GetConformer()
                    for i in range(n_heavy_mol):
                        idx = _get_heavy_atom_index(mol_h, i)
                        if idx is not None:
                            conf.SetAtomPosition(
                                idx,
                                Chem.rdGeometry.Point3D(
                                    float(docked_coords[i][0]),
                                    float(docked_coords[i][1]),
                                    float(docked_coords[i][2]),
                                ),
                            )
                    ff_bound = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props)
                    if ff_bound is not None:
                        e_lig_bound = ff_bound.CalcEnergy()
                except Exception:
                    e_lig_bound = None

        # Compute ligand energy in its minimised (relaxed) conformation
        ff_min = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props)
        if ff_min is None:
            return None
        ff_min.Minimize(maxIts=500)
        e_lig_min = ff_min.CalcEnergy()

        # Ligand strain energy (docked vs minimised)
        if e_lig_bound is not None and abs(e_lig_bound) < 1e6:
            e_strain = e_lig_bound - e_lig_min
        else:
            e_strain = 0.0

        # --- Non-polar solvation ---
        tpsa = rdMolDescriptors.CalcTPSA(mol_h)
        e_np = 0.01 * tpsa

        # --- Compute Gasteiger charges ---
        AllChem.ComputeGasteigerCharges(mol_h)
        n_heavy = mol_h.GetNumHeavyAtoms()
        lig_charges = []
        heavy_atom_indices = []
        for i in range(mol_h.GetNumAtoms()):
            if mol_h.GetAtomWithIdx(i).GetAtomicNum() == 1:
                continue
            heavy_atom_indices.append(i)
            try:
                q = float(mol_h.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
                lig_charges.append(q)
            except (ValueError, KeyError):
                lig_charges.append(0.0)

        # --- Distance-dependent dielectric solvation (ligand self) ---
        e_solv = 0.0
        conf = mol_h.GetConformer()
        for i in range(n_heavy):
            for j in range(i + 1, n_heavy):
                qi = lig_charges[i] if i < len(lig_charges) else 0.0
                qj = lig_charges[j] if j < len(lig_charges) else 0.0
                if abs(qi) < 1e-6 or abs(qj) < 1e-6:
                    continue
                pi = conf.GetAtomPosition(heavy_atom_indices[i])
                pj = conf.GetAtomPosition(heavy_atom_indices[j])
                r = pi.Distance(pj)
                if r < 0.5:
                    continue
                e_solv += 332.0 * qi * qj / (80.0 * r)

        # --- Protein-ligand interaction (when receptor PDB is available) ---
        e_receptor_interaction = 0.0
        if receptor_pdb and os.path.exists(receptor_pdb) and docked_coords is not None:
            try:
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True)
                struct = parser.get_structure("receptor", receptor_pdb)
                receptor_atoms = []
                for model in struct:
                    for chain in model:
                        for residue in chain:
                            rid = residue.get_id()
                            if rid[0] != " ":
                                continue
                            for atom in residue:
                                if atom.element and atom.element.upper() == "H":
                                    continue
                                try:
                                    pos = atom.get_vector().get_array()
                                    receptor_atoms.append(pos)
                                except Exception:
                                    pass
                if receptor_atoms:
                    rec_coords = np.array(receptor_atoms)
                    # Compute distance-dependent dielectric interaction
                    # Use vectorized calculation with strict distance clipping
                    # to prevent numerical blow-up from near-zero distances.
                    for li in range(len(docked_coords)):
                        if li >= len(lig_charges):
                            break
                        qi = lig_charges[li]
                        if abs(qi) < 1e-4:
                            continue
                        lpos = np.array(docked_coords[li])
                        dists = np.linalg.norm(rec_coords - lpos, axis=1)
                        valid_mask = (dists >= 0.5) & (dists <= 10.0)
                        valid_dists = dists[valid_mask]
                        if len(valid_dists) > 0:
                            e_receptor_interaction += float(
                                np.sum(332.0 * qi * (-0.2) / (4.0 * valid_dists))
                            )
            except Exception:
                e_receptor_interaction = 0.0

        # --- Composite score ---
        # ΔG ≈ e_strain + e_receptor_interaction + e_np
        # Returns a relative ranking score (strain penalty + interaction + solvation).
        # Lower (more negative) suggests a more favourable binding pose.
        if e_lig_bound is None:
            return None
        score = e_strain + e_receptor_interaction + e_np
        # Sanity bound: discard any pathological score beyond ±700 a.u.
        if abs(score) > 700:
            return None
        return float(score)

    except Exception:
        return None


def _get_heavy_atom_index(mol: Chem.Mol, heavy_idx: int) -> Optional[int]:
    """Return the atom index of the *heavy_idx*-th heavy atom in *mol*."""
    count = 0
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() > 1:
            if count == heavy_idx:
                return i
            count += 1
    return None


def _parse_pdbqt_heavy_coords(pdbqt_path: str) -> List[np.ndarray]:
    """Parse heavy-atom 3D coordinates from a PDBQT file. Skips hydrogens.
    
    Only reads the first MODEL (if multiple models are present), so the
    returned coordinates correspond to a single docking pose."""
    coords: List[np.ndarray] = []
    try:
        with open(pdbqt_path) as f:
            in_first_model = True
            for line in f:
                if line.startswith("MODEL"):
                    if "MODEL 1" in line or "MODEL 0" in line:
                        in_first_model = True
                    elif any(f"MODEL {i}" in line for i in range(2, 10)):
                        in_first_model = False
                    continue
                if line.startswith("ENDMDL"):
                    if in_first_model:
                        break
                    continue
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                if not in_first_model:
                    continue
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    elem = line[76:78].strip()
                except (ValueError, IndexError):
                    continue
                if elem and elem.upper() != "H":
                    coords.append(np.array([x, y, z]))
    except OSError:
        pass
    return coords


def _dock_worker(
    payload: Tuple[str, str],
    dock_func: Callable,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
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
            rec, receptor_pdbqt, center, box_size, work_dir, tag,
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

