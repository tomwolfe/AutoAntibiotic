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
    exhaustiveness: int = 32,
    num_modes: int = 3,
) -> Optional[float]:
    """
    Run a single Vina docking job. Returns best binding energy (kcal/mol)
    or None on failure.

    Args:
        exhaustiveness: Vina exhaustiveness (default 32 in science mode).
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
    exhaustiveness: int = 32,
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
        exhaustiveness: Vina exhaustiveness (default 32 in science mode).
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
    exhaustiveness: int = 16,
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
        exhaustiveness: Vina exhaustiveness parameter (default 16).

    Returns:
        List of ``(CompoundRecord, energy_or_None)`` tuples.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    if n_jobs is None:
        n_jobs = N_JOBS
    if dock_func is None:
        dock_func = dock_compound

    # Bind exhaustiveness to the dock_func so workers use the configured value
    dock_func_bound = partial(dock_func, exhaustiveness=exhaustiveness)

    # Lightweight payloads: pickling only (id, smiles) avoids shipping the Mol.
    payloads = [(rec.compound_id, rec.smiles) for rec in records]
    by_id = {rec.compound_id: rec for rec in records}

    results: List[Tuple["CompoundRecord", Optional[float]]] = []
    total = len(records)

    # In-process execution keeps small batches deterministic and testable.
    if n_jobs <= 1:
        for i, payload in enumerate(payloads):
            rec, energy, pose = _dock_worker(
                payload, dock_func_bound, receptor_pdbqt, center, box_size, work_dir, tag,
            )
            parent = by_id[rec.compound_id]
            results.append((parent, energy))
            # Propagate the active-site pose path back to the parent record so
            # downstream pose analysis (MMFF94 strain-interaction, H-bond flags,
            # mutation scan) can use it. Only the "active" tag produces a retained
            # pose.
            if pose is not None:
                parent.active_docked_pdbqt = pose
            if (i + 1) % 25 == 0:
                log.info(f"    Docked {i + 1} / {total} ({tag})")
        return results

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(
                _dock_worker, payload, dock_func_bound,
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


def rescore_mmff94_strain(
    record: "CompoundRecord",
    receptor_pdb: Optional[str] = None,
) -> Optional[float]:
    """
    MMFF94 strain-interaction rescoring score (NOT MM-GBSA).

    The score combines three physically motivated terms:
        score = e_strain + e_receptor_interaction + e_np

    where:
      - e_strain is the ligand strain energy (MMFF94 energy of the
        docked pose minus MMFF94 energy of the relaxed conformation).
      - e_receptor_interaction is an electrostatic proxy computed via
        distance-dependent dielectric interaction between Gasteiger
        ligand charges and a uniform -0.2 receptor atom charge proxy.
      - e_np is a non-polar solvation term estimated from TPSA.

    This is NOT an MM-GBSA (MM/Generalized Born Surface Area) score. It does
    NOT use Generalized Born or Poisson-Boltzmann electrostatics. The score is
    a dimensionless relative ranking value and should NOT be interpreted as a
    binding free energy in kcal/mol.

    When the docked pose is unavailable, returns None.

    A more negative score suggests a more favourable binding pose.
    The absolute value should not be interpreted as a true binding
    free energy; it is a relative ranking score for comparing candidates.

    Args:
        record: Compound record with a valid SMILES.
        receptor_pdb: Path to receptor PDB for protein-ligand
            interaction calculation.

    Returns:
        Approximate MMFF94 strain-interaction score (arbitrary units) or None.
    """
    if record is None:
        return None
    mol = record.mol if record.mol is not None else Chem.MolFromSmiles(record.smiles)
    if mol is None:
        return None

    from rdkit.Chem import AllChem, rdMolDescriptors

    try:
        pose_path = getattr(record, "active_docked_pdbqt", None)
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

        ff_min = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props)
        if ff_min is None:
            return None
        ff_min.Minimize(maxIts=500)
        e_lig_min = ff_min.CalcEnergy()

        if e_lig_bound is not None and abs(e_lig_bound) < 1e6:
            e_strain = e_lig_bound - e_lig_min
        else:
            e_strain = 0.0

        tpsa = rdMolDescriptors.CalcTPSA(mol_h)
        e_np = 0.01 * tpsa

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

        if e_lig_bound is None:
            return None
        score = e_strain + e_receptor_interaction + e_np
        if abs(score) > 700:
            return None
        return float(score)

    except Exception:
        return None


def rescore_mmffsa(
    record: "CompoundRecord",
    receptor_pdb: Optional[str] = None,
) -> Optional[float]:
    """Deprecated alias for :func:`rescore_mmff94_strain`."""
    import warnings
    warnings.warn(
        "rescore_mmffsa is deprecated; use rescore_mmff94_strain instead.",
        DeprecationWarning, stacklevel=2,
    )
    return rescore_mmff94_strain(record, receptor_pdb=receptor_pdb)


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


def set_pose_coordinates(mol: "Chem.Mol", pose_pdbqt: str) -> bool:
    """Overlay an RDKit mol (with H) heavy atoms onto a docked pose PDBQT.

    The docked pose PDBQT stores the ligand in the receptor coordinate frame.
    This sets the mol's heavy-atom positions (canonical order, H appended last
    by RDKit after ``AddHs``) to the pose coordinates so that MD starts from
    the actual docking pose rather than an arbitrary ETKDG conformer.

    Args:
        mol: RDKit mol with an embedded conformer (all atoms, including H).
        pose_pdbqt: Path to the docked pose PDBQT (first MODEL).

    Returns:
        True if heavy-atom coordinates were applied.
    """
    coords = _parse_pdbqt_heavy_coords(pose_pdbqt)
    if not coords:
        return False
    conf = mol.GetConformer()
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    n_pose = len(coords)
    if n_heavy != n_pose:
        log.warning(
            f"  set_pose_coordinates: mol heavy atoms ({n_heavy}) != pose "
            f"atoms ({n_pose}); pose overlay skipped"
        )
        return False
    for i in range(n_heavy):
        conf.SetAtomPosition(
            i,
            Chem.rdGeometry.Point3D(
                float(coords[i][0]), float(coords[i][1]), float(coords[i][2])
            ),
        )
    return True


def find_best_pose_pdbqt(compound_id: str, work_dir: str) -> Optional[str]:
    """Return the lowest-energy active-site docked pose PDBQT for a compound.

    Scans ``<work_dir>/<compound_id>_active_*_out.pdbqt`` and selects the file
    whose ``REMARK VINA RESULT`` energy is most negative (best pose across the
    multi-conformer screen).

    Args:
        compound_id: Compound ID.
        work_dir: Directory containing the docked pose files.

    Returns:
        Path to the best pose PDBQT, or None if none found.
    """
    import glob

    best_path = None
    best_energy = float("inf")
    for f in glob.glob(os.path.join(work_dir, f"{compound_id}_active_*_out.pdbqt")):
        energy = None
        try:
            with open(f) as fh:
                for line in fh:
                    if line.startswith("REMARK VINA RESULT"):
                        energy = float(line.split()[3])
                        break
        except (OSError, ValueError, IndexError):
            continue
        if energy is not None and energy < best_energy:
            best_energy = energy
            best_path = f
    return best_path


def _prepare_flexible_pdbqt(
    receptor_pdb: str,
    flex_residues: List[Tuple[str, int]],
    work_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Prepare rigid and flexible PDBQT files for Vina flexible docking.

    Extracts the specified residues from *receptor_pdb* into a separate flexible
    side-chain PDBQT, and writes the remaining (rigid) receptor as a second
    PDBQT. Returns ``(rigid_pdbqt, flex_pdbqt)``.

    Args:
        receptor_pdb: Cleaned receptor PDB file.
        flex_residues: List of ``(resname, resnum)`` tuples for flexible residues.
        work_dir: Scratch directory for intermediate files.

    Returns:
        ``(rigid_pdbqt_path, flex_pdbqt_path)`` or ``(None, None)`` on failure.
    """
    rigid_pdb = os.path.join(work_dir, "rigid_receptor.pdb")
    flex_pdb = os.path.join(work_dir, "flex_sidechains.pdb")
    rigid_pdbqt_path = os.path.join(work_dir, "rigid_receptor.pdbqt")
    flex_pdbqt_path = os.path.join(work_dir, "flex_sidechains.pdbqt")
    try:
        from Bio.PDB import PDBParser, PDBIO, Select
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("receptor", receptor_pdb)

        class FlexSelect(Select):
            def accept_residue(self, residue):
                rid = residue.get_id()
                for rname, rnum in flex_residues:
                    if residue.get_resname().strip().upper() == rname.upper() and rid[1] == rnum:
                        return True
                return False

        class RigidSelect(Select):
            def accept_residue(self, residue):
                rid = residue.get_id()
                for rname, rnum in flex_residues:
                    if residue.get_resname().strip().upper() == rname.upper() and rid[1] == rnum:
                        return False
                return True

        io = PDBIO()
        io.set_structure(struct)
        io.save(flex_pdb, FlexSelect())
        io.save(rigid_pdb, RigidSelect())

        # Convert to PDBQT using obabel
        subprocess.run(
            ["obabel", rigid_pdb, "-O", rigid_pdbqt_path, "-xr"],
            capture_output=True, timeout=300,
        )
        subprocess.run(
            ["obabel", flex_pdb, "-O", flex_pdbqt_path, "-xr"],
            capture_output=True, timeout=300,
        )
        if os.path.exists(rigid_pdbqt_path) and os.path.exists(flex_pdbqt_path):
            return rigid_pdbqt_path, flex_pdbqt_path
    except Exception as exc:
        log.warning(f"  Could not prepare flexible PDBQT: {exc}")
    return None, None


def dock_compound_flexible(
    record: "CompoundRecord",
    rigid_receptor_pdbqt: str,
    flex_receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "flex",
    timeout: Optional[int] = None,
    exhaustiveness: int = 16,
    num_modes: int = 5,
) -> Optional[float]:
    """Flexible-side-chain docking wrapper for Vina (``--flex``).

    Uses the rigid receptor PDBQT and a flexible side-chain PDBQT to allow
    specified receptor residues to move during docking. All other parameters
    mirror :func:`dock_compound`.

    Returns:
        Best binding energy (kcal/mol) or None on failure.
    """
    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            return None
        record.mol = mol

    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    lig_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_lig.pdbqt")
    out_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_out.pdbqt")

    mol_for_prep = Chem.AddHs(record.mol)
    if not mol_for_prep.GetNumConformers():
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        try:
            AllChem.EmbedMolecule(mol_for_prep, params)
        except Exception:
            pass
    if not prepare_ligand_pdbqt(mol_for_prep, lig_pdbqt):
        raise RuntimeError(f"PDBQT prep failed for {record.compound_id}")

    cmd = [
        "vina",
        "--receptor", rigid_receptor_pdbqt,
        "--flex", flex_receptor_pdbqt,
        "--ligand", lig_pdbqt,
        "--out", out_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.1f}",
        "--size_y", f"{box_size[1]:.1f}",
        "--size_z", f"{box_size[2]:.1f}",
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ]
    if timeout is None:
        timeout = VINA_TIMEOUT_S
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("1") and " " in stripped:
                parts = stripped.split()
                try:
                    energy = float(parts[1])
                    if energy > 0:
                        return None
                    return energy
                except (ValueError, IndexError):
                    continue
        return None
    except subprocess.TimeoutExpired:
        log.warning(f"  Vina flexible timeout ({timeout}s) for {record.compound_id}.")
        return None
    except Exception as exc:
        log.warning(f"  Vina flexible exception: {exc}")
        return None


def _find_flexible_residues(
    receptor_pdb: str,
    ligand_coords: List[np.ndarray],
    distance_cutoff: float = 5.0,
) -> List[Tuple[str, int]]:
    """Find all receptor residues within *distance_cutoff* Å of the ligand.

    Args:
        receptor_pdb: Path to receptor PDB file.
        ligand_coords: List of ligand heavy-atom coordinates as numpy arrays.
        distance_cutoff: Distance threshold in Å.

    Returns:
        List of ``(resname, resnum)`` tuples for flexible residues.
    """
    if not ligand_coords:
        return []
    lig_positions = np.array([c for c in ligand_coords])
    if lig_positions.ndim != 2 or lig_positions.shape[0] == 0:
        return []
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("receptor", receptor_pdb)
        flex_residues = []
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
                        except Exception:
                            continue
                        dists = np.linalg.norm(lig_positions - pos, axis=1)
                        if np.any(dists <= distance_cutoff):
                            flex_residues.append(
                                (residue.get_resname().strip(), rid[1])
                            )
                            break
        return flex_residues
    except Exception as exc:
        log.warning(f"  Could not find flexible residues: {exc}")
        return []


def dock_compound_induced_fit(
    record: "CompoundRecord",
    receptor_pdb: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    rigid_pose_pdbqt: Optional[str] = None,
    tag: str = "ifd",
    n_iterations: int = 3,
    timeout: Optional[int] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Iterative induced-fit docking with OpenMM minimization + Vina redocking.

    Algorithm (3 iterations):
      i.   Take the current best pose.
      ii.  Build an OpenMM system with the ligand + flexible residues
           unrestrained, all other protein atoms restrained
           (10 kcal/mol/Å² on backbone CA).
      iii. Minimize for 2000 steps (L-BFGS).
      iv.  Extract the minimized pocket coordinates.
      v.   Write a new receptor PDBQT from the minimized structure.
      vi.  Re-dock the ligand into the minimized pocket with Vina
           (exhaustiveness=32, num_modes=9).
      vii. Take the best pose. Repeat from (ii).

    Args:
        record: Compound record.
        receptor_pdb: Path to receptor PDB file (not PDBQT).
        center: Grid box centre for docking.
        box_size: Grid box dimensions.
        work_dir: Scratch directory.
        rigid_pose_pdbqt: Initial docked pose PDBQT. If None, uses
            ``record.active_docked_pdbqt``.
        tag: Label for temp files.
        n_iterations: Number of IFD iterations (default 3).
        timeout: Per-call Vina timeout override.

    Returns:
        ``(best_energy, final_pose_pdbqt)`` or ``(None, None)`` on failure.
    """
    import openmm
    from openmm import app, unit
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from openff.toolkit import Molecule as OffMolecule
    from rdkit import Chem
    from utils.structure_prep import assign_protonation_states, build_openmm_variant_list

    ligand_pdbqt = rigid_pose_pdbqt or getattr(record, "active_docked_pdbqt", None)
    if ligand_pdbqt is None or not os.path.exists(ligand_pdbqt):
        log.warning(f"  No initial pose for {record.compound_id}; skipping IFD")
        return None, None

    # Parse initial pose coordinates
    current_coords = _parse_pdbqt_heavy_coords(ligand_pdbqt)
    if not current_coords:
        log.warning(f"  Could not parse pose coordinates for {record.compound_id}")
        return None, None

    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    current_pose_path = ligand_pdbqt
    best_energy = None
    final_pose_path = None

    # Prepare ligand 3D structure for parameterization
    mol = Chem.MolFromSmiles(record.smiles)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return None, None
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    # Set coordinates from pose
    conf = mol.GetConformer()
    for i, coord in enumerate(current_coords):
        if i < mol.GetNumAtoms():
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                float(coord[0]), float(coord[1]), float(coord[2])
            ))

    for iteration in range(n_iterations):
        log.info(f"  IFD iteration {iteration + 1}/{n_iterations}")

        # Find flexible residues within 5 Å of current pose
        flex_residues = _find_flexible_residues(
            receptor_pdb, current_coords, distance_cutoff=5.0
        )
        flex_set = set(flex_residues)
        log.info(f"    Found {len(flex_residues)} flexible residues within 5 Å")

        # Build OpenMM system
        try:
            pdb = app.PDBFile(receptor_pdb)
            # PROPKA protonation
            propka_variants = assign_protonation_states(receptor_pdb, pH=7.4)
            modeller = app.Modeller(pdb.topology, pdb.positions)
            variant_list = build_openmm_variant_list(modeller.topology, propka_variants)
            modeller.addHydrogens(pH=7.4, variants=variant_list)

            # Receptor atom count AFTER addHydrogens (addHydrogens appends H
            # atoms to the original topology; the ligand is appended after).
            n_rec_atoms = modeller.topology.getNumAtoms()

            # Add ligand
            lig_pdb_path = os.path.join(work_dir, f"{safe_id}_{tag}_lig.pdb")
            Chem.MolToPDBFile(mol, lig_pdb_path)
            lig_pdb = app.PDBFile(lig_pdb_path)
            modeller.add(lig_pdb.topology, lig_pdb.positions)
            complex_top = modeller.topology
            complex_pos = modeller.positions

            # Parameterize
            off_mol = OffMolecule.from_rdkit(mol, allow_undefined_stereo=True)
            off_mol.assign_partial_charges(partial_charge_method="gasteiger")
            tg = SMIRNOFFTemplateGenerator(molecules=off_mol, forcefield="openff-2.0.0")
            ff = app.ForceField("amber14-all.xml")
            ff.registerTemplateGenerator(tg.generator)
            system = ff.createSystem(
                complex_top,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
            )

            # Apply restraints: 10 kcal/mol/Å² on backbone CA of non-flexible residues
            from utils.openmm_platform import position_restraint_force
            RESTRAINT_FORCE = 10.0  # kcal/mol/Å²
            # k is stored per-particle in internal OpenMM units (kJ/mol/nm²).
            RESTRAINT_FORCE_KJ = RESTRAINT_FORCE * 4.184 / (0.1 ** 2)
            restraint = position_restraint_force(RESTRAINT_FORCE_KJ, periodic=True)
            restraint.addPerParticleParameter("k")
            restraint.addPerParticleParameter("x0")
            restraint.addPerParticleParameter("y0")
            restraint.addPerParticleParameter("z0")

            n_restrained = 0
            for residue in pdb.topology.residues():
                res_key = (residue.name, int(residue.id))
                if res_key in flex_set:
                    continue  # Flexible residue — no restraint
                for atom in residue.atoms():
                    if atom.name == "CA":
                        pos = complex_pos[atom.index]
                        restraint.addParticle(
                            atom.index,
                            [RESTRAINT_FORCE_KJ, pos.x, pos.y, pos.z]
                        )
                        n_restrained += 1
                        break

            # Add restraint for ligand (keep near initial position)
            n_lig_atoms = mol.GetNumAtoms()
            for i in range(n_lig_atoms):
                idx = n_rec_atoms + i
                pos = complex_pos[idx]
                restraint.addParticle(
                    idx,
                    [RESTRAINT_FORCE_KJ, pos.x, pos.y, pos.z]
                )
                n_restrained += 1

            system.addForce(restraint)

            # Minimize
            integrator = openmm.LangevinIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
            )
            simulation = app.Simulation(complex_top, system, integrator)
            simulation.context.setPositions(complex_pos)
            simulation.minimizeEnergy(maxIterations=2000)
            state_min = simulation.context.getState(getPositions=True)
            min_pos = state_min.getPositions()

            # Extract minimized receptor coordinates
            rec_min_pdb = os.path.join(work_dir, f"{safe_id}_{tag}_iter{iteration}_rec.pdb")
            # Write minimized complex, then strip ligand
            min_complex_pdb = os.path.join(work_dir, f"{safe_id}_{tag}_iter{iteration}_complex.pdb")
            with open(min_complex_pdb, "w") as fh:
                app.PDBFile.writeFile(complex_top, min_pos, fh)

            # Write receptor-only PDB (receptor atoms + H, no ligand) by
            # writing the minimized complex and keeping only the first
            # n_rec_atoms ATOM/HETATM records (receptor + hydrogens).
            rec_top_pdb = os.path.join(work_dir, f"{safe_id}_{tag}_iter{iteration}_rec_only.pdb")
            try:
                with open(rec_top_pdb, "w") as fh_out:
                    n_written = 0
                    with open(min_complex_pdb) as fh_in:
                        for line in fh_in:
                            if line.startswith(("ATOM", "HETATM")):
                                if n_written >= n_rec_atoms:
                                    break
                                fh_out.write(line)
                                n_written += 1
                            else:
                                fh_out.write(line)
            except Exception as exc:
                log.warning(f"    Could not write minimized receptor PDB: {exc}")
                continue

            # Update ligand coordinates from minimized pose
            lig_min_coords = []
            for i in range(n_lig_atoms):
                idx = n_rec_atoms + i
                pos = min_pos[idx]
                lig_min_coords.append(np.array([
                    pos.value_in_unit(unit.angstrom) for _ in range(1)
                ]))
            # Update mol coords
            for i in range(n_lig_atoms):
                if i < mol.GetNumAtoms():
                    pos = min_pos[n_rec_atoms + i]
                    conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                        float(pos[0].value_in_unit(unit.angstrom)),
                        float(pos[1].value_in_unit(unit.angstrom)),
                        float(pos[2].value_in_unit(unit.angstrom)),
                    ))

            # Convert minimized receptor to PDBQT via obabel
            rec_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_iter{iteration}_rec.pdbqt")
            try:
                subprocess.run(
                    ["obabel", rec_top_pdb, "-O", rec_pdbqt, "-xr"],
                    capture_output=True, timeout=300,
                )
            except Exception as exc:
                log.warning(f"    obabel conversion failed: {exc}")
                continue

            if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) == 0:
                log.warning(f"    Receptor PDBQT was not created for iteration {iteration}")
                continue

            # Re-dock into minimized receptor
            lig_pdbqt_path = os.path.join(work_dir, f"{safe_id}_{tag}_iter{iteration}_lig.pdbqt")
            if not prepare_ligand_pdbqt(mol, lig_pdbqt_path):
                log.warning(f"    Ligand PDBQT prep failed for iteration {iteration}")
                continue

            out_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_iter{iteration}_out.pdbqt")
            energy = _run_vina_docking(
                rec_pdbqt, lig_pdbqt_path, out_pdbqt,
                center, box_size,
                timeout=timeout,
                exhaustiveness=32,
                num_modes=9,
            )

            if energy is not None and (best_energy is None or energy < best_energy):
                best_energy = energy
                final_pose_path = out_pdbqt
                current_pose_path = out_pdbqt
                # Update current coords for next iteration
                current_coords = _parse_pdbqt_heavy_coords(out_pdbqt)
                if current_coords:
                    for i, coord in enumerate(current_coords):
                        if i < mol.GetNumAtoms():
                            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                                float(coord[0]), float(coord[1]), float(coord[2])
                            ))
                log.info(f"    Iteration {iteration + 1}: energy = {energy:.2f} kcal/mol")

        except Exception as exc:
            log.warning(f"    IFD iteration {iteration + 1} failed: {exc}")
            continue

    if best_energy is not None:
        log.info(f"  IFD complete: best energy = {best_energy:.2f} kcal/mol")
    else:
        log.warning(f"  IFD failed to produce any valid pose for {record.compound_id}")

    return best_energy, final_pose_path


def dock_compound_ifd_catalytic(
    record: "CompoundRecord",
    receptor_pdb: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    rigid_pose_pdbqt: Optional[str] = None,
    catalytic_residues: Optional[List[Tuple[str, int]]] = None,
    n_iterations: int = 3,
    exhaustiveness: int = 32,
    timeout: Optional[int] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Light induced-fit docking with catalytic triad side-chain sampling.

    Like :func:`dock_compound_induced_fit`, but guarantees the PBP2a catalytic
    residues (Ser403, Lys406, Tyr446) are included in the flexible residue set
    regardless of their distance from the docked pose. This addresses the
    rigid-docking limitation where PBP2a's dynamic active-site loop prevents
    accurate scoring without side-chain relaxation.

    Algorithm (n_iterations):
      i.   Take the current best pose.
      ii.  Find residues within 5 Å of the ligand, UNION with catalytic_residues.
      iii. OpenMM minimize (2000 steps, L-BFGS) with non-flexible residues
           restrained (10 kcal/mol/Å² on backbone CA).
      iv.  Re-dock the ligand into the minimized pocket (Vina).

    Args:
        record: Compound record.
        receptor_pdb: Path to receptor PDB file.
        center: Grid box centre for docking.
        box_size: Grid box dimensions.
        work_dir: Scratch directory.
        rigid_pose_pdbqt: Initial docked pose PDBQT.
        catalytic_residues: List of ``(resname, resnum)`` tuples to always
            treat as flexible. Defaults to PBP2a catalytic triad.
        n_iterations: Number of IFD iterations (default 3).
        exhaustiveness: Vina exhaustiveness for re-docking (default 32).
        timeout: Per-call Vina timeout override.

    Returns:
        ``(best_energy, final_pose_pdbqt)`` or ``(None, None)`` on failure.
    """
    if catalytic_residues is None:
        catalytic_residues = [("SER", 403), ("LYS", 406), ("TYR", 446)]

    ligand_pdbqt = rigid_pose_pdbqt or getattr(record, "active_docked_pdbqt", None)
    if ligand_pdbqt is None or not os.path.exists(ligand_pdbqt):
        log.warning(f"  No initial pose for {record.compound_id}; skipping IFD")
        return None, None

    current_coords = _parse_pdbqt_heavy_coords(ligand_pdbqt)
    if not current_coords:
        log.warning(f"  Could not parse pose coordinates for {record.compound_id}")
        return None, None

    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    current_pose_path = ligand_pdbqt
    best_energy = None
    final_pose_path = None

    mol = Chem.MolFromSmiles(record.smiles)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return None, None
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    conf = mol.GetConformer()
    for i, coord in enumerate(current_coords):
        if i < mol.GetNumAtoms():
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                float(coord[0]), float(coord[1]), float(coord[2])
            ))

    try:
        import openmm
        from openmm import app, unit
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        from openff.toolkit import Molecule as OffMolecule
        from utils.structure_prep import assign_protonation_states, build_openmm_variant_list
        from utils.openmm_platform import position_restraint_force
    except ImportError as exc:
        log.warning(f"  OpenMM dependencies missing for IFD: {exc}")
        return None, None

    RESTRAINT_FORCE = 10.0
    RESTRAINT_FORCE_KJ = RESTRAINT_FORCE * 4.184 / (0.1 ** 2)

    for iteration in range(n_iterations):
        log.info(f"  IFD-catalytic iteration {iteration + 1}/{n_iterations}")

        # Find residues near ligand AND force catalytic residues as flexible
        flex_near = _find_flexible_residues(
            receptor_pdb, current_coords, distance_cutoff=5.0
        )
        flex_set = set(flex_near) | set(catalytic_residues)

        log.info(f"    Flexible residues: {len(flex_set)} "
                 f"({len(flex_near)} near ligand + {len(catalytic_residues)} catalytic)")

        try:
            pdb = app.PDBFile(receptor_pdb)
            propka_variants = assign_protonation_states(receptor_pdb, pH=7.4)
            modeller = app.Modeller(pdb.topology, pdb.positions)
            variant_list = build_openmm_variant_list(modeller.topology, propka_variants)
            modeller.addHydrogens(pH=7.4, variants=variant_list)

            n_rec_atoms = modeller.topology.getNumAtoms()

            lig_pdb_path = os.path.join(work_dir, f"{safe_id}_ifdc_it{iteration}_lig.pdb")
            Chem.MolToPDBFile(mol, lig_pdb_path)
            lig_pdb = app.PDBFile(lig_pdb_path)
            modeller.add(lig_pdb.topology, lig_pdb.positions)
            complex_top = modeller.topology
            complex_pos = modeller.positions

            off_mol = OffMolecule.from_rdkit(mol, allow_undefined_stereo=True)
            off_mol.assign_partial_charges(partial_charge_method="gasteiger")
            tg = SMIRNOFFTemplateGenerator(molecules=off_mol, forcefield="openff-2.0.0")
            ff = app.ForceField("amber14-all.xml")
            ff.registerTemplateGenerator(tg.generator)
            system = ff.createSystem(
                complex_top,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
            )

            restraint = position_restraint_force(RESTRAINT_FORCE_KJ, periodic=True)
            restraint.addPerParticleParameter("k")
            restraint.addPerParticleParameter("x0")
            restraint.addPerParticleParameter("y0")
            restraint.addPerParticleParameter("z0")

            for residue in pdb.topology.residues():
                res_key = (residue.name, int(residue.id))
                if res_key in flex_set:
                    continue
                for atom in residue.atoms():
                    if atom.name == "CA":
                        pos = complex_pos[atom.index]
                        restraint.addParticle(
                            atom.index,
                            [RESTRAINT_FORCE_KJ, pos.x, pos.y, pos.z]
                        )
                        break

            n_lig_atoms = mol.GetNumAtoms()
            for i in range(n_lig_atoms):
                idx = n_rec_atoms + i
                pos = complex_pos[idx]
                restraint.addParticle(
                    idx,
                    [RESTRAINT_FORCE_KJ, pos.x, pos.y, pos.z]
                )

            system.addForce(restraint)

            integrator = openmm.LangevinIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
            )
            simulation = app.Simulation(complex_top, system, integrator)
            simulation.context.setPositions(complex_pos)
            simulation.minimizeEnergy(maxIterations=2000)
            state_min = simulation.context.getState(getPositions=True)
            min_pos = state_min.getPositions()

            min_complex_pdb = os.path.join(work_dir, f"{safe_id}_ifdc_it{iteration}_cx.pdb")
            with open(min_complex_pdb, "w") as fh:
                app.PDBFile.writeFile(complex_top, min_pos, fh)

            rec_top_pdb = os.path.join(work_dir, f"{safe_id}_ifdc_it{iteration}_rec.pdb")
            try:
                with open(rec_top_pdb, "w") as fh_out:
                    n_written = 0
                    with open(min_complex_pdb) as fh_in:
                        for line in fh_in:
                            if line.startswith(("ATOM", "HETATM")):
                                if n_written >= n_rec_atoms:
                                    break
                                fh_out.write(line)
                                n_written += 1
                            else:
                                fh_out.write(line)
            except Exception as exc:
                log.warning(f"    Could not write minimized receptor: {exc}")
                continue

            for i in range(n_lig_atoms):
                if i < mol.GetNumAtoms():
                    pos = min_pos[n_rec_atoms + i]
                    conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                        float(pos[0].value_in_unit(unit.angstrom)),
                        float(pos[1].value_in_unit(unit.angstrom)),
                        float(pos[2].value_in_unit(unit.angstrom)),
                    ))

            rec_pdbqt = os.path.join(work_dir, f"{safe_id}_ifdc_it{iteration}_rec.pdbqt")
            try:
                subprocess.run(
                    ["obabel", rec_top_pdb, "-O", rec_pdbqt, "-xr"],
                    capture_output=True, timeout=300,
                )
            except Exception as exc:
                log.warning(f"    obabel failed: {exc}")
                continue

            if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) == 0:
                log.warning(f"    Receptor PDBQT not created for iteration {iteration}")
                continue

            lig_pdbqt_path = os.path.join(work_dir, f"{safe_id}_ifdc_it{iteration}_lig.pdbqt")
            if not prepare_ligand_pdbqt(mol, lig_pdbqt_path):
                continue

            out_pdbqt = os.path.join(work_dir, f"{safe_id}_ifdc_it{iteration}_out.pdbqt")
            energy = _run_vina_docking(
                rec_pdbqt, lig_pdbqt_path, out_pdbqt,
                center, box_size,
                timeout=timeout,
                exhaustiveness=exhaustiveness,
                num_modes=9,
            )

            if energy is not None and (best_energy is None or energy < best_energy):
                best_energy = energy
                final_pose_path = out_pdbqt
                current_pose_path = out_pdbqt
                current_coords = _parse_pdbqt_heavy_coords(out_pdbqt)
                if current_coords:
                    for i, coord in enumerate(current_coords):
                        if i < mol.GetNumAtoms():
                            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                                float(coord[0]), float(coord[1]), float(coord[2])
                            ))
                log.info(f"    Iteration {iteration + 1}: IFD-energy = {energy:.2f} kcal/mol")

        except Exception as exc:
            log.warning(f"    IFD-catalytic iteration {iteration + 1} failed: {exc}")
            continue

    if best_energy is not None:
        log.info(f"  IFD-catalytic complete: best energy = {best_energy:.2f} kcal/mol")
    else:
        log.warning(f"  IFD-catalytic failed for {record.compound_id}")

    return best_energy, final_pose_path


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

