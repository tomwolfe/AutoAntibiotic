#!/usr/bin/env python3
"""Induced-fit docking orchestration for top candidates.

This module provides a high-level orchestration function that runs IFD
on a list of CompoundRecord objects, persisting poses to output/ifd_poses/<CID>/
and updating the records with ifd_energy and ifd_pose_pdbqt fields.

Usage:
    from utils.ifd import run_ifd_orchestration

    records = [...]  # list of CompoundRecord
    results = run_ifd_orchestration(
        records=records,
        receptor_pdb="output/workdir/PBP2a_holo_clean.pdb",
        active_center=(23.95, 27.87, 88.53),
        active_box=(18.7, 29.1, 21.4),
        work_dir="output/workdir",
        n_iterations=3,
        output_dir="output",
    )
"""

from __future__ import annotations

import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


def run_ifd_orchestration(
    records: list,
    receptor_pdb: str,
    active_center: tuple,
    active_box: tuple,
    work_dir: str,
    n_iterations: int = 3,
    output_dir: Optional[str] = None,
) -> list:
    """Run induced-fit docking on a list of compound records.

    Args:
        records: List of CompoundRecord objects with active_docked_pdbqt paths.
        receptor_pdb: Path to the receptor PDB file.
        active_center: Grid box centre (x, y, z) in Angstroms.
        active_box: Grid box dimensions (dx, dy, dz) in Angstroms.
        work_dir: Scratch directory for intermediate files.
        n_iterations: Number of IFD iterations (default 3).
        output_dir: Output directory for IFD poses. If None, uses "output".

    Returns:
        List of updated CompoundRecord objects with ifd_energy and
        ifd_pose_pdbqt fields populated for successful IFD runs.
    """
    from utils.docking import dock_compound_induced_fit, _parse_pdbqt_heavy_coords

    if output_dir is None:
        output_dir = "output"

    output_path = Path(output_dir) / "ifd_poses"
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    n_success = 0

    for rec in records:
        pose_pdbqt = getattr(rec, "active_docked_pdbqt", None)
        if pose_pdbqt is None or not os.path.exists(pose_pdbqt):
            results.append(rec)
            continue

        ifd_energy, ifd_pose = dock_compound_induced_fit(
            rec, receptor_pdb, active_center, active_box,
            work_dir, rigid_pose_pdbqt=pose_pdbqt, tag="ifd",
            n_iterations=n_iterations,
        )

        if ifd_energy is not None and ifd_pose is not None:
            rec.ifd_energy = ifd_energy
            rec.ifd_pose_pdbqt = ifd_pose
            n_success += 1
            results.append(rec)

            # Persist the induced-fit pose
            cid_dir = output_path / rec.compound_id
            cid_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(ifd_pose, cid_dir / "ifd_pose.pdbqt")
            except Exception as exc:
                log.warning(f"  Could not persist IFD pose for {rec.compound_id}: {exc}")

            log.info(f"    {rec.compound_id}: IFD energy={ifd_energy:.2f} kcal/mol")
        else:
            rec.ifd_energy = None
            rec.ifd_pose_pdbqt = None
            results.append(rec)

    log.info(f"  IFD completed for {n_success}/{len(records)} candidates")
    return results
