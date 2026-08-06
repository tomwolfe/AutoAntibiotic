#!/usr/bin/env python3
"""
Conserved active-site water network analysis across production MD trajectories.

The DUD-E rigid-docking benchmark fails for PBP2a in part because the shallow,
polar, solvent-exposed active site is water-mediated: the protocol strips all
waters, so conserved water molecules and their displacement penalties are not
modelled (see output/conserved_waters.json, which found conserved water
positions supporting 1VQQ / 4DKI).

This script streams each candidate's production trajectory (DCD + solvated
topology) frame by frame and:

  1. For every water oxygen, computes the fraction of sampled frames within
     ``CATALYTIC_CUTOFF_A`` (5.5 A) of the catalytic triad
     (Ser403.OG / Lys406.NZ / Tyr446.OH)  ->  *occupancy*.
  2. For each such site water, tracks contiguous residence through the
     trajectory (waters are identity-stable by atom index within one replica),
     reporting the longest continuous stay as *residence_time_ps*.
  3. Tests, per candidate/replica, whether the previously identified conserved
     water positions (output/conserved_waters.json) are still occupied (a
     water oxygen within ``CONSERVED_MATCH_A``, default 1.5 A) and whether the
     ligand heavy atoms displace them.

Memory-safe: the DCD is streamed frame by frame (never materialised in RAM), so
this runs on full 100 ns / 426k-atom trajectories. A frame stride
(``--frame``) bounds the per-replica runtime on very long trajectories.

Usage:
    python scripts/water_analysis.py                     # all candidates
    python scripts/water_analysis.py --cid BRICS_0022
    python scripts/water_analysis.py --frame 10          # sample every 10th frame

Outputs:
    output/water_analysis.json   - per-candidate / per-replica summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("water_analysis")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
MD_OUT = OUT / "md_explicit"
CONSERVED_JSON = OUT / "conserved_waters.json"
RECEPTOR_PDB = OUT / "workdir" / "PBP2a_holo_clean.pdb"

CATALYTIC = {
    "SER403_OG": ("SER", 403, "OG"),
    "LYS406_NZ": ("LYS", 406, "NZ"),
    "TYR446_OH": ("TYR", 446, "OH"),
}
CATALYTIC_CUTOFF_A = 5.5
CONSERVED_MATCH_A = 1.5
LIGAND_DISPLACE_A = 2.0
WATER_NAMES = {"HOH", "WAT", "H2O", "TIP3"}
LIGAND_NAMES = {"LIG", "UNL", "MOL", "MOL2"}
DEFAULT_FRAME_STRIDE = 5


def _load_conserved():
    if not CONSERVED_JSON.is_file():
        log.warning("conserved_waters.json not found; disabling conserved checks")
        return []
    data = json.loads(CONSERVED_JSON.read_text())
    return [c["coord_A"] for c in (data.get("conserved_waters") or [])]


def _reference_catalytic_coords():
    """Reference (nm) coordinates of the catalytic atoms in the clean receptor."""
    import openmm.app as app
    pdb = app.PDBFile(str(RECEPTOR_PDB))
    refs = {}
    for label, (resname, resnum, atomname) in CATALYTIC.items():
        for residue in pdb.topology.residues():
            if residue.name == resname and str(residue.id) == str(resnum):
                for atom in residue.atoms():
                    if atom.name == atomname:
                        p = pdb.positions[atom.index]
                        refs[label] = (atomname, p.x, p.y, p.z)
                        break
                break
    missing = [k for k in CATALYTIC if k not in refs]
    if missing:
        raise RuntimeError(f"Could not locate catalytic atoms {missing} in {RECEPTOR_PDB}")
    return refs


def _find_catalytic_indices(pdb):
    """Match catalytic atoms in a solvated (renumbered) topology by coordinates."""
    refs = _reference_catalytic_coords()
    indices = []
    positions = np.array([[v.x, v.y, v.z] for v in pdb.positions]) * 10.0  # nm -> A
    for label, (atomname, rx, ry, rz) in refs.items():
        ref = np.array([rx, ry, rz]) * 10.0
        best_idx, best_d = None, float("inf")
        for residue in pdb.topology.residues():
            for atom in residue.atoms():
                if atom.name != atomname:
                    continue
                d = float(np.linalg.norm(positions[atom.index] - ref))
                if d < best_d:
                    best_d, best_idx = d, atom.index
        if best_idx is None or best_d > 2.0:
            raise RuntimeError(
                f"catalytic atom {label} not matched by coordinates (min d={best_d:.2f})")
        indices.append(best_idx)
    return indices


def _frame_dt_ps(replica_dir: Path) -> float:
    try:
        summary = replica_dir / "summary.json"
        if summary.is_file():
            rs = json.loads(summary.read_text()).get("production", {})
            n_frames = rs.get("n_frames")
            npt_ns = rs.get("npt_duration_ns")
            if n_frames and npt_ns:
                return float(npt_ns) * 1000.0 / float(n_frames)
    except Exception:
        pass
    return 10.0  # default report interval (5000 steps x 2 fs)


def analyze_replica(candidate_dir: Path, replica_dir: Path,
                    conserved_positions: list, frame_stride: int) -> dict:
    """Stream one replica's DCD and compute water-network statistics."""
    import openmm.app as app

    dcd = replica_dir / "trajectory.dcd"
    top_pdb = replica_dir / "topology.pdb"
    out = {
        "replica": replica_dir.name,
        "dcd_found": dcd.is_file(),
        "topology_found": top_pdb.is_file(),
        "n_frames_analysed": 0,
        "n_site_waters_detected": 0,
        "site_waters": [],
        "max_residence_ps": None,
        "conserved": {},
    }
    if not (dcd.is_file() and top_pdb.is_file()):
        return out

    try:
        pdb = app.PDBFile(str(top_pdb))
        # Solvated topologies renumber residues; locate the catalytic triad by
        # matching coordinates against the clean receptor reference.
        catal_atoms = _find_catalytic_indices(pdb)
    except Exception as exc:
        out["error"] = f"topology/catalytic-atom resolution failed: {exc}"
        return out

    water_oxy = []
    for residue in pdb.topology.residues():
        if residue.name in WATER_NAMES:
            for atom in residue.atoms():
                if atom.name in ("O", "OW", "O1"):
                    water_oxy.append(atom.index)
                    break
    water_oxy_arr = np.asarray(water_oxy, dtype=np.int64)

    lig_atoms = []
    for residue in pdb.topology.residues():
        if residue.name in LIGAND_NAMES:
            lig_atoms.extend(a.index for a in residue.atoms())
    lig_arr = np.asarray(lig_atoms, dtype=np.int64)
    if lig_arr.size == 0:
        for residue in reversed(list(pdb.topology.residues())):
            if residue.name not in WATER_NAMES:
                lig_arr = np.asarray([a.index for a in residue.atoms()], dtype=np.int64)
                break

    dt_ps = _frame_dt_ps(replica_dir)
    n_contact = np.zeros(len(water_oxy), dtype=np.int64)
    run = np.zeros(len(water_oxy), dtype=np.int64)
    max_run = np.zeros(len(water_oxy), dtype=np.int64)
    conserved_occ = np.zeros(len(conserved_positions), dtype=np.int64)
    n_lig_disp = 0
    n_frames = 0
    import mdtraj as md
    try:
        md_top = md.Topology.from_openmm(pdb.topology)
        # Stream the DCD in chunks (never materialise the full trajectory);
        # the stride samples every Nth frame directly on disk.
        for chunk in md.iterload(str(dcd), top=md_top, chunk=50, stride=frame_stride):
            xyz_a = chunk.xyz * 10.0  # nm -> A
            for positions in xyz_a:
                n_frames += 1
                catal_xyz = positions[catal_atoms]           # (3, 3)
                water_xyz = positions[water_oxy_arr] if water_oxy_arr.size else np.zeros((0, 3))

                if water_xyz.shape[0]:
                    d = np.linalg.norm(catal_xyz[:, None, :] - water_xyz[None, :, :], axis=2)
                    in_contact = d.min(axis=0) < CATALYTIC_CUTOFF_A   # per water O
                else:
                    in_contact = np.zeros(0, dtype=bool)

                n_contact += in_contact
                run = (run + 1) * in_contact
                max_run = np.maximum(max_run, run)

                if conserved_positions and water_xyz.shape[0]:
                    con = np.asarray(conserved_positions, dtype=float)
                    dc = np.linalg.norm(con[:, None, :] - water_xyz[None, :, :], axis=2)
                    conserved_occ += (dc.min(axis=1) < CONSERVED_MATCH_A).astype(np.int64)

                if conserved_positions and lig_arr.size:
                    lig_xyz = positions[lig_arr]
                    dc_lig = np.linalg.norm(
                        np.asarray(conserved_positions, dtype=float)[:, None, :]
                        - lig_xyz[None, :, :], axis=2)
                    n_lig_disp += int(np.any(dc_lig.min(axis=1) < LIGAND_DISPLACE_A))
    except Exception as exc:
        # Typically a topology/DCD atom-count mismatch (stale topology.pdb
        # from an earlier solvation box): re-running explicit_solvent_md with
        # the same command regenerates a consistent topology+DCD pair. Report
        # it and skip this replica.
        out["error"] = f"DCD streaming failed (stale topology?): {exc}"
        return out

    if n_frames == 0:
        out["error"] = "no frames read"
        return out

    site = []
    for i in range(len(water_oxy)):
        if n_contact[i] == 0:
            continue
        site.append({
            "atom_index": int(water_oxy[i]),
            "occupancy": round(float(n_contact[i]) / n_frames, 3),
            "n_frames_contact": int(n_contact[i]),
            "max_contiguous_frames": int(max_run[i]),
            "residence_time_ps": round(float(max_run[i] * dt_ps * frame_stride), 1),
        })
    site.sort(key=lambda w: -w["occupancy"])

    out["n_frames_analysed"] = n_frames
    out["n_site_waters_detected"] = len(site)
    out["site_waters"] = site[:20]
    out["max_residence_ps"] = (
        round(float(max_run.max() * dt_ps * frame_stride), 1) if max_run.size else None
    )
    out["conserved"] = {
        "n_positions": len(conserved_positions),
        "occupancy": [round(float(o) / n_frames, 3) for o in conserved_occ],
        "ligand_displacement_fraction": round(float(n_lig_disp) / n_frames, 3),
        "catalytic_cutoff_A": CATALYTIC_CUTOFF_A,
        "conserved_match_A": CONSERVED_MATCH_A,
    }
    return out


def analyze_candidate(candidate_dir: Path, frame_stride: int) -> dict:
    conserved = _load_conserved()
    result = {"compound_id": candidate_dir.name, "n_replicas": 0, "replicas": []}
    reps = sorted((d for d in candidate_dir.glob("replica_*") if d.is_dir()),
                  key=lambda d: d.name)
    for rep in reps:
        result["replicas"].append(analyze_replica(candidate_dir, rep, conserved, frame_stride))
    result["n_replicas"] = len(reps)
    return result


def main():
    parser = argparse.ArgumentParser(description="Conserved active-site water analysis")
    parser.add_argument("--cid", type=str, default=None,
                        help="Compound ID to analyse (default: all candidates with DCDs)")
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME_STRIDE,
                        help="Sample every Nth frame of the DCD (default: %d)" % DEFAULT_FRAME_STRIDE)
    args = parser.parse_args()

    if not MD_OUT.is_dir():
        log.error("No MD output at %s", MD_OUT)
        sys.exit(1)

    candidates = sorted(d for d in MD_OUT.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
    if args.cid:
        candidates = [c for c in candidates if c.name == args.cid]
        if not candidates:
            log.error("Candidate %s not found", args.cid)
            sys.exit(1)

    all_results = [analyze_candidate(cand, args.frame) for cand in candidates]
    (OUT / "water_analysis.json").write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Wrote %s", OUT / "water_analysis.json")

    for res in all_results:
        for rep in res["replicas"]:
            log.info("  %s %s: %d site-water(s), max residence=%s ps",
                     res["compound_id"], rep["replica"],
                     rep.get("n_site_waters_detected", 0),
                     rep.get("max_residence_ps"))
    sys.exit(0)


if __name__ == "__main__":
    main()