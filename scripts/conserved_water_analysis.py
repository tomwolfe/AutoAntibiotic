#!/usr/bin/env python3
"""
Conserved active-site water analysis for the PBP2a crystal structures.

Rigid-receptor Vina docking is performed against *water-stripped* receptors:
``clean_pdb_structure`` removes all HETATM records (including ordered waters)
before PDBQT preparation. If ordered waters occupy the active site in the
source crystal structures, the docking protocol cannot represent
water-mediated contacts or displacement penalties -- a plausible contributor
to the poor DUD-E enrichment (AUC ~ 0.13) reported for this target.

This script characterises that limitation directly:

  1. Parses the source crystal structures (1VQQ apo, 3ZG0 holo, 4DKI holo).
  2. Locates the catalytic triad (Ser403.OG / Lys406.NZ / Tyr446.OH).
  3. Finds ordered water molecules within a cutoff of any catalytic atom.
  4. Superposes all structures onto the reference (1VQQ, chain A) using a
     Kabsch fit on active-site C-alpha atoms, then clusters the superposed
     waters to identify *conserved* water positions present in >= 2
     structures.

The finding is reported as a JSON verdict and, if conserved waters exist,
they are exported as a PDB (in the reference frame) so users can optionally
prepare a water-included receptor PDBQT and re-run the enrichment benchmark
with waters retained ("docking with waters").

Usage:
    python scripts/conserved_water_analysis.py
    python scripts/conserved_water_analysis.py --cutoff 4.0 --cluster-tol 1.5

Outputs:
    output/conserved_waters.json            — per-structure + conserved clusters
    output/conserved_waters.pdb             — conserved waters in ref frame (docking)
    output/figures/publication/conserved_waters.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from Bio.PDB import PDBParser

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("water")

REPO = Path(__file__).resolve().parent.parent
PDB_DIR = REPO / "output" / "pdb"
OUT = REPO / "output"
FIGS_OUT = OUT / "figures" / "publication"

# Catalytic triad atoms (resname, resseq, atom name) in the PBP2a numbering.
CATALYTIC = {
    "SER403_OG": ("SER", 403, "OG"),
    "LYS406_NZ": ("LYS", 406, "NZ"),
    "TYR446_OH": ("TYR", 446, "OH"),
}
REFERENCE = "1VQQ"
CHAIN = "A"
RESIDUE_WINDOW = (350, 470)  # C-alpha window for the Kabsch superposition fit


def _parse(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_path.stem, str(pdb_path))


def _catalytic_coords(structure) -> list[tuple[str, np.ndarray]]:
    coords = []
    model = structure[0]
    for label, (resname, resseq, atom_name) in CATALYTIC.items():
        try:
            residue = model[CHAIN][resseq]
        except KeyError:
            continue
        if residue.resname.strip() != resname:
            continue
        if atom_name not in residue:
            continue
        coords.append((label, residue[atom_name].get_coord().astype(float)))
    return coords


def _waters(structure) -> list[np.ndarray]:
    model = structure[0]
    waters = []
    for residue in model.get_residues():
        if residue.resname.strip() in ("HOH", "WAT", "SOL"):
            for atom in residue:
                if atom.element == "O":
                    waters.append(atom.get_coord().astype(float))
    return waters


def _c_alpha_atoms(structure) -> dict[tuple[str, int], np.ndarray]:
    """Active-site C-alpha coordinates keyed by (resname, resseq)."""
    model = structure[0]
    out = {}
    for residue in model.get_residues():
        rid = residue.id[1]
        if not (RESIDUE_WINDOW[0] <= rid <= RESIDUE_WINDOW[1]):
            continue
        if "CA" not in residue:
            continue
        out[(residue.resname.strip(), rid)] = residue["CA"].get_coord().astype(float)
    return out


def _kabsch(P, Q):
    """Best-fit rotation + translation mapping mobile points P onto Q (both Nx3)."""
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    Pc = P - pc
    Qc = Q - qc
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) if min(U.shape) > 1 else 1.0
    D = np.diag([1.0, 1.0, d]) if S.size > 2 else np.eye(3)
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    return R, t


def _superpose(mobile, reference_ca):
    """Superpose *mobile* structure onto reference C-alpha set; returns R, t."""
    mob_ca = _c_alpha_atoms(mobile)
    common = [k for k in mob_ca if k in reference_ca]
    if len(common) < 3:
        return None, None
    P = np.asarray([mob_ca[k] for k in common])
    Q = np.asarray([reference_ca[k] for k in common])
    return _kabsch(P, Q)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Conserved active-site water analysis for PBP2a structures"
    )
    parser.add_argument("--pdb-dir", type=str, default=str(PDB_DIR),
                        help="Directory containing the source crystal PDBs")
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="Active-site water cutoff from any catalytic atom (A)")
    parser.add_argument("--cluster-tol", type=float, default=1.5,
                        help="Distance tolerance for clustering conserved waters (A)")
    parser.add_argument("--min-structures", type=int, default=2,
                        help="Min supporting structures to call a water 'conserved'")
    parser.add_argument("--reference", type=str, default=REFERENCE,
                        help="Reference structure for superposition (default 1VQQ)")
    args = parser.parse_args(argv)

    pdb_dir = Path(args.pdb_dir)
    names = ["1VQQ", "3ZG0", "4DKI"]
    structures = {}
    for name in names:
        path = pdb_dir / f"{name}.pdb"
        if not path.is_file():
            log.warning(f"  structure not found: {path}")
            continue
        structures[name] = _parse(path)
    if not structures:
        log.error(f"No source PDBs found in {pdb_dir}")
        sys.exit(1)

    ref_name = args.reference if args.reference in structures else next(iter(structures))
    log.info(f"  Structures: {sorted(structures)}; reference: {ref_name}")

    ref_structure = structures[ref_name]
    ref_cat = _catalytic_coords(ref_structure)
    ref_ca = _c_alpha_atoms(ref_structure)
    if not ref_cat:
        log.error(f"Catalytic triad not found in reference {ref_name}")
        sys.exit(1)
    cat_atoms = np.asarray([c[1] for c in ref_cat])

    per_structure = {}
    conserved_water_coords = []
    for name, structure in structures.items():
        cat = _catalytic_coords(structure)
        waters = _waters(structure)
        R, t = _superpose(structure, ref_ca)
        transform = (R, t)
        if R is None:
            log.warning(f"  {name}: superposition failed (few shared C-alphas); "
                        f"reporting active-site waters in native frame only")
        else:
            log.info(f"  {name}: Kabsch superposed on {len(ref_ca)} ref C-alpha refs")
        active = []
        if cat:
            cat_xyz = np.asarray([c[1] for c in cat])
            for w in waters:
                if np.min(np.linalg.norm(cat_xyz - w, axis=1)) <= args.cutoff:
                    coord = (R @ w + t) if R is not None else w
                    active.append((coord, w))
            if active and R is not None:
                conserved_water_coords.append((name, [a[0] for a in active]))
        per_structure[name] = {
            "n_waters_total": len(waters),
            "n_catalytic_atoms_found": len(cat),
            "n_active_site_waters": len(active),
            "superposed": R is not None,
            "active_site_waters": [
                {"superposed_coord_A": [round(float(c), 3) for c in coord],
                 "native_coord_A": [round(float(w[0]), 3), round(float(w[1]), 3),
                                    round(float(w[2]), 3)]}
                for coord, w in active
            ],
        }
        log.info(f"  {name}: {len(waters)} total waters, "
                 f"{len(active)} within {args.cutoff} A of catalytic atoms")

    # Cluster superposed waters across structures to find conserved positions.
    clusters = []
    for name, coords in conserved_water_coords:
        for c in coords:
            placed = False
            for cl in clusters:
                if np.linalg.norm(cl["centroid"] - c) <= args.cluster_tol:
                    cl["members"].append((name, c))
                    cl["centroid"] = np.mean(
                        [m[1] for m in cl["members"]], axis=0
                    )
                    placed = True
                    break
            if not placed:
                clusters.append({"centroid": c, "members": [(name, c)]})

    conserved = [
        cl for cl in clusters
        if len({m[0] for m in cl["members"]}) >= args.min_structures
    ]
    conserved_xyz = [cl["centroid"] for cl in conserved]

    verdict = {
        "reference_structure": ref_name,
        "catalytic_cutoff_A": args.cutoff,
        "cluster_tolerance_A": args.cluster_tol,
        "min_structures": args.min_structures,
        "per_structure": per_structure,
        "n_conserved_water_positions": len(conserved),
        "conserved_waters": [
            {"coord_A": [round(float(c), 3) for c in cl["centroid"]],
             "supporting_structures": sorted({m[0] for m in cl["members"]}),
             "n_instances": len(cl["members"])}
            for cl in conserved
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "conserved_waters.json", "w") as fh:
        json.dump(verdict, fh, indent=2)

    # Export conserved waters as a PDB (reference frame) for optional
    # water-included receptor preparation / docking-with-waters runs.
    if conserved_xyz:
        with open(OUT / "conserved_waters.pdb", "w") as fh:
            for i, c in enumerate(conserved_xyz, start=1):
                fh.write(
                    f"HETATM{i:5d}  O   HOH A{2000 + i:4d}    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00 20.00           O\n"
                )
            fh.write("END\n")
        log.info(f"  Exported {len(conserved_xyz)} conserved waters: "
                 f"{OUT / 'conserved_waters.pdb'}")

    if len(conserved) > 0:
        msg = (
            f"{len(conserved)} conserved active-site water position(s) found across "
            f"{len(structures)} PBP2a structures. The current docking protocol strips "
            f"all waters before PDBQT preparation, so water-mediated contacts and "
            f"displacement penalties are not modelled -- a candidate contributor to "
            f"the poor rigid-docking enrichment for this target. Use "
            f"output/conserved_waters.pdb to prepare a water-included receptor and "
            f"re-run the benchmark."
        )
        classification = "WATERS_PRESENT"
    else:
        msg = (
            f"No conserved active-site water positions within {args.cutoff} A of the "
            f"catalytic triad were found across the PBP2a structures (or superposition "
            f"was insufficient). The water-stripped docking protocol is therefore not "
            f"missing conserved catalytic waters for this target; the poor enrichment "
            f"cannot be attributed to omitted conserved waters."
        )
        classification = "NO_CONSERVED_WATERS"
    verdict["classification"] = classification
    verdict["summary"] = msg
    with open(OUT / "conserved_waters.json", "w") as fh:
        json.dump(verdict, fh, indent=2)

    _plot(per_structure, conserved_xyz, args)
    log.info("")
    log.info(f"  Classification: {classification}")
    log.info(f"  {msg}")
    sys.exit(0)


def _plot(per_structure, conserved_xyz, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    names = list(per_structure)
    counts = [per_structure[n]["n_active_site_waters"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, counts, color="#2c7fb8")
    axes[0].set_ylabel(f"Waters within {args.cutoff} A of catalytic triad")
    axes[0].set_title("Active-site ordered waters")
    axes[0].axhline(0, color="grey", lw=0.5)

    ax = axes[1]
    for n in names:
        ws = per_structure[n]["active_site_waters"]
        pts = np.asarray([w["superposed_coord_A"] for w in ws]) if ws else np.empty((0, 3))
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 1], alpha=0.4, s=20, label=n)
    if conserved_xyz:
        cx = np.asarray(conserved_xyz)
        ax.scatter(cx[:, 0], cx[:, 1], marker="*", s=160, c="red",
                   label="conserved")
    ax.set_xlabel("X (A, ref frame)")
    ax.set_ylabel("Y (A, ref frame)")
    ax.set_title("Superposed active-site waters")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(str(FIGS_OUT / "conserved_waters.png"), dpi=300)
    plt.close(fig)
    log.info(f"  Plot: {FIGS_OUT / 'conserved_waters.png'}")


if __name__ == "__main__":
    main()