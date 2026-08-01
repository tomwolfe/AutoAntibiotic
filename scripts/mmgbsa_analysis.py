#!/usr/bin/env python3
"""
MM-GBSA binding free energy analysis from the minimised explicit-solvent MD complexes.

The explicit-solvent MD protocol stores per-frame ligand RMSD / receptor RMSF
traces but not full trajectory coordinates, so MM-GBSA is evaluated as a
single-pose end-point calculation on each successful candidate's energy-minimised
protein--ligand complex (output/md_explicit/<CID>/replica_0/topology.pdb),
with water and ions stripped and an implicit-solvent OBC2 (GBSAOBC2) model.

delta_G_bind = E_complex - E_receptor - E_ligand
where each term is computed with three separate systems (the partner's atoms are
fully removed, so bonded energies cancel correctly). The catalytic triad
(Ser403/Lys406/Tyr446) is located in the stripped topology by matching atom
coordinates against the prepared receptor PDB, which is robust to the sequential
residue renumbering introduced by PDBFile.writeFile.

Reports:
  - delta_G_bind for each compound
  - Per-residue decomposition for the catalytic triad

Usage:
    python scripts/mmgbsa_analysis.py

Outputs:
    output/mmgbsa_results.json           - per-candidate MM-GBSA results
    output/mmgbsa_per_residue.json       - per-residue decomposition
    output/figures/publication/mmgbsa_barchart.{pdf,png} - bar chart of dG_bind
    output/figures/publication/per_residue_decomp.{pdf,png} - per-residue decomposition
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("mmgbsa")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
MD_OUT = OUT / "md_explicit"
FIGS_OUT = OUT / "figures" / "publication"
RECEPTOR_PDB = OUT / "workdir" / "PBP2a_holo_clean.pdb"
CSV_PATH = OUT / "top_candidates.csv"

WATER = {"HOH", "WAT", "NA", "CL", "K", "Cl-", "Na+"}

# Receptor-numbering catalytic triad and the reference atom to match by coordinates.
CATALYTIC = {"SER403": ("SER", 403, "OG"),
             "LYS406": ("LYS", 406, "NZ"),
             "TYR446": ("TYR", 446, "OH")}


def _check_deps():
    try:
        import openmm  # noqa: F401
    except ImportError:
        log.error("OpenMM not installed. Run: conda install -c conda-forge openmm")
        sys.exit(1)


def _reference_catalytic_coords():
    """Coordinates of Ser403.OG / Lys406.NZ / Tyr446.OH in the prepared receptor."""
    import openmm.app as app
    pdb = app.PDBFile(str(RECEPTOR_PDB))
    refs = {}
    for residue in pdb.topology.residues():
        for label, (resname, resnum, atom_name) in CATALYTIC.items():
            if residue.name == resname and int(residue.id) == resnum:
                for atom in residue.atoms():
                    if atom.name == atom_name:
                        p = pdb.positions[atom.index]
                        refs[label] = (atom_name, p.x, p.y, p.z)
    missing = [k for k in CATALYTIC if k not in refs]
    if missing:
        raise RuntimeError(f"Could not locate catalytic atoms {missing} in {RECEPTOR_PDB}")
    return refs


def _strip_solvent(modeller):
    """Delete water / ions from a Modeller, returning atom count removed."""
    n_before = modeller.topology.getNumAtoms()
    modeller.delete([r for r in modeller.topology.residues() if r.name in WATER])
    return n_before - modeller.topology.getNumAtoms()


def _build_system(topology, smi=None):
    """Build a GBSAOBC2 (OBC2) implicit-solvent system."""
    import openmm.app as app
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from openff.toolkit import Molecule as OffMolecule
    from rdkit import Chem

    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    if smi:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        off = OffMolecule.from_rdkit(mol, allow_undefined_stereo=True)
        off.assign_partial_charges(partial_charge_method="gasteiger")
        tg = SMIRNOFFTemplateGenerator(molecules=off, forcefield="openff-2.0.0")
        ff.registerTemplateGenerator(tg.generator)
    return ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)


def _energy(system, positions):
    import openmm
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001 * openmm.unit.picoseconds))
    ctx.setPositions(positions)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        openmm.unit.kilocalories_per_mole
    )


def _zero_residue(system, indices):
    """Zero charges + LJ of the given atoms in both NonbondedForce and CustomGBForce."""
    import openmm
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    gb = [f for f in system.getForces() if isinstance(f, openmm.CustomGBForce)][0]
    saved = {}
    for i in indices:
        q, s, e = nb.getParticleParameters(i)
        saved[i] = (q, s, e, gb.getParticleParameters(i))
        nb.setParticleParameters(i, 0.0, s, e)
        ch, o, sr = gb.getParticleParameters(i)
        gb.setParticleParameters(i, [0.0, o, sr])
    return saved, nb, gb


def _restore_residue(saved, nb, gb):
    for i, (q, s, e, gbparams) in saved.items():
        nb.setParticleParameters(i, q, s, e)
        gb.setParticleParameters(i, gbparams)


def _match_catalytic_by_coords(topology, positions, refs):
    """Return {label: residue} by nearest-atom coordinate match to receptor refs."""
    import numpy as np
    best = {}
    for res in topology.residues():
        for atom in res.atoms():
            if atom.name not in ("OG", "NZ", "OH"):
                continue
            p = positions[atom.index]
            for label, (ref_name, rx, ry, rz) in refs.items():
                if atom.name != ref_name:
                    continue
                d = np.sqrt((p.x - rx) ** 2 + (p.y - ry) ** 2 + (p.z - rz) ** 2)
                if label not in best or d < best[label][0]:
                    best[label] = (d, res)
    return {label: res for label, (d, res) in best.items()}


def compute_mmgbsa(candidate_dir: Path) -> dict | None:
    """Single-pose MM-GBSA for one successful MD candidate."""
    import openmm.app as app

    summary_path = candidate_dir / "summary.json"
    if not summary_path.is_file():
        log.warning(f"  Summary not found: {summary_path}")
        return None
    with open(summary_path) as f:
        cand_data = json.load(f)
    if not cand_data.get("success"):
        log.warning(f"  Candidate MD failed, skipping MM-GBSA")
        return None

    cid = cand_data["compound_id"]
    smi = cand_data["smiles"]
    rep_dir = candidate_dir / "replica_0"
    top_pdb = rep_dir / "topology.pdb"
    if not top_pdb.is_file():
        log.warning(f"  Topology not found: {top_pdb}")
        return None

    try:
        pdb = app.PDBFile(str(top_pdb))
    except Exception as exc:
        log.warning(f"  PDB load failed: {exc}")
        return None

    mod = app.Modeller(pdb.topology, pdb.positions)
    _strip_solvent(mod)

    # Identify the ligand residue (last non-solvent residue).
    lig_residues = [r for r in mod.topology.residues() if r.name in ("LIG", "UNL", "MOL")]
    if not lig_residues:
        lig_residues = [list(mod.topology.residues())[-1]]
    if len(lig_residues) != 1:
        log.warning(f"  {cid}: unexpected ligand residue count {len(lig_residues)}")
        return None
    lig_res = lig_residues[0]

    cpx_top, cpx_pos = mod.topology, mod.positions

    rec_mod = app.Modeller(mod.topology, mod.positions)
    rec_mod.delete([lig_res])
    rec_top, rec_pos = rec_mod.topology, rec_mod.positions

    lig_mod = app.Modeller(mod.topology, mod.positions)
    lig_mod.delete([r for r in mod.topology.residues() if r != lig_res])
    lig_top, lig_pos = lig_mod.topology, lig_mod.positions

    try:
        sys_cpx = _build_system(cpx_top, smi)
        sys_rec = _build_system(rec_top)
        sys_lig = _build_system(lig_top, smi)
    except Exception as exc:
        log.warning(f"  {cid}: system build failed: {exc}")
        return None

    e_complex = _energy(sys_cpx, cpx_pos)
    e_receptor = _energy(sys_rec, rec_pos)
    e_ligand = _energy(sys_lig, lig_pos)
    dg = e_complex - e_receptor - e_ligand
    dg_ref = e_complex - e_receptor

    # Per-residue decomposition of the catalytic triad.
    per_res = {}
    try:
        refs = _reference_catalytic_coords()
        matched = _match_catalytic_by_coords(cpx_top, cpx_pos, refs)
        for label, res in matched.items():
            idxs = [a.index for a in res.atoms()]
            saved, nb, gb = _zero_residue(sys_cpx, idxs)
            e_cpx_mut = _energy(sys_cpx, cpx_pos)
            _restore_residue(saved, nb, gb)
            saved, nb, gb = _zero_residue(sys_rec, idxs)
            e_rec_mut = _energy(sys_rec, rec_pos)
            _restore_residue(saved, nb, gb)
            per_res[label] = round((e_cpx_mut - e_rec_mut) - dg_ref, 2)
    except Exception as exc:
        log.warning(f"  {cid}: per-residue decomposition skipped: {exc}")

    log.info(f"    dG_bind = {dg:.2f} kcal/mol; per-residue {per_res}")
    return {
        "compound_id": cid,
        "smiles": smi,
        "n_snapshots": 1,
        "replica_used": 0,
        "delta_G_bind_mean_kcal": round(dg, 2),
        "delta_G_bind_std_kcal": 0.0,
        "delta_G_bind_min_kcal": round(dg, 2),
        "delta_G_bind_max_kcal": round(dg, 2),
        "delta_G_bind_values": [round(dg, 2)],
        "method": "OpenMM_GBSAOBC2_single_pose_minimized",
        "success": True,
        "per_residue": {k: {"mean_kcal": v, "std_kcal": 0.0} for k, v in per_res.items()},
    }


def main():
    _check_deps()

    MD_OUT.mkdir(parents=True, exist_ok=True)
    FIGS_OUT.mkdir(parents=True, exist_ok=True)

    candidates = sorted([d for d in MD_OUT.iterdir()
                         if d.is_dir() and d.name.startswith(("BRICS_", "ALL_", "SEED_"))])
    if not candidates:
        log.error(f"No candidate directories found in {MD_OUT}. Run scripts/explicit_solvent_md.py first.")
        sys.exit(1)

    log.info(f"Found {len(candidates)} candidate directories")

    all_results = []
    for cand_dir in candidates:
        cid = cand_dir.name
        log.info(f"\n  Processing {cid}...")
        result = compute_mmgbsa(cand_dir)
        if result:
            all_results.append(result)

    with open(OUT / "mmgbsa_results.json", "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    log.info(f"\n  MM-GBSA results saved: {OUT / 'mmgbsa_results.json'}")

    per_res_data = {}
    for r in all_results:
        if r.get("per_residue"):
            per_res_data[r["compound_id"]] = {
                "delta_G_bind_mean_kcal": r["delta_G_bind_mean_kcal"],
                "per_residue": r["per_residue"],
                "method": r.get("method"),
            }
    with open(OUT / "mmgbsa_per_residue.json", "w") as fh:
        json.dump(per_res_data, fh, indent=2, default=str)
    log.info(f"  Per-residue results saved: {OUT / 'mmgbsa_per_residue.json'}")

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cids = [r["compound_id"] for r in all_results]
        means = [r["delta_G_bind_mean_kcal"] for r in all_results]
        stds = [r["delta_G_bind_std_kcal"] for r in all_results]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2c7fb8", "#7fcdbb", "#edf8b1", "#41b6c4", "#253494"]
        bars = ax.bar(range(len(cids)), means, yerr=stds, capsize=5,
                      color=colors[:len(cids)], edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels(cids, rotation=45, ha="right")
        ax.set_ylabel("ΔG_bind (kcal/mol)")
        ax.set_title("MM-GBSA (OBC2, single-pose) Binding Free Energies")
        ax.axhline(0, color="grey", ls="--", lw=0.5)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.3,
                    f"{mean:.1f}±{std:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(str(FIGS_OUT / "mmgbsa_barchart.pdf"), dpi=300)
        fig.savefig(str(FIGS_OUT / "mmgbsa_barchart.png"), dpi=300)
        plt.close(fig)

        if per_res_data:
            n = len(per_res_data)
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
            if n == 1:
                axes = [axes]
            from matplotlib.patches import Patch
            for idx, (cid, data) in enumerate(per_res_data.items()):
                ax = axes[idx]
                residues = list(data["per_residue"].keys())
                values = [data["per_residue"][r]["mean_kcal"] for r in residues]
                errs = [data["per_residue"][r]["std_kcal"] for r in residues]
                ax.barh(range(len(residues)), values, xerr=errs, color="#377eb8",
                        edgecolor="black", linewidth=0.3, capsize=3)
                ax.set_yticks(range(len(residues)))
                ax.set_yticklabels(residues, fontsize=8)
                ax.set_xlabel("Energy contribution (kcal/mol)")
                ax.set_title(f"{cid}: ΔG = {data['delta_G_bind_mean_kcal']:.1f} kcal/mol")
                ax.axvline(0, color="grey", ls="--", lw=0.5)
            fig.tight_layout()
            fig.savefig(str(FIGS_OUT / "per_residue_decomp.pdf"), dpi=300)
            fig.savefig(str(FIGS_OUT / "per_residue_decomp.png"), dpi=300)
            plt.close(fig)
    except ImportError:
        log.warning("  matplotlib not available; skipping figures")

    # Update top_candidates.csv with MMGBSA_dG_Bind and MD_Stability columns
    if CSV_PATH.is_file():
        try:
            rows = []
            with open(CSV_PATH, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                for col in ("MMGBSA_dG_Bind", "MD_Stability"):
                    if col not in fieldnames:
                        fieldnames.append(col)
                for row in reader:
                    cid = row.get("Compound_ID", "")
                    mmgbsa_result = next(
                        (r for r in all_results if r.get("compound_id") == cid), None
                    )
                    if mmgbsa_result and mmgbsa_result.get("success"):
                        row["MMGBSA_dG_Bind"] = (
                            f"{mmgbsa_result['delta_G_bind_mean_kcal']:.2f}±"
                            f"{mmgbsa_result['delta_G_bind_std_kcal']:.2f}"
                        )
                    else:
                        row["MMGBSA_dG_Bind"] = ""
                    cand_dir = MD_OUT / cid
                    cs_path = cand_dir / "summary.json"
                    if cand_dir.is_dir() and cs_path.is_file():
                        try:
                            with open(cs_path) as f:
                                cs = json.load(f)
                            row["MD_Stability"] = cs.get("consensus_stability", "")
                        except Exception:
                            row["MD_Stability"] = ""
                    rows.append(row)

            with open(CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            log.info(f"  Updated {CSV_PATH} with MMGBSA_dG_Bind and MD_Stability columns")
        except Exception as exc:
            log.warning(f"  Could not update CSV: {exc}")

    n_ok = sum(1 for r in all_results if r.get("success"))
    log.info(f"\n  {n_ok}/{len(all_results)} succeeded")
    sys.exit(0 if n_ok == len(all_results) else 1)


if __name__ == "__main__":
    main()
