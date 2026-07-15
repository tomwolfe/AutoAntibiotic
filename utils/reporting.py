#!/usr/bin/env python3
"""
Reporting & artifact generation for the AutoAntibiotic discovery pipeline.

This module owns Phase 5: the CSV/JSON report, 2D structure images, the 2D
interaction diagrams, and the PyMOL visualization script. It depends only on
RDKit, NumPy, pandas, and ``config.constants`` so it can be imported without
pulling in the full orchestrator.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from Bio.PDB import PDBParser

from config.constants import OUTPUT_DIR, CSV_REPORT, protocol_trust

# A module-level logger sharing the pipeline's "AutoAntibiotic" logger name so
# that handlers configured in discovery_pipeline capture these messages too.
log = logging.getLogger("AutoAntibiotic")


def _key_residue_coords(receptor_pdb: str) -> Dict[str, List[np.ndarray]]:
    """
    Return the 3D coordinates of the key catalytic residue atoms in *receptor_pdb*.

    Only the polar H-bond donor/acceptor atoms are collected:
        Ser403 → OG, Lys406 → NZ, Tyr446 → OH.

    Returns a dict ``{resname: [np.ndarray, ...]}`` (empty lists for absent
    residues). Used to highlight ligand atoms that engage these residues in the
    2D interaction diagram.
    """
    targets_map = {
        "Ser403": [("SER", 403, "OG")],
        "Lys406": [("LYS", 406, "NZ")],
        "Tyr446": [("TYR", 446, "OH")],
    }
    atom_coords: Dict[str, List[np.ndarray]] = {k: [] for k in targets_map}
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("receptor", receptor_pdb)
        for model in struct:
            for chain in model:
                for residue in chain:
                    for resname, entries in targets_map.items():
                        for entry in entries:
                            aname = entry[-1]
                            resname_expected = entry[0] if len(entry) > 2 else ""
                            resno_expected = entry[1] if len(entry) > 2 else -1
                            if (
                                resname_expected
                                and residue.get_resname().strip().upper()
                                != resname_expected.upper()
                            ):
                                continue
                            if resno_expected >= 0 and residue.get_id()[1] != resno_expected:
                                continue
                            if aname in residue:
                                atom_coords[resname].append(
                                    residue[aname].get_vector().get_array()
                                )
    except Exception as exc:
        log.warning(f"  Could not parse receptor for key residues: {exc}")
    return atom_coords


def _pose_heavy_atom_coords(pdbqt_path: str) -> List[np.ndarray]:
    """
    Parse heavy-atom 3D coordinates (in file order) from a docked ligand PDBQT.

    Hydrogen lines are skipped so the returned list indexes the same heavy
    atoms as a SMILES-derived :class:`Chem.Mol` (PDBQT preparation appends
    hydrogens after the heavy atoms, preserving heavy-atom order).
    """
    coords: List[np.ndarray] = []
    try:
        with open(pdbqt_path) as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
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


def generate_csv_report(
    top10: List[CompoundRecord],
    validation_ok: bool = False,
    holo_pdb_path: Optional[str] = None,
    mode: str = "science",
    redock_rmsd: Optional[float] = None,
    csv_report: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Phase 5.1 — Write top_candidates.csv with all required columns.

    Columns:
        Compound_ID, SMILES, PBP2a_Allosteric_Energy, PBP2a_Active_Energy,
        Human_Trypsin_Energy, Human_CES1_Energy, Selectivity_Index,
        Selectivity_Confidence, Max_Similarity, Passes_Lipinski,
        QED_Score, Binding_Mode_Notes, Protocol_RMSD, protocol_trust,
        H_Bond_Ser403, H_Bond_Lys406, H_Bond_Tyr446.

    Returns path to CSV.
    """
    log.info("─── Phase 5: Reporting ───")
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if csv_report is None:
        csv_report = CSV_REPORT
    output_dir = Path(output_dir)
    csv_report = Path(csv_report)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_mock = (mode == "ci")

    rows = []
    for rec in top10:
        # Per-residue H-bond flags derived from the interaction fingerprint
        # captured during Phase 4 (min distance to the conserved residue; a
        # contact is flagged when the ligand approaches within 3.5 Å).
        inter = getattr(rec, "interactions", None)
        if inter:
            h_ser = bool(inter.get("min_dist_Ser403", float("inf")) < 3.5)
            h_lys = bool(inter.get("min_dist_Lys406", float("inf")) < 3.5)
            h_tyr = bool(inter.get("min_dist_Tyr446", float("inf")) < 3.5)
        else:
            h_ser = h_lys = h_tyr = False

        # ── Trust Badge columns ──
        # Protocol_RMSD: the raw redocking RMSD value (in Å) for every row, shown
        # as a plain float (e.g. "1.234") wherever a real measurement exists. In
        # CI/mock mode the protocol is never redocked, so it reads "SKIPPED".
        protocol_rmsd_str = "SKIPPED" if is_mock else (
            f"{redock_rmsd:.3f}" if redock_rmsd is not None else "N/A"
        )

        # protocol_trust: a single quick-glance trust badge so chemists
        # immediately see protocol quality. The canonical mapping logic lives in
        # ``config.constants.protocol_trust`` so it remains the single source of
        # truth for these exact output strings.
        protocol_trust_val = protocol_trust(mode, redock_rmsd)

        rows.append({
            "Compound_ID": rec.compound_id,
            "SMILES": rec.smiles,
            "PBP2a_Allosteric_Energy": (
                f"{rec.pb2pa_allosteric_energy:.2f}" if rec.pb2pa_allosteric_energy is not None
                else "N/A"
            ),
            "PBP2a_Active_Energy": (
                f"{rec.pb2pa_active_energy:.2f}" if rec.pb2pa_active_energy is not None
                else "N/A"
            ),
            "Human_Trypsin_Energy": (
                f"{rec.human_trypsin_energy:.2f}" if rec.human_trypsin_energy is not None
                else "N/A"
            ),
            "Human_CES1_Energy": (
                f"{rec.human_ces1_energy:.2f}" if rec.human_ces1_energy is not None
                else "N/A"
            ),
            "Selectivity_Index": (
                f"{rec.selectivity_index:.2f}" if rec.selectivity_index is not None
                else "N/A"
            ) + ("" if rec.selectivity_confidence == "High" else " (low-conf)"),
            "Selectivity_Confidence": (
                "Unassessed" if rec.selectivity_confidence == "None"
                else rec.selectivity_confidence
            ) + (" (mock)" if is_mock else ""),
            "Max_Similarity": f"{rec.max_similarity:.3f}",
            "Passes_Lipinski": str(rec.passes_lipinski),
            "QED_Score": f"{rec.qed_score:.3f}",
            "Binding_Mode_Notes": rec.resistance_notes.replace("; ", " | "),
            "Protocol_RMSD": protocol_rmsd_str,
            "protocol_trust": protocol_trust_val,
            "H_Bond_Ser403": str(h_ser),
            "H_Bond_Lys406": str(h_lys),
            "H_Bond_Tyr446": str(h_tyr),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_report, index=False)
    log.info(f"  CSV report saved: {csv_report}")

    json_path = Path(str(csv_report)).with_suffix(".json")
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)
    log.info(f"  JSON candidates saved: {json_path}")

    return str(csv_report)


def generate_images(
    top3: List[CompoundRecord],
    output_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Phase 5.2 — Save 2D structure PNGs for the top 3 candidates.

    Returns list of file paths.
    """
    paths = []
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    for i, rec in enumerate(top3):
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol

        img_path = output_dir / f"top{i + 1}_{rec.compound_id}.png"
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(rec.mol)
            drawer.FinishDrawing()
            drawer.WriteDrawingText(str(img_path))
            paths.append(str(img_path))
            log.info(f"  Image saved: {img_path}")
        except Exception as exc:
            log.warning(f"  Failed to render {rec.compound_id}: {exc}")

    return paths


def generate_interaction_diagram(
    record: CompoundRecord,
    receptor_pdb: str,
    output_path: str,
) -> Optional[str]:
    """
    Phase 5.2b — Render a 2D interaction diagram for a single compound.

    Draws the ligand (from ``record.mol`` / SMILES) and overlays the key
    binding interactions: ligand atoms that approach within H-bond distance of
    Ser403, Lys406, or Tyr446 (parsed from the docked pose in
    ``record.active_docked_pdbqt``) are highlighted in red. A legend line
    summarises which conserved residues are engaged.

    The diagram is saved to *output_path* (typically
    ``output/interaction_<compound_id>.png``) and the path is returned.

    If the heavyweight pose/atom mapping is unavailable, the function still
    draws the ligand and annotates the detected key interactions from
    ``record.interactions`` — so a chemist always gets a visual artefact.
    """
    mol = record.mol
    if mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            log.warning(
                f"  Cannot render ligand for {record.compound_id} "
                "(invalid SMILES)."
            )
            return None

    try:
        from rdkit.Chem.Draw import rdMolDraw2D as _draw
    except Exception as exc:
        log.warning(f"  RDKit drawing unavailable: {exc}")
        return None

    # ── Determine which ligand atoms engage the key residues ──
    highlight_atoms: List[int] = []
    inter = getattr(record, "interactions", None)
    pose = getattr(record, "active_docked_pdbqt", None)

    if pose and os.path.exists(pose) and receptor_pdb and os.path.exists(receptor_pdb):
        try:
            key_coords = _key_residue_coords(receptor_pdb)
            pose_coords = _pose_heavy_atom_coords(pose)
            n_lig = mol.GetNumAtoms()
            # Heavy-atom order in the docked pose matches the SMILES-derived mol
            # (H are appended by the PDBQT prep), so index i maps directly.
            for i, p in enumerate(pose_coords):
                if i >= n_lig:
                    break
                for resname, coords in key_coords.items():
                    if not coords:
                        continue
                    cutoff = 3.8 if resname == "Lys406" else 3.5
                    if any(np.linalg.norm(p - c) < cutoff for c in coords):
                        highlight_atoms.append(i)
                        break
        except Exception as exc:
            log.warning(f"  Atom-level interaction highlight failed: {exc}")
            highlight_atoms = []

    # De-duplicate while preserving order.
    highlight_atoms = list(dict.fromkeys(highlight_atoms))

    # ── Build the legend from the interaction fingerprint ──
    parts = []
    if inter:
        if inter.get("Ser403_contact"):
            parts.append("Ser403 H-bond")
        if inter.get("Lys406_Hbond"):
            parts.append("Lys406 H-bond")
        if inter.get("Tyr446_Hbond"):
            parts.append("Tyr446 H-bond")
    legend = "; ".join(parts) if parts else "No key H-bonds detected"

    try:
        drawer = _draw.MolDraw2DCairo(500, 500)
        drawer.drawOptions().highlightColor = _draw.Color(1.0, 0.0, 0.0)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            legend=legend,
        )
        drawer.FinishDrawing()
        drawer.WriteDrawingText(output_path)
        log.info(f"  Interaction diagram saved: {output_path}")
        return output_path
    except Exception as exc:
        log.warning(f"  Could not render interaction diagram: {exc}")
        return None


def generate_pymol_script(
    top_candidates: List[CompoundRecord],
    targets: dict,
    output_dir: str,
) -> str:
    """
    Phase 5.3 — Write a PyMOL session script (``visualization.pml``) that loads
    the PBP2a receptor, the top 3 candidate active-site poses, highlights the
    conserved catalytic residues (Ser403, Lys406, Tyr446) as sticks, and colours
    each ligand by element for quick medicinal-chemist inspection.

    Returns the path to the generated ``.pml`` file.
    """
    pml_path = os.path.join(output_dir, "visualization.pml")
    receptor_pdb = targets.get("PBP2a", {}).get("cleaned_pdb")

    lines = [
        "# Auto-generated PyMOL visualization script (AutoAntibiotic)",
        "# Load with: pymol -l visualization.pml",
        "",
    ]

    if receptor_pdb and os.path.exists(receptor_pdb):
        lines.append(f"load {receptor_pdb!r}, PBP2a")
        # Highlight the conserved catalytic residues as sticks.
        lines.append(
            "select conserved_residues, "
            "(resn SER and resi 403) or "
            "(resn LYS and resi 406) or "
            "(resn TYR and resi 446)"
        )
        lines.append("show sticks, conserved_residues")
        lines.append("color magenta, conserved_residues")
        lines.append("")
    else:
        log.warning(
            "  PBP2a cleaned PDB not available; PyMOL script will skip receptor load."
        )

    loaded = 0
    for i, rec in enumerate(top_candidates[:3]):
        pose = getattr(rec, "active_docked_pdbqt", None)
        if not pose or not os.path.exists(pose):
            log.warning(
                f"  No active-site pose for {rec.compound_id}; skipping in PyMOL script."
            )
            continue
        name = f"Ligand_{i + 1}_{rec.compound_id}"
        lines.append(f"load {pose!r}, {name}")
        # Show the ligand as sticks coloured by element for quick inspection.
        lines.append(f"color byelement, {name}")
        lines.append(f"show sticks, {name}")

        # ── Dashed hydrogen-bond lines from the pose-derived interaction fingerprint ──
        # Draw a dashed measurement line for each conserved residue whose
        # ligand→residue distance is within the H-bond cutoff (Ser403/Tyr446 < 3.5 Å,
        # Lys406 < 3.8 Å). These come straight from the record.interactions dict
        # computed during Phase 4, so the script reflects the real measured pose.
        inter = getattr(rec, "interactions", None)
        if inter:
            ser_d = inter.get("min_dist_Ser403", float("inf"))
            lys_d = inter.get("min_dist_Lys406", float("inf"))
            tyr_d = inter.get("min_dist_Tyr446", float("inf"))
            if np.isfinite(ser_d) and ser_d < 3.5:
                lines.append(
                    f"distance hbond_ser_{i + 1}, {name}, resi 403 & name OG, cutoff=3.5"
                )
                lines.append("dash wid 2.0")
            if np.isfinite(lys_d) and lys_d < 3.8:
                lines.append(
                    f"distance hbond_lys_{i + 1}, {name}, resi 406 & name NZ, cutoff=3.8"
                )
                lines.append("dash wid 2.0")
            if np.isfinite(tyr_d) and tyr_d < 3.5:
                lines.append(
                    f"distance hbond_tyr_{i + 1}, {name}, resi 446 & name OH, cutoff=3.5"
                )
                lines.append("dash wid 2.0")
        lines.append("")
        loaded += 1

    lines.append("")
    lines.append("# Orient the view")
    lines.append("zoom")
    lines.append("orient")

    with open(pml_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    log.info(
        f"  PyMOL script saved: {pml_path} "
        f"({loaded} ligand pose(s) loaded)."
    )
    return pml_path


def _print_single_summary(rec: CompoundRecord) -> None:
    """Print a concise single-compound screening summary table to stdout."""
    inter = getattr(rec, "interactions", None)
    key_interactions = []
    if inter:
        if inter.get("Ser403_contact"):
            key_interactions.append("Ser403")
        if inter.get("Lys406_Hbond"):
            key_interactions.append("Lys406")
        if inter.get("Tyr446_Hbond"):
            key_interactions.append("Tyr446")
    key_str = ", ".join(key_interactions) if key_interactions else "None detected"

    fmt = lambda v: f"{v:.2f} kcal/mol" if v is not None else "N/A"

    print("\n" + "=" * 64)
    print("  SINGLE-COMPOUND SCREEN SUMMARY")
    print("=" * 64)
    print(f"  {'Compound ID':<20}: {rec.compound_id}")
    print(f"  {'SMILES':<20}: {rec.smiles}")
    print(f"  {'Allosteric Energy':<20}: {fmt(rec.pb2pa_allosteric_energy)}")
    print(f"  {'Active Energy':<20}: {fmt(rec.pb2pa_active_energy)}")
    print(f"  {'Key Interactions':<20}: {key_str}")
    print("=" * 64 + "\n")
