from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .config import CONFIG, CompoundRecord
from .io_utils import ensure_output_dir, log


def generate_csv_report(top10: List[CompoundRecord]) -> str:
    """Phase 5.1 — Write top_candidates.csv with all required columns."""
    log.info("─── Phase 5: Reporting ───")
    ensure_output_dir()

    scoring_method = "Vina" if top10[0].pb2pa_allosteric_energy is not None else "RDKit Shape (fallback)"
    rows: List[Dict[str, str]] = []
    for rec in top10:
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
            "Shape_Score": (
                f"{rec.shape_score:.2f}" if rec.shape_score is not None
                else "N/A"
            ),
            "Selectivity_Index": (
                f"{rec.selectivity_index:.2f}" if rec.selectivity_index is not None
                else "N/A"
            ),
            "Max_Similarity": f"{rec.max_similarity:.3f}",
            "Passes_Lipinski": str(rec.passes_lipinski),
            "QED_Score": f"{rec.qed_score:.3f}",
            "Scoring_Method": scoring_method,
            "Binding_Mode_Notes": rec.resistance_notes.replace("; ", " | "),
        })

    df = pd.DataFrame(rows)
    df.to_csv(CONFIG.output_dir / "top_candidates.csv", index=False)
    log.info(f'  CSV report saved: {CONFIG.output_dir / "top_candidates.csv"}')
    return str(CONFIG.output_dir / "top_candidates.csv")


def generate_images(top3: List[CompoundRecord]) -> List[str]:
    """Phase 5.2 — Save 2D structure PNGs for the top 3 candidates."""
    paths: List[str] = []
    for i, rec in enumerate(top3):
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol

        img_path = CONFIG.output_dir / f"top{i + 1}_{rec.compound_id}.png"
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


def generate_html_report(
    top10: List[CompoundRecord],
    top50: List[CompoundRecord],
    output_dir: Path,
) -> Tuple[str, str, str]:
    """Phase 5.3 — Generate an HTML report with embedded matplotlib figures.

    Creates scatter plot, QED histogram, and HTML page.
    Returns (html_path, scatter_path, hist_path).
    """
    log.info("─── Phase 5: HTML Report Generation ───")

    scatter_data = [
        (r.pb2pa_allosteric_energy, r.selectivity_index, r.compound_id)
        for r in top10
        if r.pb2pa_allosteric_energy is not None and r.selectivity_index is not None
    ]
    if scatter_data:
        fig, ax = plt.subplots(figsize=(9, 6))
        energies = [d[0] for d in scatter_data]
        sis = [d[1] for d in scatter_data]
        cids = [d[2] for d in scatter_data]
        ax.scatter(energies, sis, c="steelblue", s=60, edgecolors="black")
        for x, y, cid in zip(energies, sis, cids):
            ax.annotate(cid, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax.axhline(y=CONFIG.selectivity_index_threshold, color="red", linestyle="--", alpha=0.6,
                   label=f"SI threshold = {CONFIG.selectivity_index_threshold}")
        ax.set_xlabel("Allosteric Binding Energy (kcal/mol)", fontsize=12)
        ax.set_ylabel("Selectivity Index", fontsize=12)
        ax.set_title("Top Candidates: Binding Energy vs Selectivity", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        scatter_path = os.path.join(str(output_dir), "energy_vs_selectivity.png")
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Scatter plot saved: {scatter_path}")
    else:
        scatter_path = ""

    qeds = [r.qed_score for r in top50 if r.qed_score > 0]
    if qeds:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.hist(qeds, bins=20, edgecolor="black", color="mediumseagreen", alpha=0.8)
        ax.axvline(x=0.6, color="red", linestyle="--", alpha=0.6, label="QED cutoff = 0.6")
        ax.set_xlabel("QED Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("QED Distribution (Top 50 Candidates)", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        hist_path = os.path.join(str(output_dir), "qed_histogram.png")
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  QED histogram saved: {hist_path}")
    else:
        hist_path = ""

    table_rows = ""
    for i, rec in enumerate(top10):
        allosteric = f"{rec.pb2pa_allosteric_energy:.2f}" if rec.pb2pa_allosteric_energy is not None else "N/A"
        active = f"{rec.pb2pa_active_energy:.2f}" if rec.pb2pa_active_energy is not None else "N/A"
        si = f"{rec.selectivity_index:.2f}" if rec.selectivity_index is not None else "N/A"
        qed = f"{rec.qed_score:.3f}" if rec.qed_score else "N/A"
        table_rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td>{rec.compound_id}</td>"
            f"<td style='font-size:0.8em;max-width:300px;word-break:break-all;'>{rec.smiles}</td>"
            f"<td>{allosteric}</td>"
            f"<td>{active}</td>"
            f"<td>{si}</td>"
            f"<td>{qed}</td>"
            f"<td>{rec.resistance_notes}</td>"
            f"</tr>\n"
        )

    scatter_img = ""
    if scatter_path:
        scatter_img = (
            '<h2>Binding Energy vs Selectivity</h2>\n'
            f'<img src="energy_vs_selectivity.png" alt="Energy vs Selectivity" style="max-width:800px;">\n'
        )
    hist_img = ""
    if hist_path:
        hist_img = (
            '<h2>QED Score Distribution</h2>\n'
            f'<img src="qed_histogram.png" alt="QED Histogram" style="max-width:800px;">\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoAntibiotic Discovery Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; }}
h1 {{ color: #1a5276; }}
h2 {{ color: #2e86c1; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background-color: #2e86c1; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
img {{ border: 1px solid #ddd; border-radius: 4px; padding: 4px; }}
.footer {{ margin-top: 30px; color: #777; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>AutoAntibiotic Discovery Pipeline — Top Candidates Report</h1>
<p>Generated by AutoAntibiotic v3.2 | MRSA PBP2a Inhibitor Screening</p>
<hr>

{scatter_img}

{hist_img}

<h2>Top {len(top10)} Candidates</h2>
<table>
<tr>
  <th>Rank</th>
  <th>ID</th>
  <th>SMILES</th>
  <th>Allosteric (kcal/mol)</th>
  <th>Active (kcal/mol)</th>
  <th>Selectivity Index</th>
  <th>QED</th>
  <th>Resistance Notes</th>
</tr>
{table_rows}
</table>

<div class="footer">
<p>Pipeline completed successfully. See <code>top_candidates.csv</code> for full data.</p>
</div>
</body>
</html>"""

    html_path = os.path.join(str(output_dir), "report.html")
    with open(html_path, "w") as f:
        f.write(html)
    log.info(f"  HTML report saved: {html_path}")

    return html_path, scatter_path, hist_path


def print_summary(
    n_total: int, n_filtered: int,
    top10: List[CompoundRecord],
    validation_ok: bool, redock_rmsd: Optional[float],
    deps: Dict[str, Any],
) -> None:
    """Log a final pipeline summary."""
    n_docked = sum(1 for r in top10 if r.pb2pa_allosteric_energy is not None)
    n_selectivity_pass = sum(
        1 for r in top10
        if r.selectivity_index is not None and r.selectivity_index >= CONFIG.selectivity_index_threshold
    )

    log.info("=" * 60)
    log.info("  PIPELINE SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total compounds generated:     {n_total}")
    log.info(f"  After filtering:               {n_filtered}")
    log.info(f"  Top candidates reported:       {len(top10)}")
    log.info(f"  Successfully docked:           {n_docked}")
    log.info(f"  Selectivity pass (SI >= 2.0):  {n_selectivity_pass}")
    log.info(f"  Docking engine:                {'Vina' if deps.get('USE_VINA', False) else 'RDKit Shape (fallback)'}")
    if redock_rmsd is not None:
        log.info(f"  Redocking RMSD:                {redock_rmsd:.3f} Å")
    else:
        log.info("  Redocking RMSD:                N/A")
    log.info(f"  Redocking validated:           {validation_ok}")
    log.info(f'  CSV report:                    {CONFIG.output_dir / "top_candidates.csv"}')
    log.info("=" * 60)
