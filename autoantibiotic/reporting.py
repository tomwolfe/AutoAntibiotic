from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.decomposition import PCA

from .config import CONFIG
from .models import CompoundRecord
from .io_utils import ensure_output_dir, log

# Imported lazily (in generate_html_report) to avoid circular imports
# _get_meta_scorer is used to compute SHAP explanations


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
            "ML_Score": (
                f"{rec.ml_score:.2f}" if rec.ml_score is not None
                else "N/A"
            ),
            "Selectivity_Index": (
                f"{rec.selectivity_index:.2f}" if rec.selectivity_index is not None
                else "N/A"
            ),
            "Max_Similarity": f"{rec.max_similarity:.3f}",
            "IFP_Score": (
                f"{rec.ifp_score:.3f}" if rec.ifp_score is not None else "N/A"
            ),
            "Passes_Lipinski": str(rec.passes_lipinski),
            "QED_Score": f"{rec.qed_score:.3f}",
            "ADMET_Flags": "; ".join(rec.admet_flags) if rec.admet_flags else "N/A",
            "Scoring_Method": scoring_method,
            "Docking_Method": rec.docking_method,
            "Binding_Mode_Notes": rec.resistance_notes.replace("; ", " | "),
        })

    csv_path = CONFIG.output_dir / CONFIG.csv_report_name
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    log.info(f'  CSV report saved: {csv_path}')
    return str(csv_path)


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


def _make_scatter_plot(top10: List[CompoundRecord]) -> str:
    """Create an interactive scatter plot of Binding Energy vs Selectivity."""
    scatter_data = [
        (r.pb2pa_allosteric_energy, r.selectivity_index, r.compound_id, r.qed_score)
        for r in top10
        if r.pb2pa_allosteric_energy is not None and r.selectivity_index is not None
    ]
    if not scatter_data:
        return ""

    energies = [d[0] for d in scatter_data]
    sis = [d[1] for d in scatter_data]
    cids = [d[2] for d in scatter_data]
    qeds = [d[3] for d in scatter_data]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=energies,
        y=sis,
        mode="markers+text",
        text=cids,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            size=10,
            color=qeds,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="QED"),
            line=dict(width=1, color="black"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Energy: %{x:.2f} kcal/mol<br>"
            "SI: %{y:.2f}<br>"
            "QED: %{customdata:.3f}"
            "<extra></extra>"
        ),
        customdata=np.array(qeds),
    ))

    fig.add_hline(
        y=CONFIG.selectivity_index_threshold,
        line_dash="dash",
        line_color="red",
        opacity=0.6,
        annotation_text=f"SI threshold = {CONFIG.selectivity_index_threshold}",
        annotation_position="right",
    )

    fig.update_layout(
        title="Top Candidates: Binding Energy vs Selectivity",
        xaxis_title="Allosteric Binding Energy (kcal/mol)",
        yaxis_title="Selectivity Index",
        template="plotly_white",
        height=500,
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _make_qed_histogram(top50: List[CompoundRecord]) -> str:
    """Create an interactive histogram of QED scores."""
    qeds = [r.qed_score for r in top50 if r.qed_score > 0]
    if not qeds:
        return ""

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=qeds,
        nbinsx=20,
        marker=dict(
            color="mediumseagreen",
            line=dict(color="black", width=1),
        ),
        hovertemplate="QED: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(
        x=CONFIG.qed_threshold,
        line_dash="dash",
        line_color="red",
        opacity=0.6,
        annotation_text=f"QED cutoff = {CONFIG.qed_threshold}",
        annotation_position="top",
    )

    fig.update_layout(
        title="QED Distribution (Top 50 Candidates)",
        xaxis_title="QED Score",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        bargap=0.1,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _make_pca_plot(top50: List[CompoundRecord]) -> str:
    """Create a PCA projection of Morgan fingerprints coloured by QED score.

    Uses :class:`sklearn.decomposition.PCA` to reduce 2048-bit Morgan
    fingerprints to two principal components for visualising chemical
    diversity.
    """
    valid: List[CompoundRecord] = []
    fps: List[np.ndarray] = []

    for r in top50:
        mol = r.mol
        if mol is None:
            mol = Chem.MolFromSmiles(r.smiles)
            if mol is None:
                continue
            r.mol = mol
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, CONFIG.morgan_radius, nBits=CONFIG.morgan_nbits,
        )
        fps.append(np.array(fp, dtype=np.float64))
        valid.append(r)

    if len(valid) < 3:
        log.warning(f"  PCA plot requires ≥3 compounds, got {len(valid)}. Skipping.")
        return ""

    X = np.vstack(fps)
    pca = PCA(n_components=2, random_state=CONFIG.random_seed)
    coords = pca.fit_transform(X)

    var_ratio = pca.explained_variance_ratio_
    cids = [r.compound_id for r in valid]
    qeds = [r.qed_score for r in valid]
    energies = [
        r.pb2pa_allosteric_energy if r.pb2pa_allosteric_energy is not None
        else (r.shape_score or 0.0)
        for r in valid
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        text=cids,
        marker=dict(
            size=8,
            color=qeds,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="QED"),
            line=dict(width=0.5, color="black"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "PC1: %{x:.2f}<br>"
            "PC2: %{y:.2f}<br>"
            "QED: %{marker.color:.3f}<br>"
            "Energy: %{customdata:.2f}"
            "<extra></extra>"
        ),
        customdata=np.array(energies),
    ))

    fig.update_layout(
        title="Chemical Diversity (PCA of Morgan Fingerprints, Top 50)",
        xaxis_title=f"PC1 ({var_ratio[0] * 100:.1f}% variance)",
        yaxis_title=f"PC2 ({var_ratio[1] * 100:.1f}% variance)",
        template="plotly_white",
        height=500,
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def generate_html_report(
    top10: List[CompoundRecord],
    top50: List[CompoundRecord],
    output_dir: Path,
) -> Tuple[str, str, str]:
    """Phase 5.3 — Generate an HTML report with interactive Plotly charts.

    Creates scatter plot (Energy vs Selectivity), QED histogram, and
    PCA diversity plot as interactive Plotly charts embedded directly in
    the HTML page.

    Returns (html_path, scatter_path, hist_path).  The *scatter_path* and
    *hist_path* are empty strings since charts are now embedded inline.
    """
    log.info("─── Phase 5: HTML Report Generation ───")

    scatter_div = _make_scatter_plot(top10)
    hist_div = _make_qed_histogram(top50)
    pca_div = _make_pca_plot(top50)

    # ── SHAP explanations for each record ──────────────────────────
    from .ml_scoring.meta_scorer import _get_meta_scorer as _get_shap_scorer
    _meta_scorer_instance = _get_shap_scorer()
    shap_explanations: Dict[str, str] = {}
    if _meta_scorer_instance is not None:
        for rec in top10:
            try:
                shap_dict = _meta_scorer_instance.explain_prediction(rec)
                if shap_dict:
                    sorted_by_val = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
                    top_pos = [f"+{k}:{v:.3f}" for k, v in sorted_by_val[:3] if v > 0]
                    top_neg = [f"{k}:{v:.3f}" for k, v in sorted_by_val[-3:] if v < 0]
                    parts = top_pos + top_neg
                    shap_explanations[rec.compound_id] = ", ".join(parts) if parts else "N/A"
            except Exception:
                shap_explanations[rec.compound_id] = "N/A"

    table_rows = ""
    for i, rec in enumerate(top10):
        allosteric = f"{rec.pb2pa_allosteric_energy:.2f}" if rec.pb2pa_allosteric_energy is not None else "N/A"
        active = f"{rec.pb2pa_active_energy:.2f}" if rec.pb2pa_active_energy is not None else "N/A"
        si = f"{rec.selectivity_index:.2f}" if rec.selectivity_index is not None else "N/A"
        qed = f"{rec.qed_score:.3f}" if rec.qed_score else "N/A"
        ml_score = f"{rec.ml_score:.2f}" if rec.ml_score is not None else "N/A"
        admet_str = "; ".join(rec.admet_flags) if rec.admet_flags else "N/A"
        shap_str = shap_explanations.get(rec.compound_id, "N/A")

        poor_admet = (
            "poor_solubility" in admet_str.lower()
            or "high herg" in admet_str.lower()
            or "lipinski violation" in admet_str.lower()
        )
        row_style = ' style="background-color:#ffcccc;"' if poor_admet else ""

        is_shape_fallback = rec.docking_method == "ShapeFallback"
        method_badge = (
            f'<span style="background-color:#ff9800;color:#fff;padding:2px 6px;'
            f'border-radius:4px;font-size:0.8em;">{rec.docking_method}</span>'
            if is_shape_fallback
            else f'<span style="background-color:#4caf50;color:#fff;padding:2px 6px;'
                 f'border-radius:4px;font-size:0.8em;">{rec.docking_method}</span>'
        )

        table_rows += (
            f"<tr{row_style}>"
            f"<td>{i + 1}</td>"
            f"<td>{rec.compound_id}</td>"
            f"<td style='font-size:0.8em;max-width:300px;word-break:break-all;'>{rec.smiles}</td>"
            f"<td>{allosteric}</td>"
            f"<td>{active}</td>"
            f"<td>{ml_score}</td>"
            f"<td>{si}</td>"
            f"<td>{qed}</td>"
            f"<td style='font-size:0.8em;max-width:180px;'>{shap_str}</td>"
            f"<td style='font-size:0.8em;max-width:200px;'>{admet_str}</td>"
            f"<td>{method_badge}</td>"
            f"<td style='font-size:0.8em;max-width:250px;'>{rec.resistance_notes}</td>"
            f"</tr>\n"
        )

    plotly_js = (
        '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'
    )

    def _section(title: str, content: str) -> str:
        if not content:
            return ""
        return f"<h2>{title}</h2>\n{content}\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoAntibiotic Discovery Report</title>
{plotly_js}
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; }}
h1 {{ color: #1a5276; }}
h2 {{ color: #2e86c1; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background-color: #2e86c1; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
tr.admet-warning {{ background-color: #ffcccc; }}
.plotly-graph-div {{ margin: 10px 0; }}
.footer {{ margin-top: 30px; color: #777; font-size: 0.9em; }}
.warning-text {{ color: #cc0000; font-weight: bold; }}
.admet-info {{ font-size: 0.8em; max-width: 200px; word-break: break-word; }}
</style>
</head>
<body>
<h1>AutoAntibiotic Discovery Pipeline — Top Candidates Report</h1>
<p>Generated by AutoAntibiotic v3.2 | MRSA PBP2a Inhibitor Screening</p>
<hr>

{_section("Binding Energy vs Selectivity", scatter_div)}

{_section("QED Score Distribution", hist_div)}

{_section("Chemical Diversity (PCA of Morgan Fingerprints)", pca_div)}

<h2>Top {len(top10)} Candidates</h2>
<p>Rows highlighted in <span class="warning-text">red</span> have poor ADMET flags despite strong docking scores.</p>
<table>
<tr>
  <th>Rank</th>
  <th>ID</th>
  <th>SMILES</th>
  <th>Allosteric (kcal/mol)</th>
  <th>Active (kcal/mol)</th>
  <th>ML Score</th>
  <th>Selectivity Index</th>
  <th>QED</th>
  <th>Top SHAP Features</th>
  <th>ADMET Flags</th>
  <th>Docking Method</th>
  <th>Resistance Notes</th>
</tr>
{table_rows}
</table>

<div class="footer">
<p>Pipeline completed successfully. See <code>top_candidates.csv</code> for full data.</p>
</div>
</body>
</html>"""

    html_path = os.path.join(str(output_dir), CONFIG.html_report_name)
    with open(html_path, "w") as f:
        f.write(html)
    log.info(f"  HTML report saved: {html_path}")

    return html_path, "", ""


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
    ifp_scores = [r.ifp_score for r in top10 if r.ifp_score is not None]
    if ifp_scores:
        log.info(f"  Average IFP similarity (top 10): {np.mean(ifp_scores):.3f}")
    log.info(f"  Redocking validated:           {validation_ok}")
    log.info(f'  CSV report:                    {CONFIG.output_dir / CONFIG.csv_report_name}')
    log.info("=" * 60)
