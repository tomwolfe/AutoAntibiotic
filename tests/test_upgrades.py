"""Unit tests for the scientific upgrades added in the review-gap closing pass.

Covers:
  * DUD-E benchmark bootstrap confidence intervals (Phase 2)
  * Trajectory-frame sampling + solvent-stripping index mapping (Phase 5)
  * Selectivity_Index_CI merge helper (Phase 6)

These are pure-function tests; no Vina / OpenMM invocation.
"""
import importlib.util
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _load(name):
    """Load a scripts/<name> module by file path (scripts/ is not a package)."""
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
# Phase 2 — DUD-E benchmark bootstrapped enrichment metrics
# ─────────────────────────────────────────────────────────────────────────
def test_enrichment_metrics_separates_strong_signal():
    dude = _load("dude_benchmark.py")
    rng = np.random.RandomState(0)
    # Actives dock to ~ -11 kcal/mol, decoys ~ -5: clean separation.
    E = np.r_[rng.normal(-11, 0.5, 60), rng.normal(-5, 1, 600)]
    labels = np.r_[np.ones(60), np.zeros(600)].astype(int)
    m = dude.enrichment_metrics(labels, -E)
    assert m["auc"] > 0.9
    assert m["ef_1pct"] >= 5.0
    assert m["ef_5pct"] >= 5.0


def test_enrichment_small_n_ef_not_degenerate():
    dude = _load("dude_benchmark.py")
    labels = np.array([1, 0, 0, 0, 0])
    scores = np.array([9, 3, 4, 5, 6])  # active best
    m = dude.enrichment_metrics(labels, scores)
    assert m["ef_1pct"] == 5.0
    assert m["auc"] == 1.0


def test_bootstrap_cis_bounds_and_range():
    dude = _load("dude_benchmark.py")
    rng = np.random.RandomState(0)
    E = np.r_[rng.normal(-11, 0.3, 60), rng.normal(-5, 1, 600)]
    labels = np.r_[np.ones(60), np.zeros(600)].astype(int)
    cis = dude.bootstrap_cis(labels, -E, n_resamples=100, seed=7)
    for key, (mean, lo, hi) in cis.items():
        assert lo <= mean <= hi, key
        if key == "auc":
            assert 0.0 <= lo <= hi <= 1.0


# ─────────────────────────────────────────────────────────────────────────
# Phase 5 — trajectory MM-GBSA frame sampling / index mapping
# ─────────────────────────────────────────────────────────────────────────
def test_select_trajectory_frames_last_window():
    mm = _load("mmgbsa_analysis.py")
    frames = mm.select_trajectory_frames(n_frames=100, dt_ps_per_frame=100.0,
                                         window_ns=5.0, sample_ps=100.0)
    assert frames and frames[0] >= 50  # only the last 5 ns
    assert len(frames) <= 51


def test_select_trajectory_frames_unknown_dt_fallback():
    mm = _load("mmgbsa_analysis.py")
    frames = mm.select_trajectory_frames(n_frames=40, dt_ps_per_frame=None)
    assert frames == list(range(20, 40))  # final 50% fallback


def test_select_trajectory_frames_default_window_is_50ns():
    # The ensemble-MMGBSA protocol (review gap) samples the last 50 ns by
    # default; with a 50 ns trajectory the whole file is used (start == 0).
    mm = _load("mmgbsa_analysis.py")
    frames = mm.select_trajectory_frames(n_frames=500, dt_ps_per_frame=100.0)
    assert frames[0] == 0  # 50 ns window == full 50 ns production
    # For a 100 ns trajectory only the last 50 ns are sampled.
    frames100 = mm.select_trajectory_frames(n_frames=1000, dt_ps_per_frame=100.0)
    assert frames100[0] == 500
    assert len(frames100) <= 501


def test_select_trajectory_frames_100ps_cadence():
    # Every 100 ps from the last window, at 10 ps/frame -> every 10 frames.
    # 200 ns at 10 ps/frame = 20,000 frames; last 50 ns = frames [15000, 20000).
    mm = _load("mmgbsa_analysis.py")
    frames = mm.select_trajectory_frames(n_frames=20000, dt_ps_per_frame=10.0,
                                         window_ns=50.0, sample_ps=100.0)
    assert frames[0] == 15000
    assert all((b - a) == 10 for a, b in zip(frames, frames[1:]))
    assert frames[-1] < 20000


def test_positions_for_indices():
    mm = _load("mmgbsa_analysis.py")
    pos = np.arange(9).reshape(3, 3).tolist()
    out = mm.positions_for_indices(pos, [0, 2])
    assert out == [[0, 1, 2], [6, 7, 8]]


# ─────────────────────────────────────────────────────────────────────────
# Phase 6 — SI confidence-interval merge helper
# ─────────────────────────────────────────────────────────────────────────
def test_integrate_si_ci_merge_map(tmp_path):
    mod = _load("integrate_si_ci.py")
    csv_path = tmp_path / "ci.csv"
    csv_path.write_text(
        "Compound_ID,Selectivity_Index_CI\n"
        "BRICS_0022,2.13 ± 0.11 [1.92–2.34]\n"
        "ALL_QU04,2.07 ± 0.10 [1.86–2.28]\n"
    )
    ci_map = mod.load_ci_map(csv_path)
    assert ci_map["BRICS_0022"].startswith("2.13 ± 0.11")
    assert ci_map.get("MISSING", "N/A") == "N/A"


# ─────────────────────────────────────────────────────────────────────────
# Phase 7 (Obj5) — Transparent MD-informed composite re-rank (honesty)
# ─────────────────────────────────────────────────────────────────────────
def test_composite_rerank_ranks_md_evidence_above_unvalidated(tmp_path, monkeypatch):
    cr = _load("composite_rerank.py")
    # Base docking CSV: two high-SI candidates, one with no MD data at all.
    csv_p = tmp_path / "top_candidates.csv"
    csv_p.write_text(
        "Compound_ID,Selectivity_Index,SI_Tier\n"
        "BRICS_0022,2.13,Strong\n"
        "ALL_QU04,2.07,Strong\n"
        "ALL_NOMD,3.50,Strong\n"
    )
    # MD summary: BRICS_0022 = Stable (D3-validated), ALL_QU04 = Dissociated.
    md_dir = tmp_path / "md"; md_dir.mkdir()
    (md_dir / "BRICS_0022").mkdir()
    (md_dir / "BRICS_0022" / "summary.json").write_text(
        '{"success": true, "stability_class_d3": "Validated"}')
    (md_dir / "ALL_QU04").mkdir()
    (md_dir / "ALL_QU04" / "summary.json").write_text(
        '{"success": true, "stability_class_d3": "Dissociated"}')

    monkeypatch.setattr(cr, "CSV_PATH", csv_p)
    monkeypatch.setattr(cr, "MD_OUT", md_dir)
    monkeypatch.setattr(cr, "MMGBSA_PATH", tmp_path / "mmgbsa_results.json")
    monkeypatch.setattr(cr, "WATER_PATH", tmp_path / "water_analysis.json")

    records = cr._load_md_stability()
    assert records["BRICS_0022"]["stability_class"] == "Validated"
    assert records["ALL_QU04"]["stability_class"] == "Dissociated"
    assert "ALL_NOMD" not in records

    ranked = cr._build_and_sort_records(list(csv.DictReader(open(csv_p))),
                                        md_stab=records, traj_mg={}, water={})
    ids = [r["compound_id"] for r in ranked]
    # The unvalidated high-SI candidate (ALL_NOMD) must rank BELOW the
    # MD-validated ones, even though its docking SI is highest.
    assert ids.index("ALL_NOMD") > ids.index("BRICS_0022")
    assert ids.index("ALL_NOMD") > ids.index("ALL_QU04")
    assert ranked[0]["compound_id"] == "BRICS_0022"
    # Honesty: the unvalidated candidate must be flagged, with no stability.
    nomd = next(r for r in ranked if r["compound_id"] == "ALL_NOMD")
    assert nomd["evidence"] == "unvalidated"
    assert nomd["d3_stability"] is None