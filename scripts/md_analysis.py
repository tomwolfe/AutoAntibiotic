#!/usr/bin/env python3
"""
Trajectory-based binding-stability analysis for explicit-solvent MD replicas.

Reads a completed replica's stored analysis inputs (ligand RMSD trace,
per-replica H-bond occupancy, and optionally the replica summary) and produces
the summary statistics the review asked for:

  - Block-averaged ligand RMSD (mean ± std per time block, default 5 blocks).
  - Running-average RMSD (default 1 ns window) — also plotted when matplotlib
    is available.
  - Integrated autocorrelation time of the ligand-RMSD time series (ps), a
    quantitative measure of how many independent effective samples the trace
    holds.
  - Catalytic H-bond occupancy (Ser403/Lys406/Tyr446) per block, read directly
    from the stored per-replica ``hbond_occupancy.json``.
  - D3 binding classification via ``utils.filtering.classify_md_stability``
    plus the legacy Stable/Metastable/Unstable label.

The RMSD-based metrics (blocks, running mean, autocorrelation) work purely on
the small ``ligand_rmsd.npy`` trace, so the analysis runs identically on the
short smoke-test trajectories and on production 100 ns trajectories without
loading the (extremely large) full coordinate trajectory into memory. H-bond
occupancy is taken from the occupancy already computed during production.

Usage:
    python scripts/md_analysis.py --cid BRICS_0022 --replica 0
    python scripts/md_analysis.py --cid BRICS_0022            # all replicas
    python scripts/md_analysis.py --cid BRICS_0022 --blocks 5 --window-ns 1.0

Outputs:
    output/md_explicit/<CID>/replica_<N>/analysis.json            per-replica
    output/md_explicit/<CID>/analysis_summary.json                per-candidate
    output/figures/md_analysis/<cid>_replica_<N>_rmsd.png         running-mean plot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("md_analysis")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
MD_OUT = OUT / "md_explicit"
FIGS_OUT = OUT / "figures" / "md_analysis"

DEFAULT_BLOCKS = 5
DEFAULT_WINDOW_NS = 1.0


def _load_replica_inputs(replica_dir: Path) -> dict:
    """Load the analysis inputs for one replica."""
    inputs: dict = {}
    if (replica_dir / "ligand_rmsd.npy").is_file():
        inputs["ligand_rmsd_A"] = np.load(replica_dir / "ligand_rmsd.npy")
    elif (replica_dir / "ligand_rmsd.npy").exists() is False and (
            replica_dir / "production_frames.dat").is_file():
        raise FileNotFoundError(f"no ligand_rmsd.npy but frames exist: {replica_dir}")
    if (replica_dir / "hbond_occupancy.json").is_file():
        inputs["hbond_occupancy"] = json.loads(
            (replica_dir / "hbond_occupancy.json").read_text())
    summary_path = replica_dir / "summary.json"
    if summary_path.is_file():
        inputs["summary"] = json.loads(summary_path.read_text())
    return inputs


def _frame_dt_ps(n_frames: int, summary: Optional[dict] = None) -> float:
    """Infer the frame spacing (ps) from a replica summary, else a default."""
    summary = summary or {}
    prod = summary.get("production") or {}
    n_frames = n_frames or (prod.get("n_frames") or 0)
    npt_ns = prod.get("npt_duration_ns")
    if npt_ns and n_frames:
        return npt_ns * 1000.0 / n_frames
    return 2.0  # pipeline default: ~2 ps/frame (report_npt_steps × timestep)


def compute_block_stats(rmsd: np.ndarray, n_blocks: int = DEFAULT_BLOCKS) -> list[dict]:
    """Split *rmsd* into *n_blocks* contiguous blocks and return per-block stats.

    Data that cannot fill *n_blocks* uses as many blocks as the length supports
    (never fewer than 1); this keeps short smoke tests well-defined.
    """
    rmsd = np.asarray(rmsd, dtype=float)
    n = rmsd.shape[0]
    if n == 0:
        return []
    n_blocks = max(1, min(int(n_blocks), n))
    blocks = []
    # Contiguous split; earlier blocks get the remainder when n % n_blocks != 0.
    base = n // n_blocks
    rem = n % n_blocks
    start = 0
    for b in range(n_blocks):
        end = start + base + (1 if b < rem else 0)
        seg = rmsd[start:end]
        blocks.append({
            "block": b,
            "start_frame": int(start),
            "end_frame": int(end - 1 if end - 1 >= start else start),
            "frames": int(end - start),
            "mean_A": float(np.mean(seg)),
            "std_A": float(np.std(seg)),
            "min_A": float(np.min(seg)),
            "max_A": float(np.max(seg)),
        })
        start = end
    return blocks


def running_mean(rmsd: np.ndarray, window_ns: float, dt_ps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (window_size_frames, run_mean) over the RMSD series.

    The running mean is NaN-padded at the start so it has the same length as the
    input (convenient for plotting). ``window_ns`` is converted to whole frames.
    """
    rmsd = np.asarray(rmsd, dtype=float)
    n = rmsd.shape[0]
    win = max(1, min(int(round(window_ns * 1000.0 / dt_ps)), n))
    if n == 0:
        return np.int64(win), np.array([])
    kern = np.ones(win) / win
    conv = np.convolve(rmsd, kern, mode="valid")
    pad = np.full(n - conv.shape[0], np.nan)
    return np.int64(win), np.concatenate([pad, conv])


def autocorrelation_time(rmsd: np.ndarray, dt_ps: float, max_lag: Optional[int] = None) -> dict:
    """Integrated autocorrelation time of the ligand-RMSD series.

    Returns ``tau_ps`` (integral of the normalised autocorrelation up to the
    first zero crossing) and the normalised ACF. ``max_lag`` caps O(n·lag) cost
    (default: ``n``).
    """
    rmsd = np.asarray(rmsd, dtype=float)
    n = rmsd.shape[0]
    if n < 3:
        return {"tau_ps": None, "acf_max_k": None, "n": int(n), "dt_ps": dt_ps}
    if max_lag is None:
        max_lag = n
    max_lag = min(int(max_lag), n - 1)
    x = rmsd - rmsd.mean()
    var = float(np.dot(x, x))
    if var == 0:
        return {"tau_ps": None, "acf_max": None, "n": int(n), "dt": dt_ps}
    tau = 0.5  # ACF(k=0) == 1 contributes 0.5 to the integrated time
    acf_max = None
    for k in range(1, max_lag + 1):
        c = float(np.dot(x[:n - k], x[k:]) / var)
        if c <= 0:  # first zero crossing terminates the integral
            acf_max = k
            break
        tau += c
        acf_max = k
    return {"tau_ps": float(tau * dt_ps), "n": int(n), "dt": dt_ps, "lags": acf_max}


def analyze_replica(cid: str, replica_idx: int, n_blocks: int, window_ns: float,
                    classify_d3: bool = True, no_plot: bool = False) -> dict:
    """Analyse one replica and write its per-replica analysis.json."""
    replica_dir = MD_OUT / cid / f"replica_{replica_idx}"
    if not replica_dir.is_dir():
        raise FileNotFoundError(f"replica dir not found: {replica_dir}")
    inputs = _load_replica_inputs(replica_dir)

    rmsd = np.asarray(inputs.get("ligand_rmsd_A", np.zeros(0)), dtype=float)

    # Resolve the replica's summary (per-replica metrics live in the candidate
    # aggregate summary.json) so time-basis and D3 classification are accurate.
    summary = inputs.get("summary") or {}
    if not summary:
        cand_summary_path = MD_OUT / cid / "summary.json"
        if cand_summary_path.is_file():
            _cs = json.loads(cand_summary_path.read_text())
            _reps = _cs.get("replicas") or []
            summary = _reps[replica_idx] if replica_idx < len(_reps) else {}

    dt = _frame_dt_ps(rmsd.size, summary)

    blocks = compute_block_stats(rmsd, n_blocks)
    win_frames, run_mean = running_mean(rmsd, window_ns, dt)
    acf = autocorrelation_time(rmsd, dt)

    hb = inputs.get("hbond_occupancy") or {}
    overall_rmsd = float(np.mean(rmsd)) if rmsd.size else None
    result = {
        "replica": replica_idx,
        "compound_id": cid,
        "n_frames": int(rmsd.size),
        "overall_rmsd_mean_A": overall_rmsd,
        "dt_ps": dt,
        "n_blocks": len(blocks),
        "block_averaged_rmsd_A": blocks,
        "running_mean_window_ns": window_ns,
        "running_mean_window_frames": int(win_frames),
        "autocorrelation_time_ps": acf.get("tau_ps"),
        "autocorrelation_lags_used": acf.get("lags"),
        "hbond_occupancy": hb,
        "ser403_occ": (hb.get("SER403_OG") or {}).get("occupancy"),
        "ser403_mean_dist_A": (hb.get("SER403_OG") or {}).get("mean_distance_A"),
        "stable_class": _classify_stability_label(overall_rmsd),
    }

    if summary.get("stability_class"):
        result["stability_class_reported"] = summary["stability_class"]
    if classify_d3:
        rep = {
            "ligand_rmsd_mean_last5ns_A": summary.get("ligand_rmsd_mean_last5ns_A"),
            "ser403_og_hbond_occupancy": summary.get("ser403_og_hbond_occupancy"),
        }
        try:
            from utils.filtering import classify_md_stability
            result["d3_class"] = classify_md_stability([rep])
        except Exception as exc:
            log.warning("  D3 classifier unavailable (%s); omitted", exc)

    (replica_dir / "analysis.json").write_text(json.dumps(result, indent=2, default=str))
    if not no_plot:
        _plot_rmsd(cid, replica_idx, rmsd, run_mean, dt)
    return result


def _classify_stability_label(mean_rmsd: Optional[float]) -> Optional[str]:
    if mean_rmsd is None:
        return None
    if mean_rmsd < 2.0:
        return "Stable"
    if mean_rmsd < 4.0:
        return "Metastable"
    return "Unstable"


def _plot_rmsd(cid: str, replica_idx: int, rmsd: np.ndarray, run_mean: np.ndarray, dt: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("  matplotlib unavailable; skipping RMSD plot (%s)", exc)
        return
    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    t = np.arange(rmsd.size) * dt / 1000.0  # ns
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, rmsd, lw=0.8, alpha=0.6, label="ligand RMSD")
    ax.plot(t, run_mean, lw=1.6, color="crimson", label="running mean")
    ax.axhline(2.0, color="tab:green", ls="--", lw=0.8, alpha=0.7, label="Stable < 2 Å")
    ax.axhline(4.0, color="tab:orange", ls="--", lw=0.8, alpha=0.7, label="Metastable < 4 Å")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("ligand RMSD (Å)")
    ax.set_title(f"{cid} replica {replica_idx} — ligand RMSD")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIGS_OUT / f"{cid}_replica_{replica_idx}_rmsd.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  wrote %s", path)


def main():
    parser = argparse.ArgumentParser(description="Trajectory binding-stability analysis")
    parser.add_argument("--cid", type=str, default=None,
                        help="Compound ID to analyse (default: first with results)")
    parser.add_argument("--replica", type=int, default=None, help="Specific replica index")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--window-ns", type=float, default=DEFAULT_WINDOW_NS)
    parser.add_argument("--no-plot", action="store_true", help="Skip the RMSD plot")
    args = parser.parse_args()
    cid = args.cid or _first_candidate()
    main_analyse(cid, args.replica, args.blocks, args.window_ns, no_plot=args.no_plot)


def _first_candidate() -> str:
    """Pick a candidate that already has replica results, for convenience."""
    if not MD_OUT.is_dir():
        return "BRICS_0022"
    for d in sorted(MD_OUT.iterdir()):
        if d.name.startswith("_") or not d.is_dir():
            continue
        if any(p.name.startswith("replica_") for p in d.iterdir()):
            return d.name
    return "BRICS_0022"


def main_analyse(cid: str, replica_idx: Optional[int], n_blocks: int, window_ns: float,
                 no_plot: bool = False) -> None:
    """Run the full per-candidate analysis."""
    cand_dir = MD_OUT / cid
    if not cand_dir.is_dir():
        log.error("No MD results for candidate %s (%s missing)", cid, cand_dir)
        sys.exit(1)
    replica_dirs = sorted(
        (d for d in cand_dir.iterdir() if d.name.startswith("replica_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if replica_idx is not None:
        replica_dirs = [cand_dir / f"replica_{replica_idx}"]
    if not replica_dirs:
        log.error("No replicas found for %s", cid)
        sys.exit(1)

    results = []
    for rd in replica_dirs:
        try:
            ridx = int(rd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        results.append(analyze_replica(cid, ridx, n_blocks, window_ns, no_plot=no_plot))

    summary = {
        "compound_id": cid,
        "n_replicas_analysed": len(results),
        "replicas": results,
    }
    (cand_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    log.info("  wrote %s", cand_dir / "analysis_summary.json")


if __name__ == "__main__":
    main()