#!/usr/bin/env python3
"""
Saturation analysis for the DUD-E style PBP2a enrichment benchmark (D4 / D5).

The single exhaustiveness-8 DUD-E run reports AUC = 0.134 (FAIL). A single
setting cannot tell us whether that poor discrimination is:

  (i)   an *undersampling* artifact  -- rigid-receptor Vina at shallow
        exhaustiveness does not sample enough near-native poses to rank
        actives ahead of property-matched decoys, so raising exhaustiveness
        would recover enrichment; or

  (ii)  a *fundamental* limitation -- rigid-receptor Vina scored against the
        apo (1VQQ) conformer cannot separate weak PBP2a beta-lactam actives
        (pIC50 3.7--5.0) from property-matched decoys on the shallow, polar,
        solvent-exposed active site, so no exhaustiveness recovers AUC >= 0.7.

This script answers that question by sweeping the docking exhaustiveness
across multiple levels (default 8, 16, 32, 64) and plotting AUC / EF_{1%} /
BEDROC as a function of exhaustiveness. If the metrics plateau at a level
well below the validation thresholds, the poor discrimination is
characterised as fundamental to the target--method pairing rather than an
undersampling artifact -- and that is reported as a *validated negative
result* (a key finding about PBP2a druggability), not hidden.

Per-exhaustiveness results are cached (output/enrichment_saturation_cache/)
so an interrupted run can resume and so full sweeps do not re-dock already
completed settings. A `--limit` flag supports quick smoke tests on a subset
of the benchmark set (for validation of the plumbing, not publication metrics).

Usage:
    AUTOANTIBIOTIC_MODE=science python scripts/enrichment_saturation_analysis.py
    AUTOANTIBIOTIC_MODE=science python scripts/enrichment_saturation_analysis.py \\
        --exhaustiveness-list 8,16,32,64
    AUTOANTIBIOTIC_MODE=science python scripts/enrichment_saturation_analysis.py \\
        --limit 24            # quick smoke test on 24 compounds

Outputs:
    output/enrichment_saturation.json     — per-exhaustiveness metrics + verdict
    output/enrichment_saturation.png      — AUC / EF_{1%} / BEDROC vs exhaustiveness
    output/enrichment_saturation_cache/   — per-exhaustiveness score caches (resume)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import discovery_pipeline as P
from config.constants import ACTIVE_BOX_SIZE, ACTIVE_SITE_RESIDUES
from utils.docking import dock_compound
import scripts.dude_benchmark as DB

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("saturation")

REPO = DB.REPO
OUT = DB.OUT
CACHE = os.path.join(OUT, "enrichment_saturation_cache")

# Validation thresholds shared with verify_success.py / dude_benchmark.py.
AUC_MIN = 0.7

DEFAULT_EXHAUSTIVENESS = (8, 16, 32, 64)


def records_fingerprint(records, labels) -> str:
    """Stable hash over (compound_id, smiles, label) so cached scores are only
    reused when the actives+decoys set is identical across runs."""
    entries = sorted((r.compound_id, r.smiles, str(l)) for r, l in zip(records, labels))
    blob = "|".join("\x1f".join(e) for e in entries).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(exhaustiveness: int) -> str:
    return os.path.join(CACHE, f"ex{exhaustiveness}.json")


def _load_cache(exhaustiveness: int, fingerprint: str) -> dict | None:
    path = _cache_path(exhaustiveness)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as exc:
        log.warning(f"  cache read failed for ex={exhaustiveness}: {exc}")
        return None
    if data.get("records_fingerprint") != fingerprint:
        log.info(f"  cache ex={exhaustiveness} stale (set changed); discarding")
        return None
    return data


def _save_cache(exhaustiveness: int, data: dict) -> None:
    os.makedirs(CACHE, exist_ok=True)
    with open(_cache_path(exhaustiveness), "w") as fh:
        json.dump(data, fh, indent=2)


def _dock_and_score(
    records, labels, receptor_pdbqt, active_center, active_box,
    work_dir, exhaustiveness, fingerprint, limit=None,
) -> dict:
    """Run (or reuse cached) docking at one exhaustiveness and compute metrics."""
    cached = _load_cache(exhaustiveness, fingerprint)
    if cached is not None:
        log.info(f"  ex={exhaustiveness}: using cached {len(cached['scores'])} scores")
        return cached

    dock_records = records
    tag = f"sat_ex{exhaustiveness}"
    if limit:
        dock_records = records[:limit]
        tag += "_smoke"
    log.info(f"  ex={exhaustiveness}: docking {len(dock_records)} compounds...")
    results = P._dock_compounds_parallel(
        dock_records, receptor_pdbqt, active_center, active_box,
        work_dir, tag,
        exhaustiveness=exhaustiveness,
    )
    energies = {rec.compound_id: energy for rec, energy in results}
    ids = [r.compound_id for r in dock_records]
    scores = [-(energies[cid] if energies[cid] is not None else 1e9) for cid in ids]
    labels_arr = np.asarray([labels[ids.index(c)] for c in ids], dtype=int)

    fp, tp, auc = DB.compute_roc(labels_arr, scores)
    metrics = DB.enrichment_metrics(labels_arr, scores, alpha=20.0)
    cis = DB.bootstrap_cis(labels_arr, scores, n_resamples=1000, seed=DB.RANDOM_SEED)

    data = {
        "exhaustiveness": exhaustiveness,
        "records_fingerprint": fingerprint,
        "n_compounds": len(ids),
        "n_actives": int(labels_arr.sum()),
        "n_decoys": len(ids) - int(labels_arr.sum()),
        "auc": metrics["auc"],
        "ef_1pct": metrics["ef_1pct"],
        "ef_5pct": metrics["ef_5pct"],
        "ef_10pct": metrics["ef_10pct"],
        "bedrock_alpha20": metrics["bedrock_alpha20"],
        "ci_95_bootstrap_1000": cis,
        "scores": {cid: energies.get(cid) for cid in ids},
        "labels": {cid: int(labels[ids.index(cid)]) for cid in ids},
    }
    _save_cache(exhaustiveness, data)
    log.info(f"  ex={exhaustiveness}: AUC={metrics['auc']:.3f} "
             f"EF_1%={metrics['ef_1pct']:.2f} BEDROC={metrics['bedrock_alpha20']:.3f}")
    return data


def _assess_saturation(results: list[dict]) -> dict:
    """Classify whether enrichment saturates at/above the validation threshold
    or plateaus below it (fundamental limitation)."""
    levels = sorted(r["exhaustiveness"] for r in results)
    aucs = {r["exhaustiveness"]: r["auc"] for r in results}
    max_auc = max(aucs.values())
    last_auc = aucs[levels[-1]]
    # Saturation defined as a <=0.05 change in AUC over the upper half of the sweep.
    changes = [abs(aucs[levels[i + 1]] - aucs[levels[i]]) for i in range(len(levels) - 1)]
    plateau = bool(changes) and max(changes[-max(1, len(changes) // 2):]) <= 0.05
    threshold_met = max_auc >= AUC_MIN
    if threshold_met:
        classification = "PASS"
        interpretation = (
            "Enrichment reached the validation threshold at some exhaustiveness; "
            "the primary-screen failure at ex=8 was an undersampling artifact."
        )
    elif plateau:
        classification = "FUNDAMENTAL_LIMITATION"
        interpretation = (
            "AUC saturates at (or below) the validation threshold as exhaustiveness "
            "increases; rigid-receptor Vina against the apo conformer cannot separate "
            "the weak PBP2a actives from property-matched decoys. Reported as a "
            "validated negative result on PBP2a/rigid-docking discrimination."
        )
    else:
        classification = "UNSATURATED"
        interpretation = (
            "Metrics were still increasing with exhaustiveness; a higher setting may "
            "yet recover enrichment. Extend the sweep to higher exhaustiveness before "
            "concluding the failure is fundamental."
        )
    return {
        "exhaustiveness_levels": levels,
        "auc_by_exhaustiveness": {k: round(v, 4) for k, v in aucs.items()},
        "last_auc": round(last_auc, 4),
        "max_auc": round(max_auc, 4),
        "plateau": plateau,
        "auc_min": AUC_MIN,
        "classification": classification,
        "interpretation": interpretation,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep docking exhaustiveness to characterise DUD-E enrichment saturation"
    )
    parser.add_argument("--exhaustiveness-list", type=str, default="8,16,32,64",
                        help="Comma-separated exhaustiveness levels to sweep (default 8,16,32,64)")
    parser.add_argument("--decoy-pool", type=str, default=None,
                        help="SMILES library CSV(es) used to source decoys")
    parser.add_argument("--limit", type=int, default=None,
                        help="Dock only the first N benchmark compounds (smoke test only)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached per-exhaustiveness results and re-dock")
    args = parser.parse_args(argv)

    config = P.load_config()
    config["mode"] = "science"
    deps = P.check_dependencies()
    if not deps["USE_VINA"]:
        log.error("Vina required for saturation analysis. Aborting.")
        sys.exit(1)

    levels = tuple(int(x) for x in args.exhaustiveness_list.split(",") if x.strip())
    if not levels:
        parser.error("--exhaustiveness-list must contain at least one level")

    pool_path = args.decoy_pool or DB.DEFAULT_DECOY_POOL
    if not all(os.path.exists(p) for p in pool_path.split(",")):
        log.error(f"Decoy pool files not found: {pool_path}")
        sys.exit(1)

    os.makedirs(CACHE, exist_ok=True)
    pdb_dir = os.path.join(OUT, "pdb_dude")
    work_dir = os.path.join(OUT, "workdir_saturation")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Receptor prep with the same caching wrapper as dude_benchmark.
    _orig_clean = P.clean_pdb_structure
    def _cached_clean(pdb_path, out_path, **kw):
        pdbqt = out_path.replace(".pdb", ".pdbqt")
        if (os.path.exists(out_path) and os.path.getsize(out_path) > 0
                and os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0):
            return pdbqt
        return _orig_clean(pdb_path, out_path, **kw)
    P.clean_pdb_structure = _cached_clean

    targets = P.prepare_targets(pdb_dir, work_dir, deps, config=config)
    pb2pa = targets["PBP2a"]
    receptor_pdbqts = pb2pa["receptor_pdbqts"]
    active_center = pb2pa["active_center"]
    cleaned_pdb = pb2pa["cleaned_pdb"]
    active_box = P._auto_box_size(
        cleaned_pdb, active_center, ACTIVE_BOX_SIZE,
        min_size=15.0, max_size=20.0, site_residues=ACTIVE_SITE_RESIDUES,
    )
    receptor_pdbqt = receptor_pdbqts[0]
    log.info(f"  Docking against apo receptor (1VQQ): {receptor_pdbqt}")

    records, labels = DB.load_benchmark(
        os.path.join(DB.DATA, "active_site_actives.csv"), pool_path,
        n_decoys_per_active=DB.DECOYS_PER_ACTIVE,
    )
    fingerprint = records_fingerprint(records, labels)
    log.info(f"  Benchmark set: {sum(labels)} actives, {len(labels) - sum(labels)} decoys")

    results = []
    for ex in levels:
        r = _dock_and_score(
            records, labels, receptor_pdbqt, active_center, active_box,
            work_dir, ex, fingerprint, limit=args.limit,
        )
        results.append(r)

    assessment = _assess_saturation(results)
    payload = {
        "method": "DUD-E style enrichment saturation sweep (rigid Vina, apo 1VQQ)",
        "decoy_pool": pool_path,
        "exhaustiveness_sweep": levels,
        "results": results,
        "assessment": assessment,
    }
    with open(os.path.join(OUT, "enrichment_saturation.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info(f"  Sierra: {os.path.join(OUT, 'enrichment_saturation.json')}")

    _plot(results, assessment)

    log.info("")
    log.info("=" * 62)
    log.info("  ENRICHMENT SATURATION SWEEP")
    log.info("  " + "-" * 58)
    log.info(f"  {'ex':<6}{'AUC':<10}{'EF_1%':<10}{'EF_5%':<10}{'BEDROC':<10}")
    for r in results:
        log.info(f"  {r['exhaustiveness']:<6}{r['auc']:<10.3f}{r['ef_1pct']:<10.2f}"
                 f"{r['ef_5pct']:<10.2f}{r['bedrock_alpha20']:<10.3f}")
    log.info("  " + "-" * 58)
    log.info(f"  Classification: {assessment['classification']}")
    log.info(f"  {assessment['interpretation']}")
    log.info("=" * 62)
    sys.exit(0)


def _plot(results: list[dict], assessment: dict) -> None:
    levels = sorted(r["exhaustiveness"] for r in results)
    exes = np.asarray(levels)
    aucs = np.asarray([next(r["auc"] for r in results if r["exhaustiveness"] == e) for e in levels])
    efs = np.asarray([next(r["ef_1pct"] for r in results if r["exhaustiveness"] == e) for e in levels])
    bdr = np.asarray([next(r["bedrock_alpha20"] for r in results if r["exhaustiveness"] == e) for e in levels])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(exes, aucs, "o-", color="#2c7fb8", lw=2, label="ROC-AUC")
    ax.plot(exes, bdr, "s-", color="#7fcdbb", lw=2, label="BEDROC($\\alpha$=20)")
    ax.axhline(AUC_MIN, color="red", ls="--", lw=1, label=f"pass threshold (AUC={AUC_MIN})")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="random (0.5)")
    for e, a in zip(exes, aucs):
        ax.annotate(f"{a:.2f}", (e, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Vina exhaustiveness")
    ax.set_ylabel("Metric")
    ax.set_title(f"PBP2a DUD-E enrichment vs exhaustiveness "
                 f"({assessment['classification']})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "enrichment_saturation.png"), dpi=300)
    plt.close(fig)
    log.info(f"  Plot: {os.path.join(OUT, 'enrichment_saturation.png')}")


if __name__ == "__main__":
    main()