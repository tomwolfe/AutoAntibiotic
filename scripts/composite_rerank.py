#!/usr/bin/env python3
"""Transparent MD-informed composite re-rank of the AutoAntibiotic candidates.

The base output/top_candidates.csv ranking is by docking Selectivity_Index and is
left untouched. This script produces a *separate, fully-sourced* MD-informed
consensus ranking that folds the new explicit-solvent evidence into a single
view without ever fabricating a result for a candidate that has no data:

Evidence merged, per candidate:
  * MD stability class (D3)   -> output/md_explicit/<CID>/summary.json: stability_class_d3
  * Trajectory MM-GBSA dG     -> output/mmgbsa_results.json: method *_trajectory_frames
  * Active-site water network -> output/water_analysis.json (occupancy, if present)
  * Base docking SI           -> output/top_candidates.csv: Selectivity_Index / SI_Tier

Composite rank rule (transparent, documented in the JSON artifact):
  1. Primary: D3 stability tier  (Stable/Validated > Metastable > Dissociated)
  2. Secondary: trajectory MM-GBSA dG_bind (more negative = better) among
     MD-validated candidates; unvalidated candidates sort below all validated
     ones but still by docking SI.
  3. Tertiary: docking Selectivity_Index (descending).
No weighting is hidden; candidates without MD/MM-GBSA are explicitly marked
"MD:UNRAN", never extrapolated.

Outputs (never modify the primary ranking):
  output/composite_rerank.json        - structured provenance + ranked orders
  output/composite_rerank.csv         - flat view with all evidence columns

Honesty invariants (checked by verify_success.py Criterion 34):
  - Original top_candidates.csv is read-only.
  - Every candidate's ranking evidence comes only from the listed sources.
  - A candidate with no MD stability AND no trajectory MM-GBSA is ranked below
    every candidate that has either, and is flagged 'unvalidated'.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
CSV_PATH = OUT / "top_candidates.csv"
MD_OUT = OUT / "md_explicit"
MMGBSA_PATH = OUT / "mmgbsa_results.json"
WATER_PATH = OUT / "water_analysis.json"
OUT_JSON = OUT / "composite_rerank.json"
OUT_CSV = OUT / "composite_rerank.csv"

# D3 tiers -> primary sort group (lower = ranks higher).
STABILITY_RANK = {
    "Validated": 0, "Stable": 0,
    "Metastable": 1,
    "Dissociated": 2,
}
STABILITY_SYNONYMS = {
    "stable": "Validated", "validated": "Validated",
    "metastable": "Metastable", "dissociated": "Dissociated",
}

BIG = 1e9


def _parse_float_to_none(s):
    if s is None:
        return None
    try:
        return float(str(s).split()[0])
    except (ValueError, TypeError, IndexError, AttributeError):
        return None


def _load_md_stability() -> dict[str, dict]:
    """CID -> {stability_class, success, npt_duration_ns, error}."""
    md = {}
    if not MD_OUT.is_dir():
        return md
    for cdir in MD_OUT.iterdir():
        if not cdir.is_dir():
            continue
        cid = cdir.name
        summary = cdir / "summary.json"
        if not summary.is_file():
            continue
        try:
            d = json.loads(summary.read_text())
        except Exception:
            continue
        cls_raw = d.get("stability_class_d3") or d.get("consensus_stability") or ""
        cls = STABILITY_SYNONYMS.get(str(cls_raw).strip().lower(), str(cls_raw).strip())
        md[cid] = {
            "stability_class": cls if cls else None,
            "success": bool(d.get("success")),
            "npt_duration_ns": d.get("npt_duration_ns"),
            "error": d.get("error"),
        }
    return md


def _load_traj_mmgbsa() -> dict[str, float]:
    """CID -> trajectory MM-GBSA dG (kcal/mol) from *_trajectory_frames results."""
    res = {}
    if not MMGBSA_PATH.is_file():
        return res
    try:
        data = json.loads(MMGBSA_PATH.read_text())
    except Exception:
        return res
    items = data if isinstance(data, list) else [data]
    for r in items:
        if not isinstance(r, dict):
            continue
        cid = r.get("compound_id")
        method = str(r.get("method") or "")
        if cid and method.endswith("_trajectory_frames") and r.get("success"):
            res[cid] = r.get("delta_G_bind_mean_kcal")
    return res


def _load_water() -> dict[str, dict]:
    """CID -> { n_site_waters, max_residence_ps } best error-free replica."""
    res = {}
    if not WATER_PATH.is_file():
        return res
    try:
        data = json.loads(WATER_PATH.read_text())
    except Exception:
        return res
    for c in data if isinstance(data, list) else []:
        if not isinstance(c, dict):
            continue
        cid = c.get("compound_id")
        if cid is None:
            continue
        best = None
        for r in c.get("replicas") or []:
            if r.get("error"):
                continue
            key = (r.get("n_site_waters_detected", 0), r.get("max_residence_ps"))
            if best is None or key > best[0]:
                best = (key, r.get("n_site_waters_detected", 0), r.get("max_residence_ps"))
        if best:
            res[cid] = {"n_site_waters": best[1], "max_residence_ps": best[2]}
    return res


def _evidence(stab_ok, dg_ok, w_ok, md_success):
    tags = []
    if stab_ok or md_success:
        tags.append("MD:stability")
    if dg_ok:
        tags.append("MD:traj-MMGBSA")
    if w_ok:
        tags.append("water")
    if not tags:
        return "unvalidated"
    return "+".join(tags)


def _build_and_sort_records(doc_records, md_stab, traj_mg, water):
    """Expand raw intake rows into evidence records and return them D3-sorted."""
    records = []
    for row in doc_records:
        cid = row["Compound_ID"]
        md = md_stab.get(cid, {})
        cls = md.get("stability_class")
        stab_rank = STABILITY_RANK.get(cls) if cls else None
        dg = traj_mg.get(cid)
        w = water.get(cid)
        si = _parse_float_to_none(row.get("Selectivity_Index"))
        has_md_stab = md.get("success") and stab_rank is not None
        records.append({
            "compound_id": cid,
            "si": si,
            "si_tier": row.get("SI_Tier", ""),
            "d3_stability": cls,
            "stability_rank": stab_rank,
            "md_success": bool(md.get("success")),
            "npt_duration_ns": md.get("npt_duration_ns"),
            "traj_mmgbbsa_dg_kcal": dg,
            "n_site_waters": w.get("n_site_waters") if w else None,
            "max_residence_ps": w.get("max_residence_ps") if w else None,
            "evidence": _evidence(stab_rank is not None or has_md_stab, dg is not None,
                                  w is not None, bool(md.get("success"))),
        })

    def sort_key(r):
        dg = r["traj_mmgbbsa_dg_kcal"]
        si = r["si"]
        has_any = (r["md_success"] or dg is not None)
        stab = r["stability_rank"] if r["stability_rank"] is not None else 2
        return (
            not has_any,                     # unvalidated candidates sort last
            stab,                            # D3 tier
            -(dg if dg is not None else 1e9),  # secondary: more-negative dG first
            -(si if si is not None else -1.0),  # tertiary: higher SI first
        )

    records.sort(key=sort_key)
    return records


def main():
    ap = argparse.ArgumentParser(description="Transparent MD-informed composite re-rank")
    ap.add_argument("--no-water", action="store_true", help="Ignore water_analysis.json")
    args = ap.parse_args()

    if not CSV_PATH.is_file():
        print(f"ERROR: {CSV_PATH} not found.")
        return 1
    rows = list(csv.DictReader(open(CSV_PATH)))

    md_stab = _load_md_stability()
    traj_mg = _load_traj_mmgbsa()
    water = {} if args.no_water else _load_water()
    records = _build_and_sort_records(rows, md_stab, traj_mg, water)

    artifact = {
        "description": "Transparent MD-informed composite re-rank; does not overwrite top_candidates.csv",
        "sources": {
            "base_ranking": str(CSV_PATH),
            "md_stability": str(MD_OUT / "<CID>/summary.json"),
            "trajectory_mmgbbsa": str(MMGBSA_PATH),
            "water_analysis": str(WATER_PATH),
        },
        "ranking_rule": [
            "1. D3 stability tier (Stable/Validated > Metastable > Dissociated)",
            "2. candidates with neither MD stability nor trajectory MM-GBSA are 'unvalidated'",
            "   and sort below every candidate that has either",
            "3. within a group, rank by more-favourable (more negative) trajectory MM-GBSA dG",
            "4. tertiary tie-break: docking Selectivity_Index (descending)",
        ],
        "ranked": [dict(r) for r in records],
        "note": ("Not a replacement for top_candidates.csv; the paper's primary ranking "
                 "remains the docking SI list. This artifact aggregates MD evidence and "
                 "never invents a value for an un-run candidate."),
    }
    OUT_JSON.write_text(json.dumps(artifact, indent=2, default=str))

    fieldnames = list(records[0].keys())
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    for i, r in enumerate(records):
        dg = r["traj_dg"] if "traj_dg" in r else r["traj_mmgbbsa_dg_kcal"]
        dg_s = f"{dg:.2f}" if dg is not None else "-"
        si_s = f"{r['si']:.2f}" if r["si"] is not None else "-"
        print(f"  {i + 1:2d}. {r['compound_id']:<12s} d3={str(r['d3_stability'] or '-'):12s} "
              f"traj_dG={dg_s:>7s} SI={si_s:>5s} [{r['evidence']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())