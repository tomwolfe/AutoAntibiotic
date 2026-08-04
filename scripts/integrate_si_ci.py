#!/usr/bin/env python3
"""
Integrate the 3-seed Selectivity-Index confidence interval into the main report.

The single-report ``Selectivity_Index`` is one Vina run; AutoDock Vina's
stochastic search carries ~±2 kcal/mol noise, so a single value cannot separate
"Strong" (> 2.0) from "Promising" (> 1.5) when candidates are within that noise
band. To make that distinction statistically meaningful, a 3-seed re-docking
(exhaustiveness=32, seeds 0/1/2) against PBP2a + trypsin + CES1 was run (its
per-seed results are kept in output/top_candidates_ci.csv), and each
Selectivity Index is reported as:

    Selectivity_Index_CI = "mean ± std [low–high]"

This script merges that precomputed column into the canonical
output/top_candidates.csv (and its JSON mirror) keyed by Compound_ID, so the
confidence interval is surfaced alongside the single-report value instead of
living in a separate on-off file.

Usage:
    python scripts/integrate_si_ci.py

Outputs:
    output/top_candidates.csv / .json with the Selectivity_Index_CI column set.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("integrate_si_ci")

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "output" / "top_candidates.csv"
CI_PATH = REPO / "output" / "top_candidates_ci.csv"


def load_ci_map(path: Path) -> dict:
    """Map Compound_ID -> Selectivity_Index_CI string."""
    if not path.is_file():
        return {}
    with open(path, newline="") as f:
        return {r["Compound_ID"]: r.get("Selectivity_Index_CI", "N/A")
                for r in csv.DictReader(f)}


def main() -> int:
    if not CSV_PATH.is_file():
        log.error(f"  top_candidates.csv not found: {CSV_PATH}")
        return 1
    ci_map = load_ci_map(CI_PATH)
    log.info(f"  Loaded {len(ci_map)} Selectivity_Index_CI entries from {CI_PATH.name}")

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    if "Selectivity_Index_CI" not in fieldnames:
        fieldnames.append("Selectivity_Index_CI")

    used = 0
    for row in rows:
        row["Selectivity_Index_CI"] = ci_map.get(row["Compound_ID"], "N/A")
        if row["Selectivity_Index_CI"].strip() not in ("", "N/A"):
            used += 1

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    json_path = CSV_PATH.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    log.info(f"  Integrated Selectivity_Index_CI for {used}/{len(rows)} candidates")
    log.info(f"  Updated {CSV_PATH} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())