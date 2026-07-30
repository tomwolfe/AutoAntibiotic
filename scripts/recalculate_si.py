#!/usr/bin/env python3
"""
Recalculate Selectivity Index from existing docking results with the minimum
binding energy threshold applied.

Reads output/top_candidates.csv, applies MIN_BINDING_ENERGY = 1.0 threshold
to off-target energies, recomputes SI, and writes updated CSV/JSON.

Usage:
    python scripts/recalculate_si.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import (
    MIN_BINDING_ENERGY,
    SELECTIVITY_PANEL_TARGETS,
    SELECTIVITY_INDEX_THRESHOLD,
    SI_STRONG_THRESHOLD,
    SI_PROMISING_THRESHOLD,
    CEFTAROLINE_CONTROL_E,
    OUTPUT_DIR,
    CSV_REPORT,
)


def si_tier(si):
    if si is None:
        return "N/A"
    if si >= SI_STRONG_THRESHOLD:
        return "Strong"
    if si >= SI_PROMISING_THRESHOLD:
        return "Promising"
    return "Weak"


def recalculate_si_row(row):
    pb2pa_best_str = row.get("PBP2a_Best_Energy", "N/A")
    trypsin_str = row.get("Human_Trypsin_Energy", "N/A")
    ces1_str = row.get("Human_CES1_Energy", "N/A")

    pb2pa_best = float(pb2pa_best_str) if pb2pa_best_str not in ("N/A", "") else None
    trypsin_e = float(trypsin_str) if trypsin_str not in ("N/A", "") else None
    ces1_e = float(ces1_str) if ces1_str not in ("N/A", "") else None

    si_vs_ceft = abs(pb2pa_best) / CEFTAROLINE_CONTROL_E if pb2pa_best is not None else None

    sel_panel = {t.lower() for t in SELECTIVITY_PANEL_TARGETS}

    panel_valid = []
    for label, energy in [("trypsin", trypsin_e), ("ces1", ces1_e)]:
        if label not in sel_panel:
            continue
        if energy is not None and abs(energy) >= MIN_BINDING_ENERGY and energy <= -0.01:
            panel_valid.append(energy)

    selectivity_index = None
    selectivity_confidence = "None"
    si_provisional = None

    if len(panel_valid) >= 2:
        si = abs(pb2pa_best) / np.mean([abs(e) for e in panel_valid]) if pb2pa_best is not None else None
        selectivity_index = si
        selectivity_confidence = "High"
    elif len(panel_valid) == 1 and pb2pa_best is not None:
        si_prov = abs(pb2pa_best) / abs(panel_valid[0])
        si_provisional = si_prov
        selectivity_confidence = "Low"

    return selectivity_index, selectivity_confidence, si_provisional, si_vs_ceft


def _parse_si(si_str):
    if si_str in ("N/A", "N/A (low-conf)", ""):
        return None
    try:
        return float(si_str.split()[0])
    except (ValueError, IndexError):
        return None


def main():
    csv_path = Path(CSV_REPORT)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    updated = []
    for row in rows:
        si, conf, si_prov, si_vs_ceft = recalculate_si_row(row)

        row["Selectivity_Index"] = (
            f"{si:.2f}" if si is not None else "N/A"
        ) + ("" if conf == "High" else " (low-conf)")

        row["Selectivity_Confidence"] = "Unassessed" if conf == "None" else conf
        row["Selectivity_Index_TwoTarget"] = (
            f"{si:.2f}" if si is not None else "N/A"
        )
        row["SI_vs_Ceftaroline"] = (
            f"{si_vs_ceft:.2f}" if si_vs_ceft is not None else "N/A"
        )
        row["SI_Tier"] = si_tier(si)
        row["Passes_Selectivity_Gate"] = str(
            si is not None and si >= SELECTIVITY_INDEX_THRESHOLD
        )
        row["SI_Provisional"] = (
            f"{si_prov:.2f}" if si_prov is not None else "N/A"
        )
        row["HIGH_TOXICITY_RISK"] = str(
            si is not None and si < 1.0
        )

        updated.append(row)

    # Sort: passing (SI >= 1.5) compounds first by SI descending,
    # then below-gate compounds by PBP2a best energy (most negative first).
    passing = [r for r in updated if r.get("Selectivity_Index", "N/A") != "N/A (low-conf)"
               and r.get("Selectivity_Index", "N/A") != "N/A"]
    passing = [r for r in passing if _parse_si(r.get("Selectivity_Index", "N/A")) is not None
               and _parse_si(r.get("Selectivity_Index", "N/A")) >= SI_PROMISING_THRESHOLD]
    passing.sort(key=lambda r: _parse_si(r.get("Selectivity_Index", "N/A")) or 0.0, reverse=True)
    below = [r for r in updated if r not in passing]
    below.sort(key=lambda r: float(r.get("PBP2a_Best_Energy", "0")) if r.get("PBP2a_Best_Energy", "N/A") not in ("N/A", "") else float("inf"))
    updated = passing + below

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(updated)
    print(f"Updated {csv_path}")

    json_path = csv_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"Updated {json_path}")

    print("\nRecalculation complete.")
    print(f"  MIN_BINDING_ENERGY = {MIN_BINDING_ENERGY} kcal/mol")
    n_strong = sum(1 for r in updated if r.get("SI_Tier") == "Strong")
    n_promising = sum(1 for r in updated if r.get("SI_Tier") == "Promising")
    print(f"  Strong: {n_strong}, Promising: {n_promising}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
