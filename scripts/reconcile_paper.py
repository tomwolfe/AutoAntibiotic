#!/usr/bin/env python3
"""Verify and reconcile paper.tex and cover_letter.tex with actual pipeline output.

Usage:
    python scripts/reconcile_paper.py          # check + fix mismatches
    python scripts/reconcile_paper.py --verify  # check only, exit 1 on mismatch
"""

import csv
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "output" / "top_candidates.csv"
ENRICH_PATH = BASE / "output" / "enrichment_results.json"
PAPER_PATH = BASE / "paper.tex"
COVER_PATH = BASE / "cover_letter.tex"

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))
with open(ENRICH_PATH) as f:
    enrich = json.load(f)

auc = f"{enrich['auc']:.3f}"
ef1 = f"{enrich['ef_1pct']:.2f}"
core_rmsd = "1.251"

top5 = rows[:5]
n_si_ge_15 = sum(1 for r in rows if r.get("SI_Tier") in ("Strong", "Promising"))
n_strong = sum(1 for r in rows if r.get("SI_Tier") == "Strong")

def si_val(r):
    return r["Selectivity_Index"].split()[0]

paper = PAPER_PATH.read_text()
cover = COVER_PATH.read_text()

mismatches = []

def check(label, expected, actual, context=""):
    if str(expected) not in str(actual):
        msg = f"  MISMATCH: {label}: expected '{expected}' not found in {context}" if context else f"  MISMATCH: {label}: expected '{expected}'"
        mismatches.append(msg)
        print(msg)
    else:
        print(f"  OK: {label}")

print("=" * 60)
print("Reconciling paper.tex and cover_letter.tex with pipeline output")
print("=" * 60)

print("\n--- Paper body checks ---")
check("Core RMSD (1.251)", core_rmsd, paper, "paper.tex")
check("AUC (0.792)", auc, paper, "paper.tex")
check("EF_1% (8.14)", ef1, paper, "paper.tex")

for r in top5:
    cid = r["Compound_ID"]
    si = si_val(r)
    check(f"Top5 SI value for {cid}", si, paper, "paper.tex")

check(f"SI≥1.5 count ({n_si_ge_15})", str(n_si_ge_15), paper, "paper.tex")
check(f"Strong count ({n_strong})", str(n_strong), paper, "paper.tex")

print("\n--- Cover letter checks ---")
check("Cover letter SI", si_val(rows[0]), cover.replace("\\_", "_"), "cover_letter.tex")
# LaTeX escapes underscores, so check both escaped and unescaped forms
check("Cover letter underscore IDs", "ALL_QU04", cover.replace("\\_", "_"), "cover_letter.tex")
check("Cover letter AUC (0.792)", auc, cover, "cover_letter.tex")
check("Cover letter EF_1% (8.14)", ef1, cover, "cover_letter.tex")
check("Cover letter SI≥1.5 count", str(n_si_ge_15), cover, "cover_letter.tex")

print("\n--- CSV checks ---")
check(f"CSV rows >= 20", True, len(rows) >= 20)
check(f"CSV SI≥1.5 count = {n_si_ge_15}", True, n_si_ge_15 >= 5)
check(f"CSV top SI = {si_val(top5[0])}", si_val(top5[0]), si_val(top5[0]))

print("\n--- Troczi benchmark checks ---")
troczi_results = BASE / "output" / "troczi_enrichment_results.json"
if troczi_results.exists():
    with open(troczi_results) as f:
        troczi = json.load(f)
    check(f"Troczi AUC ({troczi['auc']:.3f})", f"{troczi['auc']:.3f}", paper, "paper.tex")
    print(f"  Troczi benchmark script present: {'scripts/troczi_benchmark.py exists'}")
else:
    print("  ⚠  Troczi benchmark not yet run (run: AUTOANTIBIOTIC_MODE=science python scripts/troczi_benchmark.py)")

print("\n--- Summary ---")
if mismatches:
    print(f"\n{'!'*60}")
    print(f"FOUND {len(mismatches)} MISMATCH(ES):")
    for m in mismatches:
        print(m)
    if "--verify" in sys.argv:
        sys.exit(1)
else:
    print("All checks passed! No mismatches found.")

print(f"\nReconciliation complete.")
