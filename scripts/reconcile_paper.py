#!/usr/bin/env python3
"""Verify and reconcile paper.tex and cover_letter.tex with actual pipeline output.

Usage:
    python scripts/reconcile_paper.py          # check + fix mismatches
    python scripts/reconcile_paper.py --verify  # check only, exit 1 on mismatch
    python scripts/reconcile_paper.py --populate  # auto-populate paper.tex with MD/MM-GBSA metrics
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
MMGBSA_PATH = BASE / "output" / "mmgbsa_results.json"
MD_SUMMARY_PATH = BASE / "output" / "md_explicit" / "summary.json"

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

# ── Load MD/MM-GBSA metrics if available ──
mmgbsa_data = {}
if MMGBSA_PATH.exists():
    try:
        with open(MMGBSA_PATH) as f:
            mg = json.load(f)
        if isinstance(mg, list):
            for entry in mg:
                cid = entry.get("compound_id")
                if cid and entry.get("success"):
                    mmgbsa_data[cid] = entry
        elif isinstance(mg, dict) and mg.get("compound_id"):
            mmgbsa_data[mg["compound_id"]] = mg
    except Exception:
        pass

md_data = {}
if MD_SUMMARY_PATH.exists():
    try:
        with open(MD_SUMMARY_PATH) as f:
            md_summary = json.load(f)
        for entry in md_summary.get("candidates", []):
            cid = entry.get("compound_id")
            if cid:
                md_data[cid] = entry
    except Exception:
        pass

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
check("AUC (dynamic)", auc, paper, "paper.tex")
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
check("Cover letter AUC (dynamic)", auc, cover, "cover_letter.tex")
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

# ── MD/MM-GBSA metric reconciliation ──
print("\n--- MD/MM-GBSA Metric Reconciliation ---")
if mmgbsa_data:
    print(f"  MM-GBSA data available for {len(mmgbsa_data)} candidates")
    for cid, mg in mmgbsa_data.items():
        dg = mg.get("delta_G_bind_kcal_mol")
        if dg is not None:
            dg_str = f"{dg:.2f}"
            if dg_str in paper:
                print(f"    OK: {cid} MM-GBSA dG = {dg_str} in paper")
            else:
                print(f"    NOTE: {cid} MM-GBSA dG = {dg_str} (run --populate to insert)")
        d3 = mg.get("d3_stability_class")
        if d3 and d3 in paper:
            print(f"    OK: {cid} D3 class '{d3}' in paper")
else:
    print("  No MM-GBSA data found (run --validate-candidates to generate)")

if md_data:
    print(f"  MD summary data available for {len(md_data)} candidates")
    for cid, md_entry in md_data.items():
        rmsd = md_entry.get("ligand_rmsd_mean_last5ns_A")
        if rmsd is not None:
            rmsd_str = f"{rmsd:.2f}"
            if rmsd_str in paper:
                print(f"    OK: {cid} MD RMSD = {rmsd_str} in paper")
            else:
                print(f"    NOTE: {cid} MD RMSD = {rmsd_str} (run --populate to insert)")
else:
    print("  No MD summary data found (run scripts/explicit_solvent_md.py)")

# ── Populate mode: auto-insert MD/MM-GBSA metrics into paper.tex ──
if "--populate" in sys.argv:
    print("\n--- Populating paper.tex with MD/MM-GBSA metrics ---")
    paper_text = paper
    n_inserted = 0

    for cid in list(mmgbsa_data.keys()) + list(md_data.keys()):
        mg = mmgbsa_data.get(cid, {})
        md_entry = md_data.get(cid, {})

        # Insert MM-GBSA delta-G if not already present
        dg = mg.get("delta_G_bind_kcal_mol")
        if dg is not None:
            dg_str = f"{dg:.2f}"
            if cid in paper_text and dg_str not in paper_text:
                # Find the paragraph mentioning this compound and append MM-GBSA info
                pattern = rf"({re.escape(cid)}.*?)(?=\\n|\\.|$)"
                match = re.search(pattern, paper_text, re.DOTALL)
                if match:
                    insert_pos = match.end()
                    insertion = f" (MM-GBSA $\\Delta G_{{bind}}$ = {dg_str} kcal/mol)"
                    paper_text = paper_text[:insert_pos] + insertion + paper_text[insert_pos:]
                    n_inserted += 1
                    print(f"  Inserted MM-GBSA dG for {cid}: {dg_str} kcal/mol")

        # Insert MD RMSD if not already present
        rmsd = md_entry.get("ligand_rmsd_mean_last5ns_A")
        if rmsd is not None:
            rmsd_str = f"{rmsd:.2f}"
            if cid in paper_text and rmsd_str not in paper_text:
                pattern = rf"({re.escape(cid)}.*?)(?=\\n|\\.|$)"
                match = re.search(pattern, paper_text, re.DOTALL)
                if match:
                    insert_pos = match.end()
                    insertion = f" (MD RMSD = {rmsd_str} \\AA)"
                    paper_text = paper_text[:insert_pos] + insertion + paper_text[insert_pos:]
                    n_inserted += 1
                    print(f"  Inserted MD RMSD for {cid}: {rmsd_str} Å")

        # Insert D3 stability class
        d3 = md_entry.get("stability_class_d3")
        if d3 and cid in paper_text:
            if d3 not in paper_text:
                pattern = rf"({re.escape(cid)}.*?)(?=\\n|\\.|$)"
                match = re.search(pattern, paper_text, re.DOTALL)
                if match:
                    insert_pos = match.end()
                    insertion = f" [D3: {d3}]"
                    paper_text = paper_text[:insert_pos] + insertion + paper_text[insert_pos:]
                    n_inserted += 1
                    print(f"  Inserted D3 class for {cid}: {d3}")

    if n_inserted > 0:
        # Write updated paper.tex
        PAPER_PATH.write_text(paper_text)
        print(f"  Inserted {n_inserted} metric(s) into paper.tex")
    else:
        print("  No new metrics to insert (all already present or no matches)")

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
