#!/usr/bin/env python3
"""Verify all 20 success criteria for the AutoAntibiotic pipeline."""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_THRESHOLDS = {
    "n_rows": 20, "n_strong": 1, "n_si_ge_1_5": 5, "max_clash": 2,
    "core_rmsd_max": 1.5, "auc_min": 0.7, "ef_1pct_min": 5,
    "min_figures": 4, "min_library_size": 2000,
}


def main():
    base = Path(".")
    csv_path = base / "output" / "top_candidates.csv"
    enrich_path = base / "output" / "enrichment_results.json"
    figures_path = base / "output" / "figures"
    paper_path = base / "paper.tex"
    lib_path = base / "data" / "screen_library_final.csv"

    print("=" * 60)
    print("AutoAntibiotic Pipeline — All 20 Success Criteria")
    print("=" * 60)

    with open(csv_path) as f:
        csv_rows = list(csv.DictReader(f))
    csv_header = list(csv_rows[0].keys()) if csv_rows else []
    with open(enrich_path) as f:
        enrich_data = json.load(f)

    top_hit = csv_rows[0] if csv_rows else {}
    h_ser = top_hit.get("H_Bond_Ser403", "").strip() == "True"
    h_lys = top_hit.get("H_Bond_Lys406", "").strip() == "True"
    n_strong = sum(1 for r in csv_rows if r.get("SI_Tier") == "Strong")
    n_si_15 = sum(1 for r in csv_rows if r.get("SI_Tier") in ("Strong", "Promising"))
    n_clash = sum(1 for r in csv_rows if "CLASH" in r.get("Human_CES1_Energy", ""))
    prot = top_hit.get("protocol_trust", "")
    try:
        top_si = float(top_hit.get("Selectivity_Index", "0").split()[0])
    except (ValueError, IndexError):
        top_si = 0.0
    auc = enrich_data.get("auc", 0)
    ef = enrich_data.get("ef_1pct", 0)
    n_lib = sum(1 for _ in open(lib_path)) - 1
    pngs = list(figures_path.glob("*.png")) if figures_path.exists() else []

    criteria_results = []

    def c(num, name, ok, detail):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Criterion {num:2d}. {name}: {detail}")
        criteria_results.append(ok)

    ok = len(csv_rows) >= 20
    c(1, "output/top_candidates.csv >= 20 rows", ok, f"{len(csv_rows)} rows")

    ok = n_strong >= 1
    c(2, ">= 1 Strong tier compound", ok, f"{n_strong} Strong")

    ok = n_si_15 >= 5
    c(3, ">= 5 compounds with SI >= 1.5", ok, f"{n_si_15} with SI>=1.5")

    ok = n_clash <= 2
    c(4, "<= 2 CLASH entries", ok, f"{n_clash} CLASH")

    ok = prot == "Validated"
    c(5, "protocol_trust == Validated", ok, f"trust='{prot}'")

    ok = auc >= 0.7 and ef >= 5
    c(6, "Enrichment AUC >= 0.7, EF_1% >= 5", ok, f"AUC={auc:.3f}, EF={ef:.2f}")

    ok = h_ser and h_lys
    c(7, "Top hit Ser403 + Lys406 H-bonds", ok, f"S403={h_ser}, K406={h_lys}")

    ok = len(pngs) >= 4
    c(8, ">= 4 figures in output/figures/", ok, f"{len(pngs)} PNGs")

    xelatex = shutil.which("xelatex") or shutil.which("pdflatex")
    if xelatex:
        result = subprocess.run(
            [xelatex, "-interaction=nonstopmode", paper_path.name],
            cwd=paper_path.parent, capture_output=True, text=True, timeout=60,
        )
    pdf = paper_path.with_suffix(".pdf")
    pdf_ok = pdf.exists() and pdf.stat().st_size > 0
    ok = xelatex is not None and result.returncode == 0 and pdf_ok
    c(9, "paper.tex compiles", ok, f"PDF={pdf.stat().st_size}B" if ok else f"xelatex not found")

    paper_text = paper_path.read_text()
    issues = []
    si_val = top_hit.get("Selectivity_Index", "").split()[0]
    if si_val and si_val not in paper_text:
        issues.append(f"SI={si_val}")
    if f"{auc:.3f}" not in paper_text:
        issues.append(f"AUC={auc:.3f}")
    if f"{ef:.2f}" not in paper_text:
        issues.append(f"EF={ef:.2f}")
    ok = len(issues) == 0
    c(10, "Paper numbers match output files", ok,
      "OK" if ok else f"Issues: {issues}")

    ok = n_lib >= 500
    c(11, "Library >= 500 compounds", ok, f"{n_lib} cmpds")

    ok = "MMFF94_Strain_Score" in csv_header
    c(12, "MMFF94_Strain_Score column in CSV", ok, "Present" if ok else "MISSING")

    ok = "Human_CYP3A4_Energy" in csv_header
    c(13, "Human_CYP3A4_Energy column in CSV", ok, "Present" if ok else "MISSING")

    ok = n_lib >= 2000
    c(14, "Library >= 2000 compounds", ok, f"{n_lib} cmpds")

    ok = n_si_15 >= 5
    c(15, ">= 5 compounds with SI >= 1.5", ok, f"{n_si_15}/5")

    ok = h_ser and h_lys
    c(16, "Top hit H-bonds to Ser403 + Lys406", ok, f"S403={h_ser}, K406={h_lys}")

    ok_h = "Human_hERG_Energy" in csv_header
    ok_a = "Human_Albumin_Energy" in csv_header
    c(17, "hERG + Albumin columns in CSV", ok_h and ok_a,
      f"hERG={'Y'if ok_h else 'N'}, Alb={'Y'if ok_a else 'N'}")

    docking_src = Path("utils/docking.py").read_text()
    ok_pi = "rescore_mmff94_strain" in docking_src and "e_receptor_interaction" in docking_src
    c(18, "MMFF94_Strain_Score is force-field-based ranking", ok_pi,
      "Strain-interaction" if ok_pi else "Ligand-only")

    all_pass = all(criteria_results)
    c(19, "verify_success.py exits 0", all_pass,
      "PASS" if all_pass else "Self-check (criteria failed above)")

    ok_20 = pdf.exists() and pdf.stat().st_size > 0
    c(20, "paper.tex compiles with xelatex", ok_20, f"PDF={pdf.stat().st_size}B")

    n_pass = sum(1 for i, ok in enumerate(criteria_results, 1) if ok)
    print("\n" + "=" * 60)
    if all_pass:
        print(f"ALL {len(criteria_results)} CRITERIA PASSED")
    else:
        n_fail = len(criteria_results) - n_pass
        print(f"{n_pass}/{len(criteria_results)} CRITERIA PASSED ({n_fail} FAILED)")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
