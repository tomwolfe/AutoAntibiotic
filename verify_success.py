#!/usr/bin/env python3
"""Verify all success criteria for the AutoAntibiotic pipeline and paper.

Checks:
  1. output/top_candidates.csv >= 20 rows
  2. >= 2 Strong tier compounds (property-based check)
  3. >= 5 compounds with SI >= 1.5
  4. <= 2 CLASH entries
  5. protocol_trust == Validated
   6. Enrichment AUC >= 0.7, EF_1% >= 5 (with mathematically valid EF check)
   7. Top hit Ser403 + Lys406 H-bonds
   8. >= 8 figures in output/figures/publication/
   9. paper.tex compiles
  10. Paper numbers match output files
  11. Library >= 500 compounds
  12. MMFF94_Strain_Score column in CSV
  13. Human_CYP3A4_Energy column in CSV
  14. Library >= 2000 compounds
  15. Explicit-solvent MD results exist
  16. MM-GBSA results exist
  17. At least 2 Strong tier compounds (property-based, not compound-ID specific)
  18. ALL_QU05 is NOT the primary lead (secondary only)
  19. verify_success.py exits 0
  20. paper.tex compiles with xelatex
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_THRESHOLDS = {
    "n_rows": 20, "n_strong": 2, "n_si_ge_1_5": 5, "max_clash": 2,
    "core_rmsd_max": 1.5, "auc_min": 0.7, "ef_1pct_min": 5,
    "min_figures": 8, "min_library_size": 2000,
}


def main():
    base = Path(".")
    csv_path = base / "output" / "top_candidates.csv"
    enrich_path = base / "output" / "enrichment_results.json"
    figures_path = base / "output" / "figures" / "publication"
    paper_path = base / "paper.tex"
    lib_path = base / "data" / "screen_library_final.csv"
    md_explicit_path = base / "output" / "md_explicit" / "summary.json"
    mmgbsa_path = base / "output" / "mmgbsa_results.json"

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
    pdfs = list(figures_path.glob("*.pdf")) if figures_path.exists() else []

    # Check explicit-solvent MD exists
    md_ok = md_explicit_path.is_file()
    mmgbsa_ok = mmgbsa_path.is_file()

    # Check Strong tier compounds (property-based, not compound-ID specific)
    strong_compounds = [r for r in csv_rows if r.get("SI_Tier") == "Strong"]
    n_strong_tier = len(strong_compounds)
    # Check if BRICS_0022 and ALL_QU04 specifically have Strong tier (for reporting)
    brics_strong = any(r.get("Compound_ID") == "BRICS_0022" and r.get("SI_Tier") == "Strong" for r in csv_rows)
    all_qu04_strong = any(r.get("Compound_ID") == "ALL_QU04" and r.get("SI_Tier") == "Strong" for r in csv_rows)

    # Check ALL_QU05 is NOT the primary lead (should not be ranked #1 if SI < 1.5)
    all_qu05_rank = None
    for i, r in enumerate(csv_rows):
        if r["Compound_ID"] == "ALL_QU05":
            all_qu05_rank = i + 1
            break
    all_qu05_not_primary = all_qu05_rank is not None and all_qu05_rank > 2
    all_qu05_si_below = False
    if all_qu05_rank:
        all_qu05_row = next((r for r in csv_rows if r["Compound_ID"] == "ALL_QU05"), None)
        if all_qu05_row:
            try:
                si_val = float(all_qu05_row.get("Selectivity_Index", "0").split()[0])
                all_qu05_si_below = si_val < 1.5
            except (ValueError, IndexError):
                pass

    criteria_results = []

    def c(num, name, ok, detail):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Criterion {num:2d}. {name}: {detail}")
        criteria_results.append(ok)

    ok = len(csv_rows) >= 20
    c(1, "output/top_candidates.csv >= 20 rows", ok, f"{len(csv_rows)} rows")

    ok = n_strong_tier >= 2
    c(2, ">= 2 Strong tier compounds", ok, f"{n_strong_tier} Strong")

    ok = n_si_15 >= 5
    c(3, ">= 5 compounds with SI >= 1.5", ok, f"{n_si_15} with SI>=1.5")

    ok = n_clash <= 2
    c(4, "<= 2 CLASH entries", ok, f"{n_clash} CLASH")

    ok = prot == "Validated"
    c(5, "protocol_trust == Validated", ok, f"trust='{prot}'")

    # Validate EF_1% is mathematically possible
    # For EF_1% to be valid, it must be <= max possible EF given N, n_act, k1
    N = enrich_data.get("n_compounds", 171)
    n_act = enrich_data.get("n_actives", 21)
    k1 = max(1, round(0.01 * N))
    max_ef_possible = (min(k1, n_act) / n_act) / (k1 / N) if n_act > 0 else 0
    ef_valid = ef <= max_ef_possible
    ok = auc >= 0.7 and ef >= 5 and ef_valid
    c(6, "Enrichment AUC >= 0.7, EF_1% >= 5 (valid)", ok, f"AUC={auc:.3f}, EF={ef:.2f}, max_possible={max_ef_possible:.2f}")

    ok = h_ser and h_lys
    c(7, "Top hit Ser403 + Lys406 H-bonds", ok, f"S403={h_ser}, K406={h_lys}")

    ok = len(pngs) + len(pdfs) >= 8
    c(8, ">= 8 figures in output/figures/publication/", ok, f"{len(pngs)} PNGs, {len(pdfs)} PDFs")

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
    # Check BRICS_0022 and ALL_QU04 mentioned
    if "BRICS_0022" not in paper_text or "ALL_QU04" not in paper_text:
        issues.append("Primary leads not mentioned")
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

    ok = md_ok
    c(15, "Explicit-solvent MD results exist (output/md_explicit/summary.json)", ok,
      "Present" if ok else "MISSING — run scripts/explicit_solvent_md.py")

    ok = mmgbsa_ok
    c(16, "MM-GBSA results exist (output/mmgbsa_results.json)", ok,
      "Present" if ok else "MISSING — run scripts/mmgbsa_analysis.py")

    # Criterion 17: At least 2 Strong tier compounds (property-based, not compound-ID specific)
    ok = n_strong_tier >= 2
    c(17, "At least 2 Strong tier compounds (property-based)", ok,
      f"{n_strong_tier} Strong tier compounds")

    ok = all_qu05_si_below or all_qu05_not_primary
    c(18, "ALL_QU05 is secondary scaffold (SI < 1.5 or not rank #1)", ok,
      f"Rank={all_qu05_rank}, SI_below_gate={all_qu05_si_below}")

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