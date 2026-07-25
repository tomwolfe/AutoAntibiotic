#!/usr/bin/env python3
"""Verify all success criteria for the AutoAntibiotic pipeline."""

import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

REQUIRED_THRESHOLDS = {
    "n_rows": 20,
    "n_strong": 1,
    "n_si_ge_1_5": 3,
    "max_clash": 2,
    "core_rmsd_max": 1.5,
    "auc_min": 0.7,
    "ef_1pct_min": 5,
    "min_figures": 4,
    "min_library_size": 500,
}


def check_criterion(level: int, name: str, result: bool, detail: str) -> None:
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] Criterion {level}.{name}: {detail}")


def verify_csv(path: Path) -> Tuple[bool, List[str]]:
    details = []
    if not path.exists():
        return False, ["CSV file not found"]
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) < REQUIRED_THRESHOLDS["n_rows"]:
        details.append(f"Only {len(rows)} rows (need >= {REQUIRED_THRESHOLDS['n_rows']})")
        return False, details
    details.append(f"{len(rows)} rows (need >= {REQUIRED_THRESHOLDS['n_rows']})")

    strong = [r for r in rows if r.get("SI_Tier") == "Strong"]
    if len(strong) < REQUIRED_THRESHOLDS["n_strong"]:
        details.append(f"Only {len(strong)} Strong (need >= {REQUIRED_THRESHOLDS['n_strong']})")
        return False, details
    details.append(f"{len(strong)} Strong tier compounds")

    si_passing = [r for r in rows if r.get("SI_Tier") in ("Strong", "Promising")]
    if len(si_passing) < REQUIRED_THRESHOLDS["n_si_ge_1_5"]:
        details.append(f"Only {len(si_passing)} with SI >= 1.5 (need >= {REQUIRED_THRESHOLDS['n_si_ge_1_5']})")
        return False, details
    details.append(f"{len(si_passing)} compounds with SI >= 1.5")

    clash = [r for r in rows if "CLASH" in r.get("Human_CES1_Energy", "")]
    if len(clash) > REQUIRED_THRESHOLDS["max_clash"]:
        details.append(f"{len(clash)} CLASH entries (max {REQUIRED_THRESHOLDS['max_clash']})")
        return False, details
    details.append(f"{len(clash)} CLASH entries (max {REQUIRED_THRESHOLDS['max_clash']})")

    top_hit = rows[0]
    h_ser = top_hit.get("H_Bond_Ser403", "").strip() == "True"
    h_lys = top_hit.get("H_Bond_Lys406", "").strip() == "True"
    if not (h_ser and h_lys):
        details.append(f"Top hit missing H-bonds: Ser403={h_ser}, Lys406={h_lys}")
        return False, details
    details.append(f"Top hit {top_hit['Compound_ID']}: Ser403 HB={h_ser}, Lys406 HB={h_lys}")

    protocol = top_hit.get("protocol_trust", "")
    if protocol != "Validated":
        details.append(f"Protocol trust is '{protocol}' (need 'Validated')")
        return False, details
    details.append(f"Protocol trust: {protocol}")

    # Check MMGBSA_Score column
    if "MMGBSA_Score" not in rows[0]:
        details.append("MMGBSA_Score column not found in CSV")
        return False, details
    details.append("MMGBSA_Score column present")

    # Check Human_CYP3A4_Energy column
    if "Human_CYP3A4_Energy" not in rows[0]:
        details.append("Human_CYP3A4_Energy column not found in CSV")
        return False, details
    details.append("Human_CYP3A4_Energy column present")

    return True, details


def verify_enrichment(path: Path) -> Tuple[bool, List[str]]:
    details = []
    if not path.exists():
        return False, ["enrichment_results.json not found"]
    with open(path) as f:
        data = json.load(f)
    auc = data.get("auc", 0)
    ef = data.get("ef_1pct", 0)
    if auc < REQUIRED_THRESHOLDS["auc_min"]:
        details.append(f"AUC={auc:.3f} (need >= {REQUIRED_THRESHOLDS['auc_min']})")
        return False, details
    details.append(f"AUC={auc:.3f} (need >= {REQUIRED_THRESHOLDS['auc_min']})")
    if ef < REQUIRED_THRESHOLDS["ef_1pct_min"]:
        details.append(f"EF_1%={ef:.2f} (need >= {REQUIRED_THRESHOLDS['ef_1pct_min']})")
        return False, details
    details.append(f"EF_1%={ef:.2f} (need >= {REQUIRED_THRESHOLDS['ef_1pct_min']})")
    return True, details


def verify_figures(path: Path) -> Tuple[bool, List[str]]:
    details = []
    pngs = list(path.glob("*.png"))
    if len(pngs) < REQUIRED_THRESHOLDS["min_figures"]:
        details.append(f"Only {len(pngs)} PNGs (need >= {REQUIRED_THRESHOLDS['min_figures']})")
        return False, details
    details.append(f"{len(pngs)} PNG figures")
    return True, details


def verify_library(lib_path: Path) -> Tuple[bool, List[str]]:
    details = []
    if not lib_path.exists():
        return False, ["screen_library_final.csv not found"]
    with open(lib_path) as f:
        n_lines = sum(1 for _ in f)
    n_compounds = n_lines - 1
    if n_compounds < REQUIRED_THRESHOLDS["min_library_size"]:
        details.append(f"Only {n_compounds} compounds (need >= {REQUIRED_THRESHOLDS['min_library_size']})")
        return False, details
    details.append(f"{n_compounds} compounds (need >= {REQUIRED_THRESHOLDS['min_library_size']})")
    return True, details


def verify_paper_compiles(paper_path: Path) -> Tuple[bool, List[str]]:
    import subprocess
    details = []
    result = subprocess.run(
        ["/opt/homebrew/bin/xelatex", "-interaction=nonstopmode", paper_path.name],
        cwd=paper_path.parent,
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        errors = [l for l in result.stdout.split("\n") if "Error" in l][:5]
        details.append(f"xelatex failed: {'; '.join(errors)}")
        return False, details
    pdf = paper_path.with_suffix(".pdf")
    if pdf.exists() and pdf.stat().st_size > 0:
        details.append(f"PDF generated: {pdf.name} ({pdf.stat().st_size} bytes)")
    return True, details


def verify_paper_numbers(paper_path: Path, csv_path: Path, enrich_path: Path) -> Tuple[bool, List[str]]:
    details = []
    with open(paper_path) as f:
        text = f.read()

    issues = []

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    top_hit = rows[0]
    si = top_hit.get("Selectivity_Index", "")
    if si:
        cnt = text.count(si)
        if cnt < 2:
            issues.append(f"SI value {si} not found in paper text")

    with open(enrich_path) as f:
        enrich = json.load(f)
    auc_str = f"{enrich['auc']:.3f}"
    if auc_str not in text:
        issues.append(f"AUC {auc_str} not found in paper")
    ef_str = f"{enrich['ef_1pct']:.2f}"
    if ef_str not in text:
        issues.append(f"EF_1% {ef_str} not found in paper")

    n_clash = sum(1 for r in rows if "CLASH" in r.get("Human_CES1_Energy", ""))
    if n_clash == 0:
        if "No candidates were lost to steric clashes" not in text:
            issues.append("CSV has 0 CLASH entries but paper doesn't state this clearly")

    details.append(f"Paper text checks: {len(issues)} potential mismatches")
    for iss in issues:
        details.append(f"  Mismatch: {iss}")
    return len(issues) == 0, details


def main():
    base = Path(".")
    csv_path = base / "output" / "top_candidates.csv"
    enrich_path = base / "output" / "enrichment_results.json"
    figures_path = base / "output" / "figures"
    paper_path = base / "paper.tex"
    lib_path = base / "data" / "screen_library_final.csv"

    print("=" * 60)
    print("AutoAntibiotic Pipeline — Success Criteria Verification")
    print("=" * 60)

    all_pass = True

    # Criterion 1: CSV has >= 20 rows
    print("\n1. output/top_candidates.csv has >= 20 rows")
    ok, details = verify_csv(csv_path)
    all_pass &= ok
    for d in details:
        print(f"    {d}")

    # Criterion 2: >= 1 compound with SI_Tier == "Strong"
    # Criterion 3: >= 3 compounds with SI >= 1.5
    # Criterion 4: <= 2 CLASH entries
    # Criterion 5: protocol_trust == "Validated"
    # Criterion 7: Top hit has H_Bond_Ser403 and H_Bond_Lys406
    # (all covered in verify_csv)

    # Criterion 6: enrichment AUC >= 0.7 and EF_1% >= 5
    print("\n6. Enrichment: AUC >= 0.7, EF_1% >= 5")
    ok, details = verify_enrichment(enrich_path)
    all_pass &= ok
    for d in details:
        print(f"    {d}")

    # Criterion 8: >= 4 figure PNGs
    print("\n8. output/figures/ has >= 4 .png files")
    if not figures_path.exists():
        print("    FAIL: figures directory not found")
        all_pass = False
    else:
        ok, details = verify_figures(figures_path)
        all_pass &= ok
        for d in details:
            print(f"    {d}")

    # Criterion 9: paper.tex compiles
    print("\n9. paper.tex compiles cleanly")
    ok, details = verify_paper_compiles(paper_path)
    all_pass &= ok
    for d in details:
        print(f"    {d}")

    # Criterion 10: Paper numbers match output files
    print("\n10. Paper numbers match output files")
    ok, details = verify_paper_numbers(paper_path, csv_path, enrich_path)
    all_pass &= ok
    for d in details:
        print(f"    {d}")

    # Criterion 11: Library has >= 500 compounds
    print("\n11. Library size >= 500 compounds")
    ok, details = verify_library(lib_path)
    all_pass &= ok
    for d in details:
        print(f"    {d}")

    # Criterion 12: MMGBSA_Score column exists (verified in verify_csv)

    # Criterion 13: Human_CYP3A4_Energy column exists (verified in verify_csv)

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CRITERIA PASSED")
    else:
        print("SOME CRITERIA FAILED — see details above")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
