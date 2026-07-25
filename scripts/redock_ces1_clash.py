"""
Re-dock CLASH compounds against CES1 with corrected grid-box parameters.

This script addresses the CES1 docking failure where compounds were rejected
because the grid box was too large (max_size=25.0, padding=4.0) and the
centroid check was too permissive (max_dist=22.0). After the fix, the box
is max_size=18.0, padding=2.0 and the centroid check is max_dist=11.0.

Usage:
    python scripts/redock_ces1_clash.py
"""
import json
import os
import sys
import logging
import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import (
    CES1_CATALYTIC_RESIDUES, SELECTIVITY_BOX_SIZE,
    SELECTIVITY_PANEL_TARGETS, SI_PROMISING_THRESHOLD,
    SI_STRONG_THRESHOLD, CEFTAROLINE_CONTROL_E,
)
from discovery_pipeline import _auto_box_size, _offtarget_dock_with_centroid_check
from utils.docking import dock_compound
from utils.structure_prep import compute_residue_centroid
from utils.library_gen import CompoundRecord

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("AutoAntibiotic")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
WORK_DIR = os.path.join(OUTPUT_DIR, "workdir")
CSV_PATH = os.path.join(OUTPUT_DIR, "top_candidates.csv")
JSON_PATH = os.path.join(OUTPUT_DIR, "top_candidates.json")
CES1_PDB = os.path.join(WORK_DIR, "CES1_clean.pdb")
CES1_PDBQT = os.path.join(WORK_DIR, "CES1_clean.pdbqt")


def compute_si(pb2pa_energy, human_energies):
    """Compute mechanism-restricted selectivity index."""
    if pb2pa_energy is None:
        return None
    valid = [e for e in human_energies if e is not None and e <= 0.0]
    if not valid:
        return None
    return abs(pb2pa_energy) / np.mean(valid)


def main():
    log.info("=== Re-docking CLASH compounds against CES1 (corrected params) ===")

    # Load current results
    df = pd.read_csv(CSV_PATH)
    with open(JSON_PATH) as f:
        json_records = json.load(f)

    # Identify CLASH compounds
    clash_mask = df["Human_CES1_Energy"].astype(str).str.contains("CLASH", na=False)
    clash_df = df[clash_mask].copy()
    log.info(f"Found {len(clash_df)} CLASH compounds to re-dock:")
    for _, row in clash_df.iterrows():
        log.info(f"  {row['Compound_ID']}")

    if len(clash_df) == 0:
        log.info("No CLASH compounds found. Exiting.")
        return

    # Compute CES1 center and corrected box
    ces1_center = compute_residue_centroid(CES1_PDB, CES1_CATALYTIC_RESIDUES, use_ca=False)
    log.info(f"CES1 center: {ces1_center}")

    ces1_box = _auto_box_size(
        CES1_PDB, ces1_center, SELECTIVITY_BOX_SIZE,
        min_size=15.0, max_size=18.0, padding=2.0,
        site_residues=CES1_CATALYTIC_RESIDUES,
    )
    log.info(f"CES1 box (corrected): {ces1_box}")

    # Create centroid-check dock function
    ces1_dock_func = _offtarget_dock_with_centroid_check(ces1_center, max_dist=11.0)

    # Re-dock each CLASH compound
    updates = {}
    for _, row in clash_df.iterrows():
        cid = row["Compound_ID"]
        smiles = row["SMILES"]
        log.info(f"\nRe-docking {cid}...")

        rec = CompoundRecord(compound_id=cid, smiles=smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning(f"  Could not parse SMILES for {cid}")
            updates[cid] = None
            continue

        try:
            energy = ces1_dock_func(
                rec, CES1_PDBQT, ces1_center, ces1_box,
                WORK_DIR, "ces1", timeout=300,
            )
            if energy is not None:
                log.info(f"  {cid}: CES1 energy = {energy:.2f} kcal/mol")
            else:
                log.info(f"  {cid}: Still CLASH (no pose)")
            updates[cid] = energy
        except Exception as exc:
            log.warning(f"  {cid}: Docking failed: {exc}")
            updates[cid] = None

    # Update CSV
    for idx, row in clash_df.iterrows():
        cid = row["Compound_ID"]
        new_energy = updates.get(cid)
        if new_energy is not None:
            df.at[idx, "Human_CES1_Energy"] = f"{new_energy:.2f}"
            df.at[idx, "Human_OffTarget_Max_Energy"] = f"{new_energy:.2f}"

            # Recompute SI
            pb2pa_best = row["PBP2a_Best_Energy"]
            tryp_energy = row["Human_Trypsin_Energy"]
            human_energies = []
            if isinstance(tryp_energy, str) and "CLASH" not in tryp_energy:
                try:
                    human_energies.append(float(tryp_energy))
                except ValueError:
                    pass
            human_energies.append(new_energy)

            si = compute_si(pb2pa_best, human_energies)
            if si is not None:
                df.at[idx, "Selectivity_Index"] = f"{si:.2f}"
                df.at[idx, "Selectivity_Index_TwoTarget"] = f"{si:.2f}"
                df.at[idx, "SI_Provisional"] = f"{si:.2f}"
                df.at[idx, "Selectivity_Confidence"] = "High"
                if si >= SI_STRONG_THRESHOLD:
                    df.at[idx, "SI_Tier"] = "Strong"
                elif si >= SI_PROMISING_THRESHOLD:
                    df.at[idx, "SI_Tier"] = "Promising"
                else:
                    df.at[idx, "SI_Tier"] = "Below gate"
                df.at[idx, "Passes_Selectivity_Gate"] = si >= SI_PROMISING_THRESHOLD
                df.at[idx, "SI_vs_Ceftaroline"] = f"{abs(pb2pa_best) / CEFTAROLINE_CONTROL_E:.2f}"

    # Save updated CSV
    df.to_csv(CSV_PATH, index=False)
    log.info(f"\nUpdated CSV saved to {CSV_PATH}")

    # Update JSON
    for rec in json_records:
        cid = rec.get("Compound_ID") or rec.get("compound_id")
        if cid in updates:
            new_energy = updates[cid]
            if new_energy is not None:
                rec["Human_CES1_Energy"] = f"{new_energy:.2f}"
                rec["Human_OffTarget_Max_Energy"] = f"{new_energy:.2f}"

                pb2pa_best = rec.get("PBP2a_Best_Energy")
                tryp_str = rec.get("Human_Trypsin_Energy", "")
                human_energies = []
                if isinstance(tryp_str, str) and "CLASH" not in tryp_str:
                    try:
                        human_energies.append(float(tryp_str))
                    except ValueError:
                        pass
                human_energies.append(new_energy)

                si = compute_si(pb2pa_best, human_energies)
                if si is not None:
                    rec["Selectivity_Index"] = f"{si:.2f}"
                    rec["Selectivity_Index_TwoTarget"] = f"{si:.2f}"
                    rec["SI_Provisional"] = f"{si:.2f}"
                    rec["Selectivity_Confidence"] = "High"
                    if si >= SI_STRONG_THRESHOLD:
                        rec["SI_Tier"] = "Strong"
                    elif si >= SI_PROMISING_THRESHOLD:
                        rec["SI_Tier"] = "Promising"
                    else:
                        rec["SI_Tier"] = "Below gate"
                    rec["Passes_Selectivity_Gate"] = si >= SI_PROMISING_THRESHOLD
                    rec["SI_vs_Ceftaroline"] = f"{abs(pb2pa_best) / CEFTAROLINE_CONTROL_E:.2f}"

    with open(JSON_PATH, "w") as f:
        json.dump(json_records, f, indent=2)
    log.info(f"Updated JSON saved to {JSON_PATH}")

    # Summary
    n_clash_remaining = sum(1 for v in updates.values() if v is None)
    n_resolved = sum(1 for v in updates.values() if v is not None)
    log.info(f"\n=== Summary ===")
    log.info(f"  Re-docked: {len(updates)}")
    log.info(f"  Resolved (got valid pose): {n_resolved}")
    log.info(f"  Still CLASH: {n_clash_remaining}")


if __name__ == "__main__":
    main()
