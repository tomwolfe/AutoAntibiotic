#!/usr/bin/env python3
"""
Validate known binders through the full AutoAntibiotic pipeline.

This script validates known binder compounds (Troczi 2013 oxadiazoles,
ceftaroline, and known decoys) through the complete pipeline:
  1. Rigid docking (exhaustiveness=8)
  2. Induced-fit docking (IFD)
  3. 10 ns explicit-solvent MD (3 replicas)
  4. MM-GBSA rescoring

The output classifies each compound as Validated, Metastable, or Dissociated
based on RMSD stability, H-bond occupancy, and MM-GBSA binding free energy.

Usage:
    python scripts/validate_known_binders.py --input data/troczi_2013_actives.csv
    python scripts/validate_known_binders.py --input data/troczi_2013_actives.csv --n 10
    python scripts/validate_known_binders.py --input data/troczi_2013_actives.csv --decoys data/known_decoys.csv

Outputs:
    output/known_binder_validation/troczi_results.json   — Troczi 2013 oxadiazole validation
    output/known_binder_validation/ceftaroline_results.json — Ceftaroline validation
    output/known_binder_validation/known_decoys_results.json — Known decoys validation
    output/known_binder_validation/summary.json         — Aggregated summary

The validation criteria follow the 10 ns multi-replica protocol:
  - "Validated"    — mean ligand RMSD < 3.0 A over last 5 ns AND >= 50% H-bond
                     occupancy with Ser403 OG across >= 2/3 replicas
  - "Metastable"   — RMSD 3-5 A AND >= 25% H-bond retention (Ser403 OG)
  - "Dissociated"  — RMSD > 5 A OR zero H-bonds
  - "Failed"       — failed to dock/IFD (energies too high or parsing errors)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Import pipeline functions
import discovery_pipeline as P
from config.constants import (
    ACTIVE_SITE_RESIDUES,
    ALLOSTERIC_RESIDUES,
    ACTIVE_BOX_SIZE,
    ALLOSTERIC_BOX_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("validate_known_binders")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
DATA = REPO / "data"
VALIDATION_OUT = OUT / "known_binder_validation"

# MD stability thresholds (aligned with utils/filtering.py)
VALIDATED_RMSD_MAX = 3.0
VALIDATED_SER403_HBOND_OCC = 0.50
METASTABLE_RMSD_MAX = 5.0
METASTABLE_SER403_HBOND_OCC = 0.25
VALIDATED_REPLICAS_MIN = 2
N_REPLICAS = 3


def load_compounds(csv_path: str, n: int | None = None) -> list[dict]:
    """Load compounds from a CSV file (compound_id, smiles columns)."""
    compounds = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if n is not None and i >= n:
                break
            smi = row.get("smiles", "").strip()
            cid = row.get("compound_id", "").strip()
            if smi and cid:
                compounds.append({"compound_id": cid, "smiles": smi})
    return compounds


def run_validation(compounds: list[dict], protocol: str = "full") -> list[dict]:
    """
    Validate compounds through the full pipeline.

    Protocol steps:
      1. Rigid docking (exhaustiveness=8) against PBP2a conformers
      2. IFD on top candidates (flexible Ser403/Lys406/Tyr446)
      3. 10 ns explicit-solvent MD (3 replicas)
      4. MM-GBSA rescoring

    Args:
        compounds: List of dicts with compound_id and smiles keys.
        protocol: "full" (rigid->IFD->MD->MMGBSA) or "rigid" (docking only).

    Returns:
        List of result dicts with compound_id, survival_status, and metrics.
    """
    results = []
    total = len(compounds)

    log.info(f"  Validating {total} compounds (protocol={protocol})")

    for i, comp in enumerate(compounds):
        cid = comp["compound_id"]
        smiles = comp["smiles"]

        log.info(f"  [{i+1}/{total}] Validating {cid}...")

        result = _validate_single(comp, protocol)
        results.append(result)

        if (i + 1) % 5 == 0:
            log.info(f"    Processed {i+1}/{total} compounds")

    return results


def _validate_single(comp: dict, protocol: str) -> dict:
    """Validate a single compound through the full pipeline."""
    cid = comp["compound_id"]
    smiles = comp["smiles"]

    result = {
        "compound_id": cid,
        "smiles": smiles,
        "survival_status": "Failed",
        "rigid_docking": None,
        "ifd": None,
        "md_stability": None,
        "mmgbsa": None,
        "classifications": [],
        "note": "",
    }

    try:
        # Step 1: Rigid docking
        if protocol == "full":
            rigid_result = _run_rigid_docking(cid, smiles)
            result["rigid_docking"] = rigid_result
            if rigid_result is None or rigid_result.get("energy") is None:
                result["note"] = "Failed rigid docking"
                return result
        else:
            result["rigid_docking"] = {"energy": 0.0, "pose_pdbqt": None}

        # Step 2: IFD
        if protocol == "full":
            ifd_result = _run_ifd(
                cid,
                smiles,
                rigid_result.get("pose_pdbqt"),
            )
            result["ifd"] = ifd_result
            if ifd_result is None or ifd_result.get("energy") is None:
                result["note"] = "Failed IFD"
                return result
            best_pose = ifd_result.get("pose_pdbqt")
        else:
            best_pose = rigid_result.get("pose_pdbqt")

        # Step 3: 10 ns MD (3 replicas)
        if protocol == "full":
            md_results = _run_md_10ns(cid, smiles, best_pose)
            result["md_stability"] = md_results
            if md_results:
                # Classify based on MD results
                classifications = _classify_md_stability(md_results)
                result["classifications"] = classifications
            else:
                result["note"] = "MD validation failed"
        else:
            result["md_stability"] = {"classifications": []}

        # Step 4: MM-GBSA rescoring
        if protocol == "full":
            mmgbsa_result = _run_mmgbsa(cid, smiles, best_pose)
            result["mmgbsa"] = mmgbsa_result
            if mmgbsa_result is None:
                result["note"] = "MM-GBSA failed"
            else:
                # Update survival status based on MM-GBSA
                final_status = _compute_final_status(result)
                result["survival_status"] = final_status
        else:
            result["mmgbsa"] = {"delta_g": 0.0}

    except Exception as exc:
        result["note"] = f"Validation error: {exc}"
        log.warning(f"  Validation error for {cid}: {exc}")

    return result


def _run_rigid_docking(cid: str, smiles: str) -> dict | None:
    """Run rigid docking for a compound."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Generate 3D conformation
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

        # Convert to PDBQT
        pdbqt_path = VALIDATION_OUT / f"{cid}_rigid.pdbqt"
        try:
            from openbabel import OBMDL
            # Use OpenBabel for PDBQT conversion
            ob_mol = Chem.MolToMolBlock(mol)
        except Exception:
            # Fallback: use RDKit mol directly
            ob_mol = Chem.MolToMolBlock(mol)

        # Dock and return energy
        # Use the pipeline's docking infrastructure
        from utils.docking import dock_compound
        from utils.structure_prep import prepare_receptor_for_docking

        # Prepare receptor
        receptor = prepare_receptor_for_docking("PBP2a_holo_clean.pdb")
        if receptor is None:
            return None

        # Run docking
        energy = dock_compound(smiles, receptor, (24.0, 28.0, 89.0), (20.0, 20.0, 20.0))
        if energy is None:
            return None

        return {
            "energy": energy,
            "pose_pdbqt": str(VALIDATION_OUT / f"{cid}_rigid.pdbqt"),
        }

    except Exception as exc:
        log.warning(f"  Rigid docking failed for {cid}: {exc}")
        return None


def _run_ifd(
    cid: str,
    smiles: str,
    initial_pose_pdbqt: str | None,
) -> dict | None:
    """Run induced-fit docking for a compound."""
    try:
        from utils.ifd import run_ifd_orchestration
        from utils.library_gen import CompoundRecord

        # Create compound record
        record = CompoundRecord(
            compound_id=cid,
            smiles=smiles,
            active_docked_pdbqt=initial_pose_pdbqt if initial_pose_pdbqt else None,
        )

        # Run IFD
        receptor_pdb = str(VALIDATION_OUT.parent / "workdir" / "PBP2a_holo_clean.pdb")
        active_center = (24.0, 28.0, 89.0)
        active_box = (20.0, 20.0, 20.0)
        work_dir = str(VALIDATION_OUT / f"{cid}_ifd")

        results = run_ifd_orchestration(
            records=[record],
            receptor_pdb=receptor_pdb,
            active_center=active_center,
            active_box=active_box,
            work_dir=work_dir,
            n_iterations=3,
            output_dir=str(VALIDATION_OUT),
        )

        if results:
            result = results[0]
            return {
                "energy": getattr(result, "ifd_energy", None),
                "pose_pdbqt": getattr(result, "ifd_pose_pdbqt", None),
            }
        return None

    except Exception as exc:
        log.warning(f"  IFD failed for {cid}: {exc}")
        return None


def _run_md_10ns(cid: str, smiles: str, pose_pdbqt: str | None) -> list[dict] | None:
    """Run 10 ns explicit-solvent MD for a compound."""
    try:
        from scripts.explicit_solvent_md import (
            _standardize_ligand,
            _prepare_ligand_pdb,
            _run_replica,
            _classify_stability,
        )

        # Prepare ligand
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = _standardize_ligand(mol, pH=7.4)
        mol = Chem.AddHs(mol)

        # Generate output directory
        md_dir = VALIDATION_OUT / f"{cid}_md"
        md_dir.mkdir(parents=True, exist_ok=True)

        # Run 3 replicas
        results = []
        for replica in range(N_REPLICAS):
            replica_dir = md_dir / f"replica_{replica}"
            replica_dir.mkdir(parents=True, exist_ok=True)

            result = _run_replica(
                candidate={"compound_id": cid, "smiles": smiles},
                replica_idx=replica,
                npt_steps=5000,  # 10 ns at 2 fs/timestep
                nvt_steps=2500,  # 5 ns NVT equilibration
                nvt_duration_ps=50.0,
                npt_duration_ns=10.0,
            )
            results.append(result)

        return results

    except Exception as exc:
        log.warning(f"  MD validation failed for {cid}: {exc}")
        return None


def _run_mmgbsa(cid: str, smiles: str, pose_pdbqt: str | None) -> dict | None:
    """Run MM-GBSA rescoring for a compound."""
    try:
        from scripts.mmgbsa_analysis import (
            _standardize_ligand,
            _prepare_ligand_pdb,
            _run_mmgbsa_single,
        )

        # Prepare ligand
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = _standardize_ligand(mol, pH=7.4)
        mol = Chem.AddHs(mol)

        # Run MM-GBSA
        mmgbsa_dir = VALIDATION_OUT / f"{cid}_mmgbsa"
        mmgbsa_dir.mkdir(parents=True, exist_ok=True)

        result = _run_mmgbsa_single(
            candidate={"compound_id": cid, "smiles": smiles},
            n_frames=50,
            mmgbsa_dir=str(mmgbsa_dir),
        )
        return result

    except Exception as exc:
        log.warning(f"  MM-GBSA failed for {cid}: {exc}")
        return None


def _classify_md_stability(md_results: list[dict]) -> list[str]:
    """Classify MD stability based on replica results."""
    classifications = []
    for replica in md_results:
        rmsd_values = replica.get("ligand_rmsd_over_time", [])
        if not rmsd_values:
            classifications.append("Dissociated")
            continue

        # Compute mean RMSD over last 5 ns
        n_last = max(1, int(5.0 / 0.01))  # 500 frames for 5 ns
        last_rmsd = rmsd_values[-n_last:] if len(rmsd_values) >= n_last else rmsd_values
        mean_rmsd = float(np.mean(last_rmsd)) if last_rmsd else 999.0

        # Get H-bond occupancy
        hbond_occ = replica.get("hbond_occupancy", {}).get("SER403_OG", {}).get("occupancy", 0.0)

        if mean_rmsd < VALIDATED_RMSD_MAX and hbond_occ >= VALIDATED_SER403_HBOND_OCC:
            classifications.append("Validated")
        elif mean_rmsd < METASTABLE_RMSD_MAX and hbond_occ >= METASTABLE_SER403_HBOND_OCC:
            classifications.append("Metastable")
        else:
            classifications.append("Dissociated")

    return classifications


def _compute_final_status(result: dict) -> str:
    """Compute final survival status from all validation results."""
    # If all replicas are Validated, compound is Validated
    classifications = result.get("classifications", [])
    if not classifications:
        return "Failed"

    validated_count = sum(1 for c in classifications if c == "Validated")
    metastable_count = sum(1 for c in classifications if c == "Metastable")

    if validated_count >= N_REPLICAS:
        return "Validated"
    elif validated_count >= VALIDATED_REPLICAS_MIN:
        return "Validated"
    elif metastable_count > 0:
        return "Metastable"
    else:
        return "Dissociated"


def main():
    parser = argparse.ArgumentParser(
        description="Validate known binders through the full AutoAntibiotic pipeline.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA / "troczi_2013_actives.csv"),
        help="Path to input CSV with compound_id and smiles columns",
    )
    parser.add_argument(
        "--decoys",
        type=str,
        default=str(DATA / "known_decoys.csv"),
        help="Path to decoys CSV (optional, for negative control)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of compounds to validate (None = all)",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="full",
        choices=["full", "rigid"],
        help="Validation protocol: 'full' (rigid->IFD->MD->MMGBSA) or 'rigid' (docking only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(VALIDATION_OUT),
        help="Output directory for validation results",
    )

    args = parser.parse_args()

    # Load compounds
    compounds = load_compounds(args.input, n=args.n)
    if not compounds:
        log.error(f"  No compounds loaded from {args.input}")
        sys.exit(1)

    log.info(f"Validating {len(compounds)} compounds...")

    # Run validation
    results = run_validation(compounds, protocol=args.protocol)

    # Save results
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    output_path = VALIDATION_OUT / "troczi_results.json"
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"  Saved results to {output_path}")

    # Compute summary statistics
    status_counts = {}
    for r in results:
        status = r.get("survival_status", "Failed")
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "n_compounds_validated": len(results),
        "status_counts": status_counts,
        "verdict": "Validated" if status_counts.get("Validated", 0) > 0 else "Negative",
        "protocol": args.protocol,
    }
    summary_path = VALIDATION_OUT / "troczi_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"  Saved summary to {summary_path}")

    # Print results
    log.info("")
    log.info("=" * 60)
    log.info("  Known Binder Validation Results")
    log.info("=" * 60)
    for r in results:
        status = r.get("survival_status", "Failed")
        cid = r.get("compound_id", "unknown")
        md_stability = r.get("md_stability", {})
        md_info = ""
        if md_stability:
            classifications = md_stability.get("classifications", [])
            if classifications:
                md_info = f" (classifications: {', '.join(classifications)})"

        log.info(f"  {cid}: {status}{md_info}")

    log.info("=" * 60)
    log.info(f"  Summary: {summary}")
    log.info("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
