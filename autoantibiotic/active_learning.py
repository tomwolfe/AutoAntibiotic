from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem

from .config import CONFIG
from .io_utils import log


def _parse_active_learning_csv(csv_path: str) -> Tuple[List[str], List[float]]:
    """Parse a CSV file with {smiles, ic50} columns.

    Returns a tuple of (smiles_list, ic50_values).
    Raises ValueError if required columns are missing or values are invalid.
    """
    smiles_list: List[str] = []
    ic50_values: List[float] = []

    if not os.path.exists(csv_path):
        log.warning(f"  Active learning CSV not found: {csv_path}")
        return smiles_list, ic50_values

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            # Validate column headers
            if reader.fieldnames is None:
                log.warning("  Active learning CSV has no headers.")
                return smiles_list, ic50_values

            has_smiles = "smiles" in {c.strip().lower() for c in reader.fieldnames}
            has_ic50 = "ic50" in {c.strip().lower() for c in reader.fieldnames}

            if not has_smiles or not has_ic50:
                log.warning(
                    "  Active learning CSV must contain 'smiles' and 'ic50' columns."
                )
                return smiles_list, ic50_values

            for row_num, row in enumerate(reader, start=2):
                try:
                    smi = row.get("smiles", "").strip()
                    ic50_str = row.get("ic50", "").strip()
                    if not smi or not ic50_str:
                        continue
                    ic50_val = float(ic50_str)
                    if ic50_val <= 0:
                        continue
                    pIC50 = -np.log10(ic50_val)
                    smiles_list.append(smi)
                    ic50_values.append(pIC50)
                except (ValueError, TypeError, KeyError):
                    log.debug(f"  Skipping invalid row {row_num} in active learning CSV.")
                    continue
    except Exception as exc:
        log.warning(f"  Failed to parse active learning CSV: {exc}")

    return smiles_list, ic50_values


def retrain_meta_scorer(new_data_path: str, model_path: Optional[str] = None) -> bool:
    """Retrain the MetaScorer with new active-learning data from a CSV file.

    Loads a CSV containing {smiles, ic50} columns, converts IC50 values
    to pIC50, and calls :meth:`MetaScorer.retrain_with_new_data` to
    append the new data to the existing training set and refit.

    Args:
        new_data_path: Path to the CSV file with {smiles, ic50} columns.
        model_path: Optional path to save the retrained model.
            Defaults to ``CONFIG.meta_scorer_model_path``.

    Returns:
        True if retraining succeeded, False otherwise.

    Examples
    --------
    >>> # Assuming a CSV file exists with 'smiles' and 'ic50' columns:
    >>> retrain_meta_scorer("output/new_active_data.csv")
    True
    """
    from .ml_scoring.meta_scorer import MetaScorer

    if not os.path.exists(new_data_path):
        log.warning(f"  Active learning CSV not found: {new_data_path}")
        return False

    smiles_list, pIC50_values = _parse_active_learning_csv(new_data_path)

    if len(smiles_list) < 2:
        log.warning(
            f"  Active learning data too small ({len(smiles_list)} entries). "
            "Need at least 2 valid entries for retraining."
        )
        return False

    log.info(
        f"  Active learning: loading {len(smiles_list)} new data points for retraining."
    )

    # Load or create MetaScorer
    scorer = MetaScorer(model_path=model_path or CONFIG.meta_scorer_model_path)
    if not scorer.load():
        log.warning("  MetaScorer model not available for retraining. Training from scratch.")

    # Get existing training data from the scorer
    existing_actives = list(scorer._training_actives) if hasattr(scorer, "_training_actives") else []
    existing_inactives = list(scorer._training_inactives) if hasattr(scorer, "_training_inactives") else []

    # Determine which are actives vs inactives based on pIC50 threshold
    # pIC50 > 5.0 (~10 µM) is considered active
    active_threshold_pIC50 = 5.0
    new_actives = [smi for smi, pval in zip(smiles_list, pIC50_values) if pval > active_threshold_pIC50]
    new_inactives = [smi for smi, pval in zip(smiles_list, pIC50_values) if pval <= active_threshold_pIC50]

    log.info(
        f"  Active learning: {len(new_actives)} actives / {len(new_inactives)} inactives."
    )

    try:
        scorer.retrain_with_new_data(new_actives, new_inactives)
        log.info("  Active learning: MetaScorer retrained successfully.")
        return True
    except Exception as exc:
        log.warning(f"  Active learning retraining failed: {exc}")
        return False
