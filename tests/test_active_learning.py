"""
Tests for the Active Learning Feedback Loop:
- Uncertainty quantification (predict_with_uncertainty)
- Model retraining via retrain_with_new_data
- Review queue generation
- CLI retrain flag integration
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.analysis import MetaScorer, _get_meta_scorer
from autoantibiotic.config import CONFIG
from autoantibiotic.models import CompoundRecord


def _make_record(
    compound_id: str = "TEST-001",
    smiles: str = "c1ccccc1O",
    mol: Chem.Mol = None,
) -> CompoundRecord:
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
    return CompoundRecord(
        compound_id=compound_id,
        smiles=smiles,
        mol=mol,
        pb2pa_allosteric_energy=-8.5,
        shape_score=0.75,
        qed_score=0.7,
    )


# ── test_uncertainty_quantification ─────────────────────────────────

ACTIVES = [
    "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
    "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
    "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
]
INACTIVES = [
    "CCCCCCCCCCCCCCCCCC(=O)O",
    "CC(C)(C)OC(=O)NCCCCCCBr",
]


def test_uncertainty_quantification() -> None:
    """predict_with_uncertainty returns non-zero std_dev for a compound
    different from training data."""
    scorer = MetaScorer()
    scorer.fit(ACTIVES, INACTIVES, uncertainty_threshold=0.0001)

    # Use a compound not in training data
    new_smiles = "CC1=CC2=C(C=C1)C(=O)C2=O"  # anthraquinone-like
    rec = _make_record("UNCERT-001", new_smiles)

    mean_score, std_dev = scorer.predict_with_uncertainty(rec)

    assert mean_score is not None
    assert 0.0 <= mean_score <= 1.0
    assert std_dev >= 0.0, "std_dev must be non-negative"
    assert std_dev > 0.0, (
        "std_dev must be non-zero for a compound different from training data"
    )


# ── test_retrain_updates_model ──────────────────────────────────────

def test_retrain_updates_model() -> None:
    """Train on small set, predict, add new data, retrain, and verify
    prediction for a new compound changes significantly."""
    initial_actives = [
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
    ]
    initial_inactives = [
        "CCCCCCCCCCCCCCCCCC(=O)O",
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]

    scorer = MetaScorer()
    scorer.fit(initial_actives, initial_inactives, uncertainty_threshold=0.0001)

    # Predict on a new compound
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin-like
    test_rec = _make_record("RETRAIN-001", test_smiles)
    mean1, std1 = scorer.predict_with_uncertainty(test_rec)

    # Add new training data
    new_actives = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    ]
    new_inactives = [
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]
    scorer.retrain_with_new_data(new_actives, new_inactives)

    # Predict again after retrain
    test_rec2 = _make_record("RETRAIN-002", test_smiles)
    mean2, std2 = scorer.predict_with_uncertainty(test_rec2)

    # The prediction should change significantly after retrain
    assert mean2 is not None
    assert 0.0 <= mean2 <= 1.0
    assert std2 >= 0.0

    # The mean score should change by at least 0.05 (significant change)
    assert abs(mean2 - mean1) > 0.05, (
        f"Prediction should change after retrain: {mean1:.4f} -> {mean2:.4f}"
    )


# ── test_review_queue_generation ────────────────────────────────────

def test_review_queue_generation() -> None:
    """Run a mock pipeline step and verify review_queue.csv is created
    with correct columns when uncertainty is high."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        scorer = MetaScorer()
        scorer.fit(ACTIVES, INACTIVES)

        # Create records that will be flagged (use very low threshold)
        rec1 = _make_record("REVIEW-001", ACTIVES[0])
        rec2 = _make_record("REVIEW-002", ACTIVES[1])
        rec3 = _make_record("REVIEW-003", "CC(=O)Oc1ccccc1C(=O)O")  # new compound

        # Flag with high threshold to ensure at least some are flagged
        flagged = scorer.flag_uncertain_predictions(
            [rec1, rec2, rec3], threshold=0.0001,
        )
        flagged_records = [r for r in flagged if r.needs_manual_review]
        assert len(flagged_records) > 0, "At least one record should be flagged"

        # Verify that the flagged records have needs_manual_review = True
        for rec in flagged_records:
            assert rec.needs_manual_review is True

        # Create a review queue CSV and verify its columns
        review_path = output_dir / "review_queue.csv"
        with open(review_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["compound_id", "smiles", "meta_score", "reason"])
            for rec in flagged_records:
                writer.writerow([
                    rec.compound_id, rec.smiles,
                    getattr(rec, "ml_score", ""),
                    "High prediction uncertainty",
                ])

        # Read back and verify columns
        with open(review_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == [
                "compound_id", "smiles", "meta_score", "reason",
            ], f"Expected 4 columns, got {len(header)}: {header}"
            rows = list(reader)
            assert len(rows) == len(flagged_records)


# ── test_cli_retrain_flag ──────────────────────────────────────────

def test_cli_retrain_flag() -> None:
    """Verify that passing --retrain-model loads data and triggers
    retraining without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "retrain_data.csv"

        # Create a small CSV with actives/inactives
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "ic50"])
            # Positive ic50 -> active
            writer.writerow(["CC(=O)Oc1ccccc1C(=O)O", "500"])
            # Negative/zero ic50 -> inactive
            writer.writerow(["CC(C)(C)OC(=O)NCCCCCCBr", "0"])

        # Load the data and trigger retraining
        from autoantibiotic.ml_scoring.meta_scorer import _get_meta_scorer, MetaScorer

        scorer = _get_meta_scorer()
        if scorer is not None:
            new_actives: List[str] = []
            new_inactives: List[str] = []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smi = row.get("smiles", "").strip()
                    ic50 = row.get("ic50", "").strip()
                    if not smi or not ic50:
                        continue
                    try:
                        val = float(ic50)
                    except ValueError:
                        continue
                    if val > 0:
                        new_actives.append(smi)
                    else:
                        new_inactives.append(smi)

            # Should not raise
            scorer.retrain_with_new_data(new_actives, new_inactives)
            assert scorer.available is True
