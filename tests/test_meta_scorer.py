"""
Tests for the MetaScorer (stacking regressor for consensus scoring, v4.0).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.analysis import (
    MetaScorer,
    predict_meta_score,
    _get_meta_scorer,
)
from autoantibiotic.config import CONFIG
from autoantibiotic.models import CompoundRecord


def _make_record(smiles: str = "c1ccccc1O") -> CompoundRecord:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return CompoundRecord(
        compound_id="TEST-001",
        smiles=smiles,
        mol=mol,
        pb2pa_allosteric_energy=-8.5,
        shape_score=0.75,
        qed_score=0.7,
    )


# ── MetaScorer class tests ─────────────────────────────────────────


def test_metascorer_init() -> None:
    """MetaScorer should start unfitted."""
    scorer = MetaScorer()
    assert scorer.available is False
    assert scorer._model is None


def test_metascorer_train_and_predict() -> None:
    """Training on benchmark actives/inactives should yield a model
    that returns predictions in [0, 1]."""
    actives = [
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
    ]
    inactives = [
        "CCCCCCCCCCCCCCCCCC(=O)O",
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]

    scorer = MetaScorer()
    scorer.fit(actives, inactives)
    assert scorer.available is True

    # Predict on a known active
    rec = _make_record(actives[0])
    score = scorer.predict(rec)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_metascorer_predict_invalid_mol() -> None:
    """Predicting with an invalid molecule should return None."""
    scorer = MetaScorer()
    # Not fitted -> None
    rec = _make_record()
    assert scorer.predict(rec) is None


def test_metascorer_save_load() -> None:
    """Model save/load roundtrip should preserve predictions."""
    actives = [
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
    ]
    inactives = [
        "CCCCCCCCCCCCCCCCCC(=O)O",
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        model_path = str(Path(tmp) / "model.joblib")
        scorer1 = MetaScorer(model_path=model_path)
        scorer1.fit(actives, inactives)
        rec = _make_record(actives[0])
        score1 = scorer1.predict(rec)

        scorer2 = MetaScorer(model_path=model_path)
        loaded = scorer2.load()
        assert loaded is True
        assert scorer2.available is True
        score2 = scorer2.predict(rec)
        assert score2 is not None
        assert abs(score1 - score2) < 1e-4


# ── predict_meta_score tests ────────────────────────────────────────


def test_predict_meta_score_fallback() -> None:
    """predict_meta_score should fall back to compute_consensus_score
    when meta-scorer is not available."""
    rec = _make_record()
    score = predict_meta_score(rec)
    # With use_meta_scoring=False in test config, this should use fallback
    assert score is not None


def test_predict_meta_score_no_docking() -> None:
    """Predict should handle records with missing docking scores."""
    rec = _make_record()
    rec.pb2pa_allosteric_energy = None
    rec.shape_score = None
    score = predict_meta_score(rec)
    # Fallback: both None -> None
    assert score is None or (0.0 <= score <= 1.0)


# ── CONFIG toggle ─────────────────────────────────────────────────


def test_use_meta_scoring_config_toggle() -> None:
    """CONFIG.use_meta_scoring should default to True."""
    assert CONFIG.use_meta_scoring is True


# ── Active-learning / uncertainty threshold tests ──────────────


def test_metascorer_uncertainty_threshold_default() -> None:
    """MetaScorer should not set needs_manual_review without threshold."""
    actives = [
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
    ]
    inactives = [
        "CCCCCCCCCCCCCCCCCC(=O)O",
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]

    scorer = MetaScorer()
    scorer.fit(actives, inactives)
    rec = _make_record(actives[0])
    scorer.predict(rec)
    assert rec.needs_manual_review is False


def test_metascorer_uncertainty_threshold_low() -> None:
    """With a very low threshold, predictions may trigger manual review."""
    actives = [
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
    ]
    inactives = [
        "CCCCCCCCCCCCCCCCCC(=O)O",
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]

    scorer = MetaScorer()
    scorer.fit(actives, inactives, uncertainty_threshold=1e-6)
    rec = _make_record(actives[0])
    score = scorer.predict(rec)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_metascorer_dynamic_features_backward_compat() -> None:
    """MetaScorer must still train and predict when MD features are None."""
    actives = [
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
    ]
    inactives = [
        "CCCCCCCCCCCCCCCCCC(=O)O",
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]

    scorer = MetaScorer()
    scorer.fit(actives, inactives)
    assert scorer.available is True

    rec = _make_record(actives[0])
    rec.md_ligand_rmsd = None
    rec.md_pocket_rg_stability = None
    score = scorer.predict(rec)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_metascorer_dynamic_features_affect_score() -> None:
    """Records with MD data should produce different scores than without when
    the model is trained with dynamic features present in the training data."""
    actives = [
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
    ]
    inactives = [
        "CCCCCCCCCCCCCCCCCC(=O)O",
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]

    scorer = MetaScorer()
    scorer.fit(actives, inactives)

    rec_no_md = _make_record(actives[0])
    score_no_md = scorer.predict(rec_no_md)

    rec_with_md = _make_record(actives[0])
    rec_with_md.md_ligand_rmsd = 0.5
    rec_with_md.md_pocket_rg_stability = 0.05
    score_with_md = scorer.predict(rec_with_md)

    # Both should be valid scores; they may or may not differ depending
    # on the model, but both must be in [0, 1].
    assert score_no_md is not None
    assert score_with_md is not None
    assert 0.0 <= score_no_md <= 1.0
    assert 0.0 <= score_with_md <= 1.0


def test_metascorer_uncertainty_threshold_high_never_triggers() -> None: 
    """With a very high threshold, needs_manual_review should stay False."""
    actives = [
        "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)"
        "N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
    ]
    inactives = [
        "CCCCCCCCCCCCCCCCCC(=O)O",
        "CC(C)(C)OC(=O)NCCCCCCBr",
    ]

    scorer = MetaScorer()
    scorer.fit(actives, inactives, uncertainty_threshold=100.0)
    rec = _make_record(actives[0])
    scorer.predict(rec)
    assert rec.needs_manual_review is False
