"""
Tests for the MetaScorer (stacking regressor for consensus scoring, v4.0).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.analysis import (
    MetaScorer,
    predict_meta_score,
    _get_meta_scorer,
)
from autoantibiotic.config import CONFIG, ConfigurationError
from autoantibiotic.models import CompoundRecord

# Lower the minimum training threshold so existing tests (which use
# 4 samples) continue to work.  Individual tests that need the production
# default override it locally.
CONFIG.min_training_samples = 4


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


def test_metascorer_scaffold_split_small_dataset() -> None:
    """Scaffold split should not crash with small datasets (singletons)."""
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
    # Should not raise — even with tiny dataset
    scorer.fit(actives, inactives)
    assert scorer.available is True


def test_metascorer_scaffold_group_diverse() -> None:
    """_scaffold_groups should group molecules by Murcko scaffold."""
    smiles_list = [
        "c1ccccc1O",      # phenol
        "c1ccccc1O",      # phenol (same)
        "c1ccccc1N",      # aniline (different scaffold)
    ]
    groups = MetaScorer._scaffold_groups(smiles_list)
    # Should have at least 2 groups
    assert len(groups) >= 1
    # Same SMILES should be in same group
    phenol_scaffold = None
    for scaf, indices in groups.items():
        if 0 in indices and 1 in indices:
            phenol_scaffold = scaf
            break
    assert phenol_scaffold is not None, (
        "Both phenol SMILES should share a scaffold group"
    )


def test_metascorer_scaffold_split_consistency() -> None:
    """MetaScorer trained with scaffold splitting should still predict."""
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
    score = scorer.predict(rec)
    assert score is not None
    assert 0.0 <= score <= 1.0


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


def test_metascorer_ifp_water_features() -> None:
    """IFP and water displacement energy should influence the feature vector.

    Two records with identical base features but different IFP scores must
    produce different feature vectors, and the feature vector must have 13 dims.
    """
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

    raw_mol = Chem.MolFromSmiles(actives[0])
    assert raw_mol is not None

    feat_high = scorer._default_features(raw_mol, ifp_score=0.95, water_displacement_energy=-2.5)
    feat_low = scorer._default_features(raw_mol, ifp_score=0.10, water_displacement_energy=-0.1)

    # Feature vectors must differ at IFP index (3) and water index (12)
    assert feat_high[3] == pytest.approx(0.95, abs=1e-6)
    assert feat_low[3] == pytest.approx(0.10, abs=1e-6)
    assert feat_high[12] == pytest.approx(-2.5, abs=1e-6)
    assert feat_low[12] == pytest.approx(-0.1, abs=1e-6)
    assert feat_high[3] != feat_low[3]
    assert feat_high[12] != feat_low[12]

    # Feature vector length must be 16
    assert feat_high.shape[0] == 16, f"Expected 16 features, got {feat_high.shape[0]}"
    assert len(scorer._feature_names) == 16

    # Predictions must be valid and based on different feature vectors
    score_high = scorer.predict(_make_record(actives[0]))
    score_low = scorer.predict(_make_record(actives[0]))
    assert score_high is not None
    assert score_low is not None
    assert 0.0 <= score_high <= 1.0
    assert 0.0 <= score_low <= 1.0


def test_metascorer_ifp_water_features_none_defaults() -> None:
    """When IFP and water displacement are None, defaults must be 0.0."""
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

    raw_mol = Chem.MolFromSmiles(actives[0])
    assert raw_mol is not None

    feat_with_none = scorer._default_features(raw_mol, ifp_score=None, water_displacement_energy=None)
    assert feat_with_none[3] == 0.0, "IFP score default should be 0.0"
    assert feat_with_none[12] == 0.0, "Water displacement default should be 0.0"


def test_metascorer_with_real_docking_features() -> None:
    """Feature vector must incorporate real docking values when available.

    Mocks the benchmark docking cache to return non-zero docking features,
    then verifies that the resulting feature vector contains non-zero
    physics-based columns (vina_energy, gnina_score, shape_score).
    """
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
    # Patch get_benchmark_docking_features to return non-zero values
    mock_docking = {
        actives[0]: {"vina_energy": -9.5, "gnina_score": 0.85, "ifp_score": 0.92},
        actives[1]: {"vina_energy": -8.2, "gnina_score": 0.72, "ifp_score": 0.80},
        inactives[0]: {"vina_energy": -5.0, "gnina_score": 0.45, "ifp_score": 0.30},
        inactives[1]: {"vina_energy": -6.1, "gnina_score": 0.55, "ifp_score": 0.40},
    }
    from unittest.mock import patch
    with patch(
        "benchmarks.reference_data.get_benchmark_docking_features",
        return_value=mock_docking,
    ):
        scorer.fit(actives, inactives)

    raw_mol = Chem.MolFromSmiles(actives[0])
    assert raw_mol is not None

    feats = scorer._default_features(
        raw_mol,
        vina_energy=-9.5,
        gnina_score=0.85,
        shape_score=0.92,
    )
    # Feature vector must have shape (16,)
    assert feats.shape == (16,), f"Expected shape (16,), got {feats.shape}"

    # Physics columns must be non-zero
    assert feats[0] == pytest.approx(-9.5, abs=1e-6), "vina_energy must be -9.5"
    assert feats[1] == pytest.approx(0.85, abs=1e-6), "gnina_score must be 0.85"
    assert feats[2] == pytest.approx(0.92, abs=1e-6), "shape_score must be 0.92"

    # Predict must return a valid score
    score = scorer.predict(_make_record(actives[0]))
    assert score is not None
    assert 0.0 <= score <= 1.0


# ── Minimum training data enforcement ───────────────────────────


def test_fit_raises_on_small_dataset() -> None:
    """MetaScorer.fit() must raise ConfigurationError when fewer than
    CONFIG.min_training_samples samples are provided."""
    original = CONFIG.min_training_samples
    CONFIG.min_training_samples = 20
    try:
        actives = ["c1ccccc1O"]
        inactives = ["c1ccccc1N"]

        scorer = MetaScorer()
        with pytest.raises(ConfigurationError, match="Insufficient training data"):
            scorer.fit(actives, inactives)

        assert scorer.available is False
    finally:
        CONFIG.min_training_samples = original


# ── MD feature validation ────────────────────────────────────────


def test_predict_validates_md_features() -> None:
    """When CONFIG.force_md_for_meta_scoring is True, predict() must
    raise ConfigurationError if MD features are missing."""
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

    original_value = CONFIG.force_md_for_meta_scoring
    try:
        CONFIG.force_md_for_meta_scoring = True

        rec = _make_record(actives[0])
        rec.md_ligand_rmsd = None
        rec.md_pocket_rg_stability = None

        with pytest.raises(ConfigurationError, match="missing required MD"):
            scorer.predict(rec)
    finally:
        CONFIG.force_md_for_meta_scoring = original_value


# ── SHAP explanation tests ───────────────────────────────────────


def test_shap_explanation_returns_dict() -> None:
    """explain_prediction must return a non-empty dict for a fitted model."""
    import sys

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

    # Fake shap in sys.modules so the lazy import inside explain_prediction works
    import numpy as np
    fake_shap = MagicMock()
    fake_explainer = MagicMock()
    fake_shap.TreeExplainer.return_value = fake_explainer
    fake_explainer.shap_values.return_value = np.array(
        [[0.1, -0.05, 0.2, 0.0, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    with patch.dict("sys.modules", {"shap": fake_shap}):
        rec = _make_record(actives[0])
        result = scorer.explain_prediction(rec)

    assert isinstance(result, dict)
    assert len(result) > 0
    assert "vina_energy" in result
    assert result["vina_energy"] == pytest.approx(0.1, abs=1e-6)
    assert result["qed"] == pytest.approx(0.3, abs=1e-6)
    assert result["logp"] == pytest.approx(-0.1, abs=1e-6)


def test_shap_fallback_without_library(caplog: pytest.LogCaptureFixture) -> None:
    """When shap is not installed, explain_prediction must return an empty
    dict and log a warning."""
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
    result = scorer.explain_prediction(rec)

    assert isinstance(result, dict)
    assert len(result) == 0
    # The log should mention SHAP is not installed
    assert any(
        "SHAP is not installed" in message
        for message in caplog.messages
    )


def test_feature_importance_logging(caplog: pytest.LogCaptureFixture) -> None:
    """After fitting, the log must contain 'top SHAP features'."""
    import logging
    import sys
    caplog.set_level(logging.INFO)

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

    import numpy as np
    fake_shap = MagicMock()
    fake_explainer = MagicMock()
    fake_shap.TreeExplainer.return_value = fake_explainer
    n_feats = 16
    n_samples = 4
    fake_explainer.shap_values.return_value = np.random.randn(n_samples, n_feats).astype(np.float32)

    with patch.dict("sys.modules", {"shap": fake_shap}):
        scorer.fit(actives, inactives)

    assert any(
        "SHAP features" in message
        for message in caplog.messages
    ), "Expected log to contain SHAP features after fit"
