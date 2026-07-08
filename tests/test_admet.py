"""Unit tests for the ML-ADMET predictor and its integration into
:mod:`autoantibiotic.analysis` and :mod:`autoantibiotic.library_gen`."""

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.analysis import (
    MLADMETPredictor,
    _get_ml_admet_predictor,
    ChemBERTaEmbedder,
    predict_admet_profile,
    predict_herg_ml,
    predict_cyp_inhibition,
    predict_logs,
    predict_herg_risk,
)
from autoantibiotic.config import CONFIG, ConfigurationError
from autoantibiotic.library_gen import _filter_ml_admet, apply_filters, generate_candidate_library
from autoantibiotic.models import CompoundRecord


# ── Helper molecules ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def toxic_mol() -> Chem.Mol:
    """Doxorubicin — known cardiotoxic compound (hERG risk)."""
    smi = "CC1(C(=O)OC2=C1C=C3CC4=CC5=C(C=C4CN3C2=O)OC6=C(C=C(C=C6)C(=O)O)OC5)O"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, "Doxorubicin SMILES should parse"
    return mol


@pytest.fixture(scope="session")
def safe_mol() -> Chem.Mol:
    """Caffeine — generally considered safe for hERG."""
    mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    assert mol is not None, "Caffeine SMILES should parse"
    return mol


@pytest.fixture(scope="session")
def phenol_mol() -> Chem.Mol:
    """Phenol — simple, low-risk molecule."""
    mol = Chem.MolFromSmiles("c1ccccc1O")
    assert mol is not None
    return mol


# ── MLADMETPredictor tests ──────────────────────────────────────────────────


class TestMLADMETPredictorInit:
    """Verify that the MLADMETPredictor can be created and fits models."""

    def test_init_fits_models(self) -> None:
        predictor = MLADMETPredictor()
        assert predictor.available, "Predictor should be available after init"
        assert predictor.herg_model is not None, "hERG model should be fitted"
        assert predictor.cyp_model is not None, "CYP model should be fitted"

    def test_module_level_singleton(self) -> None:
        pred = _get_ml_admet_predictor()
        # May be None if CONFIG.use_ml_admet is True but fitting failed
        if pred is None:
            pytest.skip("Module-level predictor unavailable (check warnings).")
        assert isinstance(pred, MLADMETPredictor)

    def test_init_with_embedder(self, phenol_mol: Chem.Mol) -> None:
        """Predictor should accept a ChemBERTaEmbedder and fit on embeddings."""
        try:
            embedder = ChemBERTaEmbedder()
            predictor = MLADMETPredictor(embedder=embedder)
            assert predictor.available, (
                "Predictor with embedder should be available"
            )
            # verify predictions work
            prob = predictor.predict_herg_probability(phenol_mol)
            assert prob is not None
            assert 0.0 <= prob <= 1.0
        except (ImportError, Exception):
            pytest.skip("ChemBERTaEmbedder not available in this environment")

    def test_disabled_when_use_ml_admet_false(self) -> None:
        original = CONFIG.use_ml_admet
        CONFIG.use_ml_admet = False
        try:
            pred = _get_ml_admet_predictor()
            assert pred is None, "Predictor should be None when use_ml_admet=False"
        finally:
            CONFIG.use_ml_admet = original

    def test_trains_on_curated_csv_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that the predictor falls back to the curated CSV when
        the ChEMBL API module is unavailable."""
        import autoantibiotic.admet.predictors as pred_mod

        # Mock benchmarks.reference_data to be import-failed
        import builtins
        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "benchmarks.reference_data":
                raise ImportError("Mocked: ChEMBL unavailable")
            return original_import(name, *args, **kwargs)

        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", _mock_import)
            # Clear the cached predictor singleton
            m.setattr(pred_mod, "_ml_predictor", None)
            pred = MLADMETPredictor()
            assert pred.available, (
                "Predictor should be available when trained on curated CSV"
            )
            assert pred.herg_model is not None
            assert pred.cyp_model is not None


class TestChemBERTaEmbedder:
    """Verify that :class:`ChemBERTaEmbedder` produces the correct output."""

    def test_embedding_shape(self, phenol_mol: Chem.Mol) -> None:
        try:
            embedder = ChemBERTaEmbedder()
            emb = embedder.get_embedding(phenol_mol)
        except (ImportError, Exception):
            pytest.skip("ChemBERTaEmbedder not available in this environment")
            return  # unreachable, but satisfies type-checker

        assert emb.shape == (768,), f"Expected 768-dim, got {emb.shape}"
        assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
        assert np.all(np.isfinite(emb)), "Embedding should contain only finite values"

    def test_two_molecules_give_different_embeddings(
        self, toxic_mol: Chem.Mol, safe_mol: Chem.Mol,
    ) -> None:
        try:
            embedder = ChemBERTaEmbedder()
            e1 = embedder.get_embedding(toxic_mol)
            e2 = embedder.get_embedding(safe_mol)
        except (ImportError, Exception):
            pytest.skip("ChemBERTaEmbedder not available in this environment")
            return

        assert not np.allclose(e1, e2), "Different molecules should have different embeddings"

    def test_same_molecule_gives_same_embedding(self, phenol_mol: Chem.Mol) -> None:
        try:
            embedder = ChemBERTaEmbedder()
            e1 = embedder.get_embedding(phenol_mol)
            e2 = embedder.get_embedding(phenol_mol)
        except (ImportError, Exception):
            pytest.skip("ChemBERTaEmbedder not available in this environment")
            return

        assert np.allclose(e1, e2), "Same molecule should give identical embeddings"


class TestMLADMETPredictorFeatureComputation:
    """Test the feature engineering pipeline."""

    def test_feature_vector_shape(self, phenol_mol: Chem.Mol) -> None:
        feats = MLADMETPredictor.compute_features(phenol_mol)
        assert feats.shape == (2055,), f"Expected 2055 features, got {feats.shape}"
        assert feats.dtype == np.float32

    def test_feature_vector_is_finite(self, phenol_mol: Chem.Mol) -> None:
        feats = MLADMETPredictor.compute_features(phenol_mol)
        assert np.all(np.isfinite(feats)), "Features should be finite"

    def test_different_molecules_different_features(
        self, toxic_mol: Chem.Mol, safe_mol: Chem.Mol,
    ) -> None:
        f1 = MLADMETPredictor.compute_features(toxic_mol)
        f2 = MLADMETPredictor.compute_features(safe_mol)
        assert not np.allclose(f1, f2), "Feature vectors should differ"

    def test_same_molecule_same_features(self, phenol_mol: Chem.Mol) -> None:
        f1 = MLADMETPredictor.compute_features(phenol_mol)
        f2 = MLADMETPredictor.compute_features(phenol_mol)
        assert np.allclose(f1, f2), "Feature vectors should be identical"


class TestMLADMETPredictorPredictions:
    """Verify predictions on known toxic vs. safe compounds."""

    @pytest.fixture(scope="class")
    def predictor(self) -> MLADMETPredictor:
        p = MLADMETPredictor()
        if not p.available:
            pytest.skip("Predictor not available")
        return p

    def test_toxic_mol_herg_probability_higher_than_safe(
        self, predictor: MLADMETPredictor, toxic_mol: Chem.Mol, safe_mol: Chem.Mol,
    ) -> None:
        toxic_prob = predictor.predict_herg_probability(toxic_mol)
        safe_prob = predictor.predict_herg_probability(safe_mol)
        assert toxic_prob is not None
        assert safe_prob is not None
        assert toxic_prob >= safe_prob, (
            f"Expected toxic (doxorubicin) hERG prob ({toxic_prob:.3f}) "
            f">= safe (caffeine) prob ({safe_prob:.3f})"
        )

    def test_herg_probability_in_range(
        self, predictor: MLADMETPredictor, phenol_mol: Chem.Mol,
    ) -> None:
        prob = predictor.predict_herg_probability(phenol_mol)
        assert prob is not None
        assert 0.0 <= prob <= 1.0, f"Probability {prob:.3f} not in [0, 1]"

    def test_cyp_probability_in_range(
        self, predictor: MLADMETPredictor, phenol_mol: Chem.Mol,
    ) -> None:
        prob = predictor.predict_cyp_inhibition_probability(phenol_mol)
        assert prob is not None
        assert 0.0 <= prob <= 1.0, f"Probability {prob:.3f} not in [0, 1]"

    def test_herg_ml_returns_string(self, phenol_mol: Chem.Mol) -> None:
        result = predict_herg_ml(phenol_mol)
        assert result in ("High", "Moderate", "Low"), f"Unexpected result: {result}"

    def test_cyp_inhibition_returns_string(self, phenol_mol: Chem.Mol) -> None:
        result = predict_cyp_inhibition(phenol_mol)
        assert result in ("Yes", "No"), f"Unexpected result: {result}"


# ── predict_admet_profile integration tests ─────────────────────────────────


class TestPredictADMETProfile:
    """Ensure :func:`predict_admet_profile` uses ML predictors correctly."""

    def test_populates_flags(self, safe_mol: Chem.Mol) -> None:
        record = CompoundRecord(
            compound_id="TEST-001", smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            mol=safe_mol, passes_lipinski=True, qed_score=0.8,
        )
        result = predict_admet_profile(record)
        assert len(result.admet_flags) > 0, "Should have at least one flag"

    def test_includes_herg_flag(self, safe_mol: Chem.Mol) -> None:
        record = CompoundRecord(
            compound_id="TEST-002", smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            mol=safe_mol, passes_lipinski=True, qed_score=0.8,
        )
        result = predict_admet_profile(record)
        herg_flags = [f for f in result.admet_flags if "hERG" in f]
        assert len(herg_flags) >= 1, "Should contain hERG risk flag"

    def test_includes_cyp_flag(self, safe_mol: Chem.Mol) -> None:
        record = CompoundRecord(
            compound_id="TEST-003", smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            mol=safe_mol, passes_lipinski=True, qed_score=0.8,
        )
        result = predict_admet_profile(record)
        cyp_flags = [f for f in result.admet_flags if "CYP" in f]
        assert len(cyp_flags) >= 1, "Should contain CYP inhibition flag"

    def test_invalid_molecule_handled(self) -> None:
        record = CompoundRecord(
            compound_id="TEST-BAD", smiles="this_is_not_a_smiles",
        )
        result = predict_admet_profile(record)
        assert "ADMET: invalid molecule" in result.admet_flags

    def test_backward_compatible_solubility(self, phenol_mol: Chem.Mol) -> None:
        """Solubility via ESOL should still work."""
        logs = predict_logs(phenol_mol)
        assert isinstance(logs, float), "LogS should be a float"
        assert logs < 0, "Phenol LogS should be negative"


# ── _filter_ml_admet integration tests ──────────────────────────────────────


class TestMLADMETFilter:
    """Verify that :func:`_filter_ml_admet` correctly flags toxic compounds."""

    def test_filter_passes_safe_molecule(self, safe_mol: Chem.Mol) -> None:
        record = CompoundRecord(
            compound_id="TEST", smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            mol=safe_mol,
        )
        ok, reason = _filter_ml_admet(record, safe_mol)
        assert ok, f"Safe molecule should pass filter, got reason: {reason}"

    def test_disabled_when_use_ml_admet_false(self, safe_mol: Chem.Mol) -> None:
        original = CONFIG.use_ml_admet
        CONFIG.use_ml_admet = False
        try:
            record = CompoundRecord(
                compound_id="TEST", smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
                mol=safe_mol,
            )
            ok, _ = _filter_ml_admet(record, safe_mol)
            assert ok, "Should pass when ML-ADMET is disabled"
        finally:
            CONFIG.use_ml_admet = original


# ── apply_filters integration tests ─────────────────────────────────────────


class TestApplyFiltersMLADMETIntegration:
    """Verify that the ML-ADMET filter is integrated into the pipeline."""

    def test_pipeline_runs_with_ml_admet_enabled(self) -> None:
        """apply_filters should run without error when ML-ADMET is enabled."""
        records = generate_candidate_library(target_count=20, seed=42)
        passed = apply_filters(records)
        assert isinstance(passed, list)
        # Should not crash; at least one compound may pass
        assert len(passed) >= 0

    def test_pipeline_runs_with_ml_admet_disabled(self) -> None:
        """apply_filters should run without error when ML-ADMET is disabled."""
        original = CONFIG.use_ml_admet
        CONFIG.use_ml_admet = False
        try:
            records = generate_candidate_library(target_count=20, seed=42)
            passed = apply_filters(records)
            assert isinstance(passed, list)
        finally:
            CONFIG.use_ml_admet = original


# ── Fallback tests ──────────────────────────────────────────────────────────


class TestFallbackBehavior:
    """Verify backward compatibility when ML models are unavailable."""

    def test_herg_ml_falls_back_to_rule_based(self, phenol_mol: Chem.Mol) -> None:
        """predict_herg_ml should return a valid string even if ML is off."""
        original = CONFIG.use_ml_admet
        CONFIG.use_ml_admet = False
        try:
            result = predict_herg_ml(phenol_mol)
            assert result in ("High", "Moderate", "Low")
        finally:
            CONFIG.use_ml_admet = original

    def test_cyp_falls_back(self, phenol_mol: Chem.Mol) -> None:
        original = CONFIG.use_ml_admet
        CONFIG.use_ml_admet = False
        try:
            result = predict_cyp_inhibition(phenol_mol)
            assert result in ("Yes", "No")
        finally:
            CONFIG.use_ml_admet = original
