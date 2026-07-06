"""Tests for active learning retraining stub."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoantibiotic.active_learning import retrain_meta_scorer, _parse_active_learning_csv


class TestParseActiveLearningCSV:
    """Tests for CSV parsing in active learning."""

    @pytest.fixture
    def valid_csv_path(self, tmp_path):
        csv_path = tmp_path / "active_data.csv"
        csv_path.write_text(
            "smiles,ic50\n"
            "CC(=O)OC,1e-6\n"
            "CC(C)O,5e-5\n"
            "CCO,1e-3\n"
        )
        return str(csv_path)

    @pytest.fixture
    def invalid_csv_path(self, tmp_path):
        csv_path = tmp_path / "invalid_data.csv"
        csv_path.write_text(
            "name,value\n"
            "foo,1.0\n"
            "bar,2.0\n"
        )
        return str(csv_path)

    @pytest.fixture
    def empty_csv_path(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        return str(csv_path)

    def test_parse_valid_csv(self, valid_csv_path):
        smiles, pIC50 = _parse_active_learning_csv(valid_csv_path)
        assert len(smiles) == 3
        assert len(pIC50) == 3
        # pIC50 = -log10(ic50)
        assert abs(pIC50[0] - (-np.log10(1e-6))) < 1e-10
        assert abs(pIC50[1] - (-np.log10(5e-5))) < 1e-10
        assert abs(pIC50[2] - (-np.log10(1e-3))) < 1e-10

    def test_parse_missing_columns(self, invalid_csv_path):
        smiles, pIC50 = _parse_active_learning_csv(invalid_csv_path)
        assert len(smiles) == 0
        assert len(pIC50) == 0

    def test_parse_empty_csv(self, empty_csv_path):
        smiles, pIC50 = _parse_active_learning_csv(empty_csv_path)
        assert len(smiles) == 0
        assert len(pIC50) == 0

    def test_parse_nonexistent_file(self):
        smiles, pIC50 = _parse_active_learning_csv("/nonexistent/file.csv")
        assert len(smiles) == 0
        assert len(pIC50) == 0

    def test_parse_invalid_values(self, tmp_path):
        csv_path = tmp_path / "invalid_values.csv"
        csv_path.write_text(
            "smiles,ic50\n"
            "CCO,not_a_number\n"
            "CC(=O)OC,1e-6\n"
        )
        smiles, pIC50 = _parse_active_learning_csv(str(csv_path))
        # Only the valid entry should be parsed
        assert len(smiles) == 1
        assert "CC(=O)OC" in smiles

    def test_parse_negative_ic50(self, tmp_path):
        csv_path = tmp_path / "negative_ic50.csv"
        csv_path.write_text(
            "smiles,ic50\n"
            "CCO,-100.0\n"
            "CC(=O)OC,1e-6\n"
        )
        smiles, pIC50 = _parse_active_learning_csv(str(csv_path))
        # Only the valid positive entry should be parsed
        assert len(smiles) == 1

    def test_empty_string_columns(self, tmp_path):
        csv_path = tmp_path / "empty_columns.csv"
        csv_path.write_text(
            "smiles,ic50\n"
            ",\n"
            "CCO,1e-6\n"
        )
        smiles, pIC50 = _parse_active_learning_csv(str(csv_path))
        assert len(smiles) == 1
        assert "CCO" in smiles


class TestRetrainMetaScorer:
    """Tests for active learning retraining."""

    @pytest.fixture
    def csv_path(self, tmp_path):
        csv_path = tmp_path / "training_data.csv"
        csv_path.write_text(
            "smiles,ic50\n"
            "CC(=O)OC,1e-6\n"
            "CC(C)O,5e-5\n"
        )
        return str(csv_path)

    def test_retrain_with_valid_csv(self, csv_path):
        result = retrain_meta_scorer(csv_path)
        assert result is True

    def test_retrain_with_nonexistent_file(self):
        result = retrain_meta_scorer("/nonexistent/file.csv")
        assert result is False

    def test_retrain_with_too_few_entries(self, tmp_path):
        csv_path = tmp_path / "too_few.csv"
        csv_path.write_text(
            "smiles,ic50\n"
            "CCO,1e-6\n"
        )
        result = retrain_meta_scorer(str(csv_path))
        assert result is False

    def test_retrain_model_path_specified(self, csv_path, tmp_path):
        model_path = str(tmp_path / "model.joblib")
        result = retrain_meta_scorer(csv_path, model_path=model_path)
        assert result is True

    def test_retrain_invalid_csv_path(self):
        result = retrain_meta_scorer("invalid.csv")
        assert result is False


# Import numpy for the test
import numpy as np
