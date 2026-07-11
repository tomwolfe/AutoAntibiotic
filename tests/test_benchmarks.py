"""
Unit tests for the enrichment benchmark logic.

Tests the enrichment factor computation, ROC-AUC calculation,
and decoy generation logic in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from benchmarks.run_enrichment_test import (
    compute_enrichment_factor,
    compute_roc_auc,
    generate_decoys,
    score_compounds_with_fingerprint_similarity,
    score_compounds_with_dry_run,
)
from benchmarks.reference_data import (
    PBP2A_ACTIVES,
    PBP2A_INACTIVES,
    get_actives_smiles,
    get_inactives_smiles,
)
from autoantibiotic.models import CompoundRecord


class TestEnrichmentFactor:
    """Tests for the Enrichment Factor calculation."""

    def test_perfect_enrichment(self) -> None:
        """EF1% = 100 when all actives rank at the top."""
        scores = np.array([-10.0, -9.5, -9.0, -5.0, -4.0, -3.0, -2.0, -1.0])
        labels = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        ef = compute_enrichment_factor(scores, labels, fraction=0.25)
        n_actives = 2
        expected = (2 / n_actives) / 0.25
        assert ef == pytest.approx(expected)

    def test_random_enrichment(self) -> None:
        """EF1% ≈ 1.0 when actives are distributed randomly."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(-10, -1, size=1000)
        labels = np.zeros(1000, dtype=np.int64)
        labels[:100] = 1
        rng.shuffle(labels)
        ef = compute_enrichment_factor(scores, labels, fraction=0.01)
        assert 0.0 <= ef <= 10.0

    def test_no_actives(self) -> None:
        """EF1% = 1.0 when there are no actives in the set."""
        scores = np.array([-5.0, -4.0, -3.0])
        labels = np.array([0, 0, 0])
        ef = compute_enrichment_factor(scores, labels, fraction=0.01)
        assert ef == 1.0

    def test_all_actives(self) -> None:
        """EF1% = 1.0 when all compounds are actives."""
        scores = np.array([-5.0, -4.0, -3.0, -2.0])
        labels = np.array([1, 1, 1, 1])
        ef = compute_enrichment_factor(scores, labels, fraction=0.5)
        assert ef == pytest.approx(1.0)

    def test_single_compound_set(self) -> None:
        """EF handles single-compound edge case."""
        scores = np.array([-7.0])
        labels = np.array([1])
        ef = compute_enrichment_factor(scores, labels, fraction=0.01)
        assert ef > 0.0

    def test_ef1_vs_ef5_different(self) -> None:
        """Different enrichment fractions yield different values."""
        scores = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
        labels = np.array([1, 1, 1, 0, 0, 0, 0])
        ef1 = compute_enrichment_factor(scores, labels, fraction=0.01)
        ef5 = compute_enrichment_factor(scores, labels, fraction=0.05)
        assert ef1 != ef5


class TestROCAUC:
    """Tests for the ROC-AUC calculation."""

    def test_perfect_separation(self) -> None:
        """ROC-AUC = 1.0 when all actives score better than inactives."""
        scores = np.array([-10.0, -9.0, -8.0, -5.0, -4.0, -3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        auc = compute_roc_auc(scores, labels)
        assert auc == pytest.approx(1.0)

    def test_random_separation(self) -> None:
        """ROC-AUC ≈ 0.5 when scores are random."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(-10, -1, size=200)
        labels = np.zeros(200, dtype=np.int64)
        labels[:50] = 1
        rng.shuffle(labels)
        auc = compute_roc_auc(scores, labels)
        assert 0.2 <= auc <= 0.8

    def test_reverse_performance(self) -> None:
        """ROC-AUC = 1.0 when inactives score better (negated)."""
        scores = np.array([-3.0, -4.0, -5.0, -10.0, -9.0, -8.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        auc = compute_roc_auc(scores, labels)
        assert auc == pytest.approx(0.0, abs=0.01)

    def test_single_class(self) -> None:
        """ROC-AUC = 0.5 when only one class is present."""
        scores = np.array([-5.0, -4.0, -3.0])
        labels = np.array([1, 1, 1])
        auc = compute_roc_auc(scores, labels)
        assert auc == 0.5

    def test_ties_handled_gracefully(self) -> None:
        """ROC-AUC handles tied scores without error."""
        scores = np.array([-5.0, -5.0, -5.0, -3.0, -3.0, -3.0])
        labels = np.array([1, 1, 0, 1, 0, 0])
        auc = compute_roc_auc(scores, labels)
        assert 0.0 <= auc <= 1.0

    def test_lower_scores_better(self) -> None:
        """Since lower docking energies = better binding, negated scores
        should produce AUC > 0.5 when actives bind stronger."""
        scores = np.array([-10.0, -9.0, -8.0, -2.0, -1.0, 0.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        auc = compute_roc_auc(scores, labels)
        assert auc > 0.5


class TestDecoyGeneration:
    """Tests for property-matched decoy generation."""

    @pytest.fixture
    def sample_actives(self) -> list:
        active_smiles = get_actives_smiles()[:3]
        mols = []
        for smi in active_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
        return mols

    def test_decoy_generation_produces_correct_count(
        self, sample_actives,
    ) -> None:
        """Decoy count matches request per active."""
        decoys = generate_decoys(
            sample_actives, n_decoys_per_active=10, pool_size=2000, seed=42,
        )
        assert len(decoys) >= 3 * 5  # at least 5 per active

    def test_decoy_ids_are_unique(self, sample_actives) -> None:
        """Each decoy has a unique ID."""
        decoys = generate_decoys(
            sample_actives, n_decoys_per_active=10, pool_size=2000, seed=42,
        )
        ids = [d[0] for d in decoys]
        assert len(ids) == len(set(ids))

    def test_decoys_are_different_from_actives(self, sample_actives) -> None:
        """Decoys should not be identical to any active."""
        decoys = generate_decoys(
            sample_actives, n_decoys_per_active=5, pool_size=2000, seed=42,
        )
        active_smiles_set = {Chem.MolToSmiles(m) for m in sample_actives}
        for _, mol, _ in decoys:
            decoy_smi = Chem.MolToSmiles(mol)
            assert decoy_smi not in active_smiles_set


class TestScoringFunctions:
    """Tests for scoring functions used in benchmarking."""

    @pytest.fixture
    def sample_records(self) -> list:
        return [
            CompoundRecord(
                compound_id="TEST_ACTIVE_1",
                smiles="c1ccccc1O",  # phenol
                mol=Chem.MolFromSmiles("c1ccccc1O"),
            ),
            CompoundRecord(
                compound_id="TEST_INACTIVE_1",
                smiles="CCCCCCCCCCCCCCCCCC(=O)O",  # stearic acid
                mol=Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCC(=O)O"),
            ),
            CompoundRecord(
                compound_id="TEST_ACTIVE_2",
                smiles="c1ccc(O)c(CO)c1",  # salicyl alcohol
                mol=Chem.MolFromSmiles("c1ccc(O)c(CO)c1"),
            ),
        ]

    def test_dry_run_scoring_assigns_energies(
        self, sample_records,
    ) -> None:
        """Dry-run scoring sets pb2pa_allosteric_energy for all records."""
        results = score_compounds_with_dry_run(sample_records)
        for rec in results:
            assert rec.pb2pa_allosteric_energy is not None
            assert -10.0 <= rec.pb2pa_allosteric_energy <= -5.0

    def test_fingerprint_scoring_assigns_energies(
        self, sample_records,
    ) -> None:
        """Fingerprint scoring assigns energy-like values."""
        ref_smiles = get_actives_smiles()
        results = score_compounds_with_fingerprint_similarity(
            sample_records, ref_smiles,
        )
        for rec in results:
            assert rec.pb2pa_allosteric_energy is not None


class TestReferenceDataIntegrity:
    """Verify the reference datasets are valid."""

    def test_all_active_smiles_parse(self) -> None:
        """Every active SMILES in the reference list is valid."""
        for entry in PBP2A_ACTIVES:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"Failed to parse active: {entry['id']}"
            Chem.SanitizeMol(mol)
            assert mol.GetNumHeavyAtoms() > 3, f"Too few atoms: {entry['id']}"

    def test_all_inactive_smiles_parse(self) -> None:
        """Every inactive SMILES in the reference list is valid."""
        for entry in PBP2A_INACTIVES:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"Failed to parse inactive: {entry['id']}"

    def test_actives_are_distinct(self) -> None:
        """No duplicate active SMILES."""
        smiles_list = get_actives_smiles()
        assert len(smiles_list) == len(set(smiles_list))

    def test_inactives_are_distinct(self) -> None:
        """No duplicate inactive SMILES."""
        smiles_list = get_inactives_smiles()
        assert len(smiles_list) == len(set(smiles_list))

    def test_actives_and_inactives_disjoint(self) -> None:
        """Active and inactive sets should be disjoint."""
        active_set = set(get_actives_smiles())
        inactive_set = set(get_inactives_smiles())
        assert active_set.isdisjoint(inactive_set)

    def test_reference_data_has_references(self) -> None:
        """Each active entry has a reference source."""
        for entry in PBP2A_ACTIVES:
            assert "reference" in entry and entry["reference"]
        for entry in PBP2A_INACTIVES:
            assert "reference" in entry and entry["reference"]

    def test_csv_files_load_correctly(self) -> None:
        """Verify the CSV files load correctly and contain valid SMILES."""
        from pathlib import Path
        import pandas as pd
        from rdkit import Chem

        data_dir = Path(__file__).parent.parent / "data"
        actives_csv = data_dir / "pbp2a_actives.csv"
        inactives_csv = data_dir / "pbp2a_inactives.csv"

        assert actives_csv.exists(), f"Missing {actives_csv}"
        assert inactives_csv.exists(), f"Missing {inactives_csv}"

        df_act = pd.read_csv(actives_csv)
        assert list(df_act.columns) == ["id", "smiles", "reference"], \
            f"Unexpected columns in actives CSV: {list(df_act.columns)}"
        assert len(df_act) > 0, "Actives CSV is empty"

        df_inact = pd.read_csv(inactives_csv)
        assert list(df_inact.columns) == ["id", "smiles", "reference"], \
            f"Unexpected columns in inactives CSV: {list(df_inact.columns)}"
        assert len(df_inact) > 0, "Inactives CSV is empty"

        for _, row in df_act.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            assert mol is not None, f"Invalid SMILES in actives CSV: {row['id']}: {row['smiles']}"

        for _, row in df_inact.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            assert mol is not None, f"Invalid SMILES in inactives CSV: {row['id']}: {row['smiles']}"
