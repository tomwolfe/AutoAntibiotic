"""Unit tests for pharmacophore-aware library generation."""

from typing import Optional

import numpy as np
import pytest
from rdkit import Chem

from autoantibiotic.library_gen import (
    _build_allosteric_pharmacophore,
    check_pharmacophore_match,
    generate_pharmacophore_aware_library,
)


class TestBuildAllostericPharmacophore:
    """``_build_allosteric_pharmacophore`` returns a valid feature dict."""

    def test_returns_dict_or_none(self) -> None:
        query = _build_allosteric_pharmacophore()
        assert query is None or isinstance(query, dict)

    def test_has_required_keys(self) -> None:
        query = _build_allosteric_pharmacophore()
        if query is not None:
            assert "feat_types" in query
            assert "residue_map" in query

    def test_contains_three_features(self) -> None:
        query = _build_allosteric_pharmacophore()
        if query is not None:
            assert len(query["feat_types"]) == 3
            assert "Donor" in query["feat_types"]
            assert "Acceptor" in query["feat_types"]
            assert "Hydrophobe" in query["feat_types"]

    def test_residue_map_complete(self) -> None:
        query = _build_allosteric_pharmacophore()
        if query is not None:
            assert "TYR159" in query["residue_map"]
            assert "ALA237" in query["residue_map"]
            assert "MET241" in query["residue_map"]


class TestCheckPharmacophoreMatch:
    """``check_pharmacophore_match`` correctly identifies feature matches."""

    def _get_query(self) -> Optional[dict]:
        return _build_allosteric_pharmacophore()

    def test_molecule_with_all_three_features(self) -> None:
        """A molecule with donor, acceptor, and hydrophobic regions."""
        mol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1O")  # acetaminophen: donor (OH), acceptor (C=O), hydrophobe (ring)
        assert mol is not None
        query = self._get_query()
        result = check_pharmacophore_match(mol, query, min_matches=2)
        assert result is True

    def test_molecule_with_no_features_returns_false(self) -> None:
        """Ethane has no donor/acceptor/hydrophobe features."""
        mol = Chem.MolFromSmiles("CC")
        assert mol is not None
        query = self._get_query()
        if query is not None:
            result = check_pharmacophore_match(mol, query, min_matches=2)
            assert result is False

    def test_molecule_with_one_feature_fails_strict(self) -> None:
        """Propane (hydrophobe only) should fail min_matches=2."""
        mol = Chem.MolFromSmiles("CCC")
        assert mol is not None
        query = self._get_query()
        if query is not None:
            result = check_pharmacophore_match(mol, query, min_matches=2)
            assert result is False

    def test_min_matches_one_passes_with_one_feature(self) -> None:
        """Propane passes when min_matches=1 (Hydrophobe detected)."""
        mol = Chem.MolFromSmiles("CCC")
        assert mol is not None
        query = self._get_query()
        if query is not None:
            result = check_pharmacophore_match(mol, query, min_matches=1)
            assert result is True

    def test_none_query_passes_through(self) -> None:
        """When query is None, the check passes (graceful fallback)."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        result = check_pharmacophore_match(mol, query=None)
        assert result is True

    def test_acceptor_only(self) -> None:
        """Formaldehyde (H2C=O) is an acceptor only."""
        mol = Chem.MolFromSmiles("C=O")
        assert mol is not None
        query = self._get_query()
        if query is not None:
            result = check_pharmacophore_match(mol, query, min_matches=2)
            assert result is False

    def test_donor_and_acceptor(self) -> None:
        """Formic acid has donor (OH) and acceptor (C=O)."""
        mol = Chem.MolFromSmiles("C(=O)O")
        assert mol is not None
        query = self._get_query()
        if query is not None:
            result = check_pharmacophore_match(mol, query, min_matches=2)
            assert result is True


class TestGeneratePharmacophoreAwareLibrary:
    """``generate_pharmacophore_aware_library`` produces enriched libraries."""

    def test_returns_list_of_records(self) -> None:
        library = generate_pharmacophore_aware_library(
            target_count=10, seed=42,
        )
        assert isinstance(library, list)
        if library:
            rec = library[0]
            assert hasattr(rec, "compound_id")
            assert hasattr(rec, "smiles")

    def test_respects_target_count(self) -> None:
        library = generate_pharmacophore_aware_library(
            target_count=10, seed=42,
        )
        assert len(library) <= 10

    def test_with_pocket_coords(self) -> None:
        """Passing dummy coords should not break anything."""
        dummy_coords = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        library = generate_pharmacophore_aware_library(
            target_count=5, seed=42,
            allosteric_pocket_coords=dummy_coords,
        )
        assert isinstance(library, list)

    def test_molecules_pass_pharmacophore_check(self) -> None:
        """All returned molecules should satisfy the pharmacophore."""
        query = _build_allosteric_pharmacophore()
        if query is None:
            pytest.skip("Pharmacophore factory unavailable")
        library = generate_pharmacophore_aware_library(
            target_count=10, seed=42,
        )
        for rec in library:
            mol = rec.mol
            if mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
            assert mol is not None
            assert check_pharmacophore_match(mol, query, min_matches=2) is True
