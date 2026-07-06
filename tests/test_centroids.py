"""Unit tests for 3D coordinate centroid calculation."""

import numpy as np
import pytest

from autoantibiotic.structure_prep import compute_residue_centroid


class TestResidueCentroidShape:
    """``compute_residue_centroid`` returns correctly shaped arrays."""

    def test_allosteric_site_returns_3_element_array(self, test_pdb_path: str) -> None:
        resid_list = ["ASN159", "GLU237", "ARG241"]
        centroid = compute_residue_centroid(test_pdb_path, resid_list)
        assert isinstance(centroid, np.ndarray), "Should return numpy array"
        assert centroid.shape == (3,), f"Expected shape (3,), got {centroid.shape}"

    def test_allosteric_site_values_are_finite(self, test_pdb_path: str) -> None:
        resid_list = ["ASN159", "GLU237", "ARG241"]
        centroid = compute_residue_centroid(test_pdb_path, resid_list)
        assert np.all(np.isfinite(centroid)), "All centroid values should be finite"

    def test_allosteric_site_centroid_value(self, test_pdb_path: str) -> None:
        resid_list = ["ASN159", "GLU237", "ARG241"]
        centroid = compute_residue_centroid(test_pdb_path, resid_list)
        expected = np.array([2.5, 2.5, 2.5])
        np.testing.assert_allclose(centroid, expected, atol=1e-6)

    def test_active_site_returns_3_element_array(self, test_pdb_path: str) -> None:
        resid_list = ["SER403"]
        centroid = compute_residue_centroid(test_pdb_path, resid_list)
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (3,)

    def test_active_site_centroid_value(self, test_pdb_path: str) -> None:
        resid_list = ["SER403"]
        centroid = compute_residue_centroid(test_pdb_path, resid_list)
        expected = np.array([4.5, 4.5, 4.5])
        np.testing.assert_allclose(centroid, expected, atol=1e-6)

    def test_single_residue_returns_3_element_array(self, test_pdb_path: str) -> None:
        centroid = compute_residue_centroid(test_pdb_path, ["ASN159"])
        assert centroid.shape == (3,)

    def test_duplicate_residues_handled(self, test_pdb_path: str) -> None:
        centroid = compute_residue_centroid(test_pdb_path, ["ASN159", "ASN159"])
        assert centroid.shape == (3,)
        assert np.all(np.isfinite(centroid))


class TestResidueCentroidErrors:
    """Error handling for centroid calculation."""

    def test_missing_residue_raises(self, test_pdb_path: str) -> None:
        with pytest.raises(ValueError, match="No matching residues"):
            compute_residue_centroid(test_pdb_path, ["GLY999"])
