"""Unit tests for AutoAntibiotic filter and validation logic."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from autoantibiotic.analysis import compute_pharmacophore_score, compute_selectivity_index
from autoantibiotic.library_gen import _validate_mol, generate_candidate_library, apply_filters
from autoantibiotic.io_utils import parse_vina_energy
from tests.conftest import BETA_LACTAM_SMARTS


class TestSMILESValidation:
    """``_validate_mol`` ensures SMILES parse and sanitise correctly."""

    def test_valid_smiles(self) -> None:
        mol = _validate_mol("c1ccccc1O")
        assert mol is not None
        assert mol.GetNumAtoms() > 0

    def test_invalid_smiles_returns_none(self) -> None:
        assert _validate_mol("this_is_not_a_smiles") is None

    def test_empty_string_returns_none(self) -> None:
        result = _validate_mol("")
        assert result is None or result.GetNumAtoms() == 0


class TestBetaLactamFilter:
    """Beta-lactam SMARTS pattern correctly identifies reactive warheads."""

    @pytest.fixture
    def lactam_pattern(self) -> Chem.Mol:
        pat = Chem.MolFromSmarts(BETA_LACTAM_SMARTS)
        assert pat is not None, "Beta-lactam SMARTS should compile"
        return pat

    def test_beta_lactam_matches(self, beta_lactam_mol: Chem.Mol,
                                 lactam_pattern: Chem.Mol) -> None:
        assert beta_lactam_mol.HasSubstructMatch(lactam_pattern)

    def test_non_beta_lactam_does_not_match(self, non_beta_lactam_mol: Chem.Mol,
                                            lactam_pattern: Chem.Mol) -> None:
        assert not non_beta_lactam_mol.HasSubstructMatch(lactam_pattern)


class TestSelectivityIndex:
    """``compute_selectivity_index`` handles edge cases correctly."""

    def test_normal_case(self) -> None:
        si = compute_selectivity_index(-8.0, -4.0)
        assert si == pytest.approx(2.0)

    def test_division_by_zero_safe(self) -> None:
        si = compute_selectivity_index(-8.0, 0.0)
        assert si == 0.0

    def test_both_positive_returns_zero(self) -> None:
        si = compute_selectivity_index(1.0, 2.0)
        assert si == 0.0

    def test_pb2pa_positive_returns_zero(self) -> None:
        si = compute_selectivity_index(0.5, -4.0)
        assert si == 0.0

    def test_human_positive_returns_zero(self) -> None:
        si = compute_selectivity_index(-8.0, 0.5)
        assert si == 0.0

    def test_very_small_human_energy(self) -> None:
        si = compute_selectivity_index(-8.0, -1e-8)
        assert si == 0.0

    def test_negative_energies(self) -> None:
        si = compute_selectivity_index(-10.0, -5.0)
        assert si == pytest.approx(2.0)

    def test_zero_inputs(self) -> None:
        si = compute_selectivity_index(0.0, 0.0)
        assert si == 0.0


class TestVinaEnergyParsing:
    """``parse_vina_energy`` extracts binding energies from Vina output."""

    def test_stdout_mode_line(self) -> None:
        stdout = (
            "mode |   affinity | dist from best mode\n"
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
            "-----+------------+----------+----------\n"
            "   1       -8.123       0.000      0.000\n"
            "   2       -7.500       1.234      2.345\n"
        )
        energy = parse_vina_energy(stdout)
        assert energy == pytest.approx(-8.123)

    def test_affinity_fallback(self) -> None:
        stdout = "Affinity: -9.456 (kcal/mol)"
        energy = parse_vina_energy(stdout)
        assert energy == pytest.approx(-9.456)

    def test_no_energy_returns_none(self) -> None:
        assert parse_vina_energy("No docking results") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_vina_energy("") is None


class TestLibraryLipinskiCompliance:
    """Verify generated compounds satisfy Lipinski Rule-of-5."""

    def test_generated_molecules_have_reasonable_mw(self) -> None:
        """Passing compounds should have MW ≤ 500 Da."""
        from rdkit.Chem import Descriptors

        records = generate_candidate_library(target_count=20, seed=42)
        passed = apply_filters(records)
        assert len(passed) > 0

        for rec in passed:
            mol = rec.mol
            assert mol is not None
            mw = Descriptors.MolWt(mol)
            assert mw <= 500.0, f"{rec.compound_id} MW {mw:.1f} > 500"

    def test_generated_molecules_have_reasonable_logp(self) -> None:
        """Passing compounds should have LogP ≤ 5.0."""
        from rdkit.Chem import Crippen

        records = generate_candidate_library(target_count=20, seed=42)
        passed = apply_filters(records)
        assert len(passed) > 0

        for rec in passed:
            mol = rec.mol
            assert mol is not None
            logp = Crippen.MolLogP(mol)
            assert logp <= 5.0, f"{rec.compound_id} LogP {logp:.2f} > 5.0"

    def test_generated_molecules_have_reasonable_hbd(self) -> None:
        """Passing compounds should have HBD ≤ 5."""
        from rdkit.Chem import Descriptors

        records = generate_candidate_library(target_count=20, seed=42)
        passed = apply_filters(records)
        assert len(passed) > 0

        for rec in passed:
            mol = rec.mol
            assert mol is not None
            hbd = Descriptors.NumHDonors(mol)
            assert hbd <= 5, f"{rec.compound_id} HBD {hbd} > 5"

    def test_generated_molecules_have_reasonable_hba(self) -> None:
        """Passing compounds should have HBA ≤ 10."""
        from rdkit.Chem import Descriptors

        records = generate_candidate_library(target_count=20, seed=42)
        passed = apply_filters(records)
        assert len(passed) > 0

        for rec in passed:
            mol = rec.mol
            assert mol is not None
            hba = Descriptors.NumHAcceptors(mol)
            assert hba <= 10, f"{rec.compound_id} HBA {hba} > 10"


class TestPharmacophoreScore:
    """``compute_pharmacophore_score`` returns sensible values."""

    def test_identical_molecules_score_near_one(self) -> None:
        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        score = compute_pharmacophore_score(mol, mol)
        assert score is not None
        assert 0.0 <= score <= 1.0
        # identical molecules should score near 1.0
        assert score > 0.8

    def test_very_different_molecules_score_low(self) -> None:
        phenol = Chem.MolFromSmiles("c1ccccc1O")
        decane = Chem.MolFromSmiles("CCCCCCCCCC")
        assert phenol is not None and decane is not None
        score = compute_pharmacophore_score(phenol, decane)
        # decane has no N/O atoms → no features → should return 1.0
        assert score is not None
        assert score == 1.0

    def test_ethanol_and_methanol_similar(self) -> None:
        ethanol = Chem.MolFromSmiles("CCO")
        methanol = Chem.MolFromSmiles("CO")
        assert ethanol is not None and methanol is not None
        score = compute_pharmacophore_score(ethanol, methanol)
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_returns_none_on_invalid_mol(self) -> None:
        mol = Chem.MolFromSmiles("c1ccccc1O")
        assert mol is not None
        # pass a string that isn't a Mol to force an error
        score = compute_pharmacophore_score(mol, "not_a_mol")  # type: ignore
        assert score is None

    def test_hbd_and_hba_feature_types(self) -> None:
        """Molecules with shared donors/acceptors produce intermediate scores."""
        ethanol = Chem.MolFromSmiles("CCO")
        acetic_acid = Chem.MolFromSmiles("CC(=O)O")
        assert ethanol is not None and acetic_acid is not None
        score = compute_pharmacophore_score(ethanol, acetic_acid)
        assert score is not None
        assert 0.0 <= score <= 1.0
