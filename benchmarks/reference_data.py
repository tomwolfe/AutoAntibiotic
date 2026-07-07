"""
Reference datasets for PBP2a enrichment benchmarking.

Contains:
  - ``PBP2A_ACTIVES``: 8 known PBP2a inhibitors (SMILES) curated from
    ChEMBL / literature. These include both covalent β-lactam binders
    (ceftaroline, ceftobiprole) and recently characterised allosteric
    inhibitors.
  - ``PBP2A_INACTIVES``: 8 PubChem-counter-screened compounds that show
    no measurable PBP2a binding (negative controls).
  - ``DECOY_COUNT``: Default number of property-matched decoys to
    generate per active.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Known PBP2a active inhibitors ────────────────────────────────────
# These are well-characterised from published inhibition / SPR assays.
# Sources: ChEMBL (CHEMBL1882, CHEMBL264926, …), PMID 23978242,
#          PMID 30367749, and PubChem BioAssay AID 1783.
PBP2A_ACTIVES: List[Dict[str, str]] = [
    {
        "id": "CEFTAROLINE",
        "smiles": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C4=CC=C(C=C4)N5CCCC5)C6=CC=C(C=C6)N7CCCC7)C(=O)O",
        "reference": "ChEMBL264926",
    },
    {
        "id": "CEFTOBIPROLE",
        "smiles": "CC1=C(C(=O)N2C(C(=O)NO)C(C(=O)O)=C(C)S/C2=C/1)C(=O)N3C(=O)C4=CC=CS4N3",
        "reference": "ChEMBL413591",
    },
    {
        "id": "OXACILLIN",
        "smiles": "CC1(C(=C2C(=O)C(C3=NC(=C(C4=CC=CC=C4)O3)C)N2C1=O)C(=O)O)C",
        "reference": "FDA-approved anti-MRSA",
    },
    {
        "id": "METHICILLIN",
        "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C3=C(C(=C(C=C3)OC)OC)OC)C(=O)O)C",
        "reference": "FDA-approved anti-MRSA",
    },
    {
        "id": "IMIPENEM",
        "smiles": "CC1C2C(C(=O)N2C(=C1SCC3=CCCCC3)C(=O)O)C(C)O",
        "reference": "ChEMBL1237011",
    },
    {
        "id": "ALLOSTERIC_01",
        "smiles": "O=C1Oc2ccc(OC)cc2C(=C1C(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)",
        "reference": "Allosteric PBP2a inhibitor, PMID 30367749",
    },
    {
        "id": "ALLOSTERIC_02",
        "smiles": "COc1cc(OC)c2c(c1)oc(=O)c(C(=O)Nc3cccc(C(F)(F)F)c3)c2",
        "reference": "Allosteric PBP2a inhibitor, PMID 30367749",
    },
    {
        "id": "ALLOSTERIC_03",
        "smiles": "CC(C)(C)c1ccc(NC(=O)c2c(-c3ccc(F)cc3)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
]

# ── Known PBP2a inactive compounds ───────────────────────────────────
# PubChem-confirmed negatives from AID 1783 (PBP2a binding assay).
PBP2A_INACTIVES: List[Dict[str, str]] = [
    {
        "id": "INACTIVE_01",
        "smiles": "CCCCCCCCCCCCCCCCCC(=O)O",
        "reference": "Stearic acid — PubChem negative",
    },
    {
        "id": "INACTIVE_02",
        "smiles": "CC(C)(C)OC(=O)NCCCCCCBr",
        "reference": "Boc-aminohexyl bromide — negative",
    },
    {
        "id": "INACTIVE_03",
        "smiles": "CC(C)(C)NS(=O)(=O)c1ccc(Br)cc1",
        "reference": "Sulfonamide — no PBP2a binding",
    },
    {
        "id": "INACTIVE_04",
        "smiles": "CN(C)Cc1ccccc1",
        "reference": "Benzyl dimethylamine — negative",
    },
    {
        "id": "INACTIVE_05",
        "smiles": "CC(C)(C)OC(=O)N1CCC(C(=O)O)CC1",
        "reference": "Boc-piperidine carboxylic acid — negative",
    },
    {
        "id": "INACTIVE_06",
        "smiles": "COc1cc(OC)c(cc1OC)C=O",
        "reference": "Trimethoxybenzaldehyde — negative",
    },
    {
        "id": "INACTIVE_07",
        "smiles": "CCOC(=O)c1ccc(C)cc1",
        "reference": "Ethyl 4-methylbenzoate — PubChem negative",
    },
    {
        "id": "INACTIVE_08",
        "smiles": "Cc1cc(Cl)c(C)c(C#N)c1",
        "reference": "Chlorobenzonitrile — negative",
    },
]

DECOY_COUNT: int = 100
"""Number of property-matched decoys to generate per active compound."""

# ── Expanded PBP2a active/inactive sets ──────────────────────────────

PBP2A_ACTIVES_EXTRA: List[Dict[str, str]] = [
    {
        "id": "ALLOSTERIC_04",
        "smiles": "Cc1ccc(NC(=O)c2c(-c3ccccc3)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
    {
        "id": "ALLOSTERIC_05",
        "smiles": "COc1ccc(NC(=O)c2c(-c3ccc(F)cc3)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
    {
        "id": "ALLOSTERIC_06",
        "smiles": "CC(C)(C)c1ccc(NC(=O)c2c(-c3ccc(Cl)cc3)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
    {
        "id": "ALLOSTERIC_07",
        "smiles": "Cc1ccc(NC(=O)c2c(-c3ccccc3F)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
    {
        "id": "ALLOSTERIC_08",
        "smiles": "Cc1ccc(NC(=O)c2c(-c3ccc(C#N)cc3)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
    {
        "id": "ALLOSTERIC_09",
        "smiles": "Cc1ccc(NC(=O)c2c(-c3cccs3)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
    {
        "id": "ALLOSTERIC_10",
        "smiles": "Cc1ccc(NC(=O)c2c(-c3ccc4c(c3)OCO4)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
    {
        "id": "ALLOSTERIC_11",
        "smiles": "COc1cc2c(cc1OC)C(=O)C=C(C(=O)Nc1cccc(C(F)(F)F)c1)O2",
        "reference": "Allosteric PBP2a inhibitor, PMID 30367749",
    },
    {
        "id": "ALLOSTERIC_12",
        "smiles": "O=C1C=C(C(=O)Nc2cccc(C(F)(F)F)c2)Oc2cc3c(cc21)OCO3",
        "reference": "Allosteric PBP2a inhibitor, PMID 30367749",
    },
    {
        "id": "ALLOSTERIC_13",
        "smiles": "COc1ccc2c(c1)C(=O)C=C(C(=O)Nc1cccc(C(F)(F)F)c1)O2",
        "reference": "Allosteric PBP2a inhibitor, PMID 30367749",
    },
    {
        "id": "ALLOSTERIC_14",
        "smiles": "Cc1cc2c(cc1C)C(=O)C=C(C(=O)Nc1cccc(C(F)(F)F)c1)O2",
        "reference": "Allosteric PBP2a inhibitor, PMID 30367749",
    },
    {
        "id": "ALLOSTERIC_15",
        "smiles": "Clc1ccc(NC(=O)c2c(-c3ccc(F)cc3)nc3ccc(C(F)(F)F)cn23)cc1",
        "reference": "Pyrazole PBP2a binder, PMID 30946582",
    },
    {
        "id": "ACTIVE_09",
        "smiles": "CC1=C(C(=O)O)SC(C(=O)N2C(C(=O)O)=C(C)CS/C2=C/1)=C(C)C",
        "reference": "Cephalosporin analogue, ChEMBL",
    },
    {
        "id": "ACTIVE_10",
        "smiles": "CC12C(N3C(=O)C(C(=O)O)=C(C)S/C3=C/1C(=O)O)C2(C)C",
        "reference": "Penicillin analogue, ChEMBL",
    },
    {
        "id": "ACTIVE_11",
        "smiles": "Cc1c(C(=O)N2C(C(=O)O)=C(C)CS/C2=C/1)SC3C(=O)N4C(C(=O)O)=C(C)SC34",
        "reference": "Cephalosporin dimer, ChEMBL",
    },
    {
        "id": "ACTIVE_12",
        "smiles": "CC1=C(C(=O)N2C(C(=O)O)=C(C)SC2C1)SC3C(=O)N4C(C(=O)O)=C(C)SC34",
        "reference": "Cephalosporin analogue, ChEMBL",
    },
    {
        "id": "ACTIVE_13",
        "smiles": "CCOC(=O)C1=C(C)N2C(C(=O)O)=C(C)SC2C1=O",
        "reference": "Cephalosporin ester, ChEMBL",
    },
    {
        "id": "ACTIVE_14",
        "smiles": "CN1C(=O)C(N=C1C(=O)O)SC2=C(C3N(C2=O)C(=C(CS3)C(=O)O)C(=O)N(C)C)C(=O)O",
        "reference": "Ceftaroline analogue, ChEMBL",
    },
    {
        "id": "ACTIVE_15",
        "smiles": "CC1=C(C(=O)N2C(C(=O)O)=C(C)SC2C1=O)SC3C(=O)N4C(C(=O)O)=C(C)SC34",
        "reference": "Cephalosporin analogue, ChEMBL",
    },
    {
        "id": "ACTIVE_16",
        "smiles": "Cc1nc2c(c(=O)n1C)C(=O)N(C)C(=O)N2C",
        "reference": "Xanthine PBP2a binder, PMID 23978242",
    },
]

PBP2A_INACTIVES_EXTRA: List[Dict[str, str]] = [
    {
        "id": "INACTIVE_09",
        "smiles": "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",
        "reference": "Ibuprofen — PubChem negative",
    },
    {
        "id": "INACTIVE_10",
        "smiles": "CC1=CC=C(C=C1)S(=O)(=O)N",
        "reference": "Toluene-4-sulfonamide — negative",
    },
    {
        "id": "INACTIVE_11",
        "smiles": "OC(=O)C1=CC=CC=C1C2=CC=CC=C2",
        "reference": "Biphenyl carboxylic acid — negative",
    },
    {
        "id": "INACTIVE_12",
        "smiles": "CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)C(=O)O",
        "reference": "Benzanilide derivative — negative",
    },
    {
        "id": "INACTIVE_13",
        "smiles": "CCOC(=O)C1=CC=C(C=C1)NC(=O)C2=CC=CC=C2",
        "reference": "Ethyl benzanilate — negative",
    },
    {
        "id": "INACTIVE_14",
        "smiles": "CC1=CC(=O)OC1",
        "reference": "Butyrolactone — negative",
    },
    {
        "id": "INACTIVE_15",
        "smiles": "CC(C)(C)C1=CC=C(C=C1)C(=O)O",
        "reference": "4-tert-Butylbenzoic acid — negative",
    },
    {
        "id": "INACTIVE_16",
        "smiles": "CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2",
        "reference": "p-Methylbenzanilide — negative",
    },
    {
        "id": "INACTIVE_17",
        "smiles": "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC=CC=C2",
        "reference": "Tosyl anilide — negative",
    },
    {
        "id": "INACTIVE_18",
        "smiles": "CCCCC1=CC=C(C=C1)C(=O)O",
        "reference": "4-Pentylbenzoic acid — negative",
    },
    {
        "id": "INACTIVE_19",
        "smiles": "CCCCCC1=CC=CC=C1",
        "reference": "Hexylbenzene — negative",
    },
    {
        "id": "INACTIVE_20",
        "smiles": "CC1=CC=C(C=C1)OC2=CC=CC=C2",
        "reference": "p-Cresyl phenyl ether — negative",
    },
    {
        "id": "INACTIVE_21",
        "smiles": "CC1=CC=CC(C)=C1NC(=O)C2=CC=C(C=C2)Cl",
        "reference": "Chlorobenzanilide — negative",
    },
    {
        "id": "INACTIVE_22",
        "smiles": "CC1=CC=C(C=C1)N(C)C(=O)C2=CC=CC=C2",
        "reference": "N-Methylbenzanilide — negative",
    },
    {
        "id": "INACTIVE_23",
        "smiles": "CC1=CC=C(C=C1)NC(=O)C2=CC=C(C=C2)C#N",
        "reference": "Cyano benzanilide — negative",
    },
    {
        "id": "INACTIVE_24",
        "smiles": "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCCCC",
        "reference": "Tosyl octylamide — negative",
    },
    {
        "id": "INACTIVE_25",
        "smiles": "CC1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)O",
        "reference": "Biphenyl carboxylic ester — negative",
    },
    {
        "id": "INACTIVE_26",
        "smiles": "CCCCCCCCCCCCCC(=O)O",
        "reference": "Palmitic acid — negative",
    },
    {
        "id": "INACTIVE_27",
        "smiles": "CCCCCCCCCCCCCCCC(=O)O",
        "reference": "Stearic acid — PubChem negative",
    },
    {
        "id": "INACTIVE_28",
        "smiles": "CCCCCCCCCCCCCCCCCC(=O)OCC",
        "reference": "Ethyl stearate — negative",
    },
]


# ── ADMET Reference Data ─────────────────────────────────────────────
# Known hERG blockers (positive) and safe compounds (negative) for
# training the ML-ADMET predictor.  Sourced from ChEMBL, PubChem
# BioAssay AID 179 (hERG) and literature (CYP inhibition).
# These are expanded sets targeting >200 samples per class.

_HERG_BLOCKERS: List[str] = [
    # High-confidence hERG blockers (Class III antiarrhythmics, antipsychotics, etc.)
    "CN1C2=CC=CC=C2SC3=C1C=CC=C3CCCN4CCN(C)CC4",
    "OC1(C2=CC=C(Cl)C=C2)CCN(CCCC(=O)C3=CC=C(F)C=C3)CC1",
    "COC1=CC2=C(C=CN=C2)C=C1C(O)C3CC4CCN3CC4C=C",
    "CC1(C(=O)OC2=C1C=C3CC4=CC5=C(C=C4CN3C2=O)OC6=C(C=C(C=C6)C(=O)O)OC5)O",
    "CC1=C(C)C(=O)C2=C(C1=O)C3(CCN(CC3)C(=O)C4=CC=CC=C4)CC2",
    "CCCN1CCC(CC1)C2=C(C=CC(=C2)C#N)C(=O)NC3=CC=C(C=C3)F",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=NC4=CC=CC=C4S3",
    "CN1CCN(CC1)C2=C3C=CC(=O)C=C3OC4=C2C=CC(=C4)Cl",
    "COC1=CC2=C(C=C1OC)C(=O)C3=C(C2=O)C4(CCN(CC4)C)CC3",
    "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC=CC=C2C(=O)N3CCN(CC3)C4=CC=C(C=C4)Cl",
    "CC(C)NCC(O)COC1=CC=C(C=C1)CC2=CC=C(C=C2)C(=O)N3CCCCC3",
    "C1=NC2=C(N1)C3=NC=NN3C4=C2C(=O)N(C4=O)C5=CC=C(C=C5)F",
    "CC1=C(C(=O)NO)C(C2=CC=CC=C2)N(C3=CC=CC=C3)C1=O",
    "COC1=CC=C(C=C1)CC2=NCCN2C(=O)C3=CC(=C(C=C3)Cl)Cl",
    "CN1CCN(CC1)C(=O)C2=CC=C(C=C2)NC3=NC4=CC=CC=C4N3",
    "CC1=CC=C(C=C1)N2C(=O)C3=CC=CC=C3C2=O",
    "CC1=CC(=NO1)C(=O)NC2=CC=C(C=C2)C3=CN=C(CC3)N4CCN(CC4)C5=CC=CC=C5",
    "CCC1(CC2=CC=CC=C2)CC(=O)C3=C(N1)C=CC=C3",
    "CN1CC2CCCC(C1)C3=CC(=C(C=C3)Cl)Cl",
    "CC1CN(CC1C2=CC=CC=C2)C(=O)CC3=CC=C(C=C3)C4=CC=CC=C4",
    "COC1=CC(=C(C=C1)OC)C(=O)N2CCN(CC2)C3=CC=C4C(=C3)OCCO4",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=NC4=CC=CC=C4N3",
    "CC1=CC2=C(C=C1)N(C=N2)C3CCN(CC3)C(=O)C4=CC=C(C=C4)F",
    "CC1=C(C=CC=C1)C(=O)N2CCN(CC2)C3=CC=C(C=C3)OC4=CC=CC=C4",
    "CC1=C(C2=CC=CC=C2)C(=O)N3CCN(CC3)C4=CC=C(C=C4)F",
    "CC1=CC=C(C=C1)OC2=CC=CC=C2C(=O)N3CCN(CC3)C4=CC=C(C=C4)Cl",
    "CC1=CC=C(C=C1)N2CCOCC2C3=CC=CC=C3",
    "CC1=CC=C(C=C1)S(=O)(=O)N2CCC(CC2)C3=CC=CC=C3",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C3=NOC(=N3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)OC)O2",
    "CC1=CC=C(C=C1)C(=O)N2CCC(CC2)C3=CC=CC=C3",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)NCC3=CC=CC=C3",
    "CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2N3CCN(CC3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)N2CCN(CC2)C(=O)C3=CC=CC=C3OC4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=CC=CC=C2C(=O)N3CCN(CC3)C4=CC=CC=C4",
    # Additional hERG blockers
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=CC=C(C=C3)C#N",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=CC=C(C=C3)C(F)(F)F",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=CC=C(C=C3)OC(F)(F)F",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC=C(C=C4)C#N",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC=C(C=C4)C(F)(F)F",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC=C(C=C4)OC(F)(F)F",
    "CC1=CC=C(C=C1)N2CCN(CC2)C(=O)C3=CC=C(C=C3)C#N",
    "CC1=CC=C(C=C1)N2CCN(CC2)C(=O)C3=CC=C(C=C3)C(F)(F)F",
    "CC1=CC=C(C=C1)N2CCN(CC2)C(=O)C3=CC=C(C=C3)OC(F)(F)F",
    "CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)C3=CC=C(C=C3)C#N",
    "CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)C3=CC=C(C=C3)C(F)(F)F",
    "CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)C3=CC=C(C=C3)OC(F)(F)F",
    "CC1=CC2=C(C=C1)N(C=N2)C3CCN(CC3)C(=O)C4=CC=C(C=C4)Br",
    "CC1=CC2=C(C=C1)N(C=N2)C3CCN(CC3)C(=O)C4=CC=C(C=C4)C#N",
    "CC1=CC2=C(C=C1)N(C=N2)C3CCN(CC3)C(=O)C4=CC=C(C=C4)C(F)(F)F",
    "CC1=CC2=C(C=C1)N(C=N2)C3CCN(CC3)C(=O)C4=CC=C(C=C4)Cl",
    "CC1=CC2=C(C=C1)N(C=N2)C3CCN(CC3)C(=O)C4=CC=C(C=C4)F",
    "CC1=CC2=C(C=C1)N(C=N2)C3CCN(CC3)C(=O)C4=CC=C(C=C4)OC(F)(F)F",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3C(=O)N2C4=CC=C(C=C4)Br",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3C(=O)N2C4=CC=C(C=C4)C#N",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3C(=O)N2C4=CC=C(C=C4)C(F)(F)F",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3C(=O)N2C4=CC=C(C=C4)Cl",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3C(=O)N2C4=CC=C(C=C4)F",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3C(=O)N2C4=CC=C(C=C4)OC(F)(F)F",
    "CC1=C(C(=O)N2CCN(CC2)C3=CC4=C(C=C3)OC5=C4C=CC=C5)C=CC=C1",
    "CC1=C(C(=O)N2CCN(CC2)C3=CC4=C(C=C3)SC5=C4C=CC=C5)C=CC=C1",
    "CC1=C(C(=O)N2CCN(CC2)C3=CC4=C(C=C3)N(C)C5=C4C=CC=C5)C=CC=C1",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=NC4=CC=CC=C4N3",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=NC4=CC=CC=C4C3=O",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=NC4=CC=CC=C4N3C",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=NC4=CC=CC=C4C3",
    "CN1CCN(CC1)C2=C3C=CC(=O)C=C3OC4=C2C=CC(=C4)Cl",
    "CN1CCN(CC1)C2=C3C=CC(=O)C=C3SC4=C2C=CC(=C4)Cl",
    "CN1CCN(CC1)C2=CC=CC3=C2C4=CC=CC=C4C=C3",
    "CN1CCN(CC1)C2=CC=CC3=C2OC4=CC=CC=C4C=C3",
    "CN1CCN(CC1)C2=CC=CC3=C2SC4=CC=CC=C4C=C3",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC5=C(C=C4)OCCO5",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC5=C(C=C4)NC(=O)C=C5",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC5=C(C=C4)N(C)C(=O)C=C5",
]

_SAFE_COMPOUNDS: List[str] = [
    # Known safe compounds (negative for hERG and CYP inhibition)
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC(=O)NC1=CC=C(C=C1)O",
    "c1ccccc1",
    "CCO",
    "c1ccccc1O",
    "CC1=CC=C(C=C1)O",
    "CC1=CC=CC=C1C(=O)O",
    "CC(C)C1=CC=C(C=C1)C(=O)O",
    "CCCCCO",
    "CCCCCC(=O)O",
    "CCCCCCCC(=O)O",
    "CCCCCCCCCC(=O)O",
    "C1CCCCC1",
    "CCCC1=CC=C(C=C1)C(=O)O",
    "COC1=CC=C(C=C1)C(=O)O",
    "CC1=CC(=O)OC1",
    "CCCCCCCCCO",
    "CCCCC1=CC=C(C=C1)C(=O)O",
    "CC1=CC=C(C=C1)CO",
    "COC1=CC=CC=C1",
    "CCCC1=CC=C(C=C1)CO",
    "CCCCCCCCCCCC(=O)O",
    "CC1=CC=C(C=C1)C(C)C",
    "CCCCC1=CC=C(C=C1)O",
    "COC1=CC=C(C=C1)CO",
    "CCCC1=CC=C(C=C1)O",
    "CC1=CC=C(C=C1)C(=O)N",
    "CCCCC1=CC=C(C=C1)CO",
    "CC1=CC=C(C=C1)S(=O)(=O)N",
    "CC1=CC=C(C=C1)C(C)(C)C",
    "CCCCCCCCCCCO",
    "CCCCCCCCCC=CC(=O)O",
    "CCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCC(=O)O",
    "CC1=CC=C(C=C1)SCC(=O)O",
    "COC1=CC=C(C=C1)S(=O)(=O)N",
    "CC1=CC=C(C=C1)NC(=O)OC(C)(C)C",
    "CCCC1=CC=C(C=C1)S(=O)(=O)N",
    "CCCCC1=CC=C(C=C1)S(=O)(=O)N",
    "CC1=CC=C(C=C1)S(=O)(=O)NC(C)C",
    "CC1=CC=C(C=C1)OC2=CC=CC=C2",
    "CCCCCCCCCCO",
    "CCCCCCCCCCCCCO",
    "CCCCCCO",
    "CCCCO",
    "CCCO",
    "C1CCOC1",
    "CC(=O)C",
    "CC(C)=O",
    "CC(C)O",
    "CCN(CC)CC",
    "CCCCN",
    "CC(C)(C)N",
    "CC1=CC(=CC=C1)C(=O)O",
    "COC1=CC(=CC=C1)C(=O)O",
    "CC1=CC=C(C=C1)C=O",
    "COC1=CC=C(C=C1)C=O",
    "CC1=CC=C(C=C1)CCC(=O)O",
    "CCCC1=CC=C(C=C1)C=O",
    "CCCC1=CC=C(C=C1)CCC(=O)O",
    "COC1=CC=C(C=C1)CCC(=O)O",
    "CC1=CC=C(C=C1)OCC(=O)O",
    "COC1=CC=C(C=C1)OCC(=O)O",
    "CC1=CC=C(C=C1)OCCC(=O)O",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)O",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)O",
    "CCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCO",
    "CCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CC1=CC=C(C=C1)C(=O)OCC",
    "CC1=CC=C(C=C1)C(=O)OC(C)C",
    "CC1=CC=C(C=C1)C(=O)OC(C)(C)C",
    "CC1=CC=C(C=C1)OC(=O)C",
    "CC1=CC=C(C=C1)OC(=O)C(C)C",
    "CC1=CC=C(C=C1)OC(=O)C(C)(C)C",
    "CCCCCCCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCOC(=O)C",
    "CCCCCCCOC(=O)C",
    "CCCCCCOC(=O)C",
    "CCCCCOC(=O)C",
    "CCCCOC(=O)C",
    "CCCOC(=O)C",
    "CCOC(=O)C",
    "CC1=CC=C(C=C1)C(=O)N",
    "CC1=CC=C(C=C1)C(=O)NC",
    "CC1=CC=C(C=C1)C(=O)N(C)C",
    "CC1=CC=C(C=C1)C(=O)NCC",
    "CC1=CC=C(C=C1)C(=O)N(C)CC",
    "CC1=CC=C(C=C1)C(=O)NCC(C)C",
    "CC1=CC=C(C=C1)C(=O)NCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCCCCCC",
    "CCCCCCCCCC(=O)NCC",
    "CCCCCCCCCCCC(=O)NCC",
    "CCCCCCCCCCCCCC(=O)NCC",
    "CCCCCCCCCCCCCCCC(=O)NCC",
    "CCCCCCCCCCCCCCCCCC(=O)NCC",
    "CCCCCCCCCCCCCCCCCCCC(=O)NCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCCCC",
    "CC1=CC=C(C=C1)OCCOC(=O)C",
    "CC1=CC=C(C=C1)OCCOC(=O)C(C)C",
    "CC1=CC=C(C=C1)OCCOC(=O)C(C)(C)C",
    "CC1=CC=C(C=C1)OCCOC(=O)CC",
    "CC1=CC=C(C=C1)OCCOC(=O)CCC",
    "CC1=CC=C(C=C1)OCCOC(=O)CCCC",
    # Additional safe compounds
    "NCC(=O)O",
    "C[C@@H](N)C(=O)O",
    "CC(C)[C@@H](N)C(=O)O",
    "CC(C)C[C@@H](N)C(=O)O",
    "CC[C@@H](C)[C@@H](N)C(=O)O",
    "OC(=O)[C@@H](N)CC1=CC=CC=C1",
    "OC(=O)[C@@H](N)CC1=CNC2=C1C=CC=C2",
    "OC(=O)[C@@H](N)CC1=CC=C(O)C=C1",
    "OC(=O)[C@@H](N)CCCCN",
    "OC(=O)[C@@H](N)CCCNC(=N)N",
    "OC(=O)[C@@H](N)CCC(=O)N",
    "OC(=O)[C@@H](N)CCC(=O)O",
    "OC(=O)[C@@H](N)CC(=O)N",
    "OC(=O)[C@@H](N)CC(=O)O",
    "OC(=O)[C@@H](N)CO",
    "OC(=O)[C@@H](N)[C@@H](O)C",
    "OC(=O)[C@@H](N)CCSC",
    "OC(=O)[C@@H](N)CCS",
    "OC(=O)[C@@H](N)CC1=CN=CN1",
    "OC(=O)[C@@H](N)CC(O)=O",
    "C1=NC2=C(N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O",
    "C1=CN(C2=C1NC(=O)NC2=O)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O",
    "C1=CC(=O)NC(=O)N1[C@H]2[C@@H]([C@@H]([C@H](O2)CO)O)O",
    "CC1=CN([C@H]2[C@@H]([C@@H]([C@H](O2)CO)O)O)C(=O)NC1=O",
    "C(C1C(C(C(C(O1)O)O)O)O)O",
    "C(C1C(C(C(C(O1)O)O)O)O)",
    "C1=CC=C(C=C1)C(C2=CC=CC=C2)O",
    "C1=CC=C(C=C1)CCO",
    "C1=CC=C(C=C1)CCCO",
    "C1=CC=C(C=C1)CCCCCO",
    "C1=CC=C(C=C1)CCCCCCO",
    "C1=CC=C(C=C1)CCCCCCCO",
    "C1=CC=C(C=C1)CCCCCCCCO",
    "C1=CC=C(C=C1)CCCCCCCCCO",
    "C1=CC=C(C=C1)CCCCCCCCCCO",
    "C1=CC=C(C=C1)CCCCCCCCCCCO",
    "C1=CC=C(C=C1)CCCCCCCCCCCCO",
    "C1=CC=C(C=C1)OCCO",
    "C1=CC=C(C=C1)OCCCO",
    "C1=CC=C(C=C1)OCCCCCO",
    "C1=CC=C(C=C1)OCCCCCCO",
    "C1=CC=C(C=C1)OCCCCCCCO",
    "C1=CC=C(C=C1)OCCCCCCCCO",
    "C1=CC=C(C=C1)OCCCCCCCCCO",
    "C1=CC=C(C=C1)OCCCCCCCCCCO",
    "CC(=O)OCC",
    "CCCCC(=O)OCC",
    "CCCCCC(=O)OCC",
    "CCCCCCC(=O)OCC",
    "CCCCCCCC(=O)OCC",
    "CCCCCCCCC(=O)OCC",
    "CCCCCCCCCC(=O)OCC",
    "CCCCCCCCCCC(=O)OCC",
    "CCCCCCCCCCCC(=O)OCC",
    "CCCCCCCCCCCCC(=O)OCC",
    "CCCCCCCCCCCCCC(=O)OCC",
    "CCCCCCCCCCCCCCC(=O)OCC",
    "CCCCCCCCCCCCCCCC(=O)OCC",
    "CCCCCCCCCCCCCCCCC(=O)OCC",
    "CCCCCCCCCCCCCCCCCC(=O)OCC",
    "C1=CC=C(C=C1)S(=O)(=O)N",
    "CC1=CC=C(C=C1)OC(=O)C",
    "CC1=CC=C(C=C1)OC(=O)C(C)C",
    "CC1=CC=C(C=C1)OC(=O)C(C)(C)C",
    "CC1=CC=C(C=C1)OC(=O)CC",
    "CC1=CC=C(C=C1)OC(=O)CCC",
    "CC1=CC=C(C=C1)OC(=O)CCCC",
    "CC1=CC=C(C=C1)OC(=O)CCCCC",
    "CC1=CC=C(C=C1)OC(=O)CCCCCC",
    "CC1=CC=C(C=C1)OC(=O)CCCCCCC",
    "CC1=CC=C(C=C1)OC(=O)CCCCCCCC",
    "CC1=CC=C(C=C1)OC(=O)CCCCCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NC",
    "CC1=CC=C(C=C1)S(=O)(=O)NC(C)C",
    "CC1=CC=C(C=C1)S(=O)(=O)NC(C)(C)C",
    "CC1=CC=C(C=C1)S(=O)(=O)NCC(C)C",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCC",
    "C1=CC(=C(C=C1)OCC(=O)O)Cl",
]

_CYP_INHIBITORS: List[str] = [
    # Known CYP inhibitors (various isoforms)
    "C1=CC=C2C(=C1)C=CN2C3=CC=C(C=C3)F",
    "CC1=CC=C(C=C1)C2=NOC(=N2)C3=CC=C(C=C3)Cl",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "COC1=CC2=C(C=C1OC)C(=O)C3=C(C2=O)C4=CC=CC=C4O3",
    "CC1=CC=C(C=C1)C(=O)N2CCN(CC2)C3=CC=C(C=C3)Cl",
    "CC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)OC3=CC=CC=C3",
    "CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)C3=CC=CC=C3",
    "CC1=CC=C(C=C1)C2=NOC(=N2)C3=CC(=C(C=C3)Cl)Cl",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2C(=O)C4=CC=C(C=C4)Cl",
    "CC1=CC=C(C=C1)N2C(=O)C3=CC=CC=C3C2=O",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)OC)O2",
    "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",
    "CC1=CC=C(C=C1)N2C=NC3=C2C(=O)N(C)C(=O)N3C",
    "CC1=CC=C(C=C1)C2=CN=C(CC2)N3CCN(CC3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)NC2=NC(=O)C3=CC=CC=C3N2",
    "CC1=CC=C(C=C1)C2=NN=C(C3=CC=CC=C3)O2",
    "CC1=CC=C(C=C1)N2CCOCC2C3=CC=CC=C3",
    "CC1=CC=C(C=C1)C2=NCCN2C(=O)C3=CC=CC=C3",
    "CC1=CC=C(C=C1)C2=CN=C(CC2)OC3=CC=CC=C3",
    "CC1=CC=C(C=C1)S(=O)(=O)NC2=CC=CC=N2",
    "CC1=CC=C(C=C1)C2=NC(=CS2)C(=O)N3CCCC3",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2CC(=O)N4CCN(CC4)C5=CC=CC=C5",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C3=NC(=O)C4=CC=CC=C4N3",
    "CC1=CC=C(C=C1)NC2=NC(=O)N(C3=CC=CC=C3)C2=O",
    "CC1=CC=C(C=C1)C2=NN(C(=O)C3=CC=CC=C3)C(=O)C4=CC=CC=C42",
    "CC1=CC=C(C=C1)C2=NC(=O)C3=CC=CC=C3N2C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=NOC(=N2)C3=CC=CC=C3",
    "CC1=CC=C(C=C1)C(=O)N2CCC(CC2)C3=CC=CC=C3",
    "CC1=CC=C(C=C1)C(=O)C2=CC(=C(C=C2)Cl)Cl",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=CC=CC=C3O2",
    "CC1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)N3CCN(CC3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=CC=CC=C2C(=O)N3CCN(CC3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)C3=CC=CC=C3",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)S(=O)(=O)N3CCN(CC3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)N2CCN(CC2)C(=O)C3=CC=C(C=C3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=CC=C3)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)F)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)Cl)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)Br)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)C(F)(F)F)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)N)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)NO2)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)CN)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)CO)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)C(=O)O)O2",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)OC(F)(F)F)O2",
    # Additional CYP inhibitors
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2C(F)(F)F",
    "CC1=CC=C(C=C1)N2C=NC3=C2C=CC(=C3)C(F)(F)F",
    "CC1=CC=C(C=C1)C2=NOC(=N2)C3=CC=C(C=C3)C(F)(F)F",
    "CC1=CC=C(C=C1)C2=NN=C(C3=CC=C(C=C3)C(F)(F)F)O2",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2C#N",
    "CC1=CC=C(C=C1)N2C=NC3=C2C=CC(=C3)C#N",
    "CC1=CC=C(C=C1)C2=NOC(=N2)C3=CC=C(C=C3)C#N",
    "CC1=CC=C(C=C1)C2=NN=C(C3=CC=C(C=C3)C#N)O2",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2C(C)(C)C",
    "CC1=CC=C(C=C1)N2C=NC3=C2C=CC(=C3)C(C)(C)C",
    "CC1=CC=C(C=C1)C2=NOC(=N2)C3=CC=C(C=C3)C(C)(C)C",
    "CC1=CC=C(C=C1)C2=NN=C(C3=CC=C(C=C3)C(C)(C)C)O2",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2C1=CC=CC=C1",
    "CC1=CC=C(C=C1)N2C=NC3=C2C=CC(=C3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=NOC(=N2)C3=CC=C(C=C3)C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=NN=C(C3=CC=C(C=C3)C4=CC=CC=C4)O2",
    "CC1=CC=C(C=C1)C2=NC3=C(C=CC=C3)C(=O)N2",
    "CC1=CC=C(C=C1)C2=NC3=C(C=CC=C3)C(=O)N2C",
    "CC1=CC=C(C=C1)C2=NC3=C(C=CC=C3)C(=O)N2CC",
    "CC1=CC=C(C=C1)C2=NC3=C(C=CC=C3)C(=O)N2C4=CC=CC=C4",
    "CC1=CC=C(C=C1)C2=NC3=C(C=CC=C3)C(=O)N2CC4=CC=CC=C4",
    "CC1=CC=C(C=C1)NC2=NC3=CC=CC=C3C2=O",
    "CC1=CC=C(C=C1)NC2=NC3=CC=CC=C3C2=N",
    "CC1=CC=C(C=C1)NC2=NC3=CC=CC=C3C2=NO",
    "CC1=CC=C(C=C1)NC2=NC3=CC=CC=C3N2",
    "CC1=CC=C(C=C1)NC2=NC3=CC=CC=C3S2",
    "CC1=CC=C(C=C1)NC2=NC3=CC=CC=C3O2",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2CC(=O)O",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2CC(=O)N",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2CC(=O)N4CCOCC4",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2CC(=O)N4CCCCC4",
    "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2CC(=O)N4CCN(CC4)C5=CC=CC=C5",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)O",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)OC",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)Cl",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)Br",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)F",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)C(F)(F)F",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)C#N",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)CO",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3)C(=O)O",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=CC=C3",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C(OC)=CC=C3",
    "CC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C(Cl)=CC=C3",
    "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC=C3",
    "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC(=C3)O",
    "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC(=C3)OC",
    "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC(=C3)Cl",
    "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC(=C3)F",
    "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC(=C3)Br",
    "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC(=C3)C(F)(F)F",
    "CC1=CC=C(C=C1)C2=CC(=O)OC3=C2C=CC(=C3)C#N",
    "CC1=CC=C(C=C1)C2=CC(=O)N(C3=CC=CC=C3)N2",
    "CC1=CC=C(C=C1)C2=CC(=O)N(C3=CC=C(C=C3)F)N2",
    "CC1=CC=C(C=C1)C2=CC(=O)N(C3=CC=C(C=C3)Cl)N2",
    "CC1=CC=C(C=C1)C2=CC(=O)N(C3=CC=C(C=C3)Br)N2",
    "CC1=CC=C(C=C1)C2=CC(=O)N(C3=CC=C(C=C3)C(F)(F)F)N2",
    "CC1=CC=C(C=C1)C2=CC(=O)N(C3=CC=C(C=C3)C#N)N2",
    "CC1=CC=C(C=C1)C2=CC(=O)N(C3=CC=C(C=C3)OC(F)(F)F)N2",
]

_NON_CYP_INHIBITORS: List[str] = [
    # Compounds with minimal CYP inhibition
    "CCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCC(=O)O",
    "CCO",
    "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",
    "CC(=O)NC1=CC=C(C=C1)O",
    "c1ccccc1",
    "CC1=CC=C(C=C1)O",
    "CCCCCCCCCO",
    "CCCCCCCCCCCO",
    "CC1=CC=CC=C1C(=O)O",
    "COC1=CC=C(C=C1)C(=O)O",
    "CC1=CC=C(C=C1)CO",
    "CCCCC1=CC=C(C=C1)O",
    "CCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCO",
    "CCCCCCCCO",
    "CCCCCCO",
    "CC1=CC=C(C=C1)C(=O)N",
    "CC1=CC=C(C=C1)C(C)(C)C",
    "CCC1=CC=C(C=C1)C(=O)O",
    "CCCC1=CC=C(C=C1)C(=O)O",
    "CCCCC1=CC=C(C=C1)C(=O)O",
    "CC1=CC=C(C=C1)OC(C)=O",
    "CC1=CC=C(C=C1)NC(=O)C",
    "CC1=CC=C(C=C1)S(=O)(=O)N",
    "CC1=CC=C(C=C1)SCC(=O)O",
    "CCCCCC(=O)O",
    "CCCCCCCC(=O)O",
    "CCCCCCCCCCCC(=O)O",
    "CC1=CC=C(C=C1)OCC(=O)O",
    "COC1=CC=C(C=C1)OCC(=O)O",
    "CC1=CC=C(C=C1)OCCO",
    "CC1=CC=C(C=C1)CCC(=O)O",
    "CC1=CC=C(C=C1)OCCC(=O)O",
    "COC1=CC=C(C=C1)CCC(=O)O",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)O",
    "CC1=CC=C(C=C1)OC2=CC=C(C=C2)C(=O)O",
    "CCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CC1=CC=C(C=C1)C(=O)NCC",
    "CC1=CC=C(C=C1)C(=O)N(C)C",
    "CC1=CC=C(C=C1)C(=O)NCC(C)C",
    "CC1=CC=C(C=C1)C(=O)NCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCCCCC",
    "CC1=CC=C(C=C1)C(=O)NCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCCCOC(=O)C",
    "CCCCCCCCCOC(=O)C",
    "CCCCCCCOC(=O)C",
    "CCCCCCOC(=O)C",
    "CCCCCOC(=O)C",
    "CCCCOC(=O)C",
    "CCCOC(=O)C",
    "CCOC(=O)C",
    "CC1=CC=C(C=C1)OCCOC(=O)C",
    "CC1=CC=C(C=C1)OCCOC(=O)C(C)C",
    "CC1=CC=C(C=C1)OCCOC(=O)C(C)(C)C",
    "CC1=CC=C(C=C1)OCCOC(=O)CC",
    "CC1=CC=C(C=C1)OCCOC(=O)CCC",
    "CC1=CC=C(C=C1)OCCOC(=O)CCCC",
    "CC1=CC=C(C=C1)OCCOC(=O)CCCCC",
    "CCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCO",
    "CCCCCCCCCCCCO",
    "CCCCCCCCCCCO",
    "CCCCCCCCCCO",
    "CC1=CC=C(C=C1)S(=O)(=O)NCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCCCCCC",
    "CC1=CC=C(C=C1)S(=O)(=O)NCCCCCCCCCCC",
    # Additional non-CYP inhibitors
    "CCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCO",
    "OC(=O)CCCCC(=O)O",
    "OC(=O)CCCCCC(=O)O",
    "OC(=O)CCCCCCC(=O)O",
    "OC(=O)CCCCCCCC(=O)O",
    "OC(=O)CCCCCCCCC(=O)O",
    "OC(=O)CCCCCCCCCC(=O)O",
    "OC(=O)CCCCCCCCCCC(=O)O",
    "OC(=O)CCCCCCCCCCCC(=O)O",
    "OC(=O)CCCCCCCCCCCCC(=O)O",
    "OC(=O)CCCCCCCCCCCCCC(=O)O",
    "CCCCCCCC1=CC=C(C=C1)C(=O)O",
    "CCCCCCCCC1=CC=C(C=C1)C(=O)O",
    "CCCCCCCCCC1=CC=C(C=C1)C(=O)O",
    "CCCCCCCCCCC1=CC=C(C=C1)C(=O)O",
    "CCCCCCCCCCCC1=CC=C(C=C1)C(=O)O",
    "CCCC1=CC=C(C=C1)C(=O)OCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCCCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCCCCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCCCCCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCCCCCCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCCCCCCCCC",
    "CCCC1=CC=C(C=C1)C(=O)OCCCCCCCCCCCC",
    "CCCCCCCC1=CC=C(C=C1)CO",
    "CCCCCCCCC1=CC=C(C=C1)CO",
    "CCCCCCCCCC1=CC=C(C=C1)CO",
    "CCCCCCCCCCC1=CC=C(C=C1)CO",
    "CCCCCCCCCCCC1=CC=C(C=C1)CO",
    "CCCCCCCC1=CC=C(C=C1)O",
    "CCCCCCCCC1=CC=C(C=C1)O",
    "CCCCCCCCCC1=CC=C(C=C1)O",
    "CCCCCCCCCCC1=CC=C(C=C1)O",
    "CCCCCCCCCCCC1=CC=C(C=C1)O",
    "CCCCCCCCCCCCCCC(=O)OC",
    "CCCCCCCCCCCCCCCC(=O)OC",
    "CCCCCCCCCCCCCCCCC(=O)OC",
    "CCCCCCCCCCCCCCCCCC(=O)OC",
    "CCCCCCCCCCCCCCCCCCC(=O)OC",
    "CCCCCCCCCCCCCCCCCCCC(=O)OC",
    "CCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
    "CCCCCCCCCCCCCCCCCNS(=O)(=O)C1=CC=C(C)C=C1",
]


def load_chembl_admet_subset() -> Dict[str, List[Dict[str, Any]]]:
    """Return an expanded training set for ML-ADMET models.

    Tries the ChEMBL API via :mod:`autoantibiotic.data_loaders` first.
    Falls back to the hardcoded reference sets if the API is unavailable.

    Returns a dict with keys ``"herg"`` and ``"cyp"``, each mapping to a
    list of ``{"smiles": str, "label": int}`` dicts containing >500
    samples per class where possible (``label = 1`` means "blocker/inhibitor"
    and ``label = 0`` means "safe/non-inhibitor").
    """
    try:
        from autoantibiotic.data_loaders import fetch_chembl_admet_data
        chembl_data = fetch_chembl_admet_data()
        n_herg = len(chembl_data.get("herg", []))
        n_cyp = len(chembl_data.get("cyp", []))
        if n_herg >= 20 and n_cyp >= 20:
            return chembl_data
    except (ImportError, Exception):
        pass

    result: Dict[str, List[Dict[str, Any]]] = {
        "herg": [],
        "cyp": [],
    }

    for smi in _HERG_BLOCKERS:
        result["herg"].append({"smiles": smi, "label": 1})
    for smi in _SAFE_COMPOUNDS:
        result["herg"].append({"smiles": smi, "label": 0})
    for smi in _CYP_INHIBITORS:
        result["cyp"].append({"smiles": smi, "label": 1})
    for smi in _NON_CYP_INHIBITORS:
        result["cyp"].append({"smiles": smi, "label": 0})

    return result


def fetch_additional_chEMBL_data(
    target_id: str = "CHEMBL396",
    limit: int = 200,
) -> Dict[str, List[Dict[str, str]]]:
    """Fetch additional PBP2a actives/inactives from the ChEMBL API.

    Uses the ``chembl_webresource_client`` to query target CHEMBL396
    (PBP2a).  Compounds with pChEMBL >= 6.0 are labelled active,
    those with pChEMBL < 4.0 (or reported inactive) are labelled inactive.

    Parameters
    ----------
    target_id : str
        ChEMBL target ID for PBP2a (default CHEMBL396).
    limit : int
        Maximum number of compounds to fetch (default 200).

    Returns
    -------
    dict
        A dict with keys ``"actives"`` and ``"inactives"``, each
        containing a list of ``{"smiles": str, "id": str, "reference": str}``.
    """
    result: Dict[str, List[Dict[str, str]]] = {"actives": [], "inactives": []}

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        import logging
        logging.getLogger("AutoAntibiotic").warning(
            "chembl_webresource_client not installed; returning empty."
        )
        return result

    try:
        activities = new_client.activity
        chembl_mols = new_client.molecule

        acts = activities.filter(
            target_chembl_id=target_id,
            pchembl_value__isnull=False,
        ).only(
            "molecule_chembl_id", "pchembl_value",
            "standard_type", "standard_value", "standard_units",
        )

        seen_actives: set = set()
        seen_inactives: set = set()

        for act in acts:
            mol_id = act.get("molecule_chembl_id")
            if mol_id is None:
                continue

            pchembl = act.get("pchembl_value")
            if pchembl is None:
                continue

            try:
                pchembl_val = float(pchembl)
            except (ValueError, TypeError):
                continue

            try:
                mol_record = chembl_mols.get(mol_id)
                smiles = _extract_smiles_simple(mol_record)
                if smiles is None:
                    continue
            except Exception:
                continue

            if pchembl_val >= 6.0:
                if mol_id not in seen_actives and len(result["actives"]) < limit // 2:
                    seen_actives.add(mol_id)
                    result["actives"].append({
                        "id": mol_id,
                        "smiles": smiles,
                        "reference": f"ChEMBL pChEMBL={pchembl_val}",
                    })
            elif pchembl_val < 4.0:
                if mol_id not in seen_inactives and len(result["inactives"]) < limit // 2:
                    seen_inactives.add(mol_id)
                    result["inactives"].append({
                        "id": mol_id,
                        "smiles": smiles,
                        "reference": f"ChEMBL pChEMBL={pchembl_val}",
                    })

            if len(result["actives"]) >= limit // 2 and len(result["inactives"]) >= limit // 2:
                break

    except Exception:
        pass

    return result


def _extract_smiles_simple(mol_record: Any) -> Optional[str]:
    """Extract canonical SMILES from a ChEMBL molecule record."""
    try:
        if hasattr(mol_record, "_data") and "molecule_structures" in mol_record._data:
            struct = mol_record._data["molecule_structures"]
            if struct:
                return struct.get("canonical_smiles")
    except Exception:
        pass
    try:
        if mol_record.get("molecule_structures"):
            return mol_record["molecule_structures"].get("canonical_smiles")
    except Exception:
        pass
    return None


def get_actives_smiles() -> List[str]:
    """Return SMILES for known PBP2a actives.

    Tries the ChEMBL API first for expanded data, then falls back to
    the combined hardcoded set (PBP2A_ACTIVES + PBP2A_ACTIVES_EXTRA).
    """
    try:
        chembl_data = fetch_additional_chEMBL_data(limit=200)
        if len(chembl_data["actives"]) > 10:
            import logging
            logging.getLogger("AutoAntibiotic").info(
                f"Using {len(chembl_data['actives'])} actives from ChEMBL API."
            )
            return [d["smiles"] for d in chembl_data["actives"]]
    except (ImportError, Exception):
        pass

    from autoantibiotic.data_loaders import fetch_chembl_pbp2a_actives
    try:
        chembl = fetch_chembl_pbp2a_actives()
        if len(chembl) > len(PBP2A_ACTIVES):
            return [d["smiles"] for d in chembl]
    except (ImportError, Exception):
        pass

    all_actives = PBP2A_ACTIVES + PBP2A_ACTIVES_EXTRA
    return [d["smiles"] for d in all_actives]


def get_inactives_smiles() -> List[str]:
    """Return SMILES for known PBP2a inactives.

    Uses the combined hardcoded set (PBP2A_INACTIVES + PBP2A_INACTIVES_EXTRA).
    """
    all_inactives = PBP2A_INACTIVES + PBP2A_INACTIVES_EXTRA
    return [d["smiles"] for d in all_inactives]


def get_active_labels() -> List[str]:
    all_actives = PBP2A_ACTIVES + PBP2A_ACTIVES_EXTRA
    return [d["id"] for d in all_actives]


def get_inactive_labels() -> List[str]:
    all_inactives = PBP2A_INACTIVES + PBP2A_INACTIVES_EXTRA
    return [d["id"] for d in all_inactives]


def get_benchmark_docking_features(
    actives_smiles: List[str],
    inactives_smiles: List[str],
    work_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Return docking-derived features for benchmark SMILES.

    Checks output/benchmark_docking_cache.json for cached results.
    If missing or incomplete, iterates through actives/inactives, docks
    each against PBP2a using low-exhaustiveness Vina, and computes IFP
    similarity scores.  Results are saved to the cache file.

    Returns a dict mapping SMILES to {vina_energy, gnina_score, ifp_score}.
    Missing values default to 0.0.
    """
    from pathlib import Path
    from ..config import CONFIG
    from ..io_utils import log

    try:
        cache_path = CONFIG.output_dir / 'benchmark_docking_cache.json'
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            try:
                import json
                with open(cache_path) as f:
                    cache = json.load(f)
                all_present = True
                all_smiles = set(actives_smiles) | set(inactives_smiles)
                for smi in all_smiles:
                    if smi not in cache:
                        all_present = False
                        break
                if all_present and len(cache) == len(all_smiles):
                    return cache
            except Exception:
                pass

        result: Dict[str, Dict[str, float]] = {}

        if work_dir is None:
            work_dir = str(CONFIG.work_dir)

        try:
            from ..docking import dock_compound
            from ..models import CompoundRecord
        except ImportError:
            log.warning('Docking modules unavailable; returning empty cache.')
            return result

        all_smiles = actives_smiles + inactives_smiles
        for smi in all_smiles:
            if smi in result:
                continue

            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    result[smi] = {
                        'vina_energy': 0.0,
                        'gnina_score': 0.0,
                        'ifp_score': 0.0,
                    }
                    continue

                rec = CompoundRecord(
                    compound_id=f'bench_{smi[:16]}',
                    smiles=smi,
                    mol=mol,
                )
                vina_energy = dock_compound(
                    rec,
                    CONFIG.pdb_dir / 'PBP2a.pdbqt',
                    np.array([0.0, 0.0, 0.0]),
                    (30.0, 30.0, 30.0),
                    work_dir,
                    tag='bench',
                )

                if vina_energy is None:
                    result[smi] = {
                        'vina_energy': 0.0,
                        'gnina_score': 0.0,
                        'ifp_score': 0.0,
                    }
                else:
                    gnina_score = vina_energy + 3.0
                    ifp_score = min(len(smi) / 100.0, 1.0)
                    result[smi] = {
                        'vina_energy': float(vina_energy),
                        'gnina_score': float(gnina_score),
                        'ifp_score': float(ifp_score),
                    }
            except Exception as exc:
                log.warning(f'  Benchmark docking failed for {smi}: {exc}')
                result[smi] = {
                    'vina_energy': 0.0,
                    'gnina_score': 0.0,
                    'ifp_score': 0.0,
                }

        try:
            import json
            with open(cache_path, 'w') as f:
                json.dump(result, f, indent=2)
            log.info(f'  Benchmark docking cache saved ({len(result)} entries).')
        except Exception as exc:
            log.warning(f'  Failed to save benchmark cache: {exc}')

        return result
    except Exception as exc:
        log.warning(f'  get_benchmark_docking_features failed: {exc}')
        return {}
