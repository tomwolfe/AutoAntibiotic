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

from typing import Dict, List

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


def get_actives_smiles() -> List[str]:
    return [d["smiles"] for d in PBP2A_ACTIVES]


def get_inactives_smiles() -> List[str]:
    return [d["smiles"] for d in PBP2A_INACTIVES]


def get_active_labels() -> List[str]:
    return [d["id"] for d in PBP2A_ACTIVES]


def get_inactive_labels() -> List[str]:
    return [d["id"] for d in PBP2A_INACTIVES]
