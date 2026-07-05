from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .io_utils import log

try:
    from chembl_webresource_client.new_client import new_client
    _HAVE_CHEMBL = True
except ImportError:
    _HAVE_CHEMBL = False

PBP2A_CHEMBL_ID: str = "CHEMBL396"
"""ChEMBL target ID for PBP2a (Penicillin-binding protein 2a,
also known as PBP2a' or mecA product)."""


def fetch_chembl_pbp2a_actives(
    max_compounds: int = 500,
) -> List[Dict[str, Any]]:
    """Fetch known PBP2a inhibitors from ChEMBL via the web resource client.

    Returns a list of ``{"smiles": str, "id": str, "reference": str}``
    dicts for compounds with reported IC50 / Ki / Kd <= 10 µM against
    target CHEMBL396.

    Falls back to the hardcoded benchmark list if the API is unavailable
    or returns fewer than 10 compounds.
    """
    if not _HAVE_CHEMBL:
        log.warning("chembl_webresource_client not installed; falling back to hardcoded PBP2a data.")
        return _fallback_pbp2a_actives()

    try:
        activities = new_client.activity
        chembl_mols = new_client.molecule

        acts = activities.filter(
            target_chembl_id=PBP2A_CHEMBL_ID,
            pchembl_value__isnull=False,
        ).only(
            "molecule_chembl_id", "pchembl_value",
            "standard_type", "standard_value", "standard_units",
        )

        results: List[Dict[str, Any]] = []
        seen_mols: set = set()
        for act in acts:
            mol_id = act.get("molecule_chembl_id")
            if mol_id is None or mol_id in seen_mols:
                continue

            pchembl = act.get("pchembl_value")
            if pchembl is None or float(pchembl) < 5.0:
                continue

            try:
                mol_record = chembl_mols.get(mol_id)
                if mol_record is None:
                    continue
                smiles = _extract_smiles(mol_record)
                if smiles is None:
                    continue
            except Exception:
                continue

            seen_mols.add(mol_id)
            results.append({
                "id": mol_id,
                "smiles": smiles,
                "reference": f"ChEMBL pChEMBL={pchembl}",
            })

            if len(results) >= max_compounds:
                break

        if len(results) < 10:
            log.warning(
                f"ChEMBL returned only {len(results)} PBP2a actives; "
                "falling back to hardcoded data."
            )
            return _fallback_pbp2a_actives()

        log.info(f"Fetched {len(results)} PBP2a active compounds from ChEMBL.")
        return results

    except Exception as exc:
        log.warning(f"ChEMBL API call failed ({exc}); using hardcoded PBP2a data.")
        return _fallback_pbp2a_actives()


def _extract_smiles(mol_record: Any) -> Optional[str]:
    """Try to extract a canonical SMILES from a ChEMBL molecule record."""
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


def _fallback_pbp2a_actives() -> List[Dict[str, Any]]:
    from benchmarks.reference_data import PBP2A_ACTIVES
    return [
        {"id": d["id"], "smiles": d["smiles"], "reference": d["reference"]}
        for d in PBP2A_ACTIVES
    ]


def _fallback_pbp2a_inactives() -> List[Dict[str, Any]]:
    from benchmarks.reference_data import PBP2A_INACTIVES
    return [
        {"id": d["id"], "smiles": d["smiles"], "reference": d["reference"]}
        for d in PBP2A_INACTIVES
    ]


def fetch_chembl_admet_data() -> Dict[str, List[Dict[str, Any]]]:
    """Fetch hERG and CYP inhibition data from ChEMBL.

    Returns the same shape as ``load_chembl_admet_subset()``:
    ``{"herg": [...], "cyp": [...]}`` with >500 entries per class if
    the API is available.

    Falls back to the hardcoded benchmark data if the API fails.
    """
    if not _HAVE_CHEMBL:
        log.warning("chembl_webresource_client not installed; using hardcoded ADMET data.")
        return _fallback_admet_data()

    result: Dict[str, List[Dict[str, Any]]] = {"herg": [], "cyp": []}

    try:
        activities = new_client.activity
        chembl_mols = new_client.molecule

        herg_target = new_client.target.filter(
            pref_name__icontains="hERG",
        ).only("target_chembl_id")
        herg_target_id = herg_target[0]["target_chembl_id"] if herg_target else None

        if herg_target_id:
            herg_acts = activities.filter(
                target_chembl_id=herg_target_id,
                pchembl_value__isnull=False,
                standard_type__in=["IC50", "Ki"],
            ).only("molecule_chembl_id", "pchembl_value", "standard_relation")

            for act in herg_acts:
                mol_id = act.get("molecule_chembl_id")
                pchembl = act.get("pchembl_value")
                if mol_id is None or pchembl is None:
                    continue
                try:
                    mol_record = chembl_mols.get(mol_id)
                    smiles = _extract_smiles(mol_record) if mol_record else None
                    if smiles is None:
                        continue
                except Exception:
                    continue
                label = 1 if float(pchembl) >= 5.0 else 0
                result["herg"].append({"smiles": smiles, "label": label})
                if len(result["herg"]) >= 600:
                    break

    except Exception as exc:
        log.warning(f"ChEMBL hERG fetch failed ({exc}); using hardcoded ADMET data.")
        return _fallback_admet_data()

    try:
        cyp_targets = new_client.target.filter(
            pref_name__icontains="CYP",
        ).only("target_chembl_id")
        cyp_target_ids = [t["target_chembl_id"] for t in cyp_targets[:5]]

        if cyp_target_ids:
            cyp_acts = activities.filter(
                target_chembl_id__in=cyp_target_ids,
                pchembl_value__isnull=False,
            ).only("molecule_chembl_id", "pchembl_value")

            for act in cyp_acts:
                mol_id = act.get("molecule_chembl_id")
                pchembl = act.get("pchembl_value")
                if mol_id is None or pchembl is None:
                    continue
                try:
                    mol_record = chembl_mols.get(mol_id)
                    smiles = _extract_smiles(mol_record) if mol_record else None
                    if smiles is None:
                        continue
                except Exception:
                    continue
                label = 1 if float(pchembl) >= 5.0 else 0
                result["cyp"].append({"smiles": smiles, "label": label})
                if len(result["cyp"]) >= 600:
                    break

    except Exception as exc:
        log.warning(f"ChEMBL CYP fetch failed ({exc}); using hardcoded ADMET data.")
        return _fallback_admet_data()

    if len(result["herg"]) < 20 and len(result["cyp"]) < 20:
        log.warning("ChEMBL returned too few ADMET compounds; falling back to hardcoded data.")
        return _fallback_admet_data()

    log.info(
        f"Fetched {len(result['herg'])} hERG and {len(result['cyp'])} CYP "
        "compounds from ChEMBL."
    )
    return result


def _fallback_admet_data() -> Dict[str, List[Dict[str, Any]]]:
    from benchmarks.reference_data import load_chembl_admet_subset
    return load_chembl_admet_subset()
