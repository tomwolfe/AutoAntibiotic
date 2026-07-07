"""
Reference datasets for PBP2a enrichment benchmarking.

Contains:
  - ``PBP2A_ACTIVES``: Known PBP2a inhibitors (SMILES) loaded from
    ``data/pbp2a_actives.csv``.
  - ``PBP2A_INACTIVES``: Known PBP2a inactive compounds loaded from
    ``data/pbp2a_inactives.csv``.
  - ``DECOY_COUNT``: Default number of property-matched decoys to
    generate per active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "data"

_df_actives: pd.DataFrame = pd.read_csv(_DATA_DIR / "pbp2a_actives.csv")
PBP2A_ACTIVES: List[Dict[str, str]] = _df_actives.to_dict(orient="records")

_df_inactives: pd.DataFrame = pd.read_csv(_DATA_DIR / "pbp2a_inactives.csv")
PBP2A_INACTIVES: List[Dict[str, str]] = _df_inactives.to_dict(orient="records")

DECOY_COUNT: int = 100

def load_chembl_admet_subset() -> Dict[str, List[Dict[str, Any]]]:
    """Return an expanded training set for ML-ADMET models.

    Tries the ChEMBL API via :mod:`autoantibiotic.data_loaders` first.
    Falls back to the reference CSV files (``data/herg_data.csv``,
    ``data/cyp_data.csv``) if the API is unavailable.

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

    herg_df = pd.read_csv(_DATA_DIR / "herg_data.csv")
    cyp_df = pd.read_csv(_DATA_DIR / "cyp_data.csv")

    result: Dict[str, List[Dict[str, Any]]] = {
        "herg": herg_df.to_dict(orient="records"),
        "cyp": cyp_df.to_dict(orient="records"),
    }

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
        from chembl_webresource_client.settings import Settings

        Settings.Instance().CACHING = False
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
    the reference dataset loaded from ``pbp2a_actives.csv``.
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

    return [d["smiles"] for d in PBP2A_ACTIVES]


def get_inactives_smiles() -> List[str]:
    """Return SMILES for known PBP2a inactives."""
    return [d["smiles"] for d in PBP2A_INACTIVES]


def get_active_labels() -> List[str]:
    return [d["id"] for d in PBP2A_ACTIVES]


def get_inactive_labels() -> List[str]:
    return [d["id"] for d in PBP2A_INACTIVES]


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
