from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from .config import CONFIG, CompoundRecord
from .io_utils import log

_HAVE_GNINA: bool = False
_HAVE_RF_SCORE: bool = False
_HAVE_TRANSFORMERS: bool = False

try:
    result = subprocess.run(
        ["gnina", "--help"], capture_output=True, text=True, timeout=10,
    )
    _HAVE_GNINA = result.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
    pass

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    _HAVE_RF_SCORE = True
except ImportError:
    pass

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _HAVE_TRANSFORMERS = True
except ImportError:
    pass


def _compute_rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute a standard set of RDKit descriptors for ML scoring."""
    descs: List[float] = []
    for name, fn in Descriptors.descList:
        try:
            val = fn(mol)
            descs.append(val if val is not None else 0.0)
        except Exception:
            descs.append(0.0)
    return np.array(descs, dtype=np.float64)


def _rescore_with_gnina(
    top_candidates: List[CompoundRecord],
    receptor_pdbqt: str,
    work_dir: str,
) -> List[CompoundRecord]:
    """Rescore top candidates using GNINA if available.

    GNINA is a deep-learning-enhanced version of AutoDock Vina that uses
    convolutional neural networks for scoring.  This wrapper writes each
    candidate as a PDBQT, invokes ``gnina``, and parses the CNN score
    from the output.
    """
    cnn_scores: Dict[str, Optional[float]] = {}
    for rec in top_candidates:
        lig_dir = os.path.join(work_dir, f"gnina_{rec.compound_id}")
        os.makedirs(lig_dir, exist_ok=True)
        lig_pdbqt = os.path.join(lig_dir, "ligand.pdbqt")
        out_pdbqt = os.path.join(lig_dir, "out.pdbqt")

        try:
            from .docking import prepare_ligand_pdbqt
            if not prepare_ligand_pdbqt(rec.mol, lig_pdbqt):
                cnn_scores[rec.compound_id] = None
                continue

            cmd = [
                "gnina",
                "--receptor", receptor_pdbqt,
                "--ligand", lig_pdbqt,
                "--out", out_pdbqt,
                "--score_only",
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=CONFIG.vina_timeout_s,
            )
            stdout = proc.stdout + proc.stderr
            cnn_score = _parse_gnina_cnn_score(stdout)
            cnn_scores[rec.compound_id] = cnn_score
        except Exception as exc:
            log.warning(f"  GNINA rescoring failed for {rec.compound_id}: {exc}")
            cnn_scores[rec.compound_id] = None
        finally:
            for f in (lig_pdbqt, out_pdbqt):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(lig_dir)
            except OSError:
                pass

    for rec in top_candidates:
        rec.ml_score = cnn_scores.get(rec.compound_id)
    return top_candidates


def _parse_gnina_cnn_score(gnina_output: str) -> Optional[float]:
    """Parse the CNN affinity score from GNINA's output."""
    for line in gnina_output.splitlines():
        if "CNN score" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "score" and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        pass
        if "Affinity:" in line and "CNN" in line:
            m = line.split("Affinity:")[-1].strip().split()[0]
            try:
                return float(m)
            except ValueError:
                pass
    return None


def _compute_rf_features(mol: Chem.Mol) -> np.ndarray:
    """Compute a fixed-length feature vector for RF-Score-VS.

    Uses 200 rdkit descriptors (2D) plus Morgan fingerprint counts.
    """
    morgan = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=1024,
    )
    morgan_arr = np.array(morgan, dtype=np.float64)
    descs = _compute_rdkit_descriptors(mol)
    return np.concatenate([descs, morgan_arr])


def _train_rf_on_vina_data(
    candidates: List[CompoundRecord],
) -> Any:
    """Train a quick Random Forest regressor on Vina docking energies."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for rec in candidates:
        if rec.pb2pa_allosteric_energy is None or rec.mol is None:
            continue
        mol = rec.mol
        try:
            feats = _compute_rf_features(mol)
            X_list.append(feats)
            y_list.append(rec.pb2pa_allosteric_energy)
        except Exception:
            continue

    if len(X_list) < 10:
        return None

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=CONFIG.random_seed,
    )
    rf.fit(X_scaled, y)
    return scaler, rf


def _rescore_with_rf(
    top_candidates: List[CompoundRecord],
    model: Any,
) -> List[CompoundRecord]:
    """Rescore using a trained Random Forest model (RF-Score-VS style)."""
    if model is None:
        log.warning("  RF model not available for rescoring.")
        return top_candidates

    scaler, rf = model
    for rec in top_candidates:
        if rec.mol is None:
            continue
        try:
            feats = _compute_rf_features(rec.mol).reshape(1, -1)
            feats_scaled = scaler.transform(feats)
            pred = float(rf.predict(feats_scaled)[0])
            rec.ml_score = -abs(pred)  # normalise to negative (energy-like)
        except Exception as exc:
            log.warning(f"  RF rescoring failed for {rec.compound_id}: {exc}")
            rec.ml_score = None
    return top_candidates


_ID_DRUG = "[C@@]12CC[C@@]3(C)[C@]1(C[C@@H](O)[C@]2(C3=O)O)C(=O)O"
_ID_SEQUENCE = "MKKITIWLISLLVLSISFSTNSEYERISFKNKANFDSAVSK"

_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"


def _rescore_with_chemberta(
    top_candidates: List[CompoundRecord],
) -> List[CompoundRecord]:
    """Rescore using a pre-trained ChemBERTa model via transformers.

    Falls back gracefully if transformers or the model are unavailable.
    """
    if not _HAVE_TRANSFORMERS:
        log.warning("  transformers not installed; skipping ChemBERTa rescore.")
        return top_candidates

    try:
        tokenizer = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            _CHEMBERTA_MODEL, num_labels=1,
        )
    except Exception as exc:
        log.warning(f"  ChemBERTa model load failed ({exc}); skipping rescore.")
        return top_candidates

    for rec in top_candidates:
        try:
            inputs = tokenizer(
                rec.smiles, return_tensors="pt", padding=True, truncation=True,
            )
            outputs = model(**inputs)
            pIC50 = float(outputs.logits.detach().numpy().flatten()[0])
            rec.ml_score = -abs(float(pIC50))  # normalise to negative energy-like
        except Exception as exc:
            log.warning(f"  ChemBERTa scoring failed for {rec.compound_id}: {exc}")
            rec.ml_score = None
    return top_candidates


def rescore_with_ml(
    top_candidates: List[CompoundRecord],
    receptor_pdbqt: str,
    work_dir: str,
) -> List[CompoundRecord]:
    """Rescore the top Vina candidates using the best available ML method.

    Selection priority:
      1. **GNINA**: if the ``gnina`` binary is on ``$PATH``.
      2. **RF-Score-VS**: a Random Forest model trained on Vina energies
         and RDKit descriptors.
      3. **ChemBERTa**: a pre-trained Transformer model regressing SMILES
         → pIC50 (requires ``transformers``).

    Each method sets ``rec.ml_score = predicted_energy_like_value``.
    Unsuccessful methods leave ``ml_score = None``, and the function
    always returns without raising.
    """
    log.info("─── ML Rescoring ───")
    n = len(top_candidates)
    log.info(f"  Rescoring {n} candidates with ML.")

    if _HAVE_GNINA:
        log.info("  Using GNINA (CNN rescoring).")
        return _rescore_with_gnina(top_candidates, receptor_pdbqt, work_dir)

    if _HAVE_RF_SCORE:
        log.info("  Training RF-Score-VS model on Vina energies…")
        model = _train_rf_on_vina_data(top_candidates)
        if model is not None:
            log.info("  Applying RF-Score-VS rescoring.")
            return _rescore_with_rf(top_candidates, model)
        log.warning("  RF model training failed (too few training points).")

    if _HAVE_TRANSFORMERS:
        log.info("  Falling back to ChemBERTa rescoring.")
        return _rescore_with_chemberta(top_candidates)

    log.warning(
        "  No ML rescoring backend available (gnina, sklearn, or transformers). "
        "Leaving ml_score = None."
    )
    return top_candidates
