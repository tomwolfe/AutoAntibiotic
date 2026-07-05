from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors

from ..config import CONFIG
from ..io_utils import log
from ..models import CompoundRecord

# ── Optional ML backend flags ───────────────────────────────────────

_HAVE_TORCH = True
_HAVE_TRANSFORMERS = True
try:
    import torch  # noqa: F401
except ImportError:
    _HAVE_TORCH = False
try:
    import transformers  # noqa: F401
except ImportError:
    _HAVE_TRANSFORMERS = False


# ── ChemBERTa Embedder ─────────────────────────────────────────────

class ChemBERTaEmbedder:
    """Extract 768-dimensional [CLS] embeddings from a pre-trained ChemBERTa
    model for use as features in downstream ML models.

    Loads the model/tokenizer once at the class level (singleton) so that
    repeated calls do not re-download.
    """

    _model: Any = None
    _tokenizer: Any = None
    _device: Any = None

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1") -> None:
        self.model_name = model_name
        self._initialize()

    def _initialize(self) -> None:
        if ChemBERTaEmbedder._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        ChemBERTaEmbedder._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        ChemBERTaEmbedder._model = AutoModel.from_pretrained(self.model_name)
        ChemBERTaEmbedder._model.eval()
        ChemBERTaEmbedder._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ChemBERTaEmbedder._model.to(ChemBERTaEmbedder._device)

    def get_embedding(self, mol: Chem.Mol) -> np.ndarray:
        """Return the 768-dim [CLS] embedding for *mol*."""
        import torch

        smiles = Chem.MolToSmiles(mol)
        inputs = self._tokenizer(
            smiles, return_tensors="pt", truncation=True, max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        emb = outputs.last_hidden_state[0, 0, :].cpu().numpy().astype(np.float32)
        return emb


# ── ML-ADMET Predictor ─────────────────────────────────────────────

class MLADMETPredictor:
    """Lightweight ML-based ADMET predictor using RDKit descriptors +
    Morgan fingerprints + RandomForest.

    Trains on a built-in reference set of known hERG blockers and safe
    compounds at initialisation time.  Provides :meth:`predict_herg_probability`
    and :meth:`predict_cyp_inhibition_probability` (both return 0–1).
    """

    _REFERENCE_COMPOUNDS: List[tuple] = [
        # (SMILES, hERG_label, CYP_label, name)
        ("OC1(C2=CC=C(Cl)C=C2)CCN(CCCC(=O)C3=CC=C(F)C=C3)CC1",   1, 1, "Haloperidol"),
        ("CN1C2=CC=CC=C2SC3=C1C=CC=C3CCCN4CCN(C)CC4",            1, 1, "Thioridazine"),
        ("COC1=CC2=C(C=CN=C2)C=C1C(O)C3CC4CCN3CC4C=C",           1, 1, "Quinidine"),
        ("CC1(C(=O)OC2=C1C=C3CC4=CC5=C(C=C4CN3C2=O)OC6=C(C=C(C=C6)C(=O)O)OC5)O", 1, 1, "Doxorubicin"),
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  0, 0, "Caffeine"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O",      0, 0, "Aspirin"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 0, 0, "Ibuprofen"),
        ("CC(=O)NC1=CC=C(C=C1)O",         0, 0, "Acetaminophen"),
        ("c1ccccc1",                       0, 0, "Benzene"),
        ("CCO",                            0, 0, "Ethanol"),
        ("c1ccccc1O",                      0, 0, "Phenol"),
    ]

    def __init__(self, embedder: Optional[ChemBERTaEmbedder] = None) -> None:
        self._embedder: Optional[ChemBERTaEmbedder] = embedder
        self.herg_model: Any = None
        self.cyp_model: Any = None
        self._fitted: bool = False
        self._ndim: int = 0
        try:
            self._fit_models()
        except Exception as exc:
            log.warning(f"ML-ADMET: Model fitting failed — {exc}. "
                         "Falling back to rule-based ADMET.")

    def _get_features(self, mol: Chem.Mol) -> np.ndarray:
        if self._embedder is not None:
            return self._embedder.get_embedding(mol)
        return self.compute_features(mol)

    @staticmethod
    def compute_features(mol: Chem.Mol) -> np.ndarray:
        """2055-dim feature vector: 2048-bit Morgan FP + 7 RDKit descriptors."""
        from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        descs = np.array([
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            float(mol.GetNumHeavyAtoms()),
        ], dtype=np.float32)
        return np.concatenate([arr, descs])

    def _fit_models(self) -> None:
        from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier

        try:
            from benchmarks.reference_data import load_chembl_admet_subset
            chembl_data = load_chembl_admet_subset()
            herg_samples = chembl_data["herg"]
            cyp_samples = chembl_data["cyp"]
            _has_expanded_data = True
        except (ImportError, Exception):
            _has_expanded_data = False
            herg_samples = []
            cyp_samples = []

        if _has_expanded_data and len(herg_samples) > 20:
            log.info(f"ML-ADMET: Using expanded reference set ({len(herg_samples)} hERG, {len(cyp_samples)} CYP).")
            X_herg: List[np.ndarray] = []
            y_herg: List[int] = []
            X_cyp: List[np.ndarray] = []
            y_cyp: List[int] = []

            for entry in herg_samples:
                mol = Chem.MolFromSmiles(entry["smiles"])
                if mol is None:
                    continue
                mol = Chem.MolFromSmiles(entry["smiles"])
                if mol is None:
                    continue
                try:
                    X_herg.append(self._get_features(mol))
                    y_herg.append(entry["label"])
                except Exception:
                    continue
            for entry in cyp_samples:
                mol = Chem.MolFromSmiles(entry["smiles"])
                if mol is None:
                    continue
                try:
                    X_cyp.append(self._get_features(mol))
                    y_cyp.append(entry["label"])
                except Exception:
                    continue

            n_pos = sum(y_herg)
            n_neg = len(y_herg) - n_pos
            if n_pos < 1 or n_neg < 1:
                log.warning("ML-ADMET: Need at least one positive and one negative "
                            "hERG reference compound — disabling ML models.")
                return

            Xh = np.array(X_herg, dtype=np.float32)
            self._ndim = Xh.shape[1]
            self.herg_model = _RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced",
            )
            self.herg_model.fit(Xh, y_herg)

            n_pos = sum(y_cyp)
            n_neg = len(y_cyp) - n_pos
            if n_pos < 1 or n_neg < 1:
                log.warning("ML-ADMET: Need at least one positive and one negative "
                            "CYP reference compound — disabling CYP model.")
                self.cyp_model = None
            else:
                Xc = np.array(X_cyp, dtype=np.float32)
                self.cyp_model = _RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced",
                )
                self.cyp_model.fit(Xc, y_cyp)

            self._fitted = True
            log.info(f"ML-ADMET: Models fitted ({len(y_herg)} hERG, {len(y_cyp)} CYP training compounds, "
                     f"{self._ndim} features).")
            return

        X_list: List[np.ndarray] = []
        y_herg: List[int] = []
        y_cyp: List[int] = []

        for smi, h_label, c_label, name in self._REFERENCE_COMPOUNDS:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                log.warning(f"ML-ADMET: Skipping '{name}' — invalid SMILES")
                continue
            try:
                X_list.append(self._get_features(mol))
                y_herg.append(h_label)
                y_cyp.append(c_label)
            except Exception as exc:
                log.warning(f"ML-ADMET: Skipping '{name}' — {exc}")
                continue

        n_pos = sum(y_herg)
        n_neg = len(y_herg) - n_pos
        if n_pos < 1 or n_neg < 1:
            log.warning("ML-ADMET: Need at least one positive and one negative "
                        "reference compound — disabling ML models.")
            return

        X = np.array(X_list, dtype=np.float32)
        self._ndim = X.shape[1]

        self.herg_model = _RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced",
        )
        self.herg_model.fit(X, y_herg)

        self.cyp_model = _RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced",
        )
        self.cyp_model.fit(X, y_cyp)

        self._fitted = True
        log.info(f"ML-ADMET: Models fitted ({len(y_herg)} training compounds, "
                 f"{self._ndim} features).")

    @property
    def available(self) -> bool:
        return self._fitted and self.herg_model is not None

    def predict_herg_probability(self, mol: Chem.Mol) -> Optional[float]:
        if not self.available:
            return None
        try:
            feats = self._get_features(mol).reshape(1, -1)
            return float(self.herg_model.predict_proba(feats)[0, 1])  # type: ignore[union-attr]
        except Exception:
            return None

    def predict_cyp_inhibition_probability(self, mol: Chem.Mol) -> Optional[float]:
        if not self.available:
            return None
        try:
            feats = self._get_features(mol).reshape(1, -1)
            return float(self.cyp_model.predict_proba(feats)[0, 1])  # type: ignore[union-attr]
        except Exception:
            return None


# Module-level predictor (lazy singleton)
_ml_predictor: Optional[MLADMETPredictor] = None
_chemberta_embedder: Optional[ChemBERTaEmbedder] = None


def _get_chemberta_embedder() -> Optional[ChemBERTaEmbedder]:
    global _chemberta_embedder
    if _chemberta_embedder is None:
        try:
            _chemberta_embedder = ChemBERTaEmbedder(
                model_name=CONFIG.chemberta_model_name,
            )
        except Exception as exc:
            log.warning(
                f"ChemBERTa embedder failed to load — {exc}. "
                "Falling back to fingerprint-based features."
            )
            _chemberta_embedder = None
    return _chemberta_embedder


def _get_ml_admet_predictor() -> Optional[MLADMETPredictor]:
    global _ml_predictor
    if not CONFIG.use_ml_admet:
        _ml_predictor = None
        return None
    if _ml_predictor is None:
        embedder: Optional[ChemBERTaEmbedder] = None

        if CONFIG.ml_admet_model_type == "chemberta_rf":
            if _HAVE_TRANSFORMERS and _HAVE_TORCH:
                embedder = _get_chemberta_embedder()
                if embedder is not None:
                    log.info("Using ChemBERTa embeddings for ML-ADMET")
                else:
                    log.warning(
                        "ChemBERTa not available, "
                        "falling back to fingerprint-based RF"
                    )
            else:
                log.warning(
                    "Transformers not available, "
                    "falling back to fingerprint-based RF"
                )
        elif CONFIG.ml_admet_model_type == "rule_based":
            _ml_predictor = None
            return None

        _ml_predictor = MLADMETPredictor(embedder=embedder)
    return _ml_predictor if _ml_predictor.available else None


# ── Solubility (ESOL) ──────────────────────────────────────────────

_LOGS_MODEL_COEFFS = {
    "c": 0.16, "MolLogP": -0.63, "MolWt": -0.0062,
    "NumRotatableBonds": -0.0034, "NumAromaticRings": -0.042,
    "HeavyAtomCount": 0.00025,
}


def predict_logs(mol: Chem.Mol) -> float:
    """Predict aqueous solubility (LogS) using a simple linear model (ESOL)."""
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    n_arom = Descriptors.NumAromaticRings(mol) if hasattr(Descriptors, "NumAromaticRings") else 0
    heavy = mol.GetNumHeavyAtoms()

    logs = (
        _LOGS_MODEL_COEFFS["c"]
        + _LOGS_MODEL_COEFFS["MolLogP"] * logp
        + _LOGS_MODEL_COEFFS["MolWt"] * mw
        + _LOGS_MODEL_COEFFS["NumRotatableBonds"] * rot
        + _LOGS_MODEL_COEFFS["NumAromaticRings"] * n_arom
        + _LOGS_MODEL_COEFFS["HeavyAtomCount"] * heavy
    )
    return logs


# ── Rule-based hERG helpers ───────────────────────────────────────

def _has_basic_nitrogen(mol: Chem.Mol) -> bool:
    basic_n_pattern = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(NC=O)]")
    if basic_n_pattern is None:
        return False
    return mol.HasSubstructMatch(basic_n_pattern)


def predict_herg_risk(mol: Chem.Mol) -> str:
    """Rule-based hERG blockage risk assessment.

    Flags compounds with:
      - LogP > 4.0 (high lipophilicity → promiscuous hERG binding)
      - AND presence of a basic nitrogen

    Returns ``"High"``, ``"Moderate"``, or ``"Low"``.
    """
    logp = Crippen.MolLogP(mol)
    has_basic_n = _has_basic_nitrogen(mol)
    if logp > 4.0 and has_basic_n:
        return "High"
    if logp > 4.0 or has_basic_n:
        return "Moderate"
    return "Low"


# ── ML-based predictions ──────────────────────────────────────────

def predict_herg_ml(mol: Chem.Mol) -> str:
    """ML-based hERG blockage risk using the global predictor.

    Returns ``"High"``, ``"Moderate"``, or ``"Low"``.
    Falls back to the rule-based method if ML not available.
    """
    predictor = _get_ml_admet_predictor()
    if predictor is None:
        return predict_herg_risk(mol)
    prob = predictor.predict_herg_probability(mol)
    if prob is None:
        return predict_herg_risk(mol)
    if prob >= CONFIG.ml_admet_herg_threshold:
        return "High"
    if prob >= CONFIG.ml_admet_herg_threshold * 0.5:
        return "Moderate"
    return "Low"


def predict_cyp_inhibition(mol: Chem.Mol) -> str:
    """ML-based CYP inhibition prediction.

    Returns ``"Yes"`` or ``"No"`` based on the probability threshold.
    Falls back to a simple rule (any basic N) if ML not available.
    """
    predictor = _get_ml_admet_predictor()
    if predictor is None:
        return "Yes" if _has_basic_nitrogen(mol) else "No"
    prob = predictor.predict_cyp_inhibition_probability(mol)
    if prob is None:
        return "Yes" if _has_basic_nitrogen(mol) else "No"
    return "Yes" if prob >= CONFIG.ml_admet_herg_threshold else "No"


# ── Full ADMET profile ────────────────────────────────────────────

def predict_admet_profile(record: CompoundRecord) -> CompoundRecord:
    """Compute ADMET properties for a compound and populate *admet_flags*.

    Evaluates:
      1. **Solubility (LogS)**: predicted via ESOL model.
      2. **hERG blockage risk**: ML-based (RandomForest) with rule-based fallback.
      3. **CYP inhibition risk**: ML-based (RandomForest) with rule-based fallback.
      4. **Lipinski Rule-of-5** and **QED** (already computed in filtering).

    Flags are appended to ``record.admet_flags``.

    Returns the same ``CompoundRecord`` with populated *admet_flags*.
    """
    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            record.admet_flags.append("ADMET: invalid molecule")
            return record
        record.mol = mol
    mol = record.mol

    flags: List[str] = []

    # Solubility
    try:
        logs = predict_logs(mol)
        if logs < CONFIG.ml_admet_solubility_threshold - 1.0:
            flags.append(f"Poor solubility (LogS={logs:.2f})")
        elif logs < CONFIG.ml_admet_solubility_threshold:
            flags.append(f"Moderate solubility (LogS={logs:.2f})")
        else:
            flags.append(f"Good solubility (LogS={logs:.2f})")
    except Exception:
        flags.append("Solubility prediction failed")

    # hERG risk — ML with rule-based fallback
    try:
        herg = predict_herg_ml(mol)
        predictor = _get_ml_admet_predictor()
        if predictor is not None:
            prob = predictor.predict_herg_probability(mol)
            prob_str = f" (ML Prob: {prob:.2f})" if prob is not None else ""
        else:
            prob_str = ""
        if herg == "High":
            flags.append(f"High hERG risk{prob_str}")
        elif herg == "Moderate":
            flags.append(f"Moderate hERG risk{prob_str}")
        else:
            flags.append(f"Low hERG risk{prob_str}")
    except Exception:
        flags.append("hERG prediction failed")

    # CYP inhibition — ML with rule-based fallback
    try:
        cyp = predict_cyp_inhibition(mol)
        if cyp == "Yes":
            flags.append("CYP inhibition predicted")
        else:
            flags.append("No CYP inhibition predicted")
    except Exception:
        flags.append("CYP prediction failed")

    # Lipinski (already computed, just annotate)
    if record.passes_lipinski:
        flags.append("Lipinski OK")
    else:
        flags.append("Lipinski violation")

    # QED
    if record.qed_score > CONFIG.qed_threshold:
        flags.append(f"QED OK ({record.qed_score:.2f})")
    else:
        flags.append(f"QED below threshold ({record.qed_score:.2f})")

    record.admet_flags = flags
    return record
