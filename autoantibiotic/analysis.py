from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, rdDistGeom, rdMolAlign, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier

from .config import CONFIG
from .models import CompoundRecord
from .docking import _parallel_dock
from .io_utils import log
from .scoring_metrics import (
    check_key_interactions,
    compute_ifp_similarity,
    compute_pharmacophore_score,
    _parse_pdbqt_ligand_coords,
    _parse_pdb_residue_coords,
)

_CacheLike = Optional[Dict[str, float]]

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


# ── ChemBERTa Embedder ─────────────────────────────────────────────────────

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
        # [CLS] token is at index 0
        emb = outputs.last_hidden_state[0, 0, :].cpu().numpy().astype(np.float32)
        return emb


# ── ML-ADMET Predictor ─────────────────────────────────────────────────────

class MLADMETPredictor:
    """Lightweight ML-based ADMET predictor using RDKit descriptors +
    Morgan fingerprints + RandomForest.

    Trains on a built-in reference set of known hERG blockers and safe
    compounds at initialisation time.  Provides :meth:`predict_herg_probability`
    and :meth:`predict_cyp_inhibition_probability` (both return 0–1).

    If PyTorch / Transformers are installed, a more sophisticated
    transformer-based predictor can be substituted in the future.
    """

    _REFERENCE_COMPOUNDS: List[tuple] = [
        # (SMILES, hERG_label, CYP_label, name)
        # hERG blockers (positive)
        ("OC1(C2=CC=C(Cl)C=C2)CCN(CCCC(=O)C3=CC=C(F)C=C3)CC1",   1, 1, "Haloperidol"),
        ("CN1C2=CC=CC=C2SC3=C1C=CC=C3CCCN4CCN(C)CC4",            1, 1, "Thioridazine"),
        ("COC1=CC2=C(C=CN=C2)C=C1C(O)C3CC4CCN3CC4C=C",           1, 1, "Quinidine"),
        ("CC1(C(=O)OC2=C1C=C3CC4=CC5=C(C=C4CN3C2=O)OC6=C(C=C(C=C6)C(=O)O)OC5)O", 1, 1, "Doxorubicin"),
        # Safe compounds (negative)
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
        self.herg_model: Optional[_RandomForestClassifier] = None
        self.cyp_model: Optional[_RandomForestClassifier] = None
        self._fitted: bool = False
        self._ndim: int = 0
        try:
            self._fit_models()
        except Exception as exc:
            log.warning(f"ML-ADMET: Model fitting failed — {exc}. "
                         "Falling back to rule-based ADMET.")

    def _get_features(self, mol: Chem.Mol) -> np.ndarray:
        """Get feature vector — either ChemBERTa embedding or RDKit descriptors."""
        if self._embedder is not None:
            return self._embedder.get_embedding(mol)
        return self.compute_features(mol)

    # ── feature engineering ──────────────────────────────────────────

    @staticmethod
    def compute_features(mol: Chem.Mol) -> np.ndarray:
        """2055-dim feature vector: 2048-bit Morgan FP + 7 RDKit descriptors."""
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

    # ── model fitting ────────────────────────────────────────────────

    def _fit_models(self) -> None:
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

    # ── prediction API ───────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._fitted and self.herg_model is not None

    def predict_herg_probability(self, mol: Chem.Mol) -> Optional[float]:
        """Probability of hERG blockage in [0, 1], or *None* if unavailable."""
        if not self.available:
            return None
        try:
            feats = self._get_features(mol).reshape(1, -1)
            return float(self.herg_model.predict_proba(feats)[0, 1])  # type: ignore[union-attr]
        except Exception:
            return None

    def predict_cyp_inhibition_probability(self, mol: Chem.Mol) -> Optional[float]:
        """Probability of CYP inhibition in [0, 1], or *None* if unavailable."""
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
    """Return a singleton ChemBERTaEmbedder, or *None* on failure."""
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
            _chemberta_embedder = None  # ensure it stays None
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
        # "rf_legacy" → embedder stays None (fingerprint RF)

        _ml_predictor = MLADMETPredictor(embedder=embedder)
    return _ml_predictor if _ml_predictor.available else None


def compute_consensus_score(
    vina_energy: Optional[float],
    shape_score: Optional[float],
    vina_weight: float = CONFIG.consensus_vina_weight,
    shape_weight: float = CONFIG.consensus_shape_weight,
) -> Optional[float]:
    """Compute a weighted consensus score from Vina and Shape scores.

    Returns ``w_vina * |vina| + w_shape * shape`` if both are available,
    or whichever single score is present, or ``None`` if neither exists.
    """
    if vina_energy is not None and shape_score is not None:
        return vina_weight * abs(vina_energy) + shape_weight * shape_score
    if vina_energy is not None:
        return abs(vina_energy)
    if shape_score is not None:
        return shape_score
    return None


def compute_selectivity_index(
    pb2pa_energy: float, human_avg_energy: float,
) -> float:
    """Selectivity Index (SI).

    SI = |PBP2a Energy| / |Human Avg Energy|

    Returns 0.0 if either energy is non-negative or human average is near zero.
    """
    if pb2pa_energy >= 0 or human_avg_energy >= 0:
        return 0.0
    return abs(pb2pa_energy) / abs(human_avg_energy) if abs(human_avg_energy) > 1e-6 else 0.0


def profile_resistance_risk(
    record: CompoundRecord,
    work_dir: str,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: tuple,
) -> str:
    """Energy-based heuristic proxy for resistance-risk profiling.

    Returns a human-readable notes string.
    """
    notes: List[str] = []

    act_thresh = CONFIG.resistance_energy_active_threshold
    allo_thresh = CONFIG.resistance_energy_allosteric_threshold
    mw_thresh = CONFIG.resistance_mw_threshold
    rot_thresh = CONFIG.resistance_rot_threshold
    qed_thresh = CONFIG.resistance_qed_threshold

    if record.pb2pa_active_energy is not None and record.pb2pa_active_energy < act_thresh:
        notes.append("Energy profile suggests interaction near catalytic Ser403 (heuristic proxy).")

    if record.pb2pa_allosteric_energy is not None and record.pb2pa_allosteric_energy < allo_thresh:
        if record.pb2pa_active_energy is None or record.pb2pa_active_energy > act_thresh:
            notes.append("Allosteric binder (Ala237/Met241/Tyr159 pocket). Novel mechanism.")

    if record.mol is not None:
        mw = Descriptors.MolWt(record.mol)
        if mw > mw_thresh:
            notes.append(f"High MW (>{mw_thresh:.0f}) — broad interaction surface, may contact multiple residues.")
        n_rot = Descriptors.NumRotatableBonds(record.mol)
        if n_rot < rot_thresh:
            notes.append(f"Rigid scaffold — reduced entropic penalty, may enhance binding specificity.")

    if record.qed_score > qed_thresh:
        notes.append(f"High drug-likeness (QED > {qed_thresh}) — good developability profile.")

    if not notes:
        notes.append("No specific resistance flags identified.")

    return "; ".join(notes)


_LOGS_MODEL_COEFFS = {
    "c": 0.16, "MolLogP": -0.63, "MolWt": -0.0062,
    "NumRotatableBonds": -0.0034, "NumAromaticRings": -0.042,
    "HeavyAtomCount": 0.00025,
}


def predict_logs(mol: Chem.Mol) -> float:
    """Predict aqueous solubility (LogS) using a simple linear model.

    The model is a re-implementation of the ESOL method (Delaney 2004)
    using RDKit descriptors:

        LogS = 0.16 - 0.63*LogP - 0.0062*MW + 0.0034*RotBonds
               - 0.042*AromRings + 0.00025*HeavyAtoms

    Returns predicted LogS in mol/L.
    """
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


def _has_basic_nitrogen(mol: Chem.Mol) -> bool:
    """Check if the molecule contains a basic nitrogen (aliphatic primary/secondary/tertiary)."""
    basic_n_pattern = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(NC=O)]")
    if basic_n_pattern is None:
        return False
    return mol.HasSubstructMatch(basic_n_pattern)


def predict_herg_risk(mol: Chem.Mol) -> str:
    """Rule-based hERG blockage risk assessment.

    Flags compounds with:
      - LogP > 4.0  (high lipophilicity → promiscuous hERG binding)
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


def analyze_selectivity_and_resistance(
    top10: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: _CacheLike = None,
    use_cache: bool = False,
) -> List[CompoundRecord]:
    """Phase 4 — Selectivity & Resistance Analysis.

    Docks top 10 against human off-targets, computes SI, profiles resistance risk.
    """
    log.info("─── Phase 4: Selectivity & Resistance Analysis ───")

    use_vina = deps.get("USE_VINA", False)
    if not use_vina:
        log.warning("  Vina unavailable — skipping selectivity docking. Flagging all as uncertain.")
        for rec in top10:
            rec.selectivity_index = 1.0
            rec.resistance_notes = "Selectivity not assessed (Vina unavailable)."
        return top10

    trypsin_target = targets.get("trypsin")
    ces1_target = targets.get("CES1")
    if trypsin_target is None or ces1_target is None:
        log.warning("  Off-target data missing — skipping selectivity docking.")
        for rec in top10:
            rec.selectivity_index = 1.0
            rec.resistance_notes = "Selectivity not assessed (off-target data missing)."
        return top10

    log.info("  Docking top 10 vs Human Trypsin (1UTN)…")
    trypsin_center = trypsin_target.get("active_center", np.array([0.0, 0.0, 0.0]))
    trypisn_items = [(r.compound_id, r.smiles) for r in top10]
    trypsin_results = _parallel_dock(
        trypisn_items, targets["trypsin"]["pdbqt"],
        trypsin_center, CONFIG.offtarget_box_size,
        work_dir, "trypsin", n_jobs=min(4, len(top10)),
        cache=cache, use_cache=use_cache,
    )
    cid_map = {r.compound_id: r for r in top10}
    for cid, energy in trypsin_results:
        if cid in cid_map:
            cid_map[cid].human_trypsin_energy = energy

    log.info("  Docking top 10 vs Human Carboxylesterase 1 (3KJZ)…")
    ces1_center = ces1_target.get("active_center", np.array([0.0, 0.0, 0.0]))
    ces1_items = [(r.compound_id, r.smiles) for r in top10]
    ces1_results = _parallel_dock(
        ces1_items, targets["CES1"]["pdbqt"],
        ces1_center, CONFIG.offtarget_box_size,
        work_dir, "ces1", n_jobs=min(4, len(top10)),
        cache=cache, use_cache=use_cache,
    )
    for cid, energy in ces1_results:
        if cid in cid_map:
            cid_map[cid].human_ces1_energy = energy

    for rec in top10:
        energies_human = [
            e for e in (rec.human_trypsin_energy, rec.human_ces1_energy)
            if e is not None
        ]
        if not energies_human:
            log.warning(f"  {rec.compound_id}: No human docking data. SI = N/A.")
            rec.selectivity_index = 1.0
            continue

        human_avg = np.mean(energies_human)
        pb2pa_best = (
            rec.pb2pa_active_energy if rec.pb2pa_active_energy is not None
            else rec.pb2pa_allosteric_energy
        )
        if pb2pa_best is None:
            rec.selectivity_index = 1.0
            continue

        si = compute_selectivity_index(pb2pa_best, human_avg)
        rec.selectivity_index = si

        if si < CONFIG.selectivity_index_threshold:
            log.warning(
                f"  {rec.compound_id}: Low selectivity (SI = {si:.2f} < {CONFIG.selectivity_index_threshold}). "
                "Flagged for off-target risk."
            )
        else:
            log.info(f"  {rec.compound_id}: SI = {si:.2f} (pass).")

    pb2pa = targets["PBP2a"]
    receptor_pdb = pb2pa["pdbqt"].replace(".pdbqt", ".pdb")
    if not os.path.isfile(receptor_pdb):
        receptor_pdb = os.path.join(
            os.path.dirname(pb2pa["pdbqt"]),
            "PBP2a_clean.pdb",
        )

    log.info("  Checking key interactions for top candidates…")
    for rec in top10:
        rec.resistance_notes = profile_resistance_risk(
            rec, work_dir,
            pb2pa["pdbqt"],
            pb2pa["allosteric_center"],
            CONFIG.allosteric_box_size,
        )

        # Generate 3-D conformer and write temporary PDBQT for IFP check
        if rec.mol is None:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            rec.mol = mol
        mol = Chem.MolFromSmiles(rec.smiles) if rec.mol is None else Chem.RWMol(rec.mol)
        mol = Chem.AddHs(mol)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = CONFIG.random_seed
        if rdDistGeom.EmbedMolecule(mol, params) >= 0:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".pdbqt", delete=False,
                ) as tmp:
                    tmp_pdbqt = tmp.name
                    conf = mol.GetConformer()
                    tmp.write("ROOT\n")
                    for i in range(mol.GetNumAtoms()):
                        atom = mol.GetAtomWithIdx(i)
                        if atom.GetAtomicNum() == 1:
                            continue
                        pt = conf.GetAtomPosition(i)
                        elem = atom.GetSymbol()
                        tmp.write(
                            f"ATOM  {i+1:5d} {elem:<4s} LIG     1    "
                            f"{pt.x:8.3f}{pt.y:8.3f}{pt.z:8.3f}  "
                            f"1.00  0.00          {elem:>2s}\n"
                        )
                    tmp.write("ENDROOT\n")
                try:
                    allosteric_hits = (
                        CONFIG.min_key_interactions > 0
                        and check_key_interactions(
                            tmp_pdbqt, receptor_pdb,
                            CONFIG.key_interaction_residues_allosteric,
                        )
                    )
                    active_hits = check_key_interactions(
                        tmp_pdbqt, receptor_pdb,
                        CONFIG.key_interaction_residues_active,
                    )
                    if not (allosteric_hits or active_hits):
                        rec.resistance_notes += (
                            "; Warning: No key interactions detected"
                        )
                finally:
                    try:
                        os.unlink(tmp_pdbqt)
                    except OSError:
                        pass
            except Exception as exc:
                log.debug(f"  IFP check failed for {rec.compound_id}: {exc}")

        # ── IFP similarity to reference Ceftaroline ──
        try:
            ref_smi = CONFIG.control_smiles.get("Ceftaroline", "")
            if ref_smi:
                ref_mol = Chem.MolFromSmiles(ref_smi)
                if ref_mol is not None and rec.mol is not None:
                    rec.ifp_score = compute_ifp_similarity(
                        rec.mol, ref_mol, receptor_pdb,
                    )
                    if rec.ifp_score < CONFIG.ifp_similarity_threshold:
                        rec.resistance_notes += (
                            f"; Warning: Low IFP similarity to reference ligand "
                            f"({rec.ifp_score:.2f})"
                        )
        except Exception as exc:
            log.debug(f"  IFP similarity failed for {rec.compound_id}: {exc}")

    # ── ADMET profiling on top 10 ──
    log.info("  Computing ADMET profiles for top candidates…")
    for rec in top10:
        predict_admet_profile(rec)
        log.debug(f"  {rec.compound_id}: {rec.admet_flags}")

    log.info("─── Phase 4 complete ───")
    return top10
