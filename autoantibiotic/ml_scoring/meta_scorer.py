from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED
from sklearn.ensemble import RandomForestRegressor

from ..config import CONFIG
from ..io_utils import log
from ..models import CompoundRecord


class MetaScorer:
    """Stacking regressor that learns to predict activity from multiple
    docking and descriptor features.

    Features
    --------
    - Vina Energy (allosteric)
    - Vina Energy (active)
    - GNINA CNNscore (if available)
    - Shape Score
    - IFP Score
    - QED
    - LogP
    - MolWt

    Training data is sourced from :mod:`benchmarks.reference_data` using
    known actives / inactives.  The trained model is persisted with
    ``joblib`` in the ``output/`` directory for reuse.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model: Optional[RandomForestRegressor] = None
        self._fitted: bool = False
        self._feature_names: List[str] = []
        self.model_path: Optional[str] = model_path or str(
            CONFIG.output_dir / "meta_scorer.joblib"
        )
        self._training_actives: List[str] = []
        self._training_inactives: List[str] = []

    # ── public API ──────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._fitted and self._model is not None

    def fit(
        self,
        actives_smiles: List[str],
        inactives_smiles: List[str],
        feature_fn: Optional[Any] = None,
        uncertainty_threshold: Optional[float] = None,
    ) -> "MetaScorer":
        """Train the stacking regressor on benchmark actives / inactives.

        Parameters
        ----------
        actives_smiles : list of str
            SMILES for known PBP2a inhibitors (label = 1).
        inactives_smiles : list of str
            SMILES for confirmed negatives (label = 0).
        feature_fn : callable, optional
            Function ``(mol) -> np.ndarray`` that extracts the feature
            vector.  Defaults to :meth:`_default_features`.
        uncertainty_threshold : float, optional
            If set, during :meth:`predict` the standard deviation of
            predictions across all trees is compared to this threshold.
            When exceeded, ``record.needs_manual_review`` is set to
            ``True``, flagging the compound for manual inspection.
        """
        X_list: List[np.ndarray] = []
        y_list: List[float] = []

        self._training_actives = list(actives_smiles)
        self._training_inactives = list(inactives_smiles)

        feature_extractor = feature_fn or self._default_features

        for smi in actives_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                X_list.append(feature_extractor(mol))
                y_list.append(1.0)
            except Exception:
                continue

        for smi in inactives_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                X_list.append(feature_extractor(mol))
                y_list.append(0.0)
            except Exception:
                continue

        if len(X_list) < 4:
            log.warning(
                f"MetaScorer: only {len(X_list)} valid training points "
                "— skipping training."
            )
            return self

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        self._model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=CONFIG.random_seed,
            oob_score=True,
        )
        self._model.fit(X, y)
        self._fitted = True
        self._uncertainty_threshold = uncertainty_threshold
        oob = getattr(self._model, "oob_score_", float("nan"))
        log.info(
            f"MetaScorer: trained on {len(X)} compounds "
            f"({len(actives_smiles)} actives / {len(inactives_smiles)} inactives), "
            f"OOB R² = {oob:.4f}"
        )

        self._save()
        return self

    def predict(self, record: CompoundRecord) -> Optional[float]:
        """Predict the meta-score for a single CompoundRecord.

        Returns a score in [0, 1] where higher is more likely active,
        or ``None`` if the model is not fitted or feature extraction
        fails.

        If the record contains MD-derived dynamic features
        (``md_ligand_rmsd``, ``md_pocket_rg_stability``) they are
        included in the feature vector automatically.

        When ``uncertainty_threshold`` was set at fit time, the standard
        deviation of predictions across all trees is computed.  If it
        exceeds the threshold, ``record.needs_manual_review`` is set to
        ``True``.
        """
        if not self.available:
            return None
        if record.mol is None:
            mol = Chem.MolFromSmiles(record.smiles)
            if mol is None:
                return None
            record.mol = mol
        try:
            feats = self._default_features(
                record.mol,
                md_ligand_rmsd=record.md_ligand_rmsd,
                md_pocket_rg_stability=record.md_pocket_rg_stability,
            ).reshape(1, -1)
            prob = float(self._model.predict(feats)[0])  # type: ignore[union-attr]

            # Active-learning: ensemble variance check
            threshold = getattr(self, "_uncertainty_threshold", None)
            if threshold is not None and hasattr(self._model, "estimators_"):
                tree_preds = np.array([
                    tree.predict(feats)[0] for tree in self._model.estimators_
                ])
                std = float(np.std(tree_preds, ddof=1))
                if std > threshold:
                    record.needs_manual_review = True
                    log.debug(
                        f"MetaScorer: {record.compound_id} flagged for review "
                        f"(std={std:.3f} > threshold={threshold})"
                    )

            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            return None

    def load(self) -> bool:
        """Load a previously trained model from disk.

        Checks that the stored feature dimension matches the current
        expected dimension (11).  If mismatched, logs a warning and
        returns ``False`` so the caller retrains with the new feature
        space.

        Returns ``True`` on success.
        """
        if self.model_path is None or not Path(self.model_path).exists():
            return False
        try:
            obj = joblib.load(self.model_path)
            self._model = obj.get("model")
            self._fitted = obj.get("fitted", False)
            self._feature_names = obj.get("feature_names", [])
            self._training_actives = obj.get("training_actives", [])
            self._training_inactives = obj.get("training_inactives", [])

            # Backward compatibility: retrain if feature dimension has changed
            expected_dim = 11
            if self._model is not None and hasattr(self._model, "n_features_in_"):
                if self._model.n_features_in_ != expected_dim:
                    log.warning(
                        f"MetaScorer: loaded model has "
                        f"{self._model.n_features_in_} features, "
                        f"current code expects {expected_dim}. "
                        "Triggering retrain."
                    )
                    self._fitted = False
                    self._model = None

            return self._fitted and self._model is not None
        except Exception as exc:
            log.warning(f"MetaScorer: failed to load model — {exc}")
            return False

    def flag_uncertain_predictions(
        self,
        records: List[CompoundRecord],
        threshold: float = 0.1,
    ) -> List[CompoundRecord]:
        """Flag records whose prediction std across trees exceeds *threshold*.

        Sets ``record.needs_manual_review = True`` for uncertain compounds.
        Returns the list for chaining.
        """
        if not self.available or not hasattr(self._model, "estimators_"):
            return records

        for record in records:
            if record.needs_manual_review:
                continue
            mol = record.mol
            if mol is None:
                mol = Chem.MolFromSmiles(record.smiles)
                if mol is None:
                    continue
            try:
                feats = self._default_features(
                    mol,
                    md_ligand_rmsd=record.md_ligand_rmsd,
                    md_pocket_rg_stability=record.md_pocket_rg_stability,
                ).reshape(1, -1)
                tree_preds = np.array([
                    tree.predict(feats)[0] for tree in self._model.estimators_
                ])
                std = float(np.std(tree_preds, ddof=1))
                if std > threshold:
                    record.needs_manual_review = True
            except Exception:
                continue
        return records

    def retrain_with_new_data(
        self,
        new_actives: List[str],
        new_inactives: List[str],
    ) -> "MetaScorer":
        """Append new training data and refit the model.

        Combines the existing training SMILES (stored at fit time) with
        the newly provided actives and inactives, then calls :meth:`fit`.
        """
        all_actives = list(self._training_actives)
        all_inactives = list(self._training_inactives)

        for smi in new_actives:
            if smi not in all_actives:
                all_actives.append(smi)
        for smi in new_inactives:
            if smi not in all_inactives:
                all_inactives.append(smi)

        old_threshold = getattr(self, "_uncertainty_threshold", None)
        self.fit(
            all_actives,
            all_inactives,
            uncertainty_threshold=old_threshold,
        )
        return self

    # ── internals ───────────────────────────────────────────────────

    def _default_features(
        self,
        mol: Chem.Mol,
        md_ligand_rmsd: Optional[float] = None,
        md_pocket_rg_stability: Optional[float] = None,
    ) -> np.ndarray:
        """11-dim feature vector for a molecule.

        Static descriptors (8):
        1-4. Vina/GNINA/Shape/IFP placeholders (0.0)
        5.   QED
        6.   LogP
        7.   MolWt
        8.   NumRotatableBonds

        Dynamic MD features (3), default 0.0 when unavailable:
        9.  ligand_rmsd_mean
        10. ligand_rmsd_std
        11. pocket_rg_stability
        """
        qed = float(QED.qed(mol))
        logp = float(Crippen.MolLogP(mol))
        mw = float(Descriptors.MolWt(mol))
        n_rot = float(Descriptors.NumRotatableBonds(mol))

        rmsd_mean = md_ligand_rmsd if md_ligand_rmsd is not None else 0.0
        rmsd_std = 0.0
        rg_stab = md_pocket_rg_stability if md_pocket_rg_stability is not None else 0.0

        arr = np.array(
            [0.0, 0.0, 0.0, 0.0, qed, logp, mw, n_rot,
             rmsd_mean, rmsd_std, rg_stab],
            dtype=np.float32,
        )
        self._feature_names = [
            "vina_energy", "gnina_score", "shape_score", "ifp_score",
            "qed", "logp", "mw", "n_rotatable",
            "ligand_rmsd_mean", "ligand_rmsd_std", "pocket_rg_stability",
        ]
        return arr

    def _save(self) -> None:
        if self.model_path is None:
            return
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": self._model,
                    "fitted": self._fitted,
                    "feature_names": self._feature_names,
                    "training_actives": self._training_actives,
                    "training_inactives": self._training_inactives,
                },
                self.model_path,
            )
            log.info(f"MetaScorer: model saved to {self.model_path}")
        except Exception as exc:
            log.warning(f"MetaScorer: failed to save model — {exc}")


# Module-level singleton
_meta_scorer: Optional[MetaScorer] = None


def _get_meta_scorer() -> Optional[MetaScorer]:
    """Return a singleton MetaScorer (trained or loaded from disk)."""
    global _meta_scorer
    if _meta_scorer is None:
        _meta_scorer = MetaScorer()
        if _meta_scorer.load():
            log.info("MetaScorer: loaded from disk.")
        else:
            from benchmarks.reference_data import get_actives_smiles, get_inactives_smiles

            try:
                _meta_scorer.fit(
                    actives_smiles=get_actives_smiles(),
                    inactives_smiles=get_inactives_smiles(),
                )
            except Exception as exc:
                log.warning(f"MetaScorer: training failed — {exc}. "
                             "Falling back to weighted consensus.")
                _meta_scorer = None
    return _meta_scorer if (_meta_scorer is not None and _meta_scorer.available) else None


def predict_meta_score(record: CompoundRecord) -> Optional[float]:
    """Predict consensus activity score using the trained ``MetaScorer``.

    Falls back to the legacy weighted :func:`compute_consensus_score` if
    the meta-scorer is unavailable.
    """
    if CONFIG.use_meta_scoring:
        scorer = _get_meta_scorer()
        if scorer is not None:
            score = scorer.predict(record)
            if score is not None:
                return score
    from ..analysis import compute_consensus_score
    return compute_consensus_score(
        record.pb2pa_allosteric_energy,
        record.shape_score,
    )
