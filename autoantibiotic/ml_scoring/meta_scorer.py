from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from ..config import CONFIG, ConfigurationError
from ..io_utils import log
from ..models import CompoundRecord


class MetaScorer:
    """Stacking regressor that learns to predict activity from multiple
    docking and descriptor features.

    Features (13-dim)
    -----------------
    - Vina Energy (allosteric) [placeholder]
    - GNINA CNNscore [placeholder]
    - Shape Score [placeholder]
    - IFP Score (docking) [placeholder]
    - QED
    - LogP
    - MolWt
    - NumRotatableBonds
    - Ligand RMSD mean (MD-derived)
    - Ligand RMSD std (MD-derived)
    - Pocket Rg stability (MD-derived)
    - IFP similarity score (from CompoundRecord)
    - Water displacement energy (from CompoundRecord)

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
        self.uses_dynamic_features: bool = False

    # ── public API ──────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._fitted and self._model is not None

    def validate_input_features(self, record: CompoundRecord) -> bool:
        """Validate that required input features are present before prediction.

        When ``CONFIG.force_md_for_meta_scoring`` is ``True``, this method
        verifies that MD-derived dynamic features (``md_ligand_rmsd`` and
        ``md_pocket_rg_stability``) are not ``None``.  If they are missing,
        a :class:`~autoantibiotic.config.ConfigurationError` is raised.

        Returns ``True`` when validation passes.
        """
        if CONFIG.force_md_for_meta_scoring:
            missing = []
            if record.md_ligand_rmsd is None:
                missing.append("md_ligand_rmsd")
            if record.md_pocket_rg_stability is None:
                missing.append("md_pocket_rg_stability")
            if missing:
                raise ConfigurationError(
                    f"MetaScorer: {record.compound_id} is missing required MD "
                    f"features: {', '.join(missing)}. "
                    "Set CONFIG.force_md_for_meta_scoring=False or run MD "
                    "validation first."
                )
        return True

    @staticmethod
    def _scaffold_groups(smiles_list: List[str]) -> Dict[str, List[int]]:
        """Group molecule indices by Murcko scaffold.

        Returns a dict mapping scaffold SMILES to list of indices
        in *smiles_list* that share that scaffold.
        """
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False,
                )
                groups[scaffold].append(i)
            except Exception:
                continue
        return groups

    def fit(
        self,
        actives_smiles: List[str],
        inactives_smiles: List[str],
        feature_fn: Optional[Any] = None,
        uncertainty_threshold: Optional[float] = None,
        md_ligand_rmsd_values: Optional[List[float]] = None,
        md_pocket_rg_stability_values: Optional[List[float]] = None,
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
        md_ligand_rmsd_values : list of float, optional
            Per-sample MD ligand RMSD values for training data.
        md_pocket_rg_stability_values : list of float, optional
            Per-sample MD pocket Rg stability values for training data.
        """
        total_samples = len(actives_smiles) + len(inactives_smiles)
        if total_samples < CONFIG.min_training_samples:
            raise ConfigurationError(
                f"Insufficient training data for MetaScorer "
                f"(found {total_samples}). Please ensure ChEMBL API "
                f"access or expand reference data."
            )

        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        smiles_for_scaffold: List[str] = []

        self._training_actives = list(actives_smiles)
        self._training_inactives = list(inactives_smiles)

        # Track whether any training sample had non-zero MD features
        md_values = md_ligand_rmsd_values or []
        rg_values = md_pocket_rg_stability_values or []
        has_nonzero_md = any(
            v is not None and v != 0.0
            for v in md_values + rg_values
        )
        self.uses_dynamic_features = has_nonzero_md

        # ── Load benchmark docking features ────────────────────────
        docking_features: Dict[str, Dict[str, float]] = {}
        try:
            from benchmarks.reference_data import get_benchmark_docking_features
            all_smiles = actives_smiles + inactives_smiles
            docking_features = get_benchmark_docking_features(
                actives_smiles, inactives_smiles,
                work_dir=str(CONFIG.work_dir),
            )
            log.info(
                f"MetaScorer: loaded {len(docking_features)} benchmark docking features."
            )
        except Exception as exc:
            log.warning(f"MetaScorer: failed to load benchmark docking data — {exc}")

        feature_extractor = feature_fn or self._default_features

        actives_md_rmsd = md_ligand_rmsd_values or []
        actives_rg_stab = md_pocket_rg_stability_values or []

        for i, smi in enumerate(actives_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                rmsd = actives_md_rmsd[i] if i < len(actives_md_rmsd) else None
                rg = actives_rg_stab[i] if i < len(actives_rg_stab) else None
                df = docking_features.get(smi, {})
                X_list.append(feature_extractor(
                    mol,
                    md_ligand_rmsd=rmsd,
                    md_pocket_rg_stability=rg,
                    vina_energy=df.get("vina_energy"),
                    gnina_score=df.get("gnina_score"),
                    shape_score=df.get("shape_score"),
                ))
                y_list.append(1.0)
                smiles_for_scaffold.append(smi)
            except Exception:
                continue

        inactives_md_rmsd = md_ligand_rmsd_values[len(actives_smiles):] if md_ligand_rmsd_values else []
        inactives_rg_stab = md_pocket_rg_stability_values[len(actives_smiles):] if md_pocket_rg_stability_values else []

        for i, smi in enumerate(inactives_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                idx = len(actives_smiles) + i
                rmsd = md_ligand_rmsd_values[idx] if md_ligand_rmsd_values and idx < len(md_ligand_rmsd_values) else None
                rg = md_pocket_rg_stability_values[idx] if md_pocket_rg_stability_values and idx < len(md_pocket_rg_stability_values) else None
                df = docking_features.get(smi, {})
                X_list.append(feature_extractor(
                    mol,
                    md_ligand_rmsd=rmsd,
                    md_pocket_rg_stability=rg,
                    vina_energy=df.get("vina_energy"),
                    gnina_score=df.get("gnina_score"),
                    shape_score=df.get("shape_score"),
                ))
                y_list.append(0.0)
                smiles_for_scaffold.append(smi)
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

        # ── Scaffold-split cross-validation ───────────────────────
        scaffold_r2 = float("nan")
        try:
            groups = self._scaffold_groups(smiles_for_scaffold)
            scaffold_ids = list(groups.keys())
            if len(scaffold_ids) >= 2:
                # Split scaffolds into train/test
                train_scaffolds, test_scaffolds = train_test_split(
                    scaffold_ids, test_size=0.3, random_state=CONFIG.random_seed,
                )
                train_idx = []
                test_idx = []
                for scaf in train_scaffolds:
                    train_idx.extend(groups[scaf])
                for scaf in test_scaffolds:
                    test_idx.extend(groups[scaf])

                if len(train_idx) >= 4 and len(test_idx) >= 2:
                    X_train_scaf = X[train_idx]
                    y_train_scaf = y[train_idx]
                    X_test_scaf = X[test_idx]
                    y_test_scaf = y[test_idx]

                    cv_model = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=8,
                        random_state=CONFIG.random_seed,
                    )
                    cv_model.fit(X_train_scaf, y_train_scaf)
                    y_pred_scaf = cv_model.predict(X_test_scaf)
                    scaffold_r2 = float(r2_score(y_test_scaf, y_pred_scaf))
                    log.info(
                        f"MetaScorer: scaffold-split R² = {scaffold_r2:.4f} "
                        f"(train={len(train_idx)}, test={len(test_idx)})"
                    )
        except Exception as exc:
            log.warning(f"MetaScorer: scaffold split failed ({exc}); skipping.")

        # ── Train final model on all data ──────────────────────────
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

        # ── Random-split validation ────────────────────────────────
        random_r2 = float("nan")
        try:
            X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
                X, y, test_size=0.3, random_state=CONFIG.random_seed,
            )
            rand_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=CONFIG.random_seed,
            )
            rand_model.fit(X_train_rand, y_train_rand)
            y_pred_rand = rand_model.predict(X_test_rand)
            random_r2 = float(r2_score(y_test_rand, y_pred_rand))
        except Exception:
            pass

        log.info(
            f"MetaScorer: trained on {len(X)} compounds "
            f"({len(actives_smiles)} actives / {len(inactives_smiles)} inactives), "
            f"OOB R² = {oob:.4f}, "
            f"Random-Split R² = {random_r2:.4f}, "
            f"Scaffold-Split R² = {scaffold_r2:.4f}"
        )

        # ── Optional SHAP analysis ─────────────────────────────────
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            shap_values = explainer.shap_values(X[:min(50, len(X))])
            mean_shap = np.abs(shap_values).mean(axis=0)
            top_n = min(5, len(mean_shap))
            top_feature_indices = np.argsort(mean_shap)[-top_n:][::-1]
            feat_names = self._feature_names or [
                f"feat_{i}" for i in range(X.shape[1])
            ]
            shap_info = ", ".join(
                f"{feat_names[i]}: {mean_shap[i]:.4f}"
                for i in top_feature_indices
            )
            log.info(f"MetaScorer: top {top_n} SHAP features: {shap_info}")
        except ImportError:
            pass

        self._save()
        return self

    def predict(self, record: CompoundRecord) -> Optional[float]:
        """Predict the meta-score for a single CompoundRecord.

        Returns a score in [0, 1] where higher is more likely active,
        or ``None`` if the model is not fitted or feature extraction
        fails.

        If the record contains MD-derived dynamic features
        (``md_ligand_rmsd``, ``md_pocket_rg_stability``) or IFP/Water
        features (``ifp_score``, ``water_displacement_energy``) they
        are included in the feature vector automatically.

        When ``uncertainty_threshold`` was set at fit time, the standard
        deviation of predictions across all trees is computed.  If it
        exceeds the threshold, ``record.needs_manual_review`` is set to
        ``True``.
        """
        if not self.available:
            return None
        self.validate_input_features(record)
        if record.mol is None:
            mol = Chem.MolFromSmiles(record.smiles)
            if mol is None:
                return None
            record.mol = mol
        # Warn if dynamic features are expected but missing
        if (
            self.uses_dynamic_features
            and (record.md_ligand_rmsd is None or record.md_pocket_rg_stability is None)
        ):
            log.warning(
                f"MetaScorer trained with dynamic features, but input lacks MD data. "
                f"Prediction for {record.compound_id} may be less accurate."
            )

        try:
            feats = self._default_features(
                record.mol,
                md_ligand_rmsd=record.md_ligand_rmsd,
                md_pocket_rg_stability=record.md_pocket_rg_stability,
                ifp_score=record.ifp_score,
                water_displacement_energy=record.water_displacement_energy,
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

    def predict_with_uncertainty(self, record: CompoundRecord) -> Tuple[float, float]:
        """Predict the meta-score and return prediction uncertainty.

        Returns a tuple ``(mean_score, std_dev)`` where ``mean_score``
        is the clipped prediction from the model and ``std_dev`` is the
        standard deviation of predictions across all tree estimators.

        Raises
        ------
        ValueError
            If the model is not fitted or feature extraction fails.
        """
        if not self.available:
            raise ValueError("MetaScorer is not fitted.")
        if record.mol is None:
            mol = Chem.MolFromSmiles(record.smiles)
            if mol is None:
                raise ValueError(f"Cannot parse SMILES for {record.compound_id}")
            record.mol = mol

        try:
            feats = self._default_features(
                record.mol,
                md_ligand_rmsd=record.md_ligand_rmsd,
                md_pocket_rg_stability=record.md_pocket_rg_stability,
                ifp_score=record.ifp_score,
                water_displacement_energy=record.water_displacement_energy,
            ).reshape(1, -1)
            prob = float(self._model.predict(feats)[0])  # type: ignore[union-attr]

            threshold = getattr(self, "_uncertainty_threshold", None)
            tree_preds = None
            std = 0.0
            if threshold is not None and hasattr(self._model, "estimators_"):
                tree_preds = np.array([
                    tree.predict(feats)[0] for tree in self._model.estimators_
                ])
                std = float(np.std(tree_preds, ddof=1))
                if std > threshold:
                    record.needs_manual_review = True

            return (float(np.clip(prob, 0.0, 1.0)), std)
        except Exception:
            raise ValueError("Feature extraction failed.")

    def load(self) -> bool:
        """Load a previously trained model from disk.

        Checks that the stored feature dimension matches the current
        expected dimension (13).  If mismatched, logs a warning and
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
            self.uses_dynamic_features = obj.get("uses_dynamic_features", False)

            # Backward compatibility: retrain if feature dimension has changed
            expected_dim = 13
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
                    ifp_score=record.ifp_score,
                    water_displacement_energy=record.water_displacement_energy,
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

    def explain_prediction(self, record: CompoundRecord) -> Dict[str, float]:
        """Return a dict mapping feature names to SHAP values for *record*.

        Uses :class:`shap.TreeExplainer` on the fitted model.  If ``shap``
        is not installed, logs a warning and returns an empty dict.

        Returns
        -------
        Dict[str, float]
            Feature-name → SHAP value.  Positive values push prediction
            toward the active (1) class.
        """
        if not self.available:
            log.warning("MetaScorer not fitted — cannot explain prediction.")
            return {}
        try:
            import shap
        except ImportError:
            log.warning(
                "SHAP is not installed. Install with: "
                "pip install autoantibiotic[explainability]"
            )
            return {}

        if record.mol is None:
            mol = Chem.MolFromSmiles(record.smiles)
            if mol is None:
                return {}
            record.mol = mol

        try:
            feats = self._default_features(
                record.mol,
                md_ligand_rmsd=record.md_ligand_rmsd,
                md_pocket_rg_stability=record.md_pocket_rg_stability,
                ifp_score=record.ifp_score,
                water_displacement_energy=record.water_displacement_energy,
            ).reshape(1, -1)
            explainer = shap.TreeExplainer(self._model)
            shap_values = explainer.shap_values(feats)
            # shap_values shape: (1, n_features) for single-output regressor
            sv = shap_values[0] if shap_values.ndim == 2 else shap_values
            feat_names = self._feature_names or [
                f"feat_{i}" for i in range(len(sv))
            ]
            return dict(zip(feat_names, map(float, sv)))
        except Exception as exc:
            log.warning(f"SHAP explanation failed: {exc}")
            return {}

    # ── internals ───────────────────────────────────────────────────

    def _default_features(
        self,
        mol: Chem.Mol,
        md_ligand_rmsd: Optional[float] = None,
        md_pocket_rg_stability: Optional[float] = None,
        ifp_score: Optional[float] = None,
        water_displacement_energy: Optional[float] = None,
        vina_energy: Optional[float] = None,
        gnina_score: Optional[float] = None,
        shape_score: Optional[float] = None,
    ) -> np.ndarray:
        """13-dim feature vector for a molecule.

        Real physics data (4):
        1.   Vina binding energy (kcal/mol)
        2.   GNINA CNN score (0-1)
        3.   Shape score (protrude distance normalised)
        4.   IFP similarity score (from CompoundRecord.ifp_score)

        Static descriptors (4):
        5.   QED
        6.   LogP
        7.   MolWt
        8.   NumRotatableBonds

        Dynamic MD features (3), default 0.0 when unavailable:
        9.  ligand_rmsd_mean
        10. ligand_rmsd_std
        11. pocket_rg_stability

        IFP / Water features (2), default 0.0 when unavailable:
        12. IFP similarity score (from CompoundRecord.ifp_score)
        13. Water displacement energy (from CompoundRecord.water_displacement_energy)
        """
        qed = float(QED.qed(mol))
        logp = float(Crippen.MolLogP(mol))
        mw = float(Descriptors.MolWt(mol))
        n_rot = float(Descriptors.NumRotatableBonds(mol))

        vina = vina_energy if vina_energy is not None else 0.0
        gnina = gnina_score if gnina_score is not None else 0.0
        shape = shape_score if shape_score is not None else 0.0

        ifp = ifp_score if ifp_score is not None else 0.0
        water_disp = water_displacement_energy if water_displacement_energy is not None else 0.0

        rmsd_mean = md_ligand_rmsd if md_ligand_rmsd is not None else 0.0
        rmsd_std = 0.0
        rg_stab = md_pocket_rg_stability if md_pocket_rg_stability is not None else 0.0

        arr = np.array(
            [vina, gnina, shape, ifp, qed, logp, mw, n_rot,
             rmsd_mean, rmsd_std, rg_stab, ifp, water_disp],
            dtype=np.float32,
        )
        self._feature_names = [
            "vina_energy", "gnina_score", "shape_score", "ifp_score",
            "qed", "logp", "mw", "n_rotatable",
            "ligand_rmsd_mean", "ligand_rmsd_std", "pocket_rg_stability",
            "ifp_score_complex", "water_displacement_energy",
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
                    "uses_dynamic_features": self.uses_dynamic_features,
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
