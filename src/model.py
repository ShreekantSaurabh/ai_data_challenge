from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.utils.logger import get_logger

class CampaignModel:
    """Wrapper class supporting Dummy, single-model boosters and a tree ensemble."""
    def __init__(self, model_type="rf", balance_classes=True, params=None) -> None:
        """Create the configurable campaign model wrapper.

        :param model_type: Identifier for the underlying estimator (dummy/rf/gbm/xgb/cat/ada/tree_ensemble).
        :param balance_classes: Whether to compute class-balanced weights during training.
        :param params: Optional parameter overrides passed to the estimator.
        """
        if model_type == "dummy":
            self.model = DummyClassifier(strategy="most_frequent")
        elif model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_split=4,
                n_jobs=-1, random_state=42)
        elif model_type == "ada":
            self.model = AdaBoostClassifier(
                n_estimators=400,
                learning_rate=0.05,
                random_state=42,
            )
        elif model_type =="gbm":
            self.model = GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42,
            )
        elif model_type == "xgb":
            self.model = XGBClassifier(
                n_estimators=500, learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric='mlogloss')
        elif model_type == "cat":
            self.model = CatBoostClassifier(
                iterations=600,
                learning_rate=0.05,
                depth=6,
                loss_function="MultiClass",
                random_seed=42,
                verbose=False,
                allow_writing_files=False)
        elif model_type == "tree_ensemble":
            rf = RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=3,
                n_jobs=-1,
                random_state=42,
            )
            ada = AdaBoostClassifier(
                n_estimators=400,
                learning_rate=0.05,
                random_state=42, 
                algorithm="SAMME.R",
            )
            gbm = GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42,
            )
            xgb = XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
            )
            cat = CatBoostClassifier(
                iterations=500,
                learning_rate=0.04,
                depth=6,
                loss_function="MultiClass",
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            )
            self.model = VotingClassifier(
                estimators=[
                    ("rf", rf),
                    ("ada", ada),
                    ("gbm", gbm),
                    ("xgb", xgb),
                    ("cat", cat),
                ],
                voting="soft",
                weights=[1, 1, 2, 2, 2],
                n_jobs=-1,
            )
        else:
            raise ValueError(
                "Invalid model_type. Choose dummy, rf, gbm, xgb, cat, ada or tree_ensemble."
            )
        self.model_type = model_type
        self.balance_classes = balance_classes
        self.class_weight_ = None

        if params:
            self.model.set_params(**params)

    def _compute_sample_weights(self, y) -> Tuple[Optional[Dict[int, float]], Optional[np.ndarray]]:
        """Derive class weights and sample weights for balancing.
        Done to counter class imbalance before any model fitting happens. 
        If balancing is turned on and you really have more than one class, it computes inverse-frequency weights 
        so that rare classes influence the loss as much as majority classes. It prevents minority classes from being ignored.

        :param y: Target vector used to compute class-frequency-based weights.
        :returns: Tuple of (class_weight mapping, sample_weight array) if balancing is enabled.
        """
        if not self.balance_classes or len(np.unique(y)) <= 1:
            return None, None

        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = dict(zip(classes, weights))
        sample_weight = np.array([class_weight[label] for label in y])
        return class_weight, sample_weight

    def train(self, X, y) -> None:
        """Fit the underlying estimator with optional cross-validated reporting.

        :param X: Feature matrix for training.
        :param y: Target labels aligned with ``X`` rows.
        """
        y = np.asarray(y)

        def _subset(data, idx):
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                return data.iloc[idx]
            return data[idx]

        class_weight, sample_weight = self._compute_sample_weights(y)

        if class_weight is not None:
            if self.model_type == "rf":
                if hasattr(self.model, "set_params"):
                    self.model.set_params(class_weight=class_weight)
            elif self.model_type == "cat":
                ordered_weights = [class_weight[c] for c in sorted(class_weight.keys())]
                if hasattr(self.model, "set_params"):
                    self.model.set_params(class_weights=ordered_weights)
            elif self.model_type == "tree_ensemble":
                ordered_weights = [class_weight[c] for c in sorted(class_weight.keys())]
                updated_estimators = []
                for name, estimator in self.model.estimators:
                    if name == "rf" and hasattr(estimator, "set_params"):
                        estimator.set_params(class_weight=class_weight)
                    elif name == "cat" and hasattr(estimator, "set_params"):
                        estimator.set_params(class_weights=ordered_weights)
                    updated_estimators.append((name, estimator))
                self.model.estimators = updated_estimators

        # Perform stratified 5-fold cross-validation and log F1_macro score
        # StratifiedKFold is done because the target distribution is imbalanced, 
        # a plain KFold could produce folds missing some classes. 
        # StratifiedKFold preserves the original class proportions in every split, 
        # so each validation fold contains all classes and the macro F1 averages are meaningful.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        X_for_split = np.asarray(X) if not hasattr(X, "iloc") else X

        for train_idx, val_idx in skf.split(X_for_split, y):
            estimator = clone(self.model)
            fit_params = {}
            if sample_weight is not None:
                fit_params["sample_weight"] = sample_weight[train_idx]

            X_train_fold = _subset(X, train_idx)
            X_val_fold = _subset(X, val_idx)

            estimator.fit(X_train_fold, y[train_idx], **fit_params)
            preds = estimator.predict(X_val_fold)
            fold_scores.append(f1_score(y[val_idx], preds, average="macro"))

        logger = get_logger("model")
        logger.info(f"[{self.model_type}] CV F1_macro: {np.mean(fold_scores):.4f}")

        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self.model.fit(X, y, **fit_params)
        self.class_weight_ = class_weight

    def predict(self, X) -> np.ndarray:
        """Generate predictions with the fitted estimator.

        :param X: Feature matrix to score.
        :returns: Array of predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return class probability estimates when the estimator supports it."""
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Underlying model does not expose predict_proba.")
        return self.model.predict_proba(X)

    def save(self, path) -> None:
        """Persist the trained estimator to disk.

        :param path: Destination path for the serialized model artifact.
        """
        joblib.dump(self.model, path)
