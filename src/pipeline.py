from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model import CampaignModel
from src.utils.logger import get_logger

def run_training(data_path: str, model_path: str, prep_path: str, model_type: str = 'gbm') -> None:
    """Execute the full training pipeline for campaign profitability models.

    :param data_path: Location of the raw campaign dataset.
    :param model_path: Destination where the trained model artifact is stored.
    :param prep_path: Destination where the fitted preprocessor is stored.
    :param model_type: Identifier of the estimator to train (e.g., ``'gbm'``, ``'ada'``, or ``'tree_ensemble'``).
    """
    log = get_logger("pipeline")
    log.info(f"Loading data from {data_path}")
    df = DataLoader(data_path).load()

    def _within_numeric_suffix(col: str, prefix: str, limit: int) -> bool:
        """Check whether a feature name matches the expected ``prefix_index`` format.

        :param col: Column name being evaluated.
        :param prefix: Feature prefix (e.g., ``'g1_'``).
        :param limit: Highest allowed numeric suffix for the prefix.
        :returns: True when the column name respects the prefix and numeric bounds.
        """
        if not col.startswith(prefix):
            return False
        suffix = col.split('_', 1)[1] if '_' in col else ""
        return suffix.isdigit() and int(suffix) <= limit

    feature_cols = [
        c for c in df.columns
        if _within_numeric_suffix(c, 'g1_', 20)
        or _within_numeric_suffix(c, 'g2_', 20)
        or _within_numeric_suffix(c, 'c_', 27)
    ]

    X = df[feature_cols].copy()
    y = df['target']

    # Remove strongly correlated predictors before mutual information scoring.
    corr_threshold = 0.8
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features: List[str] = [col for col in upper_triangle.columns if any(upper_triangle[col] > corr_threshold)]
    if high_corr_features:
        log.info("Dropping %d highly correlated features.", len(high_corr_features))
        X = X.drop(columns=high_corr_features)

    # Score remaining features with mutual information and keep the most informative ones.
    importance_quantile = 0.75
    mi_scores = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)
    mi_threshold = mi_scores.quantile(importance_quantile)
    selected_features = mi_scores[mi_scores >= mi_threshold].index.tolist()
    if not selected_features:
        fallback_k = min(5, len(mi_scores))
        selected_features = mi_scores.sort_values(ascending=False).head(fallback_k).index.tolist()
        log.warning(
            "No features cleared MI quantile %.2f; using top %d scores instead.",
            importance_quantile,
            fallback_k)

    log.info(
        "Selected %d features via mutual information thresholding (cutoff=%.4f).",
        len(selected_features),
        mi_threshold,
    )
    log.debug("Selected features: %s", selected_features)
    X = X[selected_features]

    log.info("Preprocessing data...")
    pre = Preprocessor()
    X_prep = pre.fit_transform(X)

    # Persist the feature ordering so downstream components can align inputs.
    pre.feature_names_ = list(X.columns)

    log.info(f"Training model: {model_type}")
    model = CampaignModel(model_type)
    model.train(X_prep, y)

    model.save(model_path)
    pre.save(prep_path)
    log.info(f"Model and preprocessor saved to {model_path}, {prep_path}")
