from typing import Union

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """Preprocess features: impute missing values and scale.
    computes the median for each feature column and uses it to fill missing values.
    Then standardizes the features to have zero mean and unit variance.
    """
    def __init__(self) -> None:
        """Initialize imputers and scalers used for preprocessing."""
        self.imp = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame) -> None:
        """Fit the imputing and scaling steps.

        :param X: Training feature matrix used to learn medians and scaling parameters.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = None
        self.imp.fit(X)
        self.scaler.fit(self.imp.transform(X))
        self.fitted = True

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply the learned imputing and scaling transformations.

        :param X: Dataset to transform; accepts DataFrame or numpy array.
        :returns: Transformed data with the same column ordering as during fit.
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted.")
        feature_names = getattr(self, "feature_names_", None)
        is_df = isinstance(X, pd.DataFrame)
        index = X.index if is_df else None

        if is_df:
            if feature_names is None:
                feature_names = list(X.columns)
                self.feature_names_ = feature_names
            X = X[feature_names]

        X_imp = self.imp.transform(X)
        X_scaled = self.scaler.transform(X_imp)

        if feature_names is not None:
            return pd.DataFrame(X_scaled, columns=feature_names, index=index)

        return X_scaled

    def fit_transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """Fit the preprocessing steps and immediately transform the input.

        :param X: Dataset used to both fit and transform the preprocessing pipeline.
        :returns: Transformed dataset after fitting.
        """
        self.fit(X)
        return self.transform(X)

    def save(self, path: str) -> None:
        """Persist the preprocessor object for reuse.

        :param path: Destination path where the preprocessor is serialized.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Preprocessor":
        """Load a serialized preprocessor instance.

        :param path: Source path pointing to the serialized preprocessor file.
        :returns: Restored ``Preprocessor`` instance with feature metadata populated.
        """
        preprocessor = joblib.load(path)
        if not hasattr(preprocessor, "feature_names_"):
            preprocessor.feature_names_ = []
        if preprocessor.feature_names_ is None:
            preprocessor.feature_names_ = []
        return preprocessor
