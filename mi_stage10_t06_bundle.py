"""Importable inference bundle for T06 final models."""
from __future__ import annotations

import numpy as np
import pandas as pd


class FinalModelBundle:
    """Joblib-safe estimator/calibrator pair with saved TabPFN category handling."""

    def __init__(self, target: str, model_name: str, estimator, calibrator,
                 feature_columns: list[str], category_schema: dict[str, list] | None,
                 metadata: dict):
        self.target = target
        self.model_name = model_name
        self.estimator = estimator
        self.calibrator = calibrator
        self.feature_columns = list(feature_columns)
        self.category_schema = category_schema
        self.metadata = metadata

    def predict_base_probability(self, X: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_columns).difference(X.columns)
        if missing:
            raise ValueError(f"Inference data missing model features: {sorted(missing)}")
        inputs = X.loc[:, self.feature_columns]
        if self.model_name == "TabPFN 3.5":
            if self.category_schema is None:
                raise RuntimeError("TabPFN bundle lacks its saved native category schema.")
            inputs = inputs.copy()
            for column, categories in self.category_schema.items():
                if column not in inputs:
                    raise ValueError(f"Inference data missing categorical feature: {column}")
                inputs[column] = pd.Categorical(inputs[column], categories=categories)
        probability = np.asarray(self.estimator.predict_proba(inputs), dtype=float)
        classes = np.asarray(self.estimator.classes_)
        positive = np.flatnonzero(classes == 1)
        if len(positive) != 1 or probability.ndim != 2 or probability.shape[1] != len(classes):
            raise ValueError("Finalist did not return a valid positive-class probability column.")
        result = probability[:, int(positive[0])]
        if result.ndim != 1 or len(result) != len(inputs) or not np.isfinite(result).all():
            raise ValueError("Finalist returned malformed or non-finite probabilities.")
        if ((result < 0) | (result > 1)).any():
            raise ValueError("Finalist returned probabilities outside [0, 1].")
        return result

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.calibrator.predict(self.predict_base_probability(X)), dtype=float,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = self.predict_probability(X)
        return np.column_stack([1 - positive, positive])
