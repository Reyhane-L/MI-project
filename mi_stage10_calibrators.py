"""Serializable calibration estimators used by Stage 10 and later stages."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

SEED = 42
EPS = 1e-6


def clipped_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)


def probability_logit(probability: np.ndarray) -> np.ndarray:
    p = clipped_probability(probability)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    def fit(self, probability: np.ndarray, y: np.ndarray):
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.asarray(probability, dtype=float)


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=SEED)

    def fit(self, probability: np.ndarray, y: np.ndarray):
        self.model.fit(probability_logit(probability).reshape(-1, 1), np.asarray(y, dtype=int))
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(probability_logit(probability).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probability: np.ndarray, y: np.ndarray):
        self.model.fit(clipped_probability(probability), np.asarray(y, dtype=int))
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(clipped_probability(probability)), dtype=float)


class BetaCalibrator:
    """Monotone beta calibration with nonnegative shape coefficients."""
    def __init__(self):
        self.coef_: np.ndarray | None = None

    @staticmethod
    def features(probability: np.ndarray) -> np.ndarray:
        p = clipped_probability(probability)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, probability: np.ndarray, y: np.ndarray):
        x = self.features(probability)
        labels = np.asarray(y, dtype=float)
        design = np.column_stack([x, np.ones(len(x))])

        def objective(theta):
            eta = design @ theta
            pred = 1 / (1 + np.exp(-np.clip(eta, -40, 40)))
            value = np.logaddexp(0, eta).sum() - np.dot(labels, eta)
            grad = design.T @ (pred - labels)
            return float(value), grad

        result = minimize(objective, np.array([1.0, 1.0, 0.0]), method="L-BFGS-B", jac=True,
                          bounds=[(0.0, None), (0.0, None), (None, None)],
                          options={"maxiter": 2000, "ftol": 1e-12})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Constrained beta calibration failed: {result.message}")
        self.coef_ = result.x
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("BetaCalibrator must be fitted before predict().")
        eta = self.features(probability) @ self.coef_[:2] + self.coef_[2]
        return 1 / (1 + np.exp(-np.clip(eta, -40, 40)))


class TemperatureCalibrator:
    """One-parameter temperature scaling on logit(p), without an intercept."""
    def __init__(self):
        self.temperature_: float | None = None

    def fit(self, probability: np.ndarray, y: np.ndarray):
        z = probability_logit(probability)
        labels = np.asarray(y, dtype=float)

        def objective(log_temperature: float) -> float:
            scaled = z / np.exp(log_temperature)
            return float(np.mean(np.logaddexp(0, scaled) - labels * scaled))

        result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded",
                                 options={"xatol": 1e-10, "maxiter": 1000})
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError(f"Temperature scaling fit failed: {result.message}")
        self.temperature_ = float(np.exp(result.x))
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureCalibrator must be fitted before predict().")
        z = probability_logit(probability) / self.temperature_
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def make_calibrator(method: str):
    choices = {
        "Raw": IdentityCalibrator,
        "Platt": PlattCalibrator,
        "Isotonic": IsotonicCalibrator,
        "Beta": BetaCalibrator,
        "Temperature": TemperatureCalibrator,
    }
    if method not in choices:
        raise ValueError(f"Unknown calibration method: {method}")
    return choices[method]()
