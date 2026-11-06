import unittest

import numpy as np
import pandas as pd

from mi_stage10_calibration_comparison import (
    METHODS, calibration_diagnostics, quality_metrics,
    quantile_reliability_rows, recommend_method,
)
from mi_stage10_calibrators import (
    BetaCalibrator, IsotonicCalibrator, PlattCalibrator, TemperatureCalibrator,
)


class CalibrationHelperTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 0, 0, 1, 1, 1, 0, 1] * 5, dtype=int)
        self.p = np.array([0.08, 0.18, 0.34, 0.58, 0.73, 0.91, 0.27, 0.82] * 5)

    def test_platt_outputs_probabilities(self):
        model = PlattCalibrator().fit(self.p, self.y)
        out = model.predict(self.p)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        self.assertTrue(np.all(np.diff(out[np.argsort(self.p)]) >= -1e-10))

    def test_isotonic_outputs_monotone_probabilities(self):
        model = IsotonicCalibrator().fit(self.p, self.y)
        out = model.predict(np.linspace(0, 1, 101))
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        self.assertTrue((np.diff(out) >= -1e-12).all())

    def test_beta_coefficients_nonnegative_and_mapping_monotone(self):
        model = BetaCalibrator().fit(self.p, self.y)
        self.assertGreaterEqual(model.coef_[0], 0)
        self.assertGreaterEqual(model.coef_[1], 0)
        out = model.predict(np.linspace(0, 1, 101))
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        self.assertTrue((np.diff(out) >= -1e-10).all())

    def test_temperature_is_positive_and_outputs_probabilities(self):
        model = TemperatureCalibrator().fit(self.p, self.y)
        self.assertGreater(model.temperature_, 0)
        out = model.predict(np.linspace(0, 1, 101))
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        self.assertTrue((np.diff(out) >= -1e-10).all())

    def test_quality_and_calibration_diagnostics_are_finite(self):
        metrics = quality_metrics(self.y, self.p)
        diagnostics = calibration_diagnostics(self.y, self.p)
        self.assertGreaterEqual(metrics["average_precision"], 0)
        self.assertLessEqual(metrics["average_precision"], 1)
        self.assertTrue(np.isfinite(list(diagnostics.values())).all())
        self.assertGreaterEqual(diagnostics["ece_10_equal_width"], 0)
        fitted = 1 / (1 + np.exp(-(np.log(self.p / (1 - self.p)) +
                                  diagnostics["calibration_in_large_intercept"])))
        self.assertAlmostEqual(float(fitted.mean()), float(self.y.mean()), places=6)

    def test_quantile_reliability_bins_cover_records_without_ids(self):
        rows = quantile_reliability_rows("ZSN", "Raw", self.y, self.p, n_bins=10)
        self.assertLessEqual(len(rows), 10)
        self.assertEqual(sum(r["n"] for r in rows), len(self.y))
        self.assertTrue(all("ID" not in r for r in rows))
        self.assertTrue(all(0 <= r["mean_predicted_probability"] <= 1 for r in rows))

    def test_recommendation_prefers_platt_when_differences_are_mixed(self):
        pooled = pd.DataFrame([
            {"target": "T", "method": method, "brier_score": score, "log_loss": loss}
            for method, score, loss in [("Raw", .1, .2), ("Platt", .099, .199),
                                        ("Isotonic", .099, .201), ("Beta", .101, .198),
                                        ("Temperature", .1, .2)]
        ])
        folds = pd.DataFrame([
            {"target": "T", "method": method, "outer_fold": fold,
             "brier_score": score, "log_loss": loss}
            for method in METHODS for fold in range(1, 6)
            for score, loss in [(dict(Raw=(.10, .20), Platt=(.10, .20), Isotonic=(.09, .21),
                                      Beta=(.11, .19), Temperature=(.10, .20))[method])]
        ])
        result = recommend_method(folds, pooled, "T")
        self.assertEqual(result["recommended_method"], "Platt")
        self.assertIn("inconclusive", result["decision_status"])

    def test_recommendation_allows_stable_optional_improvement(self):
        pooled_scores = {"Raw": (.11, .22), "Platt": (.10, .20), "Isotonic": (.097, .194),
                         "Beta": (.096, .193), "Temperature": (.10, .20)}
        pooled = pd.DataFrame([{"target": "T", "method": m,
                                "brier_score": pair[0], "log_loss": pair[1]}
                               for m, pair in pooled_scores.items()])
        folds = pd.DataFrame([{"target": "T", "method": m, "outer_fold": fold,
                               "brier_score": pooled_scores[m][0], "log_loss": pooled_scores[m][1]}
                              for m in METHODS for fold in range(1, 6)])
        result = recommend_method(folds, pooled, "T")
        self.assertIn(result["recommended_method"], {"Isotonic", "Beta"})


if __name__ == "__main__":
    unittest.main()
