import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

import mi_undersampling_probability_experiment as experiment


class PriorCorrectionTests(unittest.TestCase):
    def test_prior_correction_maps_sampled_prior_to_natural_prior(self):
        corrected = experiment.prior_correct([0.5], natural_prevalence=0.2)
        self.assertAlmostEqual(float(corrected[0]), 0.2)

    def test_prior_correction_preserves_order_and_boundaries(self):
        raw = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
        corrected = experiment.prior_correct(raw, natural_prevalence=0.15)
        self.assertTrue(np.all(np.diff(corrected) > 0))
        self.assertEqual(corrected[0], 0)
        self.assertEqual(corrected[-1], 1)

    def test_prior_correction_rejects_invalid_prevalence(self):
        with self.assertRaises(ValueError):
            experiment.prior_correct([0.2], natural_prevalence=0)


class ResamplingTests(unittest.TestCase):
    def test_undersampling_is_deterministic_and_balanced(self):
        y = np.array([0] * 12 + [1] * 4)
        first = experiment.undersample_indices(y, 19)
        second = experiment.undersample_indices(y, 19)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 8)
        self.assertEqual(int(y[first].sum()), 4)

    def test_undersampling_requires_both_classes(self):
        with self.assertRaises(ValueError):
            experiment.undersample_indices(np.zeros(5, dtype=int), 1)

    def test_calibration_split_is_stratified_disjoint_and_exhaustive(self):
        y = np.array([0] * 80 + [1] * 20)
        model_fit, calibration = experiment.calibration_split_positions(y, 17)
        self.assertEqual(len(np.intersect1d(model_fit, calibration)), 0)
        self.assertEqual(set(np.concatenate([model_fit, calibration])), set(range(len(y))))
        self.assertAlmostEqual(float(y[calibration].mean()), 0.2, delta=0.03)


class MetricTests(unittest.TestCase):
    def test_ece_and_metric_contract(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = experiment.probability_metrics(y, p)
        self.assertAlmostEqual(metrics["observed_prevalence"], 0.5)
        self.assertAlmostEqual(metrics["mean_predicted_probability"], 0.5)
        self.assertAlmostEqual(metrics["brier_score"], 0.025)
        self.assertAlmostEqual(metrics["ece_quantile_10"], 0.15)
        self.assertGreater(metrics["calibration_slope"], 0)


class OutputTests(unittest.TestCase):
    def test_writer_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "occupied"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                experiment.write_outputs({}, output)


class _FakeEstimator:
    def fit(self, X, y):
        self.probability = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.probability)
        return np.column_stack([1 - p, p])


class _FakeStage2:
    targets = ["DEATH_BINARY"]
    random_state = experiment.SEED
    n_splits = experiment.FOLDS
    continuous_features = []
    ordinal_features = []
    nominal_features = []

    @staticmethod
    def build_pipeline(*args):
        return _FakeEstimator()


class CrossFitTests(unittest.TestCase):
    def test_calibration_and_validation_partitions_are_audited(self):
        with mock.patch.object(experiment, "load_stage2_api", return_value=_FakeStage2()):
            y = np.array([0] * 120 + [1] * 30)
            X = pd.DataFrame({"feature": np.linspace(0, 1, len(y))})
            y_all = pd.DataFrame({"DEATH_BINARY": y})
            result = experiment.run_experiment(X, y_all)
        audit = result["fold_audit"]
        self.assertEqual(len(audit), experiment.FOLDS)
        for flag in ("model_fit_calibration_disjoint", "calibration_validation_disjoint",
                     "model_fit_validation_disjoint", "resampled_calibration_disjoint"):
            self.assertTrue(audit[flag].all())
        self.assertTrue(np.allclose(audit.calibration_prevalence, 0.2, atol=0.03))
        self.assertEqual(set(result["summary_metrics"].method), set(experiment.METHODS))
        self.assertEqual(len(result["predictions"]), len(y))
        self.assertTrue(result["predictions"].platt_probability.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
