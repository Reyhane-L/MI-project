import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
