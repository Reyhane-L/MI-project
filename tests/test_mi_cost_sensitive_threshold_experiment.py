import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import mi_cost_sensitive_threshold_experiment as experiment


class CostThresholdTests(unittest.TestCase):
    def test_metrics_and_cost(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.4, 0.3, 0.1])
        metrics = experiment._threshold_metrics(y, p, 0.4)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["cost_fn10_fp1"], 0)
        self.assertEqual(metrics["true_deaths_identified_per_100"], 50.0)
        self.assertEqual(metrics["sensitivity"], 1.0)

    def test_minimum_cost_tie_uses_highest_threshold(self):
        # All-negative sentinel costs 10. Threshold 0.7 recovers one death
        # while adding ten false positives, also costing 10; choose sentinel.
        rows = pd.DataFrame({
            "ID": np.arange(11),
            "true_label": np.array([1] + [0] * 10),
            "outer_fold": [1] * 11,
            "platt_probability": np.array([0.7] + [0.8] * 10),
        })
        result = experiment.analyze(rows, frozen_threshold=0.5)
        self.assertEqual(result["comparison"].iloc[0].cost_fn10_fp1, 10)
        self.assertGreater(result["optimal_threshold"], 0.8)

    def test_csv_round_trip_preserves_all_negative_sentinel(self):
        rows = pd.DataFrame({
            "ID": np.arange(11),
            "true_label": np.array([1] + [0] * 10),
            "outer_fold": [1] * 11,
            "platt_probability": np.array([0.7] + [0.8] * 10),
        })
        result = experiment.analyze(rows, frozen_threshold=0.5)
        with tempfile.TemporaryDirectory() as parent:
            out = Path(parent) / "roundtrip"
            experiment.write_outputs(result, rows, 0.5, out)
            curve = pd.read_csv(out / experiment.OUTPUTS[0])
        self.assertEqual(curve.threshold.nunique(), len(result["curve"]))
        sentinel = np.nextafter(float(rows.platt_probability.max()), np.inf)
        sentinel_row = curve.loc[curve.threshold == sentinel]
        self.assertEqual(len(sentinel_row), 1)
        self.assertEqual(int(sentinel_row.iloc[0].true_negatives), 10)
        self.assertEqual(int(sentinel_row.iloc[0].false_positives), 0)
        self.assertEqual(int(sentinel_row.iloc[0].false_negatives), 1)
        self.assertEqual(int(sentinel_row.iloc[0].true_positives), 0)

    def test_validation_requires_selected_ids_unique(self):
        n = 1360
        oof = pd.DataFrame({
            "target": ["DEATH_BINARY"] * n,
            "model": ["CatBoost tuned"] * n,
            "ID": np.arange(n),
            "true_label": np.tile([0, 1], n // 2),
            "outer_fold": np.tile([1, 2, 3, 4, 5], n // 5)[:n],
            "platt_probability": np.linspace(0, 1, n),
        })
        frozen = pd.DataFrame({
            "target": ["DEATH_BINARY"], "threshold": [0.15333595513662462],
            "status": ["constraint_met"], "selection_criterion": ["test"],
        })
        rows, threshold = experiment._validate_inputs(oof, frozen)
        self.assertEqual(len(rows), n)
        self.assertAlmostEqual(threshold, 0.15333595513662462)
        oof.loc[1, "ID"] = 0
        with self.assertRaises(ValueError):
            experiment._validate_inputs(oof, frozen)

    def test_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as parent:
            out = Path(parent) / "existing"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                experiment.write_outputs({}, pd.DataFrame(), 0.5, out)


if __name__ == "__main__":
    unittest.main()
