from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import mi_stage13_t09_mortality_shap as t09


class MortalityShapHelperTests(unittest.TestCase):
    def test_feature_validation_accepts_exact_development_schema(self):
        frame = pd.DataFrame({"AGE": [60, 70], "SEX": [1, 0]})
        t09.validate_development_features(frame, ["AGE", "SEX"])

    def test_feature_validation_rejects_order_mismatch_and_target_leakage(self):
        with self.assertRaisesRegex(ValueError, "names/order"):
            t09.validate_development_features(
                pd.DataFrame({"SEX": [0], "AGE": [55]}), ["AGE", "SEX"],
            )
        with self.assertRaisesRegex(ValueError, "forbidden"):
            t09.validate_development_features(
                pd.DataFrame({"AGE": [55], "DEATH_BINARY": [1]}),
                ["AGE", "DEATH_BINARY"],
            )

    def test_summary_excludes_base_value_and_sorts_by_absolute_contribution(self):
        values = np.array([
            [1.0, -4.0, 9.0],
            [-3.0, 2.0, 8.0],
        ])
        summary = t09.summarize_shap_values(values, ["small", "large"])
        self.assertEqual(summary.feature.tolist(), ["large", "small"])
        self.assertEqual(summary.mean_absolute_shap.tolist(), [3.0, 2.0])
        self.assertEqual(summary.mean_signed_shap.tolist(), [-1.0, -1.0])

    def test_summary_rejects_invalid_shape_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "one column per feature"):
            t09.summarize_shap_values(np.ones((2, 2)), ["a", "b"])
        broken = np.ones((2, 2))
        broken[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            t09.summarize_shap_values(broken, ["a"])

    def test_summary_plot_writes_expected_image_from_synthetic_values(self):
        values = np.array([
            [0.3, -0.2, 0.1], [-0.1, 0.4, -0.3], [0.2, -0.1, 0.05],
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.png"
            t09.write_summary_plot(values, ["AGE", "SEX"], output, max_features=2)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)

    def test_explanation_defines_features_and_cautiously_describes_shap(self):
        summary = pd.DataFrame({
            "feature": [f"f{i}" for i in range(10)],
            "mean_absolute_shap": np.linspace(1, 0.1, 10),
            "mean_signed_shap": np.zeros(10),
        })
        dictionary = pd.DataFrame({
            "variable": [f"f{i}" for i in range(10)],
            "description": [f"Definition {i}" for i in range(10)],
        })
        explanation = t09.build_explanation(summary, dictionary, 4)
        self.assertIn("Definition 0", explanation)
        self.assertIn("Positive SHAP values raise", explanation)
        self.assertIn("negative values lower", explanation)
        self.assertIn("heterogeneous contributions", explanation)
        self.assertIn("not causal", explanation)
        self.assertNotIn("\\n", explanation)


if __name__ == "__main__":
    unittest.main()
