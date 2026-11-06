import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

import mi_stage11_t07_evaluation as t07


class MockBundle:
    target = "DEATH_BINARY"
    model_name = "Mock model"
    feature_columns = ["feature_a", "feature_b"]
    metadata = {
        "target": "DEATH_BINARY",
        "model": "Mock model",
        "calibration_method": "Platt",
        "calibration_parameters": {"method": "Platt"},
        "threshold": 0.5,
    }

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)
        self.fit = Mock(side_effect=AssertionError("T07 must not fit the bundle."))

    def predict_probability(self, X):
        self.asserted_rows = len(X)
        return self.probabilities


class T07EvaluationTests(unittest.TestCase):
    def test_import_has_no_input_access_bundle_loading_or_prediction(self):
        with patch.object(t07.pd, "read_csv") as read_csv, patch.object(
            t07.joblib, "load",
        ) as load:
            importlib.reload(t07)
        read_csv.assert_not_called()
        load.assert_not_called()

    def test_missing_approval_exits_before_data_bundle_or_output_access(self):
        with patch.object(t07, "run_evaluation") as run, patch.object(
            t07.pd, "read_csv",
        ) as read_csv, patch.object(t07.joblib, "load") as load:
            with self.assertRaises(SystemExit) as error:
                t07.main([])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()
        read_csv.assert_not_called()
        load.assert_not_called()

    def test_direct_runner_requires_approval_before_any_path_access(self):
        with patch.object(t07.Path, "exists") as exists, patch.object(
            t07.Path, "is_file",
        ) as is_file, patch.object(t07.pd, "read_csv") as read_csv, patch.object(
            t07.joblib, "load",
        ) as load:
            with self.assertRaises(PermissionError):
                t07.run_evaluation()
        exists.assert_not_called()
        is_file.assert_not_called()
        read_csv.assert_not_called()
        load.assert_not_called()

    def test_validates_inputs_and_rejects_forbidden_predictors(self):
        X = pd.DataFrame({"feature_a": [0.0, 1.0], "feature_b": [1.0, 0.0]})
        y = pd.DataFrame({target: [0, 1] for target in t07.TARGETS})
        y["LET_IS"] = [0, 1]
        ids = pd.DataFrame({"ID": ["a", "b"]})
        t07._validate_test_inputs(X, y, ids)

        forbidden = X.assign(ID=[1, 2])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            t07._validate_test_inputs(forbidden, y, ids)

    def test_evaluates_frozen_probability_threshold_and_metrics_without_fit(self):
        labels = np.array([0, 1, 0, 1])
        probability = np.array([0.1, 0.8, 0.4, 0.6])
        ids = np.array(["r1", "r2", "r3", "r4"])
        bundle = MockBundle(probability)
        predicted = t07._probability_vector(
            bundle, pd.DataFrame({"feature_a": range(4), "feature_b": range(4)}),
            "DEATH_BINARY",
        )
        result, confusion, rows, diagnostics, bins = t07.evaluate_target(
            "DEATH_BINARY", labels, predicted, 0.5, ids,
        )

        self.assertEqual([confusion[k] for k in ("tn", "fp", "fn", "tp")], [2, 0, 0, 2])
        self.assertEqual(result["frozen_threshold"], 0.5)
        self.assertAlmostEqual(result["brier_skill_score"], 0.63)
        self.assertAlmostEqual(result["brier_baseline_score"], 0.25)
        self.assertEqual(result["sensitivity"], 1.0)
        self.assertEqual(result["specificity"], 1.0)
        self.assertEqual(rows["ID"].tolist(), ids.tolist())
        self.assertEqual(rows["predicted_label"].tolist(), [0, 1, 0, 1])
        self.assertTrue(diagnostics["diagnostic_only"])
        self.assertGreaterEqual(diagnostics["ece_10_equal_width"], 0)
        self.assertGreater(diagnostics["calibration_slope_standard_error"], 100)
        self.assertEqual(set(bins["binning"]), {"equal_width", "quantile"})
        bundle.fit.assert_not_called()

    def test_failed_calibration_slope_diagnostic_does_not_abort_evaluation(self):
        labels = np.array([0, 1, 0, 1])
        probability = np.array([0.1, 0.8, 0.4, 0.6])
        failed = OptimizeResult(success=False, message="synthetic optimizer failure", x=np.array([np.nan, np.nan]))
        with patch.object(t07, "minimize", return_value=failed):
            metrics, _, _, diagnostics, _ = t07.evaluate_target(
                "DEATH_BINARY", labels, probability, 0.5, np.array(["a", "b", "c", "d"]),
            )
        self.assertTrue(np.isfinite(metrics["auroc"]))
        self.assertTrue(np.isfinite(metrics["brier_score"]))
        self.assertEqual(diagnostics["calibration_slope_status"], "unavailable")
        self.assertIn("synthetic optimizer failure", diagnostics["calibration_slope_detail"])

    def test_saved_split_row_count_is_checked_without_recreating_split(self):
        summary = pd.DataFrame({
            "partition": ["train", "test"],
            "patients": [8, 2],
        })
        t07._validate_split_summary(summary, 2)
        with self.assertRaisesRegex(ValueError, "row count"):
            t07._validate_split_summary(summary, 3)
        fractional = pd.DataFrame({"partition": ["train", "test"], "patients": [8.0, 2.5]})
        fractional.loc[fractional["partition"] == "test", "patients"] = 2.5
        with self.assertRaisesRegex(ValueError, "row count"):
            t07._validate_split_summary(fractional, 2)

    def test_missing_inputs_and_required_columns_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.csv"
            with patch.object(t07, "INPUT_FILES", (missing, missing, missing)):
                with self.assertRaisesRegex(FileNotFoundError, "Missing explicit"):
                    t07._load_inputs()

        X = pd.DataFrame({"feature_a": [0.0, 1.0], "feature_b": [1.0, 0.0]})
        y = pd.DataFrame({target: [0, 1] for target in t07.TARGETS})
        y["LET_IS"] = [0, 1]
        ids = pd.DataFrame({"ID": ["a", "b"]})
        with self.assertRaisesRegex(ValueError, "missing required targets"):
            t07._validate_test_inputs(X, y.drop(columns="ZSN"), ids)
        with self.assertRaisesRegex(ValueError, "unique columns"):
            t07._validate_test_inputs(pd.concat([X, X.iloc[:, [0]]], axis=1), y, ids)

    def test_quantile_bins_keep_tied_probabilities_together_and_handle_constant_scores(self):
        probability = np.array([0.2, 0.2, 0.2, 0.9, 0.9])
        groups = t07._quantile_bin_indices(probability)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertEqual(groups[3], groups[4])
        self.assertNotEqual(groups[0], groups[3])

        constant = np.full(5, 0.4)
        constant_groups = t07._quantile_bin_indices(constant)
        np.testing.assert_array_equal(constant_groups, np.zeros(5, dtype=int))
        self.assertEqual(len(np.unique(constant_groups)), 1)

    def test_calibration_diagnostics_reject_nonbinary_or_nonfinite_labels(self):
        probability = np.array([0.1, 0.9])
        for labels in (np.array([0, 2]), np.array([0, np.nan])):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(ValueError, "binary"):
                    t07.calibration_diagnostics(labels, probability)

    def test_decision_curve_net_benefit_and_reference_strategies(self):
        labels = np.array([1, 0, 1, 0])
        probability = np.array([0.8, 0.7, 0.2, 0.1])
        curve = t07.decision_curve_analysis(
            labels, probability, thresholds=np.array([0.25]),
        )
        values = curve.set_index("strategy")["net_benefit"]
        self.assertAlmostEqual(values["model"], 0.25 - 0.25 * (1 / 3))
        self.assertAlmostEqual(values["treat_all"], 0.5 - 0.5 * (1 / 3))
        self.assertEqual(values["treat_none"], 0.0)

        full = t07.decision_curve_analysis(labels, probability)
        self.assertEqual(full["threshold_probability"].min(), 0.05)
        self.assertEqual(full["threshold_probability"].max(), 0.40)
        self.assertEqual(set(full["strategy"]), {"model", "treat_all", "treat_none"})

    def test_bootstrap_ranking_intervals_are_deterministic_and_count_invalid_draws(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        probability = np.array([0.1, 0.9, 0.3, 0.7, 0.4, 0.8])
        first = t07.bootstrap_ranking_confidence_intervals(
            "ZSN", labels, probability, replicates=1000, seed=31,
        )
        second = t07.bootstrap_ranking_confidence_intervals(
            "ZSN", labels, probability, replicates=1000, seed=31,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["metric"]), {"auroc", "auprc_average_precision"})
        self.assertTrue((first["bootstrap_replicates_requested"] == 1000).all())
        self.assertTrue((first["bootstrap_seed"] == 31).all())
        self.assertTrue((first["bootstrap_replicates_invalid_single_class"] > 0).all())
        self.assertTrue((
            first["bootstrap_replicates_valid"]
            + first["bootstrap_replicates_invalid_single_class"] == 1000
        ).all())
        self.assertTrue((first["ci_lower"] <= first["estimate"]).all())
        self.assertTrue((first["estimate"] <= first["ci_upper"]).all())

    def test_existing_attempt_marker_prevents_another_run_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "attempt_started"
            t07._create_attempt_marker(marker)
            original = marker.read_text(encoding="utf-8")
            with self.assertRaises(FileExistsError):
                t07._create_attempt_marker(marker)
            self.assertEqual(marker.read_text(encoding="utf-8"), original)

    def test_synthetic_end_to_end_run_writes_expected_outputs(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        target_thresholds = {"DEATH_BINARY": 0.5, "ZSN": 0.4, "FIBR_PREDS": 0.6}
        model_names = {target: f"Mock {target}" for target in t07.TARGETS}
        fake_bundles = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1 = root / "stage1"
            stage10 = root / "stage10"
            output = root / "stage11"
            stage1.mkdir()
            stage10.mkdir()
            X = pd.DataFrame({"feature_a": np.arange(6), "feature_b": np.arange(6) * 2})
            y = pd.DataFrame({target: labels for target in t07.TARGETS})
            y["LET_IS"] = labels
            ids = pd.DataFrame({"ID": [f"synthetic-{index}" for index in range(6)]})
            X.to_csv(stage1 / "X_test.csv", index=False)
            y.to_csv(stage1 / "y_test.csv", index=False)
            ids.to_csv(stage1 / "test_record_ids.csv", index=False)
            pd.DataFrame({"partition": ["train", "test"], "patients": [24, 6]}).to_csv(
                stage1 / "split_summary.csv", index=False,
            )
            (root / "mi_stage1_eda.py").write_text("# synthetic split source\n", encoding="utf-8")

            model_manifest = {
                target: {"model": model_names[target]}
                for target in t07.TARGETS
            }
            t06_manifest = {
                "stage": "T06",
                "freeze_revision": "non_tabpfn_v1",
                "targets": list(t07.TARGETS),
                "models": model_manifest,
                "calibration_parameters": {
                    target: {"method": "Platt"} for target in t07.TARGETS
                },
            }
            (stage10 / "final_model_manifest.json").write_text(
                json.dumps(t06_manifest), encoding="utf-8",
            )
            pd.DataFrame({
                "target": list(t07.TARGETS),
                "threshold": [target_thresholds[target] for target in t07.TARGETS],
            }).to_csv(stage10 / "selected_thresholds.csv", index=False)

            for index, (target, filename) in enumerate(t07.BUNDLE_FILES.items()):
                class SyntheticBundle:
                    feature_columns = ["feature_a", "feature_b"]

                    def __init__(self):
                        self.target = target
                        self.model_name = model_names[target]
                        self.metadata = {
                            "target": target,
                            "model": model_names[target],
                            "calibration_method": "Platt",
                            "calibration_parameters": {"method": "Platt"},
                            "threshold": target_thresholds[target],
                        }

                    def predict_probability(self, _features):
                        return np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

                fake_bundles[filename] = SyntheticBundle()
                (stage10 / filename).write_bytes(f"synthetic bundle {index}".encode())
            development_oof = pd.DataFrame([
                {
                    "target": target,
                    "ID": f"{target}-{index}",
                    "true_label": labels[index],
                    "raw_probability": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9][index],
                    "platt_probability": [0.15, 0.75, 0.25, 0.65, 0.35, 0.85][index],
                }
                for target in t07.TARGETS
                for index in range(6)
            ])
            development_oof.to_csv(stage10 / "cross_fitted_oof_predictions.csv", index=False)

            input_paths = (
                stage1 / "X_test.csv",
                stage1 / "y_test.csv",
                stage1 / "test_record_ids.csv",
            )
            attempt_marker = root / ".attempt_marker"
            with (
                patch.object(t07, "ROOT", root),
                patch.object(t07, "STAGE1", stage1),
                patch.object(t07, "STAGE10", stage10),
                patch.object(t07, "OUT", output),
                patch.object(t07, "ATTEMPT_MARKER_FILE", attempt_marker),
                patch.object(t07, "INPUT_FILES", input_paths),
                patch.object(t07, "SPLIT_SUMMARY_FILE", stage1 / "split_summary.csv"),
                patch.object(t07, "SPLIT_SOURCE_FILE", root / "mi_stage1_eda.py"),
                patch.object(t07, "THRESHOLD_FILE", stage10 / "selected_thresholds.csv"),
                patch.object(t07, "T06_MANIFEST_FILE", stage10 / "final_model_manifest.json"),
                patch.object(t07, "DEVELOPMENT_OOF_FILE", stage10 / "cross_fitted_oof_predictions.csv"),
                patch.object(t07.joblib, "load", side_effect=lambda path: fake_bundles[Path(path).name]),
            ):
                t07.run_evaluation(approval_granted=True)
                self.assertEqual({path.name for path in output.iterdir()}, set(t07.OUTPUT_FILES))
                self.assertTrue(attempt_marker.is_file())
                prediction_output = pd.read_csv(output / "test_predictions.csv")
                self.assertEqual(len(prediction_output), 18)
                self.assertEqual(set(prediction_output["target"]), set(t07.TARGETS))
                report = (output / "t07_evaluation_report.md").read_text(encoding="utf-8")
                self.assertIn("## Ranking quality", report)
                self.assertIn("## Probability and calibration quality", report)
                self.assertIn("## Before/after calibration comparison (development only)", report)
                comparison = pd.read_csv(output / "development_oof_calibration_comparison.csv")
                self.assertEqual(len(comparison), len(t07.TARGETS))
                intervals = pd.read_csv(output / "test_metric_confidence_intervals.csv")
                self.assertEqual(len(intervals), 2 * len(t07.TARGETS))
                self.assertTrue((intervals.bootstrap_replicates_requested == 1000).all())
                curves = pd.read_csv(output / "test_decision_curve_analysis.csv")
                self.assertEqual(len(curves), len(t07.TARGETS) * len(t07.DCA_THRESHOLDS) * 3)
                self.assertEqual(set(curves.strategy), {"model", "treat_all", "treat_none"})
                self.assertAlmostEqual(curves.threshold_probability.min(), 0.05)
                self.assertAlmostEqual(curves.threshold_probability.max(), 0.40)
                self.assertIn("## Ranking metric uncertainty", report)
                self.assertIn("## Decision curve analysis (descriptive)", report)
                manifest = json.loads((output / "t07_run_manifest.json").read_text(encoding="utf-8"))
                self.assertIn("split_alignment_evidence", manifest)
                self.assertIn("marker_sha256", manifest["one_time_execution_guard"])
                self.assertEqual(manifest["ranking_metric_confidence_intervals"]["replicates"], 1000)
                self.assertEqual(manifest["decision_curve_analysis"]["threshold_start"], 0.05)

                for artifact in output.iterdir():
                    artifact.unlink()
                output.rmdir()
                with self.assertRaises(FileExistsError):
                    t07.run_evaluation(approval_granted=True)

    def test_rejects_wrong_probability_shapes_and_invalid_thresholds(self):
        for probability in (
            np.array([[0.1], [0.9]]),
            np.array([0.1, np.nan]),
            np.array([0.1, 1.1]),
        ):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    t07.evaluate_target(
                        "DEATH_BINARY", np.array([0, 1]), probability, 0.5,
                        np.array(["a", "b"]),
                    )
        with self.assertRaises(ValueError):
            t07.evaluate_target(
                "DEATH_BINARY", np.array([0, 1]), np.array([0.1, 0.9]), float("nan"),
                np.array(["a", "b"]),
            )
        with self.assertRaisesRegex(ValueError, "binary"):
            t07.evaluate_target(
                "DEATH_BINARY", np.array([0, 0.5]), np.array([0.1, 0.9]), 0.5,
                np.array(["a", "b"]),
            )

    def test_bundle_validation_requires_exact_feature_order_and_frozen_threshold(self):
        bundle = MockBundle(np.array([0.2, 0.8]))
        manifest = {
            "models": {"DEATH_BINARY": {"model": "Mock model"}},
            "calibration_parameters": {"DEATH_BINARY": {"method": "Platt"}},
        }
        thresholds = pd.DataFrame({"target": ["DEATH_BINARY"], "threshold": [0.5]})
        X = pd.DataFrame(columns=["feature_a", "feature_b"])
        self.assertEqual(t07._validate_bundle(bundle, "DEATH_BINARY", manifest, thresholds, X), 0.5)

        reordered = X[["feature_b", "feature_a"]]
        with self.assertRaisesRegex(ValueError, "names/order"):
            t07._validate_bundle(bundle, "DEATH_BINARY", manifest, thresholds, reordered)

        bundle.metadata["threshold"] = 0.4
        with self.assertRaisesRegex(ValueError, "does not match"):
            t07._validate_bundle(bundle, "DEATH_BINARY", manifest, thresholds, X)

        bundle.metadata["threshold"] = 0.5
        bundle.metadata["calibration_parameters"] = {
            "method": "Platt", "coefficient": 99,
        }
        with self.assertRaisesRegex(ValueError, "calibrator parameters"):
            t07._validate_bundle(bundle, "DEATH_BINARY", manifest, thresholds, X)

    def test_test_labels_must_be_binary_and_ids_unique(self):
        X = pd.DataFrame({"feature_a": [0.0, 1.0], "feature_b": [1.0, 0.0]})
        y = pd.DataFrame({target: [0, 1] for target in t07.TARGETS})
        y["LET_IS"] = [0, 1]
        duplicate_ids = pd.DataFrame({"ID": ["same", "same"]})
        with self.assertRaisesRegex(ValueError, "unique"):
            t07._validate_test_inputs(X, y, duplicate_ids)
        invalid_y = y.copy()
        invalid_y.loc[0, "ZSN"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            t07._validate_test_inputs(X, invalid_y, pd.DataFrame({"ID": ["a", "b"]}))
        mismatched_death = y.copy()
        mismatched_death.loc[1, "DEATH_BINARY"] = 0
        with self.assertRaisesRegex(ValueError, "LET_IS > 0"):
            t07._validate_test_inputs(X, mismatched_death, pd.DataFrame({"ID": ["a", "b"]}))


if __name__ == "__main__":
    unittest.main()
