import base64
import io
import json
import subprocess
import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from mi_stage10_t06_freeze import (
    CALIBRATION_METHODS,
    FinalModelBundle,
    OUTPUT_FILES,
    cleanup_model_resources,
    build_estimator,
    compare_methods,
    crossfit_target,
    full_development_oof_and_model,
    load_selection_evidence,
    model_configurations,
    optional_method_qualification,
    quality_metrics,
    selection_rationale,
    select_threshold,
    summarize_method_comparison,
    _validate_outcome_values,
    _validated_binary_labels,
)
from mi_stage10_calibrators import PlattCalibrator


class IndexProbabilityEstimator:
    """Tiny injected estimator used to observe fold boundaries without model work."""

    def __init__(self, fit_records):
        self.fit_records = fit_records
        self.classes_ = np.array([0, 1])
        self.fit_ids = None

    def fit(self, X, y):
        self.fit_ids = set(X.index.tolist())
        self.fit_records.append(self)
        return self

    def predict_proba(self, X):
        evaluation_ids = set(X.index.tolist())
        if self.fit_ids.intersection(evaluation_ids):
            raise AssertionError("A prediction row was included in estimator fitting.")
        score = np.asarray(X["score"], dtype=float)
        positive = 1 / (1 + np.exp(-score))
        return np.column_stack([1 - positive, positive])


def fixed_comparison(qualified=True):
    rows = []
    methods = CALIBRATION_METHODS
    pooled_values = {
        "Raw": (0.20, 0.40),
        "Platt": (0.10, 0.20),
        "Isotonic": (0.08 if qualified else 0.0995, 0.16 if qualified else 0.1995),
        "Beta": (0.11, 0.21),
        "Temperature": (0.12, 0.22),
    }
    for method in methods:
        brier, loss = pooled_values[method]
        rows.append({
            "target": "T", "method": method, "scope": "pooled", "outer_fold": "pooled",
            "brier_score": brier, "log_loss": loss,
        })
        for fold in range(1, 6):
            if method == "Isotonic" and qualified:
                fold_metrics = (0.09, 0.18)
            elif method == "Isotonic":
                fold_metrics = (0.09, 0.18) if fold < 5 else (0.11, 0.21)
            else:
                fold_metrics = (brier, loss)
            rows.append({
                "target": "T", "method": method, "scope": "outer_fold",
                "outer_fold": fold, "brier_score": fold_metrics[0],
                "log_loss": fold_metrics[1],
            })
    return pd.DataFrame(rows)


class T06FreezeTests(unittest.TestCase):
    def test_development_outcomes_reject_fractional_and_nonbinary_values(self):
        valid = pd.DataFrame({
            "LET_IS": [0, 1, 2, 0],
            "DEATH_BINARY": [0, 1, 1, 0],
            "ZSN": [0, 1, 0, 1],
            "FIBR_PREDS": [1, 0, 1, 0],
        })
        _validate_outcome_values(valid)

        # Missing outcomes remain eligible for the established per-target dropna.
        with_missing = valid.copy()
        with_missing["ZSN"] = with_missing["ZSN"].astype(float)
        with_missing.loc[3, "ZSN"] = np.nan
        _validate_outcome_values(with_missing)

        for target, invalid_value in (
            ("ZSN", 0.5),
            ("FIBR_PREDS", 2),
            ("DEATH_BINARY", -1),
        ):
            with self.subTest(target=target, invalid_value=invalid_value):
                invalid = valid.copy()
                invalid[target] = invalid[target].astype(float)
                invalid.loc[0, target] = invalid_value
                with self.assertRaisesRegex(ValueError, "finite exact binary"):
                    _validate_outcome_values(invalid)

        with self.assertRaisesRegex(ValueError, "finite exact binary"):
            _validated_binary_labels(np.array([0, 0.5, 1]), "ZSN")

    def test_development_mortality_label_must_match_let_is_definition(self):
        y = pd.DataFrame({
            "LET_IS": [0, 1, 2, 0],
            "DEATH_BINARY": [0, 1, 1, 0],
            "ZSN": [0, 1, 0, 1],
            "FIBR_PREDS": [1, 0, 1, 0],
        })
        with self.assertRaisesRegex(ValueError, "LET_IS"):
            _validate_outcome_values(y.drop(columns="LET_IS"))

        mismatch = y.copy()
        mismatch.loc[1, "DEATH_BINARY"] = 0
        with self.assertRaisesRegex(ValueError, "LET_IS > 0"):
            _validate_outcome_values(mismatch)

        malformed_let_is = y.astype(float)
        malformed_let_is.loc[0, "LET_IS"] = np.inf
        with self.assertRaisesRegex(ValueError, "LET_IS labels must be finite"):
            _validate_outcome_values(malformed_let_is)

        missingness_mismatch = y.astype(float)
        missingness_mismatch.loc[0, "LET_IS"] = np.nan
        with self.assertRaisesRegex(ValueError, "missingness must match"):
            _validate_outcome_values(missingness_mismatch)

    def test_crossfit_rejects_fractional_labels_before_integer_conversion_or_fit(self):
        builder = Mock(side_effect=AssertionError("Invalid labels must fail before fitting."))
        X = pd.DataFrame({"score": [0, 1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "finite exact binary"):
            crossfit_target(
                X, np.array([0, 1, 0.5, 1]), np.array(["a", "b", "c", "d"]),
                "ZSN", "unused", builder,
            )
        builder.assert_not_called()

    def test_quality_metrics_include_probability_quality_and_confusion_counts(self):
        result = quality_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.6, 0.9]))
        self.assertAlmostEqual(result["brier_score"], 0.1675)
        self.assertGreater(result["average_precision"], 0)
        self.assertGreater(result["auroc"], 0)
        self.assertIn("log_loss", result)
        self.assertEqual(
            [result["true_negatives_at_0_5"], result["false_positives_at_0_5"],
             result["false_negatives_at_0_5"], result["true_positives_at_0_5"]],
            [1, 1, 0, 2],
        )

    def test_calibration_comparison_reports_each_fold_and_pooled_metrics(self):
        records = []
        for fold in range(1, 6):
            for index in range(4):
                label = index % 2
                records.append({
                    "target": "DEATH_BINARY", "outer_fold": fold,
                    "true_label": label, "raw_probability": 0.2 + 0.6 * label,
                    "platt_probability": 0.25 + 0.5 * label,
                    "isotonic_probability": 0.3 + 0.4 * label,
                    "beta_probability": 0.2 + 0.6 * label,
                    "temperature_probability": 0.2 + 0.6 * label,
                })
        result = compare_methods(pd.DataFrame(records))
        subset = result[result.target == "DEATH_BINARY"]
        self.assertEqual(len(subset), len(CALIBRATION_METHODS) * 6)
        self.assertEqual(set(subset.scope), {"outer_fold", "pooled"})
        self.assertEqual(set(subset.outer_fold[subset.scope == "outer_fold"]), {1, 2, 3, 4, 5})
        self.assertTrue({"brier_score", "log_loss", "average_precision", "auroc",
                         "true_positives_at_0_5"}.issubset(subset.columns))

    def test_summary_has_required_pooled_metrics_and_five_fold_comparison(self):
        records = []
        for target in ("DEATH_BINARY", "ZSN", "FIBR_PREDS"):
            for fold in range(1, 6):
                for label in (0, 1):
                    records.append({
                        "target": target, "outer_fold": fold, "true_label": label,
                        "raw_probability": 0.2 + 0.6 * label,
                        "platt_probability": 0.25 + 0.5 * label,
                        "isotonic_probability": 0.3 + 0.4 * label,
                        "beta_probability": 0.2 + 0.6 * label,
                        "temperature_probability": 0.2 + 0.6 * label,
                    })
        detailed = compare_methods(pd.DataFrame(records))
        qualifications = {
            target: {
                method: {"qualifies": False}
                for method in ("Isotonic", "Beta", "Temperature")
            }
            for target in ("DEATH_BINARY", "ZSN", "FIBR_PREDS")
        }
        summary = summarize_method_comparison(detailed, qualifications)
        self.assertEqual(len(summary), 3 * len(CALIBRATION_METHODS))
        self.assertTrue({
            "pooled_brier_score", "pooled_log_loss", "pooled_average_precision",
            "pooled_auroc", "fold_level_comparison", "recommendation",
        }.issubset(summary.columns))
        operating_point_metrics = (
            "accuracy_at_0_5", "precision_at_0_5", "recall_at_0_5",
            "specificity_at_0_5", "f1_at_0_5",
            "true_negatives_at_0_5", "false_positives_at_0_5",
            "false_negatives_at_0_5", "true_positives_at_0_5",
        )
        self.assertTrue({
            f"pooled_{metric}" for metric in operating_point_metrics
        }.issubset(summary.columns))
        for comparison_row in summary.itertuples(index=False):
            target_detailed = detailed.loc[detailed.target == comparison_row.target]
            pooled_detailed = target_detailed.loc[
                (target_detailed.method == comparison_row.method)
                & (target_detailed.scope == "pooled")
            ].iloc[0]
            fold_values = json.loads(comparison_row.fold_level_comparison)
            self.assertEqual(len(fold_values), 5)
            for metric in operating_point_metrics:
                self.assertAlmostEqual(
                    getattr(comparison_row, f"pooled_{metric}"),
                    pooled_detailed[metric],
                )
            for fold_value in fold_values:
                self.assertTrue(set(operating_point_metrics).issubset(fold_value))
                fold_detailed = target_detailed.loc[
                    (target_detailed.method == comparison_row.method)
                    & (target_detailed.outer_fold == fold_value["fold"])
                ].iloc[0]
                for metric in operating_point_metrics:
                    self.assertAlmostEqual(fold_value[metric], fold_detailed[metric])
        self.assertTrue(summary.loc[summary.method == "Platt", "recommendation"]
                        .str.contains("Frozen final method").all())

    def test_optional_calibrator_requires_both_relative_gains_and_four_fold_wins(self):
        qualified = optional_method_qualification(fixed_comparison(True), "T")
        self.assertTrue(qualified["Isotonic"]["qualifies"])
        self.assertFalse(optional_method_qualification(fixed_comparison(False), "T")["Isotonic"]["qualifies"])

    def test_threshold_maximizes_specificity_and_breaks_ties_upward(self):
        result = select_threshold(
            np.array([1, 1, 1, 1, 0, 0]),
            np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2]),
            minimum_sensitivity=0.5,
        )
        self.assertEqual(result["status"], "constraint_met")
        self.assertEqual(result["threshold"], 0.8)
        self.assertEqual(result["sensitivity"], 0.5)
        self.assertEqual(result["specificity"], 1.0)
        self.assertEqual(
            [result["true_negatives"], result["false_positives"],
             result["false_negatives"], result["true_positives"]],
            [2, 0, 2, 2],
        )

    def test_zero_sensitivity_constraint_includes_all_negative_cutoff(self):
        probabilities = np.array([0.2, 0.9])
        result = select_threshold(
            np.array([1, 0]), probabilities, minimum_sensitivity=0,
        )
        self.assertEqual(result["status"], "constraint_met")
        self.assertEqual(result["threshold"], np.nextafter(probabilities.max(), np.inf))
        self.assertEqual(result["sensitivity"], 0.0)
        self.assertEqual(result["specificity"], 1.0)
        self.assertEqual(
            [result["true_negatives"], result["false_positives"],
             result["false_negatives"], result["true_positives"]],
            [1, 0, 1, 0],
        )

    def test_threshold_explicitly_reports_unmet_constraint(self):
        result = select_threshold(np.array([0, 0]), np.array([0.1, 0.2]))
        self.assertEqual(result["status"], "constraint_unmet")
        self.assertIn("sensitivity >=", result["constraint"])
        self.assertIn("precision", result)
        self.assertIn("f1", result)

    def test_metrics_reject_nonfinite_and_out_of_range_probabilities(self):
        for probability in (np.array([0.1, np.nan]), np.array([0.1, 1.1])):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    quality_metrics(np.array([0, 1]), probability)

    def test_cleanup_collects_garbage_without_loading_optional_accelerator_runtime(self):
        with patch("mi_stage10_t06_freeze.gc.collect") as collect:
            cleanup_model_resources()
        collect.assert_called_once_with()

    def test_nested_crossfit_isolates_fit_rows_and_aligns_one_oof_row_per_id(self):
        n = 50
        labels = np.tile([0, 1], n // 2)
        ids = np.array([f"dev-{i:03d}" for i in range(n)])
        X = pd.DataFrame({"score": np.where(labels == 1, 1.5, -1.5)}, index=ids)
        fit_records = []

        def builder(model_name, feature_columns):
            self.assertEqual(model_name, "Injected deterministic estimator")
            self.assertEqual(feature_columns, ["score"])
            return IndexProbabilityEstimator(fit_records)

        with patch("mi_stage10_t06_freeze.cleanup_model_resources") as cleanup:
            result = crossfit_target(
                X, labels, ids, "T", "Injected deterministic estimator", builder,
            )
        self.assertEqual(len(result), n)
        self.assertEqual(result.ID.nunique(), n)
        self.assertEqual(set(result.ID), set(ids))
        self.assertEqual(result.ID.tolist(), ids.tolist())
        aligned = result.set_index("ID").loc[ids]
        np.testing.assert_array_equal(aligned.true_label.to_numpy(), labels)
        self.assertEqual(set(aligned.outer_fold), {1, 2, 3, 4, 5})
        self.assertEqual(len(fit_records), 30)
        self.assertTrue(all(estimator.fit_ids for estimator in fit_records))
        self.assertEqual(cleanup.call_count, 30)

    def test_final_development_bundles_are_staged_and_released_sequentially(self):
        labels = np.tile([0, 1], 25)
        X = pd.DataFrame({
            "score": np.where(labels == 1, 1.5, -1.5),
        }, index=[f"dev-{i:03d}" for i in range(len(labels))])
        estimator_refs = []
        dumped_refs = []

        class LifecycleEstimator:
            classes_ = np.array([0, 1])

            def __init__(self):
                alive = [index for index, ref in enumerate(estimator_refs) if ref() is not None]
                if alive:
                    raise AssertionError(f"Previous full-development estimators remain live: {alive}")
                estimator_refs.append(weakref.ref(self))

            def fit(self, X_fit, _y_fit):
                self.fit_ids = set(X_fit.index)
                return self

            def predict_proba(self, X_score):
                positive = 1 / (1 + np.exp(-np.asarray(X_score["score"], dtype=float)))
                return np.column_stack([1 - positive, positive])

        def builder(model_name, feature_columns):
            self.assertEqual(model_name, "Injected lifecycle estimator")
            self.assertEqual(feature_columns, ["score"])
            return LifecycleEstimator()

        def record_dump(bundle, _destination, compress):
            self.assertEqual(compress, 3)
            dumped_refs.append(weakref.ref(bundle.estimator))

        with patch("mi_stage10_t06_freeze.joblib.dump", new=record_dump):
            for target in ("T1", "T2"):
                scores, folds, parameters = full_development_oof_and_model(
                    X, labels, "Injected lifecycle estimator", builder,
                    bundle_path=Path(f"{target}_staged_bundle.joblib"),
                    threshold=0.5,
                    threshold_status="constraint_met",
                    target=target,
                )
                self.assertEqual(scores.shape, labels.shape)
                self.assertEqual(folds.shape, labels.shape)
                self.assertEqual(parameters["method"], "Platt")

        self.assertEqual(len(estimator_refs), 12)
        self.assertEqual(len(dumped_refs), 2)
        self.assertTrue(all(ref() is None for ref in estimator_refs))
        self.assertTrue(all(ref() is None for ref in dumped_refs))

    def test_bundle_survives_joblib_round_trip_and_predicts_probabilities(self):
        X = pd.DataFrame({"score": [-2.0, -1.0, 1.0, 2.0]})
        y = np.array([0, 0, 1, 1])
        estimator = DummyClassifier(strategy="prior").fit(X, y)
        base = estimator.predict_proba(X)[:, 1]
        calibrator = PlattCalibrator().fit(base, y)
        bundle = FinalModelBundle(
            "T", "Random Forest preliminary", estimator, calibrator,
            ["score"], None, {"calibration_method": "Platt"},
        )
        storage = io.BytesIO()
        import joblib

        joblib.dump(bundle, storage, compress=3)
        storage.seek(0)
        loaded = joblib.load(storage)
        probability = loaded.predict_probability(X)
        self.assertEqual(probability.shape, (4,))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertEqual(loaded.predict_proba(X).shape, (4, 2))
        self.assertEqual(FinalModelBundle.__module__, "mi_stage10_t06_bundle")

    def test_script_run_bundle_loads_in_fresh_process_without_main_alias(self):
        X = pd.DataFrame({"score": [-2.0, -1.0, 1.0, 2.0]})
        y = np.array([0, 0, 1, 1])
        estimator = DummyClassifier(strategy="prior").fit(X, y)
        calibrator = PlattCalibrator().fit(estimator.predict_proba(X)[:, 1], y)
        script_namespace = {"__name__": "__main__"}
        exec("from mi_stage10_t06_bundle import FinalModelBundle", script_namespace)
        script_bundle_class = script_namespace["FinalModelBundle"]
        self.assertIs(script_bundle_class, FinalModelBundle)
        bundle = script_bundle_class(
            "T", "Random Forest preliminary", estimator, calibrator,
            ["score"], None, {"calibration_method": "Platt"},
        )
        storage = io.BytesIO()
        import joblib

        joblib.dump(bundle, storage, compress=3)
        payload = base64.b64encode(storage.getvalue()).decode("ascii")
        child_code = (
            "import base64, io, json, sys, joblib, pandas as pd\n"
            "bundle = joblib.load(io.BytesIO(base64.b64decode(sys.stdin.buffer.read())))\n"
            "X = pd.DataFrame({'score': [-2.0, -1.0, 1.0, 2.0]})\n"
            "print(json.dumps({'module': type(bundle).__module__, "
            "'probability': bundle.predict_probability(X).tolist()}))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", child_code],
            input=payload,
            text=True,
            capture_output=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        loaded = json.loads(completed.stdout)
        self.assertEqual(loaded["module"], "mi_stage10_t06_bundle")
        self.assertEqual(len(loaded["probability"]), 4)
        self.assertTrue(np.isfinite(loaded["probability"]).all())
        self.assertTrue(np.all(
            (np.asarray(loaded["probability"]) >= 0)
            & (np.asarray(loaded["probability"]) <= 1)
        ))

    def test_final_output_names_match_t06_deliverables(self):
        required = {
            "final_model_selections.csv", "calibration_method_comparison.csv",
            "cross_fitted_oof_predictions.csv", "selected_thresholds.csv",
            "final_model_manifest.json", "t06_freeze_report.md",
        }
        self.assertTrue(required.issubset(OUTPUT_FILES))
        self.assertEqual(len(OUTPUT_FILES), len(set(OUTPUT_FILES)))

    def test_finalist_rationale_is_truthful_about_user_selected_near_tie(self):
        rationale = selection_rationale("FIBR_PREDS", {
            "selected_mean": 0.3011,
            "selected_sd": 0.1079,
            "selected_auroc_mean": 0.7251,
            "selected_brier_score_mean": 0.082854,
            "selected_log_loss_mean": 0.292421,
            "alternatives": [{
                "model": "Extra Trees preliminary",
                "mean": 0.3014,
                "sample_sd": 0.1252,
                "auroc_mean": 0.7217,
                "brier_score_mean": 0.082892,
                "log_loss_mean": 0.295108,
            }],
        })
        self.assertIn("essentially tied", rationale)
        self.assertIn("AP gap was 0.000300", rationale)
        self.assertIn("more balanced candidate", rationale)
        self.assertIn("AUROC 0.725100 versus 0.721700", rationale)
        self.assertIn("Brier 0.082854 versus 0.082892", rationale)
        self.assertIn("Log Loss 0.292421 versus 0.295108", rationale)
        self.assertIn("descriptive, not significance tests", rationale)

    def test_saved_development_ap_summaries_support_all_fixed_finalists(self):
        evidence, paths = load_selection_evidence()
        self.assertEqual(len(paths), 1)
        self.assertEqual(set(evidence), {"DEATH_BINARY", "ZSN", "FIBR_PREDS"})
        self.assertAlmostEqual(evidence["DEATH_BINARY"]["selected_mean"], 0.732590, places=6)
        self.assertAlmostEqual(evidence["ZSN"]["selected_mean"], 0.501775, places=6)
        self.assertAlmostEqual(evidence["FIBR_PREDS"]["selected_mean"], 0.3010538, places=6)
        self.assertAlmostEqual(evidence["FIBR_PREDS"]["selected_auroc_mean"], 0.725125, places=6)
        self.assertAlmostEqual(evidence["FIBR_PREDS"]["selected_brier_score_mean"], 0.082854, places=6)
        self.assertAlmostEqual(evidence["FIBR_PREDS"]["selected_log_loss_mean"], 0.292421, places=6)
        self.assertEqual(
            evidence["FIBR_PREDS"]["alternatives"][0]["model"],
            "Extra Trees preliminary",
        )
        self.assertAlmostEqual(evidence["FIBR_PREDS"]["alternatives"][0]["auroc_mean"], 0.721708, places=6)
        self.assertAlmostEqual(evidence["FIBR_PREDS"]["alternatives"][0]["brier_score_mean"], 0.082892, places=6)
        self.assertAlmostEqual(evidence["FIBR_PREDS"]["alternatives"][0]["log_loss_mean"], 0.295108, places=6)

    def test_tuned_finalist_configurations_are_saved_fold_medoids(self):
        configurations = model_configurations()
        death = configurations["DEATH_BINARY"]
        zsn = configurations["ZSN"]
        self.assertEqual(death["model"], "CatBoost tuned")
        self.assertEqual(zsn["model"], "Random Forest tuned")
        self.assertEqual(death["tuned_configuration_provenance"]["selected_outer_fold"], 4)
        self.assertEqual(zsn["tuned_configuration_provenance"]["selected_outer_fold"], 4)
        self.assertEqual(death["tuned_configuration_provenance"]["source"],
                         "stage6_catboost_outputs/tuning_results.csv")
        self.assertEqual(zsn["tuned_configuration_provenance"]["source"],
                         "stage3_outputs/tuning_results.csv")
        self.assertEqual(
            build_estimator("CatBoost tuned", ["SEX", "ant_im"], "DEATH_BINARY")
            .named_steps["classifier"].get_params()["iterations"],
            death["parameters"]["iterations"],
        )
        self.assertEqual(
            build_estimator("Random Forest tuned", ["AGE", "SEX"], "ZSN")
            .named_steps["classifier"].get_params()["min_samples_leaf"],
            zsn["parameters"]["min_samples_leaf"],
        )


if __name__ == "__main__":
    unittest.main()
