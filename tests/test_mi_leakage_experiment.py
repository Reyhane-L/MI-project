from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import mi_leakage_experiment as experiment


class LeakageExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage2 = experiment.load_stage2_api()

    def make_inputs(self, rows: int = 31):
        rng = np.random.default_rng(42)
        index = pd.Index(range(100, 100 + rows))
        features = {}
        for column in self.stage2.continuous_features:
            features[column] = rng.normal(size=rows)
        features["AGE"] = np.arange(rows, dtype=float)
        for column in self.stage2.ordinal_features:
            features[column] = np.arange(rows, dtype=float) % 4
        for column in self.stage2.nominal_features:
            features[column] = np.arange(rows, dtype=float) % 4
        features["SEX"] = np.arange(rows) % 2
        X = pd.DataFrame(features, index=index)
        X.loc[index[1], "AGE"] = np.nan
        X.loc[index[2], "INF_ANAM"] = np.nan
        X.loc[index[3], "ant_im"] = np.nan
        X.loc[index[4], "SEX"] = np.nan
        X.loc[index[-1], "AGE"] = 100000.0
        labels = np.tile([0, 1], (rows + 1) // 2)[:rows].astype(float)
        y_all = pd.DataFrame({
            target: labels.copy() for target in self.stage2.targets
        }, index=index)
        if rows > 30:
            y_all.iloc[-1, :] = np.nan
        return X, y_all

    def test_global_imputers_use_complete_development_predictor_table(self):
        X, y_all = self.make_inputs()
        binary = experiment.validate_inputs(X, y_all, self.stage2)
        imputers = experiment.fit_global_imputers(X, self.stage2, binary)
        expected = float(X["AGE"].median())
        observed_label_rows_only = float(X.iloc[:-1]["AGE"].median())
        self.assertNotEqual(expected, observed_label_rows_only)
        self.assertEqual(imputers["continuous"].statistics_[0], expected)

    def test_fold_local_pipeline_imputer_excludes_validation_predictors(self):
        X, y_all = self.make_inputs()
        observed_y = experiment.validate_target(
            self.stage2.targets[0], y_all[self.stage2.targets[0]], self.stage2.n_splits,
        )
        train_positions = np.arange(0, 24)
        valid_positions = np.arange(24, 30)
        X_train = X.loc[observed_y.index].iloc[train_positions].copy()
        X_valid = X.loc[observed_y.index].iloc[valid_positions].copy()
        X_valid.iloc[0, X.columns.get_loc("AGE")] = 1_000_000.0
        y_train = observed_y.iloc[train_positions].to_numpy()
        binary = experiment.validate_inputs(X, y_all, self.stage2)

        _, _, pipeline = experiment._fit_fold(
            self.stage2, X.columns.tolist(), binary, X_train, y_train, X_valid,
        )
        fitted_imputer = (
            pipeline.named_steps["preprocess"]
            .named_transformers_["continuous"]
            .named_steps["imputer"]
        )
        self.assertEqual(
            fitted_imputer.statistics_[0],
            float(X_train["AGE"].median()),
        )
        self.assertNotEqual(fitted_imputer.statistics_[0], 1_000_000.0)

    def test_experiment_uses_same_folds_and_returns_metric_schemas(self):
        X, y_all = self.make_inputs()
        folds, aggregate, target_summary, manifest = experiment.run_experiment(
            X, y_all, stage2=self.stage2, command="synthetic test",
        )
        self.assertEqual(list(folds.columns), experiment.FOLD_COLUMNS)
        self.assertEqual(list(aggregate.columns), experiment.AGGREGATE_COLUMNS)
        self.assertEqual(list(target_summary.columns), experiment.TARGET_SUMMARY_COLUMNS)
        self.assertEqual(len(folds), 3 * 5 * 2)
        self.assertEqual(len(aggregate), 3 * 2)
        self.assertEqual(len(target_summary), 3)
        self.assertTrue(np.isfinite(folds[experiment.METRICS].to_numpy()).all())
        self.assertEqual(set(folds["method"]), {
            experiment.METHOD_FOLD_LOCAL, experiment.METHOD_GLOBAL,
        })
        for target in self.stage2.targets:
            target_folds = folds.loc[folds["target"] == target]
            safe = target_folds.loc[
                target_folds["method"] == experiment.METHOD_FOLD_LOCAL
            ].sort_values("fold")
            global_method = target_folds.loc[
                target_folds["method"] == experiment.METHOD_GLOBAL
            ].sort_values("fold")
            self.assertEqual(
                safe[["train_index_sha256", "validation_index_sha256"]].to_numpy().tolist(),
                global_method[["train_index_sha256", "validation_index_sha256"]].to_numpy().tolist(),
            )
        self.assertTrue(manifest["methodology"]["same_folds"])
        self.assertFalse(manifest["test_partition_accessed"])
        self.assertEqual(target_summary["missing_label_rows"].tolist(), [1, 1, 1])

    def test_guards_reject_missing_targets_bad_labels_and_forbidden_features(self):
        X, y_all = self.make_inputs()
        broken_y = y_all.drop(columns=[self.stage2.targets[0]])
        with self.assertRaisesRegex(ValueError, "Missing required"):
            experiment.validate_inputs(X, broken_y, self.stage2)

        bad_labels = y_all.copy()
        bad_labels.loc[bad_labels.index[0], self.stage2.targets[0]] = 2
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            experiment.validate_target(
                self.stage2.targets[0], bad_labels[self.stage2.targets[0]],
                self.stage2.n_splits,
            )

        leaked_features = X.copy()
        leaked_features["DEATH_BINARY"] = 0
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            experiment.validate_inputs(leaked_features, y_all, self.stage2)

        too_few = pd.Series([0, 0, 0, 0, 1] + [np.nan] * 26)
        with self.assertRaisesRegex(ValueError, "at least 5 examples"):
            experiment.validate_target("small_target", too_few, self.stage2.n_splits)

    def test_guards_reject_nonbinary_predictors_and_row_misalignment(self):
        X, y_all = self.make_inputs()
        broken = X.copy()
        broken.loc[broken.index[0], "SEX"] = 2
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            experiment.validate_inputs(broken, y_all, self.stage2)
        with self.assertRaisesRegex(ValueError, "same unique row index"):
            experiment.validate_inputs(X, y_all.set_axis(range(len(y_all))), self.stage2)

    def test_writer_refuses_existing_output_directory(self):
        destination = Path("already-existing-output")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                experiment.write_outputs(
                    destination,
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                    {},
                )


if __name__ == "__main__":
    unittest.main()
