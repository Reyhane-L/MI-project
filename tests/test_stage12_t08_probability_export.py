from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import mi_stage12_t08_probability_export as t08


class SyntheticBundle:
    def __init__(self, target: str, model: str, calibration: dict,
                 threshold: float, probabilities: list[float]):
        self.target = target
        self.model_name = model
        self.feature_columns = ["AGE", "SEX"]
        self.metadata = {
            "target": target,
            "model": model,
            "calibration_method": "Platt",
            "calibration_parameters": calibration,
            "threshold": threshold,
        }
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.predict_calls = 0

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_calls += 1
        if len(X) != len(self._probabilities):
            raise AssertionError("Synthetic bundle saw an unexpected number of rows.")
        return self._probabilities

    def fit(self, *args, **kwargs):  # pragma: no cover - called only if regression occurs
        raise AssertionError("Frozen bundle must never be fit during T08.")


class T08ProbabilityExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stage1 = self.root / "stage1_outputs"
        self.stage10 = self.root / "stage10_t06_non_tabpfn_outputs"
        self.stage11 = self.root / "stage11_t07_outputs"
        self.out = self.root / "stage12_t08_outputs"
        self.stage1.mkdir()
        self.stage10.mkdir()
        self.stage11.mkdir()
        self.ids = pd.Series([101, 102, 103], name="ID")
        self.ids.to_frame().to_csv(self.stage1 / "test_record_ids.csv", index=False)
        pd.DataFrame({
            "partition": ["train", "test"],
            "patients": [12, 3],
        }).to_csv(self.stage1 / "split_summary.csv", index=False)
        pd.DataFrame({"AGE": [55, 67, 72], "SEX": [1, 0, 1]}).to_csv(
            self.stage1 / "X_test.csv", index=False,
        )

        self.models = dict(t08.EXPECTED_MODELS)
        self.calibrations = {
            target: {"method": "Platt", "coefficient": i + 0.5, "intercept": -0.1}
            for i, target in enumerate(t08.TARGETS)
        }
        self.thresholds = {
            "DEATH_BINARY": 0.15,
            "ZSN": 0.16,
            "FIBR_PREDS": 0.07,
        }
        self.bundles = {}
        self.bundle_hashes = {}
        fallback_probs = {
            "DEATH_BINARY": [0.11, 0.22, 0.33],
            "ZSN": [0.44, 0.55, 0.66],
            "FIBR_PREDS": [0.77, 0.88, 0.99],
        }
        for i, target in enumerate(t08.TARGETS):
            name = t08.BUNDLE_FILES[target]
            bundle_path = self.stage10 / name
            bundle_path.write_bytes(f"synthetic bundle {target}".encode())
            self.bundle_hashes[target] = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
            self.bundles[target] = SyntheticBundle(
                target, self.models[target], self.calibrations[target],
                self.thresholds[target], fallback_probs[target],
            )

        manifest = {
            "stage": "T06",
            "freeze_revision": "non_tabpfn_v1",
            "targets": list(t08.TARGETS),
            "models": {
                target: {"model": self.models[target]} for target in t08.TARGETS
            },
            "calibration_parameters": self.calibrations,
        }
        self.t06_manifest_path = self.stage10 / "final_model_manifest.json"
        self.t06_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        pd.DataFrame([
            {"target": target, "threshold": self.thresholds[target]}
            for target in t08.TARGETS
        ]).to_csv(self.stage10 / "selected_thresholds.csv", index=False)

        self.patches = [
            patch.object(t08, "ROOT", self.root),
            patch.object(t08, "STAGE1", self.stage1),
            patch.object(t08, "STAGE10", self.stage10),
            patch.object(t08, "STAGE11", self.stage11),
            patch.object(t08, "OUT", self.out),
            patch.object(t08, "X_TEST_FILE", self.stage1 / "X_test.csv"),
            patch.object(t08, "TEST_IDS_FILE", self.stage1 / "test_record_ids.csv"),
            patch.object(t08, "SPLIT_SUMMARY_FILE", self.stage1 / "split_summary.csv"),
            patch.object(t08, "T06_MANIFEST_FILE", self.t06_manifest_path),
            patch.object(t08, "THRESHOLD_FILE", self.stage10 / "selected_thresholds.csv"),
            patch.object(t08, "T07_MANIFEST_FILE", self.stage11 / "t07_run_manifest.json"),
            patch.object(t08, "T07_PREDICTIONS_FILE", self.stage11 / "test_predictions.csv"),
            patch.object(t08.joblib, "load", side_effect=self._load_bundle),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp.cleanup()

    def _load_bundle(self, path):
        return self.bundles[Path(path).name.replace("_bundle.joblib", "").upper()]

    def _write_t07_predictions(self, *, wrong_bundle_hash=False,
                               reordered_ids=False):
        manifest = {
            "stage": "T07",
            "test_rows": 3,
            "target_order": list(t08.TARGETS),
            "models": {
                target: {"model": self.models[target]} for target in t08.TARGETS
            },
            "calibration_parameters": self.calibrations,
            "thresholds": self.thresholds,
            "bundle_sha256": dict(self.bundle_hashes),
            "input_sha256": {
                "stage1_outputs/test_record_ids.csv": t08._sha256(
                    self.stage1 / "test_record_ids.csv"
                ),
                "stage1_outputs/X_test.csv": t08._sha256(
                    self.stage1 / "X_test.csv"
                ),
            },
        }
        if wrong_bundle_hash:
            manifest["bundle_sha256"]["DEATH_BINARY"] = "wrong-hash"
        (self.stage11 / "t07_run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        row_ids = [103, 102, 101] if reordered_ids else self.ids.tolist()
        rows = []
        for target_index, target in enumerate(t08.TARGETS):
            for row_index, record_id in enumerate(row_ids):
                rows.append({
                    "ID": record_id,
                    "true_label": row_index % 2,
                    "calibrated_probability": [0.1, 0.2, 0.3][row_index] + target_index * 0.1,
                    "target": target,
                    "predicted_label": 0,
                })
        pd.DataFrame(rows).to_csv(self.stage11 / "test_predictions.csv", index=False)

    def test_reuses_valid_t07_probabilities_and_maps_death_to_let_is(self):
        self._write_t07_predictions()
        self.assertTrue(t08.run_export(approval_granted=True) is None)
        submission = pd.read_csv(self.out / "submission.csv")
        self.assertEqual(submission.columns.tolist(), list(t08.OUTPUT_COLUMNS))
        self.assertEqual(len(submission), 3)
        np.testing.assert_array_equal(submission["ID"], self.ids)
        np.testing.assert_allclose(submission["LET_IS"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(submission["ZSN"], [0.2, 0.3, 0.4])
        np.testing.assert_allclose(submission["FIBR_PREDS"], [0.3, 0.4, 0.5])
        manifest = json.loads((self.out / "t08_run_manifest.json").read_text())
        self.assertEqual(manifest["probability_source"]["status"], "validated")
        self.assertFalse(manifest["test_labels_loaded"])
        self.assertFalse(manifest["thresholds_applied"])
        self.assertIn("stage1_outputs/X_test.csv", manifest["input_sha256"])
        self.assertEqual([b.predict_calls for b in self.bundles.values()], [0, 0, 0])

    def test_unsuitable_t07_bundle_provenance_falls_back_to_frozen_inference(self):
        self._write_t07_predictions(wrong_bundle_hash=True)
        t08.run_export(approval_granted=True)
        submission = pd.read_csv(self.out / "submission.csv")
        self.assertEqual(submission.columns.tolist(), list(t08.OUTPUT_COLUMNS))
        np.testing.assert_allclose(submission["LET_IS"], [0.11, 0.22, 0.33])
        np.testing.assert_allclose(submission["ZSN"], [0.44, 0.55, 0.66])
        np.testing.assert_allclose(submission["FIBR_PREDS"], [0.77, 0.88, 0.99])
        manifest = json.loads((self.out / "t08_run_manifest.json").read_text())
        self.assertEqual(manifest["probability_source"]["status"], "fallback_used")
        self.assertEqual([b.predict_calls for b in self.bundles.values()], [1, 1, 1])

    def test_missing_t07_predictions_falls_back_to_frozen_inference(self):
        t08.run_export(approval_granted=True)
        manifest = json.loads((self.out / "t08_run_manifest.json").read_text())
        self.assertEqual(manifest["probability_source"]["status"], "fallback_used")
        self.assertEqual([b.predict_calls for b in self.bundles.values()], [1, 1, 1])

    def test_misaligned_t07_ids_fall_back_to_frozen_inference(self):
        self._write_t07_predictions(reordered_ids=True)
        t08.run_export(approval_granted=True)
        manifest = json.loads((self.out / "t08_run_manifest.json").read_text())
        self.assertEqual(manifest["probability_source"]["status"], "fallback_used")
        np.testing.assert_array_equal(pd.read_csv(self.out / "submission.csv")["ID"], self.ids)
        self.assertEqual([b.predict_calls for b in self.bundles.values()], [1, 1, 1])

    def test_changed_test_features_make_t07_source_unsuitable(self):
        self._write_t07_predictions()
        run_path = self.stage11 / "t07_run_manifest.json"
        manifest = json.loads(run_path.read_text())
        manifest["input_sha256"]["stage1_outputs/X_test.csv"] = "stale-feature-hash"
        run_path.write_text(json.dumps(manifest), encoding="utf-8")
        t08.run_export(approval_granted=True)
        exported = json.loads((self.out / "t08_run_manifest.json").read_text())
        self.assertEqual(exported["probability_source"]["status"], "fallback_used")
        self.assertIn("feature-input hash", exported["probability_source"]["reason"])
        self.assertEqual([b.predict_calls for b in self.bundles.values()], [1, 1, 1])

    def test_split_summary_enforces_expected_340_row_contract(self):
        pd.DataFrame({"partition": ["train", "test"], "patients": [1360, 340]}).to_csv(
            self.stage1 / "split_summary.csv", index=False,
        )
        with self.assertRaisesRegex(ValueError, "does not match expected 340"):
            t08.run_export(approval_granted=True)
        self.assertFalse(self.out.exists())

    def test_t07_reader_selects_only_id_target_and_probability_columns(self):
        self._write_t07_predictions()
        real_read_csv = pd.read_csv
        read_options = []

        def checked_read_csv(path, *args, **kwargs):
            if Path(path) == self.stage11 / "test_predictions.csv":
                read_options.append(kwargs.get("usecols"))
            return real_read_csv(path, *args, **kwargs)

        with patch.object(t08.pd, "read_csv", side_effect=checked_read_csv):
            t08.run_export(approval_granted=True)
        self.assertEqual(read_options, [list(t08.T07_PROBABILITY_COLUMNS)])

    def test_export_requires_approval_and_refuses_overwrite(self):
        with self.assertRaises(PermissionError):
            t08.run_export(approval_granted=False)
        self._write_t07_predictions()
        t08.run_export(approval_granted=True)
        with self.assertRaises(FileExistsError):
            t08.run_export(approval_granted=True)

    def test_submission_validator_rejects_bad_header_rows_ids_and_probabilities(self):
        good = pd.DataFrame({
            "ID": self.ids,
            "LET_IS": [0.1, 0.2, 0.3],
            "ZSN": [0.2, 0.3, 0.4],
            "FIBR_PREDS": [0.3, 0.4, 0.5],
        }, columns=list(t08.OUTPUT_COLUMNS))
        t08.validate_submission(good, self.ids, 3)
        with self.assertRaises(ValueError):
            t08.validate_submission(good.rename(columns={"LET_IS": "DEATH_BINARY"}), self.ids, 3)
        with self.assertRaises(ValueError):
            t08.validate_submission(good.iloc[:2], self.ids, 3)
        duplicate_ids = good.copy()
        duplicate_ids.loc[1, "ID"] = duplicate_ids.loc[0, "ID"]
        with self.assertRaises(ValueError):
            t08.validate_submission(duplicate_ids, self.ids, 3)
        out_of_order = good.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            t08.validate_submission(out_of_order, self.ids, 3)
        invalid_probability = good.copy()
        invalid_probability.loc[0, "ZSN"] = np.nan
        with self.assertRaises(ValueError):
            t08.validate_submission(invalid_probability, self.ids, 3)
        out_of_range = good.copy()
        out_of_range.loc[2, "FIBR_PREDS"] = 1.01
        with self.assertRaises(ValueError):
            t08.validate_submission(out_of_range, self.ids, 3)


if __name__ == "__main__":
    unittest.main()
