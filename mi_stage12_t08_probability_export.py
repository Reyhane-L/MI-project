"""Create the assignment probability CSV from frozen T06 probabilities."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE10 = ROOT / "stage10_t06_non_tabpfn_outputs"
STAGE11 = ROOT / "stage11_t07_outputs"
OUT = ROOT / "stage12_t08_outputs"
X_TEST_FILE = STAGE1 / "X_test.csv"
TEST_IDS_FILE = STAGE1 / "test_record_ids.csv"
SPLIT_SUMMARY_FILE = STAGE1 / "split_summary.csv"
T06_MANIFEST_FILE = STAGE10 / "final_model_manifest.json"
THRESHOLD_FILE = STAGE10 / "selected_thresholds.csv"
T07_MANIFEST_FILE = STAGE11 / "t07_run_manifest.json"
T07_PREDICTIONS_FILE = STAGE11 / "test_predictions.csv"

TARGETS = ("DEATH_BINARY", "ZSN", "FIBR_PREDS")
EXPECTED_MODELS = {
    "DEATH_BINARY": "CatBoost tuned",
    "ZSN": "Random Forest tuned",
    "FIBR_PREDS": "Random Forest preliminary",
}
BUNDLE_FILES = {
    "DEATH_BINARY": "death_binary_bundle.joblib",
    "ZSN": "zsn_bundle.joblib",
    "FIBR_PREDS": "fibr_preds_bundle.joblib",
}
OUTPUT_COLUMNS = ("ID", "LET_IS", "ZSN", "FIBR_PREDS")
TARGET_TO_OUTPUT_COLUMN = {
    "DEATH_BINARY": "LET_IS",
    "ZSN": "ZSN",
    "FIBR_PREDS": "FIBR_PREDS",
}
OUTPUT_FILES = ("submission.csv", "t08_run_manifest.json")
T07_PROBABILITY_COLUMNS = ("ID", "target", "calibrated_probability")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_test_rows(path: Path | None = None) -> int:
    if path is None:
        path = SPLIT_SUMMARY_FILE
    summary = pd.read_csv(path)
    if not {"partition", "patients"}.issubset(summary.columns):
        raise ValueError("Stage 1 split summary lacks partition/patients columns.")
    test_rows = summary.loc[summary["partition"] == "test", "patients"]
    if len(test_rows) != 1:
        raise ValueError("Expected exactly one test row in the Stage 1 split summary.")
    try:
        value = float(test_rows.iloc[0])
    except (TypeError, ValueError) as error:
        raise ValueError("Saved test row count is not numeric.") from error
    if not np.isfinite(value) or not value.is_integer() or value < 1:
        raise ValueError("Saved test row count must be a positive finite integer.")
    return int(value)


def _load_ids(path: Path, expected_rows: int) -> tuple[pd.Series, str]:
    ids = pd.read_csv(path)
    if ids.shape[1] != 1:
        raise ValueError("Saved test IDs must contain exactly one column.")
    values = ids.iloc[:, 0].reset_index(drop=True)
    if len(values) != expected_rows:
        raise ValueError(
            f"Saved test ID count {len(values)} does not match expected {expected_rows}."
        )
    if values.isna().any() or values.duplicated().any():
        raise ValueError("Saved test IDs must be non-null and unique.")
    return values, _sha256(path)


def _load_t06_configuration() -> tuple[dict, dict[str, float]]:
    if not T06_MANIFEST_FILE.is_file() or not THRESHOLD_FILE.is_file():
        raise FileNotFoundError("Frozen T06 manifest or threshold table is missing.")
    manifest = json.loads(T06_MANIFEST_FILE.read_text(encoding="utf-8"))
    if (manifest.get("stage") != "T06"
            or manifest.get("freeze_revision") != "non_tabpfn_v1"
            or tuple(manifest.get("targets", ())) != TARGETS):
        raise ValueError("T06 manifest is not the expected non_tabpfn_v1 freeze.")
    if {target: manifest.get("models", {}).get(target, {}).get("model")
            for target in TARGETS} != EXPECTED_MODELS:
        raise ValueError("T06 manifest model mapping differs from the approved finalists.")
    if any(manifest.get("calibration_parameters", {}).get(target, {}).get("method") != "Platt"
           for target in TARGETS):
        raise ValueError("T06 manifest does not specify frozen Platt calibration for all targets.")

    thresholds_frame = pd.read_csv(THRESHOLD_FILE)
    if not {"target", "threshold"}.issubset(thresholds_frame.columns):
        raise ValueError("T06 selected-threshold table lacks target/threshold columns.")
    thresholds: dict[str, float] = {}
    for target in TARGETS:
        selected = thresholds_frame.loc[thresholds_frame["target"] == target, "threshold"]
        if len(selected) != 1:
            raise ValueError(f"Expected one T06 threshold for {target}.")
        threshold = float(selected.iloc[0])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Frozen T06 threshold is invalid for {target}.")
        thresholds[target] = threshold
    return manifest, thresholds


def _load_frozen_bundles(manifest: dict, thresholds: dict[str, float]
                         ) -> tuple[dict[str, Any], dict[str, str]]:
    bundles: dict[str, Any] = {}
    bundle_hashes: dict[str, str] = {}
    for target, filename in BUNDLE_FILES.items():
        path = STAGE10 / filename
        if not path.is_file():
            raise FileNotFoundError(f"Frozen model bundle is missing: {path}")
        bundle = joblib.load(path)
        metadata = getattr(bundle, "metadata", None)
        if getattr(bundle, "target", None) != target:
            raise ValueError(f"Frozen bundle target mismatch for {target}.")
        if getattr(bundle, "model_name", None) != EXPECTED_MODELS[target]:
            raise ValueError(f"Frozen bundle model mismatch for {target}.")
        if not isinstance(metadata, dict):
            raise ValueError(f"Frozen bundle metadata is invalid for {target}.")
        if metadata.get("target") != target or metadata.get("model") != EXPECTED_MODELS[target]:
            raise ValueError(f"Frozen bundle metadata mismatch for {target}.")
        if metadata.get("calibration_method") != "Platt":
            raise ValueError(f"Frozen bundle calibration is not Platt for {target}.")
        expected_calibration = manifest["calibration_parameters"][target]
        if metadata.get("calibration_parameters") != expected_calibration:
            raise ValueError(f"Frozen bundle Platt parameters mismatch for {target}.")
        threshold = float(metadata.get("threshold", float("nan")))
        if not np.isfinite(threshold) or not np.isclose(
                threshold, thresholds[target], rtol=0, atol=1e-12):
            raise ValueError(f"Frozen bundle threshold mismatch for {target}.")
        features = getattr(bundle, "feature_columns", None)
        if (not isinstance(features, list) or not features
                or len(features) != len(set(features))):
            raise ValueError(f"Frozen bundle feature list is invalid for {target}.")
        bundles[target] = bundle
        bundle_hashes[target] = _sha256(path)
    return bundles, bundle_hashes


def _probability_vector(bundle: Any, X: pd.DataFrame, target: str) -> np.ndarray:
    predict = getattr(bundle, "predict_probability", None)
    if not callable(predict):
        raise TypeError(f"Frozen bundle for {target} lacks predict_probability().")
    probability = np.asarray(predict(X), dtype=float)
    if (probability.ndim != 1 or len(probability) != len(X)
            or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError(f"Frozen bundle returned invalid calibrated probabilities for {target}.")
    return probability


def _load_t07_probabilities(
    expected_ids: pd.Series,
    expected_rows: int,
    id_sha256: str,
    t06_manifest: dict,
    thresholds: dict[str, float],
    bundle_hashes: dict[str, str],
) -> tuple[dict[str, np.ndarray] | None, dict]:
    """Read only ID/target/probability columns; never materialize T07 labels."""
    if not T07_MANIFEST_FILE.is_file() or not T07_PREDICTIONS_FILE.is_file():
        return None, {"status": "unavailable", "reason": "T07 prediction artifact or manifest is missing."}

    try:
        run = json.loads(T07_MANIFEST_FILE.read_text(encoding="utf-8"))
        if run.get("stage") != "T07" or run.get("test_rows") != expected_rows:
            raise ValueError("T07 manifest stage or test row count mismatch.")
        if run.get("target_order") != list(TARGETS):
            raise ValueError("T07 manifest target order mismatch.")
        if (run.get("models") != t06_manifest.get("models")
                or run.get("calibration_parameters") != t06_manifest.get("calibration_parameters")):
            raise ValueError("T07 model or Platt calibration metadata differs from current T06.")
        if run.get("thresholds") != thresholds:
            raise ValueError("T07 thresholds differ from the current frozen T06 table.")
        if run.get("bundle_sha256") != bundle_hashes:
            raise ValueError("T07 bundle hashes differ from current frozen T06 bundles.")
        id_key = str(TEST_IDS_FILE.relative_to(ROOT))
        if run.get("input_sha256", {}).get(id_key) != id_sha256:
            raise ValueError("T07 run ID-input hash differs from the current saved test IDs.")
        if not X_TEST_FILE.is_file():
            raise ValueError("Current test features are unavailable for T07 provenance verification.")
        x_test_sha256 = _sha256(X_TEST_FILE)
        x_key = str(X_TEST_FILE.relative_to(ROOT))
        if run.get("input_sha256", {}).get(x_key) != x_test_sha256:
            raise ValueError("T07 run feature-input hash differs from current X_test.csv.")

        # `usecols` deliberately excludes `true_label`; no test-label values are
        # returned to application code or used in the submission.
        selected = pd.read_csv(T07_PREDICTIONS_FILE, usecols=list(T07_PROBABILITY_COLUMNS))
        if len(selected) != expected_rows * len(TARGETS):
            raise ValueError("T07 prediction row count is not 340 records per target.")
        probabilities: dict[str, np.ndarray] = {}
        for target in TARGETS:
            part = selected.loc[selected["target"] == target]
            if len(part) != expected_rows:
                raise ValueError(f"T07 predictions have the wrong row count for {target}.")
            part_ids = part["ID"].reset_index(drop=True)
            if not part_ids.equals(expected_ids.reset_index(drop=True)):
                raise ValueError(f"T07 ID order does not match the saved test IDs for {target}.")
            probability = part["calibrated_probability"].to_numpy(dtype=float)
            if (not np.isfinite(probability).all()
                    or ((probability < 0) | (probability > 1)).any()):
                raise ValueError(f"T07 calibrated probabilities are invalid for {target}.")
            probabilities[target] = probability
        selected_digest = _sha256_bytes(
            selected.loc[:, list(T07_PROBABILITY_COLUMNS)].to_csv(index=False).encode("utf-8")
        )
        return probabilities, {
            "status": "validated",
            "source": str(T07_PREDICTIONS_FILE.relative_to(ROOT)),
            "manifest": str(T07_MANIFEST_FILE.relative_to(ROOT)),
            "selected_probability_rows_sha256": selected_digest,
            "columns_loaded": list(T07_PROBABILITY_COLUMNS),
            "rows_per_target": expected_rows,
            "x_test_sha256": x_test_sha256,
            "labels_loaded": False,
        }
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return None, {
            "status": "unsuitable",
            "reason": f"{type(error).__name__}: {error}",
            "fallback": "frozen T06 bundle inference on X_test.csv",
        }


def validate_submission(
    submission: pd.DataFrame,
    expected_ids: pd.Series,
    expected_rows: int,
) -> None:
    if submission.columns.tolist() != list(OUTPUT_COLUMNS):
        raise ValueError(f"Submission header must be exactly {list(OUTPUT_COLUMNS)}.")
    if len(submission) != expected_rows:
        raise ValueError(f"Submission has {len(submission)} rows; expected {expected_rows}.")
    ids = submission["ID"].reset_index(drop=True)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Submission IDs must be non-null and unique.")
    if not ids.equals(expected_ids.reset_index(drop=True)):
        raise ValueError("Submission ID order differs from the saved test ID order.")
    for column in OUTPUT_COLUMNS[1:]:
        values = submission[column].to_numpy(dtype=float)
        if (not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()):
            raise ValueError(f"Submission probabilities are invalid in {column}.")


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "catboost")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _fallback_probabilities(
    bundles: dict[str, Any], expected_rows: int
) -> tuple[dict[str, np.ndarray], str]:
    if not X_TEST_FILE.is_file():
        raise FileNotFoundError(f"Frozen inference fallback requires {X_TEST_FILE}.")
    X = pd.read_csv(X_TEST_FILE)
    if X.columns.has_duplicates or len(X) != expected_rows:
        raise ValueError("Test predictor columns or row count do not match the frozen contract.")
    baseline_features: list[str] | None = None
    probabilities: dict[str, np.ndarray] = {}
    for target in TARGETS:
        features = bundles[target].feature_columns
        if baseline_features is None:
            baseline_features = features
        elif features != baseline_features:
            raise ValueError("Frozen target bundles do not share one feature order.")
        if X.columns.tolist() != features:
            raise ValueError(f"Test predictor feature order differs from frozen {target} bundle.")
        probabilities[target] = _probability_vector(bundles[target], X, target)
    return probabilities, _sha256(X_TEST_FILE)


def run_export(*, approval_granted: bool = False) -> None:
    """Stage one export; no labels, model fitting, or thresholding are used."""
    if not approval_granted:
        raise PermissionError("T08 test-feature/ID use requires explicit user approval.")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing T08 outputs: {OUT}")
    for path in (TEST_IDS_FILE, SPLIT_SUMMARY_FILE, T06_MANIFEST_FILE,
                 THRESHOLD_FILE, *(STAGE10 / name for name in BUNDLE_FILES.values())):
        if not path.is_file():
            raise FileNotFoundError(f"Required T08 input is missing: {path}")

    expected_rows = _expected_test_rows()
    ids, ids_sha256 = _load_ids(TEST_IDS_FILE, expected_rows)
    t06_manifest, thresholds = _load_t06_configuration()
    bundles, bundle_hashes = _load_frozen_bundles(t06_manifest, thresholds)
    probabilities, source_info = _load_t07_probabilities(
        ids, expected_rows, ids_sha256, t06_manifest, thresholds, bundle_hashes,
    )
    input_hashes = {str(TEST_IDS_FILE.relative_to(ROOT)): ids_sha256}
    if probabilities is None:
        probabilities, x_test_sha256 = _fallback_probabilities(bundles, expected_rows)
        input_hashes[str(X_TEST_FILE.relative_to(ROOT))] = x_test_sha256
        source_info = {
            **source_info,
            "status": "fallback_used",
            "source": str(X_TEST_FILE.relative_to(ROOT)),
            "method": "current frozen T06 bundle predict_probability() API",
            "rows_per_target": expected_rows,
            "labels_loaded": False,
        }
    else:
        input_hashes[str(X_TEST_FILE.relative_to(ROOT))] = source_info["x_test_sha256"]

    submission = pd.DataFrame({
        "ID": ids,
        "LET_IS": probabilities["DEATH_BINARY"],
        "ZSN": probabilities["ZSN"],
        "FIBR_PREDS": probabilities["FIBR_PREDS"],
    }, columns=list(OUTPUT_COLUMNS))
    validate_submission(submission, ids, expected_rows)
    submission_bytes = submission.to_csv(index=False).encode("utf-8")
    run_manifest = {
        "stage": "T08",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "packages": _package_versions(),
        "freeze_revision": t06_manifest["freeze_revision"],
        "target_order": list(TARGETS),
        "models": {target: t06_manifest["models"][target] for target in TARGETS},
        "calibration_method": "Platt (frozen in T06)",
        "calibration_parameters": {
            target: t06_manifest["calibration_parameters"][target] for target in TARGETS
        },
        "thresholds_verified_but_not_applied": thresholds,
        "probability_source": source_info,
        "submission_columns": list(OUTPUT_COLUMNS),
        "death_probability_mapping": "DEATH_BINARY calibrated probability -> LET_IS column",
        "rows": expected_rows,
        "input_sha256": input_hashes,
        "bundle_sha256": bundle_hashes,
        "submission_sha256": _sha256_bytes(submission_bytes),
        "test_labels_loaded": False,
        "thresholds_applied": False,
        "outputs": list(OUTPUT_FILES),
    }

    with tempfile.TemporaryDirectory(prefix=".t08-staging-", dir=ROOT) as temp_root:
        stage = Path(temp_root) / OUT.name
        stage.mkdir()
        (stage / "submission.csv").write_bytes(submission_bytes)
        (stage / "t08_run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if {path.name for path in stage.iterdir()} != set(OUTPUT_FILES):
            raise RuntimeError("Staged T08 output set differs from OUTPUT_FILES.")
        if OUT.exists():
            raise FileExistsError(f"Refusing to overwrite existing T08 outputs: {OUT}")
        os.rename(stage, OUT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acknowledge-approved-test-feature-use",
        action="store_true",
        help="Acknowledge explicit approval to use locked test features and IDs; this flag does not grant approval.",
    )
    args = parser.parse_args(argv)
    if not args.acknowledge_approved_test_feature_use:
        parser.error(
            "T08 export was not run. Obtain explicit approval for test-feature/ID use, "
            "then pass --acknowledge-approved-test-feature-use."
        )
    run_export(approval_granted=True)


if __name__ == "__main__":
    main()
