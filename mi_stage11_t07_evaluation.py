"""Approval-gated, one-time evaluation of the frozen T06 models on held-out data."""
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE10 = ROOT / "stage10_t06_outputs"
OUT = ROOT / "stage11_t07_outputs"
ATTEMPT_MARKER_FILE = ROOT / ".stage11_t07_attempt_started"
TARGETS = ("DEATH_BINARY", "ZSN", "FIBR_PREDS")
TARGET_OUTCOMES = ("LET_IS", "ZSN", "FIBR_PREDS", "DEATH_BINARY")
FORBIDDEN_PREDICTORS = {
    "ID", *TARGET_OUTCOMES,
    "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK", "OTEK_LANC",
    "RAZRIV", "R_AB_1_n", "R_AB_2_n", "R_AB_3_n", "NA_R_1_n", "NA_R_2_n",
    "NA_R_3_n", "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
}
BUNDLE_FILES = {
    "DEATH_BINARY": "death_binary_bundle.joblib",
    "ZSN": "zsn_bundle.joblib",
    "FIBR_PREDS": "fibr_preds_bundle.joblib",
}
PLOT_FILES = {
    "DEATH_BINARY": "death_binary_reliability.png",
    "ZSN": "zsn_reliability.png",
    "FIBR_PREDS": "fibr_preds_reliability.png",
}
DEVELOPMENT_OOF_FILE = STAGE10 / "cross_fitted_oof_predictions.csv"
DEVELOPMENT_PLOT_FILES = {
    "DEATH_BINARY": "development_death_binary_raw_vs_platt_reliability.png",
    "ZSN": "development_zsn_raw_vs_platt_reliability.png",
    "FIBR_PREDS": "development_fibr_preds_raw_vs_platt_reliability.png",
}
DCA_PLOT_FILES = {
    "DEATH_BINARY": "death_binary_decision_curve.png",
    "ZSN": "zsn_decision_curve.png",
    "FIBR_PREDS": "fibr_preds_decision_curve.png",
}
OUTPUT_FILES = (
    "test_metrics.csv",
    "test_predictions.csv",
    "test_confusion_matrices.csv",
    "test_calibration_diagnostics.csv",
    "test_calibration_bins.csv",
    "test_metric_confidence_intervals.csv",
    "test_decision_curve_analysis.csv",
    *PLOT_FILES.values(),
    "development_oof_calibration_comparison.csv",
    *DEVELOPMENT_PLOT_FILES.values(),
    *DCA_PLOT_FILES.values(),
    "t07_evaluation_report.md",
    "t07_run_manifest.json",
)
INPUT_FILES = (
    STAGE1 / "X_test.csv",
    STAGE1 / "y_test.csv",
    STAGE1 / "test_record_ids.csv",
)
SPLIT_SUMMARY_FILE = STAGE1 / "split_summary.csv"
SPLIT_SOURCE_FILE = ROOT / "mi_stage1_eda.py"
THRESHOLD_FILE = STAGE10 / "selected_thresholds.csv"
T06_MANIFEST_FILE = STAGE10 / "final_model_manifest.json"
CALIBRATION_BINS = 10
QUANTILE_BINS = 10
DCA_THRESHOLDS = np.round(np.arange(0.05, 0.401, 0.01), 2)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42


def _validate_test_inputs(X: pd.DataFrame, y: pd.DataFrame, ids: pd.DataFrame) -> None:
    if X.empty or len(X) != len(y) or len(X) != len(ids):
        raise ValueError("Held-out X, y, and ID files must be nonempty and have equal row counts.")
    if X.columns.has_duplicates or y.columns.has_duplicates:
        raise ValueError("Held-out predictors and outcomes must have unique columns.")
    if ids.shape[1] != 1 or ids.iloc[:, 0].isna().any() or ids.iloc[:, 0].duplicated().any():
        raise ValueError("Held-out IDs must be one non-null, unique column.")
    forbidden = FORBIDDEN_PREDICTORS.intersection(X.columns)
    if forbidden:
        raise ValueError(f"Held-out predictors contain forbidden columns: {sorted(forbidden)}")
    missing = {*TARGETS, "LET_IS"}.difference(y.columns)
    if missing:
        raise ValueError(f"Held-out outcomes are missing required targets: {sorted(missing)}")
    let_is = y["LET_IS"].to_numpy()
    if pd.isna(let_is).any() or not np.isfinite(let_is.astype(float)).all():
        raise ValueError("Held-out LET_IS labels must be non-null and finite.")
    if not np.array_equal(y["DEATH_BINARY"].to_numpy(), (y["LET_IS"] > 0).astype(int)):
        raise ValueError("Held-out DEATH_BINARY does not match the frozen LET_IS > 0 definition.")
    for target in TARGETS:
        values = y[target].to_numpy()
        if pd.isna(values).any() or not set(np.unique(values)).issubset({0, 1}):
            raise ValueError(f"Held-out target {target} must contain only binary 0/1 labels.")
        if len(np.unique(values)) != 2:
            raise ValueError(f"Held-out target {target} must contain both classes for evaluation.")


def _validate_split_summary(split_summary: pd.DataFrame, row_count: int) -> None:
    if not {"partition", "patients"}.issubset(split_summary.columns):
        raise ValueError("Stage 1 split summary is missing partition/patients columns.")
    test_partition = split_summary.loc[
        split_summary["partition"] == "test", "patients"
    ]
    if len(test_partition) != 1:
        raise ValueError("Held-out row count does not match the saved Stage 1 split summary.")
    try:
        saved_count = float(test_partition.iloc[0])
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 1 test row count must be a finite integer.") from error
    if not np.isfinite(saved_count) or not saved_count.is_integer() or int(saved_count) != row_count:
        raise ValueError("Held-out row count does not match the saved Stage 1 split summary.")


def _quantile_bin_indices(probability: np.ndarray) -> np.ndarray:
    """Assign quantile bins without splitting observations with tied scores."""
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("Quantile binning requires a nonempty probability vector.")
    if not np.isfinite(p).all():
        raise ValueError("Quantile binning probabilities must be finite.")
    groups = np.asarray(
        pd.qcut(p, q=min(QUANTILE_BINS, len(p)), labels=False, duplicates="drop"),
        dtype=float,
    )
    if np.isnan(groups).all():
        # A constant score vector has one valid reliability bin.
        return np.zeros(len(p), dtype=int)
    return groups.astype(int)


def _development_oof_comparison(oof: pd.DataFrame) -> pd.DataFrame:
    required = {"target", "ID", "true_label", "raw_probability", "platt_probability"}
    if not required.issubset(oof.columns):
        raise ValueError(f"T06 development OOF is missing columns: {sorted(required - set(oof.columns))}")
    if oof["target"].isna().any() or set(oof["target"].unique()) != set(TARGETS):
        raise ValueError("T06 development OOF must contain exactly the three frozen targets.")
    rows = []
    for target in TARGETS:
        part = oof.loc[oof["target"] == target]
        if part.empty or part["ID"].isna().any() or part["ID"].duplicated().any():
            raise ValueError(f"T06 development OOF IDs must be nonempty and unique for {target}.")
        labels = part["true_label"].to_numpy()
        if pd.isna(labels).any() or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"T06 development OOF labels are invalid for {target}.")
        raw = part["raw_probability"].to_numpy(dtype=float)
        platt = part["platt_probability"].to_numpy(dtype=float)
        for method, probability in (("raw", raw), ("platt", platt)):
            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError(f"T06 {method} OOF probabilities are invalid for {target}.")
        rows.append({
            "target": target,
            "n_development_oof": int(len(part)),
            "raw_brier_score": float(brier_score_loss(labels, raw)),
            "platt_brier_score": float(brier_score_loss(labels, platt)),
            "raw_log_loss": float(log_loss(labels, raw, labels=[0, 1])),
            "platt_log_loss": float(log_loss(labels, platt, labels=[0, 1])),
            "source": "T06 nested development outer OOF; identical IDs/labels for Raw and Platt",
            "calibration_method": "Fold-local development Platt OOF; final calibration remains frozen in T06",
        })
    return pd.DataFrame(rows)


def _validate_bundle(bundle: Any, target: str, t06_manifest: dict,
                     thresholds: pd.DataFrame, X: pd.DataFrame) -> float:
    if getattr(bundle, "target", None) != target:
        raise ValueError(f"Bundle target mismatch for {target}.")
    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, dict) or metadata.get("target") != target:
        raise ValueError(f"Bundle metadata target mismatch for {target}.")
    expected_model = t06_manifest["models"][target]["model"]
    if getattr(bundle, "model_name", None) != expected_model or metadata.get("model") != expected_model:
        raise ValueError(f"Frozen model metadata mismatch for {target}.")
    if metadata.get("calibration_method") != "Platt":
        raise ValueError(f"Frozen calibration method is not Platt for {target}.")
    if t06_manifest.get("calibration_parameters", {}).get(target, {}).get("method") != "Platt":
        raise ValueError(f"T06 manifest does not confirm frozen Platt calibration for {target}.")
    saved_calibration = t06_manifest["calibration_parameters"][target]
    bundle_calibration = metadata.get("calibration_parameters")
    if not isinstance(bundle_calibration, dict) or bundle_calibration != saved_calibration:
        raise ValueError(f"Frozen calibrator parameters do not match the T06 manifest for {target}.")

    feature_columns = getattr(bundle, "feature_columns", None)
    if not isinstance(feature_columns, list) or not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError(f"Bundle feature names are invalid for {target}.")
    if X.columns.tolist() != feature_columns:
        raise ValueError(f"Held-out feature names/order differ from the frozen bundle for {target}.")
    if set(feature_columns).intersection(FORBIDDEN_PREDICTORS):
        raise ValueError(f"Frozen bundle contains forbidden features for {target}.")

    selected = thresholds.loc[thresholds["target"] == target]
    if len(selected) != 1:
        raise ValueError(f"Expected one frozen threshold row for {target}.")
    threshold = float(selected.iloc[0]["threshold"])
    bundle_threshold = float(metadata.get("threshold", float("nan")))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Frozen threshold is invalid for {target}.")
    if not np.isclose(bundle_threshold, threshold, rtol=0, atol=1e-12):
        raise ValueError(f"Bundle threshold does not match the T06 threshold table for {target}.")
    return threshold


def _probability_vector(bundle: Any, X: pd.DataFrame, target: str) -> np.ndarray:
    predict = getattr(bundle, "predict_probability", None)
    if not callable(predict):
        raise TypeError(f"Bundle for {target} has no frozen calibrated prediction API.")
    probability = np.asarray(predict(X), dtype=float)
    if probability.ndim != 1 or len(probability) != len(X):
        raise ValueError(f"Bundle returned malformed probabilities for {target}.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Bundle returned non-finite or out-of-range probabilities for {target}.")
    return probability


def calibration_diagnostics(y_true: np.ndarray, probability: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Return descriptive calibration statistics; none changes frozen predictions."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Calibration labels and probabilities must be nonempty aligned vectors.")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Calibration labels must contain only finite binary 0/1 values.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Calibration probabilities must be finite and within [0, 1].")

    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1]), CALIBRATION_BINS - 1)
    ece = 0.0
    equal_width_rows = []
    for bin_index in range(CALIBRATION_BINS):
        mask = assignments == bin_index
        if mask.any():
            mean_probability = float(p[mask].mean())
            observed_rate = float(y[mask].mean())
            ece += float(mask.mean() * abs(observed_rate - mean_probability))
            equal_width_rows.append({
                "binning": "equal_width",
                "bin": bin_index,
                "count": int(mask.sum()),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            })

    quantile_groups = _quantile_bin_indices(p)
    quantile_rows = []
    for bin_index in sorted(np.unique(quantile_groups)):
        mask = quantile_groups == bin_index
        quantile_rows.append({
            "binning": "quantile",
            "bin": int(bin_index),
            "count": int(mask.sum()),
            "mean_probability": float(p[mask].mean()),
            "observed_rate": float(y[mask].mean()),
        })

    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def objective(parameters: np.ndarray, design: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(design @ parameters, -40, 40)
        fitted_probability = 1 / (1 + np.exp(-eta))
        loss = np.logaddexp(0, eta).sum() - np.dot(y, eta)
        gradient = design.T @ (fitted_probability - y)
        return float(loss), gradient

    slope_intercept = float("nan")
    slope_value = float("nan")
    slope_standard_error = float("nan")
    slope_uncertainty_status = "unavailable"
    slope_status = "unavailable"
    slope_detail = ""
    try:
        design = np.column_stack([np.ones(len(z)), z])
        joint = minimize(
            objective, np.array([0.0, 1.0]),
            args=(design,),
            method="L-BFGS-B", jac=True,
        )
        if joint.success and np.isfinite(joint.x).all():
            slope_intercept, slope_value = map(float, joint.x)
            slope_status = "available"
            fitted = 1 / (1 + np.exp(-np.clip(design @ joint.x, -40, 40)))
            information = design.T @ (
                design * (fitted * (1 - fitted))[:, np.newaxis]
            )
            covariance = np.linalg.inv(information)
            standard_errors = np.sqrt(np.diag(covariance))
            if np.isfinite(standard_errors).all():
                slope_standard_error = float(standard_errors[1])
                slope_uncertainty_status = "available"
            else:
                slope_detail = "Calibration slope information matrix produced non-finite uncertainty."
        else:
            slope_detail = str(joint.message)
    except (FloatingPointError, OverflowError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        slope_detail = f"{type(error).__name__}: {error}"
    diagnostics = {
        "ece_10_equal_width": float(ece),
        "calibration_intercept_joint": slope_intercept,
        "calibration_slope_joint": slope_value,
        "calibration_slope_status": slope_status,
        "calibration_slope_standard_error": slope_standard_error,
        "calibration_slope_uncertainty_status": slope_uncertainty_status,
        "calibration_slope_detail": slope_detail,
        "diagnostic_only": True,
    }
    return diagnostics, pd.DataFrame(equal_width_rows + quantile_rows)


def evaluate_target(target: str, y_true: np.ndarray, probability: np.ndarray,
                    threshold: float, ids: np.ndarray) -> tuple[dict, dict, pd.DataFrame, dict, pd.DataFrame]:
    raw_labels = np.asarray(y_true)
    if raw_labels.ndim != 1 or pd.isna(raw_labels).any() or set(np.unique(raw_labels)) != {0, 1}:
        raise ValueError(f"{target} labels must be a nonempty binary 0/1 vector.")
    labels = raw_labels.astype(int)
    p = np.asarray(probability, dtype=float)
    record_ids = np.asarray(ids)
    if labels.ndim != 1 or p.ndim != 1 or record_ids.ndim != 1:
        raise ValueError("Target labels, probabilities, and IDs must be vectors.")
    if not (len(labels) == len(p) == len(record_ids)) or len(labels) == 0:
        raise ValueError("Target labels, probabilities, and IDs must be nonempty and aligned.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{target} requires both binary classes for AUROC/calibration evaluation.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{target} probabilities must be finite and within [0, 1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"{target} frozen threshold must be finite and within [0, 1].")

    predicted = (p >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    prevalence = float(labels.mean())
    brier_baseline = prevalence * (1 - prevalence)
    brier = float(brier_score_loss(labels, p))
    metrics = {
        "target": target,
        "n": int(len(labels)),
        "positive_prevalence": prevalence,
        "auroc": float(roc_auc_score(labels, p)),
        "auprc_average_precision": float(average_precision_score(labels, p)),
        "brier_score": brier,
        "brier_baseline_score": brier_baseline,
        "brier_skill_score": float(1 - brier / brier_baseline),
        "log_loss": float(log_loss(labels, p, labels=[0, 1])),
        "frozen_threshold": float(threshold),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
    }
    confusion = {
        "target": target,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    diagnostics, bins = calibration_diagnostics(labels, p)
    diagnostics = {"target": target, **diagnostics}
    predictions = pd.DataFrame({
        "ID": record_ids,
        "true_label": labels,
        "calibrated_probability": p,
        "predicted_label": predicted,
        "target": target,
    })
    return metrics, confusion, predictions, diagnostics, bins


def decision_curve_analysis(y_true: np.ndarray, probability: np.ndarray,
                            thresholds: np.ndarray = DCA_THRESHOLDS) -> pd.DataFrame:
    """Calculate descriptive net benefit for model and treat-all/none policies."""
    raw_y = np.asarray(y_true)
    p = np.asarray(probability, dtype=float)
    threshold_grid = np.asarray(thresholds, dtype=float)
    if raw_y.ndim != 1 or p.ndim != 1 or len(raw_y) == 0 or len(raw_y) != len(p):
        raise ValueError("DCA labels and probabilities must be nonempty aligned vectors.")
    if (pd.isna(raw_y).any() or not set(np.unique(raw_y)).issubset({0, 1})
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError("DCA requires binary labels and finite probabilities in [0, 1].")
    y = raw_y.astype(int)
    if threshold_grid.ndim != 1 or len(threshold_grid) == 0:
        raise ValueError("DCA thresholds must be a nonempty vector.")
    if (not np.isfinite(threshold_grid).all()
            or ((threshold_grid <= 0) | (threshold_grid >= 1)).any()):
        raise ValueError("DCA thresholds must be finite and strictly between 0 and 1.")

    prevalence = float(y.mean())
    rows = []
    for threshold in threshold_grid:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        odds = float(threshold / (1 - threshold))
        rows.extend((
            {
                "threshold_probability": float(threshold),
                "strategy": "model",
                "net_benefit": float(tp / len(y) - (fp / len(y)) * odds),
            },
            {
                "threshold_probability": float(threshold),
                "strategy": "treat_all",
                "net_benefit": float(prevalence - (1 - prevalence) * odds),
            },
            {
                "threshold_probability": float(threshold),
                "strategy": "treat_none",
                "net_benefit": 0.0,
            },
        ))
    return pd.DataFrame(rows)


def bootstrap_ranking_confidence_intervals(
    target: str,
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Percentile CIs from deterministic paired row bootstrap; count invalid draws."""
    raw_y = np.asarray(y_true)
    p = np.asarray(probability, dtype=float)
    if raw_y.ndim != 1 or p.ndim != 1 or len(raw_y) == 0 or len(raw_y) != len(p):
        raise ValueError("Bootstrap labels and probabilities must be nonempty aligned vectors.")
    if pd.isna(raw_y).any() or not set(np.unique(raw_y)).issubset({0, 1}):
        raise ValueError("Bootstrap requires finite binary labels.")
    y = raw_y.astype(int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Bootstrap requires both binary classes in the evaluation sample.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Bootstrap probabilities must be finite and within [0, 1].")
    if not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError("Bootstrap replicate count must be a positive integer.")

    estimates = {
        "auroc": (float(roc_auc_score(y, p)), []),
        "auprc_average_precision": (float(average_precision_score(y, p)), []),
    }
    invalid_single_class = 0
    rng = np.random.default_rng(seed)
    n = len(y)
    for _ in range(replicates):
        sample = rng.integers(0, n, size=n)
        sample_y = y[sample]
        if len(np.unique(sample_y)) != 2:
            invalid_single_class += 1
            continue
        sample_p = p[sample]
        estimates["auroc"][1].append(float(roc_auc_score(sample_y, sample_p)))
        estimates["auprc_average_precision"][1].append(
            float(average_precision_score(sample_y, sample_p))
        )

    rows = []
    for metric, (point_estimate, values) in estimates.items():
        lower, upper = (np.percentile(values, [2.5, 97.5]) if values else (np.nan, np.nan))
        rows.append({
            "target": target,
            "metric": metric,
            "estimate": point_estimate,
            "ci_level": 0.95,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "bootstrap_replicates_requested": int(replicates),
            "bootstrap_replicates_valid": int(len(values)),
            "bootstrap_replicates_invalid_single_class": int(invalid_single_class),
            "bootstrap_seed": int(seed),
            "interval_method": "percentile paired row bootstrap with replacement",
            "status": "available" if values else "unavailable_no_valid_draws",
        })
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_attempt_marker(path: Path) -> None:
    """Prevent retrying after any approved run attempt has started."""
    with path.open("x", encoding="utf-8") as marker:
        marker.write(f"Started UTC: {datetime.now(timezone.utc).isoformat()}\n")


def _package_versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib", "tabpfn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _write_reliability_plot(target: str, bins: pd.DataFrame, path: Path) -> None:
    quantile = bins.loc[bins["binning"] == "quantile"]
    if quantile.empty:
        raise ValueError(f"No nonempty reliability bins available for {target}.")
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(
        quantile["mean_probability"], quantile["observed_rate"],
        marker="o", label="Frozen Platt probabilities",
    )
    axis.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title=f"{target}: held-out reliability (quantile bins)",
    )
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_decision_curve_plot(target: str, curve: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    for strategy in ("model", "treat_all", "treat_none"):
        part = curve.loc[curve["strategy"] == strategy]
        if part.empty:
            raise ValueError(f"Decision curve is missing {strategy} for {target}.")
        axis.plot(
            part["threshold_probability"], part["net_benefit"],
            label={"model": "Frozen model", "treat_all": "Treat all", "treat_none": "Treat none"}[strategy],
        )
    axis.set(
        xlim=(0.05, 0.40),
        xlabel="Threshold probability",
        ylabel="Net benefit",
        title=f"{target}: decision curve (descriptive)",
    )
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_development_comparison_plot(target: str, oof: pd.DataFrame, path: Path) -> None:
    part = oof.loc[oof["target"] == target]
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for label, column in (
        ("Raw base probability", "raw_probability"),
        ("Fold-local Platt probability", "platt_probability"),
    ):
        probability = part[column].to_numpy(dtype=float)
        labels = part["true_label"].to_numpy(dtype=float)
        quantile_groups = _quantile_bin_indices(probability)
        mean_probability = []
        observed_rate = []
        for bin_index in sorted(np.unique(quantile_groups)):
            mask = quantile_groups == bin_index
            mean_probability.append(float(probability[mask].mean()))
            observed_rate.append(float(labels[mask].mean()))
        axis.plot(mean_probability, observed_rate, marker="o", label=label)
    axis.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title=f"{target}: T06 development OOF calibration",
    )
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report(path: Path, metrics: pd.DataFrame, confusion: pd.DataFrame,
                  diagnostics: pd.DataFrame, development_comparison: pd.DataFrame,
                  confidence_intervals: pd.DataFrame,
                  decision_curves: pd.DataFrame, manifest: dict) -> None:
    lines = [
        "# T07: final held-out evaluation",
        "",
        "This is the single approval-gated evaluation of the frozen T06 bundles on the fixed Stage 1 test partition. "
        "The model, Platt calibrators, and thresholds were not fitted or changed using held-out labels. "
        "Calibration intercept/slope are descriptive diagnostics only and did not alter predictions.",
        "",
        "Ranking quality is summarized by AUROC and average precision (AUPRC). Probability/calibration quality is summarized separately by Brier score, Brier Skill Score against the observed-prevalence baseline, log loss, ECE, and calibration slope/intercept. "
        "Threshold metrics use each threshold loaded from the frozen bundle and cross-checked against `selected_thresholds.csv`. "
        "Test-set reliability diagrams describe frozen Platt predictions only. Raw-versus-Platt calibration comparisons below use T06 development outer-OOF predictions on the same IDs and labels; they do not use test labels or fit a new calibrator. "
        "Reliability diagrams use quantile bins; ECE uses ten equal-width bins. Empty equal-width bins are omitted from the diagnostics table.",
        "",
        "## Ranking quality",
        "",
        "| Target | Model | AUROC | AUPRC / AP |",
        "|---|---|---:|---:|",
    ]
    for row in metrics.to_dict(orient="records"):
        target = row["target"]
        model = manifest["models"][target]["model"]
        lines.append(
            f"| {target} | {model} | {row['auroc']:.6f} | "
            f"{row['auprc_average_precision']:.6f} |"
        )
    lines.extend([
        "",
        "## Ranking metric uncertainty",
        "",
        "Intervals are percentile 95% confidence intervals from a seeded paired row bootstrap of this single final evaluation. The frozen model and threshold were not refit or selected in bootstrap replicates. Replicates with only one observed class are omitted from interval estimation and counted explicitly.",
        "",
        "| Target | Metric | Estimate | 95% CI | Valid draws | Invalid single-class draws | Seed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in confidence_intervals.to_dict(orient="records"):
        lines.append(
            f"| {row['target']} | {row['metric']} | {row['estimate']:.6f} | "
            f"[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] | "
            f"{row['bootstrap_replicates_valid']} | "
            f"{row['bootstrap_replicates_invalid_single_class']} | {row['bootstrap_seed']} |"
        )
    lines.extend(["", "## Probability and calibration quality", "",
                  "| Target | Frozen calibration | Brier | Prevalence baseline Brier | Brier Skill Score | Log loss | ECE (10 equal-width) | Joint intercept | Joint slope | Slope SE | Slope status |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in metrics.to_dict(orient="records"):
        diagnostic = diagnostics.loc[diagnostics["target"] == row["target"]].iloc[0]
        lines.append(
            f"| {row['target']} | Platt | {row['brier_score']:.6f} | "
            f"{row['brier_baseline_score']:.6f} | {row['brier_skill_score']:.6f} | "
            f"{row['log_loss']:.6f} | "
            f"{diagnostic['ece_10_equal_width']:.6f} | "
            f"{diagnostic['calibration_intercept_joint']:.6g} | "
            f"{diagnostic['calibration_slope_joint']:.6g} | "
            f"{diagnostic['calibration_slope_standard_error']:.6g} | "
            f"{diagnostic['calibration_slope_status']} |"
        )
    lines.extend(["", "## Frozen-threshold classification", "",
                  "| Target | Threshold | Sensitivity | Specificity | Precision | F1 | Accuracy | TN | FP | FN | TP |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in metrics.to_dict(orient="records"):
        target = row["target"]
        conf = confusion.loc[confusion["target"] == target].iloc[0]
        lines.append(
            f"| {target} | {row['frozen_threshold']:.8f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | "
            f"{row['precision']:.4f} | {row['f1']:.4f} | {row['accuracy']:.4f} | "
            f"{conf['tn']} | {conf['fp']} | {conf['fn']} | {conf['tp']} |"
        )
    lines.extend([
        "",
        "## Decision curve analysis (descriptive)",
        "",
        "Net benefit is reported for the frozen calibrated probabilities over thresholds from 5% through 40% in one percentage-point steps, with treat-all and treat-none reference strategies. The curve uses the single evaluation sample only and is descriptive; it did not select or change the frozen operating threshold. See `test_decision_curve_analysis.csv` and the target-specific plots.",
    ])
    for target in TARGETS:
        part = decision_curves.loc[
            (decision_curves["target"] == target)
            & (decision_curves["strategy"] == "model")
        ]
        lines.append(
            f"- {target}: {len(part)} thresholds, from "
            f"{part['threshold_probability'].min():.2f} to "
            f"{part['threshold_probability'].max():.2f}."
        )
    lines.extend(["", "## Before/after calibration comparison (development only)", "",
                  "These descriptive metrics compare raw model probabilities with fold-local Platt probabilities on the same T06 outer-OOF development records. They are not test metrics and do not change the frozen T06 calibration.",
                  "",
                  "| Target | OOF records | Raw Brier | Platt Brier | Raw Log Loss | Platt Log Loss |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in development_comparison.to_dict(orient="records"):
        lines.append(
            f"| {row['target']} | {row['n_development_oof']} | "
            f"{row['raw_brier_score']:.6f} | {row['platt_brier_score']:.6f} | "
            f"{row['raw_log_loss']:.6f} | {row['platt_log_loss']:.6f} |"
        )
    lines.extend([
        "",
        "## Protocol and limitations",
        "",
        f"- Test records: {manifest['test_rows']}; input row order and IDs were preserved.",
        "- Brier Skill Score is `1 - Brier_model / (pi * (1 - pi))`, where `pi` is observed prevalence for the target in this evaluated partition, matching the assignment guide's prevalence-baseline formula.",
        "- Stage 1 source constructs the saved predictor, outcome, and ID files from the same `test_idx`; the test-row count is checked against Stage 1 `split_summary.csv`. Those CSVs contain no shared per-row key or alignment fingerprint, so the existing artifacts cannot independently prove X-to-ID correspondence. The run preserves their saved row order and records their hashes; it does not recreate the split.",
        "- Each target was evaluated independently. Test labels were used only for evaluation metrics, threshold confusion metrics, and descriptive calibration diagnostics.",
        "- Calibration diagnostics are not a refit of the frozen Platt calibrators; no result from this report may be used for model selection, recalibration, or threshold tuning.",
        "- This output contains evaluation labels for audit. It is not a T08 submission export.",
        "- Run hashes and exact input paths are recorded in `t07_run_manifest.json`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [str(path) for path in INPUT_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing explicit Stage 1 held-out inputs: {missing}")
    X, y, ids = [pd.read_csv(path) for path in INPUT_FILES]
    _validate_test_inputs(X, y, ids)
    return X, y, ids


def _load_bundles() -> dict[str, Any]:
    bundles = {}
    for target, filename in BUNDLE_FILES.items():
        path = STAGE10 / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen model bundle for {target}: {path}")
        bundles[target] = joblib.load(path)
    return bundles


def run_evaluation(*, approval_granted: bool = False) -> None:
    """Run only after the user separately approves held-out access."""
    if not approval_granted:
        raise PermissionError(
            "Held-out access is blocked until explicit user approval under AGENTS.md."
        )
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing T07 outputs: {OUT}")
    for path in (
        *INPUT_FILES, SPLIT_SUMMARY_FILE, SPLIT_SOURCE_FILE, THRESHOLD_FILE,
        T06_MANIFEST_FILE, DEVELOPMENT_OOF_FILE,
        *(STAGE10 / f for f in BUNDLE_FILES.values()),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required explicit T07 input is missing: {path}")
    _create_attempt_marker(ATTEMPT_MARKER_FILE)

    t06_manifest = json.loads(T06_MANIFEST_FILE.read_text(encoding="utf-8"))
    if t06_manifest.get("stage") != "T06" or tuple(t06_manifest.get("targets", ())) != TARGETS:
        raise ValueError("T06 manifest stage/targets do not match the frozen evaluation contract.")
    split_summary = pd.read_csv(SPLIT_SUMMARY_FILE)
    thresholds = pd.read_csv(THRESHOLD_FILE)
    if not {"target", "threshold"}.issubset(thresholds.columns):
        raise ValueError("T06 selected-threshold table is missing required columns.")
    development_oof = pd.read_csv(DEVELOPMENT_OOF_FILE)
    development_comparison = _development_oof_comparison(development_oof)

    X, y, ids = _load_inputs()
    _validate_split_summary(split_summary, len(X))
    bundles = _load_bundles()
    id_values = ids.iloc[:, 0].to_numpy()
    metrics_rows = []
    confusion_rows = []
    prediction_parts = []
    diagnostics_rows = []
    bin_parts = []
    confidence_interval_parts = []
    decision_curve_parts = []
    thresholds_used = {}

    for target in TARGETS:
        bundle = bundles[target]
        threshold = _validate_bundle(bundle, target, t06_manifest, thresholds, X)
        probability = _probability_vector(bundle, X, target)
        metrics, confusion, predictions, diagnostics, bins = evaluate_target(
            target, y[target].to_numpy(), probability, threshold, id_values,
        )
        metrics_rows.append(metrics)
        confusion_rows.append(confusion)
        prediction_parts.append(predictions)
        diagnostics_rows.append(diagnostics)
        confidence_interval_parts.append(
            bootstrap_ranking_confidence_intervals(
                target, y[target].to_numpy(), probability,
                replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
            )
        )
        decision_curve = decision_curve_analysis(y[target].to_numpy(), probability)
        decision_curve.insert(0, "target", target)
        decision_curve_parts.append(decision_curve)
        bins.insert(0, "target", target)
        bin_parts.append(bins)
        thresholds_used[target] = threshold

    metrics_frame = pd.DataFrame(metrics_rows)
    confusion_frame = pd.DataFrame(confusion_rows)
    predictions_frame = pd.concat(prediction_parts, ignore_index=True)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    calibration_bins = pd.concat(bin_parts, ignore_index=True)
    confidence_intervals = pd.concat(confidence_interval_parts, ignore_index=True)
    decision_curves = pd.concat(decision_curve_parts, ignore_index=True)

    input_hashes = {str(path.relative_to(ROOT)): _sha256(path)
                    for path in (*INPUT_FILES, SPLIT_SUMMARY_FILE, SPLIT_SOURCE_FILE,
                                 DEVELOPMENT_OOF_FILE,
                                 THRESHOLD_FILE, T06_MANIFEST_FILE)}
    bundle_hashes = {
        target: _sha256(STAGE10 / filename)
        for target, filename in BUNDLE_FILES.items()
    }
    run_manifest = {
        "stage": "T07",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "packages": _package_versions(),
        "test_rows": int(len(X)),
        "target_order": list(TARGETS),
        "thresholds": thresholds_used,
        "calibration_method": "Platt (frozen in T06)",
        "calibration_parameters": {
            target: t06_manifest["calibration_parameters"][target]
            for target in TARGETS
        },
        "models": {target: t06_manifest["models"][target] for target in TARGETS},
        "bundle_sha256": bundle_hashes,
        "input_sha256": input_hashes,
        "input_paths": [str(path.relative_to(ROOT)) for path in INPUT_FILES],
        "development_calibration_source": str(DEVELOPMENT_OOF_FILE.relative_to(ROOT)),
        "development_calibration_comparison": (
            "T06 outer OOF raw and fold-local Platt probabilities; same development rows; descriptive only."
        ),
        "split_row_count_source": str(SPLIT_SUMMARY_FILE.relative_to(ROOT)),
        "split_generation_source": str(SPLIT_SOURCE_FILE.relative_to(ROOT)),
        "split_design": {
            "method": "Stratified train/test split",
            "test_size": 0.20,
            "random_state": 42,
            "stratification_target": "DEATH_BINARY",
        },
        "split_alignment_evidence": (
            "Stage 1 source uses one test_idx for X/y/IDs and saved test row count is checked; "
            "test CSVs have no shared row-level key/fingerprint, so exact X-to-ID correspondence "
            "cannot be independently verified from current artifacts."
        ),
        "approval_procedure": (
            "Separate explicit user approval under AGENTS.md is required. The CLI flag only "
            "acknowledges that approval and cannot independently verify it."
        ),
        "threshold_source": str(THRESHOLD_FILE.relative_to(ROOT)),
        "t06_manifest": str(T06_MANIFEST_FILE.relative_to(ROOT)),
        "evaluation_protocol": "One held-out evaluation only; no model/calibrator/threshold selection or tuning.",
        "one_time_execution_guard": {
            "marker": str(ATTEMPT_MARKER_FILE.relative_to(ROOT)),
            "marker_sha256": _sha256(ATTEMPT_MARKER_FILE),
            "policy": (
                "The exclusive marker is created before input reads and retained after success or failure. "
                "Deleting the output directory does not permit another evaluation."
            ),
        },
        "test_labels_used_for": (
            "Evaluation metrics, descriptive calibration diagnostics, ranking-metric "
            "bootstrap intervals, and decision curve analysis only."
        ),
        "ranking_metric_confidence_intervals": {
            "metrics": ["auroc", "auprc_average_precision"],
            "method": "percentile paired row bootstrap with replacement",
            "confidence_level": 0.95,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "single_class_draws": "excluded and counted; no metric estimate for invalid draw",
        },
        "decision_curve_analysis": {
            "threshold_start": float(DCA_THRESHOLDS.min()),
            "threshold_end": float(DCA_THRESHOLDS.max()),
            "threshold_step": 0.01,
            "strategies": ["model", "treat_all", "treat_none"],
            "purpose": "descriptive only; does not select or change the frozen threshold",
        },
        "outputs": list(OUTPUT_FILES),
    }

    with tempfile.TemporaryDirectory(prefix=".t07-staging-", dir=ROOT) as temp_root:
        stage = Path(temp_root) / OUT.name
        stage.mkdir()
        metrics_frame.to_csv(stage / "test_metrics.csv", index=False)
        predictions_frame.to_csv(stage / "test_predictions.csv", index=False)
        confusion_frame.to_csv(stage / "test_confusion_matrices.csv", index=False)
        diagnostics_frame.to_csv(stage / "test_calibration_diagnostics.csv", index=False)
        calibration_bins.to_csv(stage / "test_calibration_bins.csv", index=False)
        confidence_intervals.to_csv(
            stage / "test_metric_confidence_intervals.csv", index=False,
        )
        decision_curves.to_csv(stage / "test_decision_curve_analysis.csv", index=False)
        development_comparison.to_csv(
            stage / "development_oof_calibration_comparison.csv", index=False,
        )
        for target, output_name in DEVELOPMENT_PLOT_FILES.items():
            _write_development_comparison_plot(
                target, development_oof, stage / output_name,
            )
        for target, filename in PLOT_FILES.items():
            _write_reliability_plot(
                target,
                calibration_bins.loc[calibration_bins["target"] == target],
                stage / filename,
            )
        for target, filename in DCA_PLOT_FILES.items():
            _write_decision_curve_plot(
                target,
                decision_curves.loc[decision_curves["target"] == target],
                stage / filename,
            )
        _write_report(
            stage / "t07_evaluation_report.md", metrics_frame, confusion_frame,
            diagnostics_frame, development_comparison, confidence_intervals,
            decision_curves, run_manifest,
        )
        (stage / "t07_run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        unexpected = set(path.name for path in stage.iterdir()).difference(OUTPUT_FILES)
        missing_outputs = set(OUTPUT_FILES).difference(path.name for path in stage.iterdir())
        if unexpected or missing_outputs:
            raise RuntimeError(
                f"Staged T07 output set mismatch; missing={sorted(missing_outputs)}, "
                f"unexpected={sorted(unexpected)}"
            )
        if OUT.exists():
            raise FileExistsError(f"Refusing to overwrite existing T07 outputs: {OUT}")
        os.rename(stage, OUT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acknowledge-separate-user-approval",
        dest="acknowledge_separate_user_approval",
        action="store_true",
        help="Acknowledge that explicit user approval was separately obtained; this flag does not grant approval.",
    )
    args = parser.parse_args(argv)
    if not args.acknowledge_separate_user_approval:
        parser.error(
            "T07 was not run. Obtain explicit user approval under AGENTS.md first, then pass "
            "--acknowledge-separate-user-approval."
        )
    run_evaluation(approval_granted=args.acknowledge_separate_user_approval)


if __name__ == "__main__":
    main()
