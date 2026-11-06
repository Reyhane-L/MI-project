"""T06: development-only cross-fitted calibration and model freeze.

The approved execution command is:
    .venv/bin/python mi_stage10_t06_freeze.py --approve-development-only-run

The script reads only explicit Stage 1 development inputs. Importing it and
running its focused tests does not create output directories or fit models.
"""
from __future__ import annotations

import argparse
import gc
import importlib
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mi_stage10_calibrators import SEED, clipped_probability, make_calibrator
from mi_stage10_t06_bundle import FinalModelBundle


ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE3 = ROOT / "stage3_outputs"
STAGE8 = ROOT / "stage8_tabpfn_outputs"
OUT = ROOT / "stage10_t06_outputs"
TARGETS = ("DEATH_BINARY", "ZSN", "FIBR_PREDS")
FINALISTS = {
    "DEATH_BINARY": "TabPFN 3.5",
    "ZSN": "TabPFN 3.5",
    "FIBR_PREDS": "Random Forest preliminary",
}
CALIBRATION_METHODS = ("Raw", "Platt", "Isotonic", "Beta", "Temperature")
OUTER_FOLDS = 5
INNER_FOLDS = 5
DEFAULT_MIN_SENSITIVITY = 0.80
TABPFN_CATEGORICAL = ("SEX", "ant_im", "lat_im", "inf_im", "post_im")
CONTINUOUS = (
    "AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT",
    "K_BLOOD", "Na_BLOOD", "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD",
    "L_BLOOD", "ROE",
)
ORDINAL = ("INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST", "GB", "DLIT_AG", "ZSN_A", "TIME_B_S")
NOMINAL = ("ant_im", "lat_im", "inf_im", "post_im")
FORBIDDEN_PREDICTORS = {
    "ID", "LET_IS", "DEATH_BINARY", "ZSN", "FIBR_PREDS",
    "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK", "OTEK_LANC",
    "RAZRIV", "R_AB_1_n", "R_AB_2_n", "R_AB_3_n", "NA_R_1_n", "NA_R_2_n",
    "NA_R_3_n", "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
}
OUTPUT_FILES = (
    "final_model_selections.csv",
    "calibration_method_comparison.csv",
    "cross_fitted_oof_predictions.csv",
    "selected_thresholds.csv",
    "final_model_manifest.json",
    "death_binary_bundle.joblib",
    "zsn_bundle.joblib",
    "fibr_preds_bundle.joblib",
    "death_binary_raw_vs_platt_reliability.png",
    "zsn_raw_vs_platt_reliability.png",
    "fibr_preds_raw_vs_platt_reliability.png",
    "fit_warnings.csv",
    "t06_freeze_report.md",
)


def cleanup_model_resources() -> None:
    """Collect released estimator objects and empty the MPS cache when available."""
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = getattr(mps_backend, "is_available", None)
    empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
    if callable(mps_available) and mps_available() and callable(empty_cache):
        try:
            empty_cache()
        except (AttributeError, RuntimeError):
            pass


def fold_categories(train: pd.DataFrame, evaluation: pd.DataFrame,
                    categorical_columns: tuple[str, ...] = TABPFN_CATEGORICAL
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Stage 8's native pandas categories, learning vocabulary on train only."""
    train = train.copy()
    evaluation = evaluation.copy()
    for column in categorical_columns:
        if column not in train or column not in evaluation:
            raise ValueError(f"Native categorical column is missing: {column}")
        categories = pd.Index(train[column].dropna().unique()).sort_values()
        train[column] = pd.Categorical(train[column], categories=categories)
        evaluation[column] = pd.Categorical(evaluation[column], categories=categories)
    return train, evaluation


def final_category_schema(train: pd.DataFrame,
                          categorical_columns: tuple[str, ...] = TABPFN_CATEGORICAL) -> dict[str, list]:
    """Save categories learned from the full development training data."""
    schema = {}
    for column in categorical_columns:
        if column not in train:
            raise ValueError(f"Native categorical column is missing: {column}")
        schema[column] = pd.Index(train[column].dropna().unique()).sort_values().tolist()
    return schema


def make_rf_pipeline(binary_columns: list[str]) -> Pipeline:
    """Reproduce Stage 3 preliminary forest with fold-local preprocessing."""
    prep = ColumnTransformer([
        ("continuous", SimpleImputer(strategy="median", keep_empty_features=True), list(CONTINUOUS)),
        ("ordinal", SimpleImputer(strategy="most_frequent"), list(ORDINAL)),
        ("binary", SimpleImputer(strategy="most_frequent"), binary_columns),
        ("ecg", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(NOMINAL)),
    ], remainder="drop")
    return Pipeline([
        ("preprocess", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=1, max_features="sqrt",
            class_weight=None,
        )),
    ])


def build_estimator(model_name: str, feature_columns: list[str]):
    if model_name == "TabPFN 3.5":
        classifier = importlib.import_module("tabpfn").TabPFNClassifier
        return classifier(
            random_state=42,
            device="auto",
            n_preprocessing_jobs=1,
            memory_saving_mode="auto",
            show_progress_bar=False,
        )
    if model_name == "Random Forest preliminary":
        typed = set(CONTINUOUS) | set(ORDINAL) | set(NOMINAL)
        return make_rf_pipeline([column for column in feature_columns if column not in typed])
    raise ValueError(f"Unsupported T06 finalist: {model_name}")


def _positive_probability(estimator, X: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1 or probability.ndim != 2 or probability.shape[1] != len(classes):
        raise ValueError("Finalist did not return a valid positive-class probability column.")
    result = probability[:, int(positive[0])]
    if result.ndim != 1 or len(result) != len(X) or not np.isfinite(result).all():
        raise ValueError("Finalist returned malformed or non-finite probabilities.")
    if ((result < 0) | (result > 1)).any():
        raise ValueError("Finalist returned probabilities outside [0, 1].")
    return result


def fit_predict_base(estimator, model_name: str, X_train: pd.DataFrame,
                     y_train: np.ndarray, X_evaluate: pd.DataFrame) -> np.ndarray:
    """Fit on only the supplied training fold and score the untouched evaluation fold."""
    if set(X_train.index).intersection(X_evaluate.index):
        raise ValueError("Training and evaluation row indexes overlap.")
    if model_name == "TabPFN 3.5":
        train_native, evaluate_native = fold_categories(X_train, X_evaluate)
        estimator.fit(train_native, np.asarray(y_train, dtype=int))
        return _positive_probability(estimator, evaluate_native)
    estimator.fit(X_train, np.asarray(y_train, dtype=int))
    return _positive_probability(estimator, X_evaluate)


def oof_base_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    estimator_builder,
    *,
    n_splits: int = INNER_FOLDS,
    seed: int = SEED,
    warning_sink: list[dict] | None = None,
    target: str = "injected_test",
    parent_fold: int = 0,
    phase: str = "oof_base_model",
) -> tuple[np.ndarray, np.ndarray]:
    """Return one deterministic stratified OOF score and fold number per input row."""
    labels = np.asarray(y, dtype=int)
    if len(X) != len(labels):
        raise ValueError("X and y lengths differ.")
    if set(np.unique(labels)) != {0, 1} or np.bincount(labels).min() < n_splits:
        raise ValueError("Both classes need at least n_splits observations.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.full(len(labels), np.nan, dtype=float)
    folds = np.zeros(len(labels), dtype=np.int8)
    for fold, (train_idx, validation_idx) in enumerate(splitter.split(X, labels), 1):
        estimator = estimator_builder(model_name, list(X.columns))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scores[validation_idx] = fit_predict_base(
                    estimator, model_name, X.iloc[train_idx], labels[train_idx],
                    X.iloc[validation_idx],
                )
        finally:
            del estimator
            cleanup_model_resources()
        if warning_sink is not None:
            warning_sink.extend(_warning_entries(
                caught, target, model_name, parent_fold or fold, "Raw", f"{phase}_fold_{fold}",
            ))
        folds[validation_idx] = fold
    if not np.isfinite(scores).all() or (folds == 0).any():
        raise RuntimeError("Cross-fitting did not produce exactly one finite score per row.")
    return scores, folds


def quality_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=int)
    raw_probability = np.asarray(probability, dtype=float)
    if (not np.isfinite(raw_probability).all()
            or ((raw_probability < 0) | (raw_probability > 1)).any()):
        raise ValueError("Metrics require finite probabilities in [0, 1].")
    p = clipped_probability(raw_probability)
    if len(y) != len(p) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Metrics require aligned probabilities and both binary classes.")
    predicted = (p >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "accuracy_at_0_5": float(accuracy_score(y, predicted)),
        "precision_at_0_5": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, predicted, zero_division=0)),
        "specificity_at_0_5": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1_at_0_5": float(f1_score(y, predicted, zero_division=0)),
        "true_negatives_at_0_5": int(tn),
        "false_positives_at_0_5": int(fp),
        "false_negatives_at_0_5": int(fn),
        "true_positives_at_0_5": int(tp),
    }


def compare_methods(oof: pd.DataFrame) -> pd.DataFrame:
    """Summarize Raw and all four calibration methods by fold and pooled OOF."""
    rows = []
    present_targets = [target for target in TARGETS if target in set(oof.target)]
    for target in present_targets:
        target_rows = oof.loc[oof.target == target]
        if target_rows.empty:
            raise ValueError(f"OOF predictions missing {target}.")
        for method in CALIBRATION_METHODS:
            column = f"{method.lower()}_probability"
            for fold in [1, 2, 3, 4, 5, "pooled"]:
                part = target_rows if fold == "pooled" else target_rows.loc[target_rows.outer_fold == fold]
                if part.empty:
                    raise ValueError(f"Missing OOF rows for {target}/{method}/fold {fold}.")
                row = {
                    "target": target,
                    "method": method,
                    "scope": "pooled" if fold == "pooled" else "outer_fold",
                    "outer_fold": fold,
                    "n": int(len(part)),
                }
                row.update(quality_metrics(part.true_label.to_numpy(), part[column].to_numpy()))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_method_comparison(
    detailed: pd.DataFrame, qualification: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """Return one requested comparison row per target and calibration method."""
    operating_point_fields = (
        "accuracy_at_0_5", "precision_at_0_5", "recall_at_0_5",
        "specificity_at_0_5", "f1_at_0_5",
        "true_negatives_at_0_5", "false_positives_at_0_5",
        "false_negatives_at_0_5", "true_positives_at_0_5",
    )
    confusion_count_fields = {
        "true_negatives_at_0_5", "false_positives_at_0_5",
        "false_negatives_at_0_5", "true_positives_at_0_5",
    }
    rows = []
    for target in TARGETS:
        target_rows = detailed.loc[detailed.target == target]
        if target_rows.empty:
            raise ValueError(f"Calibration comparison missing {target}.")
        for method in CALIBRATION_METHODS:
            pooled = target_rows.loc[
                (target_rows.method == method) & (target_rows.scope == "pooled")
            ]
            folds = target_rows.loc[
                (target_rows.method == method) & (target_rows.scope == "outer_fold")
            ].sort_values("outer_fold")
            if len(pooled) != 1 or len(folds) != OUTER_FOLDS:
                raise ValueError(f"Incomplete calibration comparison for {target}/{method}.")
            if method == "Platt":
                recommendation = "Frozen final method (assignment-required Platt)."
            elif method in qualification[target]:
                screen = qualification[target][method]
                recommendation = (
                    "Meets optional screen; descriptive only, Platt remains frozen."
                    if screen["qualifies"] else
                    "Does not meet optional screen; retain frozen Platt."
                )
            else:
                recommendation = "Raw identity reference; Platt remains frozen."
            fold_values = [
                {
                    "fold": int(row.outer_fold),
                    "brier_score": float(row.brier_score),
                    "log_loss": float(row.log_loss),
                    "average_precision": float(row.average_precision),
                    "auroc": float(row.auroc),
                    **{
                        field: (
                            int(getattr(row, field))
                            if field in confusion_count_fields else float(getattr(row, field))
                        )
                        for field in operating_point_fields
                    },
                }
                for row in folds.itertuples(index=False)
            ]
            row = pooled.iloc[0]
            summary_row = {
                "target": target,
                "method": method,
                "pooled_brier_score": float(row.brier_score),
                "pooled_log_loss": float(row.log_loss),
                "pooled_average_precision": float(row.average_precision),
                "pooled_auroc": float(row.auroc),
                "fold_level_comparison": json.dumps(fold_values, separators=(",", ":")),
                "recommendation": recommendation,
                "optional_screen": json.dumps(qualification[target].get(method, {}), separators=(",", ":")),
            }
            summary_row.update({
                f"pooled_{field}": (
                    int(row[field]) if field in confusion_count_fields else float(row[field])
                )
                for field in operating_point_fields
            })
            rows.append(summary_row)
    return pd.DataFrame(rows)


def optional_method_qualification(comparison: pd.DataFrame, target: str) -> dict[str, dict]:
    """Screen optional calibration for >=1% pooled improvements and >=4/5 paired wins."""
    table = comparison[comparison.target == target]
    platt = table[(table.method == "Platt") & (table.scope == "pooled")]
    if len(platt) != 1:
        raise ValueError(f"Pooled Platt metrics missing or duplicated for {target}.")
    baseline = platt.iloc[0]
    fold_platt = table[(table.method == "Platt") & (table.scope == "outer_fold")].set_index("outer_fold")
    if set(fold_platt.index) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Expected five Platt fold metrics for {target}.")
    result = {}
    for method in ("Isotonic", "Beta", "Temperature"):
        pooled = table[(table.method == method) & (table.scope == "pooled")]
        folds = table[(table.method == method) & (table.scope == "outer_fold")].set_index("outer_fold")
        if len(pooled) != 1 or set(folds.index) != {1, 2, 3, 4, 5}:
            raise ValueError(f"Missing calibration metrics for {target}/{method}.")
        row = pooled.iloc[0]
        brier_wins = int((folds.brier_score < fold_platt.brier_score).sum())
        log_loss_wins = int((folds.log_loss < fold_platt.log_loss).sum())
        brier_improvement = float((baseline.brier_score - row.brier_score) / baseline.brier_score)
        log_loss_improvement = float((baseline.log_loss - row.log_loss) / baseline.log_loss)
        result[method] = {
            "pooled_brier_relative_improvement": brier_improvement,
            "pooled_log_loss_relative_improvement": log_loss_improvement,
            "fold_brier_wins_vs_platt": brier_wins,
            "fold_log_loss_wins_vs_platt": log_loss_wins,
            "qualifies": bool(
                brier_improvement >= 0.01 and log_loss_improvement >= 0.01
                and brier_wins >= 4 and log_loss_wins >= 4
            ),
        }
    return result


def select_threshold(y_true: np.ndarray, probability: np.ndarray,
                     minimum_sensitivity: float = DEFAULT_MIN_SENSITIVITY) -> dict:
    """Maximize specificity subject to sensitivity, breaking specificity ties upward."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Threshold inputs must be aligned finite probabilities in [0, 1].")
    if not 0 <= minimum_sensitivity <= 1:
        raise ValueError("minimum_sensitivity must be between zero and one.")
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0 or not set(np.unique(y)).issubset({0, 1}):
        return {
            "threshold": float("nan"), "status": "constraint_unmet",
            "constraint": f"sensitivity >= {minimum_sensitivity:.2f}",
            "sensitivity": float("nan"), "specificity": float("nan"),
            "precision": float("nan"), "f1": float("nan"), "accuracy": float("nan"),
            "true_negatives": 0, "false_positives": 0,
            "false_negatives": 0, "true_positives": 0,
        }
    all_negative_cutoff = np.nextafter(float(np.max(p)), np.inf)
    candidates = np.unique(np.concatenate(([0.0], p, [all_negative_cutoff])))
    eligible = []
    for threshold in candidates:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = positives - tp
        tn = negatives - fp
        sensitivity = tp / positives
        specificity = tn / negatives
        if sensitivity + 1e-12 >= minimum_sensitivity:
            eligible.append((specificity, float(threshold), sensitivity, tn, fp, fn, tp))
    if not eligible:
        return {
            "threshold": float("nan"), "status": "constraint_unmet",
            "constraint": f"sensitivity >= {minimum_sensitivity:.2f}",
            "sensitivity": float("nan"), "specificity": float("nan"),
            "precision": float("nan"), "f1": float("nan"), "accuracy": float("nan"),
            "true_negatives": 0, "false_positives": 0,
            "false_negatives": 0, "true_positives": 0,
        }
    best_specificity = max(entry[0] for entry in eligible)
    best = max(
        (entry for entry in eligible if np.isclose(entry[0], best_specificity, atol=1e-12, rtol=0)),
        key=lambda entry: entry[1],
    )
    specificity, threshold, sensitivity, tn, fp, fn, tp = best
    return {
        "threshold": threshold, "status": "constraint_met",
        "constraint": f"sensitivity >= {minimum_sensitivity:.2f}",
        "sensitivity": sensitivity, "specificity": specificity,
        "true_negatives": tn, "false_positives": fp,
        "false_negatives": fn, "true_positives": tp,
        "accuracy": (tn + tp) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
    }


def load_selection_evidence() -> tuple[dict[str, dict], list[Path]]:
    """Load saved development CV summaries used to substantiate the fixed finalists."""
    paths = [STAGE3 / "aggregate_metrics.csv", STAGE8 / "aggregate_metrics.csv"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing explicit development model summaries: {missing}")
    summaries = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"target", "model", "average_precision_mean", "average_precision_std"}
    if not required.issubset(summaries.columns):
        raise ValueError("Saved model summaries lack AP means/sample SDs.")
    evidence = {}
    for target, selected_model in FINALISTS.items():
        target_rows = summaries.loc[summaries.target == target]
        selected = target_rows.loc[target_rows.model == selected_model]
        if selected.empty:
            raise ValueError(f"Missing development AP evidence for {target}/{selected_model}.")
        selected = selected.iloc[-1]
        alternatives = target_rows.loc[target_rows.model != selected_model].drop_duplicates(
            "model", keep="last",
        ).sort_values("average_precision_mean", ascending=False)
        evidence[target] = {
            "selected_mean": float(selected.average_precision_mean),
            "selected_sd": float(selected.average_precision_std),
            "alternatives": [
                {
                    "model": str(row.model),
                    "mean": float(row.average_precision_mean),
                    "sample_sd": float(row.average_precision_std),
                }
                for row in alternatives.itertuples(index=False)
            ],
        }
    return evidence, paths


def selection_rationale(target: str, evidence: dict) -> str:
    selected_ap = evidence["selected_mean"]
    alternatives = evidence["alternatives"]
    best = alternatives[0] if alternatives else None
    if target in {"DEATH_BINARY", "ZSN"}:
        comparison = (
            f"TabPFN mean AP {selected_ap:.4f} (fold SD {evidence['selected_sd']:.4f}) "
            f"exceeded the strongest saved alternative, {best['model']} "
            f"({best['mean']:.4f} +/- {best['sample_sd']:.4f})."
            if best else "No saved alternative was available."
        )
    else:
        comparison = (
            f"The user-specified preliminary Random Forest mean AP was "
            f"{selected_ap:.4f} (fold SD {evidence['selected_sd']:.4f}); "
            f"the highest saved alternative was {best['model']} at "
            f"{best['mean']:.4f} (fold SD {best['sample_sd']:.4f}). "
            "This near tie is descriptive; the supplied finalist is retained "
            "without retuning."
            if best else "No saved alternative was available."
        )
    return comparison + " Five-fold estimates are descriptive, not significance tests."


def crossfit_target(
    X: pd.DataFrame,
    y: np.ndarray,
    ids: np.ndarray,
    target: str,
    model_name: str,
    estimator_builder,
    *,
    prepare_fit_predict=fit_predict_base,
    warning_sink: list[dict] | None = None,
) -> pd.DataFrame:
    """Nested five-fold cross-fitting; each row is scored by an outer-held-out model."""
    labels = np.asarray(y, dtype=int)
    ids = np.asarray(ids)
    if len(X) != len(labels) or len(ids) != len(labels):
        raise ValueError("X, y, and development IDs are not row aligned.")
    if pd.isna(ids).any() or pd.Series(ids).duplicated().any():
        raise ValueError("Eligible development IDs must be non-null and unique.")
    if set(np.unique(labels)) != {0, 1} or np.bincount(labels).min() < OUTER_FOLDS:
        raise ValueError(f"{target} needs both classes with at least {OUTER_FOLDS} rows each.")
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(splitter.split(X, labels), 1):
        X_train, X_valid = X.iloc[outer_train_idx], X.iloc[outer_valid_idx]
        y_train, y_valid = labels[outer_train_idx], labels[outer_valid_idx]
        inner_scores, _ = oof_base_predictions(
            X_train, y_train, model_name, estimator_builder,
            n_splits=INNER_FOLDS, seed=SEED,
            warning_sink=warning_sink, target=target, parent_fold=outer_fold,
            phase="inner_base_model",
        )
        calibrators = {}
        for method in CALIBRATION_METHODS:
            if method == "Raw":
                continue
            calibrator = make_calibrator(method)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                calibrator.fit(inner_scores, y_train)
            calibrators[method] = calibrator
            if warning_sink is not None:
                warning_sink.extend(_warning_entries(
                    caught, target, model_name, outer_fold, method, "inner_calibrator",
                ))
        outer_model = estimator_builder(model_name, list(X.columns))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                raw = prepare_fit_predict(outer_model, model_name, X_train, y_train, X_valid)
        finally:
            del outer_model
            cleanup_model_resources()
        if warning_sink is not None:
            warning_sink.extend(_warning_entries(
                caught, target, model_name, outer_fold, "Raw", "outer_model",
            ))
        probabilities = {"raw_probability": raw}
        for method, calibrator in calibrators.items():
            calibrated = np.asarray(calibrator.predict(raw), dtype=float)
            if (not np.isfinite(calibrated).all()
                    or ((calibrated < 0) | (calibrated > 1)).any()):
                raise ValueError(f"{target}/{method} produced invalid outer probabilities.")
            probabilities[f"{method.lower()}_probability"] = calibrated
        for offset, row_id in enumerate(ids[outer_valid_idx]):
            row = {
                "target": target, "model": model_name, "ID": row_id,
                "true_label": int(y_valid[offset]), "outer_fold": outer_fold,
                "__row_position": int(outer_valid_idx[offset]),
            }
            row.update({name: float(values[offset]) for name, values in probabilities.items()})
            rows.append(row)
    result = pd.DataFrame(rows).sort_values("__row_position").drop(columns="__row_position").reset_index(drop=True)
    if len(result) != len(labels) or result.duplicated(["target", "ID"]).any():
        raise RuntimeError(f"Outer OOF rows are not one-to-one with eligible IDs for {target}.")
    expected = set(ids.tolist())
    if set(result.ID.tolist()) != expected:
        raise RuntimeError(f"Outer OOF IDs do not align with eligible development IDs for {target}.")
    return result


def _warning_entries(caught, target: str, model_name: str, fold: int,
                     method: str, phase: str) -> list[dict]:
    return [{
        "target": target, "model": model_name, "fold": fold, "method": method,
        "phase": phase, "category": warning.category.__name__,
        "message": str(warning.message),
    } for warning in caught]


def full_development_oof_and_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    estimator_builder,
    *,
    bundle_path: Path,
    threshold: float,
    threshold_status: str,
    warning_sink: list[dict] | None = None,
    target: str = "full_development",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit the final calibrator/model and stage a serialized bundle without retaining it."""
    labels = np.asarray(y, dtype=int)
    scores, folds = oof_base_predictions(
        X, labels, model_name, estimator_builder, n_splits=5, seed=SEED,
        warning_sink=warning_sink, target=target, phase="final_calibrator_base_model",
    )
    calibrator = make_calibrator("Platt")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrator.fit(scores, labels)
    if warning_sink is not None:
        warning_sink.extend(_warning_entries(
            caught, target, model_name, 0, "Platt", "final_calibrator",
        ))
    category_schema = final_category_schema(X) if model_name == "TabPFN 3.5" else None
    parameters = calibration_parameters(calibrator)
    _fit_and_stage_final_bundle(
        X=X,
        labels=labels,
        model_name=model_name,
        estimator_builder=estimator_builder,
        calibrator=calibrator,
        category_schema=category_schema,
        target=target,
        threshold=threshold,
        threshold_status=threshold_status,
        bundle_path=bundle_path,
        warning_sink=warning_sink,
    )
    return scores, folds, parameters


def calibration_parameters(calibrator) -> dict:
    """Serialize fitted Platt coefficients and fitting configuration."""
    model = calibrator.model
    return {
        "method": "Platt",
        "input_transform": "logit(clipped_probability)",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "classes": [int(value) for value in model.classes_],
        "C": float(model.C),
        "solver": str(model.solver),
        "max_iter": int(model.max_iter),
        "random_state": int(model.random_state),
    }


def _fit_and_stage_final_bundle(
    *,
    X: pd.DataFrame,
    labels: np.ndarray,
    model_name: str,
    estimator_builder,
    calibrator,
    category_schema: dict[str, list] | None,
    target: str,
    threshold: float,
    threshold_status: str,
    bundle_path: Path,
    warning_sink: list[dict] | None,
) -> None:
    """Fit, serialize, and release one full-development finalist before returning."""
    estimator = estimator_builder(model_name, list(X.columns))
    bundle = None
    try:
        if model_name == "TabPFN 3.5":
            train_native, _ = fold_categories(X, X.iloc[:0])
        else:
            train_native = X
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(train_native, labels)
        if warning_sink is not None:
            warning_sink.extend(_warning_entries(
                caught, target, model_name, 0, "Platt", "final_model",
            ))
        parameters = calibration_parameters(calibrator)
        bundle = FinalModelBundle(
            target=target,
            model_name=model_name,
            estimator=estimator,
            calibrator=calibrator,
            feature_columns=list(X.columns),
            category_schema=category_schema,
            metadata={
                "date_utc": datetime.now(timezone.utc).isoformat(),
                "target": target,
                "model": model_name,
                "calibration_method": "Platt",
                "calibration_parameters": parameters,
                "threshold": threshold,
                "threshold_status": threshold_status,
                "seed": SEED,
                "full_development_calibrator_oof_folds": 5,
                "full_development_oof_positive_prevalence": float(labels.mean()),
                "class_weight": None,
                "resampling": None,
                "category_schema": category_schema,
            },
        )
        joblib.dump(bundle, bundle_path, compress=3)
    finally:
        if bundle is not None:
            del bundle
        del estimator
        cleanup_model_resources()


def _package_versions() -> dict[str, str | None]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "tabpfn", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _validate_development_inputs(X: pd.DataFrame, y_all: pd.DataFrame,
                                 ids: pd.DataFrame, dictionary: pd.DataFrame) -> None:
    if not set(TARGETS).issubset(y_all.columns):
        raise ValueError("One or more required outcomes are missing from development labels.")
    if len(X) != len(y_all) or len(ids) != len(X):
        raise ValueError("Stage 1 development X, y, and ID rows are not aligned.")
    if ids.shape[1] != 1 or ids.iloc[:, 0].isna().any() or ids.iloc[:, 0].duplicated().any():
        raise ValueError("Development IDs must be one non-null, unique column.")
    if FORBIDDEN_PREDICTORS.intersection(X.columns):
        raise ValueError("Development predictors contain an ID, outcome, or forbidden field.")
    expected = dictionary.loc[
        dictionary.stage1_role == "admission-time predictor", "variable"
    ].tolist()
    if X.columns.tolist() != expected:
        raise ValueError("Development predictors do not match the admission-time dictionary.")
    if set(TABPFN_CATEGORICAL).difference(X.columns):
        raise ValueError("Stage 8 native categorical predictors are missing.")
    if set(CONTINUOUS + ORDINAL + NOMINAL).difference(X.columns):
        raise ValueError("Stage 3 preliminary forest feature groups are incomplete.")


def _run_crossfit(
    estimator_builder=build_estimator,
    minimum_sensitivity: float = DEFAULT_MIN_SENSITIVITY,
) -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite T06 outputs: {OUT}")
    with tempfile.TemporaryDirectory(prefix=".t06-staging-", dir=ROOT) as staging_root:
        stage_out = Path(staging_root) / OUT.name
        stage_out.mkdir()
        _run_crossfit_staged(estimator_builder, minimum_sensitivity, stage_out)
        if OUT.exists():
            raise FileExistsError(f"Refusing to overwrite T06 outputs: {OUT}")
        stage_out.rename(OUT)


def _run_crossfit_staged(
    estimator_builder,
    minimum_sensitivity: float,
    stage_out: Path,
) -> None:
    inputs = [
        STAGE1 / "X_train.csv",
        STAGE1 / "y_train.csv",
        STAGE1 / "train_record_ids.csv",
        STAGE1 / "column_dictionary.csv",
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing explicit development-only inputs: {missing}")
    X_all, y_all, ids_all, dictionary = [pd.read_csv(path) for path in inputs]
    _validate_development_inputs(X_all, y_all, ids_all, dictionary)
    selection_evidence, selection_paths = load_selection_evidence()
    id_column = ids_all.columns[0]
    warnings_found: list[dict] = []
    oof_parts = []
    fitted_calibration = {}
    thresholds = []

    for target in TARGETS:
        y_target = y_all[target].dropna().astype("int8")
        X_target = X_all.loc[y_target.index].copy()
        ids = ids_all.loc[y_target.index, id_column].to_numpy()
        model_name = FINALISTS[target]
        outer = crossfit_target(
            X_target, y_target.to_numpy(), ids, target, model_name,
            estimator_builder, warning_sink=warnings_found,
        )
        threshold = select_threshold(
            outer.true_label.to_numpy(), outer.platt_probability.to_numpy(),
            minimum_sensitivity=minimum_sensitivity,
        )
        thresholds.append({
            "target": target,
            "selection_criterion": "maximize specificity subject to minimum sensitivity",
            "source": "pooled five-fold outer Platt OOF",
            "minimum_sensitivity": minimum_sensitivity,
            **threshold,
        })
        final_scores, final_folds, parameters = full_development_oof_and_model(
            X_target, y_target.to_numpy(), model_name, estimator_builder,
            bundle_path=stage_out / f"{target.lower()}_bundle.joblib",
            threshold=threshold["threshold"],
            threshold_status=threshold["status"],
            warning_sink=warnings_found, target=target,
        )
        fitted_calibration[target] = parameters
        outer["final_calibrator_oof_raw_probability"] = final_scores
        outer["final_calibrator_oof_fold"] = final_folds
        oof_parts.append(outer)

    oof = pd.concat(oof_parts, ignore_index=True)
    comparison = compare_methods(oof)
    selections = []
    qualification = {}
    for target in TARGETS:
        optional = optional_method_qualification(comparison, target)
        qualification[target] = optional
        selections.append({
            "target": target,
            "selected_estimator": FINALISTS[target],
            "configuration": json.dumps(model_configurations()[target], sort_keys=True),
            "development_auprc_evidence": json.dumps(
                selection_evidence[target], separators=(",", ":"),
            ),
            "selection_rationale": selection_rationale(target, selection_evidence[target]),
            "selected_calibration": "Platt",
            "calibration_frozen": True,
            "optional_methods_can_override_platt": False,
            "decision": "Platt is frozen for the final model regardless of optional-method screen.",
        })
    comparison_output = summarize_method_comparison(comparison, qualification)

    threshold_frame = pd.DataFrame(thresholds)
    pd.DataFrame(selections).to_csv(stage_out / "final_model_selections.csv", index=False)
    comparison_output.to_csv(stage_out / "calibration_method_comparison.csv", index=False)
    oof.to_csv(stage_out / "cross_fitted_oof_predictions.csv", index=False)
    threshold_frame.to_csv(stage_out / "selected_thresholds.csv", index=False)

    _write_reliability_plots(oof, stage_out)
    manifest = {
        "stage": "T06",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "development_inputs": [str(path.relative_to(ROOT)) for path in inputs],
        "model_selection_development_summaries": [
            str(path.relative_to(ROOT)) for path in selection_paths
        ],
        "no_stage9_predictions_used_as_outer_scores": True,
        "targets": list(TARGETS),
        "models": model_configurations(),
        "model_selection_evidence": {
            target: {
                **selection_evidence[target],
                "rationale": selection_rationale(target, selection_evidence[target]),
            }
            for target in TARGETS
        },
        "calibration_parameters": fitted_calibration,
        "outer_cross_fitting": {"folds": 5, "shuffle": True, "random_state": 42, "stratified": True},
        "inner_calibrator_training": {"folds": 5, "shuffle": True, "random_state": 42, "stratified": True},
        "final_calibrator_training": {"full_development_oof_folds": 5, "random_state": 42},
        "calibration_methods": list(CALIBRATION_METHODS),
        "optional_calibrator_screen": {
            "pooled_brier_improvement_relative_to_platt": ">= 1%",
            "pooled_log_loss_improvement_relative_to_platt": ">= 1%",
            "per_metric_fold_wins": "at least 4/5",
            "qualification_results": qualification,
            "required_method_frozen": "Platt; optional screen never overrides it",
        },
        "threshold_rule": {
            "source": "pooled outer cross-fitted Platt OOF",
            "minimum_sensitivity": minimum_sensitivity,
            "selection": "maximize specificity; among ties choose largest threshold",
            "clinical_note": "0.80 is the selected operating-point assumption, not a universal clinical requirement.",
        },
        "packages": _package_versions(),
        "warnings": warnings_found,
        "expected_outputs": list(OUTPUT_FILES),
    }
    pd.DataFrame(warnings_found, columns=[
        "target", "model", "fold", "method", "phase", "category", "message",
    ]).to_csv(stage_out / "fit_warnings.csv", index=False)
    (stage_out / "final_model_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_value) + "\n"
    )
    _write_report(stage_out, comparison_output, threshold_frame, selections, warnings_found)
    missing_outputs = [name for name in OUTPUT_FILES if not (stage_out / name).is_file()]
    if missing_outputs:
        raise RuntimeError(f"T06 did not create its exact output set: {missing_outputs}")


def model_configurations() -> dict:
    return {
        "DEATH_BINARY": {
            "model": "TabPFN 3.5",
            "parameters": {
                "random_state": 42, "device": "auto", "n_preprocessing_jobs": 1,
                "memory_saving_mode": "auto", "show_progress_bar": False,
            },
            "native_categorical_columns": list(TABPFN_CATEGORICAL),
            "category_vocabulary": "learned from each model's training partition only",
            "class_weight": None, "resampling": None,
        },
        "ZSN": {
            "model": "TabPFN 3.5",
            "parameters": {
                "random_state": 42, "device": "auto", "n_preprocessing_jobs": 1,
                "memory_saving_mode": "auto", "show_progress_bar": False,
            },
            "native_categorical_columns": list(TABPFN_CATEGORICAL),
            "category_vocabulary": "learned from each model's training partition only",
            "class_weight": None, "resampling": None,
        },
        "FIBR_PREDS": {
            "model": "Random Forest preliminary",
            "parameters": {
                "n_estimators": 200, "random_state": 42, "max_features": "sqrt",
                "class_weight": None, "n_jobs": 1, "max_depth": None,
                "min_samples_leaf": 1, "min_samples_split": 2, "bootstrap": True,
            },
            "preprocessing": {
                "continuous": "fold-local median imputation",
                "ordinal_binary": "fold-local most-frequent imputation",
                "nominal": "fold-local most-frequent imputation then one-hot with unknown ignored",
                "scaling": "none",
            },
            "class_weight": None, "resampling": None,
        },
    }


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_reliability_plots(oof: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for target in TARGETS:
        part = oof[oof.target == target]
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
        for method, column in (("Raw", "raw_probability"), ("Platt", "platt_probability")):
            probability = part[column].to_numpy()
            labels = part.true_label.to_numpy()
            edges = np.linspace(0, 1, 11)
            bin_index = np.minimum(np.digitize(probability, edges[1:-1]), 9)
            mean_probability, observed = [], []
            for bin_number in range(10):
                selected = bin_index == bin_number
                if selected.any():
                    mean_probability.append(float(probability[selected].mean()))
                    observed.append(float(labels[selected].mean()))
            axis.plot(mean_probability, observed, marker="o", label=method)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
                 ylabel="Observed positive rate", title=f"{target}: development OOF reliability")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{target.lower()}_raw_vs_platt_reliability.png", dpi=180)
        plt.close(fig)


def _write_report(out: Path, comparison: pd.DataFrame, thresholds: pd.DataFrame,
                  selections: list[dict], warnings_found: list[dict]) -> None:
    lines = [
        "# T06 development-only model and calibration freeze",
        "",
        "Final calibration is Platt for all targets. Optional calibration methods",
        "are descriptive comparisons and cannot override the required frozen Platt method.",
        "All finalists use natural development prevalence without class weighting or resampling.",
        "Fresh seed-42 nested cross-fitting produced every outer and inner base-model",
        "prediction; Stage 9 raw predictions were not reused.",
        "",
        "| Target | Finalist | Development AP evidence (mean +/- fold SD) | Platt pooled Brier | Platt pooled Log Loss | Platt pooled AP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for selection in selections:
        target = selection["target"]
        row = comparison[(comparison.target == target) & (comparison.method == "Platt")].iloc[0]
        evidence = json.loads(selection["development_auprc_evidence"])
        lines.append(
            f"| {target} | {selection['selected_estimator']} | "
            f"{evidence['selected_mean']:.4f} +/- {evidence['selected_sd']:.4f} | "
            f"{row.pooled_brier_score:.6f} | {row.pooled_log_loss:.6f} | "
            f"{row.pooled_average_precision:.6f} |"
        )
        lines.append(f"\nSelection rationale for {target}: {selection['selection_rationale']}")
    lines.extend(["", "## Calibration method comparisons", "",
                  "| Target | Method | Pooled Brier | Pooled Log Loss | Pooled AP | Pooled AUROC | Recommendation |",
                  "|---|---|---:|---:|---:|---:|---|"])
    for row in comparison.to_dict(orient="records"):
        lines.append(
            f"| {row['target']} | {row['method']} | {row['pooled_brier_score']:.6f} | "
            f"{row['pooled_log_loss']:.6f} | {row['pooled_average_precision']:.6f} | "
            f"{row['pooled_auroc']:.6f} | {row['recommendation']} |"
        )
    lines.extend(["", "## Frozen development thresholds", "",
                  "| Target | Threshold | Status | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in thresholds.to_dict(orient="records"):
        threshold = row["threshold"]
        lines.append(
            f"| {row['target']} | {threshold:.8f} | {row['status']} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | "
            f"{row['precision']:.4f} | {row['f1']:.4f} | "
            f"{row['true_negatives']} | {row['false_positives']} | "
            f"{row['false_negatives']} | {row['true_positives']} |"
        )
    lines.extend([
        "",
        "Outer performance is development-only and descriptive. The full-development",
        "bundle calibrator is fitted to fresh five-fold cross-fitted base scores; its",
        "base estimator is then refit on all eligible development rows. The OOF and",
        "full-development estimators share fixed configurations and probability APIs;",
        "their training sizes differ, so calibration transfer remains a limitation.",
        "The Stage 9 comparison selected tuned settings using outer-training labels,",
        "which weakened independence of its inner calibration scores. T06 fixes the",
        "finalists/configurations and generates new inner OOF scores inside each outer",
        "training partition; outer-validation labels never fit their model/calibrator.",
        "The sensitivity minimum of 0.80 is an operating-point assumption, not a",
        "universal clinical requirement.",
        "",
        f"Captured fit/calibrator warnings: {len(warnings_found)} (details in fit_warnings.csv and manifest).",
        "No held-out evaluation is included or authorized by this freeze.",
        "",
    ])
    (out / "t06_freeze_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve-development-only-run", action="store_true",
        help="Explicitly authorize the bounded development-only fitting run and output creation.",
    )
    parser.add_argument(
        "--minimum-sensitivity", type=float, default=DEFAULT_MIN_SENSITIVITY,
        help="Development-OOF threshold constraint; default 0.80.",
    )
    args = parser.parse_args(argv)
    if not args.approve_development_only_run:
        parser.error("No models were fit. Obtain approval, then pass --approve-development-only-run.")
    if not np.isfinite(args.minimum_sensitivity) or not 0 <= args.minimum_sensitivity <= 1:
        parser.error("--minimum-sensitivity must be finite and between 0 and 1.")
    _run_crossfit(minimum_sensitivity=args.minimum_sensitivity)


if __name__ == "__main__":
    main()
