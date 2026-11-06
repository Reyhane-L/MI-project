"""Stage 9: nested development-only calibration comparison of finalists.

This script deliberately refits candidate estimators inside each outer
calibration fold. It never reads held-out test artifacts and refuses to
overwrite the Stage 9 output directory. The compute-intensive model run must
be explicitly authorized before execution.
"""
from __future__ import annotations

import ast
import gc
import os
from pathlib import Path
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage9-mplconfig"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import mi_stage3_random_forest as stage3
import mi_stage4_extratrees as stage4
import mi_stage6_catboost as stage6
import mi_stage8_tabpfn as stage8

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE3 = ROOT / "stage3_outputs"
STAGE4 = ROOT / "stage4_extratrees_outputs"
STAGE6 = ROOT / "stage6_catboost_outputs"
OUT = ROOT / "stage9_finalist_calibration_outputs"
SEED, OUTER_FOLDS, INNER_FOLDS = 42, 5, 4
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]
CANDIDATES = {
    "DEATH_BINARY": ["TabPFN 3.5", "CatBoost tuned"],
    "ZSN": ["TabPFN 3.5", "Random Forest tuned"],
    "FIBR_PREDS": ["Extra Trees preliminary", "Random Forest preliminary"],
}
EPS = 1e-6


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_sigmoid(raw_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling on logits with a weakly regularized two-parameter model."""
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=SEED)
    calibrator.fit(logit(raw_train), y_train.astype(int))
    return calibrator


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=int)
    bins = np.linspace(0, 1, 11)
    ids = np.minimum(np.digitize(p, bins[1:-1], right=False), 9)
    ece = 0.0
    for b in range(10):
        selected = ids == b
        if selected.any():
            ece += float(selected.mean() * abs(y[selected].mean() - p[selected].mean()))
    slope_fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=SEED)
    slope_fit.fit(logit(p), y)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10_equal_width_bins": ece,
        "calibration_intercept_diagnostic": float(slope_fit.intercept_[0]),
        "calibration_slope_diagnostic": float(slope_fit.coef_[0, 0]),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
        "specificity_at_0_5": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def params_by_fold(path: Path) -> dict[tuple[str, int], dict]:
    frame = pd.read_csv(path)
    result = {}
    for row in frame.itertuples(index=False):
        raw = getattr(row, "best_parameters", "")
        if isinstance(raw, str) and raw.strip():
            params = ast.literal_eval(raw)
        else:
            params = {}
        result[(row.target, int(getattr(row, "outer_fold")))] = params
    return result


def make_model(name: str, binary: list[str], target: str, fold: int,
               rf_params: dict, et_params: dict, cb_params: dict):
    if name == "Random Forest preliminary":
        return stage3.make_pipeline(binary)
    if name == "Random Forest tuned":
        model = stage3.make_pipeline(binary)
        return model.set_params(**{f"classifier__{k}": v for k, v in rf_params[(target, fold)].items()})
    if name == "Extra Trees preliminary":
        return stage4.make_pipeline(binary, n_estimators=200)
    if name == "CatBoost tuned":
        model = stage6.make_pipeline()
        return model.set_params(**{f"classifier__{k}": v for k, v in cb_params[(target, fold)].items()})
    if name == "TabPFN 3.5":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(random_state=SEED, device="auto", n_preprocessing_jobs=1,
                                memory_saving_mode="auto", show_progress_bar=False)
    raise ValueError(f"Unsupported candidate: {name}")


def fit_candidate(model, X_fit: pd.DataFrame, y_fit: np.ndarray, name: str) -> None:
    if name == "CatBoost tuned":
        # Stage 6 passes these as fit metadata to the CatBoost classifier
        # through the sklearn Pipeline; they are not constructor parameters.
        model.fit(X_fit, y_fit, classifier__cat_features=stage6.CATEGORICAL)
    else:
        model.fit(X_fit, y_fit)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 9 outputs: {OUT}")
    required = [
        STAGE1 / "X_train.csv", STAGE1 / "y_train.csv", STAGE1 / "train_record_ids.csv",
        STAGE1 / "column_dictionary.csv", STAGE3 / "tuning_results.csv",
        STAGE4 / "tuning_results.csv", STAGE6 / "tuning_results.csv",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing explicit development inputs: {missing}")
    X, y_all, ids, dictionary = [pd.read_csv(p) for p in required[:4]]
    stage6.validate_inputs(X, y_all, ids, dictionary)
    if len(X) != len(y_all) or len(ids) != len(X):
        raise ValueError("Development X/y/ID artifacts are not row aligned.")
    if "ID" in X.columns or set(TARGETS).intersection(X.columns):
        raise ValueError("ID or outcomes appear in development predictors.")
    id_col = ids.columns[0]
    typed = stage3.CONTINUOUS + stage3.ORDINAL + stage3.NOMINAL
    binary = [c for c in X.columns if c not in typed]
    rf_params = params_by_fold(required[4])
    et_params = params_by_fold(required[5])
    cb_params = params_by_fold(required[6])

    # Import TabPFN only after the output/input guards. No network/cache action is requested here.
    from importlib.metadata import version
    if version("tabpfn") != stage8.EXPECTED_TABPFN_VERSION:
        raise RuntimeError(f"Expected TabPFN {stage8.EXPECTED_TABPFN_VERSION}.")

    fold_rows, oof_rows, warning_rows = [], [], []
    for target in TARGETS:
        y = y_all[target].dropna().astype("int8")
        Xt, idt = X.loc[y.index].copy(), ids.loc[y.index, id_col]
        if y.value_counts().min() < OUTER_FOLDS:
            raise ValueError(f"Insufficient class counts for {target}.")
        outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        splits = list(outer.split(Xt, y))
        for outer_fold, (train_idx, valid_idx) in enumerate(splits, 1):
            X_train, X_valid = Xt.iloc[train_idx], Xt.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx].to_numpy(), y.iloc[valid_idx].to_numpy()
            inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + outer_fold)
            inner_splits = list(inner.split(X_train, y_train))
            for name in CANDIDATES[target]:
                warnings_seen = []
                # The fixed fold-specific tuned settings were selected using only this outer
                # training partition. Inner calibration scores and the final outer refit remain
                # isolated from outer validation labels.
                inner_oof = np.full(len(y_train), np.nan, dtype=float)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    for itr, iva in inner_splits:
                        model = make_model(name, binary, target, outer_fold, rf_params, et_params, cb_params)
                        X_fit, X_score = X_train.iloc[itr], X_train.iloc[iva]
                        if name == "TabPFN 3.5":
                            fit_native, score_native = stage8.fold_categories(X_fit, X_score)
                            model.fit(fit_native, y_train[itr])
                            cls = np.asarray(model.classes_)
                            inner_oof[iva] = model.predict_proba(score_native)[:, int(np.flatnonzero(cls == 1)[0])]
                        else:
                            fit_candidate(model, X_fit, y_train[itr], name)
                            cls = np.asarray(model.named_steps["classifier"].classes_)
                            inner_oof[iva] = model.predict_proba(X_score)[:, int(np.flatnonzero(cls == 1)[0])]
                        del model
                        gc.collect()
                    if not np.isfinite(inner_oof).all():
                        raise ValueError(f"Missing inner OOF probabilities: {target}/{name}/fold {outer_fold}.")
                    calibrator = fit_sigmoid(inner_oof, y_train)
                    final_model = make_model(name, binary, target, outer_fold, rf_params, et_params, cb_params)
                    if name == "TabPFN 3.5":
                        fit_native, valid_native = stage8.fold_categories(X_train, X_valid)
                        final_model.fit(fit_native, y_train)
                        cls = np.asarray(final_model.classes_)
                        raw = final_model.predict_proba(valid_native)[:, int(np.flatnonzero(cls == 1)[0])]
                    else:
                        fit_candidate(final_model, X_train, y_train, name)
                        cls = np.asarray(final_model.named_steps["classifier"].classes_)
                        raw = final_model.predict_proba(X_valid)[:, int(np.flatnonzero(cls == 1)[0])]
                    calibrated = calibrator.predict_proba(logit(raw))[:, 1]
                warnings_seen = [f"{w.category.__name__}: {w.message}" for w in caught]
                raw_metrics = calibration_metrics(y_valid, raw)
                cal_metrics = calibration_metrics(y_valid, calibrated)
                fold_rows.append({
                    "target": target, "model": name, "outer_fold": outer_fold,
                    "n_outer_train": len(train_idx), "n_outer_validation": len(valid_idx),
                    "inner_calibration_folds": INNER_FOLDS,
                    "threshold": 0.5,
                    **{f"raw_{k}": v for k, v in raw_metrics.items()},
                    **{f"sigmoid_{k}": v for k, v in cal_metrics.items()},
                    "fit_warnings": " | ".join(warnings_seen),
                })
                for j, pos in enumerate(valid_idx):
                    oof_rows.append({
                        "target": target, "model": name, "ID": idt.iloc[pos],
                        "outer_fold": outer_fold, "true_label": int(y_valid[j]),
                        "raw_probability": float(raw[j]), "sigmoid_probability": float(calibrated[j]),
                    })
                if warnings_seen:
                    warning_rows.append({"target": target, "model": name,
                                         "outer_fold": outer_fold, "warnings": " | ".join(warnings_seen)})
                del final_model, calibrator
                gc.collect()

    folds, oof = pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)
    expected = {
        (target, model, record_id)
        for target, models in CANDIDATES.items()
        for model in models
        for record_id in ids.loc[y_all[target].notna(), id_col]
    }
    observed = set(zip(oof.target, oof.model, oof.ID))
    if observed != expected or len(oof) != len(expected):
        raise ValueError("OOF predictions do not align to unique eligible development IDs.")
    if folds.groupby(["target", "model"]).outer_fold.nunique().ne(OUTER_FOLDS).any():
        raise ValueError("Candidate evaluation is missing outer folds.")
    if not np.isfinite(oof[["raw_probability", "sigmoid_probability"]].to_numpy()).all():
        raise ValueError("OOF probability output contains nonfinite values.")

    # Write only after the full run has completed and passed alignment checks.
    OUT.mkdir()
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    folds.to_csv(OUT / "fold_level_metrics.csv", index=False)
    pd.DataFrame(warning_rows, columns=["target", "model", "outer_fold", "warnings"]).to_csv(
        OUT / "warnings.csv", index=False)
    summary_rows = []
    metrics = ["average_precision", "auroc", "brier_score", "log_loss",
               "ece_10_equal_width_bins", "calibration_intercept_diagnostic",
               "calibration_slope_diagnostic", "accuracy_at_0_5", "precision_at_0_5",
               "recall_at_0_5", "specificity_at_0_5", "f1_at_0_5"]
    for target, models in CANDIDATES.items():
        for name in models:
            part = oof[(oof.target == target) & (oof.model == name)]
            y = part.true_label.to_numpy(dtype=int)
            for stage, column in [("raw", "raw_probability"), ("sigmoid", "sigmoid_probability")]:
                vals = calibration_metrics(y, part[column].to_numpy())
                summary_rows.append({"target": target, "model": name, "probability_stage": stage,
                                     "n_oof": len(part), **vals})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "oof_summary.csv", index=False)
    aggregate_rows = []
    for (target, model), group in folds.groupby(["target", "model"], sort=False):
        row = {"target": target, "model": model, "folds": int(group.outer_fold.nunique())}
        for stage in ["raw", "sigmoid"]:
            for metric in metrics:
                col = f"{stage}_{metric}"
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_sd"] = float(group[col].std(ddof=1))
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(OUT / "fold_aggregate_metrics.csv", index=False)
    aggregate = pd.DataFrame(aggregate_rows)
    result_lines = [
        "# Stage 9 finalist results and calibration diagnostics", "",
        "All values are development-only five-fold estimates. Parentheses show sample SD across outer folds. "
        "The sigmoid calibration is cross-fitted inside each outer training partition. Probabilities are "
        "probabilities clipped to [1e-6, 1-1e-6] for probability metrics and logit diagnostics. AP and AUROC use the same "
        "monotonic sigmoid mapping, so ranking metrics should be unchanged apart from numerical ties.", "",
        "## Fold mean (sample SD)", "",
        "| Target | Provisional candidate | AP raw → sigmoid | AUROC raw → sigmoid | Brier raw → sigmoid | Log Loss raw → sigmoid |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        def show(metric: str) -> str:
            return (f"{getattr(row, f'raw_{metric}_mean'):.3f} ± {getattr(row, f'raw_{metric}_sd'):.3f} → "
                    f"{getattr(row, f'sigmoid_{metric}_mean'):.3f} ± {getattr(row, f'sigmoid_{metric}_sd'):.3f}")
        result_lines.append(
            f"| {row.target} | {row.model} | {show('average_precision')} | {show('auroc')} | "
            f"{show('brier_score')} | {show('log_loss')} |"
        )
    result_lines += ["", "## Pooled OOF calibration diagnostics", "",
        "ECE uses ten equal-width bins. Calibration slope/intercept are descriptive pooled-OOF diagnostics "
        "fit on the same OOF labels and probabilities; they are not independent test estimates.", "",
        "| Target | Candidate | Stage | ECE | Intercept diagnostic | Slope diagnostic |",
        "|---|---|---|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        result_lines.append(
            f"| {row.target} | {row.model} | {row.probability_stage} | "
            f"{row.ece_10_equal_width_bins:.4f} | {row.calibration_intercept_diagnostic:.4f} | "
            f"{row.calibration_slope_diagnostic:.4f} |"
        )
    result_lines += ["", "## Interpretation", "",
        "These results compare only the user-specified provisional candidates. Five folds are descriptive, "
        "not a significance test. The fixed 0.5 threshold is a reference; model, calibration, and threshold "
        "choices remain provisional and are not frozen here. Hyperparameters for tuned candidates were "
        "selected using each full outer-training partition, so inner calibration OOF scores are not fully "
        "independent of those setting-selection labels; the untouched outer validation fold evaluates the "
        "complete procedure. Final fitting should freeze or reselect settings inside its calibration folds.", ""]
    (OUT / "stage9_results.md").write_text("\n".join(result_lines), encoding="utf-8")
    (OUT / "stage9_report.md").write_text(
        "# Stage 9: finalist calibration comparison\n\n"
        "Development-only nested cross-fitting compares the requested provisional finalists. "
        "For each outer fold, Platt scaling is fit on four-fold OOF predictions generated inside "
        "the outer training rows; the base model is refit on the outer training rows and scored "
        "once on the untouched outer validation fold. Preprocessing remains inside each model fit. "
        "Fixed fold-specific tuning settings for tuned finalists come from saved development-only "
        "tuning runs. Those settings were selected using the full outer training partition, so the "
        "inner calibration scores are not fully independent of labels used in setting selection; "
        "the untouched outer validation fold still evaluates the complete procedure. Final fitting "
        "should freeze or reselect hyperparameters inside its calibration folds. No test artifacts are read.\n\n"
        "AP is the primary ranking metric; AUROC, Brier, Log Loss, 10-bin ECE, calibration slope "
        "and intercept diagnostics, and fixed 0.5 threshold metrics are also reported. The slope "
        "and intercept diagnostics in pooled OOF summaries are descriptive fit diagnostics, not "
        "independent test estimates. Five outer folds are descriptive and do not establish a "
        "statistical winner. Final model, calibration, and threshold choices remain provisional "
        "pending review of these results and the separate freeze record.\n\n"
        "TabPFN runs use the already installed approved version. This script does not download or "
        "install packages or model weights. See the dated run log for runtime, package versions, "
        "commands, warnings, and artifact checks.\n",
        encoding="utf-8")
    print(f"Stage 9 complete: {len(oof)} OOF rows across {len(folds)} candidate-fold evaluations.")


if __name__ == "__main__":
    main()
