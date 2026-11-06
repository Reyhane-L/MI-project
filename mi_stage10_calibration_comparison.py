"""Stage 10: compare calibration methods for fixed development finalists.

Only explicit Stage 1 development files and Stage 9 development OOF outputs
are read. The locked partition is not referenced. New outputs are written to
an isolated directory and never overwritten.
"""
from __future__ import annotations

import gc
import json
import os
import argparse
from pathlib import Path
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage10-mplconfig"))
os.environ.setdefault("SKB_DATA_DIRECTORY", str(Path(tempfile.gettempdir()) / "mi-stage10-skrub-data"))
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/private/tmp/tabpfn-stage8-cache")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import mi_stage3_random_forest as stage3
import mi_stage8_tabpfn as stage8
from mi_stage10_calibrators import (
    BetaCalibrator, IdentityCalibrator, IsotonicCalibrator, PlattCalibrator,
    TemperatureCalibrator, clipped_probability, make_calibrator,
)

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE3 = ROOT / "stage3_outputs"
STAGE8 = ROOT / "stage8_tabpfn_outputs"
STAGE9 = ROOT / "stage9_finalist_calibration_outputs"
OUT = ROOT / "stage10_calibration_comparison_outputs"
SEED, OUTER_FOLDS, INNER_FOLDS = 42, 5, 4
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]
SELECTED_MODELS = {
    "DEATH_BINARY": "TabPFN 3.5",
    "ZSN": "TabPFN 3.5",
    "FIBR_PREDS": "Random Forest preliminary",
}
METHODS = ["Raw", "Platt", "Isotonic", "Beta", "Temperature"]
SCORED = ["average_precision", "auroc", "brier_score", "log_loss"]


def quality_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = clipped_probability(probability)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
        "specificity_at_0_5": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def calibration_diagnostics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    """Joint intercept/slope, calibration-in-large, and equal-width ECE."""
    y = np.asarray(y_true, dtype=float)
    p = clipped_probability(probability)
    z = np.log(p / (1 - p))

    def loss_gradient(theta: np.ndarray, design: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ theta
        pred = 1 / (1 + np.exp(-np.clip(eta, -40, 40)))
        loss = np.logaddexp(0, eta).sum() - np.dot(y, eta)
        grad = design.T @ (pred - y)
        return float(loss), grad

    joint = minimize(loss_gradient, np.array([0.0, 1.0]), args=(np.column_stack([np.ones(len(z)), z]),),
                     method="L-BFGS-B", jac=True)
    if not joint.success:
        raise RuntimeError(f"Joint calibration diagnostic failed: {joint.message}")
    # Calibration-in-the-large is an intercept with slope fixed at one:
    # estimate alpha in logit(y) = alpha + offset(logit(p)).
    def cil_loss_gradient(alpha: np.ndarray) -> tuple[float, np.ndarray]:
        eta = z + alpha[0]
        pred = 1 / (1 + np.exp(-np.clip(eta, -40, 40)))
        loss = np.logaddexp(0, eta).sum() - np.dot(y, eta)
        grad = np.array([float(np.sum(pred - y))])
        return float(loss), grad

    cil = minimize(cil_loss_gradient, np.array([0.0]), method="L-BFGS-B", jac=True)
    if not cil.success:
        raise RuntimeError(f"Calibration-in-the-large fit failed: {cil.message}")
    bins = np.linspace(0, 1, 11)
    which = np.minimum(np.digitize(p, bins[1:-1]), 9)
    ece = 0.0
    for b in range(10):
        mask = which == b
        if mask.any():
            ece += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return {
        "calibration_intercept_joint": float(joint.x[0]),
        "calibration_slope_joint": float(joint.x[1]),
        "calibration_in_large_intercept": float(cil.x[0]),
        "ece_10_equal_width": ece,
    }


def recommend_method(fold_metrics: pd.DataFrame, pooled_metrics: pd.DataFrame,
                     target: str) -> dict:
    """Apply the transparent stability preference: alternatives must beat Platt materially and consistently."""
    f = fold_metrics[fold_metrics.target == target]
    p = pooled_metrics[pooled_metrics.target == target].set_index("method")
    if set(p.index) != set(METHODS):
        raise ValueError(f"Pooled calibration methods incomplete for {target}.")
    base = "Platt"
    reasons = []
    raw_wins = {
        metric: int((f[f.method == "Raw"].set_index("outer_fold")[metric] <
                     f[f.method == base].set_index("outer_fold")[metric]).sum())
        for metric in ["brier_score", "log_loss"]
    }
    raw_beats = all(float(p.loc["Raw", metric]) < float(p.loc[base, metric]) for metric in raw_wins)
    if raw_beats and all(wins >= 4 for wins in raw_wins.values()):
        reasons.append("Raw probabilities have lower pooled Brier and Log Loss and beat Platt in at least 4/5 folds on both metrics.")
        return {"target": target, "recommended_method": "Raw", "decision_status": "descriptive recommendation",
                "rationale": " ".join(reasons), "raw_fold_wins_vs_platt_brier": raw_wins["brier_score"],
                "raw_fold_wins_vs_platt_log_loss": raw_wins["log_loss"]}

    qualifying = []
    for method in ["Isotonic", "Beta", "Temperature"]:
        wins = {
            metric: int((f[f.method == method].set_index("outer_fold")[metric] <
                         f[f.method == base].set_index("outer_fold")[metric]).sum())
            for metric in ["brier_score", "log_loss"]
        }
        material = all(float(p.loc[method, metric]) <= 0.99 * float(p.loc[base, metric]) for metric in wins)
        stable = all(value >= 4 for value in wins.values())
        if material and stable:
            qualifying.append((method, wins, float(p.loc[method, "brier_score"]), float(p.loc[method, "log_loss"])))
    if qualifying:
        qualifying.sort(key=lambda x: (x[2], x[3]))
        method, wins, _, _ = qualifying[0]
        reasons.append("The optional method improves pooled Brier and Log Loss by at least 1% and wins at least 4/5 paired folds on both.")
        return {"target": target, "recommended_method": method, "decision_status": "descriptive recommendation",
                "rationale": " ".join(reasons), "raw_fold_wins_vs_platt_brier": raw_wins["brier_score"],
                "raw_fold_wins_vs_platt_log_loss": raw_wins["log_loss"],
                "recommended_fold_wins_vs_platt_brier": wins["brier_score"],
                "recommended_fold_wins_vs_platt_log_loss": wins["log_loss"]}

    platt_wins = {
        metric: int((f[f.method == base].set_index("outer_fold")[metric] <
                     f[f.method == "Raw"].set_index("outer_fold")[metric]).sum())
        for metric in ["brier_score", "log_loss"]
    }
    if all(float(p.loc[base, metric]) < float(p.loc["Raw", metric]) for metric in platt_wins) and all(v >= 3 for v in platt_wins.values()):
        reasons.append("Platt improves pooled Brier and Log Loss versus raw; optional methods did not meet the prespecified stable-improvement screen.")
        status = "Platt recommended; optional methods inconclusive"
    else:
        reasons.append("Differences are mixed or small; retain Platt as the assignment-required default while noting that evidence is inconclusive.")
        status = "inconclusive; Platt default"
    return {"target": target, "recommended_method": "Platt", "decision_status": status,
            "rationale": " ".join(reasons), "raw_fold_wins_vs_platt_brier": raw_wins["brier_score"],
            "raw_fold_wins_vs_platt_log_loss": raw_wins["log_loss"],
            "platt_fold_wins_vs_raw_brier": platt_wins["brier_score"],
            "platt_fold_wins_vs_raw_log_loss": platt_wins["log_loss"]}


def quantile_reliability_rows(target: str, method: str, y: np.ndarray,
                              probability: np.ndarray, n_bins: int = 10) -> list[dict]:
    p = clipped_probability(probability)
    frame = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": p})
    frame["bin"] = pd.qcut(frame.p, q=n_bins, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in frame.groupby("bin", observed=True, sort=True):
        rows.append({"target": target, "method": method, "bin": int(bin_id) + 1,
                     "n": len(group), "mean_predicted_probability": float(group.p.mean()),
                     "observed_positive_rate": float(group.y.mean()),
                     "probability_min": float(group.p.min()), "probability_max": float(group.p.max())})
    return rows


def validate_inputs(X: pd.DataFrame, y_all: pd.DataFrame, ids: pd.DataFrame,
                    dictionary: pd.DataFrame, raw_oof: pd.DataFrame) -> None:
    stage8.validate_inputs(X, y_all, ids, dictionary)
    if ids.shape[1] != 1 or ids.iloc[:, 0].isna().any() or ids.iloc[:, 0].duplicated().any():
        raise ValueError("Development IDs must be a single non-null unique column.")
    required = {"target", "model", "ID", "outer_fold", "true_label", "raw_probability"}
    if required.difference(raw_oof.columns):
        raise ValueError(f"Stage 9 OOF file missing columns: {sorted(required.difference(raw_oof.columns))}")
    if raw_oof.duplicated(["target", "model", "ID"]).any():
        raise ValueError("Stage 9 contains duplicate candidate/target/ID OOF rows.")
    if not set(TARGETS).issubset(y_all.columns):
        raise ValueError("Required targets are missing from development labels.")


def build_model(model_name: str, binary: list[str]):
    if model_name == "TabPFN 3.5":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(random_state=SEED, device="auto", n_preprocessing_jobs=1,
                                memory_saving_mode="auto", show_progress_bar=False)
    if model_name == "Random Forest preliminary":
        return stage3.make_pipeline(binary)
    raise ValueError(f"Unsupported fixed Stage 10 model: {model_name}")


def fit_predict(model, model_name: str, X_fit: pd.DataFrame, y_fit: np.ndarray,
                X_eval: pd.DataFrame) -> np.ndarray:
    if model_name == "TabPFN 3.5":
        fit_native, eval_native = stage8.fold_categories(X_fit, X_eval)
        model.fit(fit_native, y_fit)
        classes = np.asarray(model.classes_)
        return np.asarray(model.predict_proba(eval_native)[:, int(np.flatnonzero(classes == 1)[0])], dtype=float)
    model.fit(X_fit, y_fit)
    classes = np.asarray(model.named_steps["classifier"].classes_)
    return np.asarray(model.predict_proba(X_eval)[:, int(np.flatnonzero(classes == 1)[0])], dtype=float)


def exact_configurations() -> dict:
    return {
        "DEATH_BINARY": {"model": "TabPFN 3.5", "parameters": {
            "random_state": SEED, "device": "auto", "n_preprocessing_jobs": 1,
            "memory_saving_mode": "auto", "show_progress_bar": False},
            "class_weight": None, "resampling": None, "prior_correction": "N/A: no class weighting/resampling; calibration uses naturally prevalent development OOF labels."},
        "ZSN": {"model": "TabPFN 3.5", "parameters": {
            "random_state": SEED, "device": "auto", "n_preprocessing_jobs": 1,
            "memory_saving_mode": "auto", "show_progress_bar": False},
            "class_weight": None, "resampling": None, "prior_correction": "N/A: no class weighting/resampling; calibration uses naturally prevalent development OOF labels."},
        "FIBR_PREDS": {"model": "Random Forest preliminary", "parameters": {
            "n_estimators": 200, "random_state": SEED, "n_jobs": 1,
            "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1,
            "min_samples_split": 2, "class_weight": None},
            "preprocessing": "Fold-fitted median imputation for continuous, most-frequent for ordinal/binary/nominal, one-hot nominal; scaling omitted.",
            "class_weight": None, "resampling": None, "prior_correction": "N/A: no class weighting/resampling; calibration uses naturally prevalent development OOF labels."},
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def repair_existing_outputs() -> None:
    """Recompute diagnostics from saved predictions and stabilize bundle pickles; no model fits."""
    if not OUT.is_dir():
        raise FileNotFoundError(f"Cannot repair missing Stage 10 outputs: {OUT}")
    required = ["outer_oof_predictions.csv", "outer_fold_metrics.csv", "pooled_oof_metrics.csv",
                "fold_metric_aggregates.csv", "method_recommendations.csv"]
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 10 repair inputs are missing: {missing}")
    oof = pd.read_csv(OUT / "outer_oof_predictions.csv")
    folds = pd.read_csv(OUT / "outer_fold_metrics.csv")
    pooled = pd.read_csv(OUT / "pooled_oof_metrics.csv")
    diag_cols = ["calibration_intercept_joint", "calibration_slope_joint",
                 "calibration_in_large_intercept", "ece_10_equal_width"]
    for index, row in folds.iterrows():
        part = oof[(oof.target == row.target) & (oof.calibration_method == row.method) &
                   (oof.outer_fold == row.outer_fold)]
        if len(part) != int(row.n_validation):
            raise ValueError(f"Saved OOF/fold row mismatch in diagnostic repair: {row.target}/{row.method}/{row.outer_fold}")
        diagnostic = calibration_diagnostics(part.true_label.to_numpy(), part.calibrated_probability.to_numpy())
        for column in diag_cols:
            folds.loc[index, column] = diagnostic[column]
    for index, row in pooled.iterrows():
        part = oof[(oof.target == row.target) & (oof.calibration_method == row.method)]
        if len(part) != int(row.n_oof):
            raise ValueError(f"Saved OOF/pooled row mismatch in diagnostic repair: {row.target}/{row.method}")
        diagnostic = calibration_diagnostics(part.true_label.to_numpy(), part.calibrated_probability.to_numpy())
        for column in diag_cols:
            pooled.loc[index, column] = diagnostic[column]

    aggregate_rows = []
    aggregate_metrics = [*SCORED, "ece_10_equal_width", "calibration_intercept_joint",
                         "calibration_slope_joint", "calibration_in_large_intercept"]
    for (target, model_name, method), group in folds.groupby(["target", "model", "method"], sort=False):
        result = {"target": target, "model": model_name, "method": method, "folds": int(group.outer_fold.nunique())}
        for metric in aggregate_metrics:
            result[f"{metric}_mean"] = float(group[metric].mean())
            result[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(result)
    fold_aggregate = pd.DataFrame(aggregate_rows)
    recommendations = pd.DataFrame([recommend_method(folds, pooled, target) for target in TARGETS])

    reliability_rows = []
    for target in TARGETS:
        for method, part in oof[oof.target == target].groupby("calibration_method", sort=False):
            reliability_rows.extend(quantile_reliability_rows(
                target, method, part.true_label.to_numpy(), part.calibrated_probability.to_numpy()))
    reliability = pd.DataFrame(reliability_rows)

    # Only diagnostic/result summaries are refreshed; the saved OOF probabilities stay byte-for-byte untouched.
    folds.to_csv(OUT / "outer_fold_metrics.csv", index=False)
    pooled.to_csv(OUT / "pooled_oof_metrics.csv", index=False)
    fold_aggregate.to_csv(OUT / "fold_metric_aggregates.csv", index=False)
    recommendations.to_csv(OUT / "method_recommendations.csv", index=False)
    reliability.to_csv(OUT / "quantile_reliability_bins.csv", index=False)
    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Ideal")
        for method, part in reliability[reliability.target == target].groupby("method", sort=False):
            ax.plot(part.mean_predicted_probability, part.observed_positive_rate, marker="o", label=method)
        ax.set(xlabel="Mean predicted probability (quantile bin)", ylabel="Observed positive rate",
               title=f"{target}: development OOF calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(OUT / f"{target.lower()}_reliability.png", dpi=180); plt.close(fig)

    # Old Stage 10 joblibs referenced classes through __main__. Bind those names
    # temporarily, load the already fitted estimators, and reserialize them with
    # stable import paths from mi_stage10_calibrators. No model is refitted.
    import __main__
    from mi_stage10_calibrators import (
        BetaCalibrator, IdentityCalibrator, IsotonicCalibrator, PlattCalibrator,
        TemperatureCalibrator,
    )
    for cls in [BetaCalibrator, IdentityCalibrator, IsotonicCalibrator,
                PlattCalibrator, TemperatureCalibrator]:
        setattr(__main__, cls.__name__, cls)
    for target in TARGETS:
        path = OUT / f"{target.lower()}_full_development_bundle.joblib"
        bundle = joblib.load(path)
        if bundle.get("target") != target or bundle.get("calibration_method") != recommendations.loc[
            recommendations.target == target, "recommended_method"].iloc[0]:
            raise ValueError(f"Full-development bundle metadata mismatch for {target}.")
        temporary = path.with_name(path.name + ".stable.tmp")
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, path)
        del bundle
        gc.collect()

    configs = json.loads((OUT / "model_configurations.json").read_text(encoding="utf-8"))
    for target in TARGETS:
        configs[target]["recommended_calibration_method"] = recommendations.loc[
            recommendations.target == target, "recommended_method"].iloc[0]
    (OUT / "model_configurations.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

    report = (OUT / "stage10_report.md").read_text(encoding="utf-8")
    report += ("\n\n## Diagnostic correction record\n\n"
               "Calibration-in-the-large is the fitted intercept in a logistic model with slope fixed at 1 and "
               "`logit(raw probability)` included as an offset. A post-run audit found the first computation omitted "
               "that offset; diagnostic columns, method recommendations, quantile bins, reliability plots, and this "
               "report were recomputed from the unchanged saved OOF probabilities. The full-development model bundles "
               "were reserialized with stable import paths; their fitted models and calibration parameters were not refit.\n")
    (OUT / "stage10_report.md").write_text(report, encoding="utf-8")
    print("Stage 10 diagnostic repair complete; no estimator or calibrator was refit.")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 10 outputs: {OUT}")
    paths = [STAGE1 / "X_train.csv", STAGE1 / "y_train.csv", STAGE1 / "train_record_ids.csv",
             STAGE1 / "column_dictionary.csv", STAGE9 / "oof_predictions.csv"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing explicit development inputs: {missing}")
    X, y_all, ids, dictionary = [pd.read_csv(p) for p in paths[:4]]
    raw_all = pd.read_csv(paths[4])
    validate_inputs(X, y_all, ids, dictionary, raw_all)
    id_col = ids.columns[0]
    typed = stage3.CONTINUOUS + stage3.ORDINAL + stage3.NOMINAL
    binary = [column for column in X if column not in typed]

    fold_rows, oof_rows, warning_rows = [], [], []
    inner_rows = []
    full_oof_by_target = {}
    for target in TARGETS:
        model_name = SELECTED_MODELS[target]
        y = y_all[target].dropna().astype("int8")
        Xt, ids_t = X.loc[y.index].copy(), ids.loc[y.index, id_col]
        stage9 = raw_all[(raw_all.target == target) & (raw_all.model == model_name)].copy()
        if len(stage9) != len(y) or set(stage9.ID) != set(ids_t):
            raise ValueError(f"Stage 9 selected raw OOF predictions do not align for {target}.")
        stage9 = stage9.set_index("ID").loc[ids_t].reset_index()
        outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        splits = list(outer.split(Xt, y))
        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            y_train, y_valid = y.iloc[train_idx].to_numpy(), y.iloc[valid_idx].to_numpy()
            X_train, X_valid = Xt.iloc[train_idx], Xt.iloc[valid_idx]
            valid_ids = ids_t.iloc[valid_idx].to_numpy()
            saved_fold = stage9.set_index("ID").loc[valid_ids]
            if set(saved_fold.outer_fold.astype(int)) != {fold}:
                raise ValueError(f"Stage 9 outer-fold membership mismatch: {target}/fold {fold}.")
            raw_outer = saved_fold.raw_probability.to_numpy(dtype=float)
            if not np.isfinite(raw_outer).all() or ((raw_outer < 0) | (raw_outer > 1)).any():
                raise ValueError(f"Invalid Stage 9 raw probabilities: {target}/fold {fold}.")

            inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold)
            inner_oof = np.full(len(y_train), np.nan)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for inner_fold, (itr, iva) in enumerate(inner.split(X_train, y_train), 1):
                    model = build_model(model_name, binary)
                    pred = fit_predict(model, model_name, X_train.iloc[itr], y_train[itr], X_train.iloc[iva])
                    inner_oof[iva] = pred
                    inner_rows.append({"target": target, "model": model_name, "outer_fold": fold,
                                       "inner_fold": inner_fold, "n_inner_train": len(itr),
                                       "n_inner_validation": len(iva),
                                       "inner_training_prevalence": float(y_train[itr].mean())})
                    del model
                    gc.collect()
                if not np.isfinite(inner_oof).all():
                    raise ValueError(f"Inner calibration scores incomplete for {target}/fold {fold}.")
                calibrated: dict[str, np.ndarray] = {"Raw": raw_outer}
                for method in METHODS[1:]:
                    calibrator = make_calibrator(method).fit(inner_oof, y_train)
                    calibrated[method] = calibrator.predict(raw_outer)
                labels = y_valid
                for method, probs in calibrated.items():
                    score = quality_metrics(labels, probs)
                    diagnostic = calibration_diagnostics(labels, probs)
                    fold_rows.append({"target": target, "model": model_name,
                                      "outer_fold": fold, "method": method,
                                      "n_validation": len(valid_idx), **score, **diagnostic})
                    for j, record_id in enumerate(valid_ids):
                        oof_rows.append({"target": target, "model": model_name,
                                         "calibration_method": method, "ID": record_id,
                                         "outer_fold": fold, "true_label": int(labels[j]),
                                         "raw_probability": float(raw_outer[j]),
                                         "calibrated_probability": float(probs[j])})
                for warning in caught:
                    warning_rows.append({"target": target, "model": model_name,
                                         "outer_fold": fold, "warning_category": warning.category.__name__,
                                         "warning_message": str(warning.message)})
            del inner_oof, calibrated
            gc.collect()
        full_oof_by_target[target] = stage9

    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    if folds.duplicated(["target", "model", "outer_fold", "method"]).any():
        raise ValueError("Duplicate outer-fold metrics.")
    if folds.groupby(["target", "method"]).outer_fold.nunique().ne(OUTER_FOLDS).any():
        raise ValueError("At least one target/method is missing an outer fold.")
    eligible_rows = sum(int(y_all[t].notna().sum()) * len(METHODS) for t in TARGETS)
    if len(oof) != eligible_rows or oof.duplicated(["target", "calibration_method", "ID"]).any():
        raise ValueError("Calibrated OOF rows are incomplete or duplicated.")
    if not np.isfinite(oof[["raw_probability", "calibrated_probability"]].to_numpy()).all():
        raise ValueError("Calibrated OOF data contain non-finite probabilities.")

    pooled_rows = []
    for target in TARGETS:
        for method, group in oof[oof.target == target].groupby("calibration_method", sort=False):
            y = group.true_label.to_numpy(dtype=int)
            p = group.calibrated_probability.to_numpy(dtype=float)
            pooled_rows.append({"target": target, "model": SELECTED_MODELS[target], "method": method,
                                "n_oof": len(group), **quality_metrics(y, p),
                                **calibration_diagnostics(y, p)})
    pooled = pd.DataFrame(pooled_rows)
    fold_aggregate_rows = []
    for (target, model, method), group in folds.groupby(["target", "model", "method"], sort=False):
        row = {"target": target, "model": model, "method": method, "folds": len(group)}
        for metric in [*SCORED, "ece_10_equal_width", "calibration_intercept_joint",
                       "calibration_slope_joint", "calibration_in_large_intercept"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        fold_aggregate_rows.append(row)
    fold_aggregate = pd.DataFrame(fold_aggregate_rows)

    paired_rows = []
    fold_idx = folds.set_index(["target", "method", "outer_fold"])
    for target in TARGETS:
        for method in METHODS[1:]:
            for comparator in ["Raw", "Platt"]:
                if method == comparator:
                    continue
                for metric in SCORED:
                    win_count = 0
                    for fold in range(1, OUTER_FOLDS + 1):
                        candidate = float(fold_idx.loc[(target, method, fold), metric])
                        baseline = float(fold_idx.loc[(target, comparator, fold), metric])
                        delta = candidate - baseline
                        better = delta < -1e-12 if metric in {"brier_score", "log_loss"} else delta > 1e-12
                        win_count += int(better)
                        paired_rows.append({"target": target, "candidate_method": method,
                                            "comparator_method": comparator, "outer_fold": fold,
                                            "metric": metric, "candidate_value": candidate,
                                            "comparator_value": baseline, "candidate_minus_comparator": delta,
                                            "candidate_better": bool(better)})
                    for row in paired_rows[-OUTER_FOLDS:]:
                        row["candidate_fold_wins_out_of_5"] = win_count
    paired = pd.DataFrame(paired_rows)

    recommendations = pd.DataFrame([recommend_method(folds, pooled, target) for target in TARGETS])
    reliability_rows = []
    for target in TARGETS:
        for method, group in oof[oof.target == target].groupby("calibration_method", sort=False):
            reliability_rows.extend(quantile_reliability_rows(
                target, method, group.true_label.to_numpy(), group.calibrated_probability.to_numpy()))
    reliability = pd.DataFrame(reliability_rows)

    # Save only after all cross-validation, completeness, and alignment checks succeed.
    OUT.mkdir()
    oof.to_csv(OUT / "outer_oof_predictions.csv", index=False)
    folds.to_csv(OUT / "outer_fold_metrics.csv", index=False)
    fold_aggregate.to_csv(OUT / "fold_metric_aggregates.csv", index=False)
    pooled.to_csv(OUT / "pooled_oof_metrics.csv", index=False)
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)
    reliability.to_csv(OUT / "quantile_reliability_bins.csv", index=False)
    recommendations.to_csv(OUT / "method_recommendations.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(OUT / "inner_fit_manifest.csv", index=False)
    pd.DataFrame(warning_rows, columns=["target", "model", "outer_fold", "warning_category", "warning_message"]).to_csv(
        OUT / "warnings.csv", index=False)

    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Ideal")
        for method, part in reliability[reliability.target == target].groupby("method", sort=False):
            ax.plot(part.mean_predicted_probability, part.observed_positive_rate, marker="o", label=method)
        ax.set(xlabel="Mean predicted probability (quantile bin)", ylabel="Observed positive rate",
               title=f"{target}: development OOF calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(OUT / f"{target.lower()}_reliability.png", dpi=180); plt.close(fig)

    configs = exact_configurations()
    from importlib.metadata import version
    feature_types = stage3.CONTINUOUS + stage3.ORDINAL + stage3.NOMINAL
    binary_features = [column for column in X.columns if column not in feature_types]
    for target, model_name in SELECTED_MODELS.items():
        model = build_model(model_name, binary)
        estimator = model if model_name == "TabPFN 3.5" else model.named_steps["classifier"]
        configs[target]["package_versions"] = {
            "scikit_learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "model_package": version("tabpfn" if model_name == "TabPFN 3.5" else "scikit-learn")}
        configs[target]["effective_estimator_parameters"] = json_safe(estimator.get_params(deep=False))
        if model_name == "TabPFN 3.5":
            configs[target]["categorical_features"] = list(stage8.CATEGORICAL)
            configs[target]["missing_value_handling"] = "Native TabPFN handling; category vocabularies inferred from each fit partition."
        else:
            configs[target]["preprocessing"] = {
                "continuous_median_imputed": list(stage3.CONTINUOUS),
                "ordinal_mode_imputed_numeric": list(stage3.ORDINAL),
                "binary_mode_imputed_numeric": binary_features,
                "nominal_mode_imputed_onehot": list(stage3.NOMINAL),
                "scaling": None,
            }
        del model
    for target in TARGETS:
        recommendation = recommendations.loc[recommendations.target == target, "recommended_method"].iloc[0]
        model_name = SELECTED_MODELS[target]
        y = y_all[target].dropna().astype("int8")
        Xt = X.loc[y.index].copy()
        stage9 = full_oof_by_target[target].set_index("ID").loc[ids.loc[y.index, id_col]]
        p_full_oof = stage9.raw_probability.to_numpy(dtype=float)
        calibrator = make_calibrator(recommendation).fit(p_full_oof, y.to_numpy())
        final_model = build_model(model_name, binary)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model_name == "TabPFN 3.5":
                fit_native, _ = stage8.fold_categories(Xt, Xt)
                final_model.fit(fit_native, y.to_numpy())
            else:
                final_model.fit(Xt, y.to_numpy())
        if caught:
            for warning in caught:
                warning_rows.append({"target": target, "model": model_name, "outer_fold": "full_development",
                                     "warning_category": warning.category.__name__,
                                     "warning_message": str(warning.message)})
        bundle = {"target": target, "model_name": model_name, "model": final_model,
                  "calibration_method": recommendation, "calibrator": calibrator,
                  "model_config": configs[target], "seed": SEED,
                  "calibrator_training_source": "Stage 9 selected-model development OOF probabilities and labels"}
        joblib.dump(bundle, OUT / f"{target.lower()}_full_development_bundle.joblib", compress=3)
        del final_model, calibrator
        gc.collect()
    # Rewrite warning output after full-development fits, preserving earlier warnings.
    pd.DataFrame(warning_rows, columns=["target", "model", "outer_fold", "warning_category", "warning_message"]).to_csv(
        OUT / "warnings.csv", index=False)
    (OUT / "model_configurations.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

    lines = ["# Stage 10: calibration method comparison", "",
             "The user-specified base models are fixed: TabPFN 3.5 for DEATH_BINARY and ZSN; Random Forest preliminary for FIBR_PREDS. "
             "Stage 9 raw outer OOF probabilities and the same target-specific five-fold assignments are used for evaluation. "
             "Only the four inner, training-side model fits per outer fold are regenerated to fit Platt, isotonic, beta, and temperature calibrators. "
             "The outer validation rows are never used to fit a calibrator. No held-out partition was opened.", "",
             "## Method recommendations", "",
             "| Target | Recommended method | Status | Rationale |", "|---|---|---|---|"]
    for row in recommendations.itertuples(index=False):
        lines.append(f"| {row.target} | {row.recommended_method} | {row.decision_status} | {row.rationale} |")
    lines += ["", "The recommendation screen assesses Brier and Log Loss first, checks fold consistency, and uses ECE/reliability and calibration diagnostics as context. "
              "An optional method is recommended only if it improves pooled Brier and Log Loss by at least 1% and wins at least four of five paired folds on both metrics. "
              "If differences are small or mixed, Platt is preferred; raw is recommended only when consistently better on both probability scores. "
              "These are descriptive rules, not significance tests. See `pooled_oof_metrics.csv`, `outer_fold_metrics.csv`, `paired_comparisons.csv`, "
              "`quantile_reliability_bins.csv`, and reliability plots for full evidence.", "",
              "## Model configuration and prior correction", "",
              "Exact configurations are recorded in `model_configurations.json`. All selected base models use natural development prevalence; none uses class weights or resampling. "
              "Prior correction is therefore N/A. Full-development bundles contain the selected estimator and calibrator fitted using development data only. "
              "They are preparation artifacts; held-out evaluation remains a separate approval-gated task.", "",
              "The pooled ECE uses ten equal-width bins. Reliability plots and `quantile_reliability_bins.csv` use ten quantile bins (fewer if ties collapse bins). "
              "Joint intercept/slope and calibration-in-large diagnostics are reported for fold and pooled estimates. Threshold metrics use a fixed 0.5 reference only; "
              "threshold selection is not part of this stage.", ""]
    (OUT / "stage10_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Sta age 10 complete: {len(oof)} outer OOF rows; {len(folds)} fold-method rows; {len(warning_rows)} warnings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-existing", action="store_true",
                        help="Recompute calibration diagnostics and reserialize bundles from existing Stage 10 outputs; performs no model fits.")
    args = parser.parse_args()
    if args.repair_existing:
        repair_existing_outputs()
    else:
        main()
