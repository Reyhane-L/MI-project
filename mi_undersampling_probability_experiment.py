from __future__ import annotations

import json
import hashlib
import importlib.metadata
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from mi_leakage_experiment import load_stage2_api
from mi_stage10_calibrators import PlattCalibrator


ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
X_PATH = STAGE1 / "X_train.csv"
Y_PATH = STAGE1 / "y_train.csv"
OUT = ROOT / "undersampling_calibration_outputs"
TARGET = "DEATH_BINARY"
SEED = 42
FOLDS = 5
METHODS = ("raw_undersampled", "prior_corrected", "platt")
OUTPUTS = (
    "oof_predictions.csv", "fold_metrics.csv", "fold_audit.csv",
    "summary_metrics.csv", "reliability_bins.csv", "reliability_comparison.png",
    "experiment_report.md", "run_manifest.json",
)


def calibration_split_positions(y_train: np.ndarray, seed: int,
                                calibration_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Split outer training labels into model-fit and natural calibration rows."""
    labels = np.asarray(y_train, dtype=np.int8)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration splitting requires both binary classes.")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between zero and one.")
    if np.bincount(labels).min() < 2:
        raise ValueError("Each class needs at least two rows to split calibration data.")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=calibration_fraction,
                                     random_state=seed)
    model_fit, calibration = next(splitter.split(np.zeros(len(labels)), labels))
    if np.intersect1d(model_fit, calibration).size or len(model_fit) + len(calibration) != len(labels):
        raise RuntimeError("Model-fit and calibration positions must be disjoint and exhaustive.")
    if set(np.unique(labels[calibration])) != {0, 1}:
        raise ValueError("Calibration subset must contain both classes.")
    return model_fit, calibration


def prior_correct(probability, natural_prevalence: float,
                  sampled_prevalence: float = 0.5) -> np.ndarray:
    """Apply Section 11.3 odds-based prior-probability correction."""
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be a finite vector in [0, 1].")
    if not 0 < natural_prevalence < 1 or not 0 < sampled_prevalence < 1:
        raise ValueError("Both prevalences must be strictly between zero and one.")
    beta = (natural_prevalence / (1 - natural_prevalence)) / (
        sampled_prevalence / (1 - sampled_prevalence)
    )
    denominator = beta * p - p + 1
    result = beta * p / denominator
    result = np.where(p == 0, 0.0, np.where(p == 1, 1.0, result))
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Prior correction produced invalid probabilities.")
    return result


def undersample_indices(y: np.ndarray, seed: int) -> np.ndarray:
    """Select a deterministic 1:1 subset from a training fold only."""
    labels = np.asarray(y)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("undersampling requires both binary classes.")
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    n_each = min(len(class0), len(class1))
    if n_each < 1:
        raise ValueError("undersampling requires at least one row per class.")
    rng = np.random.RandomState(seed)
    chosen = np.concatenate((rng.choice(class0, n_each, replace=False),
                             rng.choice(class1, n_each, replace=False)))
    return np.sort(chosen)


def _calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit, np.asarray(y, dtype=np.int8))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _ece_and_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    bins = np.minimum(np.searchsorted(edges[1:-1], p, side="right"), len(edges) - 2)
    records = []
    ece = 0.0
    for b in range(len(edges) - 1):
        take = bins == b
        if not take.any():
            continue
        avg_p = float(np.mean(p[take]))
        observed = float(np.mean(y[take]))
        ece += int(take.sum()) / len(y) * abs(observed - avg_p)
        records.append({
            "bin": b + 1, "n": int(take.sum()),
            "mean_predicted_probability": avg_p,
            "observed_prevalence": observed,
        })
    return float(ece), records


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y, dtype=np.int8)
    prob = np.asarray(p, dtype=float)
    if len(labels) != len(prob) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("aligned binary labels and probabilities are required.")
    ece, _ = _ece_and_bins(labels, prob)
    intercept, slope = _calibration_slope_intercept(labels, prob)
    return {
        "n": int(len(labels)),
        "observed_prevalence": float(labels.mean()),
        "mean_predicted_probability": float(prob.mean()),
        "average_precision": float(average_precision_score(labels, prob)),
        "auroc": float(roc_auc_score(labels, prob)),
        "brier_score": float(brier_score_loss(labels, prob)),
        "log_loss": float(log_loss(labels, np.clip(prob, 1e-15, 1 - 1e-15))),
        "ece_quantile_10": ece,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _position_digest(positions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(positions, dtype=np.int64).tobytes()).hexdigest()


def run_experiment(X: pd.DataFrame, y_all: pd.DataFrame) -> dict[str, Any]:
    if not X.index.equals(y_all.index) or len(X) != len(y_all):
        raise ValueError("Development predictors and labels must be row-aligned.")
    api = load_stage2_api()
    if TARGET not in y_all or any(c in X for c in ("ID", "LET_IS", *api.targets)):
        raise ValueError("Mortality label must exist and IDs/outcomes must not be predictors.")
    y_series = y_all[TARGET]
    observed = y_series.dropna()
    if set(observed.unique()) != {0, 1} or int(observed.value_counts().min()) < FOLDS:
        raise ValueError("Observed mortality labels must be binary with >=5 rows per class.")
    X_obs = X.loc[observed.index]
    y = observed.to_numpy(dtype=np.int8)
    binary_features = [c for c in X.columns if c not in (
        api.continuous_features + api.ordinal_features + api.nominal_features
    )]
    splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X_obs, y))
    oof = np.full(len(y), np.nan)
    corrected = np.full(len(y), np.nan)
    platt = np.full(len(y), np.nan)
    fold_assign = np.zeros(len(y), dtype=np.int8)
    fold_rows = []
    audit_rows = []
    warning_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        if np.intersect1d(train_idx, valid_idx).size:
            raise RuntimeError("Fold train/validation positions overlap.")
        y_train = y[train_idx]
        train_prevalence = float(y_train.mean())
        model_fit_local, calibration_local = calibration_split_positions(
            y_train, SEED + fold,
        )
        y_model_fit = y_train[model_fit_local]
        sampled_local = undersample_indices(y_model_fit, SEED + fold)
        sample_y = y_model_fit[sampled_local]
        if float(sample_y.mean()) != 0.5:
            raise RuntimeError("Training subset is not balanced 1:1.")
        model = api.build_pipeline(X.columns.tolist(), binary_features)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The classifier and all learned preprocessing see only the sampled
            # model-fit subset. The calibration subset remains untouched.
            model.fit(X_obs.iloc[train_idx[model_fit_local]].iloc[sampled_local], sample_y)
            raw_calibration = np.asarray(
                model.predict_proba(X_obs.iloc[train_idx[calibration_local]])[:, 1], dtype=float,
            )
            calibrator = PlattCalibrator()
            calibrator.fit(raw_calibration, y_train[calibration_local])
            raw = np.asarray(model.predict_proba(X_obs.iloc[valid_idx])[:, 1], dtype=float)
            platt_probability = np.asarray(calibrator.predict(raw), dtype=float)
        for warning in caught:
            warning_rows.append({
                "fold": fold, "category": warning.category.__name__,
                "message": str(warning.message),
            })
        fixed = prior_correct(raw, train_prevalence, float(sample_y.mean()))
        for method, probability in (("raw", raw), ("prior_corrected", fixed),
                                    ("platt", platt_probability),
                                    ("calibration_raw", raw_calibration)):
            if (not np.isfinite(probability).all()
                    or ((probability < 0) | (probability > 1)).any()):
                raise ValueError(f"Fold {fold} returned invalid {method} probabilities.")
        oof[valid_idx] = raw
        corrected[valid_idx] = fixed
        platt[valid_idx] = platt_probability
        fold_assign[valid_idx] = fold
        for method, prob in ((METHODS[0], raw), (METHODS[1], fixed), (METHODS[2], platt_probability)):
            metrics = probability_metrics(y[valid_idx], prob)
            fold_rows.append({
                "fold": fold, "method": method,
                "n_outer_train": len(train_idx), "n_model_fit_natural": len(model_fit_local),
                "n_train_resampled": len(sampled_local),
                "n_calibration": len(calibration_local), "n_validation": len(valid_idx),
                "natural_training_prevalence": train_prevalence,
                "sampled_training_prevalence": float(sample_y.mean()),
                "calibration_prevalence": float(y_train[calibration_local].mean()),
                **metrics,
            })
        model_fit_global = train_idx[model_fit_local]
        sampled_global = train_idx[model_fit_local[sampled_local]]
        calibration_global = train_idx[calibration_local]
        audit_rows.append({
            "fold": fold,
            "n_outer_train": len(train_idx), "outer_train_prevalence": train_prevalence,
            "n_model_fit_natural": len(model_fit_local),
            "model_fit_prevalence": float(y[model_fit_global].mean()),
            "n_train_resampled": len(sampled_global),
            "sampled_prevalence": float(y[sampled_global].mean()),
            "n_calibration": len(calibration_global),
            "calibration_positive_count": int(y[calibration_global].sum()),
            "calibration_prevalence": float(y[calibration_global].mean()),
            "n_validation": len(valid_idx),
            "validation_prevalence": float(y[valid_idx].mean()),
            "model_fit_calibration_disjoint": not bool(np.intersect1d(model_fit_global, calibration_global).size),
            "calibration_validation_disjoint": not bool(np.intersect1d(calibration_global, valid_idx).size),
            "model_fit_validation_disjoint": not bool(np.intersect1d(model_fit_global, valid_idx).size),
            "resampled_calibration_disjoint": not bool(np.intersect1d(sampled_global, calibration_global).size),
            "outer_train_positions": json.dumps(observed.index.to_numpy()[train_idx].tolist()),
            "model_fit_positions": json.dumps(observed.index.to_numpy()[model_fit_global].tolist()),
            "sampled_positions": json.dumps(observed.index.to_numpy()[sampled_global].tolist()),
            "calibration_positions": json.dumps(observed.index.to_numpy()[calibration_global].tolist()),
            "validation_positions": json.dumps(observed.index.to_numpy()[valid_idx].tolist()),
            "outer_train_position_sha256": _position_digest(observed.index.to_numpy()[train_idx]),
            "model_fit_position_sha256": _position_digest(observed.index.to_numpy()[model_fit_global]),
            "calibration_position_sha256": _position_digest(observed.index.to_numpy()[calibration_global]),
            "validation_position_sha256": _position_digest(observed.index.to_numpy()[valid_idx]),
        })
        del model
    if (np.isnan(oof).any() or np.isnan(corrected).any() or np.isnan(platt).any()
            or (fold_assign == 0).any()):
        raise RuntimeError("Each observed development row must receive one OOF prediction.")

    predictions = pd.DataFrame({
        "row_position": observed.index.to_numpy(), "true_label": y,
        "outer_fold": fold_assign, "raw_undersampled_probability": oof,
        "prior_corrected_probability": corrected,
        "platt_probability": platt,
    })
    summary_rows, reliability_rows = [], []
    for method, probability in ((METHODS[0], oof), (METHODS[1], corrected), (METHODS[2], platt)):
        metrics = probability_metrics(y, probability)
        summary_rows.append({"method": method, **metrics})
        _, bins = _ece_and_bins(y, probability)
        for item in bins:
            reliability_rows.append({"method": method, **item})
    manifest = {
        "experiment": "Development-only 1:1 mortality undersampling with prior and Platt calibration comparisons",
        "inputs": [str(X_PATH.relative_to(ROOT)), str(Y_PATH.relative_to(ROOT))],
        "test_partition_accessed": False,
        "target": TARGET,
        "estimator": "Stage 2 regularized LogisticRegression (C=1.0, lbfgs, max_iter=2000)",
        "preprocessing": "Stage 2 fold-local imputation, scaling, and one-hot pipeline fit only on the sampled model-fit subset",
        "resampling": "Uniform random undersampling to 1:1 within each outer fold model-fit subset; seed=42+fold",
        "platt_calibration": {
            "implementation": "mi_stage10_calibrators.PlattCalibrator",
            "source": "stratified 20% calibration subset of each outer training fold, kept at natural prevalence",
            "fitting_sequence": "fit base model on disjoint undersampled model-fit subset; score calibration subset; fit Platt on those scores and natural labels; score outer validation once",
            "seed": "42+outer_fold",
        },
        "correction": {
            "section": "AI-HW2-Guide.pdf Section 11.3, prior-probability correction",
            "formula": "p = beta * ps / (beta * ps - ps + 1)",
            "beta": "odds(natural training-fold prevalence) / odds(1:1 sampled training prevalence)",
            "prevalence_source": "training fold only; validation labels are not used in correction",
        },
        "cv": {"strategy": "StratifiedKFold", "folds": FOLDS, "shuffle": True, "seed": SEED,
               "calibration_split": "StratifiedShuffleSplit, 20% of outer training rows, seed=42+fold"},
        "command": "PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp .venv/bin/python -B mi_undersampling_probability_experiment.py",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "logistic_regression": {
            "l1_ratio": 0.0, "C": 1.0, "solver": "lbfgs", "max_iter": 2000,
            "class_weight": None, "random_state": SEED,
        },
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
        "warnings": warning_rows,
        "output_files": list(OUTPUTS),
    }
    return {
        "predictions": predictions,
        "fold_metrics": pd.DataFrame(fold_rows),
        "fold_audit": pd.DataFrame(audit_rows),
        "summary_metrics": pd.DataFrame(summary_rows),
        "reliability": pd.DataFrame(reliability_rows),
        "manifest": manifest,
    }


def write_outputs(result: dict[str, Any], out: Path = OUT) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    result["predictions"].to_csv(out / OUTPUTS[0], index=False, float_format="%.12g")
    result["fold_metrics"].to_csv(out / OUTPUTS[1], index=False, float_format="%.12g")
    result["fold_audit"].to_csv(out / OUTPUTS[2], index=False, float_format="%.12g")
    result["summary_metrics"].to_csv(out / OUTPUTS[3], index=False, float_format="%.12g")
    result["reliability"].to_csv(out / OUTPUTS[4], index=False, float_format="%.12g")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="black", label="Ideal")
    for method, group in result["reliability"].groupby("method", sort=False):
        ax.plot(group.mean_predicted_probability, group.observed_prevalence,
                marker="o", label=method.replace("_", " "))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed death prevalence", title="1:1 undersampling: OOF reliability")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / OUTPUTS[5], dpi=180)
    plt.close(fig)
    summary = result["summary_metrics"].set_index("method")
    raw, fixed = summary.loc[METHODS[0]], summary.loc[METHODS[1]]
    platt = summary.loc[METHODS[2]]
    report = [
        "# Development-only 1:1 undersampling probability experiment", "",
        "A Stage 2 regularized logistic model was fit in five stratified seed-42 folds. Each outer training fold was split into a 20% stratified natural-prevalence calibration subset and a disjoint model-fit subset. The model-fit subset alone was randomly undersampled to 1:1, and its preprocessing/model pipeline was fit only there. The untouched natural-prevalence calibration subset supplied out-of-fit base scores and labels to Platt; outer validation labels did not participate in any fitting.", "",
        "The Section 11.3 prior correction is `p = beta * ps / (beta * ps - ps + 1)`, with `beta = odds(natural outer-training prevalence) / odds(sampled model-fit prevalence)`. Platt calibration uses `mi_stage10_calibrators.PlattCalibrator` fit on the separate calibration subset.", "",
        f"Across pooled development OOF rows, observed death prevalence was **{raw.observed_prevalence:.4f}**. Mean predicted probability was **{raw.mean_predicted_probability:.4f} raw**, **{fixed.mean_predicted_probability:.4f} prior-corrected**, and **{platt.mean_predicted_probability:.4f} Platt-calibrated**.", "",
        "| Method | Mean probability | Brier | Log loss | ECE (10 quantile bins) | Calibration intercept | Calibration slope | AP | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary.iterrows():
        report.append(
            f"| {method} | {row.mean_predicted_probability:.4f} | {row.brier_score:.4f} | {row.log_loss:.4f} | {row.ece_quantile_10:.4f} | {row.calibration_intercept:.4f} | {row.calibration_slope:.4f} | {row.average_precision:.4f} | {row.auroc:.4f} |"
        )
    report += ["", "All three estimates are evaluated on the same natural-prevalence OOF records. Raw, prior-corrected, and Platt probabilities are retained as separate comparisons. Results describe this model and development sample; the locked test partition was not used.", ""]
    (out / OUTPUTS[6]).write_text("\n".join(report), encoding="utf-8")
    (out / OUTPUTS[7]).write_text(json.dumps(result["manifest"], indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    for path in (X_PATH, Y_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    X, y_all = pd.read_csv(X_PATH), pd.read_csv(Y_PATH)
    result = run_experiment(X, y_all)
    write_outputs(result)
    print(result["summary_metrics"].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Warnings captured: {len(result['manifest']['warnings'])}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
