"""Stage 3: compare preliminary and tuned Random Forests on training data only.

Run from the project root with `.venv/bin/python mi_stage3_random_forest.py`.
The locked Stage 1 test files are never read.
"""
from __future__ import annotations

from pathlib import Path
import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage3-mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE2 = ROOT / "stage2_outputs"
OUT = ROOT / "stage3_outputs"
SEED, OUTER_FOLDS, INNER_FOLDS, SEARCH_ITER = 42, 5, 3, 8
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]
CONTINUOUS = ["AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT",
              "K_BLOOD", "Na_BLOOD", "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD", "L_BLOOD", "ROE"]
ORDINAL = ["INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST", "GB", "DLIT_AG", "ZSN_A", "TIME_B_S"]
NOMINAL = ["ant_im", "lat_im", "inf_im", "post_im"]
METRICS = ["average_precision", "auroc", "brier_score", "brier_skill_score",
           "accuracy_at_0_5", "precision_at_0_5", "recall_at_0_5",
           "specificity_at_0_5", "f1_at_0_5"]
MODELS = ["Dummy prior prevalence", "Logistic regression",
          "Random Forest preliminary", "Random Forest tuned"]


def make_pipeline(binary: list[str]) -> Pipeline:
    # Imputation/encoding are fit within the CV training fold. Trees need no scaling.
    prep = ColumnTransformer([
        ("continuous", SimpleImputer(strategy="median", keep_empty_features=True), CONTINUOUS),
        ("ordinal", SimpleImputer(strategy="most_frequent"), ORDINAL),
        ("binary", SimpleImputer(strategy="most_frequent"), binary),
        ("ecg", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), NOMINAL),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("classifier", RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=1))])


def score(y: np.ndarray, p: np.ndarray, prev: float) -> dict:
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    brier = brier_score_loss(y, p)
    return {
        "average_precision": average_precision_score(y, p), "auroc": roc_auc_score(y, p),
        "brier_score": brier,
        "brier_skill_score": 1 - brier / (prev * (1-prev)) if prev not in (0, 1) else np.nan,
        "accuracy_at_0_5": accuracy_score(y, pred),
        "precision_at_0_5": precision_score(y, pred, zero_division=0),
        "recall_at_0_5": recall_score(y, pred, zero_division=0),
        "specificity_at_0_5": tn / (tn + fp) if tn + fp else np.nan,
        "f1_at_0_5": f1_score(y, pred, zero_division=0),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), part in folds.groupby(["target", "model"], sort=False):
        row = {"target": target, "model": model, "folds": len(part)}
        for metric in METRICS:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_std"] = part[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    X = pd.read_csv(STAGE1 / "X_train.csv")
    yall = pd.read_csv(STAGE1 / "y_train.csv")
    dictionary = pd.read_csv(STAGE1 / "column_dictionary.csv")
    expected = dictionary.loc[dictionary.stage1_role == "admission-time predictor", "variable"].tolist()
    if X.columns.tolist() != expected or len(X) != len(yall):
        raise ValueError("Stage 1 training data do not match the predictor dictionary or row counts.")
    if not set(TARGETS).issubset(yall.columns):
        raise ValueError("One or more required Stage 2 targets are missing.")
    typed = CONTINUOUS + ORDINAL + NOMINAL
    binary = [c for c in X.columns if c not in typed]
    for col in binary:
        if not set(X[col].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Expected binary codes in {col}.")

    rows, targets, tuning = [], [], []
    space = {"classifier__max_depth": [None, 5, 10],
             "classifier__max_features": ["sqrt", .5],
             "classifier__min_samples_leaf": [1, 3, 5],
             "classifier__class_weight": [None, "balanced_subsample"]}
    for target in TARGETS:
        y = yall[target].dropna().astype(int)
        if set(y.unique()) != {0, 1} or y.value_counts().min() < OUTER_FOLDS:
            raise ValueError(f"{target} is not suitable for stratified binary CV.")
        Xt = X.loc[y.index]
        targets.append({"target": target, "usable_training_observations": len(y),
            "missing_training_labels": int(yall[target].isna().sum()),
            "positive_cases": int(y.sum()), "negative_cases": int(len(y)-y.sum()),
            "positive_prevalence": float(y.mean())})
        outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        for fold, (itr, iva) in enumerate(outer.split(Xt, y), 1):
            xa, xv = Xt.iloc[itr], Xt.iloc[iva]
            ya, yv = y.iloc[itr].to_numpy(), y.iloc[iva].to_numpy()
            prev = float(ya.mean())
            # Fixed, preliminary forest on the common outer folds.
            rf = make_pipeline(binary)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rf.fit(xa, ya)
                p = rf.predict_proba(xv)[:, 1]
            row = {"target": target, "model": "Random Forest preliminary", "fold": fold,
                "n_train": len(itr), "n_validation": len(iva),
                "validation_positive_cases": int(yv.sum()),
                "validation_negative_cases": int(len(yv)-yv.sum()),
                "training_prevalence": prev,
                "fit_warnings": " | ".join(f"{w.category.__name__}: {w.message}" for w in caught),
                "best_parameters": ""}
            row.update(score(yv, p, prev)); rows.append(row)

            # Parameters are selected only inside the outer training rows; outer
            # validation rows score the selected model once, after that selection.
            inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED+fold)
            search = RandomizedSearchCV(make_pipeline(binary), space, n_iter=SEARCH_ITER,
                scoring="average_precision", cv=inner, random_state=SEED+fold,
                n_jobs=-1, refit=True, error_score="raise")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                search.fit(xa, ya)
                p = search.predict_proba(xv)[:, 1]
            params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
            warning_text = " | ".join(f"{w.category.__name__}: {w.message}" for w in caught)
            tuning.append({"target": target, "outer_fold": fold,
                "inner_cv_best_average_precision": search.best_score_,
                "best_parameters": str(params), "candidates_evaluated": SEARCH_ITER,
                "search_warnings": warning_text})
            row = {"target": target, "model": "Random Forest tuned", "fold": fold,
                "n_train": len(itr), "n_validation": len(iva),
                "validation_positive_cases": int(yv.sum()),
                "validation_negative_cases": int(len(yv)-yv.sum()),
                "training_prevalence": prev, "fit_warnings": warning_text,
                "best_parameters": str(params)}
            row.update(score(yv, p, prev)); rows.append(row)

    folds = pd.DataFrame(rows)
    sums = aggregate(folds)
    stage2 = pd.read_csv(STAGE2 / "aggregate_metrics.csv")
    stage2 = stage2[stage2.target.isin(TARGETS) & stage2.model.isin(MODELS[:2])]
    comparison = pd.concat([stage2, sums], ignore_index=True)
    comparison["model"] = pd.Categorical(comparison.model, MODELS, ordered=True)
    comparison = comparison.sort_values(["target", "model"]).reset_index(drop=True)
    folds.to_csv(OUT / "fold_level_metrics.csv", index=False)
    pd.DataFrame(targets).to_csv(OUT / "target_summary.csv", index=False)
    sums.to_csv(OUT / "aggregate_metrics.csv", index=False)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "tuning_results.csv", index=False)

    colors = ["#A5A5A5", "#4472C4", "#70AD47", "#ED7D31"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (metric, title) in zip(axes, [("average_precision", "Average precision"),
            ("auroc", "AUROC"), ("brier_score", "Brier score (lower is better)")]):
        for j, (model, color) in enumerate(zip(MODELS, colors)):
            part = comparison[comparison.model == model].set_index("target").loc[TARGETS]
            ax.bar(np.arange(3)+(j-1.5)*.19, part[f"{metric}_mean"], .19,
                   yerr=part[f"{metric}_std"], capsize=2, label=model, color=color)
        ax.set_title(title); ax.set_xticks(np.arange(3), ["Death", "ZSN", "FIBR_PREDS"])
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Stage 2 and Stage 3: five-fold model comparison")
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=180); plt.close(fig)

    lines = ["# Stage 3: Random Forest", "",
        f"Uses {len(X)} Stage 1 training rows and {X.shape[1]} predictors only. The locked test partition is not opened. The original 80/20 split is preserved. Logistic Regression and Dummy results are read from Stage 2 aggregates.", "",
        "Continuous values use fold-fitted median imputation; binary and ordinal codes use fold-fitted mode imputation and stay numeric. The four ECG morphology fields are mode-imputed and one-hot encoded, as in Stage 2. Scaling is omitted because tree splits do not depend on feature units. `KFK_BLOOD`, which is all-missing, is retained as a constant feature.", "",
        "The preliminary forest uses 200 trees and a fixed seed. Tuning uses eight randomized combinations per outer training fold, with three-fold stratified inner CV and Average Precision as the selection metric. It varies maximum tree depth, features considered at each split, minimum leaf size, and class weighting. The five outer folds estimate performance after parameter selection; the inner search scores are not reported as final performance. See `tuning_results.csv` for per-fold settings.", "",
        "Means (sample SD) across five folds; probability metrics use predicted probabilities and threshold metrics use 0.5.", "",
        "| Target | Model | AP | AUROC | Brier | Brier skill | Accuracy | Precision | Recall | Specificity | F1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for target in TARGETS:
        for model in MODELS:
            part = comparison[(comparison.target == target) & (comparison.model == model)]
            if part.empty: continue
            r = part.iloc[0]
            cells = [f"{r[m+'_mean']:.3f} ({r[m+'_std']:.3f})" for m in METRICS]
            lines.append(f"| `{target}` | {model} | " + " | ".join(cells) + " |")
    lines += ["", "Scores are internal CV estimates; five-fold SDs are descriptive, not confidence intervals. The search is intentionally restrained, and repeated model development can still make estimates optimistic. Class weighting may affect calibration. The 0.5 threshold is only a reference. Highly incomplete features remain imputed. Consider evaluating a finalized model once on the locked test partition after model selection is complete.", "",
        f"scikit-learn `{sklearn.__version__}`; seed `{SEED}`. Outputs: fold metrics, aggregate metrics, three-model comparison, tuning results, target summary, and plot. Stage 2 files are not modified."]
    (OUT / "stage3_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(comparison[["target", "model", "average_precision_mean", "auroc_mean", "brier_score_mean"]].to_string(index=False))
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
