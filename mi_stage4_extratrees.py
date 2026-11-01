"""Stage 4.2: Extra Trees evaluation with nested, training-only CV.

Run from the project root with `.venv/bin/python mi_stage4_extratrees.py`.
No held-out test files are read.
"""
from __future__ import annotations

from pathlib import Path
import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage4-mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
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
STAGE3 = ROOT / "stage3_outputs"
OUT = ROOT / "stage4_extratrees_outputs"
SEED, OUTER_FOLDS, INNER_FOLDS, SEARCH_ITER = 42, 5, 3, 6
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]
CONTINUOUS = ["AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT",
              "K_BLOOD", "Na_BLOOD", "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD", "L_BLOOD", "ROE"]
ORDINAL = ["INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST", "GB", "DLIT_AG", "ZSN_A", "TIME_B_S"]
NOMINAL = ["ant_im", "lat_im", "inf_im", "post_im"]
METRICS = ["average_precision", "auroc", "brier_score", "brier_skill_score",
           "accuracy_at_0_5", "precision_at_0_5", "recall_at_0_5",
           "specificity_at_0_5", "f1_at_0_5"]
ET_MODELS = ["Extra Trees preliminary", "Extra Trees tuned"]
COMPARATORS = ["Logistic regression", "Random Forest preliminary", "Random Forest tuned"]
ALL_MODELS = ["Dummy prior prevalence", "Logistic regression",
              "Random Forest preliminary", "Random Forest tuned", *ET_MODELS]
PRIMARY = ["average_precision", "auroc", "brier_score"]
FOLD_SIGNATURE = ["n_train", "n_validation", "validation_positive_cases",
                 "validation_negative_cases", "training_prevalence"]


def make_pipeline(binary_features: list[str], *, n_estimators: int = 200) -> Pipeline:
    """Stage 3 preprocessing, fitted within each CV training fold."""
    prep = ColumnTransformer([
        # Retain KFK_BLOOD when a training fold has no observed measurements.
        ("continuous", SimpleImputer(strategy="median", keep_empty_features=True), CONTINUOUS),
        ("ordinal", SimpleImputer(strategy="most_frequent"), ORDINAL),
        ("binary", SimpleImputer(strategy="most_frequent"), binary_features),
        ("ecg", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), NOMINAL),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("classifier", ExtraTreesClassifier(
        n_estimators=n_estimators, random_state=SEED, n_jobs=1,
    ))])


def check_probabilities(probability: np.ndarray, classes: np.ndarray,
                        target: str, model_name: str, fold: int) -> None:
    if 1 not in classes:
        raise ValueError(f"{target} {model_name} fold {fold}: fitted model lacks positive class.")
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError(f"{target} {model_name} fold {fold}: missing/non-finite probabilities.")
    if ((probability < -1e-12) | (probability > 1 + 1e-12)).any():
        raise ValueError(f"{target} {model_name} fold {fold}: probabilities outside [0, 1].")


def score(y_true: np.ndarray, probability: np.ndarray, prevalence: float) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    brier = brier_score_loss(y_true, probability)
    null_brier = prevalence * (1 - prevalence)
    return {
        "average_precision": average_precision_score(y_true, probability),
        "auroc": roc_auc_score(y_true, probability),
        "brier_score": brier,
        "brier_skill_score": 1 - brier / null_brier if null_brier else np.nan,
        "accuracy_at_0_5": accuracy_score(y_true, prediction),
        "precision_at_0_5": precision_score(y_true, prediction, zero_division=0),
        "recall_at_0_5": recall_score(y_true, prediction, zero_division=0),
        "specificity_at_0_5": tn / (tn + fp) if tn + fp else np.nan,
        "f1_at_0_5": f1_score(y_true, prediction, zero_division=0),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def warning_text(caught: list[warnings.WarningMessage]) -> str:
    return " | ".join(f"{w.category.__name__}: {w.message}" for w in caught)


def assert_reference_folds(reference: pd.DataFrame, target: str, model: str,
                           y: pd.Series, splits: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """Check saved fold signatures against deterministic current split labels."""
    source = reference[(reference.target == target) & (reference.model == model)]
    if source.duplicated(["fold"]).any() or set(source.fold.astype(int)) != set(range(1, OUTER_FOLDS + 1)):
        raise ValueError(f"{target}: {model} does not have exactly one row for each expected fold.")
    checks = 0
    for fold, (train_pos, valid_pos) in enumerate(splits, start=1):
        saved = source.loc[source.fold.astype(int) == fold].iloc[0]
        y_train = y.iloc[train_pos].to_numpy()
        y_valid = y.iloc[valid_pos].to_numpy()
        actual = {
            "n_train": len(train_pos), "n_validation": len(valid_pos),
            "validation_positive_cases": int(y_valid.sum()),
            "validation_negative_cases": int(len(y_valid) - y_valid.sum()),
            "training_prevalence": float(y_train.mean()),
        }
        for column, value in actual.items():
            if column not in source:
                raise ValueError(f"{model} fold file is missing {column} for fold verification.")
            matches = (np.isclose(float(saved[column]), value, atol=1e-12)
                       if column == "training_prevalence" else int(saved[column]) == value)
            if not matches:
                raise ValueError(f"{target}/{model}/fold {fold}: {column} does not match regenerated split.")
            checks += 1
    return checks


def aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), part in folds.groupby(["target", "model"], sort=False):
        row = {"target": target, "model": model, "folds": len(part)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train_paths = [STAGE1 / "X_train.csv", STAGE1 / "y_train.csv", STAGE1 / "column_dictionary.csv"]
    reference_paths = [STAGE2 / "fold_level_metrics.csv", STAGE2 / "aggregate_metrics.csv",
                       STAGE3 / "fold_level_metrics.csv", STAGE3 / "aggregate_metrics.csv"]
    for path in train_paths + reference_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Deliberately do not enumerate, open, or read X_test.csv/y_test.csv.
    X = pd.read_csv(train_paths[0])
    y_all = pd.read_csv(train_paths[1])
    dictionary = pd.read_csv(train_paths[2])
    expected_columns = dictionary.loc[dictionary.stage1_role == "admission-time predictor", "variable"].tolist()
    if X.columns.tolist() != expected_columns or len(X) != len(y_all):
        raise ValueError("Stage 1 training rows/features do not match the saved dictionary.")
    if not set(TARGETS).issubset(y_all.columns):
        raise ValueError("One or more required targets are missing from y_train.csv.")
    typed = CONTINUOUS + ORDINAL + NOMINAL
    if not set(typed).issubset(X.columns):
        raise ValueError("A declared feature group is missing from X_train.csv.")
    binary = [column for column in X.columns if column not in typed]
    if len(typed) + len(binary) != X.shape[1]:
        raise ValueError("Feature groups do not partition the predictor columns.")
    for column in binary:
        codes = set(X[column].dropna().unique())
        if not codes.issubset({0, 1}):
            raise ValueError(f"Binary feature {column} contains unexpected codes: {codes}")

    stage2_folds = pd.read_csv(reference_paths[0])
    stage2_aggregate = pd.read_csv(reference_paths[1])
    stage3_folds = pd.read_csv(reference_paths[2])
    stage3_aggregate = pd.read_csv(reference_paths[3])
    rows, target_rows, tuning_rows = [], [], []
    search_space = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [None, 8, 16],
        "classifier__max_features": ["sqrt", 0.5],
        "classifier__min_samples_leaf": [1, 3, 5],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__class_weight": [None, "balanced"],
    }
    all_fold_checks = 0

    for target in TARGETS:
        y = y_all[target].dropna().astype(int)
        if set(y.unique()) != {0, 1} or y.value_counts().min() < OUTER_FOLDS:
            raise ValueError(f"{target} labels are not suitable for stratified binary CV.")
        X_target = X.loc[y.index]
        splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X_target, y))

        # Stage 2/3 CSVs contain fold counts, not patient IDs. Recreate each
        # deterministic fold and check its saved size, class counts and prevalence.
        for ref, model in [
            (stage2_folds, "Dummy prior prevalence"),
            (stage2_folds, "Logistic regression"),
            (stage3_folds, "Random Forest preliminary"),
            (stage3_folds, "Random Forest tuned"),
        ]:
            all_fold_checks += assert_reference_folds(ref, target, model, y, splits)

        target_rows.append({
            "target": target, "usable_training_observations": len(y),
            "missing_training_labels": int(y_all[target].isna().sum()),
            "positive_cases": int(y.sum()), "negative_cases": int(len(y) - y.sum()),
            "positive_prevalence": float(y.mean()),
            "outer_folds": OUTER_FOLDS, "fold_signature_checks": 4 * OUTER_FOLDS * len(FOLD_SIGNATURE),
            "fold_signature_checks_passed": 4 * OUTER_FOLDS * len(FOLD_SIGNATURE),
        })

        for fold, (train_pos, valid_pos) in enumerate(splits, start=1):
            X_train, X_valid = X_target.iloc[train_pos], X_target.iloc[valid_pos]
            y_train = y.iloc[train_pos].to_numpy()
            y_valid = y.iloc[valid_pos].to_numpy()
            train_prevalence = float(y_train.mean())
            # The preliminary model is intentionally modest and fixed.
            preliminary = make_pipeline(binary, n_estimators=200)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                preliminary.fit(X_train, y_train)
                classes = preliminary.named_steps["classifier"].classes_
                positive_col = int(np.flatnonzero(classes == 1)[0])
                probability = preliminary.predict_proba(X_valid)[:, positive_col]
            check_probabilities(probability, classes, target, ET_MODELS[0], fold)
            row = {
                "target": target, "model": ET_MODELS[0], "fold": fold,
                "n_train": len(train_pos), "n_validation": len(valid_pos),
                "validation_positive_cases": int(y_valid.sum()),
                "validation_negative_cases": int(len(y_valid) - y_valid.sum()),
                "training_prevalence": train_prevalence,
                "probability_check_passed": True,
                "probability_min": float(probability.min()), "probability_max": float(probability.max()),
                "fit_warnings": warning_text(caught), "best_parameters": "",
            }
            row.update(score(y_valid, probability, train_prevalence)); rows.append(row)

            # Serial inner search bounds resource use on the 8 GB MacBook Air.
            inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                                    random_state=SEED + fold)
            search = RandomizedSearchCV(
                make_pipeline(binary), param_distributions=search_space,
                n_iter=SEARCH_ITER, scoring="average_precision", cv=inner,
                random_state=SEED + fold, n_jobs=1, refit=True,
                error_score="raise", return_train_score=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                search.fit(X_train, y_train)
                tuned_classes = search.best_estimator_.named_steps["classifier"].classes_
                positive_col = int(np.flatnonzero(tuned_classes == 1)[0])
                probability = search.predict_proba(X_valid)[:, positive_col]
            check_probabilities(probability, tuned_classes, target, ET_MODELS[1], fold)
            params = {key.replace("classifier__", ""): value
                      for key, value in search.best_params_.items()}
            warning_summary = warning_text(caught)
            tuning_rows.append({
                "target": target, "outer_fold": fold,
                "inner_cv_best_average_precision": float(search.best_score_),
                "best_parameters": str(params), "candidates_evaluated": SEARCH_ITER,
                "inner_cv_folds": INNER_FOLDS, "search_n_jobs": 1,
                "search_warnings": warning_summary,
            })
            row = {
                "target": target, "model": ET_MODELS[1], "fold": fold,
                "n_train": len(train_pos), "n_validation": len(valid_pos),
                "validation_positive_cases": int(y_valid.sum()),
                "validation_negative_cases": int(len(y_valid) - y_valid.sum()),
                "training_prevalence": train_prevalence,
                "probability_check_passed": True,
                "probability_min": float(probability.min()), "probability_max": float(probability.max()),
                "fit_warnings": warning_summary, "best_parameters": str(params),
            }
            row.update(score(y_valid, probability, train_prevalence)); rows.append(row)

    et_folds = pd.DataFrame(rows)
    et_aggregate = aggregate(et_folds)
    # Pull already-computed baseline and RF metrics; never refit those models here.
    stage2_rows = stage2_aggregate[
        stage2_aggregate.target.isin(TARGETS) &
        stage2_aggregate.model.isin(["Dummy prior prevalence", "Logistic regression"])]
    stage3_rows = stage3_aggregate[
        stage3_aggregate.target.isin(TARGETS) & stage3_aggregate.model.isin(COMPARATORS[1:])]
    comparison = pd.concat([stage2_rows, stage3_rows, et_aggregate], ignore_index=True)
    if set(comparison.model.unique()) != set(ALL_MODELS):
        raise ValueError("Could not assemble all six models in the aggregate comparison.")
    if comparison.duplicated(["target", "model"]).any() or len(comparison) != len(TARGETS) * len(ALL_MODELS):
        raise ValueError("Aggregate comparison has missing or duplicate target/model rows.")
    comparison["model"] = pd.Categorical(comparison.model, ALL_MODELS, ordered=True)
    comparison = comparison.sort_values(["target", "model"]).reset_index(drop=True)

    # Pair every Extra Trees fold with LR and both RF folds, using target/fold.
    et_compare = et_folds[["target", "model", "fold", *PRIMARY]].rename(
        columns={m: f"extra_trees_{m}" for m in PRIMARY})
    comparator_rows = pd.concat([
        stage2_folds[stage2_folds.model == "Logistic regression"],
        stage3_folds[stage3_folds.model.isin(COMPARATORS[1:])],
    ], ignore_index=True)
    comparator_rows = comparator_rows[["target", "model", "fold", *PRIMARY,
        "n_validation", "validation_positive_cases", "validation_negative_cases"]].rename(
            columns={**{m: f"comparator_{m}" for m in PRIMARY},
                     "model": "comparator_model"})
    paired = et_compare.merge(comparator_rows, on=["target", "fold"], how="outer",
                              validate="many_to_many", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Extra Trees and existing model fold rows failed to pair.")
    paired = paired.drop(columns="_merge")
    for metric in PRIMARY:
        delta = paired[f"extra_trees_{metric}"] - paired[f"comparator_{metric}"]
        paired[f"extra_trees_minus_comparator_{metric}"] = delta
        better = delta < 0 if metric == "brier_score" else delta > 0
        paired[f"extra_trees_beats_comparator_{metric}"] = better
        paired[f"tie_{metric}"] = np.isclose(delta, 0.0, atol=1e-12)
    # Metric-specific direction is explicit in these columns.
    paired["average_precision_direction_for_improvement"] = "higher"
    paired["auroc_direction_for_improvement"] = "higher"
    paired["brier_score_direction_for_improvement"] = "lower"
    if len(paired) != len(TARGETS) * OUTER_FOLDS * len(ET_MODELS) * len(COMPARATORS):
        raise ValueError("Unexpected number of paired comparisons.")

    OUT.mkdir(exist_ok=True)
    et_folds.to_csv(OUT / "fold_level_metrics.csv", index=False)
    pd.DataFrame(target_rows).to_csv(OUT / "target_summary.csv", index=False)
    et_aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    paired.to_csv(OUT / "paired_comparison.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "tuning_results.csv", index=False)

    colors = ["#A5A5A5", "#4472C4", "#70AD47", "#ED7D31", "#5B9BD5", "#C55A11"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    plot_metrics = [("average_precision", "Average Precision (higher is better)"),
                    ("auroc", "AUROC (higher is better)"),
                    ("brier_score", "Brier score (lower is better)")]
    x = np.arange(len(TARGETS))
    width = .13
    for ax, (metric, title) in zip(axes, plot_metrics):
        for index, (model, color) in enumerate(zip(ALL_MODELS, colors)):
            subset = comparison[comparison.model == model].set_index("target").loc[TARGETS]
            ax.bar(x + (index - 2.5) * width, subset[f"{metric}_mean"], width,
                   yerr=subset[f"{metric}_std"], capsize=2, label=model, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, ["Death", "ZSN", "FIBR_PREDS"])
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Six-model comparison: five-fold cross-validation means ± fold SD")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Stage 4.2: Extra Trees Classifier", "",
        "## Scope and design", "",
        f"This stage evaluates `DEATH_BINARY`, `ZSN`, and `FIBR_PREDS` on {len(X)} Stage 1 training rows and {X.shape[1]} predictors. Only `X_train.csv` and `y_train.csv` are opened; the locked 20% test set is not used. Rows missing the target label are excluded for that target. No Stage 1, Stage 2, or Stage 3 files are modified.", "",
        f"For each target, evaluation uses shuffled `StratifiedKFold(n_splits={OUTER_FOLDS}, random_state={SEED})`. Preprocessing is in a scikit-learn Pipeline and is refit within each fold. Continuous predictors use median imputation; binary and ordinal codes use most-frequent imputation and remain numeric; four ECG morphology fields use most-frequent imputation and one-hot encoding. Scaling is omitted because tree split rules do not depend on feature units. The all-missing `KFK_BLOOD` column is retained as a constant feature.", "",
        "## Extra Trees models and tuning", "",
        "The preliminary classifier uses 200 randomized trees, default Extra Trees split settings, and a fixed seed. Extra Trees chooses split thresholds more randomly than a standard Random Forest; this can reduce correlation among trees and may improve or worsen generalization. The preliminary model gives a fixed-setting reference.", "",
        f"Tuning uses a restrained randomized search ({SEARCH_ITER} candidates per outer fold) with three-fold stratified inner CV and Average Precision as the selection metric. It explores `n_estimators` (100–200), `max_depth`, `max_features`, `min_samples_leaf`, `min_samples_split`, and `class_weight`. Search is serial (`n_jobs=1`) to limit memory and CPU use on the 8 GB MacBook Air. Each outer fold's validation rows are held out from its inner search and scored only after refitting the selected parameters on that outer training fold. Selected settings and inner AP are in `tuning_results.csv`.", "",
        "## Fold and probability checks", "",
        f"Recreated all target-specific folds with the Stage 2/3 splitter. Compared each of five folds per target to saved Stage 2 Dummy and Logistic Regression rows and Stage 3 preliminary and tuned Random Forest rows: {all_fold_checks} fold-signature checks passed. Checks cover training/validation size, validation positive/negative counts, and training prevalence. Existing metric files do not contain patient IDs, so exact record-by-record membership cannot be independently checked from those files; the deterministic splitter and unchanged training row order are used to reproduce membership.", "",
        "Each Extra Trees prediction vector is checked for finite values and probabilities within [0,1]. A missing class or invalid probability stops the script. Fit and search warnings are captured in the fold and tuning CSVs; fitting errors stop execution. No outer-fold threshold optimization is performed. Threshold metrics use `probability >= 0.5`.", "",
        "## Results", "",
        "The table reports mean (sample standard deviation) across five outer folds. AP, AUROC, and Brier use predicted probabilities. Higher AP/AUROC is better; lower Brier is better. These fold standard deviations are descriptive, not confidence intervals.", "",
        "| Target | Model | AP | AUROC | Brier | Brier skill | Accuracy | Precision | Recall | Specificity | F1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for model in ALL_MODELS:
            result = comparison[(comparison.target == target) & (comparison.model == model)].iloc[0]
            cells = [f"{result[m+'_mean']:.3f} ({result[m+'_std']:.3f})" for m in METRICS]
            lines.append(f"| `{target}` | {model} | " + " | ".join(cells) + " |")
    lines += ["", "### Reading these results", ""]
    for target in TARGETS:
        subset = comparison[comparison.target == target]
        ap_leader = subset.loc[subset.average_precision_mean.idxmax()]
        brier_leader = subset.loc[subset.brier_score_mean.idxmin()]
        lines.append(
            f"- **{target}:** highest mean AP is {ap_leader['model']} "
            f"({ap_leader['average_precision_mean']:.3f}); lowest mean Brier is "
            f"{brier_leader['model']} ({brier_leader['brier_score_mean']:.3f}). "
            "Check the paired fold file for how often Extra Trees beats each comparator; small mean differences and five folds do not establish a reliable winner."
        )
    lines += ["", "### Paired comparisons", "",
        "`paired_comparison.csv` contains fold-level differences for each Extra Trees version against Logistic Regression and both Random Forest versions. Differences are Extra Trees minus comparator. A positive AP/AUROC difference favors Extra Trees; a negative Brier difference favors Extra Trees. The file also records fold wins and ties. No statistical significance is claimed from five folds.", ""]
    for target in TARGETS:
        lines.append(f"**{target}**")
        for et_model in ET_MODELS:
            pieces = []
            for comparator in COMPARATORS:
                part = paired[(paired.target == target) & (paired.model == et_model) &
                              (paired.comparator_model == comparator)]
                counts = []
                for metric, short in [("average_precision", "AP"), ("auroc", "AUROC"), ("brier_score", "Brier")]:
                    count = int(part[f"extra_trees_beats_comparator_{metric}"].sum())
                    counts.append(f"{short} {count}/5")
                pieces.append(f"vs {comparator}: " + ", ".join(counts))
            lines.append(f"- {et_model}: " + "; ".join(pieces))
        lines.append("")
    lines += [
        "## Method differences and limitations", "",
        "The evaluation folds and imputation/encoding choices match Stage 3. This stage changes the estimator to Extra Trees and uses six randomized candidates rather than Stage 3's eight, while tuning additional tree parameters. Stage 3 parallelized its search with `n_jobs=-1`; Stage 4 deliberately uses serial fitting to reduce resource pressure. Since fold files store fold metrics rather than patient IDs, fold matching is verified through reproducible splitting plus saved fold signatures, not a direct record-ID comparison.", "",
        "This is internal validation on a small dataset. Five folds do not establish statistical significance, and repeated model selection can make apparent performance optimistic. Class weighting can affect probability calibration. The fixed 0.5 threshold is only a reference, not a clinical decision rule. Highly incomplete measurements remain imputed and may contribute little information.", "",
        f"scikit-learn `{sklearn.__version__}`; random seed `{SEED}`. Run `.venv/bin/python mi_stage4_extratrees.py`.", "",
        "## Output files", "",
        "- `fold_level_metrics.csv`: preliminary and tuned Extra Trees metrics, confusion counts, probability checks, warnings, and selected parameters.",
        "- `target_summary.csv`: target counts and fold-signature check totals.",
        "- `aggregate_metrics.csv`: Extra Trees means and sample standard deviations.",
        "- `model_comparison.csv`: all six models across all targets.",
        "- `paired_comparison.csv`: fold-level paired Extra Trees differences against Logistic Regression and both Random Forest models.",
        "- `tuning_results.csv`: selected parameters and inner-CV AP for every outer fold.",
        "- `model_comparison.png`: AP, AUROC, and Brier comparison plot.",
        "- `stage4_report.md`: methodology, checks, results, comparisons, and limitations.",
    ]
    (OUT / "stage4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Fold signatures checked: {all_fold_checks} passed")
    print(comparison[["target", "model", "average_precision_mean", "auroc_mean",
                      "brier_score_mean"]].to_string(index=False))
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
