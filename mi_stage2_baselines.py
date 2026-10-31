
from __future__ import annotations

from pathlib import Path
import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage2-mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parent
STAGE1_DIR = ROOT / "stage1_outputs"
OUTPUT_DIR = ROOT / "stage2_outputs"
RANDOM_STATE = 42
N_SPLITS = 5
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]

# The assignment's codebook identifies these as ordered codes, not nominal
# labels. They stay numeric and are mode-imputed, but are not standardized.
ORDINAL_FEATURES = [
    "INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST", "GB", "DLIT_AG",
    "ZSN_A", "TIME_B_S",
]

# These measurements have physical units and continuous interpretation in the
# assignment dictionary. Scale them for regularized logistic regression.
CONTINUOUS_FEATURES = [
    "AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT",
    "K_BLOOD", "Na_BLOOD", "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD",
    "L_BLOOD", "ROE",
]

# ECG morphology codes 0-4 describe distinct QRS patterns. The guide does not
# establish a numeric severity distance between those patterns, so one-hot
# encoding avoids treating the code number as a linear measurement.
NOMINAL_FEATURES = ["ant_im", "lat_im", "inf_im", "post_im"]


def build_pipeline(feature_columns: list[str], binary_features: list[str]) -> Pipeline:
    """Create fold-fitted imputation, encoding, scaling, and logistic model."""
    preprocess = ColumnTransformer(
        transformers=[
            ("continuous", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), CONTINUOUS_FEATURES),
            ("ordered_codes", SimpleImputer(strategy="most_frequent"),
             ORDINAL_FEATURES),
            ("binary", SimpleImputer(strategy="most_frequent"), binary_features),
            ("ecg_patterns", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), NOMINAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            l1_ratio=0.0, C=1.0, solver="lbfgs", max_iter=2000,
            class_weight=None, random_state=RANDOM_STATE,
        )),
    ])


def score_probabilities(y_true: np.ndarray, probability: np.ndarray,
                        threshold: np.ndarray, train_prevalence: float) -> dict:
    """Return probabilistic and default-threshold metrics for one fold."""
    tn, fp, fn, tp = confusion_matrix(y_true, threshold, labels=[0, 1]).ravel()
    baseline_brier = train_prevalence * (1.0 - train_prevalence)
    brier = brier_score_loss(y_true, probability)
    return {
        "average_precision": average_precision_score(y_true, probability),
        "auroc": roc_auc_score(y_true, probability),
        "brier_score": brier,
        "brier_skill_score": 1.0 - brier / baseline_brier if baseline_brier else np.nan,
        "accuracy_at_0_5": accuracy_score(y_true, threshold),
        "precision_at_0_5": precision_score(y_true, threshold, zero_division=0),
        "recall_at_0_5": recall_score(y_true, threshold, zero_division=0),
        "specificity_at_0_5": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1_at_0_5": f1_score(y_true, threshold, zero_division=0),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    x_path, y_path = STAGE1_DIR / "X_train.csv", STAGE1_DIR / "y_train.csv"
    id_path = STAGE1_DIR / "train_record_ids.csv"
    for path in [x_path, y_path, id_path, STAGE1_DIR / "column_dictionary.csv",
                 STAGE1_DIR / "training_missingness.csv"]:
        if not path.is_file():
            raise FileNotFoundError(path)

    X = pd.read_csv(x_path)
    y_all = pd.read_csv(y_path)
    train_ids = pd.read_csv(id_path)
    dictionary = pd.read_csv(STAGE1_DIR / "column_dictionary.csv")
    missingness = pd.read_csv(STAGE1_DIR / "training_missingness.csv")
    if len(X) != len(y_all) or len(X) != len(train_ids):
        raise ValueError("Stage 1 training predictors, outcomes, and IDs are misaligned in length.")
    if train_ids.iloc[:, 0].isna().any() or train_ids.iloc[:, 0].duplicated().any():
        raise ValueError("Training record IDs are missing or duplicated.")
    if X.columns.duplicated().any() or not X.columns.is_unique:
        raise ValueError("Duplicate feature columns found in X_train.csv.")
    if not set(TARGETS).issubset(y_all.columns):
        raise ValueError(f"Missing required Stage 2 targets: {sorted(set(TARGETS)-set(y_all.columns))}")
    if "LET_IS" not in y_all or not np.array_equal(
            y_all["DEATH_BINARY"].to_numpy(), (y_all["LET_IS"] > 0).astype("int8").to_numpy()):
        raise ValueError("Stage 1 DEATH_BINARY is inconsistent with LET_IS > 0.")
    if len(dictionary) != 124 or not (dictionary["stage1_role"] == "admission-time predictor").sum() == X.shape[1]:
        raise ValueError("Stage 1 dictionary and predictor count do not match.")
    allowed_features = dictionary.loc[
        dictionary["stage1_role"] == "admission-time predictor", "variable"
    ].tolist()
    if X.columns.tolist() != allowed_features:
        raise ValueError("X_train feature order differs from the Stage 1 dictionary.")
    if not set(CONTINUOUS_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES).issubset(X.columns):
        raise ValueError("A declared feature type is absent from X_train.")
    binary_features = [c for c in X.columns if c not in
                       CONTINUOUS_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES]
    if len(CONTINUOUS_FEATURES) + len(ORDINAL_FEATURES) + len(NOMINAL_FEATURES) + len(binary_features) != X.shape[1]:
        raise ValueError("Feature type groups do not partition X_train.")
    for feature in binary_features:
        observed = set(X[feature].dropna().unique())
        if not observed.issubset({0, 1}):
            raise ValueError(f"Feature {feature} is declared binary but has codes {observed}.")

    high_missing = missingness[missingness["missing_rate"] > 0.50]
    if high_missing.empty:
        missingness_note = "No training feature exceeds 50% missingness."
    else:
        missingness_note = "; ".join(
            f"{r.feature} {r.missing_rate:.1%}" for r in high_missing.itertuples()
        )

    target_rows = []
    fold_rows = []
    metric_names = [
        "average_precision", "auroc", "brier_score", "brier_skill_score",
        "accuracy_at_0_5", "precision_at_0_5", "recall_at_0_5",
        "specificity_at_0_5", "f1_at_0_5",
    ]
    comparison_rows = []
    for target in TARGETS:
        target_series = y_all[target]
        missing_target = int(target_series.isna().sum())
        observed_y = target_series.dropna().astype(int)
        codes = set(observed_y.unique())
        if codes != {0, 1}:
            raise ValueError(f"{target} must be observed binary 0/1; got {sorted(codes)}.")
        positive = int(observed_y.sum())
        negative = int(len(observed_y) - positive)
        prevalence = positive / len(observed_y)
        if observed_y.value_counts().min() < N_SPLITS:
            raise ValueError(f"{target} has fewer than {N_SPLITS} examples in a class; stratification is infeasible.")
        target_rows.append({
            "target": target, "usable_training_observations": len(observed_y),
            "missing_training_labels": missing_target,
            "positive_cases": positive, "negative_cases": negative,
            "positive_prevalence": prevalence,
        })

        # For each target, stratification is based on observed labels. The same
        # materialized folds are passed to both estimators.
        X_target = X.loc[observed_y.index]
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=RANDOM_STATE)
        splits = list(splitter.split(X_target, observed_y))
        for fold_number, (train_positions, valid_positions) in enumerate(splits, start=1):
            X_fold_train = X_target.iloc[train_positions]
            X_fold_valid = X_target.iloc[valid_positions]
            y_fold_train = observed_y.iloc[train_positions].to_numpy()
            y_fold_valid = observed_y.iloc[valid_positions].to_numpy()
            fold_prevalence = float(np.mean(y_fold_train))
            for model_name in ["Dummy prior prevalence", "Logistic regression"]:
                fit_warnings = []
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if model_name == "Dummy prior prevalence":
                        model = DummyClassifier(strategy="prior")
                    else:
                        model = build_pipeline(X.columns.tolist(), binary_features)
                    model.fit(X_fold_train, y_fold_train)
                    probability = model.predict_proba(X_fold_valid)[:, 1]
                    prediction = model.predict(X_fold_valid)
                fit_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
                row = {
                    "target": target, "model": model_name, "fold": fold_number,
                    "n_train": len(train_positions), "n_validation": len(valid_positions),
                    "validation_positive_cases": int(y_fold_valid.sum()),
                    "validation_negative_cases": int(len(y_fold_valid)-y_fold_valid.sum()),
                    "training_prevalence": fold_prevalence,
                    "fit_warnings": " | ".join(fit_warnings),
                }
                row.update(score_probabilities(y_fold_valid, probability, prediction,
                                               fold_prevalence))
                fold_rows.append(row)

        # Paired, fold-by-fold model differences, with metric direction explicit.
        target_folds = pd.DataFrame([r for r in fold_rows if r["target"] == target])
        for metric in metric_names:
            dummy_values = target_folds.loc[target_folds["model"] == "Dummy prior prevalence", metric]
            lr_values = target_folds.loc[target_folds["model"] == "Logistic regression", metric]
            lower_is_better = metric in {"brier_score"}
            difference = float(lr_values.mean() - dummy_values.mean())
            comparison_rows.append({
                "target": target, "metric": metric,
                "dummy_mean": float(dummy_values.mean()),
                "dummy_std": float(dummy_values.std(ddof=1)),
                "logistic_mean": float(lr_values.mean()),
                "logistic_std": float(lr_values.std(ddof=1)),
                "logistic_minus_dummy": difference,
                "direction_for_improvement": "lower" if lower_is_better else "higher",
                "logistic_improves_mean": difference < 0 if lower_is_better else difference > 0,
            })

    folds = pd.DataFrame(fold_rows)
    targets = pd.DataFrame(target_rows)
    folds.to_csv(OUTPUT_DIR / "fold_level_metrics.csv", index=False)
    targets.to_csv(OUTPUT_DIR / "target_summary.csv", index=False)

    aggregate_rows = []
    for (target, model), subset in folds.groupby(["target", "model"], sort=False):
        row = {"target": target, "model": model, "folds": int(len(subset))}
        for metric in metric_names:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Plot the assignment's three primary probability metrics. Standard
    # deviations across five folds are shown as descriptive error bars.
    plot_metrics = [
        ("average_precision", "Average precision (AUPRC)"),
        ("auroc", "AUROC"),
        ("brier_score", "Brier score (lower is better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"Dummy prior prevalence": "#A5A5A5", "Logistic regression": "#4472C4"}
    x_positions = np.arange(len(TARGETS))
    width = 0.34
    for ax, (metric, title) in zip(axes, plot_metrics):
        for model_index, model_name in enumerate(colors):
            rows = aggregate[aggregate["model"] == model_name].set_index("target").loc[TARGETS]
            offset = (model_index - 0.5) * width
            ax.bar(x_positions + offset, rows[f"{metric}_mean"], width,
                   yerr=rows[f"{metric}_std"], capsize=3,
                   label=model_name, color=colors[model_name])
        ax.set_title(title)
        ax.set_xticks(x_positions, ["Death", "ZSN", "FIBR_PREDS"])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Five-fold cross-validation: baseline vs logistic regression")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    # AUPRC is the assignment's primary model-selection metric; relative to
    # prevalence it is especially useful for imbalanced outcomes.
    ap = aggregate.pivot(index="target", columns="model", values="average_precision_mean")
    br = aggregate.pivot(index="target", columns="model", values="brier_score_mean")
    lines = [
        "# Stage 2: prevalence baseline and logistic regression", "",
        "## Scope and data integrity", "",
        "This stage implements only the assignment-mandated `DEATH_BINARY`, `ZSN`, and `FIBR_PREDS` tasks. It reads `X_train.csv`, `y_train.csv`, training IDs, and Stage 1 metadata. It does not open `X_test.csv` or `y_test.csv`; the 20% held-out test set remains locked. No train/test split was recreated.", "",
        f"The training predictors and outcomes contain {len(X)} aligned rows and {X.shape[1]} matching Stage 1 admission-time features. Outcome labels missing in training: " + ", ".join(f"{r['target']}={r['missing_training_labels']}" for r in target_rows) + ".", "",
        "The 12 original outcomes are not all fitted here. The assignment names binary death (`LET_IS > 0`, exported by Stage 1 as `DEATH_BINARY`), `ZSN`, and `FIBR_PREDS` as the required initial targets; the other nine outcomes are not in Stage 2 scope.", "",
        "## Prevalence baseline", "",
        "For each fold, `DummyClassifier(strategy=\"prior\")` estimates the positive prevalence from that fold's training labels and assigns that same probability to each validation patient. It does not use the predictors. It is a meaningful reference: it measures what probability quality is available from class frequency alone. A model using patient features should improve on this reference, especially in average precision and Brier score.", "",
        "## Logistic regression and preprocessing", "",
        "The model is L2-regularized logistic regression (`C=1`, `lbfgs`, `max_iter=2000`, no class weighting). L2 regularization shrinks coefficients and limits instability with many correlated features. No hyperparameter search or threshold tuning is performed. No class weighting is used so the predicted probabilities retain the natural training prevalence scale for calibration comparisons.", "",
        "All transformations are inside a scikit-learn `Pipeline`, so each fold learns imputation values, scaling parameters, and one-hot categories from that fold's training rows only. Continuous measurements are median-imputed and standardized. Binary and documented ordinal codes are mode-imputed and kept on their native numeric scale; this follows the assignment's instruction not to one-hot these codes. Four ECG QRS morphology fields (`ant_im`, `lat_im`, `inf_im`, `post_im`) have distinct code patterns with no stated numeric distance, so they are mode-imputed and one-hot encoded. No categories are learned globally before CV.", "",
        f"Stage 1 training missingness review: {missingness_note}. In particular, highly incomplete variables remain included and are imputed within each fold, rather than using validation-fold missingness to select features. This can limit the information available from extremely incomplete measurements.", "",
        f"Feature groups: {len(CONTINUOUS_FEATURES)} continuous, {len(ORDINAL_FEATURES)} ordinal, {len(NOMINAL_FEATURES)} nominal ECG codes, and {len(binary_features)} binary. The declared groups cover all {X.shape[1]} predictors.", "",
        "## Cross-validation", "",
        f"The evaluation uses one reproducible `StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})` run on the 80% training split. Stratification is performed separately for each target using its observed binary labels. Each target has both classes represented in every fold. For a given target, the exact same materialized train/validation indices are used for Dummy and logistic regression. Each fit is run explicitly; errors stop the script, warnings are captured in the fold CSV, and no fold is silently dropped.", "",
        "Cross-validation estimates performance on held-out portions of the development data while retaining most data for each fit. It supports model comparison without using the locked test set. The reported fold standard deviation is descriptive variability across five folds, not a confidence interval.", "",
        "## Metrics", "",
        "- **Average precision (AUPRC):** step-wise area under the precision-recall curve, summarizing positive-case ranking over thresholds. It is the assignment's primary selection metric and should be read against outcome prevalence; the prior baseline's score is approximately the prevalence.",
        "- **AUROC:** probability that a randomly selected positive receives a higher score than a randomly selected negative. It measures ranking discrimination and can appear strong under severe imbalance, so it is not interpreted alone.",
        "- **Brier score:** mean squared error of predicted probabilities, `mean((p-y)^2)`. Lower is better; it reflects both calibration and discrimination. The Brier skill score is also reported relative to the fold's training-prevalence reference `p_train * (1-p_train)`: 0 matches that reference, positive is better, negative is worse.",
        "- **Threshold metrics:** accuracy, precision, recall/sensitivity, specificity, and positive-class F1 are reported at the estimator's default 0.5 decision threshold, with confusion-matrix counts. This is a transparent fixed reference, not a tuned clinical threshold. For the prior dummy, all three prevalences are below 0.5, so it predicts all negatives and has zero recall; this is why accuracy is not a useful model-selection metric here.", "",
        "Each fold's scores and confusion counts are in `fold_level_metrics.csv`; means and sample standard deviations are in `aggregate_metrics.csv`. `model_comparison.csv` gives paired-model average differences and indicates whether the logistic mean is better in the appropriate metric direction.", "",
        "## Results", "",
        "| Target | Positives / n | Dummy AP | Logistic AP | Dummy AUROC | Logistic AUROC | Dummy Brier | Logistic Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in target_rows:
        target = row["target"]
        d_ap = aggregate[(aggregate.target == target) & (aggregate.model == "Dummy prior prevalence")].iloc[0]
        l_ap = aggregate[(aggregate.target == target) & (aggregate.model == "Logistic regression")].iloc[0]
        lines.append(
            f"| `{target}` | {row['positive_cases']} / {row['usable_training_observations']} | "
            f"{d_ap.average_precision_mean:.3f} ({d_ap.average_precision_std:.3f}) | "
            f"{l_ap.average_precision_mean:.3f} ({l_ap.average_precision_std:.3f}) | "
            f"{d_ap.auroc_mean:.3f} ({d_ap.auroc_std:.3f}) | "
            f"{l_ap.auroc_mean:.3f} ({l_ap.auroc_std:.3f}) | "
            f"{d_ap.brier_score_mean:.3f} ({d_ap.brier_score_std:.3f}) | "
            f"{l_ap.brier_score_mean:.3f} ({l_ap.brier_score_std:.3f}) |"
        )
    lines += ["", "The plot at `model_comparison.png` shows fold means with one standard deviation error bars. For AP, prevalence is the natural reference; for Brier, lower scores are better. Logistic regression improves mean AP and AUROC over the prior baseline for all three targets. Its Brier score is better for mortality, marginally better for FIBR_PREDS, and slightly worse for ZSN. Review the CSVs for Brier skill, threshold metrics, and every fold. These are descriptive five-fold comparisons; overlapping variability and only five folds limit certainty.", "",
        "## Limitations and interpretation", "",
        "- This is internal cross-validation on 1,360 development records, not external validation. The held-out test set has not been evaluated.",
        "- A single five-fold seed gives a reproducible estimate but may be sensitive to split variation. The fold SD is not a formal uncertainty interval.",
        "- Some predictors are highly incomplete, especially `KFK_BLOOD`; fold-local imputation is necessary but cannot recover information that was not measured.",
        "- The assignment permits binary and ordinal codes to remain numeric. Logistic regression treats their numeric steps as linear effects on log odds, which may not capture nonlinear or category-specific effects.",
        "- Several ICU-era admission predictors may be close in time to complication onset. Stage 1 follows the assignment's allowed feature list and removes its explicitly post-admission exclusions; a stricter real-time prediction study would need event timestamps.",
        "- The 0.5 threshold metrics are not tuned for clinical costs. AUPRC, AUROC, and Brier are the main probability-based comparisons in this baseline stage.", "",
        "## Reproducibility", "",
        f"Environment scikit-learn version: `{sklearn.__version__}`. Random seed: `{RANDOM_STATE}`. Run with `.venv/bin/python mi_stage2_baselines.py`. The script uses only Stage 1 training files and writes all results to `stage2_outputs/`.", "",
        "## Outputs", "",
        "- `fold_level_metrics.csv`: model-target-fold scores, validation class counts, confusion-matrix cells, and captured warnings.",
        "- `aggregate_metrics.csv`: means and sample standard deviations across the five folds for each model and target.",
        "- `target_summary.csv`: usable training sample size, missing labels, class counts, and prevalence.",
        "- `model_comparison.csv`: mean differences between logistic regression and prior baseline, with direction-aware improvement flags.",
        "- `model_comparison.png`: side-by-side AUPRC, AUROC, and Brier comparisons.",
        "- `stage2_tutorial.md`: this methodology and results report.",
    ]
    (OUTPUT_DIR / "stage2_tutorial.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Training rows: {len(X)} | predictors: {X.shape[1]} | CV folds: {N_SPLITS}")
    print(targets.to_string(index=False))
    print("\nAggregate primary metrics:")
    print(aggregate[["target", "model", "average_precision_mean", "average_precision_std",
                     "auroc_mean", "auroc_std", "brier_score_mean", "brier_score_std"]]
          .to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
