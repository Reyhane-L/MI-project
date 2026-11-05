"""Stage 6: CatBoost comparison using nested, development-only CV.

Run from the project root with `.venv/bin/python mi_stage6_catboost.py` after
CatBoost is available in the project environment. This script uses only the
Stage 1 development artifacts and does not enumerate or read held-out files.
"""
from __future__ import annotations

from pathlib import Path
import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage6-mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import catboost
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE2 = ROOT / "stage2_outputs"
STAGE3 = ROOT / "stage3_outputs"
STAGE4 = ROOT / "stage4_extratrees_outputs"
OUT = ROOT / "stage6_catboost_outputs"
SEED, OUTER_FOLDS, INNER_FOLDS, SEARCH_ITER = 42, 5, 3, 4
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]
COMPARATORS = [
    "Dummy prior prevalence", "Logistic regression",
    "Random Forest preliminary", "Random Forest tuned",
    "Extra Trees preliminary", "Extra Trees tuned",
]
ALL_MODELS = [*COMPARATORS, "CatBoost tuned"]
CONTINUOUS = [
    "AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT",
    "K_BLOOD", "Na_BLOOD", "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD",
    "L_BLOOD", "ROE",
]
ORDINAL = ["INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST", "GB", "DLIT_AG", "ZSN_A", "TIME_B_S"]
# The dictionary describes four ECG morphology codes as different patterns,
# not ordered magnitudes. SEX is a nominal two-level attribute. Pass these to
# CatBoost natively; other 0/1 clinical indicators stay numeric.
CATEGORICAL = ["SEX", "ant_im", "lat_im", "inf_im", "post_im"]
FORBIDDEN = {
    "ID", "LET_IS", "DEATH_BINARY", "ZSN", "FIBR_PREDS",
    "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK", "OTEK_LANC",
    "RAZRIV", "R_AB_1_n", "R_AB_2_n", "R_AB_3_n",
    "NA_R_1_n", "NA_R_2_n", "NA_R_3_n", "NOT_NA_1_n", "NOT_NA_2_n",
    "NOT_NA_3_n",
}
MISSING_CATEGORY = "__MI_MISSING_CATEGORY__"
METRICS = [
    "average_precision", "auroc", "brier_score", "brier_skill_score",
    "accuracy_at_0_5", "precision_at_0_5", "recall_at_0_5",
    "specificity_at_0_5", "f1_at_0_5",
]
PRIMARY = ["average_precision", "auroc", "brier_score"]
FOLD_SIGNATURE = [
    "n_train", "n_validation", "validation_positive_cases",
    "validation_negative_cases", "training_prevalence",
]
OUTPUT_NAMES = [
    "fold_level_metrics.csv", "target_summary.csv", "aggregate_metrics.csv",
    "model_comparison.csv", "paired_comparison.csv", "tuning_results.csv",
    "oof_predictions.csv", "model_comparison.png", "stage6_report.md",
]


class CatBoostFrameTransformer(BaseEstimator, TransformerMixin):
    """Convert nominal columns to stable strings without learning from data."""

    def __init__(self, categorical_features: tuple[str, ...] = tuple(CATEGORICAL)):
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatBoost input must remain a pandas DataFrame.")
        missing = set(self.categorical_features) - set(X.columns)
        if missing:
            raise ValueError(f"Categorical inputs are missing: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatBoost input must remain a pandas DataFrame.")
        result = X.copy()
        for column in self.categorical_features:
            values = result[column].astype("string").fillna(MISSING_CATEGORY)
            result[column] = values.astype(str)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("catboost_frame", CatBoostFrameTransformer()),
        ("classifier", CatBoostClassifier(
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            loss_function="Logloss",
            eval_metric="PRAUC",
        )),
    ])


def score(y_true: np.ndarray, probability: np.ndarray, prevalence: float) -> dict:
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    brier = brier_score_loss(y_true, probability)
    null_brier = prevalence * (1.0 - prevalence)
    return {
        "average_precision": average_precision_score(y_true, probability),
        "auroc": roc_auc_score(y_true, probability),
        "brier_score": brier,
        "brier_skill_score": 1.0 - brier / null_brier if null_brier else np.nan,
        "accuracy_at_0_5": accuracy_score(y_true, predicted),
        "precision_at_0_5": precision_score(y_true, predicted, zero_division=0),
        "recall_at_0_5": recall_score(y_true, predicted, zero_division=0),
        "specificity_at_0_5": tn / (tn + fp) if tn + fp else np.nan,
        "f1_at_0_5": f1_score(y_true, predicted, zero_division=0),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def warning_text(caught: list[warnings.WarningMessage]) -> str:
    return " | ".join(f"{item.category.__name__}: {item.message}" for item in caught)


def check_probability(probability: np.ndarray, classes: np.ndarray, target: str, fold: int) -> None:
    if 1 not in classes:
        raise ValueError(f"{target} fold {fold}: fitted model lacks positive class.")
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError(f"{target} fold {fold}: probabilities are missing or non-finite.")
    if ((probability < -1e-12) | (probability > 1 + 1e-12)).any():
        raise ValueError(f"{target} fold {fold}: probabilities fall outside [0, 1].")


def assert_reference_folds(
    reference: pd.DataFrame, target: str, model: str, y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> int:
    source = reference[(reference.target == target) & (reference.model == model)]
    if source.duplicated(["fold"]).any() or set(source.fold.astype(int)) != set(range(1, OUTER_FOLDS + 1)):
        raise ValueError(f"{target}: {model} lacks exactly one row for each reference fold.")
    checks = 0
    for fold, (train_pos, valid_pos) in enumerate(splits, start=1):
        saved = source.loc[source.fold.astype(int) == fold].iloc[0]
        y_train = y.iloc[train_pos].to_numpy()
        y_valid = y.iloc[valid_pos].to_numpy()
        observed = {
            "n_train": len(train_pos), "n_validation": len(valid_pos),
            "validation_positive_cases": int(y_valid.sum()),
            "validation_negative_cases": int(len(y_valid) - y_valid.sum()),
            "training_prevalence": float(y_train.mean()),
        }
        for column, value in observed.items():
            if column not in source:
                raise ValueError(f"{model} reference folds lack {column}.")
            matches = (np.isclose(float(saved[column]), value, rtol=0, atol=1e-12)
                       if column == "training_prevalence" else int(saved[column]) == value)
            if not matches:
                raise ValueError(
                    f"{target} {model} fold {fold}: {column} mismatch "
                    f"(saved={saved[column]}, current={value})."
                )
            checks += 1
    return checks


def aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), part in folds.groupby(["target", "model"], sort=False):
        row = {"target": target, "model": model, "folds": len(part)}
        for metric in METRICS:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_std"] = part[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_inputs(X: pd.DataFrame, y_all: pd.DataFrame, ids: pd.DataFrame,
                    dictionary: pd.DataFrame) -> list[str]:
    if len(X) != len(y_all) or len(X) != len(ids):
        raise ValueError("Development predictors, outcomes, and IDs are not row-aligned.")
    if X.columns.duplicated().any() or not X.columns.is_unique:
        raise ValueError("Development predictor columns are duplicated.")
    if ids.shape[1] != 1 or ids.iloc[:, 0].isna().any() or ids.iloc[:, 0].duplicated().any():
        raise ValueError("Development IDs must be one non-null, unique ID column.")
    role_exclusions = set(dictionary.loc[
        dictionary.stage1_role != "admission-time predictor", "variable"
    ])
    forbidden = FORBIDDEN | role_exclusions
    if forbidden.intersection(X.columns):
        raise ValueError(f"Forbidden ID/outcome/post-admission features in predictors: {sorted(forbidden.intersection(X.columns))}")
    if not set(TARGETS + ["LET_IS"]).issubset(y_all.columns):
        raise ValueError("Required target or LET_IS column is absent from development outcomes.")
    if not np.array_equal(y_all["DEATH_BINARY"].to_numpy(), (y_all["LET_IS"] > 0).astype("int8").to_numpy()):
        raise ValueError("DEATH_BINARY does not equal LET_IS > 0.")
    expected = dictionary.loc[dictionary.stage1_role == "admission-time predictor", "variable"].tolist()
    if X.columns.tolist() != expected:
        raise ValueError("Predictor order differs from Stage 1 admission-time dictionary roles.")
    if not set(CONTINUOUS + ORDINAL + CATEGORICAL).issubset(X.columns):
        raise ValueError("A declared continuous, ordinal, or categorical feature is absent.")
    nominal_descriptions = dictionary.set_index("variable")["description"]
    for column in ["ant_im", "lat_im", "inf_im", "post_im"]:
        description = str(nominal_descriptions[column]).lower()
        if "ecg" not in description and "myocardial infarction" not in description:
            raise ValueError(f"Unexpected dictionary semantics for nominal feature {column}.")
        values = set(X[column].dropna().unique())
        if not values.issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"Unexpected ECG morphology codes in {column}: {values}")
    if not set(X["SEX"].dropna().unique()).issubset({0, 1}):
        raise ValueError("SEX has unexpected categories; review CatBoost categorical handling.")
    other = [column for column in X.columns if column not in CONTINUOUS + ORDINAL + CATEGORICAL]
    for column in other:
        values = set(X[column].dropna().unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"Undeclared non-binary numeric feature {column}: {values}")
    return other


def build_paired(cat_folds: pd.DataFrame, references: list[pd.DataFrame]) -> pd.DataFrame:
    comparator_parts = []
    for reference in references:
        comparator_parts.append(reference[reference.model.isin(COMPARATORS)])
    comparator = pd.concat(comparator_parts, ignore_index=True)
    expected_rows = len(TARGETS) * OUTER_FOLDS * len(COMPARATORS)
    if len(comparator) != expected_rows or comparator.duplicated(["target", "model", "fold"]).any():
        raise ValueError("Reference fold metrics do not provide one row per comparator/target/fold.")
    cat = cat_folds[["target", "fold", *PRIMARY]].rename(
        columns={metric: f"catboost_{metric}" for metric in PRIMARY})
    base = comparator[["target", "model", "fold", *PRIMARY,
                       "n_validation", "validation_positive_cases", "validation_negative_cases"]]
    paired = base.merge(cat, on=["target", "fold"], validate="many_to_one", how="left")
    cat_signature = cat_folds[["target", "fold", "n_validation",
                               "validation_positive_cases", "validation_negative_cases"]]
    paired = paired.merge(cat_signature, on=["target", "fold"], validate="many_to_one",
                          suffixes=("_reference", "_catboost"))
    for column in ["n_validation", "validation_positive_cases", "validation_negative_cases"]:
        if not paired[f"{column}_reference"].eq(paired[f"{column}_catboost"]).all():
            raise ValueError(f"Paired fold validation signature differs for {column}.")
        paired = paired.drop(columns=[f"{column}_reference", f"{column}_catboost"])
    for metric in PRIMARY:
        diff_col = f"catboost_minus_comparator_{metric}"
        paired[diff_col] = paired[f"catboost_{metric}"] - paired[metric]
        if metric == "brier_score":
            paired[f"catboost_beats_comparator_{metric}"] = paired[diff_col] < 0
        else:
            paired[f"catboost_beats_comparator_{metric}"] = paired[diff_col] > 0
        paired[f"tie_{metric}"] = np.isclose(paired[diff_col], 0.0, atol=1e-12)
    paired = paired.rename(columns={m: f"comparator_{m}" for m in PRIMARY})
    return paired.sort_values(["target", "model", "fold"]).reset_index(drop=True)


def plot_comparison(comparison: pd.DataFrame) -> None:
    metrics = [("average_precision", "Average precision (higher is better)"),
               ("auroc", "AUROC (higher is better)"),
               ("brier_score", "Brier score (lower is better)")]
    colors = ["#A5A5A5", "#4472C4", "#70AD47", "#ED7D31", "#70AD47", "#ED7D31", "#8064A2"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, title) in zip(axes, metrics):
        means = comparison.pivot(index="target", columns="model", values=f"{metric}_mean").reindex(
            index=TARGETS, columns=ALL_MODELS)
        stds = comparison.pivot(index="target", columns="model", values=f"{metric}_std").reindex(
            index=TARGETS, columns=ALL_MODELS)
        means.plot(kind="bar", ax=ax, yerr=stds, capsize=2, color=colors, width=.82,
                   legend=False, rot=0)
        ax.set_title(title, fontsize=9)
        ax.set_xticklabels(["Death", "ZSN", "FIBR_PREDS"])
        ax.grid(axis="y", alpha=.25)
        ax.set_xlabel("")
    axes[-1].legend(ALL_MODELS, frameon=False, fontsize=7, loc="best")
    fig.suptitle("Stage 6 development CV: CatBoost and existing models")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def report_text(target_summary: pd.DataFrame, comparison: pd.DataFrame,
                categorical_missing: dict, numeric_missing: dict,
                fold_checks: int) -> str:
    lines = [
        "# Stage 6 — CatBoost development-CV experiment", "",
        "## Method", "",
        f"The experiment uses only the Stage 1 development split ({int(target_summary.dev_partition_rows.iloc[0])} rows; target-specific usable sample sizes are listed below), seed {SEED}, and target-specific shuffled five-fold stratified outer CV. Each outer training partition uses shuffled three-fold stratified inner CV and a {SEARCH_ITER}-candidate randomized search scored by average precision (AP). The selected estimator is refit on that complete outer training partition; its untouched outer validation fold is scored once. `RandomizedSearchCV` raises on fitting errors, and captured Python warnings are retained in the fold/tuning CSVs.", "",
        "The searched parameter values are `iterations` ∈ {250, 400, 600}, `depth` ∈ {4, 5, 6}, `learning_rate` ∈ {0.03, 0.05, 0.08}, `l2_leaf_reg` ∈ {3, 5, 7, 9}, `random_strength` ∈ {0.5, 1.0}, and `border_count` ∈ {64, 128}. Four candidates are sampled from these ranges for each outer fold; the full Cartesian grid is not exhaustively evaluated.", "",
        "The outer splitter is the same deterministic `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` used in Stages 2–4. Saved previous-stage fold metric signatures were compared for train/validation sizes, validation class counts, and training prevalence. Those summaries do not contain record-level fold membership, so exact historical membership cannot be directly proven; current folds are reproducibly reconstructed from the same development row order and split definition.", "",
        "## Categorical and missing-value handling", "",
        "`SEX` is nominal. `ant_im`, `lat_im`, `inf_im`, and `post_im` are ECG morphology codes 0–4 whose labels describe distinct patterns without an established numeric distance; these five columns are passed to CatBoost as native categorical features. Other binary presence/absence indicators remain numeric 0/1, documented ordinal codes remain numeric, and continuous measurements remain numeric. The classifier uses no class weights or resampling, preserving natural training prevalence.", "",
        f"Numerical missing values are passed through unchanged to CatBoost's native numerical missing-value handling. Categorical missing values are converted to the fixed string `{MISSING_CATEGORY}` in the pipeline transformer; this is deterministic representation, not a statistic learned across folds. In the development predictors, categorical missing counts are {categorical_missing}; numeric missing counts are {numeric_missing}.", "",
        "## Results", "",
        "All reported scores are outer-fold results. AP is the primary ranking metric; AUROC and Brier are secondary probability/ranking metrics. Precision, recall, specificity, F1, accuracy, and confusion counts use the fixed 0.5 threshold and are not threshold selection or calibration.", "",
        "| Target | Model | AP mean ± SD | AUROC mean ± SD | Brier mean ± SD |", "|---|---|---:|---:|---:|",
    ]
    report_order = comparison.assign(
        _target_order=comparison["target"].map({name: i for i, name in enumerate(TARGETS)}),
        _model_order=comparison["model"].map({name: i for i, name in enumerate(ALL_MODELS)}),
    ).sort_values(["_target_order", "_model_order"])
    for row in report_order.drop(columns=["_target_order", "_model_order"]).to_dict("records"):
        lines.append(
            f"| {row['target']} | {row['model']} | "
            f"{row['average_precision_mean']:.3f} ± {row['average_precision_std']:.3f} | "
            f"{row['auroc_mean']:.3f} ± {row['auroc_std']:.3f} | "
            f"{row['brier_score_mean']:.3f} ± {row['brier_score_std']:.3f} |"
        )
    lines += [
        "", "## Per-target comparison", "",
        "For each target and metric, CatBoost's five-fold mean is compared with the best mean among the six existing models. AP and AUROC are better when higher; Brier is better when lower. These are descriptive comparisons and do not establish statistical significance.", "",
    ]
    metric_specs = [
        ("average_precision", "AP", True),
        ("auroc", "AUROC", True),
        ("brier_score", "Brier", False),
    ]
    for target in TARGETS:
        target_comparison = comparison[comparison["target"].astype(str) == target]
        catboost_row = target_comparison[target_comparison["model"].astype(str) == "CatBoost tuned"].iloc[0]
        existing = target_comparison[target_comparison["model"].astype(str) != "CatBoost tuned"]
        lines.append(f"### {target}")
        for metric, label, higher_is_better in metric_specs:
            mean_column = f"{metric}_mean"
            best_index = existing[mean_column].idxmax() if higher_is_better else existing[mean_column].idxmin()
            comparator = existing.loc[best_index]
            cat_value = float(catboost_row[mean_column])
            comparator_value = float(comparator[mean_column])
            if np.isclose(cat_value, comparator_value, rtol=0, atol=1e-12):
                result = "matched"
            elif (cat_value > comparator_value) == higher_is_better:
                result = "led"
            else:
                result = "did not lead"
            lines.append(
                f"- {label}: CatBoost {cat_value:.3f}; best existing comparator "
                f"{str(comparator['model'])} {comparator_value:.3f}; CatBoost {result}."
            )
        lines.append("")
    lines += [
        "", f"Fold signatures checked against prior stages: {fold_checks} individual values matched.", "",
        "## Comparison and limitations", "",
        "`paired_comparison.csv` gives target/fold-level differences against the Dummy, Logistic Regression, Random Forest, and Extra Trees results. It also records wins and ties for AP, AUROC, and Brier. Five folds are too few to establish statistical significance; differences are descriptive. CatBoost is not automatically declared best. Selection for the later final model stage should consider per-target evidence on development data and calibration separately.", "",
        "This stage performs no probability calibration, threshold optimization, or held-out evaluation. Missing labels are excluded separately for each target. The small four-candidate search is an initial compute-bounded experiment and does not explore the full CatBoost parameter space.", "",
        f"Environment: scikit-learn `{sklearn.__version__}`; CatBoost `{catboost.__version__}`. Run `.venv/bin/python mi_stage6_catboost.py`.", "",
        "## Output files", "",
        "- `fold_level_metrics.csv`: CatBoost fold metrics, threshold confusion counts, probabilities, and warnings.",
        "- `target_summary.csv`: target label availability, prevalence, and fold-check counts.",
        "- `aggregate_metrics.csv`: mean and sample standard deviation by target.",
        "- `model_comparison.csv`: CatBoost plus six existing models.",
        "- `paired_comparison.csv`: fold-level comparisons against existing models.",
        "- `tuning_results.csv`: best parameters and inner-CV AP by target/fold.",
        "- `oof_predictions.csv`: one development-only outer-fold prediction per eligible record/target, with ID for traceability only.",
        "- `model_comparison.png`: AP, AUROC, and Brier comparison plot.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUT.exists():
        existing = [name for name in OUTPUT_NAMES if (OUT / name).exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite Stage 6 artifacts: {existing}")
    for path in [
        STAGE1 / "X_train.csv", STAGE1 / "y_train.csv", STAGE1 / "train_record_ids.csv",
        STAGE1 / "column_dictionary.csv",
        STAGE2 / "fold_level_metrics.csv", STAGE2 / "aggregate_metrics.csv",
        STAGE3 / "fold_level_metrics.csv", STAGE3 / "aggregate_metrics.csv",
        STAGE4 / "fold_level_metrics.csv", STAGE4 / "aggregate_metrics.csv",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    X = pd.read_csv(STAGE1 / "X_train.csv")
    y_all = pd.read_csv(STAGE1 / "y_train.csv")
    ids = pd.read_csv(STAGE1 / "train_record_ids.csv")
    dictionary = pd.read_csv(STAGE1 / "column_dictionary.csv")
    other_binary = validate_inputs(X, y_all, ids, dictionary)
    id_column = ids.columns[0]
    stage2_folds = pd.read_csv(STAGE2 / "fold_level_metrics.csv")
    stage3_folds = pd.read_csv(STAGE3 / "fold_level_metrics.csv")
    stage4_folds = pd.read_csv(STAGE4 / "fold_level_metrics.csv")
    stage2_aggregate = pd.read_csv(STAGE2 / "aggregate_metrics.csv")
    stage3_aggregate = pd.read_csv(STAGE3 / "aggregate_metrics.csv")
    stage4_aggregate = pd.read_csv(STAGE4 / "aggregate_metrics.csv")

    OUT.mkdir(exist_ok=True)
    search_space = {
        "classifier__iterations": [250, 400, 600],
        "classifier__depth": [4, 5, 6],
        "classifier__learning_rate": [0.03, 0.05, 0.08],
        "classifier__l2_leaf_reg": [3, 5, 7, 9],
        "classifier__random_strength": [0.5, 1.0],
        "classifier__border_count": [64, 128],
    }
    fold_rows, tuning_rows, oof_rows, target_rows = [], [], [], []
    all_fold_checks = 0
    references = [stage2_folds, stage3_folds, stage4_folds]
    reference_models = {
        "Dummy prior prevalence": stage2_folds,
        "Logistic regression": stage2_folds,
        "Random Forest preliminary": stage3_folds,
        "Random Forest tuned": stage3_folds,
        "Extra Trees preliminary": stage4_folds,
        "Extra Trees tuned": stage4_folds,
    }

    for target in TARGETS:
        y = y_all[target].dropna().astype(int)
        if set(y.unique()) != {0, 1} or y.value_counts().min() < OUTER_FOLDS:
            raise ValueError(f"{target} labels cannot support five-fold stratified CV.")
        X_target = X.loc[y.index]
        target_ids = ids.loc[y.index, id_column]
        splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X_target, y))
        for model, reference in reference_models.items():
            all_fold_checks += assert_reference_folds(reference, target, model, y, splits)
        target_rows.append({
            "target": target,
            "dev_partition_rows": len(X),
            "usable_training_observations": len(y),
            "missing_training_labels": int(y_all[target].isna().sum()),
            "positive_cases": int(y.sum()),
            "negative_cases": int(len(y) - y.sum()),
            "positive_prevalence": float(y.mean()),
            "outer_folds": OUTER_FOLDS,
            "reference_fold_signature_values_checked": len(reference_models) * OUTER_FOLDS * len(FOLD_SIGNATURE),
        })

        for fold, (train_pos, valid_pos) in enumerate(splits, start=1):
            X_train, X_valid = X_target.iloc[train_pos], X_target.iloc[valid_pos]
            y_train, y_valid = y.iloc[train_pos].to_numpy(), y.iloc[valid_pos].to_numpy()
            train_prevalence = float(y_train.mean())
            inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold)
            search = RandomizedSearchCV(
                make_pipeline(), param_distributions=search_space,
                n_iter=SEARCH_ITER, scoring="average_precision", cv=inner,
                random_state=SEED + fold, n_jobs=1, refit=True,
                error_score="raise", return_train_score=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # Pass CatBoost's categorical column names as fit metadata so
                # sklearn can safely clone the estimator for every CV fold.
                search.fit(X_train, y_train, classifier__cat_features=CATEGORICAL)
                best = search.best_estimator_.named_steps["classifier"]
                positive_column = int(np.flatnonzero(best.classes_ == 1)[0])
                probability = search.predict_proba(X_valid)[:, positive_column]
            check_probability(probability, best.classes_, target, fold)
            best_parameters = {
                key.replace("classifier__", ""): value
                for key, value in search.best_params_.items()
            }
            warnings_found = warning_text(caught)
            fold_row = {
                "target": target, "model": "CatBoost tuned", "fold": fold,
                "n_train": len(train_pos), "n_validation": len(valid_pos),
                "validation_positive_cases": int(y_valid.sum()),
                "validation_negative_cases": int(len(y_valid) - y_valid.sum()),
                "training_prevalence": train_prevalence,
                "probability_check_passed": True,
                "probability_min": float(probability.min()),
                "probability_max": float(probability.max()),
                "fit_warnings": warnings_found,
                "best_parameters": str(best_parameters),
            }
            fold_row.update(score(y_valid, probability, train_prevalence))
            fold_rows.append(fold_row)
            tuning_rows.append({
                "target": target, "outer_fold": fold,
                "inner_cv_best_average_precision": float(search.best_score_),
                "best_parameters": str(best_parameters),
                "candidates_evaluated": SEARCH_ITER,
                "inner_cv_folds": INNER_FOLDS,
                "search_n_jobs": 1,
                "search_warnings": warnings_found,
            })
            for record_id, true_label, predicted_probability in zip(
                    target_ids.iloc[valid_pos].tolist(), y_valid, probability):
                oof_rows.append({
                    "target": target, "ID": record_id, "outer_fold": fold,
                    "true_label": int(true_label),
                    "predicted_probability": float(predicted_probability),
                    "predicted_class_at_0_5": int(predicted_probability >= 0.5),
                })

    cat_folds = pd.DataFrame(fold_rows)
    cat_aggregate = aggregate(cat_folds)
    summaries = [
        (stage2_aggregate, {"Dummy prior prevalence", "Logistic regression"}),
        (stage3_aggregate, {"Random Forest preliminary", "Random Forest tuned"}),
        (stage4_aggregate, {"Extra Trees preliminary", "Extra Trees tuned"}),
    ]
    previous = []
    for table, names in summaries:
        subset = table[table.target.isin(TARGETS) & table.model.isin(names)]
        if len(subset) != len(TARGETS) * len(names):
            raise ValueError(f"A previous aggregate is missing expected models: {sorted(names)}")
        previous.append(subset)
    comparison = pd.concat([*previous, cat_aggregate], ignore_index=True)
    if set(comparison.model.unique()) != set(ALL_MODELS):
        raise ValueError("Aggregate model comparison does not include all seven models.")
    if comparison.duplicated(["target", "model"]).any() or len(comparison) != len(TARGETS) * len(ALL_MODELS):
        raise ValueError("Aggregate model comparison has missing or duplicate rows.")
    comparison["model"] = pd.Categorical(comparison.model, ALL_MODELS, ordered=True)
    comparison["target"] = pd.Categorical(comparison.target, TARGETS, ordered=True)
    comparison = comparison.sort_values(["target", "model"]).reset_index(drop=True)
    paired = build_paired(cat_folds, references)
    oof = pd.DataFrame(oof_rows)
    if len(oof) != sum(len(y_all[target].dropna()) for target in TARGETS):
        raise ValueError("OOF predictions do not contain exactly one row per observed target label.")
    if oof.duplicated(["target", "ID"]).any():
        raise ValueError("Duplicate OOF prediction for a target/record ID.")
    for target in TARGETS:
        expected_ids = set(ids.loc[y_all[target].notna(), id_column].tolist())
        target_oof = oof.loc[oof.target == target]
        if len(target_oof) != int(y_all[target].notna().sum()) or set(target_oof["ID"]) != expected_ids:
            raise ValueError(f"{target}: OOF predictions do not cover every eligible development ID exactly once.")
    probabilities = oof["predicted_probability"].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("OOF probabilities are missing, infinite, or outside [0, 1].")
    if not oof["predicted_class_at_0_5"].eq((probabilities >= 0.5).astype(int)).all():
        raise ValueError("OOF class labels do not match the fixed 0.5 threshold.")

    cat_missing = {column: int(X[column].isna().sum()) for column in CATEGORICAL}
    numeric_columns = CONTINUOUS + ORDINAL + other_binary
    numeric_missing = {column: int(X[column].isna().sum()) for column in numeric_columns if X[column].isna().any()}
    OUT.mkdir(exist_ok=True)
    cat_folds.to_csv(OUT / "fold_level_metrics.csv", index=False)
    pd.DataFrame(target_rows).to_csv(OUT / "target_summary.csv", index=False)
    cat_aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    paired.to_csv(OUT / "paired_comparison.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "tuning_results.csv", index=False)
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    plot_comparison(comparison)
    report = report_text(pd.DataFrame(target_rows), comparison, cat_missing,
                         numeric_missing, all_fold_checks)
    (OUT / "stage6_report.md").write_text(report, encoding="utf-8")
    print(f"Reference fold signature values checked: {all_fold_checks}")
    print(comparison[["target", "model", "average_precision_mean", "auroc_mean",
                      "brier_score_mean"]].to_string(index=False))
    print(f"OOF predictions: {len(oof)} rows; all checks passed.")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
