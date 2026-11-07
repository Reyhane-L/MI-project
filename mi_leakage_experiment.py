from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
STAGE1_DIR = ROOT / "stage1_outputs"
STAGE2_SOURCE = ROOT / "mi_stage2_baselines.py"
DEFAULT_OUTPUT_DIR = ROOT / "leakage_experiment_outputs"
X_PATH = STAGE1_DIR / "X_train.csv"
Y_PATH = STAGE1_DIR / "y_train.csv"
RANDOM_STATE = 42
N_SPLITS = 5
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]
METRICS = ["average_precision", "auroc", "brier_score"]
METHOD_GLOBAL = "global_imputation_before_cv"
METHOD_FOLD_LOCAL = "fold_local_pipeline"
FOLD_COLUMNS = [
    "target", "method", "fold", "n_train", "n_validation",
    "validation_positive_cases", "validation_negative_cases",
    "training_prevalence", "average_precision", "auroc", "brier_score",
    "train_index_sha256", "validation_index_sha256", "fit_warnings",
]
AGGREGATE_COLUMNS = [
    "target", "method", "folds",
    "average_precision_mean", "average_precision_std",
    "auroc_mean", "auroc_std",
    "brier_score_mean", "brier_score_std",
]
TARGET_SUMMARY_COLUMNS = [
    "target", "development_rows", "observed_label_rows", "missing_label_rows",
    "positive_cases", "negative_cases", "positive_prevalence",
]


@dataclass(frozen=True)
class Stage2API:
    targets: list[str]
    random_state: int
    n_splits: int
    continuous_features: list[str]
    ordinal_features: list[str]
    nominal_features: list[str]
    build_pipeline: Any


def load_stage2_api(source_path: Path = STAGE2_SOURCE) -> Stage2API:
    """Load Stage 2's literal feature groups and pipeline without importing plots."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    values: dict[str, Any] = {}
    wanted = {
        "TARGETS", "RANDOM_STATE", "N_SPLITS", "CONTINUOUS_FEATURES",
        "ORDINAL_FEATURES", "NOMINAL_FEATURES",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            for name in names:
                if name in wanted:
                    values[name] = ast.literal_eval(node.value)

    function = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "build_pipeline"),
        None,
    )
    if function is None or wanted - values.keys():
        raise ValueError(f"Could not load Stage 2 preprocessing definitions from {source_path}.")

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer as Stage2SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    namespace = {
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": Stage2SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "CONTINUOUS_FEATURES": values["CONTINUOUS_FEATURES"],
        "ORDINAL_FEATURES": values["ORDINAL_FEATURES"],
        "NOMINAL_FEATURES": values["NOMINAL_FEATURES"],
        "RANDOM_STATE": values["RANDOM_STATE"],
    }
    isolated_module = ast.Module(body=[function], type_ignores=[])
    exec(compile(isolated_module, str(source_path), "exec"), namespace)
    return Stage2API(
        targets=values["TARGETS"],
        random_state=values["RANDOM_STATE"],
        n_splits=values["N_SPLITS"],
        continuous_features=values["CONTINUOUS_FEATURES"],
        ordinal_features=values["ORDINAL_FEATURES"],
        nominal_features=values["NOMINAL_FEATURES"],
        build_pipeline=namespace["build_pipeline"],
    )


def validate_inputs(X: pd.DataFrame, y_all: pd.DataFrame, stage2: Stage2API) -> list[str]:
    if not isinstance(X, pd.DataFrame) or not isinstance(y_all, pd.DataFrame):
        raise ValueError("Predictors and outcomes must be pandas DataFrames.")
    if X.empty or X.shape[1] == 0:
        raise ValueError("Development predictors must contain rows and features.")
    if len(X) != len(y_all):
        raise ValueError("Development predictor and outcome row counts do not match.")
    if X.columns.duplicated().any() or not X.columns.is_unique:
        raise ValueError("Duplicate predictor columns are not allowed.")
    if y_all.columns.duplicated().any() or not y_all.columns.is_unique:
        raise ValueError("Duplicate outcome columns are not allowed.")
    if X.index.has_duplicates or y_all.index.has_duplicates or not X.index.equals(y_all.index):
        raise ValueError("Development predictors and outcomes must have the same unique row index.")

    forbidden = {
        "ID", "RECORD_ID", "RECORDID", "LET_IS", *stage2.targets,
    }
    leaked = sorted(forbidden.intersection(X.columns))
    if leaked:
        raise ValueError(f"Forbidden ID/outcome predictor columns: {leaked}.")
    if not set(stage2.targets).issubset(y_all.columns):
        raise ValueError(f"Missing required development outcomes: {sorted(set(stage2.targets) - set(y_all.columns))}.")

    declared = (
        stage2.continuous_features
        + stage2.ordinal_features
        + stage2.nominal_features
    )
    if len(declared) != len(set(declared)):
        raise ValueError("Stage 2 feature groups overlap.")
    missing_groups = sorted(set(declared) - set(X.columns))
    if missing_groups:
        raise ValueError(f"Stage 2 feature groups are absent from predictors: {missing_groups}.")
    binary_features = [name for name in X.columns if name not in declared]
    if set(declared).union(binary_features) != set(X.columns):
        raise ValueError("Stage 2 feature groups do not partition the predictor schema.")

    for name in X.columns:
        values = pd.to_numeric(X[name], errors="coerce")
        nonnumeric = X[name].notna() & values.isna()
        if nonnumeric.any():
            raise ValueError(f"Predictor {name} contains nonnumeric observed values.")
        observed_values = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed_values).all():
            raise ValueError(f"Predictor {name} contains infinite values.")
        if name in binary_features and not set(values.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Predictor {name} is expected to contain binary 0/1 values.")
    return binary_features


def validate_target(target: str, series: pd.Series, n_splits: int) -> pd.Series:
    observed = series.dropna()
    if observed.empty:
        raise ValueError(f"{target} has no observed development labels.")
    values = set(observed.unique())
    if values != {0, 1}:
        raise ValueError(f"{target} must contain observed binary 0/1 labels; got {sorted(values)}.")
    observed = observed.astype("int8")
    counts = observed.value_counts()
    if len(counts) != 2 or int(counts.min()) < n_splits:
        raise ValueError(
            f"{target} needs at least {n_splits} examples in each class for stratified CV."
        )
    return observed


def fit_global_imputers(
    X: pd.DataFrame, stage2: Stage2API, binary_features: list[str],
) -> dict[str, SimpleImputer]:
    """Fit the Stage 2 median/mode strategies on all development predictors."""
    imputers: dict[str, SimpleImputer] = {}
    groups = {
        "continuous": (stage2.continuous_features, "median"),
        "ordered_codes": (stage2.ordinal_features, "most_frequent"),
        "binary": (binary_features, "most_frequent"),
        "ecg_patterns": (stage2.nominal_features, "most_frequent"),
    }
    for name, (columns, strategy) in groups.items():
        if columns:
            imputers[name] = SimpleImputer(strategy=strategy).fit(X[columns])
    return imputers


def apply_global_imputers(
    X: pd.DataFrame, stage2: Stage2API, binary_features: list[str],
    imputers: dict[str, SimpleImputer],
) -> pd.DataFrame:
    imputed = X.copy()
    groups = {
        "continuous": stage2.continuous_features,
        "ordered_codes": stage2.ordinal_features,
        "binary": binary_features,
        "ecg_patterns": stage2.nominal_features,
    }
    for name, columns in groups.items():
        if columns:
            transformed = imputers[name].transform(X[columns])
            if transformed.shape != (len(X), len(columns)):
                raise ValueError(
                    f"Global {name} imputer changed the feature shape; refusing to continue."
                )
            imputed.loc[:, columns] = transformed
    return imputed


def _index_digest(positions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(positions, dtype=np.int64).tobytes()).hexdigest()


def _fit_fold(
    stage2: Stage2API,
    feature_columns: list[str],
    binary_features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
) -> tuple[np.ndarray, list[str], Any]:
    model = stage2.build_pipeline(feature_columns, binary_features)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_valid)[:, 1]
    messages = [f"{item.category.__name__}: {item.message}" for item in captured]
    return probability, messages, model


def _aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, method), subset in folds.groupby(["target", "method"], sort=False):
        if len(subset) != N_SPLITS:
            raise ValueError(f"{target}/{method} has {len(subset)} folds; expected {N_SPLITS}.")
        row: dict[str, Any] = {"target": target, "method": method, "folds": len(subset)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)


def run_experiment(
    X: pd.DataFrame,
    y_all: pd.DataFrame,
    stage2: Stage2API | None = None,
    command: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stage2 = stage2 or load_stage2_api()
    if stage2.targets != TARGETS or stage2.n_splits != N_SPLITS or stage2.random_state != RANDOM_STATE:
        raise ValueError("Stage 2 target/fold/seed settings differ from the requested experiment.")
    binary_features = validate_inputs(X, y_all, stage2)

    global_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        global_imputers = fit_global_imputers(X, stage2, binary_features)
        X_global = apply_global_imputers(X, stage2, binary_features, global_imputers)
    global_warnings.extend(
        f"{item.category.__name__}: {item.message}" for item in captured
    )

    target_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    index_checks: dict[str, list[tuple[str, str]]] = {}
    for target in stage2.targets:
        observed_y = validate_target(target, y_all[target], stage2.n_splits)
        X_target = X.loc[observed_y.index]
        X_global_target = X_global.loc[observed_y.index]
        counts = observed_y.value_counts()
        target_rows.append({
            "target": target,
            "development_rows": len(X),
            "observed_label_rows": len(observed_y),
            "missing_label_rows": int(y_all[target].isna().sum()),
            "positive_cases": int(counts.loc[1]),
            "negative_cases": int(counts.loc[0]),
            "positive_prevalence": float(observed_y.mean()),
        })

        splitter = StratifiedKFold(
            n_splits=stage2.n_splits,
            shuffle=True,
            random_state=stage2.random_state,
        )
        splits = list(splitter.split(X_target, observed_y))
        if len(splits) != stage2.n_splits:
            raise ValueError(f"{target} materialized {len(splits)} folds, expected {stage2.n_splits}.")
        validation_counts = np.zeros(len(observed_y), dtype=np.int8)
        target_checks = []
        for fold_number, (train_positions, valid_positions) in enumerate(splits, start=1):
            if np.intersect1d(train_positions, valid_positions).size:
                raise ValueError(f"{target} fold {fold_number} has overlapping train/validation rows.")
            if len(train_positions) + len(valid_positions) != len(observed_y):
                raise ValueError(f"{target} fold {fold_number} does not partition its observed rows.")
            validation_counts[valid_positions] += 1
            y_train = observed_y.iloc[train_positions].to_numpy(dtype=np.int8)
            y_valid = observed_y.iloc[valid_positions].to_numpy(dtype=np.int8)
            if set(y_train) != {0, 1} or set(y_valid) != {0, 1}:
                raise ValueError(f"{target} fold {fold_number} lacks a class in train or validation.")

            train_digest = _index_digest(train_positions)
            valid_digest = _index_digest(valid_positions)
            target_checks.append((train_digest, valid_digest))
            for method, X_method in (
                (METHOD_FOLD_LOCAL, X_target),
                (METHOD_GLOBAL, X_global_target),
            ):
                X_train = X_method.iloc[train_positions]
                X_valid = X_method.iloc[valid_positions]
                probability, fit_warnings, _ = _fit_fold(
                    stage2, X.columns.tolist(), binary_features,
                    X_train, y_train, X_valid,
                )
                if len(probability) != len(valid_positions) or not np.isfinite(probability).all():
                    raise ValueError(
                        f"{target} fold {fold_number} {method} returned invalid probabilities."
                    )
                if ((probability < 0) | (probability > 1)).any():
                    raise ValueError(
                        f"{target} fold {fold_number} {method} returned probabilities outside [0, 1]."
                    )
                row = {
                    "target": target,
                    "method": method,
                    "fold": fold_number,
                    "n_train": len(train_positions),
                    "n_validation": len(valid_positions),
                    "validation_positive_cases": int(y_valid.sum()),
                    "validation_negative_cases": int(len(y_valid) - y_valid.sum()),
                    "training_prevalence": float(y_train.mean()),
                    "average_precision": float(average_precision_score(y_valid, probability)),
                    "auroc": float(roc_auc_score(y_valid, probability)),
                    "brier_score": float(brier_score_loss(y_valid, probability)),
                    "train_index_sha256": train_digest,
                    "validation_index_sha256": valid_digest,
                    "fit_warnings": " | ".join(fit_warnings),
                }
                fold_rows.append(row)
        if not np.all(validation_counts == 1):
            raise ValueError(f"{target} CV validation folds do not cover each observed row exactly once.")
        index_checks[target] = target_checks
        safe_checks = [
            (row["train_index_sha256"], row["validation_index_sha256"])
            for row in fold_rows
            if row["target"] == target and row["method"] == METHOD_FOLD_LOCAL
        ]
        global_checks = [
            (row["train_index_sha256"], row["validation_index_sha256"])
            for row in fold_rows
            if row["target"] == target and row["method"] == METHOD_GLOBAL
        ]
        if safe_checks != global_checks or safe_checks != target_checks:
            raise AssertionError(f"{target} methods did not use identical materialized CV indices.")

    folds = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    aggregate = _aggregate(folds)
    target_summary = pd.DataFrame(target_rows, columns=TARGET_SUMMARY_COLUMNS)
    all_fit_warnings = [
        {
            "target": row["target"],
            "method": row["method"],
            "fold": int(row["fold"]),
            "warning": warning,
        }
        for row in fold_rows
        for warning in row["fit_warnings"].split(" | ")
        if warning
    ]
    manifest = {
        "experiment": "Stage 2 logistic-regression imputation-placement comparison",
        "inputs": [str(X_PATH), str(Y_PATH)],
        "stage2_source": str(STAGE2_SOURCE),
        "test_partition_accessed": False,
        "methodology": {
            "global_method": (
                "Fit Stage 2 median/mode imputers once on the complete development "
                "predictor table before using target-specific CV folds. Validation "
                "predictor values therefore contribute to global medians/modes."
            ),
            "fold_local_method": (
                "Use the Stage 2 preprocessing/model pipeline unchanged; imputers, "
                "scaler, one-hot category discovery, and logistic regression are fit "
                "only on each training fold."
            ),
            "post_imputation_fits": (
                "For both methods, scaling, one-hot category discovery, and logistic "
                "regression are fit within each training fold."
            ),
            "interpretation": (
                "This experiment demonstrates validation-predictor leakage from global "
                "imputation; leakage can shift metrics in either direction."
            ),
            "same_folds": True,
            "split_index_digests_by_target": index_checks,
        },
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits": stage2.n_splits,
            "shuffle": True,
            "random_state": stage2.random_state,
            "target_specific_on_observed_labels": True,
        },
        "model": {
            "estimator": "LogisticRegression",
            "l1_ratio": 0.0,
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": None,
            "random_state": stage2.random_state,
            "resampling": None,
            "probabilities_preserve_natural_class_prevalence": True,
        },
        "preprocessing": {
            "continuous": "median imputation then StandardScaler",
            "ordered_codes": "most_frequent imputation",
            "binary_features": "most_frequent imputation",
            "nominal_ecg_codes": "most_frequent imputation then OneHotEncoder(handle_unknown='ignore')",
        },
        "targets": stage2.targets,
        "metrics": METRICS,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "command": command or " ".join(sys.argv),
        "global_imputation_warnings": global_warnings,
        "fold_warnings": all_fit_warnings,
        "output_schemas": {
            "fold_level_metrics.csv": FOLD_COLUMNS,
            "aggregate_metrics.csv": AGGREGATE_COLUMNS,
            "target_split_summary.csv": TARGET_SUMMARY_COLUMNS,
        },
    }
    return folds, aggregate, target_summary, manifest


def write_outputs(
    output_dir: Path,
    folds: pd.DataFrame,
    aggregate: pd.DataFrame,
    target_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    folds.to_csv(output_dir / "fold_level_metrics.csv", index=False, float_format="%.12g")
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False, float_format="%.12g")
    target_summary.to_csv(
        output_dir / "target_split_summary.csv", index=False, float_format="%.12g"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if DEFAULT_OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output directory: {DEFAULT_OUTPUT_DIR}"
        )
    for path in (X_PATH, Y_PATH, STAGE2_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    X = pd.read_csv(X_PATH)
    y_all = pd.read_csv(Y_PATH)
    folds, aggregate, target_summary, manifest = run_experiment(
        X, y_all, command=" ".join(sys.argv),
    )
    write_outputs(DEFAULT_OUTPUT_DIR, folds, aggregate, target_summary, manifest)
    print(f"Development-only imputation leakage experiment: {len(folds)} fold rows")
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    warning_count = len(manifest["global_imputation_warnings"]) + len(manifest["fold_warnings"])
    print(f"Warnings captured: {warning_count}")
    if warning_count:
        for warning in manifest["global_imputation_warnings"]:
            print(f"  global imputation: {warning}")
        for item in manifest["fold_warnings"]:
            print(
                f"  {item['target']} {item['method']} fold {item['fold']}: "
                f"{item['warning']}"
            )
    print(f"Outputs: {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
