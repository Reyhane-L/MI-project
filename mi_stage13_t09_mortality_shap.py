"""Create a development-only SHAP summary for the frozen T06 mortality model."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool


ROOT = Path(__file__).resolve().parent
DEV_X_FILE = ROOT / "stage1_outputs" / "X_train.csv"
DEV_Y_FILE = ROOT / "stage1_outputs" / "y_train.csv"
DICTIONARY_FILE = ROOT / "stage1_outputs" / "column_dictionary.csv"
BUNDLE_FILE = ROOT / "stage10_t06_non_tabpfn_outputs" / "death_binary_bundle.joblib"
MANIFEST_FILE = ROOT / "stage10_t06_non_tabpfn_outputs" / "final_model_manifest.json"
OUT = ROOT / "stage13_t09_mortality_shap_outputs"
TARGET = "DEATH_BINARY"
BACKGROUND_SIZE = 64
SEED = 42
SUMMARY_FEATURES = 20
FORBIDDEN_COLUMNS = {
    "ID", "LET_IS", "DEATH_BINARY", "ZSN", "FIBR_PREDS", "PREDS_TAH",
    "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK", "OTEK_LANC", "RAZRIV",
    "R_AB_1_n", "R_AB_2_n", "R_AB_3_n", "NA_R_1_n", "NA_R_2_n",
    "NA_R_3_n", "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_development_features(X: pd.DataFrame, feature_columns: list[str]) -> None:
    if X.empty or X.columns.has_duplicates:
        raise ValueError("Development predictors must be nonempty with unique column names.")
    if X.columns.tolist() != feature_columns:
        raise ValueError("Development feature names/order differ from the frozen bundle.")
    forbidden = FORBIDDEN_COLUMNS.intersection(X.columns)
    if forbidden:
        raise ValueError(f"Development predictors contain forbidden columns: {sorted(forbidden)}")


def summarize_shap_values(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Summarize CatBoost's per-row SHAP matrix (last base-value column excluded)."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names) + 1 or matrix.shape[0] == 0:
        raise ValueError("SHAP matrix must have one column per feature plus a base-value column.")
    if not np.isfinite(matrix).all() or len(set(feature_names)) != len(feature_names):
        raise ValueError("SHAP values and feature names must be finite and unique.")
    contributions = matrix[:, :-1]
    result = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": np.mean(np.abs(contributions), axis=0),
        "mean_signed_shap": np.mean(contributions, axis=0),
    })
    return result.sort_values("mean_absolute_shap", ascending=False, kind="stable").reset_index(drop=True)


def write_summary_plot(values: np.ndarray, feature_names: list[str], path: Path,
                       *, max_features: int = SUMMARY_FEATURES, seed: int = SEED) -> None:
    """Write a deterministic SHAP-value distribution summary (beeswarm-style)."""
    matrix = np.asarray(values, dtype=float)
    summary = summarize_shap_values(matrix, feature_names)
    selected = summary.head(min(max_features, len(summary)))
    contributions = matrix[:, :-1]
    feature_index = {name: index for index, name in enumerate(feature_names)}
    rng = np.random.default_rng(seed)
    fig, axis = plt.subplots(figsize=(9, max(5, 0.34 * len(selected) + 1.5)))
    for y_index, row in enumerate(selected.itertuples(index=False)):
        column = contributions[:, feature_index[row.feature]]
        jitter = rng.uniform(-0.28, 0.28, size=len(column))
        axis.scatter(column, y_index + jitter, s=11, alpha=0.48, color="#2878a8", linewidths=0)
    axis.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_yticks(np.arange(len(selected)), selected["feature"])
    axis.invert_yaxis()
    axis.set_xlabel("SHAP value (CatBoost raw model output scale)")
    axis.set_title("Frozen DEATH_BINARY CatBoost: development SHAP summary")
    axis.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_explanation(summary: pd.DataFrame, dictionary: pd.DataFrame,
                      background_rows: int) -> str:
    required_dictionary_columns = {"variable", "description"}
    if not required_dictionary_columns.issubset(dictionary.columns):
        raise ValueError("Column dictionary must contain variable and description columns.")
    descriptions = dictionary.drop_duplicates("variable").set_index("variable")["description"].to_dict()
    top = summary.head(10)
    if top.empty or top["feature"].map(descriptions).isna().any():
        raise ValueError("Top SHAP features must have definitions in the Stage 1 column dictionary.")
    explanation = [
        "# Frozen mortality model: development SHAP summary", "",
        "This summary explains the frozen T06 `CatBoost tuned` base model for `DEATH_BINARY`. "
        "Native CatBoost Independent TreeSHAP values were computed on the development predictors in `stage1_outputs/X_train.csv`; "
        f"the reference sample contains {background_rows} development rows selected with seed {SEED}. "
        "The frozen Platt calibrator and threshold were not refit or applied to the explanations.", "",
        "## Features with largest average absolute SHAP values", "",
        "| Rank | Feature | Meaning in the Stage 1 dictionary | Mean absolute SHAP | Mean signed SHAP |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        description = str(descriptions[row.feature]).replace("|", "\\|")
        explanation.append(
            f"| {rank} | {row.feature} | {description} | "
            f"{row.mean_absolute_shap:.6g} | {row.mean_signed_shap:.6g} |"
        )
    explanation.extend([
        "", "## How to read the plot", "",
        "Each point represents one development admission. Positive SHAP values raise the frozen CatBoost model's raw death score; negative values lower it. "
        "Features are ordered by mean absolute SHAP, so the ranking measures average contribution magnitude, not a uniform direction. "
        "The top-feature rows show observations on both sides of zero, indicating heterogeneous contributions across admissions. "
        "This plot is monochrome and does not encode feature values, so it does not support claims about which feature values raise or lower the score. "
        "The model's Platt calibration is post-hoc and was not applied to these raw-score attributions. "
        "These are model associations, not causal effects. All explained and reference rows come from development data.", "",
    ])
    return "\n".join(explanation)


def _load_frozen_bundle() -> tuple[object, dict]:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("stage") != "T06" or manifest.get("freeze_revision") != "non_tabpfn_v1":
        raise ValueError("T06 manifest does not identify the active non_tabpfn_v1 freeze.")
    model = manifest.get("models", {}).get(TARGET, {})
    if model.get("model") != "CatBoost tuned":
        raise ValueError("Frozen DEATH_BINARY model is not CatBoost tuned.")
    calibration = manifest.get("calibration_parameters", {}).get(TARGET, {})
    if calibration.get("method") != "Platt":
        raise ValueError("Frozen DEATH_BINARY calibrator is not Platt.")
    bundle = joblib.load(BUNDLE_FILE)
    if (getattr(bundle, "target", None) != TARGET
            or getattr(bundle, "model_name", None) != "CatBoost tuned"
            or not isinstance(getattr(bundle, "metadata", None), dict)
            or bundle.metadata.get("calibration_method") != "Platt"
            or bundle.metadata.get("calibration_parameters") != calibration):
        raise ValueError("Frozen mortality bundle does not match the active T06 manifest.")
    estimator = getattr(bundle, "estimator", None)
    if not hasattr(estimator, "named_steps") or "catboost_frame" not in estimator.named_steps or "classifier" not in estimator.named_steps:
        raise ValueError("Frozen mortality estimator is not the expected CatBoost pipeline.")
    if estimator.named_steps["classifier"].__class__.__name__ != "CatBoostClassifier":
        raise ValueError("Frozen mortality pipeline classifier is not CatBoostClassifier.")
    return bundle, manifest


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing T09 SHAP outputs: {OUT}")
    missing = [str(path) for path in (DEV_X_FILE, DEV_Y_FILE, DICTIONARY_FILE, BUNDLE_FILE, MANIFEST_FILE) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required development/frozen inputs: {missing}")

    # Development labels are supplied only as CatBoost Pool metadata. CatBoost's
    # binary-classification SHAP API requires Pool labels even though SHAP does
    # not use them to fit, tune, calibrate, or select the frozen estimator.
    X = pd.read_csv(DEV_X_FILE)
    y_frame = pd.read_csv(DEV_Y_FILE)
    dictionary = pd.read_csv(DICTIONARY_FILE)
    if TARGET not in y_frame.columns or len(y_frame) != len(X):
        raise ValueError("Development labels must contain aligned DEATH_BINARY values.")
    labels = y_frame[TARGET].to_numpy()
    if pd.isna(labels).any() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Development DEATH_BINARY labels must be non-null and binary.")
    bundle, manifest = _load_frozen_bundle()
    validate_development_features(X, bundle.feature_columns)

    transform = bundle.estimator.named_steps["catboost_frame"]
    classifier = bundle.estimator.named_steps["classifier"]
    transformed = transform.transform(X.loc[:, bundle.feature_columns])
    categorical_features = list(transform.categorical_features)
    if not set(categorical_features).issubset(transformed.columns):
        raise ValueError("Frozen CatBoost categorical feature declarations do not match development data.")
    rng = np.random.default_rng(SEED)
    background_idx = np.sort(rng.choice(len(transformed), size=min(BACKGROUND_SIZE, len(transformed)), replace=False))
    explain_pool = Pool(transformed, label=labels, cat_features=categorical_features)
    background_pool = Pool(
        transformed.iloc[background_idx], label=labels[background_idx],
        cat_features=categorical_features,
    )
    # CatBoost's native Independent TreeSHAP; no fitting, calibration, or test data.
    shap_values = classifier.get_feature_importance(
        data=explain_pool, reference_data=background_pool, type="ShapValues",
    )
    shap_summary = summarize_shap_values(shap_values, bundle.feature_columns)

    OUT.mkdir(parents=True)
    shap_summary.to_csv(OUT / "mortality_shap_feature_summary.csv", index=False)
    write_summary_plot(shap_values, bundle.feature_columns, OUT / "mortality_shap_summary.png")
    explanation = build_explanation(shap_summary, dictionary, len(background_idx))
    (OUT / "mortality_shap_explanation.md").write_text(explanation, encoding="utf-8")

    try:
        packages = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "catboost", "matplotlib", "joblib")}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"Could not record package version: {error}") from error
    run_manifest = {
        "stage": "T09_mortality_SHAP",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": packages,
        "target": TARGET,
        "freeze_revision": manifest["freeze_revision"],
        "model": manifest["models"][TARGET]["model"],
        "calibration_method": "Platt (frozen; not applied or refit for SHAP)",
        "explainer": "CatBoost native Independent TreeSHAP (get_feature_importance(type='ShapValues'))",
        "explanation_scale": "CatBoost raw model output; SHAP contributions sum to raw score",
        "development_features_source": str(DEV_X_FILE.relative_to(ROOT)),
        "development_features_sha256": _sha256(DEV_X_FILE),
        "development_labels_source": str(DEV_Y_FILE.relative_to(ROOT)),
        "development_labels_sha256": _sha256(DEV_Y_FILE),
        "column_dictionary_source": str(DICTIONARY_FILE.relative_to(ROOT)),
        "column_dictionary_sha256": _sha256(DICTIONARY_FILE),
        "bundle_source": str(BUNDLE_FILE.relative_to(ROOT)),
        "bundle_sha256": _sha256(BUNDLE_FILE),
        "t06_manifest_source": str(MANIFEST_FILE.relative_to(ROOT)),
        "t06_manifest_sha256": _sha256(MANIFEST_FILE),
        "development_rows_explained": int(len(X)),
        "background_rows": int(len(background_idx)),
        "background_seed": SEED,
        "feature_order": list(bundle.feature_columns),
        "held_out_data_read": False,
        "development_labels_use": "DEATH_BINARY labels supplied only as CatBoost Pool metadata; not used for fitting or selection",
        "outputs": [
            "mortality_shap_feature_summary.csv", "mortality_shap_summary.png",
            "mortality_shap_explanation.md", "t09_mortality_shap_manifest.json",
        ],
    }
    (OUT / "t09_mortality_shap_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    run()
