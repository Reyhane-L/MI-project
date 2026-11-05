"""Stage 7: development-only FIBR_PREDS diagnostic analysis.

Run from the project root with `.venv/bin/python mi_stage7_fibr_diagnostics.py`.
Only explicit Stage 1 development artifacts and saved development CV summaries
are read. This script never enumerates Stage 1 files or refers to test files.
"""
from __future__ import annotations

import ast
from pathlib import Path
import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage7-mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import catboost
from catboost import Pool
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import mi_stage3_random_forest as stage3
import mi_stage4_extratrees as stage4
import mi_stage6_catboost as stage6

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE3 = ROOT / "stage3_outputs"
STAGE4 = ROOT / "stage4_extratrees_outputs"
STAGE6 = ROOT / "stage6_catboost_outputs"
OUT = ROOT / "stage7_fibr_diagnostics_outputs"
TARGET = "FIBR_PREDS"
SEED = 42
FOLDS = 5
CLIP = 1e-15
PERMUTATION_REPEATS = 3
CONTINUOUS = stage6.CONTINUOUS
ORDINAL = stage6.ORDINAL
CATEGORICAL = stage6.CATEGORICAL
FORBIDDEN = stage6.FORBIDDEN


def clipped_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP), labels=[0, 1]))


def get_params(tuning: pd.DataFrame, fold: int) -> dict:
    rows = tuning[(tuning.target == TARGET) & (tuning.outer_fold.astype(int) == fold)]
    if len(rows) != 1:
        raise ValueError(f"Expected one saved CatBoost parameter row for fold {fold}; got {len(rows)}")
    params = ast.literal_eval(rows.iloc[0].best_parameters)
    if not isinstance(params, dict):
        raise ValueError(f"Invalid CatBoost parameters for fold {fold}")
    return params


def make_catboost(params: dict, columns: list[str]):
    """Create the saved-parameter pipeline with categories present in this view."""
    model = stage6.make_pipeline()
    categories = tuple(c for c in CATEGORICAL if c in columns)
    model.set_params(catboost_frame__categorical_features=categories)
    model.set_params(**{f"classifier__{k}": v for k, v in params.items()})
    return model, list(categories)


def check_saved_folds(reference: pd.DataFrame, model: str, y: pd.Series,
                      splits: list[tuple[np.ndarray, np.ndarray]]) -> int:
    rows = reference[(reference.target == TARGET) & (reference.model == model)]
    if rows.duplicated("fold").any() or set(rows.fold.astype(int)) != set(range(1, FOLDS + 1)):
        raise ValueError(f"{model} does not have five unique reference folds")
    checks = 0
    for fold, (itr, iva) in enumerate(splits, 1):
        saved = rows.loc[rows.fold.astype(int) == fold].iloc[0]
        ytr, yva = y.iloc[itr].to_numpy(), y.iloc[iva].to_numpy()
        actual = {"n_train": len(itr), "n_validation": len(iva),
                  "validation_positive_cases": int(yva.sum()),
                  "validation_negative_cases": int(len(yva) - yva.sum()),
                  "training_prevalence": float(ytr.mean())}
        for key, value in actual.items():
            if key not in rows:
                raise ValueError(f"{model} saved folds lack {key}")
            good = np.isclose(float(saved[key]), value, atol=1e-12, rtol=0) if key == "training_prevalence" else int(saved[key]) == value
            if not good:
                raise ValueError(f"{model} fold {fold} {key} mismatch: {saved[key]} != {value}")
            checks += 1
    return checks


def summary_group_comparison(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []
    groups = {0: X.loc[y == 0], 1: X.loc[y == 1]}
    for col in X.columns:
        x0, x1 = groups[0][col], groups[1][col]
        miss0, miss1 = float(x0.isna().mean()), float(x1.isna().mean())
        numeric = col in CONTINUOUS
        if numeric:
            a, b = x0.dropna().astype(float), x1.dropna().astype(float)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
            smd = (b.mean() - a.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan
            q1, q3 = X[col].quantile([.25, .75]).tolist()
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out0 = int(((a < lower) | (a > upper)).sum())
            out1 = int(((b < lower) | (b > upper)).sum())
            rows.append({"feature": col, "feature_type": "continuous", "category": "",
                         "negative_n": len(a), "negative_mean": a.mean(), "negative_median": a.median(),
                         "negative_q1": a.quantile(.25), "negative_q3": a.quantile(.75),
                         "positive_n": len(b), "positive_mean": b.mean(), "positive_median": b.median(),
                         "positive_q1": b.quantile(.25), "positive_q3": b.quantile(.75),
                         "standardized_mean_difference": smd, "negative_outside_1_5_iqr": out0,
                         "positive_outside_1_5_iqr": out1, "negative_category_frequency": np.nan,
                         "positive_category_frequency": np.nan, "category_fibr_rate": np.nan,
                         "negative_missing_rate": miss0,
                         "positive_missing_rate": miss1, "missing_rate_difference_positive_minus_negative": miss1 - miss0})
        else:
            for value in sorted(X[col].dropna().unique(), key=lambda v: str(v)):
                mask = X[col].eq(value)
                n0, n1 = int((mask & (y == 0)).sum()), int((mask & (y == 1)).sum())
                rows.append({"feature": col, "feature_type": "nominal_or_code", "category": value,
                             "negative_n": np.nan, "negative_mean": np.nan, "negative_median": np.nan,
                             "negative_q1": np.nan, "negative_q3": np.nan, "positive_n": np.nan,
                             "positive_mean": np.nan, "positive_median": np.nan, "positive_q1": np.nan,
                             "positive_q3": np.nan, "standardized_mean_difference": np.nan,
                             "negative_outside_1_5_iqr": np.nan, "positive_outside_1_5_iqr": np.nan,
                             "negative_category_frequency": n0 / max(int((y == 0).sum()), 1),
                             "positive_category_frequency": n1 / max(int((y == 1).sum()), 1),
                             "category_fibr_rate": float(y[mask].mean()) if mask.any() else np.nan,
                             "negative_missing_rate": miss0,
                             "positive_missing_rate": miss1, "missing_rate_difference_positive_minus_negative": miss1 - miss0})
            missing_mask = X[col].isna()
            rows.append({"feature": col, "feature_type": "missingness", "category": "__MISSING__",
                         "negative_n": np.nan, "negative_mean": np.nan, "negative_median": np.nan,
                         "negative_q1": np.nan, "negative_q3": np.nan, "positive_n": np.nan,
                         "positive_mean": np.nan, "positive_median": np.nan, "positive_q1": np.nan,
                         "positive_q3": np.nan, "standardized_mean_difference": np.nan,
                         "negative_outside_1_5_iqr": np.nan, "positive_outside_1_5_iqr": np.nan,
                         "negative_category_frequency": miss0, "positive_category_frequency": miss1,
                         "category_fibr_rate": float(y[missing_mask].mean()) if missing_mask.any() else np.nan,
                         "negative_missing_rate": miss0, "positive_missing_rate": miss1,
                         "missing_rate_difference_positive_minus_negative": miss1 - miss0})
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage 7 output directory: {OUT}")
    explicit_inputs = [
        STAGE1 / "X_train.csv", STAGE1 / "y_train.csv", STAGE1 / "train_record_ids.csv",
        STAGE1 / "column_dictionary.csv", STAGE3 / "fold_level_metrics.csv",
        STAGE3 / "tuning_results.csv", STAGE4 / "fold_level_metrics.csv",
        STAGE6 / "fold_level_metrics.csv", STAGE6 / "tuning_results.csv",
        STAGE6 / "oof_predictions.csv",
    ]
    missing = [str(p) for p in explicit_inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Required explicit development input(s) missing: {missing}")
    Xall, yall, ids, dictionary = (pd.read_csv(STAGE1 / name) for name in
        ["X_train.csv", "y_train.csv", "train_record_ids.csv", "column_dictionary.csv"])
    stage6.validate_inputs(Xall, yall, ids, dictionary)
    # Stages 3/4 treat SEX as a numeric binary feature and reserve one-hot
    # encoding for the four ECG morphology codes.
    legacy_nominal = ["ant_im", "lat_im", "inf_im", "post_im"]
    binary = [c for c in Xall.columns if c not in CONTINUOUS + ORDINAL + legacy_nominal]
    if FORBIDDEN.intersection(Xall.columns):
        raise ValueError("ID, outcome, or post-admission feature found among predictors")
    y = yall[TARGET].dropna().astype(int)
    X = Xall.loc[y.index].copy()
    id_column = ids.columns[0]
    stage3_folds, stage3_tuning = pd.read_csv(STAGE3 / "fold_level_metrics.csv"), pd.read_csv(STAGE3 / "tuning_results.csv")
    stage4_folds = pd.read_csv(STAGE4 / "fold_level_metrics.csv")
    stage6_folds, tuning = pd.read_csv(STAGE6 / "fold_level_metrics.csv"), pd.read_csv(STAGE6 / "tuning_results.csv")
    oof = pd.read_csv(STAGE6 / "oof_predictions.csv")
    splitlist = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(X, y))
    fold_checks = 0
    fold_checks += check_saved_folds(stage3_folds, "Random Forest preliminary", y, splitlist)
    fold_checks += check_saved_folds(stage4_folds, "Extra Trees preliminary", y, splitlist)
    fold_checks += check_saved_folds(stage6_folds, "CatBoost tuned", y, splitlist)
    if oof.duplicated(["target", "ID"]).any():
        raise ValueError("Stage 6 OOF data contain duplicate target/ID rows")
    target_oof = oof[oof.target == TARGET]
    expected_ids = set(ids.loc[y.index, id_column])
    if set(target_oof.ID) != expected_ids or len(target_oof) != len(expected_ids):
        raise ValueError("Stage 6 FIBR OOF rows do not align with eligible development IDs")
    id_label = dict(zip(ids.loc[y.index, id_column], y.to_numpy()))
    if dict(zip(target_oof.ID, target_oof.true_label.astype(int))) != id_label:
        raise ValueError("Stage 6 FIBR OOF labels do not match development outcomes")
    expected_fold_by_id = {}
    for fold, (_, valid_pos) in enumerate(splitlist, 1):
        for record_id in ids.loc[y.index[valid_pos], id_column]:
            expected_fold_by_id[record_id] = fold
    observed_fold_by_id = dict(zip(target_oof.ID, target_oof.outer_fold.astype(int)))
    if observed_fold_by_id != expected_fold_by_id:
        raise ValueError("Stage 6 FIBR OOF fold assignments do not match reconstructed validation folds")

    # Descriptive outcome-group summaries are associations, not causal estimates.
    group_table = summary_group_comparison(X, y)
    missing_rows = []
    for col in X.columns:
        miss = X[col].isna()
        m0, m1 = float(miss[y == 0].mean()), float(miss[y == 1].mean())
        missing_rows.append({"feature": col, "negative_missing_n": int((miss & (y == 0)).sum()),
            "negative_n": int((y == 0).sum()), "negative_missing_rate": m0,
            "positive_missing_n": int((miss & (y == 1)).sum()), "positive_n": int((y == 1).sum()),
            "positive_missing_rate": m1, "positive_minus_negative": m1 - m0})
    missing_table = pd.DataFrame(missing_rows).sort_values("positive_minus_negative", key=lambda s: s.abs(), ascending=False)
    perm_rows, shap_rows, ablation_rows, fold_rows = [], [], [], []
    for fold, (itr, iva) in enumerate(splitlist, start=1):
        Xtr, Xva = X.iloc[itr].copy(), X.iloc[iva].copy()
        ytr, yva = y.iloc[itr].to_numpy(), y.iloc[iva].to_numpy()
        prev = float(ytr.mean())
        model_specs = [
            ("Random Forest preliminary", stage3.make_pipeline(binary)),
            ("Extra Trees preliminary", stage4.make_pipeline(binary)),
        ]
        cat, cat_categories = make_catboost(get_params(tuning, fold), list(Xtr.columns))
        model_specs.append(("CatBoost tuned", cat))
        predictions: dict[str, np.ndarray] = {}
        for model_name, estimator in model_specs:
            fit_warnings = ""
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if model_name == "CatBoost tuned":
                    estimator.fit(Xtr, ytr, classifier__cat_features=cat_categories)
                else:
                    estimator.fit(Xtr, ytr)
                p = estimator.predict_proba(Xva)[:, list(estimator.named_steps["classifier"].classes_).index(1)]
            fit_warnings = " | ".join(f"{w.category.__name__}: {w.message}" for w in caught)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError(f"Invalid probabilities: {model_name}, fold {fold}")
            predictions[model_name] = p
            fold_rows.append({"model": model_name, "fold": fold, "n_train": len(itr),
                "n_validation": len(iva), "validation_positive_cases": int(yva.sum()),
                "validation_negative_cases": int(len(yva) - yva.sum()), "training_prevalence": prev,
                "average_precision": average_precision_score(yva, p), "auroc": roc_auc_score(yva, p),
                "brier_score": brier_score_loss(yva, p), "log_loss": clipped_loss(yva, p),
                "fit_warnings": fit_warnings})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pi = permutation_importance(estimator, Xva, yva, scoring="average_precision",
                    n_repeats=PERMUTATION_REPEATS, random_state=SEED + fold, n_jobs=1)
            for j, feature in enumerate(X.columns):
                perm_rows.append({"model": model_name, "fold": fold, "feature": feature,
                    "importance_mean": float(pi.importances_mean[j]), "importance_std": float(pi.importances_std[j]),
                    "permutation_warnings": " | ".join(f"{w.category.__name__}: {w.message}" for w in caught)})

        # Independent Tree SHAP, with reference data drawn only from this fold's training rows.
        cat_model = model_specs[-1][1]
        rng = np.random.RandomState(SEED + fold)
        bg_idx = rng.choice(len(Xtr), size=min(64, len(Xtr)), replace=False)
        frame = cat_model.named_steps["catboost_frame"]
        background = frame.transform(Xtr.iloc[bg_idx])
        validation_frame = frame.transform(Xva)
        pool = Pool(validation_frame, label=yva, cat_features=cat_categories)
        ref_pool = Pool(background, cat_features=cat_categories)
        shap_values = cat_model.named_steps["classifier"].get_feature_importance(
            data=pool, reference_data=ref_pool, type="ShapValues")
        if shap_values.shape != (len(Xva), len(X.columns) + 1):
            raise ValueError(f"Unexpected CatBoost SHAP dimensions on fold {fold}: {shap_values.shape}")
        for j, feature in enumerate(X.columns):
            shap_rows.append({"fold": fold, "feature": feature,
                "mean_absolute_shap": float(np.abs(shap_values[:, j]).mean()),
                "mean_signed_shap": float(shap_values[:, j].mean()),
                "validation_rows_explained": len(Xva), "training_reference_rows": len(bg_idx)})

        # Fixed, exploratory CatBoost ablations; no candidate is selected from these outer scores.
        rates = Xtr.isna().mean()
        drop_features = rates[rates > .80].index.tolist()
        variants = [("CatBoost tuned baseline", Xtr, Xva, "")]
        kept = [c for c in X.columns if c not in drop_features]
        variants.append(("Drop >80% missing (training-fold rule)", Xtr[kept], Xva[kept], ";".join(drop_features)))
        missing_train = [c for c in X.columns if Xtr[c].isna().any()]
        xa, xv = Xtr.copy(), Xva.copy()
        for col in missing_train:
            flag = f"{col}__MISSING_FLAG"
            xa[flag], xv[flag] = Xtr[col].isna().astype("int8"), Xva[col].isna().astype("int8")
        variants.append(("Add training-fold missingness flags", xa, xv, ";".join(missing_train)))
        for name, xa, xv, affected in variants:
            ab_model, ab_categories = make_catboost(get_params(tuning, fold), list(xa.columns))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ab_model.fit(xa, ytr, classifier__cat_features=ab_categories)
                p = ab_model.predict_proba(xv)[:, list(ab_model.named_steps["classifier"].classes_).index(1)]
            ablation_rows.append({"candidate": name, "fold": fold, "n_train": len(itr),
                "n_validation": len(iva), "validation_positive_cases": int(yva.sum()),
                "features_used": xa.shape[1], "features_affected": affected,
                "average_precision": average_precision_score(yva, p), "auroc": roc_auc_score(yva, p),
                "brier_score": brier_score_loss(yva, p), "log_loss": clipped_loss(yva, p),
                "exploratory_outer_fold_result": True,
                "warning_text": " | ".join(f"{w.category.__name__}: {w.message}" for w in caught)})

    # Explicitly compare the reconstructed CatBoost fold scores with its saved fold output.
    local_cat = pd.DataFrame(fold_rows).query("model == 'CatBoost tuned'").sort_values("fold")
    saved_cat = stage6_folds[(stage6_folds.target == TARGET) & (stage6_folds.model == "CatBoost tuned")].sort_values("fold")
    for metric in ["average_precision", "auroc", "brier_score", "log_loss"]:
        if metric in saved_cat and not np.allclose(local_cat[metric], saved_cat[metric], atol=1e-10, rtol=0):
            raise ValueError(f"Recomputed CatBoost {metric} differs from saved Stage 6 fold result")

    perm = pd.DataFrame(perm_rows)
    stability = (perm.assign(rank=perm.groupby(["model", "fold"]).importance_mean.rank(ascending=False, method="min"))
        .groupby(["model", "feature"], as_index=False).agg(
            mean_importance=("importance_mean", "mean"), sd_fold_importance=("importance_mean", "std"),
            mean_within_fold_permutation_sd=("importance_std", "mean"),
            top10_fold_count=("rank", lambda s: int((s <= 10).sum())),
            folds_present=("fold", "nunique")))
    stability["top10_fold_fraction"] = stability.top10_fold_count / FOLDS
    shap = pd.DataFrame(shap_rows)
    shap_summary = shap.groupby("feature", as_index=False).agg(
        mean_absolute_shap=("mean_absolute_shap", "mean"), sd_fold_absolute_shap=("mean_absolute_shap", "std"),
        mean_signed_shap=("mean_signed_shap", "mean"), folds_present=("fold", "nunique"))
    fold_df = pd.DataFrame(fold_rows)
    fold_summary = fold_df.groupby("model").agg(
        average_precision_mean=("average_precision", "mean"), average_precision_sd=("average_precision", "std"),
        average_precision_min=("average_precision", "min"), average_precision_max=("average_precision", "max"),
        positive_cases_per_fold_min=("validation_positive_cases", "min"),
        positive_cases_per_fold_max=("validation_positive_cases", "max"), folds=("fold", "nunique")).reset_index()
    ablation = pd.DataFrame(ablation_rows).groupby("candidate", as_index=False).agg(
        folds=("fold", "nunique"), average_precision_mean=("average_precision", "mean"),
        average_precision_sd=("average_precision", "std"), auroc_mean=("auroc", "mean"),
        auroc_sd=("auroc", "std"), brier_score_mean=("brier_score", "mean"),
        brier_score_sd=("brier_score", "std"), log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"), exploratory_outer_fold_result=("exploratory_outer_fold_result", "all"))

    # Leave-one-positive-out sensitivity on saved OOF scores, without exporting row identifiers.
    sens_rows = []
    for model_name, group in [("CatBoost tuned", target_oof.rename(columns={"predicted_probability": "probability"}))]:
        yy, pp = group.true_label.to_numpy(dtype=int), group.probability.to_numpy(dtype=float)
        full = average_precision_score(yy, pp)
        delete = [average_precision_score(np.delete(yy, i), np.delete(pp, i)) for i in np.flatnonzero(yy == 1)]
        sens_rows.append({"model": model_name, "oof_rows": len(yy), "positive_cases": int(yy.sum()),
            "oof_average_precision": full, "leave_one_positive_out_mean": float(np.mean(delete)),
            "leave_one_positive_out_min": float(np.min(delete)), "leave_one_positive_out_max": float(np.max(delete)),
            "max_absolute_change": float(np.max(np.abs(np.asarray(delete) - full)))})
    sensitivity = pd.DataFrame(sens_rows)

    OUT.mkdir()
    group_table.to_csv(OUT / "fibr_group_comparison.csv", index=False)
    missing_table.to_csv(OUT / "fibr_missingness_analysis.csv", index=False)
    perm.to_csv(OUT / "fibr_permutation_importance.csv", index=False)
    stability.to_csv(OUT / "fibr_feature_stability.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT / "fibr_ablation_results.csv", index=False)
    fold_df.to_csv(OUT / "fibr_fold_analysis.csv", index=False)
    shap.to_csv(OUT / "fibr_shap_fold_summary.csv", index=False)
    shap_summary.to_csv(OUT / "fibr_shap_summary.csv", index=False)
    fold_summary.to_csv(OUT / "fibr_fold_summary.csv", index=False)
    sensitivity.to_csv(OUT / "fibr_oof_positive_sensitivity.csv", index=False)

    top = stability[stability.model == "CatBoost tuned"].nlargest(15, "mean_importance")
    fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(top.feature[::-1], top.mean_importance[::-1], xerr=top.sd_fold_importance[::-1], color="#4472C4")
    ax.set_title("FIBR_PREDS CatBoost permutation importance (AP decrease)"); ax.set_xlabel("Mean outer-fold AP decrease")
    fig.tight_layout(); fig.savefig(OUT / "fibr_permutation_importance.png", dpi=160); plt.close(fig)
    top_s = shap_summary.nlargest(15, "mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(top_s.feature[::-1], top_s.mean_absolute_shap[::-1], xerr=top_s.sd_fold_absolute_shap[::-1], color="#70AD47")
    ax.set_title("FIBR_PREDS CatBoost TreeSHAP importance"); ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout(); fig.savefig(OUT / "fibr_shap_importance.png", dpi=160); plt.close(fig)
    top_m = missing_table.head(20)
    fig, ax = plt.subplots(figsize=(9, 7)); ypos = np.arange(len(top_m)); height = .38
    ax.barh(ypos - height / 2, top_m.negative_missing_rate, height, color="#A5A5A5", label="Outcome 0")
    ax.barh(ypos + height / 2, top_m.positive_missing_rate, height, color="#ED7D31", label="Outcome 1")
    ax.set_yticks(ypos, top_m.feature); ax.set_xlabel("Missingness rate")
    ax.set_title("FIBR_PREDS feature missingness by outcome"); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fibr_missingness.png", dpi=160); plt.close(fig)

    # Compact distribution diagnostic: top numeric SMDs and semantic nominal categories.
    numeric_top = (group_table[group_table.feature_type == "continuous"]
                   .dropna(subset=["standardized_mean_difference"])
                   .assign(abs_smd=lambda d: d.standardized_mean_difference.abs())
                   .nlargest(4, "abs_smd"))
    nominal_rows = group_table[group_table.feature.isin(CATEGORICAL)
                               & (group_table.feature_type == "nominal_or_code")].copy()
    nominal_rows["label"] = nominal_rows.feature.astype(str) + "=" + nominal_rows.category.astype(str)
    nominal_rows = nominal_rows.nlargest(14, "category_fibr_rate")
    fig = plt.figure(figsize=(15, 6))
    # Individual horizontal panels preserve original measurement units.
    numeric_axes = fig.add_subplot(1, 2, 1)
    if len(numeric_top):
        labels = [f"{r.feature} (SMD {r.standardized_mean_difference:.2f})" for r in numeric_top.itertuples()]
        vals0 = [X.loc[y == 0, r.feature].dropna().to_numpy() for r in numeric_top.itertuples()]
        vals1 = [X.loc[y == 1, r.feature].dropna().to_numpy() for r in numeric_top.itertuples()]
        positions = np.arange(len(labels))
        numeric_axes.boxplot(vals0, positions=positions - .16, widths=.25, showfliers=False, orientation="horizontal",
                             patch_artist=True, boxprops={"facecolor": "#A5A5A5"}, medianprops={"color": "black"})
        numeric_axes.boxplot(vals1, positions=positions + .16, widths=.25, showfliers=False, orientation="horizontal",
                             patch_artist=True, boxprops={"facecolor": "#ED7D31"}, medianprops={"color": "black"})
        numeric_axes.set_yticks(positions, labels); numeric_axes.set_xlabel("Observed value (native units)")
        numeric_axes.legend([plt.Line2D([0], [0], color="#A5A5A5", lw=6), plt.Line2D([0], [0], color="#ED7D31", lw=6)],
                            ["Outcome 0", "Outcome 1"], frameon=False)
    else:
        numeric_axes.text(.5, .5, "No continuous features", ha="center", va="center")
    numeric_axes.set_title("Most different numeric features")
    nominal_ax = fig.add_subplot(1, 2, 2)
    if len(nominal_rows):
        nr = nominal_rows.sort_values("category_fibr_rate")
        nominal_ax.barh(nr.label, nr.category_fibr_rate, color="#8064A2")
        nominal_ax.set_xlabel("FIBR_PREDS rate within category")
        nominal_ax.set_title("Observed rates for nominal codes")
    else:
        nominal_ax.text(.5, .5, "No nominal category rows", ha="center", va="center")
    fig.suptitle("FIBR_PREDS outcome-group distributions (descriptive)")
    fig.tight_layout(); fig.savefig(OUT / "fibr_group_distributions.png", dpi=160)
    # Preserve the initial plot and publish a corrected horizontal-boxplot view.
    fig.savefig(OUT / "fibr_group_distributions_corrected.png", dpi=160); plt.close(fig)

    perm_top = stability[stability.model == "CatBoost tuned"].nlargest(10, "mean_importance").feature.tolist()
    shap_top = shap_summary.nlargest(10, "mean_absolute_shap").feature.tolist()
    group_top = (group_table[group_table.feature_type == "continuous"]
                 .dropna(subset=["standardized_mean_difference"])
                 .assign(abs_smd=lambda d: d.standardized_mean_difference.abs())
                 .nlargest(5, "abs_smd"))
    group_summary = ", ".join(
        f"{r.feature} (SMD {r.standardized_mean_difference:.2f}; medians {r.negative_median:.3g} vs {r.positive_median:.3g})"
        for r in group_top.itertuples())
    miss_summary = ", ".join(
        f"{r.feature} ({r.negative_missing_rate:.1%} vs {r.positive_missing_rate:.1%}; Δ {r.positive_minus_negative:+.1%})"
        for r in missing_table.head(5).itertuples())
    stable_top = stability[stability.model == "CatBoost tuned"].nlargest(8, "mean_importance")
    stability_text = ", ".join(f"{r.feature} ({r.top10_fold_count}/{FOLDS} top-10 folds)" for r in stable_top.itertuples())
    fold_performance_text = "; ".join(
        f"{r.model}: AP {r.average_precision_mean:.3f} ± {r.average_precision_sd:.3f} (range {r.average_precision_min:.3f}–{r.average_precision_max:.3f}; 27 positives/fold)"
        for r in fold_summary.itertuples())
    ablation_summary = ablation.set_index("candidate")
    baseline_name = "CatBoost tuned baseline"
    deltas = {}
    for candidate in ablation_summary.index:
        if candidate != baseline_name:
            deltas[candidate] = {
                "ap": float(ablation_summary.loc[candidate, "average_precision_mean"] - ablation_summary.loc[baseline_name, "average_precision_mean"]),
                "brier": float(ablation_summary.loc[candidate, "brier_score_mean"] - ablation_summary.loc[baseline_name, "brier_score_mean"]),
                "log_loss": float(ablation_summary.loc[candidate, "log_loss_mean"] - ablation_summary.loc[baseline_name, "log_loss_mean"]),
            }
    delta_text = "; ".join(f"{name}: ΔAP {d['ap']:+.3f}, ΔBrier {d['brier']:+.3f}, ΔLog Loss {d['log_loss']:+.3f}"
                            for name, d in deltas.items())
    if all(d["ap"] <= 0 for d in deltas.values()):
        next_recommendation = "If both pre-specified ablations fail to show a consistent AP improvement, prioritize a follow-up missingness mechanism audit and retain the full feature set for the next model comparison."
    else:
        next_recommendation = "For any ablation with a positive mean AP delta, pre-specify a focused nested-CV follow-up that tests the same idea with selection confined to inner folds and checks Brier/Log Loss; do not treat this outer-fold signal as a selected winner."
    loo = sensitivity.iloc[0]
    lines = ["# Stage 7 — FIBR_PREDS diagnostic analysis", "",
        "## Scope and method", "",
        f"This exploratory analysis uses the Stage 1 development rows with observed {TARGET} labels only (n={len(y)}, positives={int(y.sum())}, prevalence={y.mean():.3%}). It reconstructs the deterministic shuffled five-fold stratified split (seed 42) and checks saved fold sizes, class counts and training prevalence against Stage 3 Random Forest, Stage 4 Extra Trees, and Stage 6 CatBoost summaries ({fold_checks} values checked). Historical record-level fold membership is unavailable in those summaries, so matching signatures do not prove identity of every historical fold member.", "",
        "Random Forest preliminary, Extra Trees preliminary, and CatBoost tuned are fit on each outer training fold. CatBoost uses the already AP-selected Stage 6 parameters for that fold. Average precision is the primary ranking metric; permutation importance scores the decrease in validation AP after feature permutation (three repeats, one worker, seed 42 + fold). CatBoost TreeSHAP explains only validation rows and uses a deterministic reference sample drawn from the matching training fold. No dependencies were installed. Python warnings are retained in output tables; fit exceptions stop the run rather than dropping a fold.", "",
        "The positive-versus-negative tables are descriptive associations, not causal effects. Continuous-feature quartiles/outlier counts use the pooled development distribution, and nominal/ordinal categories are shown as observed codes. The leave-one-positive-out calculation uses saved Stage 6 OOF predictions and exports no patient identifiers.", "",
        "## Answers to the diagnostic questions", "",
        f"1. **Which features distinguish positive and negative cases?** The largest continuous group differences were {group_summary}. The highest CatBoost AP permutation features were {', '.join(perm_top)}; nominal category FIBR_PREDS rates are plotted in `fibr_group_distributions_corrected.png`.",
        f"2. **Could missingness explain predictive behavior?** Largest absolute missingness differences were {miss_summary}. These are associations; missingness may reflect clinical workflow or collection patterns.",
        f"3. **How stable are model performance and feature importance across folds?** {fold_performance_text}. Leading CatBoost features with top-10 fold counts are {stability_text}. CatBoost TreeSHAP leaders were {', '.join(shap_top)}.",
        f"4–5. **What did the fixed ablations show?** Mean metric deltas versus the saved-parameter CatBoost baseline were {delta_text}. These outer-fold results are exploratory and do not select a winner; fold-level scores and affected features are in `fibr_ablation_results.csv`.",
        f"6. **How sensitive is AP to a small positive sample?** Removing one positive at a time from saved OOF predictions gave AP {loo.oof_average_precision:.3f} originally, leave-one-positive-out range {loo.leave_one_positive_out_min:.3f}–{loo.leave_one_positive_out_max:.3f}, maximum absolute change {loo.max_absolute_change:.3f}. No patient IDs are included.",
        f"7. **What should be tried next?** {next_recommendation} The follow-up should use nested development CV, keep AP as the selection metric, and reserve generalization claims for the separately governed final evaluation.", "",
        "## Interpretation and limitations", "",
        "FIBR_PREDS is relatively uncommon, and there are only five outer folds; fold-level variation and permutation rankings can be noisy. Permuting correlated predictors can distribute or suppress importance. SHAP values describe model output attribution conditional on the fitted model and training-fold reference distribution, not causal effects. Group comparisons do not adjust for confounding. The ablation scores reuse outer folds and fixed CatBoost settings for diagnostics; they must not be treated as an unbiased winner-selection exercise. The Stage 6 saved OOF baseline and reconstructed folds are development-only.", "",
        f"Environment: scikit-learn `{sklearn.__version__}`, CatBoost `{catboost.__version__}`. Run `.venv/bin/python mi_stage7_fibr_diagnostics.py`. Outputs are written only to `{OUT.name}/` and the script refuses to overwrite an existing directory.", "",
        "## Output files", "",
        "- `fibr_group_comparison.csv`: outcome-group summaries for numeric and coded features.",
        "- `fibr_missingness_analysis.csv`: missingness by outcome group.",
        "- `fibr_permutation_importance.csv`, `fibr_feature_stability.csv`: fold importance and ranking stability.",
        "- `fibr_ablation_results.csv`: fold-level exploratory CatBoost ablations with AP, AUROC, Brier and Log Loss.",
        "- `fibr_fold_analysis.csv`, `fibr_fold_summary.csv`, `fibr_oof_positive_sensitivity.csv`: fold performance and OOF sensitivity.",
        "- `fibr_shap_fold_summary.csv`, `fibr_shap_summary.csv`: fold and aggregate TreeSHAP summaries.",
        "- `fibr_permutation_importance.png`, `fibr_shap_importance.png`, `fibr_missingness.png`, `fibr_group_distributions_corrected.png`: diagnostic plots. The initial `fibr_group_distributions.png` is preserved from the first run; use the corrected plot.",
    ]
    (OUT / "stage7_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
