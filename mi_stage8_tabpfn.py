"""Stage 8: development-only TabPFN outer-fold experiment.

Run with `.venv/bin/python mi_stage8_tabpfn.py` after TabPFN is installed and
its model-weight license has been accepted by the user. This script only reads
explicit development artifacts and refuses to overwrite Stage 8 outputs.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import resource
import tempfile
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage8-mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import mi_stage6_catboost as stage6

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
STAGE2 = ROOT / "stage2_outputs"
STAGE3 = ROOT / "stage3_outputs"
STAGE4 = ROOT / "stage4_extratrees_outputs"
STAGE6 = ROOT / "stage6_catboost_outputs"
OUT = ROOT / "stage8_tabpfn_outputs"
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]
CATEGORICAL = list(stage6.CATEGORICAL)
SEED, N_FOLDS, THRESHOLD, CLIP = 42, 5, 0.5, 1e-15
EXPECTED_TABPFN_VERSION = "9.2.0"
TABPFN_MODEL = "TabPFN 3.5"
REFERENCE_MODELS = [
    "Dummy prior prevalence", "Logistic regression",
    "Random Forest preliminary", "Random Forest tuned",
    "Extra Trees preliminary", "Extra Trees tuned", "CatBoost tuned",
]
METRIC_DIRECTIONS = {
    "average_precision": "higher",
    "auroc": "higher",
    "brier_score": "lower",
    "log_loss": "lower",
}


def _rss_mb() -> float:
    """Process lifetime high-water RSS in MiB (macOS reports bytes)."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if os.uname().sysname == "Darwin" else value / 1024


def _mps_driver_mb(torch_module) -> float:
    try:
        if torch_module.backends.mps.is_available():
            return float(torch_module.mps.driver_allocated_memory()) / (1024 * 1024)
    except (AttributeError, RuntimeError):
        pass
    return float("nan")


def runtime_device_status(torch_module) -> dict:
    """Record the backend auto-selection is expected to use on this run."""
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_built = bool(mps_backend.is_built()) if mps_backend is not None else False
    mps_available = bool(mps_backend.is_available()) if mps_backend is not None else False
    cuda_available = bool(torch_module.cuda.is_available())
    if cuda_available:
        selected = "cuda"
    elif mps_available:
        selected = "mps"
    else:
        selected = "cpu"
    return {
        "torch_cuda_available": cuda_available,
        "torch_mps_built": mps_built,
        "torch_mps_available": mps_available,
        "auto_device_outcome": selected,
    }


def validate_inputs(X: pd.DataFrame, y_all: pd.DataFrame, ids: pd.DataFrame,
                    dictionary: pd.DataFrame) -> None:
    stage6.validate_inputs(X, y_all, ids, dictionary)
    if not set(TARGETS).issubset(y_all.columns):
        raise ValueError("One or more required targets are missing.")
    if len(X) != len(y_all) or len(X) != len(ids):
        raise ValueError("Stage 1 development X, y, and IDs are not row-aligned.")
    if ids.shape[1] != 1 or ids.iloc[:, 0].isna().any() or ids.iloc[:, 0].duplicated().any():
        raise ValueError("Development IDs must be a single non-null unique column.")
    if set(CATEGORICAL).difference(X.columns):
        raise ValueError(f"Native categorical columns missing: {set(CATEGORICAL).difference(X.columns)}")


def check_fold_reference(reference: pd.DataFrame, target: str, model: str,
                         y: pd.Series, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    rows = reference[(reference.target == target) & (reference.model == model)]
    if rows.duplicated("fold").any() or set(rows.fold.astype(int)) != set(range(1, N_FOLDS + 1)):
        raise ValueError(f"Saved folds missing or duplicated for {target}/{model}.")
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        saved = rows.loc[rows.fold.astype(int) == fold].iloc[0]
        y_train, y_valid = y.iloc[train_idx].to_numpy(), y.iloc[valid_idx].to_numpy()
        signature = {
            "n_train": len(train_idx),
            "n_validation": len(valid_idx),
            "validation_positive_cases": int(y_valid.sum()),
            "validation_negative_cases": int(len(y_valid) - y_valid.sum()),
            "training_prevalence": float(y_train.mean()),
        }
        for col, value in signature.items():
            if col not in rows:
                raise ValueError(f"Saved folds lack {col}: {target}/{model}.")
            if col == "training_prevalence":
                good = np.isclose(float(saved[col]), value, rtol=0, atol=1e-12)
            else:
                good = int(saved[col]) == value
            if not good:
                raise ValueError(f"Saved fold signature mismatch: {target}/{model}/fold {fold}/{col}.")


def fold_categories(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep categorical meaning native; infer each category vocabulary from train only."""
    train = train.copy()
    valid = valid.copy()
    for col in CATEGORICAL:
        categories = pd.Index(train[col].dropna().unique()).sort_values()
        train[col] = pd.Categorical(train[col], categories=categories)
        # Validation-only categories become missing rather than informing the vocabulary.
        valid[col] = pd.Categorical(valid[col], categories=categories)
    return train, valid


def score_fold(y_true: np.ndarray, probability: np.ndarray) -> dict:
    if probability.ndim != 1 or len(probability) != len(y_true):
        raise ValueError("Predicted probabilities have an invalid shape.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted probabilities are non-finite or outside [0, 1].")
    predicted = (probability >= THRESHOLD).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, probability)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.clip(probability, CLIP, 1 - CLIP), labels=[0, 1])),
        "accuracy_at_0_5": float(accuracy_score(y_true, predicted)),
        "precision_at_0_5": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, predicted, zero_division=0)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def get_reference_metrics(stage2: pd.DataFrame, stage3: pd.DataFrame,
                          stage4: pd.DataFrame, stage6_paired: pd.DataFrame,
                          stage6_folds: pd.DataFrame) -> pd.DataFrame:
    """Combine prior matched-fold scores; Stage 6 supplies comparator log loss."""
    all_ref = pd.concat([stage2, stage3, stage4], ignore_index=True, sort=False)
    ref = all_ref[all_ref.model.isin(REFERENCE_MODELS[:-1])].copy()
    cb = stage6_folds[(stage6_folds.model == "CatBoost tuned") &
                      (stage6_folds.target.isin(TARGETS))].copy()
    if len(cb) != len(TARGETS) * N_FOLDS:
        raise ValueError("Expected 15 Stage 6 CatBoost fold rows.")
    cb_rows = cb[["target", "model", "fold", "average_precision", "auroc", "brier_score", "log_loss",
                  "precision_at_0_5", "recall_at_0_5", "f1_at_0_5", "true_negatives",
                  "false_positives", "false_negatives", "true_positives"]].copy()
    ref = pd.concat([ref, cb_rows], ignore_index=True, sort=False)
    if ref.duplicated(["target", "model", "fold"]).any():
        raise ValueError("Duplicate saved comparator target/model/fold rows.")
    logloss = stage6_paired[["target", "model", "fold", "comparator_log_loss"]].rename(
        columns={"comparator_log_loss": "log_loss"})
    cb_loss = stage6_folds.loc[
        (stage6_folds.model == "CatBoost tuned") & stage6_folds.target.isin(TARGETS),
        ["target", "model", "fold", "log_loss"],
    ]
    logloss = pd.concat([logloss, cb_loss], ignore_index=True)
    ref = ref.merge(logloss, on=["target", "model", "fold"], how="left", suffixes=("", "_stage6"), validate="one_to_one")
    ref["log_loss"] = ref["log_loss"].fillna(ref["log_loss_stage6"])
    ref = ref.drop(columns=["log_loss_stage6"], errors="ignore")
    if ref.log_loss.isna().any():
        raise ValueError("A prior model is missing matched-fold Log Loss.")
    expected = {(target, model, fold) for target in TARGETS for model in REFERENCE_MODELS
                for fold in range(1, N_FOLDS + 1)}
    observed = set(zip(ref.target, ref.model, ref.fold.astype(int)))
    if observed != expected:
        raise ValueError(f"Prior comparator fold rows differ from expected: {len(observed)} vs {len(expected)}.")
    return ref


def summarize_models(folds: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    metrics = [*METRIC_DIRECTIONS, "precision_at_0_5", "recall_at_0_5", "f1_at_0_5"]
    tab = folds[["target", "model", "fold", *metrics]].copy()
    ref = references[["target", "model", "fold", *metrics]].copy()
    combined = pd.concat([ref, tab], ignore_index=True)
    rows = []
    for (target, model), g in combined.groupby(["target", "model"], sort=False):
        row = {"target": target, "model": model, "folds": int(g.fold.nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1))
        rows.append(row)
    result = pd.DataFrame(rows)
    expected = len(TARGETS) * (len(REFERENCE_MODELS) + 1)
    if len(result) != expected or (result.folds != N_FOLDS).any():
        raise ValueError("Aggregate metric rows do not cover every model/target/fold.")
    return result


def compare_to_best_mean(tab_value: float, comparator_value: float, metric: str,
                         comparator_model: str) -> str:
    """State the mean difference using the metric's improvement direction."""
    delta = float(tab_value - comparator_value)
    if abs(delta) <= 1e-12:
        direction, quality = "equal", "tied"
    else:
        direction = "higher" if delta > 0 else "lower"
        is_better = (delta > 0) if METRIC_DIRECTIONS[metric] == "higher" else (delta < 0)
        quality = "better" if is_better else "worse"
    return (
        f"mean {metric.replace('_', ' ')} is {direction} ({quality}) than the best saved comparator "
        f"{comparator_model} by {abs(delta):.6f} (TabPFN {tab_value:.6f}; "
        f"comparator {comparator_value:.6f})"
    )


def make_paired(tab_folds: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = list(METRIC_DIRECTIONS)
    for tab in tab_folds.itertuples(index=False):
        group = refs[(refs.target == tab.target) & (refs.fold == tab.fold)]
        if set(group.model) != set(REFERENCE_MODELS):
            raise ValueError(f"Comparator set incomplete at {tab.target}/fold {tab.fold}.")
        for comparator in group.itertuples(index=False):
            row = {"target": tab.target, "fold": int(tab.fold), "comparator_model": comparator.model}
            for metric in metrics:
                tab_value, comp_value = float(getattr(tab, metric)), float(getattr(comparator, metric))
                delta = tab_value - comp_value
                direction = METRIC_DIRECTIONS[metric]
                row[f"tabpfn_{metric}"] = tab_value
                row[f"comparator_{metric}"] = comp_value
                row[f"tabpfn_minus_comparator_{metric}"] = delta
                row[f"tabpfn_better_{metric}"] = delta > 1e-12 if direction == "higher" else delta < -1e-12
                row[f"tie_{metric}"] = abs(delta) <= 1e-12
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(TARGETS) * N_FOLDS * len(REFERENCE_MODELS):
        raise ValueError("Paired comparison row count is incorrect.")
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage 8 output directory: {OUT}")
    paths = [
        STAGE1 / "X_train.csv", STAGE1 / "y_train.csv", STAGE1 / "train_record_ids.csv",
        STAGE1 / "column_dictionary.csv", STAGE2 / "fold_level_metrics.csv",
        STAGE3 / "fold_level_metrics.csv", STAGE4 / "fold_level_metrics.csv",
        STAGE6 / "paired_comparison.csv", STAGE6 / "fold_level_metrics.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required explicit development inputs are missing: {missing}")
    X, y_all, ids, dictionary = [pd.read_csv(path) for path in paths[:4]]
    validate_inputs(X, y_all, ids, dictionary)
    id_col = ids.columns[0]
    stage2, stage3, stage4, stage6_paired, stage6_folds = [pd.read_csv(path) for path in paths[4:]]
    reference_metrics = get_reference_metrics(stage2, stage3, stage4, stage6_paired, stage6_folds)

    # Import only after development input validation so syntax/static checks do not
    # require an installed TabPFN package.
    skrub_data_dir = Path(tempfile.gettempdir()) / "mi-stage8-skrub-data"
    skrub_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SKB_DATA_DIRECTORY", str(skrub_data_dir))
    try:
        import torch
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise RuntimeError("Install the approved TabPFN version and dependencies before running Stage 8.") from exc
    from importlib.metadata import version as package_version
    installed_tabpfn_version = package_version("tabpfn")
    if installed_tabpfn_version != EXPECTED_TABPFN_VERSION:
        raise RuntimeError(
            f"Stage 8 was reviewed for TabPFN {EXPECTED_TABPFN_VERSION}; found {installed_tabpfn_version}."
        )

    fold_rows, oof_rows = [], []
    all_reference_rows = []
    runtime_status = runtime_device_status(torch)
    for target in TARGETS:
        y = y_all[target].dropna().astype("int8")
        X_target = X.loc[y.index].copy()
        id_target = ids.loc[y.index, id_col]
        if not set(y.unique()) == {0, 1} or y.value_counts().min() < N_FOLDS:
            raise ValueError(f"{target} must contain both classes with >=5 cases per class.")
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X_target, y))
        for model_name in REFERENCE_MODELS:
            source = (stage2 if model_name in REFERENCE_MODELS[:2] else
                      stage3 if model_name.startswith("Random Forest") else
                      stage4 if model_name.startswith("Extra Trees") else stage6_folds)
            actual_name = "CatBoost tuned" if model_name == "CatBoost tuned" else model_name
            check_fold_reference(source, target, actual_name, y, splits)
        group_refs = reference_metrics[reference_metrics.target == target]
        # Stage 6 paired rows also certify that all six comparator models use
        # exactly one score per reconstructed outer fold.
        pair_rows = stage6_paired[stage6_paired.target == target]
        expected_pair = {(name, fold) for name in REFERENCE_MODELS[:-1] for fold in range(1, N_FOLDS + 1)}
        if set(zip(pair_rows.model, pair_rows.fold.astype(int))) != expected_pair:
            raise ValueError(f"Stage 6 comparator pairs are incomplete for {target}.")
        all_reference_rows.append(group_refs)

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            X_train = X_target.iloc[train_idx]
            X_valid = X_target.iloc[valid_idx]
            y_train = y.iloc[train_idx].to_numpy()
            y_valid = y.iloc[valid_idx].to_numpy()
            train_native, valid_native = fold_categories(X_train, X_valid)
            model = TabPFNClassifier(
                random_state=SEED,
                device="auto",
                n_preprocessing_jobs=1,
                memory_saving_mode="auto",
                show_progress_bar=False,
            )
            fold_warnings = []
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train_native, y_train)
                probability = model.predict_proba(valid_native)[:, list(model.classes_).index(1)]
            elapsed = time.perf_counter() - started
            fold_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
            metrics = score_fold(y_valid, np.asarray(probability, dtype=float))
            fold_rows.append({
                "target": target, "model": TABPFN_MODEL, "fold": fold,
                "n_train": len(train_idx), "n_validation": len(valid_idx),
                "validation_positive_cases": int(y_valid.sum()),
                "validation_negative_cases": int(len(y_valid) - y_valid.sum()),
                "training_prevalence": float(y_train.mean()), "threshold": THRESHOLD,
                **metrics, "fit_predict_runtime_seconds": elapsed,
                "first_fit_may_include_license_auth_or_checkpoint_download": bool(
                    target == TARGETS[0] and fold == 1
                ),
                **runtime_status,
                "process_peak_rss_mb": _rss_mb(), "mps_driver_allocated_mb_after_fold": _mps_driver_mb(torch),
                "fit_warnings": " | ".join(fold_warnings),
            })
            for pos, row_idx in enumerate(valid_idx):
                oof_rows.append({
                    "target": target, "ID": id_target.iloc[row_idx], "outer_fold": fold,
                    "true_label": int(y_valid[pos]), "predicted_probability": float(probability[pos]),
                    "predicted_class_at_0_5": int(probability[pos] >= THRESHOLD),
                })
            del model, train_native, valid_native, probability
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except (AttributeError, RuntimeError):
                    pass

    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    refs = pd.concat(all_reference_rows, ignore_index=True)
    expected_keys = {(target, record_id) for target in TARGETS
                     for record_id in ids.iloc[:, 0][y_all[target].notna()]}
    observed_keys = set(zip(oof.target, oof.ID))
    if observed_keys != expected_keys or len(oof) != len(expected_keys):
        raise ValueError("TabPFN OOF predictions do not align with eligible development IDs.")
    for target in TARGETS:
        pred = oof[oof.target == target]
        if pred.duplicated("ID").any() or pred.predicted_probability.isna().any():
            raise ValueError(f"Duplicate IDs or missing predictions in OOF output for {target}.")
        expected_labels = dict(zip(
            ids.loc[y_all[target].notna(), id_col],
            y_all.loc[y_all[target].notna(), target].astype(int),
        ))
        observed_labels = dict(zip(pred.ID, pred.true_label.astype(int)))
        if observed_labels != expected_labels:
            raise ValueError(f"OOF labels do not align with development IDs for {target}.")
    if folds.groupby("target").size().to_dict() != {target: N_FOLDS for target in TARGETS}:
        raise ValueError("Not all TabPFN target folds completed.")

    comparison = summarize_models(folds, refs)
    paired = make_paired(folds, refs)
    aggregate = comparison.copy()
    OUT.mkdir()
    folds.to_csv(OUT / "fold_level_metrics.csv", index=False)
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    paired.to_csv(OUT / "paired_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_metrics = ["average_precision", "auroc", "brier_score", "log_loss"]
    for ax, metric in zip(axes.flat, plot_metrics):
        subset = comparison.pivot(index="target", columns="model", values=f"{metric}_mean")
        errors = comparison.pivot(index="target", columns="model", values=f"{metric}_std")
        subset.plot(kind="bar", ax=ax, yerr=errors, capsize=2, width=.82)
        ax.set_title(f"{metric.replace('_', ' ').title()} (" + ("higher" if METRIC_DIRECTIONS.get(metric) == "higher" else "lower") + " is better)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Stage 8: TabPFN vs matched development CV models")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=160)
    plt.close(fig)

    config = [
        "# Stage 8 TabPFN configuration", "",
        f"- Package: `tabpfn=={installed_tabpfn_version}` (latest stable at compatibility review: `9.2.0`).",
        "- Model family: TabPFN-3.5, the default local classifier checkpoint for package 9.2.0.",
        f"- Python: `{os.sys.version.split()[0]}`; scikit-learn: `{sklearn.__version__}`; PyTorch: `{torch.__version__}`; device selection: `auto`; observed CUDA available: `{runtime_status['torch_cuda_available']}`, MPS built: `{runtime_status['torch_mps_built']}`, MPS available: `{runtime_status['torch_mps_available']}`, expected auto-selected device: `{runtime_status['auto_device_outcome']}`.",
        f"- Outer CV: shuffled stratified 5-fold, seed `{SEED}`, target-specific eligible development rows; one fold is fit and released before the next.",
        "- Predictors: all 102 permitted Stage 1 admission-time fields. Numeric fields are passed as numeric; the declared categorical fields `SEX`, `ant_im`, `lat_im`, `inf_im`, `post_im` are pandas categorical dtype. Category vocabularies are learned from each training fold only; validation-only category values map to missing.",
        "- Missing values remain missing. No scaling, imputation, one-hot encoding, class weights, resampling, calibration, or hyperparameter search is applied. Internal TabPFN preprocessing is part of the model fit on outer training data.",
        f"- Fixed classification threshold: probability `>= {THRESHOLD}`; AP is the primary ranking metric; no threshold tuning.",
        "- Classifier: `TabPFNClassifier(random_state=42, device='auto', n_preprocessing_jobs=1, memory_saving_mode='auto', show_progress_bar=False)`; other classifier parameters are defaults.",
        "- Package code is Apache-2.0; TabPFN-3.5 weights are under a separate non-commercial/non-production license. The license says it is unavailable to consumers under German BGB §13 and requires acceptance by a person acting for professional purposes. The user must review and accept it before weight download/use.",
        "- Official memory guidance reports about 3 GB VRAM for an A100 example with 1,000 training rows, 1,000 test rows, and 200 features. This is not an Apple MPS or 8 GB MacBook Air guarantee. Here folds are run sequentially with about 1,088 training and 272 validation rows and 102 predictors; measured runtime/memory are reported after execution.",
        "- Memory reporting: process lifetime peak RSS plus MPS driver allocation after each fold when available; these are diagnostics, not a universal peak-memory guarantee. Runtime is fit plus validation prediction time.",
        "- Official sources reviewed 2026-11-05: https://github.com/PriorLabs/TabPFN/releases (v9.2.0); https://docs.priorlabs.ai/models; https://docs.priorlabs.ai/troubleshooting/OOM-errors; https://docs.priorlabs.ai/models/accessing-model-weights; https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE.",
        f"- Fold signatures checked against saved Stage 2–6 records: `{len(TARGETS) * len(REFERENCE_MODELS) * N_FOLDS}` model-fold signatures (3 targets × 7 reference models × 5 folds).",
    ]
    (OUT / "tabpfn_configuration.md").write_text("\n".join(config) + "\n", encoding="utf-8")

    lines = [
        "# Stage 8 — TabPFN experiment", "",
        "## Scope and design", "",
        f"This controlled experiment uses development data only (seed {SEED}; five shuffled stratified outer folds per target). No locked test data are read. TabPFN is evaluated once per outer training fold; no outer validation data are used for model selection. Prior model metrics come from saved matched-fold Stage 2, 3, 4 and 6 outputs. The fixed threshold is 0.5. Average Precision is the primary ranking metric; threshold precision/recall/F1 and confusion counts are separate operating-point measures.",
        "Saved Stage 2–4 fold metrics and Stage 6 comparator summaries do not contain validation record IDs. This run reconstructs the seed-42 folds from the unchanged Stage 1 development row order and checks fold sizes, validation class counts, and training prevalence. Historical record-level membership therefore cannot be directly confirmed by ID; paired comparisons use the reconstructed same-numbered folds and matching signatures.", "",
        "TabPFN receives native numeric data, pandas categorical types for nominal fields, and missing values without manual imputation or one-hot encoding. The categorical vocabulary is learned on each training fold only. No class weighting, resampling, calibration, or tuning is used. Full configuration and transformations are in `tabpfn_configuration.md`.", "",
        "## Results", "",
    ]
    targets_meeting_stability_screen = []
    targets_not_meeting_stability_screen = []
    for target in TARGETS:
        g = comparison[comparison.target == target].set_index("model")
        tab = g.loc[TABPFN_MODEL]
        rank_best = comparison[(comparison.target == target) & comparison.model.isin(REFERENCE_MODELS)]
        best_ap = rank_best.loc[rank_best.average_precision_mean.idxmax()]
        best_auc = rank_best.loc[rank_best.auroc_mean.idxmax()]
        best_brier = rank_best.loc[rank_best.brier_score_mean.idxmin()]
        best_ll = rank_best.loc[rank_best.log_loss_mean.idxmin()]
        fold_pairs = paired[paired.target == target]
        win_ap = int(fold_pairs[fold_pairs.comparator_model == best_ap.model].tabpfn_better_average_precision.sum())
        ap_delta = float(tab.average_precision_mean - best_ap.average_precision_mean)
        if ap_delta > 0 and win_ap >= 4:
            targets_meeting_stability_screen.append(target)
        else:
            targets_not_meeting_stability_screen.append(target)
        lines.extend([
            f"### {target}", "",
            f"TabPFN AP {tab.average_precision_mean:.3f} ± {tab.average_precision_std:.3f}; best saved comparator AP: {best_ap.model} ({best_ap.average_precision_mean:.3f} ± {best_ap.average_precision_std:.3f}). TabPFN mean AP {'exceeded' if tab.average_precision_mean > best_ap.average_precision_mean else 'did not exceed'} the best comparator mean.",
            f"TabPFN AUROC {tab.auroc_mean:.3f} ± {tab.auroc_std:.3f}; {compare_to_best_mean(tab.auroc_mean, best_auc.auroc_mean, 'auroc', best_auc.model)}. Brier {tab.brier_score_mean:.3f} ± {tab.brier_score_std:.3f}; {compare_to_best_mean(tab.brier_score_mean, best_brier.brier_score_mean, 'brier_score', best_brier.model)}. Log Loss {tab.log_loss_mean:.3f} ± {tab.log_loss_std:.3f}; {compare_to_best_mean(tab.log_loss_mean, best_ll.log_loss_mean, 'log_loss', best_ll.model)}. Against the comparator with the highest mean AP, TabPFN won {win_ap}/5 paired AP folds.",
            f"At threshold 0.5, mean fold precision {tab.precision_at_0_5_mean:.3f}, recall {tab.recall_at_0_5_mean:.3f}, and F1 {tab.f1_at_0_5_mean:.3f}. Per-fold confusion counts are in `fold_level_metrics.csv`.", "",
        ])
    meeting_text = ", ".join(targets_meeting_stability_screen) if targets_meeting_stability_screen else "none"
    not_meeting_text = ", ".join(targets_not_meeting_stability_screen) if targets_not_meeting_stability_screen else "none"
    if targets_meeting_stability_screen:
        overall_recommendation = (
            f"The descriptive screen is met for {meeting_text} and not met for {not_meeting_text}. "
            "This supports at most a limited controlled comparison of TabPFN on the targets that passed, "
            "if separately authorized; it does not support adopting TabPFN, establish significance, or justify "
            "an automatic next experiment."
        )
    else:
        overall_recommendation = (
            "No target met the descriptive screen. The current evidence does not justify further TabPFN "
            "investigation beyond retaining its results as a comparator; it does not support adoption or "
            "establish significance, and no further experiment is started."
        )
    fibr_summary = comparison[comparison.target == "FIBR_PREDS"]
    fibr_tab = fibr_summary.loc[fibr_summary.model == TABPFN_MODEL].iloc[0]
    fibr_comparators = fibr_summary[fibr_summary.model.isin(REFERENCE_MODELS)]
    fibr_best_ap = fibr_comparators.loc[fibr_comparators.average_precision_mean.idxmax()]
    fibr_paired = paired[
        (paired.target == "FIBR_PREDS") &
        (paired.comparator_model == fibr_best_ap.model)
    ]
    fibr_ap_wins = int(fibr_paired.tabpfn_better_average_precision.sum())
    fibr_ap_delta = float(fibr_tab.average_precision_mean - fibr_best_ap.average_precision_mean)
    fibr_brier_delta = float(fibr_tab.brier_score_mean - fibr_best_ap.brier_score_mean)
    fibr_logloss_delta = float(fibr_tab.log_loss_mean - fibr_best_ap.log_loss_mean)
    fibr_ap_rule_met = fibr_ap_delta > 0 and fibr_ap_wins >= 4
    if fibr_ap_rule_met:
        fibr_recommendation = (
            f"Further focused FIBR_PREDS investigation is warranted under the predeclared descriptive rule: "
            f"TabPFN mean AP exceeds the best saved comparator ({fibr_best_ap.model}) by {fibr_ap_delta:+.3f} "
            f"and wins {fibr_ap_wins}/5 paired folds. Against that same comparator, Brier difference is "
            f"{fibr_brier_delta:+.3f} and Log Loss difference is {fibr_logloss_delta:+.3f} "
            f"({'both favor TabPFN' if fibr_brier_delta < 0 and fibr_logloss_delta < 0 else 'probability-quality metrics are mixed or do not favor TabPFN'}). "
            "Treat this as a hypothesis for a pre-specified follow-up, not evidence of statistical significance or clinical utility."
        )
    else:
        fibr_recommendation = (
            f"The current folds do not meet the descriptive rule for a dedicated FIBR_PREDS follow-up: "
            f"TabPFN mean AP difference versus the best saved comparator ({fibr_best_ap.model}) is "
            f"{fibr_ap_delta:+.3f}, with {fibr_ap_wins}/5 paired AP wins; the rule requires a positive mean AP "
            "difference and at least 4/5 wins. Against that comparator, Brier and Log Loss differences are "
            f"{fibr_brier_delta:+.3f} and {fibr_logloss_delta:+.3f}, respectively. "
            "If either probability-quality difference favors TabPFN, that is a separate exploratory signal, "
            "not a consistent ranking improvement; retain TabPFN as a comparator unless a focused question justifies follow-up."
        )
    runtimes = folds.fit_predict_runtime_seconds
    rss = folds.process_peak_rss_mb
    mps = folds.mps_driver_allocated_mb_after_fold.dropna()
    lines.extend([
        "## Overall stability recommendation", "",
        "Across targets, the transparent descriptive screen requires TabPFN mean AP to exceed the best saved comparator and win at least 4 of 5 paired folds against it. This is a practical stability screen only, not a significance test.",
        overall_recommendation, "",
        "## FIBR_PREDS follow-up recommendation", "",
        "For FIBR_PREDS, the transparent descriptive rule is that TabPFN must exceed the best saved comparator's mean AP and win at least 4 of 5 paired outer folds against that comparator. This is a pragmatic stability screen, not a significance test.",
        fibr_recommendation, "",
        "## Runtime, memory, and limitations", "",
        f"Runtime device status: CUDA available `{runtime_status['torch_cuda_available']}`, MPS built `{runtime_status['torch_mps_built']}`, MPS available `{runtime_status['torch_mps_available']}`; `device='auto'` outcome `{runtime_status['auto_device_outcome']}`.",
        f"Across {len(folds)} TabPFN fold fits, fit-plus-predict runtime was {runtimes.sum():.1f} seconds total (mean {runtimes.mean():.1f}s/fold; range {runtimes.min():.1f}–{runtimes.max():.1f}s). The first fit may include license authentication and lazy checkpoint download if the weights were not already cached. The largest recorded process peak RSS was {rss.max():.0f} MiB. " + (f"MPS driver allocation after folds ranged up to {mps.max():.0f} MiB." if len(mps) else "MPS driver allocation was unavailable; process RSS is reported instead."),
        "Five folds provide descriptive paired evidence only; they do not establish statistical significance. Clinical prediction metrics do not establish clinical utility. The model weights use a separate non-commercial/non-production license; see `tabpfn_configuration.md`. Memory references from official docs are A100 VRAM estimates, not a guarantee for Apple unified memory. Comparator metrics are from the project’s saved seed-42 fold results, with Stage 6 comparator Log Loss refits on matching folds.", "",
        "## Output files", "",
        "- `fold_level_metrics.csv`: TabPFN fold scores, threshold metrics, warnings, runtime and memory readings.",
        "- `aggregate_metrics.csv`, `model_comparison.csv`: mean and sample standard deviation across folds for TabPFN and saved comparison models.",
        "- `oof_predictions.csv`: one outer-validation probability per eligible development ID and target.",
        "- `paired_comparison.csv`: per-fold TabPFN-minus-comparator differences, with metric direction recorded by column names and booleans.",
        "- `tabpfn_configuration.md`, `stage8_report.md`, `model_comparison.png`.",
    ])
    (OUT / "stage8_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Stage 8 complete: {len(folds)} fold rows, {len(oof)} OOF probabilities; outputs: {OUT}")


if __name__ == "__main__":
    main()
