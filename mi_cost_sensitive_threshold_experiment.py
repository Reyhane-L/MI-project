from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OOF_PATH = ROOT / "stage10_t06_non_tabpfn_outputs" / "cross_fitted_oof_predictions.csv"
FROZEN_PATH = ROOT / "stage10_t06_non_tabpfn_outputs" / "selected_thresholds.csv"
OUT = ROOT / "mi_cost_sensitive_threshold_outputs_v2"
TARGET = "DEATH_BINARY"
MODEL = "CatBoost tuned"
FN_COST = 10
FP_COST = 1
REFERENCE_THRESHOLDS = {"default_0.5": 0.5, "frozen_sensitivity_0.8": None}
OUTPUTS = (
    "threshold_cost_curve.csv", "threshold_comparison.csv",
    "prediction_level.csv", "confusion_matrices.csv", "cost_curve.png",
    "sensitivity_specificity.png", "experiment_report.md", "run_manifest.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(oof: pd.DataFrame, frozen: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    required_oof = {"target", "model", "ID", "true_label", "outer_fold", "platt_probability"}
    if not required_oof.issubset(oof.columns):
        raise ValueError(f"OOF file is missing required columns: {sorted(required_oof - set(oof.columns))}")
    rows = oof.loc[(oof.target == TARGET) & (oof.model == MODEL)].copy()
    if len(rows) != 1360:
        raise ValueError(f"Expected exactly 1,360 {TARGET}/{MODEL} development OOF rows; found {len(rows)}.")
    if rows["ID"].isna().any() or rows["ID"].duplicated().any():
        raise ValueError("Selected development OOF IDs must be non-null and unique.")
    labels = pd.to_numeric(rows.true_label, errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}) or set(labels.unique()) != {0, 1}:
        raise ValueError("Development OOF mortality labels must contain both exact binary classes.")
    probabilities = pd.to_numeric(rows.platt_probability, errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all():
        raise ValueError("Platt OOF probabilities must be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Platt OOF probabilities must be within [0, 1].")
    folds = pd.to_numeric(rows.outer_fold, errors="coerce")
    if folds.isna().any() or set(folds.astype(int).unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected exactly five outer-fold labels.")
    required_frozen = {"target", "threshold", "status", "selection_criterion"}
    if not required_frozen.issubset(frozen.columns):
        raise ValueError("Frozen threshold file lacks required columns.")
    chosen = frozen.loc[frozen.target == TARGET]
    if len(chosen) != 1 or chosen.iloc[0].status != "constraint_met":
        raise ValueError("Expected one valid frozen DEATH_BINARY threshold.")
    threshold = float(chosen.iloc[0].threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Frozen threshold must be finite and in [0, 1].")
    rows["true_label"] = labels.astype(np.int8)
    rows["platt_probability"] = probabilities.astype(float)
    rows["outer_fold"] = folds.astype(np.int8)
    return rows.reset_index(drop=True), threshold


def _threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    predicted = np.asarray(p) >= threshold
    labels = np.asarray(y, dtype=np.int8)
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    positives = tp + fn
    negatives = tn + fp
    n = len(labels)
    alerts = tp + fp
    return {
        "threshold": float(threshold), "n": n,
        "observed_prevalence": positives / n,
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
        "cost_fn10_fp1": FN_COST * fn + FP_COST * fp,
        "sensitivity": tp / positives if positives else float("nan"),
        "specificity": tn / negatives if negatives else float("nan"),
        "precision": tp / alerts if alerts else 0.0,
        "alerts": alerts, "alert_rate": alerts / n,
        "true_deaths_identified_per_100": tp / n * 100,
        "missed_deaths": fn,
    }


def analyze(rows: pd.DataFrame, frozen_threshold: float) -> dict:
    y = rows.true_label.to_numpy(dtype=np.int8)
    p = rows.platt_probability.to_numpy(dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both mortality classes are required.")
    candidates = np.unique(np.concatenate(([0.0], p, [np.nextafter(float(np.max(p)), np.inf)])))
    curve = pd.DataFrame([_threshold_metrics(y, p, threshold) for threshold in candidates])
    minimum_cost = int(curve.cost_fn10_fp1.min())
    selected = curve.loc[curve.cost_fn10_fp1 == minimum_cost].sort_values("threshold").iloc[-1]
    optimal_threshold = float(selected.threshold)
    references = {
        "cost_minimizing_development_oof": optimal_threshold,
        "default_0.5": 0.5,
        "frozen_sensitivity_0.8": float(frozen_threshold),
    }
    comparison = pd.DataFrame([
        {"rule": rule, "threshold_source": (
            "empirical minimum 10*FN+FP over development OOF threshold candidates; highest threshold breaks ties"
            if rule == "cost_minimizing_development_oof" else
            "fixed reference threshold"
        ), **_threshold_metrics(y, p, threshold)}
        for rule, threshold in references.items()
    ])
    confusions = comparison[[
        "rule", "threshold", "true_negatives", "false_positives",
        "false_negatives", "true_positives", "cost_fn10_fp1",
    ]].copy()
    predictions = rows[["ID", "true_label", "outer_fold", "platt_probability"]].copy()
    for rule, threshold in references.items():
        predictions[f"predicted_{rule}"] = (p >= threshold).astype(np.int8)
    return {"curve": curve, "comparison": comparison,
            "confusions": confusions, "predictions": predictions,
            "optimal_threshold": optimal_threshold,
            "candidates": int(len(candidates))}


def _plot_outputs(result: dict, out: Path) -> None:
    curve = result["curve"]
    comparison = result["comparison"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(curve.threshold, curve.cost_fn10_fp1, color="#315b7d", linewidth=1.4)
    ax.scatter([result["optimal_threshold"]], [curve.cost_fn10_fp1.min()], color="#c04b35", zorder=3)
    ax.set(xlabel="Probability threshold (predict death if p ≥ threshold)",
           ylabel="Empirical cost: 10 × FN + FP", title="Development OOF cost by threshold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / OUTPUTS[4], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for row in comparison.itertuples(index=False):
        ax.scatter(row.specificity, row.sensitivity, s=65, label=row.rule)
        ax.annotate(f"{row.rule}\nt={row.threshold:.3f}", (row.specificity, row.sensitivity),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Specificity", ylabel="Sensitivity",
           title="Development OOF operating points")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / OUTPUTS[5], dpi=180)
    plt.close(fig)


def write_outputs(result: dict, rows: pd.DataFrame, frozen_threshold: float,
                  out: Path = OUT) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    # Preserve adjacent floating-point cutoffs, including nextafter(max(p), +inf).
    result["curve"].to_csv(out / OUTPUTS[0], index=False, float_format="%.17g")
    result["comparison"].to_csv(out / OUTPUTS[1], index=False, float_format="%.12g")
    result["predictions"].to_csv(out / OUTPUTS[2], index=False, float_format="%.12g")
    result["confusions"].to_csv(out / OUTPUTS[3], index=False, float_format="%.12g")
    _plot_outputs(result, out)
    table = result["comparison"].set_index("rule")
    optimal = table.loc["cost_minimizing_development_oof"]
    prevalence = float(rows.true_label.mean())
    lines = [
        "# Cost-sensitive mortality threshold experiment", "",
        "This post-processing experiment uses only the frozen T06 DEATH_BINARY outer development OOF Platt probabilities. It does not fit a model or calibrator and does not read Stage 11 test results.", "",
        "Decision rule: predict death when calibrated probability is greater than or equal to the threshold. Empirical cost is `10 × false negatives + false positives`; all thresholds in `{0, unique OOF probabilities, nextafter(max probability, +∞)}` were evaluated. Minimum-cost ties resolve to the highest threshold.", "",
        f"The observed OOF death prevalence is **{prevalence:.4f}** among **N=1,360** patients. The cost-minimizing threshold is **{optimal.threshold:.12g}**, with cost **{int(optimal.cost_fn10_fp1)}**. At this threshold, **{optimal.alerts / optimal.n * 100:.2f} alerts per 100 patients** are generated and **{optimal.true_deaths_identified_per_100:.2f} deaths per 100 patients** are identified.", "",
        "| Rule | Threshold | Cost | TP | FP | TN | FN | Sensitivity | Specificity | Precision | Alerts | Alert rate | Deaths identified per 100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["comparison"].itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.threshold:.6f} | {int(row.cost_fn10_fp1)} | {int(row.true_positives)} | {int(row.false_positives)} | {int(row.true_negatives)} | {int(row.false_negatives)} | {row.sensitivity:.4f} | {row.specificity:.4f} | {row.precision:.4f} | {int(row.alerts)} | {row.alert_rate:.4f} | {row.true_deaths_identified_per_100:.3f} |"
        )
    lines += ["", "‘Deaths identified per 100’ is TP divided by all evaluated patients; sensitivity is TP divided only by observed deaths. This operating-point analysis is descriptive and does not change the frozen T06 threshold or model.", ""]
    (out / OUTPUTS[6]).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {OUT}")
    for path in (OOF_PATH, FROZEN_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    oof, frozen = pd.read_csv(OOF_PATH), pd.read_csv(FROZEN_PATH)
    rows, frozen_threshold = _validate_inputs(oof, frozen)
    result = analyze(rows, frozen_threshold)
    manifest = {
        "experiment": "Development-only cost-sensitive mortality threshold analysis",
        "target": TARGET, "model": MODEL, "probability_column": "platt_probability",
        "inputs": {
            "development_oof_predictions": str(OOF_PATH.relative_to(ROOT)),
            "frozen_development_thresholds": str(FROZEN_PATH.relative_to(ROOT)),
            "sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in (OOF_PATH, FROZEN_PATH)},
        },
        "test_partition_accessed": False,
        "model_fitted": False, "calibration_fitted": False,
        "cost": {"false_negative": FN_COST, "false_positive": FP_COST,
                 "formula": "10*FN+FP"},
        "decision_rule": "predict death when calibrated probability >= threshold",
        "candidate_thresholds": "0, all unique development OOF probabilities, nextafter(max probability, +infinity)",
        "tie_break": "choose highest threshold among thresholds with minimum empirical cost",
        "references": {"default": 0.5, "frozen_development": frozen_threshold},
        "cv_provenance": "Existing T06 five-fold seed-42 outer development OOF predictions; no fold recreated.",
        "n_oof_rows": int(len(rows)), "n_unique_ids": int(rows.ID.nunique()),
        "n_deaths": int(rows.true_label.sum()), "observed_prevalence": float(rows.true_label.mean()),
        "n_threshold_candidates": result["candidates"],
        "selected_threshold": result["optimal_threshold"],
        "command": "PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp .venv/bin/python -B mi_cost_sensitive_threshold_experiment.py",
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "matplotlib": importlib.metadata.version("matplotlib"),
        },
        "output_files": list(OUTPUTS),
    }
    write_outputs(result, rows, frozen_threshold)
    (OUT / OUTPUTS[7]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(result["comparison"].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Threshold candidates: {result['candidates']}; outputs: {OUT}")


if __name__ == "__main__":
    main()
