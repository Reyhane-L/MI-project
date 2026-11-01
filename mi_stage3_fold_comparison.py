"""Compare Stage 3 Random Forest and Stage 2 Logistic Regression by CV fold.

Run from the project root with `.venv/bin/python mi_stage3_fold_comparison.py`.
The comparison is paired by target and fold and uses no test data.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage3-mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STAGE2_FILE = ROOT / "stage2_outputs" / "fold_level_metrics.csv"
STAGE3_FILE = ROOT / "stage3_outputs" / "fold_level_metrics.csv"
OUTPUT_DIR = ROOT / "stage3_outputs"
TARGETS = ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]
RF_MODELS = ["Random Forest preliminary", "Random Forest tuned"]
METRICS = ["average_precision", "auroc", "brier_score"]


def main() -> None:
    for path in [STAGE2_FILE, STAGE3_FILE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    stage2 = pd.read_csv(STAGE2_FILE)
    stage3 = pd.read_csv(STAGE3_FILE)
    required = {"target", "model", "fold", *METRICS}
    if not required.issubset(stage2.columns) or not required.issubset(stage3.columns):
        raise ValueError("Fold-level metric files are missing required columns.")

    lr = stage2.loc[(stage2["model"] == "Logistic regression") &
                    stage2["target"].isin(TARGETS)].copy()
    rf = stage3.loc[stage3["model"].isin(RF_MODELS) &
                    stage3["target"].isin(TARGETS)].copy()
    if lr.duplicated(["target", "fold"]).any():
        raise ValueError("Stage 2 Logistic Regression has duplicate target/fold rows.")
    if rf.duplicated(["target", "fold", "model"]).any():
        raise ValueError("Stage 3 Random Forest has duplicate target/fold/model rows.")
    if set(lr["target"].unique()) != set(TARGETS) or set(rf["target"].unique()) != set(TARGETS):
        raise ValueError("One or more expected targets are missing from the fold files.")

    validation_cols = [c for c in ["n_validation", "validation_positive_cases",
                                    "validation_negative_cases"]
                       if c in lr.columns and c in rf.columns]
    base = lr[["target", "fold", *METRICS, *validation_cols]].rename(
        columns={**{m: f"logistic_{m}" for m in METRICS},
                 **{c: f"lr_{c}" for c in validation_cols}})
    paired = rf[["target", "fold", "model", *METRICS, *validation_cols]].merge(
        base, on=["target", "fold"], how="outer", validate="many_to_one",
        indicator=True)
    if not paired["_merge"].eq("both").all():
        missing = paired.loc[paired["_merge"] != "both", ["target", "fold", "model", "_merge"]]
        raise ValueError(f"Could not pair all rows by target and fold:\n{missing}")
    paired = paired.drop(columns="_merge")
    for col in validation_cols:
        if not paired[col].eq(paired[f"lr_{col}"]).all():
            raise ValueError(f"Stage 2 and Stage 3 folds differ in {col}.")
    paired = paired.drop(columns=[f"lr_{c}" for c in validation_cols], errors="ignore")

    for metric in METRICS:
        paired[f"logistic_{metric}"] = paired[f"logistic_{metric}"]
        paired[f"rf_minus_logistic_{metric}"] = paired[metric] - paired[f"logistic_{metric}"]
        better = (paired[f"rf_minus_logistic_{metric}"] < 0 if metric == "brier_score"
                  else paired[f"rf_minus_logistic_{metric}"] > 0)
        tied = np.isclose(paired[f"rf_minus_logistic_{metric}"], 0.0, atol=1e-12)
        paired[f"rf_beats_logistic_{metric}"] = better
        paired[f"tie_{metric}"] = tied

    OUTPUT_DIR.mkdir(exist_ok=True)
    paired.to_csv(OUTPUT_DIR / "paired_fold_differences.csv", index=False)

    summaries = []
    for (target, model), group in paired.groupby(["target", "model"], sort=False):
        for metric in METRICS:
            diff = group[f"rf_minus_logistic_{metric}"]
            summaries.append({
                "target": target, "random_forest_model": model, "metric": metric,
                "folds": len(group), "rf_wins": int(group[f"rf_beats_logistic_{metric}"].sum()),
                "ties": int(group[f"tie_{metric}"].sum()),
                "logistic_wins": int(len(group) - group[f"rf_beats_logistic_{metric}"].sum()
                                     - group[f"tie_{metric}"].sum()),
                "mean_rf_minus_logistic": float(diff.mean()),
                "std_paired_difference": float(diff.std(ddof=1)),
            })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "paired_difference_summary.csv", index=False)

    # Each dot is one fold's RF-minus-logistic paired difference; zero marks a tie.
    metric_titles = {
        "average_precision": "AP difference (RF − Logistic Regression; higher is better)",
        "auroc": "AUROC difference (RF − Logistic Regression; higher is better)",
        "brier_score": "Brier difference (RF − Logistic Regression; lower is better)",
    }
    colors = {"Random Forest preliminary": "#70AD47", "Random Forest tuned": "#ED7D31"}
    offsets = {"Random Forest preliminary": -0.12, "Random Forest tuned": 0.12}
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, METRICS):
        col = f"rf_minus_logistic_{metric}"
        for ti, target in enumerate(TARGETS):
            for model in RF_MODELS:
                values = paired.loc[(paired.target == target) & (paired.model == model), col].to_numpy()
                jitter = rng.uniform(-0.035, 0.035, size=len(values))
                x = ti + offsets[model] + jitter
                ax.scatter(x, values, s=30, alpha=.8, color=colors[model],
                           label=model if ti == 0 else None)
                mean = float(values.mean())
                ax.scatter(ti + offsets[model], mean, marker="D", s=48,
                           edgecolor="black", linewidth=.6, color=colors[model], zorder=3)
        ax.axhline(0, color="black", linewidth=1, alpha=.7)
        ax.set_xticks(range(len(TARGETS)), ["Death", "ZSN", "FIBR_PREDS"])
        ax.set_title(metric_titles[metric], fontsize=9)
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Paired five-fold differences: Random Forest vs Logistic Regression\n"
                 "Dots = individual folds; diamonds = mean difference")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "paired_fold_differences.png", dpi=180)
    plt.close(fig)

    print("Paired folds:", len(paired))
    print(summary.to_string(index=False))
    print(f"Saved CSVs and plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
