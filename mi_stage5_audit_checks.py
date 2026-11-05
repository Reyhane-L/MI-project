"""Static/development-only checks for T05: requirement and leakage audit.

Run from the project root with `.venv/bin/python mi_stage5_audit_checks.py`.
This checker intentionally opens only Stage 1 development artifacts and
development-CV summaries. It never enumerates, opens, or reads the locked
held-out partition.
"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT / "stage1_outputs"
TARGETS = {"DEATH_BINARY", "ZSN", "FIBR_PREDS"}
FORBIDDEN_POST_ADMISSION = {
    "R_AB_1_n", "R_AB_2_n", "R_AB_3_n", "NA_R_1_n", "NA_R_2_n",
    "NA_R_3_n", "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    # Explicit development-only paths; do not replace with a directory scan.
    x = pd.read_csv(STAGE1 / "X_train.csv")
    y = pd.read_csv(STAGE1 / "y_train.csv")
    dictionary = pd.read_csv(STAGE1 / "column_dictionary.csv")

    require(x.shape == (1360, 102), "development predictor artifact is 1,360 by 102")
    require(len(y) == len(x), "development predictors and outcomes have aligned row counts")
    require(not x.columns.duplicated().any(), "development predictor names are unique")
    require("ID" not in x.columns, "record ID is absent from development predictors")
    outcome_names = set(dictionary.loc[dictionary["stage1_role"].eq(
        "outcome; stored in y, excluded from predictors"), "variable"])
    require(len(outcome_names) == 12, "dictionary identifies all 12 source outcomes")
    require(not (outcome_names & set(x.columns)), "no source outcome is a development predictor")
    require(not (FORBIDDEN_POST_ADMISSION & set(x.columns)),
            "all nine documented post-admission fields are absent from development predictors")
    require(set(TARGETS | {"LET_IS"}).issubset(y.columns),
            "development outcomes include the three required targets and LET_IS")
    require(y["DEATH_BINARY"].eq(y["LET_IS"].gt(0).astype("int8")).all(),
            "DEATH_BINARY equals LET_IS > 0 for every development row")

    role_counts = dictionary["stage1_role"].value_counts().to_dict()
    require(role_counts == {
        "admission-time predictor": 102,
        "outcome; stored in y, excluded from predictors": 12,
        "post-admission value; excluded for admission-time prediction": 9,
        "record identifier; excluded from predictors": 1,
    }, "dictionary role counts match the admission-time feature policy")
    expected_features = dictionary.loc[dictionary["stage1_role"].eq(
        "admission-time predictor"), "variable"].tolist()
    require(x.columns.tolist() == expected_features,
            "development predictor order exactly matches the admission-time dictionary rows")

    stage1_source = (ROOT / "mi_stage1_eda.py").read_text(encoding="utf-8")
    require("RANDOM_STATE = 42" in stage1_source and "TEST_SIZE = 0.20" in stage1_source,
            "Stage 1 declares deterministic seed 42 and a 20% held-out split")
    require("stratify=y[\"DEATH_BINARY\"]" in stage1_source,
            "Stage 1 stratifies the held-out split by binary mortality")

    for script_name in ["mi_stage2_baselines.py", "mi_stage3_random_forest.py", "mi_stage4_extratrees.py"]:
        source = (ROOT / script_name).read_text(encoding="utf-8")
        require("X_train.csv" in source and "y_train.csv" in source,
                f"{script_name} declares development training inputs")
        require("Pipeline" in source and "StratifiedKFold" in source,
                f"{script_name} uses pipeline-based stratified CV")
    stage2_source = (ROOT / "mi_stage2_baselines.py").read_text(encoding="utf-8")
    require(all(name in stage2_source for name in ["DummyClassifier", "LogisticRegression",
                                                    "average_precision_score", "roc_auc_score",
                                                    "brier_score_loss"]),
            "Stage 2 implements required baseline/logistic families and AP, AUROC, Brier")
    for script_name in ["mi_stage3_random_forest.py", "mi_stage4_extratrees.py"]:
        source = (ROOT / script_name).read_text(encoding="utf-8")
        require("RandomizedSearchCV" in source and 'scoring="average_precision"' in source,
                f"{script_name} selects tree settings by development-only average precision")

    required_metrics = {"average_precision_mean", "auroc_mean", "brier_score_mean"}
    expected_models = {
        "stage2_outputs/aggregate_metrics.csv": {"Dummy prior prevalence", "Logistic regression"},
        "stage3_outputs/aggregate_metrics.csv": {"Random Forest preliminary", "Random Forest tuned"},
        "stage4_extratrees_outputs/aggregate_metrics.csv": {"Extra Trees preliminary", "Extra Trees tuned"},
    }
    for relative_path, models in expected_models.items():
        summary = pd.read_csv(ROOT / relative_path)
        require(TARGETS.issubset(set(summary["target"])),
                f"{relative_path} reports all three required targets")
        require(models.issubset(set(summary["model"])),
                f"{relative_path} reports its required model family")
        require(required_metrics.issubset(summary.columns),
                f"{relative_path} includes AP, AUROC, and Brier summaries")

    print("T05 static/development-only audit checks passed.")


if __name__ == "__main__":
    main()
