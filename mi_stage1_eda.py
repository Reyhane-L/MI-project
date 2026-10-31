"""Stage 1: prepare an admission-time MI cohort and describe training data.

Run from the project root with: .venv/bin/python mi_stage1_eda.py
The script does not fit a predictive model or inspect test-set feature values
for exploratory analysis.
"""

from pathlib import Path
import os
import tempfile


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mi-stage1-mplconfig"))
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "MI.data"
DICTIONARY_PATH = ROOT / "mi_column_dictionary.csv"
OUTPUT_DIR = ROOT / "stage1_outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Ordered exactly as the user-provided MI data dictionary: ID, 111 inputs,
# then 12 outcomes. Keeping every official abbreviation makes output tables
# interpretable and preserves alignment with source positions.
SOURCE_COLUMN_NAMES = [
    "ID", "AGE", "SEX", "INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST",
    "IBS_NASL", "GB", "SIM_GIPERT", "DLIT_AG", "ZSN_A", "nr11", "nr01",
    "nr02", "nr03", "nr04", "nr07", "nr08", "np01", "np04", "np05",
    "np07", "np08", "np09", "np10", "endocr_01", "endocr_02",
    "endocr_03", "zab_leg_01", "zab_leg_02", "zab_leg_03", "zab_leg_04",
    "zab_leg_06", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT",
    "O_L_POST", "K_SH_POST", "MP_TP_POST", "SVT_POST", "GT_POST",
    "FIB_G_POST", "ant_im", "lat_im", "inf_im", "post_im", "IM_PG_P",
    "ritm_ecg_p_01", "ritm_ecg_p_02", "ritm_ecg_p_04", "ritm_ecg_p_06",
    "ritm_ecg_p_07", "ritm_ecg_p_08", "n_r_ecg_p_01", "n_r_ecg_p_02",
    "n_r_ecg_p_03", "n_r_ecg_p_04", "n_r_ecg_p_05", "n_r_ecg_p_06",
    "n_r_ecg_p_08", "n_r_ecg_p_09", "n_r_ecg_p_10", "n_p_ecg_p_01",
    "n_p_ecg_p_03", "n_p_ecg_p_04", "n_p_ecg_p_05", "n_p_ecg_p_06",
    "n_p_ecg_p_07", "n_p_ecg_p_08", "n_p_ecg_p_09", "n_p_ecg_p_10",
    "n_p_ecg_p_11", "n_p_ecg_p_12", "fibr_ter_01", "fibr_ter_02",
    "fibr_ter_03", "fibr_ter_05", "fibr_ter_06", "fibr_ter_07",
    "fibr_ter_08", "GIPO_K", "K_BLOOD", "GIPER_Na", "Na_BLOOD",
    "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD", "L_BLOOD", "ROE", "TIME_B_S",
    "R_AB_1_n", "R_AB_2_n", "R_AB_3_n", "NA_KB", "NOT_NA_KB", "LID_KB",
    "NITR_S", "NA_R_1_n", "NA_R_2_n", "NA_R_3_n", "NOT_NA_1_n",
    "NOT_NA_2_n", "NOT_NA_3_n", "LID_S_n", "B_BLOK_S_n", "ANT_CA_S_n",
    "GEPAR_S_n", "ASP_S_n", "TIKL_S_n", "TRENT_S_n", "FIBR_PREDS",
    "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK", "OTEK_LANC",
    "RAZRIV", "DRESSLER", "ZSN", "REC_IM", "P_IM_STEN", "LET_IS",
]
TARGET_NAMES = SOURCE_COLUMN_NAMES[112:]
FORBIDDEN_ADMISSION_FEATURES = [
    "R_AB_1_n", "R_AB_2_n", "R_AB_3_n", "NA_R_1_n", "NA_R_2_n",
    "NA_R_3_n", "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
]


def load_dataset() -> pd.DataFrame:
    """Read the headerless comma-delimited file and turn '?' into NaN."""
    raw = pd.read_csv(DATA_PATH, header=None, na_values=["?"], dtype="string")
    if raw.shape[1] != 124:
        raise ValueError(f"Expected 124 source columns, found {raw.shape[1]}.")

    raw.columns = SOURCE_COLUMN_NAMES
    # Convert values to numeric after missing markers have been parsed.
    data = raw.apply(pd.to_numeric, errors="coerce")
    if data["LET_IS"].isna().any():
        raise ValueError("LET_IS contains missing/non-numeric values; inspect source.")
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_dataset()
    column_metadata = pd.read_csv(DICTIONARY_PATH)
    if column_metadata["variable"].tolist() != SOURCE_COLUMN_NAMES:
        raise ValueError("Column dictionary order does not match the dataset schema.")
    descriptions = column_metadata.set_index("variable")["description"]

    # The ID is only a row identifier.  The guide excludes post-arrival
    # observations (one-based source columns 93-95 and 100-105) for admission
    # prediction. Targets are kept in y, never in the model-input table X.
    id_column = "ID"
    feature_columns = [
        name for name in data.columns
        if name != id_column and name not in TARGET_NAMES
        and name not in FORBIDDEN_ADMISSION_FEATURES
    ]
    X = data[feature_columns].copy()
    y = data[TARGET_NAMES].copy()
    y["DEATH_BINARY"] = (y["LET_IS"] > 0).astype("int8")

    # Section 9: 20% held out once, stratified by binary mortality class;
    # random_state makes row allocation repeatable. All EDA below uses train.
    row_positions = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        row_positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y["DEATH_BINARY"],
    )
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    # Keep split outputs auditable and reusable; identifiers are not predictors.
    train_ids = data.iloc[train_idx][id_column].reset_index(drop=True)
    test_ids = data.iloc[test_idx][id_column].reset_index(drop=True)
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)
    train_ids.to_csv(OUTPUT_DIR / "train_record_ids.csv", index=False)
    test_ids.to_csv(OUTPUT_DIR / "test_record_ids.csv", index=False)

    # Document every source column, its source index, and its stage 1 role.
    roles = []
    for position, name in enumerate(SOURCE_COLUMN_NAMES, start=1):
        if name == id_column:
            role = "record identifier; excluded from predictors"
        elif name in TARGET_NAMES:
            role = "outcome; stored in y, excluded from predictors"
        elif name in FORBIDDEN_ADMISSION_FEATURES:
            role = "post-admission value; excluded for admission-time prediction"
        else:
            role = "admission-time predictor"
        roles.append(role)
    column_metadata["stage1_role"] = roles
    column_metadata.to_csv(OUTPUT_DIR / "column_dictionary.csv", index=False)

    # Training-only missingness counts and rates; missing values remain missing.
    missing = pd.DataFrame({
        "feature": X_train.columns,
        "missing_count": X_train.isna().sum().to_numpy(),
        "missing_rate": X_train.isna().mean().to_numpy(),
    }).sort_values(["missing_rate", "feature"], ascending=[False, True])
    missing["description"] = missing["feature"].map(descriptions)
    missing.to_csv(OUTPUT_DIR / "training_missingness.csv", index=False)

    # Compact descriptive statistics support both continuous and coded fields.
    desc = X_train.describe().T.rename_axis("feature").reset_index()
    desc["description"] = desc["feature"].map(descriptions)
    desc.to_csv(OUTPUT_DIR / "training_feature_descriptives.csv", index=False)

    outcome_rows = []
    for outcome in ["DEATH_BINARY", "ZSN", "FIBR_PREDS"]:
        counts = y_train[outcome].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            outcome_rows.append({
                "outcome": outcome,
                "description": (
                    "Binary lethal outcome derived from LET_IS: 1-7 means death; 0 means no lethal outcome"
                    if outcome == "DEATH_BINARY" else descriptions[outcome]
                ),
                "value": value,
                "count": int(count),
                "proportion": count / len(y_train),
            })
    outcomes = pd.DataFrame(outcome_rows)
    outcomes.to_csv(OUTPUT_DIR / "training_outcome_frequencies.csv", index=False)

    # Missingness: show the 20 most-missing training predictors for readability.
    top_missing = missing.head(20).sort_values("missing_rate")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_missing["feature"], top_missing["missing_rate"] * 100,
            color="#4472C4")
    ax.set_xlabel("Missing training values (%)")
    ax.set_title("Training predictors with the most missing values")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_missingness_top20.png", dpi=180)
    plt.close(fig)

    # AGE is continuous. SEX is binary-coded; the plot shows observed counts
    # explicitly and excludes missing entries only for the histogram/bar.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(X_train["AGE"].dropna(), bins=20,
                 color="#4472C4", edgecolor="white")
    axes[0].set(title="Age in training set", xlabel="Age (years)", ylabel="Patients")
    sex_counts = X_train["SEX"].value_counts(dropna=False).sort_index()
    sex_labels = {0: "0 (female)", 1: "1 (male)"}
    axes[1].bar([sex_labels.get(value, str(value)) for value in sex_counts.index],
                sex_counts.values, color="#70AD47")
    axes[1].set(title="Sex in training set", xlabel="SEX code (guide mapping)", ylabel="Patients")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_age_sex.png", dpi=180)
    plt.close(fig)

    # Compare training mortality proportions across the two recorded sex codes.
    plot_data = pd.DataFrame({"SEX": X_train["SEX"], "death": y_train["DEATH_BINARY"]})
    mortality_by_sex = plot_data.groupby("SEX", dropna=False)["death"].agg(
        patients="size", deaths="sum", mortality_rate="mean"
    ).reset_index()
    mortality_by_sex.to_csv(OUTPUT_DIR / "training_mortality_by_sex.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([sex_labels.get(value, str(value)) for value in mortality_by_sex["SEX"]],
           mortality_by_sex["mortality_rate"] * 100, color="#ED7D31")
    ax.set(title="Training mortality rate by recorded sex code",
           xlabel="SEX code (guide mapping)", ylabel="Deaths / patients (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_mortality_by_sex.png", dpi=180)
    plt.close(fig)

    # Compare admission-time features across the two mortality classes and
    # summarize all recorded outcomes within clinically meaningful groups.
    # K_SH_POST is the dictionary's cardiogenic-shock-at-ICU-admission field.
    class_features = ["AGE", "SEX", "K_SH_POST", "O_L_POST", "INF_ANAM"]
    feature_plot, feature_axes = plt.subplots(2, 3, figsize=(14, 8))
    feature_axes = feature_axes.ravel()
    for feature, ax in zip(class_features, feature_axes):
        values = X_train[feature]
        if feature == "AGE":
            for death, label, color in [(0, "Survived", "#4472C4"),
                                        (1, "Died", "#C0504D")]:
                observed = values[y_train["DEATH_BINARY"].eq(death)].dropna()
                ax.hist(observed, bins=np.arange(20, 101, 5), alpha=0.65,
                        label=label, color=color, edgecolor="white")
            ax.set_xlabel("Age (years)")
            ax.legend(frameon=False)
        else:
            groups = []
            labels = []
            for death, class_label in [(0, "Survived"), (1, "Died")]:
                subset = values[y_train["DEATH_BINARY"].eq(death)].dropna()
                frequencies = subset.value_counts(normalize=True).sort_index()
                groups.append(frequencies)
                labels.append(class_label)
            categories = sorted(set().union(*(g.index for g in groups)))
            width = 0.36
            for j, (freq, label, color) in enumerate(zip(
                    groups, labels, ["#4472C4", "#C0504D"])):
                positions = np.arange(len(categories)) + (j - 0.5) * width
                ax.bar(positions, [freq.get(cat, 0) * 100 for cat in categories],
                       width=width, label=label, color=color)
            ax.set_xticks(np.arange(len(categories)))
            ax.set_xticklabels([str(int(c)) if float(c).is_integer() else str(c)
                                for c in categories])
            ax.set_xlabel("Recorded code")
            ax.legend(frameon=False)
        ax.set_title(descriptions[feature])
        ax.set_ylabel("Patients" if feature == "AGE" else "Within-class patients (%)")
        ax.grid(axis="y", alpha=0.2)
    feature_axes[-1].axis("off")
    feature_plot.suptitle("Admission feature distributions by mortality class")
    feature_plot.tight_layout()
    feature_plot.savefig(OUTPUT_DIR / "training_features_by_mortality_class.png", dpi=180)
    plt.close(feature_plot)

    # Group patients by age bands, recorded sex, and cardiogenic shock status.
    # For each group, report the observed frequency of every outcome. Missing
    # feature/outcome entries are omitted from that outcome's denominator.
    comparisons = {
        "SEX": {0: "Female", 1: "Male"},
        "K_SH_POST": {0: "No shock", 1: "Shock"},
        "AGE_GROUP": {"<60": "<60", "60-69": "60-69", "70-79": "70-79", "80+": "80+"},
    }
    group_data = {
        "SEX": X_train["SEX"].map(comparisons["SEX"]),
        "K_SH_POST": X_train["K_SH_POST"].map(comparisons["K_SH_POST"]),
        "AGE_GROUP": pd.cut(X_train["AGE"], bins=[0, 60, 70, 80, np.inf],
                             labels=["<60", "60-69", "70-79", "80+"] ),
    }
    outcomes_to_compare = [name for name in TARGET_NAMES if name != "LET_IS"]
    outcomes_to_compare.append("DEATH_BINARY")
    outcome_labels = {name: descriptions[name] for name in TARGET_NAMES if name != "LET_IS"}
    outcome_labels["DEATH_BINARY"] = "Death (LET_IS > 0)"
    comparison_rows = []
    for feature, groups in group_data.items():
        for group in groups.dropna().unique():
            mask = groups.eq(group)
            for outcome in outcomes_to_compare:
                observed = y_train.loc[mask, outcome].dropna()
                positive = observed.gt(0)
                comparison_rows.append({
                    "grouping_feature": feature,
                    "group": str(group),
                    "outcome": outcome,
                    "outcome_description": outcome_labels[outcome],
                    "patients_with_observed_outcome": int(len(observed)),
                    "positive_outcome_count": int(positive.sum()),
                    "positive_outcome_rate": float(positive.mean()) if len(observed) else np.nan,
                })
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(OUTPUT_DIR / "training_outcome_frequencies_by_group.csv", index=False)

    # Focus the visualization on mortality rates to keep all three comparisons
    # legible; the CSV includes frequencies for each of the 12 outcomes.
    mortality_rows = comparison_table[comparison_table["outcome"] == "DEATH_BINARY"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for feature, color in [("SEX", "#4472C4"), ("K_SH_POST", "#C0504D"),
                           ("AGE_GROUP", "#70AD47")]:
        subset = mortality_rows[mortality_rows["grouping_feature"] == feature]
        labels = [f"{feature}: {group}" for group in subset["group"]]
        ax.bar(labels, subset["positive_outcome_rate"] * 100,
               label=feature, color=color)
    ax.set_ylabel("Deaths / patients with observed outcome (%)")
    ax.set_title("Training mortality rate across admission groups")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Grouping feature", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_mortality_by_group.png", dpi=180)
    plt.close(fig)

    split_summary = pd.DataFrame([
        {"partition": "train", "patients": len(train_idx),
         "deaths": int(y_train["DEATH_BINARY"].sum()),
         "mortality_rate": y_train["DEATH_BINARY"].mean()},
        {"partition": "test", "patients": len(test_idx),
         "deaths": int(y_test["DEATH_BINARY"].sum()),
         "mortality_rate": y_test["DEATH_BINARY"].mean()},
    ])
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    # Provide a readable tutorial beside the machine-readable outputs.
    tutorial = f"""# Stage 1: MI data loading, split, and training EDA

Run this stage from the project directory with `.venv/bin/python mi_stage1_eda.py`.

## What the code does

1. **Reads the file.** `pd.read_csv(..., header=None, na_values=[\"?\"])` treats the file as headerless and converts its question-mark missing markers into pandas missing values. The 124-column guard catches an unexpected file layout early.
2. **Names and separates columns.** The ordered headers match all 124 variables in the supplied data dictionary. The first source column is a record ID. Source columns 113-124 are outcomes. For admission-time prediction, source columns 93-95 and 100-105 (pain recurrence and ICU opioid/NSAID counts) are excluded from `X`; outcomes stay in `y` for the requested EDA and future use. `column_dictionary.csv` gives each source position, official abbreviation, plain-language description, and role in this stage.
3. **Defines mortality.** `LET_IS` uses 0 for no lethal outcome and 1-7 for cause-specific death. The guide's binary endpoint is therefore `(LET_IS > 0)`, with one meaning death.
4. **Splits once using Section 9.** `train_test_split` reserves 20% for testing, uses `stratify=DEATH_BINARY` to preserve the mortality mix, and fixes `random_state={RANDOM_STATE}` to make the split reproducible. Only training predictors feed the EDA tables and figures. Test files are written for later stages, but this stage does not inspect their feature distributions.
5. **Summarizes training data.** The missingness table reports both the number and fraction missing in each allowed predictor. `describe()` gives numeric summaries. Outcome frequency and sex-stratified mortality tables make the class balance visible.
6. **Draws figures.** The first figure shows the 20 predictors with the highest training missingness; the second describes age and recorded sex code; the third shows observed mortality proportions by sex code. Histograms and proportions omit missing values only in the specific plot/statistic that needs an observed value; the data itself is not imputed or altered.

## Statistical calculations in the tables

- For each feature, `missing_count = sum(is_missing)` and `missing_rate = missing_count / 1360` (the training sample size). The rate makes columns comparable even if the split size changes.
- `describe()` reports observed count, arithmetic mean, sample standard deviation, minimum, quartiles, and maximum. Those summaries are literal for continuous fields such as age; for numeric category codes they are useful for a quick data check, not as category averages.
- Each outcome proportion is `count(value) / 1360`. For mortality by sex code, the displayed rate is `number of deaths in the code group / number of training patients in that group`; it is descriptive and does not establish causation.

## Reading the figures

- The age histogram groups observed ages into 20 bins so its shape and range are visible. The neighboring sex-code chart uses raw counts rather than proportions so the sample size for each code is clear.
- The missingness chart orders the 20 most incomplete allowed predictors by their training missing rate. It is a focused view; the CSV table includes all 102 predictors.
- The mortality-by-sex chart plots group-specific death proportions (not raw death counts), so groups of different sizes can be compared visually. The numeric values and group sizes are also in the matching CSV table.

## Important interpretation

The supplied dictionary documents every column, so the tables now use its variable abbreviations rather than positional placeholders. It defines `SEX=0` as female and `SEX=1` as male, and the figures use those labels. Missingness is descriptive only: no imputation, scaling, feature selection, or model fitting occurs here.

## Output files

CSV files contain the train/test predictors, outcomes, IDs, split counts, feature summaries, missingness, outcome frequencies, and mortality by sex code. PNG files are ready to include in a report. `X_train.csv` and `X_test.csv` contain predictors only; each `y_*.csv` contains the 12 outcomes and the derived binary mortality label.
"""
    (OUTPUT_DIR / "stage1_tutorial.md").write_text(tutorial, encoding="utf-8")

    print(f"Rows: {len(data)} | predictors after exclusions: {X.shape[1]}")
    print(f"Training rows: {len(X_train)} | test rows: {len(X_test)}")
    print(split_summary.to_string(index=False))
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
