# Stage 6 — CatBoost development-CV experiment

## Method

The experiment uses only the Stage 1 development split (1360 rows; target-specific usable sample sizes are listed below), seed 42, and target-specific shuffled five-fold stratified outer CV. Each outer training partition uses shuffled three-fold stratified inner CV and a 4-candidate randomized search scored by average precision (AP). The selected estimator is refit on that complete outer training partition; its untouched outer validation fold is scored once. `RandomizedSearchCV` raises on fitting errors, and captured Python warnings are retained in the fold/tuning CSVs.

The searched parameter values are `iterations` ∈ {250, 400, 600}, `depth` ∈ {4, 5, 6}, `learning_rate` ∈ {0.03, 0.05, 0.08}, `l2_leaf_reg` ∈ {3, 5, 7, 9}, `random_strength` ∈ {0.5, 1.0}, and `border_count` ∈ {64, 128}. Four candidates are sampled from these ranges for each outer fold; the full Cartesian grid is not exhaustively evaluated.

The outer splitter is the same deterministic `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` used in Stages 2–4. Saved previous-stage fold metric signatures were compared for train/validation sizes, validation class counts, and training prevalence. Those summaries do not contain record-level fold membership, so exact historical membership cannot be directly proven; current folds are reproducibly reconstructed from the same development row order and split definition.

## Categorical and missing-value handling

`SEX` is nominal. `ant_im`, `lat_im`, `inf_im`, and `post_im` are ECG morphology codes 0–4 whose labels describe distinct patterns without an established numeric distance; these five columns are passed to CatBoost as native categorical features. Other binary presence/absence indicators remain numeric 0/1, documented ordinal codes remain numeric, and continuous measurements remain numeric. The classifier uses no class weights or resampling, preserving natural training prevalence.

Numerical missing values are passed through unchanged to CatBoost's native numerical missing-value handling. Categorical missing values are converted to the fixed string `__MI_MISSING_CATEGORY__` in the pipeline transformer; this is deterministic representation, not a statistic learned across folds. In the development predictors, categorical missing counts are {'SEX': 0, 'ant_im': 59, 'lat_im': 56, 'inf_im': 58, 'post_im': 53}; numeric missing counts are {'AGE': 6, 'S_AD_KBRIG': 863, 'D_AD_KBRIG': 863, 'S_AD_ORIT': 220, 'D_AD_ORIT': 220, 'K_BLOOD': 308, 'Na_BLOOD': 311, 'ALT_BLOOD': 229, 'AST_BLOOD': 230, 'KFK_BLOOD': 1356, 'L_BLOOD': 103, 'ROE': 169, 'INF_ANAM': 4, 'STENOK_AN': 90, 'FK_STENOK': 62, 'IBS_POST': 43, 'GB': 7, 'DLIT_AG': 195, 'ZSN_A': 45, 'TIME_B_S': 100, 'IBS_NASL': 1296, 'SIM_GIPERT': 7, 'nr11': 18, 'nr01': 18, 'nr02': 18, 'nr03': 18, 'nr04': 18, 'nr07': 18, 'nr08': 18, 'np01': 14, 'np04': 14, 'np05': 14, 'np07': 14, 'np08': 14, 'np09': 14, 'np10': 14, 'endocr_01': 7, 'endocr_02': 7, 'endocr_03': 7, 'zab_leg_01': 6, 'zab_leg_02': 6, 'zab_leg_03': 6, 'zab_leg_04': 6, 'zab_leg_06': 6, 'O_L_POST': 10, 'K_SH_POST': 12, 'MP_TP_POST': 12, 'SVT_POST': 10, 'GT_POST': 10, 'FIB_G_POST': 10, 'ritm_ecg_p_01': 126, 'ritm_ecg_p_02': 126, 'ritm_ecg_p_04': 126, 'ritm_ecg_p_06': 126, 'ritm_ecg_p_07': 126, 'ritm_ecg_p_08': 126, 'n_r_ecg_p_01': 94, 'n_r_ecg_p_02': 94, 'n_r_ecg_p_03': 94, 'n_r_ecg_p_04': 94, 'n_r_ecg_p_05': 94, 'n_r_ecg_p_06': 94, 'n_r_ecg_p_08': 94, 'n_r_ecg_p_09': 94, 'n_r_ecg_p_10': 94, 'n_p_ecg_p_01': 94, 'n_p_ecg_p_03': 94, 'n_p_ecg_p_04': 94, 'n_p_ecg_p_05': 94, 'n_p_ecg_p_06': 94, 'n_p_ecg_p_07': 94, 'n_p_ecg_p_08': 94, 'n_p_ecg_p_09': 94, 'n_p_ecg_p_10': 94, 'n_p_ecg_p_11': 94, 'n_p_ecg_p_12': 94, 'fibr_ter_01': 9, 'fibr_ter_02': 9, 'fibr_ter_03': 9, 'fibr_ter_05': 9, 'fibr_ter_06': 9, 'fibr_ter_07': 9, 'fibr_ter_08': 9, 'GIPO_K': 306, 'GIPER_Na': 311, 'NA_KB': 520, 'NOT_NA_KB': 547, 'LID_KB': 539, 'NITR_S': 5, 'LID_S_n': 6, 'B_BLOK_S_n': 7, 'ANT_CA_S_n': 8, 'GEPAR_S_n': 12, 'ASP_S_n': 12, 'TIKL_S_n': 11, 'TRENT_S_n': 11}.

## Results

All reported scores are outer-fold results. AP is the primary ranking metric; AUROC and Brier are secondary probability/ranking metrics. Precision, recall, specificity, F1, accuracy, and confusion counts use the fixed 0.5 threshold and are not threshold selection or calibration.

| Target | Model | AP mean ± SD | AUROC mean ± SD | Brier mean ± SD |
|---|---|---:|---:|---:|
| DEATH_BINARY | Dummy prior prevalence | 0.160 ± 0.002 | 0.500 ± 0.000 | 0.134 ± 0.001 |
| DEATH_BINARY | Logistic regression | 0.651 ± 0.031 | 0.879 ± 0.017 | 0.089 ± 0.005 |
| DEATH_BINARY | Random Forest preliminary | 0.668 ± 0.020 | 0.892 ± 0.014 | 0.091 ± 0.005 |
| DEATH_BINARY | Random Forest tuned | 0.671 ± 0.037 | 0.883 ± 0.022 | 0.109 ± 0.024 |
| DEATH_BINARY | Extra Trees preliminary | 0.656 ± 0.038 | 0.877 ± 0.011 | 0.091 ± 0.003 |
| DEATH_BINARY | Extra Trees tuned | 0.669 ± 0.047 | 0.881 ± 0.016 | 0.106 ± 0.015 |
| DEATH_BINARY | CatBoost tuned | 0.733 ± 0.059 | 0.901 ± 0.022 | 0.077 ± 0.009 |
| ZSN | Dummy prior prevalence | 0.225 ± 0.002 | 0.500 ± 0.000 | 0.174 ± 0.001 |
| ZSN | Logistic regression | 0.315 ± 0.030 | 0.621 ± 0.025 | 0.179 ± 0.005 |
| ZSN | Random Forest preliminary | 0.479 ± 0.084 | 0.720 ± 0.048 | 0.153 ± 0.009 |
| ZSN | Random Forest tuned | 0.502 ± 0.083 | 0.718 ± 0.045 | 0.148 ± 0.013 |
| ZSN | Extra Trees preliminary | 0.400 ± 0.036 | 0.708 ± 0.024 | 0.158 ± 0.005 |
| ZSN | Extra Trees tuned | 0.485 ± 0.064 | 0.733 ± 0.035 | 0.155 ± 0.005 |
| ZSN | CatBoost tuned | 0.479 ± 0.049 | 0.721 ± 0.048 | 0.150 ± 0.009 |
| FIBR_PREDS | Dummy prior prevalence | 0.099 ± 0.000 | 0.500 ± 0.000 | 0.089 ± 0.000 |
| FIBR_PREDS | Logistic regression | 0.246 ± 0.068 | 0.702 ± 0.071 | 0.089 ± 0.008 |
| FIBR_PREDS | Random Forest preliminary | 0.301 ± 0.108 | 0.725 ± 0.063 | 0.083 ± 0.005 |
| FIBR_PREDS | Random Forest tuned | 0.277 ± 0.085 | 0.708 ± 0.045 | 0.084 ± 0.004 |
| FIBR_PREDS | Extra Trees preliminary | 0.301 ± 0.125 | 0.722 ± 0.054 | 0.083 ± 0.007 |
| FIBR_PREDS | Extra Trees tuned | 0.301 ± 0.123 | 0.726 ± 0.060 | 0.083 ± 0.004 |
| FIBR_PREDS | CatBoost tuned | 0.266 ± 0.068 | 0.698 ± 0.041 | 0.086 ± 0.005 |

Fold signatures checked against prior stages: 450 individual values matched.

## Per-target comparison

For each target and metric, CatBoost's five-fold mean is compared with the best mean among the six existing models. AP and AUROC are better when higher; Brier is better when lower. These are descriptive comparisons and do not establish statistical significance.

### DEATH_BINARY
- AP: CatBoost 0.733; best existing comparator Random Forest tuned 0.671; CatBoost led.
- AUROC: CatBoost 0.901; best existing comparator Random Forest preliminary 0.892; CatBoost led.
- Brier: CatBoost 0.077; best existing comparator Logistic regression 0.089; CatBoost led.

### ZSN
- AP: CatBoost 0.479; best existing comparator Random Forest tuned 0.502; CatBoost did not lead.
- AUROC: CatBoost 0.721; best existing comparator Extra Trees tuned 0.733; CatBoost did not lead.
- Brier: CatBoost 0.150; best existing comparator Random Forest tuned 0.148; CatBoost did not lead.

### FIBR_PREDS
- AP: CatBoost 0.266; best existing comparator Extra Trees preliminary 0.301; CatBoost did not lead.
- AUROC: CatBoost 0.698; best existing comparator Extra Trees tuned 0.726; CatBoost did not lead.
- Brier: CatBoost 0.086; best existing comparator Random Forest preliminary 0.083; CatBoost did not lead.

## Comparison and limitations

`paired_comparison.csv` gives target/fold-level differences against the Dummy, Logistic Regression, Random Forest, and Extra Trees results. It also records wins and ties for AP, AUROC, and Brier. Five folds are too few to establish statistical significance; differences are descriptive. CatBoost is not automatically declared best. Selection for the later final model stage should consider per-target evidence on development data and calibration separately.

This stage performs no probability calibration, threshold optimization, or held-out evaluation. Missing labels are excluded separately for each target. The small four-candidate search is an initial compute-bounded experiment and does not explore the full CatBoost parameter space.

Environment: scikit-learn `1.9.1`; CatBoost `1.2.10`. Run `.venv/bin/python mi_stage6_catboost.py`.

## Output files

- `fold_level_metrics.csv`: CatBoost fold metrics, threshold confusion counts, probabilities, and warnings.
- `target_summary.csv`: target label availability, prevalence, and fold-check counts.
- `aggregate_metrics.csv`: mean and sample standard deviation by target.
- `model_comparison.csv`: CatBoost plus six existing models.
- `paired_comparison.csv`: fold-level comparisons against existing models.
- `tuning_results.csv`: best parameters and inner-CV AP by target/fold.
- `oof_predictions.csv`: one development-only outer-fold prediction per eligible record/target, with ID for traceability only.
- `model_comparison.png`: AP, AUROC, and Brier comparison plot.
