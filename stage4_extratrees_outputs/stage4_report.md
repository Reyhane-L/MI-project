# Stage 4.2: Extra Trees Classifier

## Scope and design

This stage evaluates `DEATH_BINARY`, `ZSN`, and `FIBR_PREDS` on 1360 Stage 1 training rows and 102 predictors. Only `X_train.csv` and `y_train.csv` are opened; the locked 20% test set is not used. Rows missing the target label are excluded for that target. No Stage 1, Stage 2, or Stage 3 files are modified.

For each target, evaluation uses shuffled `StratifiedKFold(n_splits=5, random_state=42)`. Preprocessing is in a scikit-learn Pipeline and is refit within each fold. Continuous predictors use median imputation; binary and ordinal codes use most-frequent imputation and remain numeric; four ECG morphology fields use most-frequent imputation and one-hot encoding. Scaling is omitted because tree split rules do not depend on feature units. The all-missing `KFK_BLOOD` column is retained as a constant feature.

## Extra Trees models and tuning

The preliminary classifier uses 200 randomized trees, default Extra Trees split settings, and a fixed seed. Extra Trees chooses split thresholds more randomly than a standard Random Forest; this can reduce correlation among trees and may improve or worsen generalization. The preliminary model gives a fixed-setting reference.

Tuning uses a restrained randomized search (6 candidates per outer fold) with three-fold stratified inner CV and Average Precision as the selection metric. It explores `n_estimators` (100–200), `max_depth`, `max_features`, `min_samples_leaf`, `min_samples_split`, and `class_weight`. Search is serial (`n_jobs=1`) to limit memory and CPU use on the 8 GB MacBook Air. Each outer fold's validation rows are held out from its inner search and scored only after refitting the selected parameters on that outer training fold. Selected settings and inner AP are in `tuning_results.csv`.

## Fold and probability checks

Recreated all target-specific folds with the Stage 2/3 splitter. Compared each of five folds per target to saved Stage 2 Dummy and Logistic Regression rows and Stage 3 preliminary and tuned Random Forest rows: 300 fold-signature checks passed. Checks cover training/validation size, validation positive/negative counts, and training prevalence. Existing metric files do not contain patient IDs, so exact record-by-record membership cannot be independently checked from those files; the deterministic splitter and unchanged training row order are used to reproduce membership.

Each Extra Trees prediction vector is checked for finite values and probabilities within [0,1]. A missing class or invalid probability stops the script. Fit and search warnings are captured in the fold and tuning CSVs; fitting errors stop execution. No outer-fold threshold optimization is performed. Threshold metrics use `probability >= 0.5`.

## Results

The table reports mean (sample standard deviation) across five outer folds. AP, AUROC, and Brier use predicted probabilities. Higher AP/AUROC is better; lower Brier is better. These fold standard deviations are descriptive, not confidence intervals.

| Target | Model | AP | AUROC | Brier | Brier skill | Accuracy | Precision | Recall | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DEATH_BINARY` | Dummy prior prevalence | 0.160 (0.002) | 0.500 (0.000) | 0.134 (0.001) | -0.000 (0.013) | 0.840 (0.002) | 0.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) |
| `DEATH_BINARY` | Logistic regression | 0.651 (0.031) | 0.879 (0.017) | 0.089 (0.005) | 0.337 (0.039) | 0.884 (0.006) | 0.699 (0.063) | 0.493 (0.048) | 0.958 (0.015) | 0.574 (0.017) |
| `DEATH_BINARY` | Random Forest preliminary | 0.668 (0.020) | 0.892 (0.014) | 0.091 (0.005) | 0.323 (0.035) | 0.865 (0.008) | 0.827 (0.125) | 0.212 (0.071) | 0.990 (0.009) | 0.329 (0.081) |
| `DEATH_BINARY` | Random Forest tuned | 0.671 (0.037) | 0.883 (0.022) | 0.109 (0.024) | 0.184 (0.179) | 0.872 (0.017) | 0.705 (0.165) | 0.414 (0.082) | 0.959 (0.032) | 0.506 (0.040) |
| `DEATH_BINARY` | Extra Trees preliminary | 0.656 (0.038) | 0.877 (0.011) | 0.091 (0.003) | 0.322 (0.025) | 0.873 (0.008) | 0.844 (0.093) | 0.253 (0.064) | 0.990 (0.008) | 0.385 (0.075) |
| `DEATH_BINARY` | Extra Trees tuned | 0.669 (0.047) | 0.881 (0.016) | 0.106 (0.015) | 0.212 (0.112) | 0.867 (0.013) | 0.786 (0.202) | 0.364 (0.256) | 0.962 (0.048) | 0.420 (0.192) |
| `ZSN` | Dummy prior prevalence | 0.225 (0.002) | 0.500 (0.000) | 0.174 (0.001) | -0.000 (0.007) | 0.775 (0.002) | 0.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) |
| `ZSN` | Logistic regression | 0.315 (0.030) | 0.621 (0.025) | 0.179 (0.005) | -0.024 (0.029) | 0.756 (0.010) | 0.358 (0.100) | 0.124 (0.066) | 0.939 (0.015) | 0.182 (0.081) |
| `ZSN` | Random Forest preliminary | 0.479 (0.084) | 0.720 (0.048) | 0.153 (0.009) | 0.122 (0.051) | 0.793 (0.005) | 0.756 (0.120) | 0.134 (0.036) | 0.985 (0.013) | 0.224 (0.045) |
| `ZSN` | Random Forest tuned | 0.502 (0.083) | 0.718 (0.045) | 0.148 (0.013) | 0.154 (0.072) | 0.813 (0.024) | 0.772 (0.139) | 0.236 (0.076) | 0.981 (0.012) | 0.359 (0.104) |
| `ZSN` | Extra Trees preliminary | 0.400 (0.036) | 0.708 (0.024) | 0.158 (0.005) | 0.091 (0.029) | 0.770 (0.012) | 0.419 (0.175) | 0.059 (0.025) | 0.976 (0.008) | 0.103 (0.043) |
| `ZSN` | Extra Trees tuned | 0.485 (0.064) | 0.733 (0.035) | 0.155 (0.005) | 0.110 (0.028) | 0.790 (0.013) | 0.584 (0.072) | 0.305 (0.170) | 0.932 (0.055) | 0.374 (0.127) |
| `FIBR_PREDS` | Dummy prior prevalence | 0.099 (0.000) | 0.500 (0.000) | 0.089 (0.000) | 0.000 (0.000) | 0.901 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) |
| `FIBR_PREDS` | Logistic regression | 0.246 (0.068) | 0.702 (0.071) | 0.089 (0.008) | 0.007 (0.086) | 0.893 (0.008) | 0.327 (0.147) | 0.081 (0.048) | 0.982 (0.006) | 0.130 (0.073) |
| `FIBR_PREDS` | Random Forest preliminary | 0.301 (0.108) | 0.725 (0.063) | 0.083 (0.005) | 0.073 (0.056) | 0.901 (0.002) | 0.200 (0.447) | 0.007 (0.017) | 1.000 (0.000) | 0.014 (0.032) |
| `FIBR_PREDS` | Random Forest tuned | 0.277 (0.085) | 0.708 (0.045) | 0.084 (0.004) | 0.062 (0.046) | 0.901 (0.002) | 0.400 (0.418) | 0.030 (0.031) | 0.998 (0.004) | 0.054 (0.055) |
| `FIBR_PREDS` | Extra Trees preliminary | 0.301 (0.125) | 0.722 (0.054) | 0.083 (0.007) | 0.073 (0.075) | 0.901 (0.003) | 0.400 (0.418) | 0.022 (0.020) | 0.998 (0.002) | 0.042 (0.038) |
| `FIBR_PREDS` | Extra Trees tuned | 0.301 (0.123) | 0.726 (0.060) | 0.083 (0.004) | 0.068 (0.050) | 0.899 (0.005) | 0.000 (0.000) | 0.000 (0.000) | 0.998 (0.005) | 0.000 (0.000) |

### Reading these results

- **DEATH_BINARY:** highest mean AP is Random Forest tuned (0.671); lowest mean Brier is Logistic regression (0.089). Check the paired fold file for how often Extra Trees beats each comparator; small mean differences and five folds do not establish a reliable winner.
- **ZSN:** highest mean AP is Random Forest tuned (0.502); lowest mean Brier is Random Forest tuned (0.148). Check the paired fold file for how often Extra Trees beats each comparator; small mean differences and five folds do not establish a reliable winner.
- **FIBR_PREDS:** highest mean AP is Extra Trees preliminary (0.301); lowest mean Brier is Random Forest preliminary (0.083). Check the paired fold file for how often Extra Trees beats each comparator; small mean differences and five folds do not establish a reliable winner.

### Paired comparisons

`paired_comparison.csv` contains fold-level differences for each Extra Trees version against Logistic Regression and both Random Forest versions. Differences are Extra Trees minus comparator. A positive AP/AUROC difference favors Extra Trees; a negative Brier difference favors Extra Trees. The file also records fold wins and ties. No statistical significance is claimed from five folds.

**DEATH_BINARY**
- Extra Trees preliminary: vs Logistic regression: AP 2/5, AUROC 2/5, Brier 2/5; vs Random Forest preliminary: AP 2/5, AUROC 2/5, Brier 3/5; vs Random Forest tuned: AP 1/5, AUROC 2/5, Brier 4/5
- Extra Trees tuned: vs Logistic regression: AP 4/5, AUROC 2/5, Brier 1/5; vs Random Forest preliminary: AP 3/5, AUROC 2/5, Brier 0/5; vs Random Forest tuned: AP 2/5, AUROC 2/5, Brier 2/5

**ZSN**
- Extra Trees preliminary: vs Logistic regression: AP 5/5, AUROC 5/5, Brier 5/5; vs Random Forest preliminary: AP 1/5, AUROC 2/5, Brier 1/5; vs Random Forest tuned: AP 0/5, AUROC 2/5, Brier 0/5
- Extra Trees tuned: vs Logistic regression: AP 5/5, AUROC 5/5, Brier 5/5; vs Random Forest preliminary: AP 3/5, AUROC 3/5, Brier 2/5; vs Random Forest tuned: AP 2/5, AUROC 4/5, Brier 1/5

**FIBR_PREDS**
- Extra Trees preliminary: vs Logistic regression: AP 4/5, AUROC 4/5, Brier 4/5; vs Random Forest preliminary: AP 1/5, AUROC 2/5, Brier 2/5; vs Random Forest tuned: AP 3/5, AUROC 4/5, Brier 4/5
- Extra Trees tuned: vs Logistic regression: AP 4/5, AUROC 4/5, Brier 4/5; vs Random Forest preliminary: AP 3/5, AUROC 2/5, Brier 2/5; vs Random Forest tuned: AP 3/5, AUROC 4/5, Brier 3/5

## Method differences and limitations

The evaluation folds and imputation/encoding choices match Stage 3. This stage changes the estimator to Extra Trees and uses six randomized candidates rather than Stage 3's eight, while tuning additional tree parameters. Stage 3 parallelized its search with `n_jobs=-1`; Stage 4 deliberately uses serial fitting to reduce resource pressure. Since fold files store fold metrics rather than patient IDs, fold matching is verified through reproducible splitting plus saved fold signatures, not a direct record-ID comparison.

This is internal validation on a small dataset. Five folds do not establish statistical significance, and repeated model selection can make apparent performance optimistic. Class weighting can affect probability calibration. The fixed 0.5 threshold is only a reference, not a clinical decision rule. Highly incomplete measurements remain imputed and may contribute little information.

scikit-learn `1.9.1`; random seed `42`. Run `.venv/bin/python mi_stage4_extratrees.py`.

## Output files

- `fold_level_metrics.csv`: preliminary and tuned Extra Trees metrics, confusion counts, probability checks, warnings, and selected parameters.
- `target_summary.csv`: target counts and fold-signature check totals.
- `aggregate_metrics.csv`: Extra Trees means and sample standard deviations.
- `model_comparison.csv`: all six models across all targets.
- `paired_comparison.csv`: fold-level paired Extra Trees differences against Logistic Regression and both Random Forest models.
- `tuning_results.csv`: selected parameters and inner-CV AP for every outer fold.
- `model_comparison.png`: AP, AUROC, and Brier comparison plot.
- `stage4_report.md`: methodology, checks, results, comparisons, and limitations.
