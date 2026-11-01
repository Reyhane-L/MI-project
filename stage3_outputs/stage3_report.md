# Stage 3: Random Forest

Uses 1360 Stage 1 training rows and 102 predictors only. The locked test partition is not opened. The original 80/20 split is preserved. Logistic Regression and Dummy results are read from Stage 2 aggregates.

Continuous values use fold-fitted median imputation; binary and ordinal codes use fold-fitted mode imputation and stay numeric. The four ECG morphology fields are mode-imputed and one-hot encoded, as in Stage 2. Scaling is omitted because tree splits do not depend on feature units. `KFK_BLOOD`, which is all-missing, is retained as a constant feature.

The preliminary forest uses 200 trees and a fixed seed. Tuning uses eight randomized combinations per outer training fold, with three-fold stratified inner CV and Average Precision as the selection metric. It varies maximum tree depth, features considered at each split, minimum leaf size, and class weighting. The five outer folds estimate performance after parameter selection; the inner search scores are not reported as final performance. See `tuning_results.csv` for per-fold settings.

Means (sample SD) across five folds; probability metrics use predicted probabilities and threshold metrics use 0.5.

| Target | Model | AP | AUROC | Brier | Brier skill | Accuracy | Precision | Recall | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DEATH_BINARY` | Dummy prior prevalence | 0.160 (0.002) | 0.500 (0.000) | 0.134 (0.001) | -0.000 (0.013) | 0.840 (0.002) | 0.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) |
| `DEATH_BINARY` | Logistic regression | 0.651 (0.031) | 0.879 (0.017) | 0.089 (0.005) | 0.337 (0.039) | 0.884 (0.006) | 0.699 (0.063) | 0.493 (0.048) | 0.958 (0.015) | 0.574 (0.017) |
| `DEATH_BINARY` | Random Forest preliminary | 0.668 (0.020) | 0.892 (0.014) | 0.091 (0.005) | 0.323 (0.035) | 0.865 (0.008) | 0.827 (0.125) | 0.212 (0.071) | 0.990 (0.009) | 0.329 (0.081) |
| `DEATH_BINARY` | Random Forest tuned | 0.671 (0.037) | 0.883 (0.022) | 0.109 (0.024) | 0.184 (0.179) | 0.872 (0.017) | 0.705 (0.165) | 0.414 (0.082) | 0.959 (0.032) | 0.506 (0.040) |
| `ZSN` | Dummy prior prevalence | 0.225 (0.002) | 0.500 (0.000) | 0.174 (0.001) | -0.000 (0.007) | 0.775 (0.002) | 0.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) |
| `ZSN` | Logistic regression | 0.315 (0.030) | 0.621 (0.025) | 0.179 (0.005) | -0.024 (0.029) | 0.756 (0.010) | 0.358 (0.100) | 0.124 (0.066) | 0.939 (0.015) | 0.182 (0.081) |
| `ZSN` | Random Forest preliminary | 0.479 (0.084) | 0.720 (0.048) | 0.153 (0.009) | 0.122 (0.051) | 0.793 (0.005) | 0.756 (0.120) | 0.134 (0.036) | 0.985 (0.013) | 0.224 (0.045) |
| `ZSN` | Random Forest tuned | 0.502 (0.083) | 0.718 (0.045) | 0.148 (0.013) | 0.154 (0.072) | 0.813 (0.024) | 0.772 (0.139) | 0.236 (0.076) | 0.981 (0.012) | 0.359 (0.104) |
| `FIBR_PREDS` | Dummy prior prevalence | 0.099 (0.000) | 0.500 (0.000) | 0.089 (0.000) | 0.000 (0.000) | 0.901 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 1.000 (0.000) | 0.000 (0.000) |
| `FIBR_PREDS` | Logistic regression | 0.246 (0.068) | 0.702 (0.071) | 0.089 (0.008) | 0.007 (0.086) | 0.893 (0.008) | 0.327 (0.147) | 0.081 (0.048) | 0.982 (0.006) | 0.130 (0.073) |
| `FIBR_PREDS` | Random Forest preliminary | 0.301 (0.108) | 0.725 (0.063) | 0.083 (0.005) | 0.073 (0.056) | 0.901 (0.002) | 0.200 (0.447) | 0.007 (0.017) | 1.000 (0.000) | 0.014 (0.032) |
| `FIBR_PREDS` | Random Forest tuned | 0.277 (0.085) | 0.708 (0.045) | 0.084 (0.004) | 0.062 (0.046) | 0.901 (0.002) | 0.400 (0.418) | 0.030 (0.031) | 0.998 (0.004) | 0.054 (0.055) |

Scores are internal CV estimates; five-fold SDs are descriptive, not confidence intervals. The search is intentionally restrained, and repeated model development can still make estimates optimistic. Class weighting may affect calibration. The 0.5 threshold is only a reference. Highly incomplete features remain imputed. Consider evaluating a finalized model once on the locked test partition after model selection is complete.

scikit-learn `1.9.1`; seed `42`. Outputs: fold metrics, aggregate metrics, three-model comparison, tuning results, target summary, and plot. Stage 2 files are not modified.
