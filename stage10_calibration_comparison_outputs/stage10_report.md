# Stage 10: calibration method comparison

The user-specified base models are fixed: TabPFN 3.5 for DEATH_BINARY and ZSN; Random Forest preliminary for FIBR_PREDS. Stage 9 raw outer OOF probabilities and the same target-specific five-fold assignments are used for evaluation. Only the four inner, training-side model fits per outer fold are regenerated to fit Platt, isotonic, beta, and temperature calibrators. The outer validation rows are never used to fit a calibrator. No held-out partition was opened.

## Method recommendations

| Target | Recommended method | Status | Rationale |
|---|---|---|---|
| DEATH_BINARY | Platt | Platt recommended; optional methods inconclusive | Platt improves pooled Brier and Log Loss versus raw; optional methods did not meet the prespecified stable-improvement screen. |
| ZSN | Platt | Platt recommended; optional methods inconclusive | Platt improves pooled Brier and Log Loss versus raw; optional methods did not meet the prespecified stable-improvement screen. |
| FIBR_PREDS | Platt | inconclusive; Platt default | Differences are mixed or small; retain Platt as the assignment-required default while noting that evidence is inconclusive. |

The recommendation screen assesses Brier and Log Loss first, checks fold consistency, and uses ECE/reliability and calibration diagnostics as context. An optional method is recommended only if it improves pooled Brier and Log Loss by at least 1% and wins at least four of five paired folds on both metrics. If differences are small or mixed, Platt is preferred; raw is recommended only when consistently better on both probability scores. These are descriptive rules, not significance tests. See `pooled_oof_metrics.csv`, `outer_fold_metrics.csv`, `paired_comparisons.csv`, `quantile_reliability_bins.csv`, and reliability plots for full evidence.

## Model configuration and prior correction

Exact configurations are recorded in `model_configurations.json`. All selected base models use natural development prevalence; none uses class weights or resampling. Prior correction is therefore N/A. Full-development bundles contain the selected estimator and calibrator fitted using development data only. They are preparation artifacts; held-out evaluation remains a separate approval-gated task.

The pooled ECE uses ten equal-width bins. Reliability plots and `quantile_reliability_bins.csv` use ten quantile bins (fewer if ties collapse bins). Joint intercept/slope and calibration-in-large diagnostics are reported for fold and pooled estimates. Threshold metrics use a fixed 0.5 reference only; threshold selection is not part of this stage.
