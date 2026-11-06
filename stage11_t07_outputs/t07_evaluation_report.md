# T07: final held-out evaluation

This is the single approval-gated evaluation of the frozen T06 bundles on the fixed Stage 1 test partition. The model, Platt calibrators, and thresholds were not fitted or changed using held-out labels. Calibration intercept/slope are descriptive diagnostics only and did not alter predictions.

Ranking quality is summarized by AUROC and average precision (AUPRC). Probability/calibration quality is summarized separately by Brier score, Brier Skill Score against the observed-prevalence baseline, log loss, ECE, and calibration slope/intercept. Threshold metrics use each threshold loaded from the frozen bundle and cross-checked against `selected_thresholds.csv`. Test-set reliability diagrams describe frozen Platt predictions only. Raw-versus-Platt calibration comparisons below use T06 development outer-OOF predictions on the same IDs and labels; they do not use test labels or fit a new calibrator. Reliability diagrams use quantile bins; ECE uses ten equal-width bins. Empty equal-width bins are omitted from the diagnostics table.

## Ranking quality

| Target | Model | AUROC | AUPRC / AP |
|---|---|---:|---:|
| DEATH_BINARY | CatBoost tuned | 0.883644 | 0.710777 |
| ZSN | Random Forest tuned | 0.726641 | 0.595661 |
| FIBR_PREDS | Random Forest preliminary | 0.656674 | 0.191887 |

## Ranking metric uncertainty

Intervals are percentile 95% confidence intervals from a seeded paired row bootstrap of this single final evaluation. The frozen model and threshold were not refit or selected in bootstrap replicates. Replicates with only one observed class are omitted from interval estimation and counted explicitly.

| Target | Metric | Estimate | 95% CI | Valid draws | Invalid single-class draws | Seed |
|---|---|---:|---:|---:|---:|---:|
| DEATH_BINARY | auroc | 0.883644 | [0.832098, 0.929526] | 1000 | 0 | 42 |
| DEATH_BINARY | auprc_average_precision | 0.710777 | [0.592699, 0.808587] | 1000 | 0 | 42 |
| ZSN | auroc | 0.726641 | [0.660464, 0.785915] | 1000 | 0 | 42 |
| ZSN | auprc_average_precision | 0.595661 | [0.499623, 0.681412] | 1000 | 0 | 42 |
| FIBR_PREDS | auroc | 0.656674 | [0.551759, 0.762356] | 1000 | 0 | 42 |
| FIBR_PREDS | auprc_average_precision | 0.191887 | [0.118029, 0.321778] | 1000 | 0 | 42 |

## Probability and calibration quality

| Target | Frozen calibration | Brier | Prevalence baseline Brier | Brier Skill Score | Log loss | ECE (10 equal-width) | Joint intercept | Joint slope | Slope SE | Slope status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DEATH_BINARY | Platt | 0.079882 | 0.133599 | 0.402077 | 0.267979 | 0.025568 | -0.0365257 | 1.01034 | 0.13607 | available |
| ZSN | Platt | 0.153795 | 0.191834 | 0.198289 | 0.481088 | 0.043759 | 0.721189 | 1.4001 | 0.223 | available |
| FIBR_PREDS | Platt | 0.089963 | 0.092344 | 0.025784 | 0.322208 | 0.027283 | -0.812596 | 0.603193 | 0.201592 | available |

## Frozen-threshold classification

| Target | Threshold | Sensitivity | Specificity | Precision | F1 | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEATH_BINARY | 0.15333596 | 0.7593 | 0.8182 | 0.4409 | 0.5578 | 0.8088 | 234 | 52 | 13 | 41 |
| ZSN | 0.16286458 | 0.7727 | 0.4444 | 0.3269 | 0.4595 | 0.5294 | 112 | 140 | 20 | 68 |
| FIBR_PREDS | 0.07103629 | 0.7429 | 0.4623 | 0.1368 | 0.2311 | 0.4912 | 141 | 164 | 9 | 26 |

## Decision curve analysis (descriptive)

Net benefit is reported for the frozen calibrated probabilities over thresholds from 5% through 40% in one percentage-point steps, with treat-all and treat-none reference strategies. The curve uses the single evaluation sample only and is descriptive; it did not select or change the frozen operating threshold. See `test_decision_curve_analysis.csv` and the target-specific plots.
- DEATH_BINARY: 36 thresholds, from 0.05 to 0.40.
- ZSN: 36 thresholds, from 0.05 to 0.40.
- FIBR_PREDS: 36 thresholds, from 0.05 to 0.40.

## Before/after calibration comparison (development only)

These descriptive metrics compare raw model probabilities with fold-local Platt probabilities on the same T06 outer-OOF development records. They are not test metrics and do not change the frozen T06 calibration.

| Target | OOF records | Raw Brier | Platt Brier | Raw Log Loss | Platt Log Loss |
|---|---:|---:|---:|---:|---:|
| DEATH_BINARY | 1360 | 0.077211 | 0.074604 | 0.262351 | 0.253899 |
| ZSN | 1360 | 0.147352 | 0.146906 | 0.466140 | 0.464788 |
| FIBR_PREDS | 1360 | 0.082854 | 0.083415 | 0.292421 | 0.294540 |

## Protocol and limitations

- Test records: 340; input row order and IDs were preserved.
- Brier Skill Score is `1 - Brier_model / (pi * (1 - pi))`, where `pi` is observed prevalence for the target in this evaluated partition, matching the assignment guide's prevalence-baseline formula.
- Stage 1 source constructs the saved predictor, outcome, and ID files from the same `test_idx`; the test-row count is checked against Stage 1 `split_summary.csv`. Those CSVs contain no shared per-row key or alignment fingerprint, so the existing artifacts cannot independently prove X-to-ID correspondence. The run preserves their saved row order and records their hashes; it does not recreate the split.
- Each target was evaluated independently. Test labels were used only for evaluation metrics, threshold confusion metrics, and descriptive calibration diagnostics.
- Calibration diagnostics are not a refit of the frozen Platt calibrators; no result from this report may be used for model selection, recalibration, or threshold tuning.
- This output contains evaluation labels for audit. It is not a T08 submission export.
- Run hashes and exact input paths are recorded in `t07_run_manifest.json`.
