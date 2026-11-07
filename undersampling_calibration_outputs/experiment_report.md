# Development-only 1:1 undersampling probability experiment

A Stage 2 regularized logistic model was fit in five stratified seed-42 folds. Each outer training fold was split into a 20% stratified natural-prevalence calibration subset and a disjoint model-fit subset. The model-fit subset alone was randomly undersampled to 1:1, and its preprocessing/model pipeline was fit only there. The untouched natural-prevalence calibration subset supplied out-of-fit base scores and labels to Platt; outer validation labels did not participate in any fitting.

The Section 11.3 prior correction is `p = beta * ps / (beta * ps - ps + 1)`, with `beta = odds(natural outer-training prevalence) / odds(sampled model-fit prevalence)`. Platt calibration uses `mi_stage10_calibrators.PlattCalibrator` fit on the separate calibration subset.

Across pooled development OOF rows, observed death prevalence was **0.1596**. Mean predicted probability was **0.3411 raw**, **0.1768 prior-corrected**, and **0.1663 Platt-calibrated**.

| Method | Mean probability | Brier | Log loss | ECE (10 quantile bins) | Calibration intercept | Calibration slope | AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_undersampled | 0.3411 | 0.1520 | 0.4802 | 0.1831 | -1.6434 | 0.6686 | 0.5995 | 0.8590 |
| prior_corrected | 0.1768 | 0.0964 | 0.3246 | 0.0436 | -0.5319 | 0.6687 | 0.5995 | 0.8589 |
| platt | 0.1663 | 0.0938 | 0.3103 | 0.0123 | -0.0563 | 1.0116 | 0.6034 | 0.8534 |

All three estimates are evaluated on the same natural-prevalence OOF records. Raw, prior-corrected, and Platt probabilities are retained as separate comparisons. Results describe this model and development sample; the locked test partition was not used.
