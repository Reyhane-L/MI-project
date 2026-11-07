# Development-only 1:1 undersampling probability experiment

A Stage 2 regularized logistic model was fit in five stratified seed-42 folds. Random undersampling to 1:1 was performed only on each training fold; each validation fold retained its natural prevalence. The prior correction used the natural prevalence of the corresponding training fold and the sampled training prevalence, never validation labels.

The Section 11.3 prior correction is `p = beta * ps / (beta * ps - ps + 1)`, with `beta = odds(natural training prevalence) / odds(sampled training prevalence)`. Imputation, scaling, encoding, and logistic regression were fit only on the undersampled training fold.

Across pooled development OOF rows, the observed death prevalence was **0.1596**. The raw mean predicted probability was **0.3541**; after correction it was **0.1841**.

| Method | Mean probability | Brier | Log loss | ECE (10 quantile bins) | Calibration intercept | Calibration slope | AP | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_undersampled | 0.3541 | 0.1521 | 0.4840 | 0.1946 | -1.7463 | 0.7152 | 0.6133 | 0.8734 |
| prior_corrected | 0.1841 | 0.0943 | 0.3121 | 0.0389 | -0.5581 | 0.7152 | 0.6133 | 0.8734 |

Raw and corrected estimates are evaluated on identical natural-prevalence OOF records. The correction changes the prior odds but not ranking. Results describe this model and development sample; the locked test partition was not used.
