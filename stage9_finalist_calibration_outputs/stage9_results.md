# Stage 9 finalist results and calibration diagnostics

All values are development-only five-fold estimates. Values are fold means (sample SD). The sigmoid calibrator is cross-fitted inside each outer training partition. All probability metrics and logit-based diagnostics use probabilities clipped to [1e-6, 1-1e-6]. AP and AUROC use the same monotonic sigmoid mapping, so ranking metrics are unchanged apart from possible numerical ties.

## Fold mean (sample SD)

| Target | Provisional candidate | AP raw → sigmoid | AUROC raw → sigmoid | Brier raw → sigmoid | Log Loss raw → sigmoid |
|---|---|---:|---:|---:|---:|
| DEATH_BINARY | TabPFN 3.5 | 0.750 ± 0.047 → 0.750 ± 0.047 | 0.915 ± 0.017 → 0.915 ± 0.017 | 0.073 ± 0.007 → 0.073 ± 0.007 | 0.246 ± 0.021 → 0.246 ± 0.023 |
| DEATH_BINARY | CatBoost tuned | 0.733 ± 0.059 → 0.733 ± 0.059 | 0.901 ± 0.022 → 0.901 ± 0.022 | 0.077 ± 0.009 → 0.076 ± 0.010 | 0.262 ± 0.027 → 0.259 ± 0.031 |
| ZSN | TabPFN 3.5 | 0.519 ± 0.097 → 0.519 ± 0.097 | 0.732 ± 0.056 → 0.732 ± 0.056 | 0.143 ± 0.015 → 0.143 ± 0.015 | 0.453 ± 0.036 → 0.453 ± 0.038 |
| ZSN | Random Forest tuned | 0.502 ± 0.083 → 0.502 ± 0.083 | 0.718 ± 0.045 → 0.718 ± 0.045 | 0.148 ± 0.013 → 0.148 ± 0.012 | 0.465 ± 0.032 → 0.465 ± 0.031 |
| FIBR_PREDS | Extra Trees preliminary | 0.301 ± 0.125 → 0.301 ± 0.125 | 0.722 ± 0.054 → 0.722 ± 0.054 | 0.083 ± 0.007 → 0.083 ± 0.004 | 0.295 ± 0.023 → 0.294 ± 0.015 |
| FIBR_PREDS | Random Forest preliminary | 0.301 ± 0.108 → 0.301 ± 0.108 | 0.725 ± 0.063 → 0.725 ± 0.063 | 0.083 ± 0.005 → 0.083 ± 0.004 | 0.292 ± 0.021 → 0.293 ± 0.020 |

## Pooled OOF calibration diagnostics

ECE uses ten equal-width bins. Calibration slope/intercept are descriptive pooled-OOF diagnostics fit on the same OOF labels and probabilities; they are not independent test estimates.

| Target | Candidate | Stage | ECE | Intercept diagnostic | Slope diagnostic |
|---|---|---|---:|---:|---:|
| DEATH_BINARY | TabPFN 3.5 | raw | 0.0125 | -0.0261 | 1.0104 |
| DEATH_BINARY | TabPFN 3.5 | sigmoid | 0.0114 | -0.0152 | 0.9677 |
| DEATH_BINARY | CatBoost tuned | raw | 0.0286 | 0.4616 | 1.2015 |
| DEATH_BINARY | CatBoost tuned | sigmoid | 0.0217 | -0.0365 | 1.0403 |
| ZSN | TabPFN 3.5 | raw | 0.0192 | 0.0769 | 1.0766 |
| ZSN | TabPFN 3.5 | sigmoid | 0.0182 | 0.0337 | 1.0266 |
| ZSN | Random Forest tuned | raw | 0.0330 | -0.0167 | 1.0393 |
| ZSN | Random Forest tuned | sigmoid | 0.0235 | 0.0478 | 1.0327 |
| FIBR_PREDS | Extra Trees preliminary | raw | 0.0209 | -0.3829 | 0.8140 |
| FIBR_PREDS | Extra Trees preliminary | sigmoid | 0.0148 | 0.2445 | 1.1138 |
| FIBR_PREDS | Random Forest preliminary | raw | 0.0118 | -0.0200 | 1.0244 |
| FIBR_PREDS | Random Forest preliminary | sigmoid | 0.0057 | 0.2123 | 1.1080 |

## Interpretation

These results compare only the requested provisional candidates. Five folds are descriptive, not a significance test. The fixed 0.5 threshold is a reference; model, calibration, and threshold choices remain provisional and are not frozen here. Tuned candidate settings were selected using each full outer-training partition, so inner calibration OOF scores are not fully independent of those setting-selection labels; the untouched outer validation fold evaluates the complete procedure. Final fitting should freeze or reselect settings inside its calibration folds.
