# T06 development-only model and calibration freeze

Final calibration is Platt for all targets. Optional calibration methods
are descriptive comparisons and cannot override the required frozen Platt method.
All finalists use natural development prevalence without class weighting or resampling.
Fresh seed-42 nested cross-fitting produced every outer and inner base-model
prediction; Stage 9 raw predictions were not reused.

| Target | Finalist | Development AP evidence (mean +/- fold SD) | Platt pooled Brier | Platt pooled Log Loss | Platt pooled AP |
|---|---|---:|---:|---:|---:|
| DEATH_BINARY | CatBoost tuned | 0.7326 +/- 0.0591 | 0.074604 | 0.253899 | 0.736127 |

Selection rationale for DEATH_BINARY: Selected model mean AP 0.7326 (fold SD 0.0591); strongest saved alternative was Random Forest tuned at 0.6713 (fold SD 0.0373). Five-fold estimates are descriptive, not significance tests.
| ZSN | Random Forest tuned | 0.5018 +/- 0.0829 | 0.146906 | 0.464788 | 0.491265 |

Selection rationale for ZSN: Selected model mean AP 0.5018 (fold SD 0.0829); strongest saved alternative was Extra Trees tuned at 0.4850 (fold SD 0.0641). Five-fold estimates are descriptive, not significance tests.
| FIBR_PREDS | Random Forest preliminary | 0.3011 +/- 0.1079 | 0.083415 | 0.294540 | 0.235949 |

Selection rationale for FIBR_PREDS: Random Forest preliminary mean AP 0.301054 (fold SD 0.107908) was essentially tied with Extra Trees preliminary at 0.301365 (fold SD 0.125206); the AP gap was 0.000311, far smaller than either fold SD. Random Forest was retained as the more balanced candidate: mean AUROC 0.725125 versus 0.721708, mean Brier 0.082854 versus 0.082892, and mean Log Loss 0.292421 versus 0.295108 (lower is better). Five-fold estimates are descriptive, not significance tests.

## Calibration method comparisons

| Target | Method | Pooled Brier | Pooled Log Loss | Pooled AP | Pooled AUROC | Recommendation |
|---|---|---:|---:|---:|---:|---|
| DEATH_BINARY | Raw | 0.077211 | 0.262351 | 0.737533 | 0.904193 | Raw identity reference; Platt remains frozen. |
| DEATH_BINARY | Platt | 0.074604 | 0.253899 | 0.736127 | 0.904266 | Frozen final method (assignment-required Platt). |
| DEATH_BINARY | Isotonic | 0.075635 | 0.268521 | 0.725485 | 0.898888 | Does not meet optional screen; retain frozen Platt. |
| DEATH_BINARY | Beta | 0.074625 | 0.254036 | 0.736689 | 0.903625 | Does not meet optional screen; retain frozen Platt. |
| DEATH_BINARY | Temperature | 0.077245 | 0.261649 | 0.736993 | 0.904197 | Does not meet optional screen; retain frozen Platt. |
| ZSN | Raw | 0.147352 | 0.466140 | 0.488331 | 0.714266 | Raw identity reference; Platt remains frozen. |
| ZSN | Platt | 0.146906 | 0.464788 | 0.491265 | 0.715621 | Frozen final method (assignment-required Platt). |
| ZSN | Isotonic | 0.147943 | 0.491604 | 0.464004 | 0.708380 | Does not meet optional screen; retain frozen Platt. |
| ZSN | Beta | 0.147577 | 0.467197 | 0.483898 | 0.714338 | Does not meet optional screen; retain frozen Platt. |
| ZSN | Temperature | 0.147200 | 0.465711 | 0.488791 | 0.715754 | Does not meet optional screen; retain frozen Platt. |
| FIBR_PREDS | Raw | 0.082854 | 0.292421 | 0.255133 | 0.721878 | Raw identity reference; Platt remains frozen. |
| FIBR_PREDS | Platt | 0.083415 | 0.294540 | 0.235949 | 0.717663 | Frozen final method (assignment-required Platt). |
| FIBR_PREDS | Isotonic | 0.085118 | 0.319199 | 0.223243 | 0.716260 | Does not meet optional screen; retain frozen Platt. |
| FIBR_PREDS | Beta | 0.083636 | 0.295445 | 0.236894 | 0.715546 | Does not meet optional screen; retain frozen Platt. |
| FIBR_PREDS | Temperature | 0.082797 | 0.292454 | 0.258078 | 0.720995 | Does not meet optional screen; retain frozen Platt. |

## Frozen development thresholds

| Target | Threshold | Status | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEATH_BINARY | 0.15333596 | constraint_met | 0.8018 | 0.8198 | 0.4579 | 0.5829 | 937 | 206 | 43 | 174 |
| ZSN | 0.16286458 | constraint_met | 0.8007 | 0.4668 | 0.3036 | 0.4403 | 492 | 562 | 61 | 245 |
| FIBR_PREDS | 0.07103629 | constraint_met | 0.8000 | 0.4555 | 0.1394 | 0.2374 | 558 | 667 | 27 | 108 |

Outer performance is development-only and descriptive. The full-development
bundle calibrator is fitted to fresh five-fold cross-fitted base scores; its
base estimator is then refit on all eligible development rows. The OOF and
full-development estimators share fixed configurations and probability APIs;
their training sizes differ, so calibration transfer remains a limitation.
The Stage 9 comparison selected tuned settings using outer-training labels,
which weakened independence of its inner calibration scores. T06 fixes the
finalists/configurations and generates new inner OOF scores inside each outer
training partition; outer-validation labels never fit their model/calibrator.
The sensitivity minimum of 0.80 is an operating-point assumption, not a
universal clinical requirement.

Captured fit/calibrator warnings: 0 (details in fit_warnings.csv and manifest).
No held-out evaluation is included or authorized by this freeze.
