# T06 development-only model and calibration freeze

Final calibration is Platt for all targets. Optional calibration methods
are descriptive comparisons and cannot override the required frozen Platt method.
All finalists use natural development prevalence without class weighting or resampling.
Fresh seed-42 nested cross-fitting produced every outer and inner base-model
prediction; Stage 9 raw predictions were not reused.

| Target | Finalist | Development AP evidence (mean +/- fold SD) | Platt pooled Brier | Platt pooled Log Loss | Platt pooled AP |
|---|---|---:|---:|---:|---:|
| DEATH_BINARY | TabPFN 3.5 | 0.7499 +/- 0.0470 | 0.072744 | 0.245984 | 0.741173 |

Selection rationale for DEATH_BINARY: TabPFN mean AP 0.7499 (fold SD 0.0470) exceeded the strongest saved alternative, CatBoost tuned (0.7326 +/- 0.0591). Five-fold estimates are descriptive, not significance tests.
| ZSN | TabPFN 3.5 | 0.5194 +/- 0.0971 | 0.143044 | 0.452095 | 0.501976 |

Selection rationale for ZSN: TabPFN mean AP 0.5194 (fold SD 0.0971) exceeded the strongest saved alternative, Random Forest tuned (0.5018 +/- 0.0829). Five-fold estimates are descriptive, not significance tests.
| FIBR_PREDS | Random Forest preliminary | 0.3011 +/- 0.1079 | 0.083415 | 0.294540 | 0.235949 |

Selection rationale for FIBR_PREDS: The user-specified preliminary Random Forest mean AP was 0.3011 (fold SD 0.1079); the highest saved alternative was Extra Trees preliminary at 0.3014 (fold SD 0.1252). This near tie is descriptive; the supplied finalist is retained without retuning. Five-fold estimates are descriptive, not significance tests.

## Calibration method comparisons

| Target | Method | Pooled Brier | Pooled Log Loss | Pooled AP | Pooled AUROC | Recommendation |
|---|---|---:|---:|---:|---:|---|
| DEATH_BINARY | Raw | 0.072747 | 0.245669 | 0.743106 | 0.913152 | Raw identity reference; Platt remains frozen. |
| DEATH_BINARY | Platt | 0.072744 | 0.245984 | 0.741173 | 0.913035 | Frozen final method (assignment-required Platt). |
| DEATH_BINARY | Isotonic | 0.073481 | 0.274143 | 0.727151 | 0.910747 | Does not meet optional screen; retain frozen Platt. |
| DEATH_BINARY | Beta | 0.072884 | 0.246873 | 0.741566 | 0.911926 | Does not meet optional screen; retain frozen Platt. |
| DEATH_BINARY | Temperature | 0.072731 | 0.245849 | 0.741952 | 0.913075 | Does not meet optional screen; retain frozen Platt. |
| ZSN | Raw | 0.143413 | 0.453430 | 0.499662 | 0.730271 | Raw identity reference; Platt remains frozen. |
| ZSN | Platt | 0.143044 | 0.452095 | 0.501976 | 0.730932 | Frozen final method (assignment-required Platt). |
| ZSN | Isotonic | 0.143434 | 0.458409 | 0.488403 | 0.728372 | Does not meet optional screen; retain frozen Platt. |
| ZSN | Beta | 0.143180 | 0.452084 | 0.497828 | 0.731025 | Does not meet optional screen; retain frozen Platt. |
| ZSN | Temperature | 0.143297 | 0.452925 | 0.500332 | 0.730904 | Does not meet optional screen; retain frozen Platt. |
| FIBR_PREDS | Raw | 0.082854 | 0.292421 | 0.255133 | 0.721878 | Raw identity reference; Platt remains frozen. |
| FIBR_PREDS | Platt | 0.083415 | 0.294540 | 0.235949 | 0.717663 | Frozen final method (assignment-required Platt). |
| FIBR_PREDS | Isotonic | 0.085118 | 0.319199 | 0.223243 | 0.716260 | Does not meet optional screen; retain frozen Platt. |
| FIBR_PREDS | Beta | 0.083636 | 0.295445 | 0.236894 | 0.715546 | Does not meet optional screen; retain frozen Platt. |
| FIBR_PREDS | Temperature | 0.082797 | 0.292454 | 0.258078 | 0.720995 | Does not meet optional screen; retain frozen Platt. |

## Frozen development thresholds

| Target | Threshold | Status | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEATH_BINARY | 0.19570890 | constraint_met | 0.8018 | 0.8670 | 0.5337 | 0.6409 | 991 | 152 | 43 | 174 |
| ZSN | 0.17307475 | constraint_met | 0.8007 | 0.4915 | 0.3137 | 0.4508 | 518 | 536 | 61 | 245 |
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
