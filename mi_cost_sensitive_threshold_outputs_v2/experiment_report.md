# Cost-sensitive mortality threshold experiment

This post-processing experiment uses only the frozen T06 DEATH_BINARY outer development OOF Platt probabilities. It does not fit a model or calibrator and does not read Stage 11 test results.

Decision rule: predict death when calibrated probability is greater than or equal to the threshold. Empirical cost is `10 × false negatives + false positives`; all thresholds in `{0, unique OOF probabilities, nextafter(max probability, +∞)}` were evaluated. Minimum-cost ties resolve to the highest threshold.

The observed OOF death prevalence is **0.1596** among **N=1,360** patients. The cost-minimizing threshold is **0.101369151773**, with cost **505**. At this threshold, **36.91 alerts per 100 patients** are generated and **14.49 deaths per 100 patients** are identified.

| Rule | Threshold | Cost | TP | FP | TN | FN | Sensitivity | Specificity | Precision | Alerts | Alert rate | Deaths identified per 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cost_minimizing_development_oof | 0.101369 | 505 | 197 | 305 | 838 | 20 | 0.9078 | 0.7332 | 0.3924 | 502 | 0.3691 | 14.485 |
| default_0.5 | 0.500000 | 1079 | 112 | 29 | 1114 | 105 | 0.5161 | 0.9746 | 0.7943 | 141 | 0.1037 | 8.235 |
| frozen_sensitivity_0.8 | 0.153336 | 636 | 174 | 206 | 937 | 43 | 0.8018 | 0.8198 | 0.4579 | 380 | 0.2794 | 12.794 |

‘Deaths identified per 100’ is TP divided by all evaluated patients; sensitivity is TP divided only by observed deaths. This operating-point analysis is descriptive and does not change the frozen T06 threshold or model.
