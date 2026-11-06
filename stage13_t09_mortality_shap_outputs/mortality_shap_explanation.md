# Frozen mortality model: development SHAP summary

This summary explains the frozen T06 `CatBoost tuned` base model for `DEATH_BINARY`. Native CatBoost Independent TreeSHAP values were computed on the development predictors in `stage1_outputs/X_train.csv`; the reference sample contains 64 development rows selected with seed 42. The frozen Platt calibrator and threshold were not refit or applied to the explanations.

## Features with largest average absolute SHAP values

| Rank | Feature | Meaning in the Stage 1 dictionary | Mean absolute SHAP | Mean signed SHAP |
|---:|---|---|---:|---:|
| 1 | AGE | Age | 0.293472 | -0.0407317 |
| 2 | ZSN_A | Presence of chronic Heart failure (HF) in the anamnesis | 0.291058 | -0.0678362 |
| 3 | ant_im | Presence of an anterior myocardial infarction (left ventricular) (ECG changes in leads V1 – V4 ) | 0.287284 | -0.00123867 |
| 4 | TIME_B_S | Time elapsed from the beginning of the attack of CHD to the hospital | 0.26991 | -0.00586725 |
| 5 | STENOK_AN | Exertional angina pectoris in the anamnesis | 0.213083 | -0.0365636 |
| 6 | D_AD_ORIT | Diastolic blood pressure according to intensive care unit | 0.144763 | -0.0125731 |
| 7 | lat_im | Presence of a lateral myocardial infarction (left ventricular) (ECG changes in leads V5 – V6 , I, AVL) | 0.129816 | -0.0240557 |
| 8 | L_BLOOD | White blood cell count (billions per liter) | 0.126049 | -0.0218962 |
| 9 | IBS_POST | Coronary heart disease (CHD) in recent weeks, days before admission to hospital | 0.124339 | 0.00704863 |
| 10 | ANT_CA_S_n | Use of calcium channel blockers in the ICU | 0.114852 | -0.00903293 |

## How to read the plot

Each point represents one development admission. Positive SHAP values raise the frozen CatBoost model's raw death score; negative values lower it. Features are ordered by mean absolute SHAP, so the ranking measures average contribution magnitude, not a uniform direction. The top-feature rows show observations on both sides of zero, indicating heterogeneous contributions across admissions. This plot is monochrome and does not encode feature values, so it does not support claims about which feature values raise or lower the score. The model's Platt calibration is post-hoc and was not applied to these raw-score attributions. These are model associations, not causal effects. All explained and reference rows come from development data.
