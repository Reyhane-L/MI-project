# Stage 7 — FIBR_PREDS diagnostic analysis

## Scope and method

This exploratory analysis uses the Stage 1 development rows with observed FIBR_PREDS labels only (n=1360, positives=135, prevalence=9.926%). It reconstructs the deterministic shuffled five-fold stratified split (seed 42) and checks saved fold sizes, class counts and training prevalence against Stage 3 Random Forest, Stage 4 Extra Trees, and Stage 6 CatBoost summaries (75 values checked). Historical record-level fold membership is unavailable in those summaries, so matching signatures do not prove identity of every historical fold member.

Random Forest preliminary, Extra Trees preliminary, and CatBoost tuned are fit on each outer training fold. CatBoost uses the already AP-selected Stage 6 parameters for that fold. Average precision is the primary ranking metric; permutation importance scores the decrease in validation AP after feature permutation (three repeats, one worker, seed 42 + fold). CatBoost TreeSHAP explains only validation rows and uses a deterministic reference sample drawn from the matching training fold. No dependencies were installed. Python warnings are retained in output tables; fit exceptions stop the run rather than dropping a fold.

The positive-versus-negative tables are descriptive associations, not causal effects. Continuous-feature quartiles/outlier counts use the pooled development distribution, and nominal/ordinal categories are shown as observed codes. The leave-one-positive-out calculation uses saved Stage 6 OOF predictions and exports no patient identifiers.

## Answers to the diagnostic questions

1. **Which features distinguish positive and negative cases?** The largest continuous group differences were AGE (SMD 0.51; medians 62 vs 66), D_AD_KBRIG (SMD 0.21; medians 80 vs 85), L_BLOOD (SMD 0.18; medians 8 vs 8.75), S_AD_KBRIG (SMD 0.18; medians 140 vs 140), K_BLOOD (SMD -0.17; medians 4.1 vs 4). The highest CatBoost AP permutation features were nr03, n_r_ecg_p_05, AGE, AST_BLOOD, STENOK_AN, n_r_ecg_p_01, lat_im, L_BLOOD, SEX, endocr_01; nominal category FIBR_PREDS rates are plotted in `fibr_group_distributions_corrected.png`.
2. **Could missingness explain predictive behavior?** Largest absolute missingness differences were S_AD_ORIT (17.1% vs 7.4%; Δ -9.7%), D_AD_ORIT (17.1% vs 7.4%; Δ -9.7%), NOT_NA_KB (39.3% vs 48.9%; Δ +9.6%), LID_KB (38.7% vs 48.1%; Δ +9.5%), NA_KB (37.6% vs 43.7%; Δ +6.1%). These are associations; missingness may reflect clinical workflow or collection patterns.
3. **How stable are model performance and feature importance across folds?** CatBoost tuned: AP 0.266 ± 0.068 (range 0.165–0.337; 27 positives/fold); Extra Trees preliminary: AP 0.301 ± 0.125 (range 0.163–0.438; 27 positives/fold); Random Forest preliminary: AP 0.301 ± 0.108 (range 0.172–0.431; 27 positives/fold). Leading CatBoost features with top-10 fold counts are nr03 (5/5 top-10 folds), n_r_ecg_p_05 (4/5 top-10 folds), AGE (3/5 top-10 folds), AST_BLOOD (3/5 top-10 folds), STENOK_AN (2/5 top-10 folds), n_r_ecg_p_01 (2/5 top-10 folds), lat_im (2/5 top-10 folds), L_BLOOD (2/5 top-10 folds). CatBoost TreeSHAP leaders were AGE, L_BLOOD, INF_ANAM, STENOK_AN, n_r_ecg_p_05, K_BLOOD, D_AD_ORIT, SEX, LID_S_n, ANT_CA_S_n.
4–5. **What did the fixed ablations show?** Mean metric deltas versus the saved-parameter CatBoost baseline were Add training-fold missingness flags: ΔAP +0.004, ΔBrier -0.001, ΔLog Loss -0.004; Drop >80% missing (training-fold rule): ΔAP +0.010, ΔBrier -0.001, ΔLog Loss -0.005. These outer-fold results are exploratory and do not select a winner; fold-level scores and affected features are in `fibr_ablation_results.csv`.
6. **How sensitive is AP to a small positive sample?** Removing one positive at a time from saved OOF predictions gave AP 0.216 originally, leave-one-positive-out range 0.208–0.217, maximum absolute change 0.008. No patient IDs are included.
7. **What should be tried next?** For any ablation with a positive mean AP delta, pre-specify a focused nested-CV follow-up that tests the same idea with selection confined to inner folds and checks Brier/Log Loss; do not treat this outer-fold signal as a selected winner. The follow-up should use nested development CV, keep AP as the selection metric, and reserve generalization claims for the separately governed final evaluation.

## Interpretation and limitations

FIBR_PREDS is relatively uncommon, and there are only five outer folds; fold-level variation and permutation rankings can be noisy. Permuting correlated predictors can distribute or suppress importance. SHAP values describe model output attribution conditional on the fitted model and training-fold reference distribution, not causal effects. Group comparisons do not adjust for confounding. The ablation scores reuse outer folds and fixed CatBoost settings for diagnostics; they must not be treated as an unbiased winner-selection exercise. The Stage 6 saved OOF baseline and reconstructed folds are development-only.

Environment: scikit-learn `1.9.1`, CatBoost `1.2.10`. Run `.venv/bin/python mi_stage7_fibr_diagnostics.py`. Outputs are written only to `stage7_fibr_diagnostics_outputs/` and the script refuses to overwrite an existing directory.

## Output files

- `fibr_group_comparison.csv`: outcome-group summaries for numeric and coded features.
- `fibr_missingness_analysis.csv`: missingness by outcome group.
- `fibr_permutation_importance.csv`, `fibr_feature_stability.csv`: fold importance and ranking stability.
- `fibr_ablation_results.csv`: fold-level exploratory CatBoost ablations with AP, AUROC, Brier and Log Loss.
- `fibr_fold_analysis.csv`, `fibr_fold_summary.csv`, `fibr_oof_positive_sensitivity.csv`: fold performance and OOF sensitivity.
- `fibr_shap_fold_summary.csv`, `fibr_shap_summary.csv`: fold and aggregate TreeSHAP summaries.
- `fibr_permutation_importance.png`, `fibr_shap_importance.png`, `fibr_missingness.png`, `fibr_group_distributions_corrected.png`: diagnostic plots. The initial `fibr_group_distributions.png` is preserved from the first run; use the corrected plot.
