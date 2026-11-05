# Project task list

`AI-HW2-Guide.pdf` is the source of truth. A task is complete only after its acceptance criteria pass and the independent verification procedure in `verification.md` records a pass. Preserve all assignment requirements even where existing stage reports describe a narrower stage scope.

## Completed implementation stages (verification status: pending harness audit)

- [x] **T01 — Data preparation and EDA.** Existing `mi_stage1_eda.py` and `stage1_outputs/` provide the reproducible stratified split, predictor/outcome separation, dictionary roles, and training-only EDA. Confirm source schema and allowed features during final audit.
- [x] **T02 — Required baseline and linear model.** Existing `mi_stage2_baselines.py` and `stage2_outputs/` compare prior-prevalence Dummy and regularized logistic regression using development-set CV for the required targets.
- [x] **T03 — Tree model and fold comparison.** Existing `mi_stage3_random_forest.py`, `mi_stage3_fold_comparison.py`, and `stage3_outputs/` provide preliminary/tuned Random Forest results and paired fold comparisons.
- [x] **T04 — Additional Extra Trees comparison.** Existing `mi_stage4_extratrees.py` and `stage4_extratrees_outputs/` provide an additional tree-ensemble comparison and fold-signature checks. This is additional project work, not a substitute for any assignment requirement.

## Additional project experiments (outside the required assignment sequence)

- [x] **Stage 6 — CatBoost nested-CV comparison.** `mi_stage6_catboost.py` and `stage6_catboost_outputs/` compare CatBoost with the six saved prior model results using the same target-specific five-fold split signatures, three-fold AP tuning, native categorical handling, fold metrics, paired comparisons, and development-only OOF predictions. The additional Log Loss update uses clipped validation probabilities, reports fold means/sample SDs, and refits comparators on the same development folds with already-selected settings; AP remains the sole tuning metric. Acceptance: all targets/folds complete, no held-out access, one valid OOF probability per eligible development ID/target, all requested metrics/outputs validated, and independent review passes. Independent verification passed 2026-11-05. This experiment does not replace the required final model selection/calibration task T06.

## Remaining assignment work

- [x] **T05 — Requirement and leakage audit.** Map every requirement in the guide to code, output, report, or a documented not-yet-done item. Verify one stable 80/20 stratified split, seed, ID/outcome exclusion, post-admission exclusions, and development-only model selection. Acceptance: a written checklist covers all required sections and deliverables; no predictor contains an ID, outcome, or forbidden post-admission field; all learned preprocessing is fitted within CV folds. Independent verification passed 2026-11-05. Historical split provenance remains documented as a limit; it was not tested by recreating the split or inspecting held-out rows.
- [ ] **T06 — Freeze and calibrate final models.** Select the final estimator per target using development data and AUPRC; record model, parameters, and selection evidence before test access. Fit the assignment-required sigmoid/Platt calibration using development-only cross-fitting. Compare isotonic calibration if pursued, with the method and rationale documented. Acceptance: choices for all three targets are frozen in a dated log; calibration fitting never uses test labels or scores.
- [ ] **T07 — Final held-out evaluation.** Only after the approval gate in `AGENTS.md`, run the frozen final models once on the untouched test partition. Acceptance: report per-target AUROC, AUPRC/average precision, and Brier; include the selected threshold, confusion matrix and threshold metrics, and calibration results. Do not retune from test results.
- [ ] **T08 — Assignment probability export.** Create one row per held-out test record with the guide's required ID and probability columns: `ID`, death probability corresponding to `LET_IS`, `ZSN`, and `FIBR_PREDS`. Acceptance: exact documented header, unique IDs, expected row count, finite probabilities in `[0, 1]`, and row/ID alignment with the locked test partition. Do not include test labels in the submission probability file.
- [ ] **T09 — Complete report and required explanatory outputs.** Reuse the existing stage reports where appropriate. Acceptance: final report documents data split and preprocessing; gives comparable CV results for all three required model families and all three targets; reports final test results; includes before/after calibration plots, ECE and calibration slope; includes a mortality SHAP summary plot; answers the guide's four required project questions and two selected literature-review questions; and has a separate extra-credit section if applicable.
- [ ] **T10 — Reproducible delivery and independent review.** Document how to run the complete code and its dependencies; verify all required files are present and internally consistent. Acceptance: independent verifier records PASS for every applicable item in `verification.md`, all run records are present in `logs/`, and the final diff contains no unintended changes.

## Blockers and gates

- T07 and any test-partition inspection are blocked until T06 is frozen and the user approves opening the locked test partition.
- Do not infer that an existing stage output meets a final-deliverable requirement without checking it against the guide and `verification.md`.
