# T05 requirement and leakage audit

Date: 2026-11-03 (Asia/Tehran)  
Scope: static code review and development-only artifacts. This audit never
opens, enumerates, hashes, or reads `X_test.csv` or `y_test.csv`.

## Evidence boundary

Evidence is limited to source code, the assignment guide, development artifacts
(`X_train.csv`, `y_train.csv`, `train_record_ids.csv`, and the dictionary), and
development-CV summaries. A static audit establishes declared data flow and
artifact invariants; it cannot prove what a past interactive session displayed.
T06 must retain this boundary until the approval gate for T07.

## Requirement map

| Guide requirement | Evidence | Status / next owner |
|---|---|---|
| Admission-time inputs only; exclude ID, all 12 outcomes, and source columns 93-95 and 100-105 (guide §§7-8, Table 3.3). | `mi_stage1_eda.py:60-63` names the nine excluded fields; `:92-99` constructs `X` without ID, outcomes, or them. The dictionary records 102 admission predictors, 12 outcomes, nine exclusions, and one ID. `mi_stage5_audit_checks.py` verifies absence in `X_train.csv`. | PASS for current development artifact. |
| Model independent binary death from `LET_IS > 0`, `ZSN`, and `FIBR_PREDS` (guide §7, Table 7.1). | `mi_stage1_eda.py:98-99` derives `DEATH_BINARY`; Stages 2-4 contain exactly those targets. The checker verifies the derivation for every development row. | PASS. |
| One deterministic 80/20 mortality-stratified split before modeling; preserve it (guide §9). | `mi_stage1_eda.py:27-28` sets seed 42 and 0.20; `:101-121` splits with `stratify=y["DEATH_BINARY"]` and persists partition artifacts/IDs. The development artifact has 1,360 rows, consistent with 80% of 1,700. | PASS for declared/persisted design. Do not rerun Stage 1. |
| Lock held-out data; use the remaining 80% for comparison/tuning/calibration/threshold choice (guide §9). | Stage 2 reads `X_train.csv`, `y_train.csv`, training IDs, and metadata (`mi_stage2_baselines.py:113-126`); Stage 3 reads training files (`mi_stage3_random_forest.py:90-99`); Stage 4 declares only training paths and explicitly avoids test enumeration (`mi_stage4_extratrees.py:142-153`). | PASS for reviewed code path. T07 remains blocked. |
| Learned preprocessing only inside each training fold/pipeline (guide §9). | Stage 2 `Pipeline` holds imputers, scaler, encoder, and logistic estimator (`mi_stage2_baselines.py:65-89`) and is fit per fold (`:191-213`). Stages 3-4 construct tree preprocessing inside `Pipeline` and fit within outer/inner CV (`mi_stage3_random_forest.py:48-58,120-149`; `mi_stage4_extratrees.py:54-65,212-253`). No resampler or selector exists. | PASS for reviewed development code. |
| Five-fold stratified CV; report mean and SD (guide §9). | Stage 2 specifies five shuffled stratified folds with seed 42; stages 3-4 do the same. Development summaries contain five rows per target/model. | PASS. |
| Compare prevalence baseline, regularized logistic regression, and ensemble tree model for every target (guide §10.1). | Stage 2 uses `DummyClassifier(strategy="prior")` and `LogisticRegression`; Stage 3 implements Random Forest, with Extra Trees as additional work. Aggregate files contain all required targets/families. | PASS. |
| Select by AUPRC/AP; also report AUROC and Brier; do not select by accuracy (guide §§10, 12). | Tree searches specify `scoring="average_precision"`; every development aggregate contains AP, AUROC, and Brier. Threshold accuracy is supplemental. | PASS for current CV/tuning; T06 must document final selection evidence. |
| Preserve natural prevalence; explain class weights/resampling (guide §§10.3-11). | Stage 2 has `class_weight=None`; stages 3-4 persist selected tuning parameters including any `class_weight`; no resampling code exists. | PARTIAL: T06 must explain final class weights and calibrate development-only. |
| Platt/sigmoid calibration by development-only cross-fitting per final target (guide §11). | No final calibration code, freeze record, or artifact exists. | NOT YET DONE — T06. |
| Threshold chosen without test outcomes; one final frozen test evaluation after approval (guide §§8-9, 12.5). | Current 0.5 rows are CV reference metrics only. No frozen threshold or held-out evaluation exists. | NOT YET DONE — T06, then T07 approval. |
| Probability export (guide §§8, 14). | Required held-out CSV header is `ID, LET_IS, ZSN, FIBR_PREDS`, with three probabilities in `[0,1]`; no submission file exists. | NOT YET DONE — T08. |
| Final report and explanatory outputs (guide §§11-14). | Required report elements are: split/preprocessing; comparable five-fold mean/SD AP-AUROC-Brier table for all three mandatory families and targets; frozen final-model test results plus selected-threshold confusion matrix/sensitivity/specificity/PPV/F1; calibration before/after, ECE, and calibration slope; mortality tree-model SHAP summary; answers to the four required project questions and two selected literature-review questions. No complete final report exists. Current reports are internal CV only. | NOT YET DONE — T09. |
| Reproducible runnable code and requirements (guide §14). | Existing scripts are runnable with `.venv/bin/python`, but there is no committed dependency manifest and no complete delivery runbook. | NOT YET DONE — T10. |

## Findings and decisions

1. **No predictor leakage was detected in the reviewed development artifact.**
   `X_train.csv` is 1,360 × 102, has unique feature names, and contains no ID,
   outcome, or documented post-admission field. `DEATH_BINARY` equals
   `LET_IS > 0` on every development row.
2. **The split design is traceable, but historical runtime provenance is
   limited.** The source fixes seed 42 and persists IDs/partition artifacts;
   raw integer split indices were not saved separately. Do not regenerate it.
3. **Existing comparison code uses development data and fold-fitted pipelines.**
   This audit did not fit models or regenerate outputs.
4. **Final safeguards remain outstanding.** There is no freeze record,
   calibrated final estimator, development-only threshold record, or approval.
   This is T06/T07 work and does not authorize test access.
5. **Process finding retained.** `TASKS.md` checks T01-T04 while saying their
   verification is pending, whereas `AGENTS.md` requires independent passing
   verification before completion. This audit does not alter historical status.

## Repeatable verification

```sh
.venv/bin/python mi_stage5_audit_checks.py
```

The checker uses an explicit development-only file list. It does not glob, list,
or access the locked held-out files.
