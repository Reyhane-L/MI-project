# Progress handoff

## Current stage

Harness created. Existing implementation covers Stages 1–4; the final calibrated model, held-out evaluation, submission probability CSV, and complete assignment report remain to be completed. Current active task: **T06 — freeze and calibrate final models using development data only**.

## Completed

- Inspected repository structure, instructions, scripts, reports, data/output inventory, and Git state.
- Read `AI-HW2-Guide.pdf` and captured its core split, leakage, model, calibration, metric, and delivery requirements in `TASKS.md`.
- Added the project workflow instructions, task checklist, this handoff, independent verification procedures, and `logs/` directory.
- Existing implementation stages: data preparation/EDA (T01), prevalence/logistic baseline (T02), Random Forest (T03), and additional Extra Trees comparison (T04). Their final compliance has not yet been independently audited against this harness.

## Current blockers

- Final model and calibration choices have not been frozen.
- The held-out test partition remains locked. Do not open it until T06 is documented and the user approves the final evaluation.
- No test suite or committed dependency manifest was found during harness setup.

## T05 independent verification (2026-11-05, Asia/Tehran)

- The first independent review failed because `verification.md` was missing.
  Added that procedure; the initial review incident is preserved in
  `logs/2026-11-05_t05_independent_review.log`: it enumerated held-out
  filenames but did not open, read, or hash those files.
- The second independent review passed. It reran the development-only checker,
  confirmed aligned predictor/outcome/ID rows and unique, non-null training
  IDs, and reviewed the exclusion policy, fold pipelines, AP selection, and
  AP/AUROC/Brier summaries.
- The reviewer read aggregate split counts/rates only, not held-out records.
  Historical split provenance remains limited and is documented without
  recreating the split.
- T05 is complete. T06 is the first remaining unblocked task; T07 remains
  gated on the T06 freeze and explicit user approval.

## Stage 6 CatBoost experiment (2026-11-05, Asia/Tehran)

- Added `mi_stage6_catboost.py` and wrote the nine requested artifacts under
  `stage6_catboost_outputs/`. The run uses Stage 1 development artifacts only;
  it completed all three targets and five outer folds without test-set access,
  calibration, or threshold tuning.
- CatBoost 1.2.10 was installed into the existing `.venv` after approval at
  the package-install permission gate. Search used one thread, four randomized
  candidates per outer fold, three-fold inner CV, and AP.
- The independent reviewer passed code review and output verification. Checks
  confirmed 450 prior-fold signature values, 15 CatBoost fold rows, 15 tuning
  rows, 90 paired rows, 21 aggregate comparison rows, and 4,080 OOF predictions
  aligned to eligible development IDs and labels with valid probabilities and
  0.5-threshold classes. The plot was visually reviewed.
- CatBoost led AP, AUROC, and Brier among compared models for `DEATH_BINARY`.
  It did not lead those metrics for `ZSN` or `FIBR_PREDS`. Five folds are
  descriptive; no significance claim is made. Details and the failed initial
  clone attempt are retained in `logs/2026-11-05_stage6_catboost.log`.
- Stage 6 is additional comparison work; the active required assignment task
  remains T06 (final estimator selection and calibration using development
  data only).

### Stage 6 Log Loss addition (2026-11-05)

- Added numerically clipped binary Log Loss to CatBoost outer-fold metrics,
  target summaries, aggregate results, seven-model comparisons, paired fold
  differences, report, and comparison plot. AP remains the only tuning metric.
- Earlier stage outputs did not retain per-record probabilities, so the six
  prior models were refit on the same development outer folds using their
  fixed/saved settings only to calculate comparable Log Loss. No Log Loss
  tuning or selection was performed.
- Validation passed: 15 CatBoost fold values, all target-level mean/sample-SD
  summaries, seven-model comparisons, and 90 paired rows were checked;
  CatBoost fold Log Loss recomputed from saved OOF probabilities matched within
  `1e-12`. Independent review passed; details are in
  `logs/2026-11-05_stage6_logloss.log`.

## T05 implementation record (2026-11-03, Asia/Tehran)

- Added `t05_requirement_leakage_audit.md`, a requirement-by-requirement map
  from the authoritative guide to code, development artifacts, reports, and
  remaining work. It records the no-test-access boundary and outstanding T06+
  deliverables.
- Added `mi_stage5_audit_checks.py`, an explicit development-only checker. It
  verifies the current 1,360-row/102-predictor development artifact, absence
  of ID/outcomes/nine post-admission exclusions, `DEATH_BINARY = LET_IS > 0`,
  declared 80/20 seed-42 mortality-stratified split, CV/pipeline markers, and
  required targets/model families/AP-AUROC-Brier development summaries.
- The checker passed on 2026-11-03; its command, package versions, output, and
  retained `pdfinfo`/`pdftotext` unavailable-tool warning are in
  `logs/2026-11-03_t05_requirement_leakage_audit.log`.
- In the implementation run, no model was fit, no existing stage output was
  regenerated, and neither held-out file was listed, opened, read, or hashed.
  The separate initial reviewer incident is recorded in the 2026-11-05 log.

## Next steps

1. Freeze per-target final model and calibration choices on development data (T06).
2. Request the user's approval before opening the held-out partition; then perform the one-time final evaluation (T07) and produce the assignment CSV (T08).
3. Complete the report and required outputs (T09), then request independent review under T10.

## Session record

- Harness creation: read-only inspection found a clean Git worktree and no prior harness files. No implementation code or existing output was changed. Independent verification of this harness has not yet been performed.
