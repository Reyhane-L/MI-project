# Independent verification procedure

The verifier works read-only in a fresh session and reports PASS/FAIL with
commands and evidence. Follow `AGENTS.md` and the applicable task acceptance
criteria in `TASKS.md`. Do not edit implementation code, outputs, or task status.

## Data access boundary

- For T01–T06, use source code, the assignment guide, development artifacts,
  and development-only summaries. Do not enumerate, open, read, or hash
  `X_test.csv` or `y_test.csv`; do not use a directory glob that can expose
  those names.
- T07/T08 verification may inspect held-out artifacts only after the freeze
  record and explicit approval required by `AGENTS.md` are present. Verify
  that evaluation is one-time and did not feed model selection.
- Preserve all existing artifacts. A verifier must not regenerate outputs.

## T05 — Requirement and leakage audit

1. Read `AI-HW2-Guide.pdf`, `TASKS.md`, the T05 audit, its run log, and the
   relevant scripts without accessing the held-out partition.
2. Confirm the T05 map covers the guide's data/features, split, preprocessing,
   required targets and model families, metrics, calibration, threshold,
   evaluation, export, reporting, and reproducibility requirements. T06–T10
   items may be marked outstanding; they must not be represented as complete.
3. Confirm the audit checker uses explicit development-only paths and does not
   enumerate output directories or reference held-out file paths.
4. Review Stage 1 source and development artifact evidence for the declared
   deterministic 80/20 mortality-stratified split, unique IDs, and aligned
   development predictors/outcomes. Do not recreate the split or inspect the
   held-out partition to prove historical provenance; report this limit if
   saved development evidence cannot establish more.
5. Check ID, all outcome columns, and documented post-admission exclusions
   are absent from development predictors, and confirm `DEATH_BINARY` is
   derived from `LET_IS > 0`.
6. Review Stages 2–4 data paths and pipelines for development-only fitting,
   fold-contained learned preprocessing, required model families/targets,
   AP as the selection metric, and AUROC/Brier reporting. Confirm failures and
   limitations remain visible in the log.
7. Run `.venv/bin/python mi_stage5_audit_checks.py` only if the read-only
   development-only checker needs reproduction. Record its exact output.
8. Return PASS only if the above evidence supports T05 acceptance and no
   unresolved material finding remains. Do not mark T05 complete; the
   implementation orchestrator updates task status after the review.

## Later tasks

Use each task's acceptance criteria in `TASKS.md` as the checklist. T06 must
be independently checked using development-only evidence before any request
to access the held-out partition. T07/T08 require the approval gate and the
specific output checks listed in `TASKS.md`. T09/T10 require review of the
final report, required deliverables, reproducibility record, and complete
diff. Record a separate verdict and evidence for each task; a pass for one
task does not imply a pass for another.

## Additional Stage 6 — CatBoost comparison

1. Review `mi_stage6_catboost.py`, its run log, and the nine named artifacts
   under `stage6_catboost_outputs/`. Do not use a directory listing or glob;
   open only explicit Stage 6 paths.
2. Confirm the code reads development artifacts and prior-stage CV summaries,
   uses target-specific five-fold outer and three-fold inner stratified CV,
   searches by AP, keeps learned preprocessing in the model pipeline, and
   does not calibrate or tune thresholds.
3. Confirm categorical semantics, missing-value treatment, class-weight
   policy, CatBoost parameters, and warnings are documented in the script,
   tuning/fold outputs, and report.
4. Check previous-stage fold signatures; validate all targets/folds, model
   comparisons, paired rows, and aggregate means/standard deviations.
5. Using only `stage1_outputs/train_record_ids.csv` and `y_train.csv`, verify
   each eligible development record has one OOF prediction per target, IDs and
   labels align, probabilities are finite and within [0, 1], and threshold
   labels match 0.5. Do not access held-out data.
6. Verify the comparison plot and report, including searched values,
   per-target comparison conclusions, limitations, and absence of significance
   claims. Record a separate PASS/FAIL; do not change implementation outputs.
7. For the Log Loss addition, verify numeric clipping and fold calculations
   from probabilities, mean/sample SD in target and aggregate summaries,
   matched-fold Log Loss for all six comparators, paired differences with lower
   better, AP-only tuning, and the updated report/plot. Comparator refits must
   use saved settings on the same development outer folds and preserve warnings.
