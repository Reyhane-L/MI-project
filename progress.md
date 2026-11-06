# Progress handoff

## Current stage

T06 is complete. Current active task: **T07 implementation readiness only**. T07 has not been run; the held-out partition remains locked pending separate explicit user approval under `AGENTS.md`. T06 and T07 source/test reviews pass; actual held-out outputs still require an approved one-time run and separate output review. T08 and T09 remain pending.

## Completed

- Inspected repository structure, instructions, scripts, reports, data/output inventory, and Git state.
- Read `AI-HW2-Guide.pdf` and captured its core split, leakage, model, calibration, metric, and delivery requirements in `TASKS.md`.
- Added the project workflow instructions, task checklist, this handoff, independent verification procedures, and `logs/` directory.
- Existing implementation stages: data preparation/EDA (T01), prevalence/logistic baseline (T02), Random Forest (T03), and additional Extra Trees comparison (T04). Their final compliance has not yet been independently audited against this harness.

## Current blockers

- T07 and any test-partition inspection remain blocked until explicit user approval. Do not open test files before approval.
- Exact per-row X/y/ID alignment cannot be proven from the existing CSVs; Stage 1 source applies one `test_idx` to all three artifacts, and the limitation is documented.
- No committed dependency manifest is present; the existing `.venv` is used for the focused test suite.

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

### Stage 7 FIBR_PREDS diagnostics (2026-11-05)

- Added `mi_stage7_fibr_diagnostics.py` and new outputs in
  `stage7_fibr_diagnostics_outputs/`. It compares positive/negative groups and
  missingness, calculates fold-level AP permutation importance and stability,
  CatBoost TreeSHAP with a training-fold reference, fixed exploratory CatBoost
  ablations, fold metrics, and leave-one-positive-out sensitivity on saved OOF
  predictions.
- The workflow uses the development partition only, reconstructs seed-42
  stratified folds, checks saved fold signatures and OOF ID/label/fold
  alignment, and does not export patient IDs. All five folds were completed
  for three comparison models and three CatBoost diagnostic variants. Mean
  Log Loss is reported descriptively; AP remains the primary model metric.
- One initial run failed at the CatBoost TreeSHAP API because its validation
  Pool required labels; the required fold labels were added as Pool metadata,
  and the bounded rerun completed. A separate plot inspection found a label
  axis issue, so the first image is preserved and a corrected plot is provided
  and referenced in the report. Warnings and both corrections are recorded in
  `logs/2026-11-05_stage7_fibr_diagnostics.log`.
- Independent review passed implementation and generated artifact checks.
  The findings are exploratory and do not select a new final model. T06
  (development-only final model selection/calibration) remains active; no test
  partition was accessed.

### Stage 8 TabPFN experiment (2026-11-05)

- Added `mi_stage8_tabpfn.py` and the eight requested files in
  `stage8_tabpfn_outputs/`. TabPFN 3.5 ran sequentially on five seed-42
  development folds for each target, using native numerical, categorical, and
  missing-value inputs and a fixed 0.5 threshold. The user approved package/
  checkpoint downloads and personally accepted the model license.
- The run completed 15 folds and produced 4,080 OOF predictions. It used MPS;
  total fit/predict runtime was 388.7 seconds, peak process RSS was 2,050 MiB,
  and reported MPS allocation peaked at 3,041 MiB. No locked-test data were
  accessed.
- Mean AP was 0.750 for `DEATH_BINARY` (best saved comparator 0.733), 0.519
  for `ZSN` (best saved comparator 0.502), and 0.295 for `FIBR_PREDS` (best
  saved comparator 0.301). AP fold wins against the best-mean comparator were
  4/5, 4/5, and 3/5, respectively. These five-fold results are descriptive,
  not significance tests. Reported probability-quality tradeoffs are in the
  Stage 8 report.
- The independent code/output review passed, including OOF alignment, fold
  metrics, paired comparisons, and plot review. Stage 8 recommends at most a
  limited controlled comparison for `DEATH_BINARY` and `ZSN`; it does not
  recommend a FIBR_PREDS-specific follow-up under the stated screen. No further
  experiment is authorized by this result. T06 remains the active assignment
  task.

### Stage 9 provisional finalist calibration comparison (2026-11-05)

- Added `mi_stage9_finalist_calibration.py` and new outputs in
  `stage9_finalist_calibration_outputs/` for the user-specified provisional
  pairs: TabPFN 3.5/CatBoost tuned for `DEATH_BINARY`, TabPFN 3.5/Random Forest
  tuned for `ZSN`, and Extra Trees preliminary/Random Forest preliminary for
  `FIBR_PREDS`.
- The nested development-only run used five seed-42 outer folds and four inner
  calibration folds. All 150 fits completed, producing 8,160 OOF rows and 30
  candidate-fold metric rows. Raw and sigmoid-calibrated AP, AUROC, Brier, and
  Log Loss are reported; ECE and calibration slope/intercept diagnostics are
  supplemental. No held-out test artifacts were accessed and no previous
  stage outputs were modified.
- Independent review passed. The reviewer checked all 30 folds, all 8,160
  OOF rows against development IDs/labels/folds, and recomputed the four
  probability metrics for each raw/calibrated fold result. Ten TabPFN
  CPU-slow warnings were captured; failed and interrupted attempts and the
  successful run details are retained in
  `logs/2026-11-05_stage9_finalist_calibration.log`.
- Results are provisional. Saved tuned settings were selected on each full
  outer-training partition, so the inner calibration OOF scores are not
  independent of those setting-selection labels; this limitation is disclosed
  in the report. The threshold remains a fixed 0.5 reference. Final model,
  calibration, and threshold choices remain for T06; no test evaluation is
  authorized by this stage.

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

1. After authorized development-only execution, independently verify the
   fixed-finalist cross-fit, Platt calibration, and thresholds, then record the
   T06 freeze. Stage 9 is context only; its scores are not reused.
2. Request the user's approval before opening the held-out partition; then perform the one-time final evaluation (T07) and produce the assignment CSV (T08).
3. Complete the report and required outputs (T09), then request independent review under T10.

### T06 implementation (2026-11-05; independent verification pending)

- Added `mi_stage10_t06_freeze.py` with the fixed provisional finalists,
  nested five-fold development-only cross-fitting, five calibrators and the
  optional-method screen, Platt-frozen thresholds, full-development
  cross-fitted Platt bundles, saved TabPFN categories, and the approval-gated
  exact output set. Added focused injected-estimator tests and T06 verification
  steps in `verification.md`; refined T06 acceptance in `TASKS.md`.
- `.venv/bin/python -m unittest discover -s tests -p 'test_stage10*.py' -v`
  passed all 22 tests after the review findings were fixed. The suite also
  validates saved AP evidence, output names, and per-fold comparison encoding.
  Pylance syntax check and diagnostics were clean; initial implementation/test
  errors and their corrections, plus the visible category warning, are recorded
  in `logs/2026-11-05_t06_implementation.log`.
- An independent read-only code review passed after two findings were fixed:
  the comparison CSV now preserves pooled/fold operating-point metrics, and
  configurable zero-sensitivity thresholding includes an all-negative cutoff.
  A successful approved model-fitting run and output verification remain
  pending; T06 is not complete.
- At the time of this earlier implementation record, no finalist model had
  been fit and no approved run had been invoked. A later approved run failed
  before completing its first outer validation prediction; see the correction
  record below. T06 remains incomplete and awaits independent verification.

### T06 memory lifecycle correction (2026-11-05)

- The subsequent approved T06 run failed during its first `DEATH_BINARY` outer
  validation prediction (272 validation rows) with
  `TabPFNMPSOutOfMemoryError`: 4.45 GiB MPS allocation against a 3.73 GiB
  limit. The run was not retried. Failure details and the documented
  approval-gated invocation are recorded in
  `logs/2026-11-05_t06_implementation.log`.
- Updated fold lifecycles to release each estimator after prediction, collect
  garbage, and clear the MPS cache where available. Full-development models
  are serialized one at a time and released before the next target fit.
  Complete output sets are built in a temporary project-root staging location
  and only published after success, so model failure does not leave a partial
  `stage10_t06_outputs/`.
- Focused T06 tests passed (16 tests), including injected-estimator cleanup
  and sequential bundle lifecycle checks. Pylance syntax validation passed for
  the implementation; Python AST parsing passed for both changed Python files.
  Follow-up rerun of the focused tests also passed all 16 tests; AST and
  whitespace checks passed, with the category-handling `Pandas4Warning`
  remaining. An independent read-only review passed the cleanup and staging
  correction. After separate user approval, one offline development-only retry
  completed with exit code 0 and created all 13 named outputs in
  `stage10_t06_outputs/`; the OOF CSV has 4,080 rows for verification. The exact
  command and run evidence are in `logs/2026-11-05_t06_implementation.log`.
  The run did not access held-out data or invoke T07.
- T06 is **not complete**. Independent verification of generated outputs,
  OOF/label alignment, thresholds, comparisons, and manifest passed. Bundle
  verification failed: ordinary `joblib.load` in a fresh process cannot resolve
  `__main__.FinalModelBundle`. The full independent verdict and evidence are
  recorded in `verification.md`. The class now lives in importable
  `mi_stage10_t06_bundle.py`, with a fresh-process serialization regression
  test; 17 focused tests pass and independent code review passed. The generated
  bundles were replaced only after explicit approval, using the existing fitted
  estimators/calibrators and stable importable class; fresh ordinary
  `joblib.load` subprocesses passed for all three. No model fitting occurred.
  The independent follow-up then passed ordinary fresh-process loading and
  three-row prediction checks for all bundles, including target/configuration,
  Platt parameters, threshold metadata, finite bounded probabilities, exact
  13-file output membership, and absence of temporary files. T06 is complete;
  detailed PASS/FAIL history and follow-up evidence are in `verification.md`.
  No held-out access or T07 run occurred; T07 remains gated by separate user
  approval.

## Session record

- T06 complete: independent implementation, output, and repaired-bundle
  verification passed. T07 is still blocked pending separate explicit approval
  to open the locked test partition.

- Harness creation: read-only inspection found a clean Git worktree and no prior harness files. No implementation code or existing output was changed. Independent verification of this harness has not yet been performed.

### T07 implementation-only handoff (2026-11-06)

- Added the approval-gated evaluator and focused synthetic tests. The final
  focused suite passed 11 tests, including a temporary synthetic end-to-end
  output run, failure-tolerant calibration-slope diagnostics, Brier Skill
  Score, saved split row-count validation, and no-access checks without the
  approval acknowledgement. AST parsing and whitespace checks passed.
- The guide's approval process was reviewed. The CLI flag is only an
  acknowledgement and does not establish approval. T07 was not run and no
  held-out artifact was opened, hashed, loaded, or enumerated; no
  `stage11_t07_outputs/` directory was created.
- Stage 1 source and `split_summary.csv` support split provenance and test
  count checks, but the existing X/y/ID CSVs have no shared row-level key or
  fingerprint. Exact X-to-ID correspondence cannot be proven from current
  artifacts without additional approved evidence. This limitation is recorded
  in `verification.md` and the T07 run manifest design.
- At this implementation-only checkpoint, independent review was pending
  because the code-review agent failed to start due model-routing
  unavailability. The later readiness audit records the successful review
  attempt and targeted follow-up. T07 remains unchecked/incomplete and
  approval-gated. See `logs/2026-11-06_t07_implementation.log`.

### T07 readiness audit addendum (2026-11-06)

- Rechecked `AGENTS.md`, `TASKS.md`, this handoff, `verification.md`, the T07
  implementation and tests, Stage 1 split-generation code, the T06 bundle API,
  and relevant `AI-HW2-Guide.pdf` requirements. The guide specifies the Brier
  Skill Score baseline `pi * (1 - pi)` and reliability diagrams; the
  submission requirements also call for before/after calibration diagrams.
- Confirmed Stage 1 applies one `test_idx` to predictors, outcomes, and IDs.
  No split reconstruction or source-data hash was performed: downstream split
  recreation is prohibited, and no prior source-data hash is recorded. Saved
  CSVs have no shared row-level key, so exact X-to-ID alignment remains
  unverified.
- Confirmed T06 development outer-OOF raw and fold-local Platt probabilities
  provide the safer before/after calibration comparison on the same
  development observations. The T06 bundle also exposes base and calibrated
  probability APIs. No new calibrator is fitted.
- Added an exclusive attempt marker so an approved run cannot be retried after
  input access begins; the marker remains after success or failure, even if
  outputs are removed. Synthetic tests use a temporary marker and verify
  retry refusal after output deletion. No held-out input was read, hashed, or
  enumerated, no bundle was loaded, and no T07 output was created.
- T07 and T09 remain incomplete. The acceptance checklist, validation results,
  and independent-review status are recorded in `verification.md` and the T07
  run log. The initial reviewer found one medium retry-guard issue; the
  durable-marker correction received a targeted independent PASS. No held-out
  evaluation is authorized or performed.

### Non-TabPFN finalist replacement (2026-11-06; independent review passed)

- Replaced the active T06 finalist mapping with CatBoost tuned for
  `DEATH_BINARY`, Random Forest tuned for `ZSN`, and preliminary Random Forest
  for `FIBR_PREDS`, based on existing development-only Stage 3/4/6 results.
  Tuned settings are fixed from saved outer-fold choices via a deterministic
  Hamming-medoid rule; the rule and source folds are recorded in the new T06
  manifest and selection CSV.
- Wrote a new development-only T06 freeze to
  `stage10_t06_non_tabpfn_outputs/`, preserving the prior
  `stage10_t06_outputs/`. Fresh Platt calibrators and thresholds were generated
  for all targets. The thresholds are sourced from pooled outer Platt OOF and
  each satisfies the 0.80 sensitivity constraint.
- T07 now reads only the replacement output directory and requires manifest
  revision `non_tabpfn_v1`. Active T06/T07 imports no TabPFN/PyTorch runtime;
  historical experiments and the shared environment were retained. No
  dependency manifest exists in the project.
- The full suite passed (45 tests). Fresh subprocesses loaded each replacement
  bundle and confirmed predictions on three development rows are finite,
  bounded, and equal to the saved Platt transform of base probabilities;
  threshold metadata matched the new threshold table. Exact command, prior
  run errors/corrections, and outputs are in
  `logs/2026-11-06_non_tabpfn_replacement.log`.
- Independent implementation/output review passed after one targeted
  documentation correction. No held-out inputs were accessed and T07 was not
  run; T07 remains approval-gated.

### FIBR_PREDS selection-rationale review correction (2026-11-06)

- Independent review found the first replacement rationale omitted supporting
  metrics for retaining preliminary Random Forest despite Extra Trees having
  a 0.000311 higher mean AP. The rationale, manifest, selection CSV, and report
  now state that RF's AP 0.301054 (SD 0.107908) and ET's 0.301365 (SD
  0.125206) are effectively tied relative to fold variation, while RF has
  slightly better mean AUROC (0.725125 vs 0.721708), Brier (0.082854 vs
  0.082892), and Log Loss (0.292421 vs 0.295108).
- Added a regression test for this explanation and evidence. Rebuilt only
  `stage10_t06_non_tabpfn_outputs/` after validating a complete staged set;
  original `stage10_t06_outputs/` remains untouched.
- Full suite passed again (45); all three regenerated bundles passed fresh
  subprocess load, development-row prediction, Platt mapping, and threshold
  metadata checks. The correction is logged in
  `logs/2026-11-06_non_tabpfn_replacement.log`. Independent review of the
  corrected rationale and regenerated freeze passed; T07 remains unrun and
  approval-gated.
