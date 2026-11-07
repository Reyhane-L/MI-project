# Independent verification procedure

## T09 Question 2 — 1:1 undersampling probability correction (2026-11-07)

**Independent read-only verdict: PASS.** The reviewer checked the revised
source, tests, exact output set, and saved evidence without opening the locked
test partition.

- Confirmed five-fold stratified seed-42 CV on Stage 1 development inputs only.
  The Stage 2 model/preprocessing fit only on a disjoint, undersampled
  model-fit subset. A disjoint 20% stratified calibration subset retained its
  natural class distribution; the existing `PlattCalibrator` was fit only to
  base scores and labels from that subset. Outer validation labels were not
  used for model, preprocessing, prior-correction, or Platt fitting.
- Confirmed raw, prior-corrected, and Platt scores cover the same 1,360 OOF
  rows. Independently checked disjointness audit flags, five folds with three
  method rows each, finite bounded probabilities, and exact eight-output
  membership. Recomputed pooled probability metrics.
- Nine focused synthetic tests passed. Nine fold-local `KFK_BLOOD` median
  imputation warnings (folds 1, 2, and 4) remain in the manifest and log.
- Pooled observed prevalence was 0.15956; mean probabilities were 0.34112
  raw, 0.17676 prior-corrected, and 0.16628 Platt. Brier scores were 0.15202,
  0.09637, and 0.09376; log loss 0.48017, 0.32459, and 0.31032; quantile ECE
  0.18309, 0.04356, and 0.01229, respectively.

This completes only the Question 2 experiment; it does not complete T09 or
authorize any further held-out data access. The prior-correction-only output
set is preserved; the T09 answer uses the newer three-method experiment.

## T09 Question 3 — Cost-sensitive threshold (2026-11-07)

**Independent read-only verdict: PASS after targeted correction.** The initial
review found that 12-digit CSV formatting collapsed the maximum score and the
all-negative `nextafter` sentinel. The original output directory was preserved;
the corrected output was written to `mi_cost_sensitive_threshold_outputs_v2/`.

- Confirmed the experiment reuses T06's selected CatBoost-tuned, Platt-calibrated
  `DEATH_BINARY` outer OOF probabilities and saved development threshold
  (0.1533359551). No model/calibrator was fit and no test or Stage 11 data was
  opened. Both input hashes are in the manifest.
- Recomputed all three comparison points on the same 1,360 OOF records.
  Minimum cost `10*FN+FP` was 505 at threshold 0.101369151773
  (TP=197, FP=305, TN=838, FN=20); threshold 0.5 cost 1,079 and frozen
  threshold cost 636.
- Confirmed round-trip threshold precision preserves all 1,362 candidate
  thresholds in the saved curve, including the all-negative sentinel (TN=1,143,
  FP=0, FN=217, TP=0, cost=2,170). Five focused tests passed, including a
  serialization regression test. The exact eight-file v2 output set is present.
- The report distinguishes 36.91 alerts per 100 patients from 14.49 deaths
  identified per 100 patients. The selected empirical threshold is descriptive
  and does not modify the frozen model or threshold.

This completes only the Question 3 development experiment; it does not
complete T09 or authorize further held-out access.

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

## T06 — Development-only finalist and calibration freeze

1. Read `mi_stage10_t06_freeze.py`, `tests/test_stage10_t06_freeze.py`, the
   dated T06 implementation log, and the T06 checklist. Do not invoke the
   approval-gated model run, run T07, create `stage10_t06_non_tabpfn_outputs/`, or inspect
   any held-out rows.
2. Confirm all runtime data paths are explicit Stage 1 development paths:
   `X_train.csv`, `y_train.csv`, `train_record_ids.csv`, and
   `column_dictionary.csv`, plus the saved development-only Stage 3, Stage 4,
   and Stage 6 aggregate AP summaries used for finalist evidence. Confirm no Stage 9
   prediction input is consumed and no held-out input is referenced.
3. Check that the fixed models and parameters match the manifest: tuned
   CatBoost for `DEATH_BINARY`, tuned Random Forest for `ZSN`, and preliminary
   Random Forest for `FIBR_PREDS`. Confirm tuned configurations are selected
   from saved Stage 6/3 outer-fold best configurations by the documented
   Hamming-medoid rule, with earliest-fold tie breaking, then held fixed across
   cross-fitting. Confirm the CatBoost and RF preprocessing pipelines are
   importable and fold-local; no class weighting or resampling is introduced.
4. Review the nested cross-fitting path: five seed-42 stratified outer folds;
   fresh five-fold stratified inner OOF base predictions from fixed settings
   for each outer training partition; each calibrator fit only on inner OOF
   scores/labels; outer validation scored once by a model fitted only on its
   outer training partition. Verify ID/label/fold alignment and one outer OOF
   row per eligible development ID/target. Confirm calibrated output
   probabilities are finite and bounded before they are written.
5. Run only the focused unit tests:
   `.venv/bin/python -m unittest discover -s tests -p 'test_stage10_t06_freeze.py' -v`.
   Confirm the tests use an injected deterministic estimator / tiny dummy
   model and do not fit a project estimator on held-out data. Check
   metric/confusion calculations, the two-part optional-calibrator criterion,
   threshold sensitivity/specificity and tie-breaking, fold isolation, OOF ID
   alignment, medoid configuration provenance, and joblib bundle prediction.
6. If the user later approves and a T06 output set exists, review only the
   explicitly named artifacts listed by `OUTPUT_FILES` in the script. Check
   exact output-set membership, manifest/warnings/config provenance, one
   calibration-comparison row per target/method, and five fold-level metric
   records embedded in each row. Check the selection CSV contains saved
   development AP means/SDs and a rationale, and that the OOF CSV has one row
   per eligible development ID/target, aligned to labels/folds with finite
   bounded raw/calibrated probabilities. Verify selected threshold metrics
   and confusion counts, including unmet-constraint reporting, Platt freeze
   statements, the three loadable bundles, saved calibration parameters, and
   reliability plots. Do not regenerate outputs. Recompute saved metrics from
   the OOF CSV and verify the optional-method screen independently.
7. Return PASS only if the implementation, tests, and any approved artifacts
   meet all T06 criteria and no material finding remains. Record evidence and
   verdict here; implementation status must remain pending until this fresh
   independent review passes. Do not infer approval for T07 from a T06 pass.

### Non-TabPFN finalist replacement review (2026-11-06)

**Overall verdict: PASS — replacement T06 complete.** Independent read-only
review checked the active implementation and the new development-only freeze.

| Acceptance criterion | Verdict | Evidence |
|---|---|---|
| Replacement finalists and fixed tuned configurations | PASS | CatBoost tuned for `DEATH_BINARY`, Random Forest tuned for `ZSN`, and preliminary Random Forest for `FIBR_PREDS`; tuned settings match the Stage 6/3 Hamming medoids and tie rule. |
| FIBR_PREDS selection trade-off | PASS | Manifest/report state AP means and fold SDs, the 0.000311 AP gap, and RF's slightly better AUROC/Brier/Log Loss; regression tests cover the rationale. |
| Nested development cross-fitting, fold-local preprocessing, calibration and thresholds | PASS | Source, saved OOF, calibration, and threshold artifacts reviewed; all thresholds use replacement-model pooled outer Platt OOF and meet sensitivity >= 0.80. |
| Three bundles load and predict in fresh processes | PASS | All three loaded via ordinary `joblib.load` and produced finite probabilities in `[0, 1]` on development rows; outputs matched Platt transforms and threshold metadata matched the replacement table. |
| Exact output membership and T07 routing | PASS | Replacement directory contains the 13 declared T06 outputs; T07 requires `freeze_revision == "non_tabpfn_v1"` and points only to the new directory. |
| Tests and active TabPFN/PyTorch assumptions | PASS | Full test suite: 45 passed; no TabPFN/PyTorch imports or inference assumptions remain in active T06/T07 code. Historical experiment scripts/results and shared environment were retained. |
| Held-out data boundary | PASS | No held-out input was accessed and T07 was not run. |

The initial independent review requested the supporting-metric explanation for
retaining RF on `FIBR_PREDS`; implementation added the evidence and regression
test, regenerated only the new T06 output set, and the follow-up review passed.
This T06 PASS does not authorize T07 or access to held-out data.

### T06 implementation-only review record (2026-11-05)

- Independent read-only code review: **PASS for implementation**, after two
  findings were fixed. The reviewer confirmed that pooled and per-fold
  comparison records retain accuracy, precision, recall, specificity, F1, and
  TN/FP/FN/TP at 0.5; and that zero-minimum-sensitivity thresholding includes
  an all-negative cutoff.
- The reviewer checked the fixed-finalist evidence against the saved Stage 3
  and Stage 8 aggregates and found no other material implementation or
  leakage issue. Its focused T06 test run passed 14 tests.
- The implementation session reran the combined Stage 10 suite: 22 tests
  passed. Pylance syntax validation and diagnostics were clean.
- No finalist models were fitted and no T06 output directory/artifact exists.
  Thus output consistency, actual bundle loading from generated files, and
  run-time warning/provenance checks are still pending. T06 is **not complete**;
  this record does not authorize T07 or held-out access.

### T06 memory-lifecycle correction review (2026-11-05)

- Independent read-only code review: **PASS for the memory-lifecycle and
  output-staging correction**. The reviewer confirmed that inner/outer
  estimators and final estimator/bundle references are released in `finally`
  blocks, and that staged output is published only after the expected files
  exist. The associated injected-estimator tests cover cleanup calls and
  sequential release.
- The implementation session reran
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_stage10_t06_freeze.py' -v`:
  **16 tests passed**. AST parsing and `git diff --check` passed; the known
  category-handling `Pandas4Warning` remains.
- No finalist fit/retry, T06 artifact generation, held-out access, or T07 run
  occurred. `stage10_t06_outputs/` remains absent. This review does not verify
  generated artifacts or complete T06; a newly approved development run and
  independent output checks remain pending.

### T06 generated-output verification (2026-11-06)

**Overall verdict: FAIL.** The independent reviewer used only the named T06
outputs and explicit Stage 1 development files.

| Acceptance criterion | Verdict | Evidence |
|---|---|---|
| Explicit development-only inputs, fixed finalists/configurations, fold-local categorical schema, and nested five-fold cross-fitting | PASS | Source review matched the checklist; no held-out file was accessed. |
| Focused injected-estimator tests | PASS | 16 tests passed; the known category-handling `Pandas4Warning` appeared. |
| Exact output-set membership, manifest expected-output list, and reliability PNGs | PASS | Exactly 13 `OUTPUT_FILES`; all three plots are valid PNGs. |
| OOF ID/label/fold alignment and one prediction per eligible record/target | PASS | 4,080 rows (1,360 per target); labels, IDs, outer folds, and final-calibrator folds match the development files and recomputed seed-42 folds. |
| Probability validity | PASS | All six probability columns are finite and in `[0,1]`. |
| Calibration comparisons and optional-method screens | PASS | 15 target/method rows with five fold records each; independently recomputed pooled/fold metrics had zero mismatches; all nine optional screens match and none qualifies. |
| Finalists, saved development AUPRC evidence, Platt freeze, and thresholds | PASS | Configurations/rationales match; Platt remains frozen; independently recomputed thresholds/metrics match and meet 0.80 sensitivity. |
| Manifest, provenance, warnings, and report consistency | PASS | Manifest and warning CSV agree; zero fit warnings; provenance/configurations/freeze statements are consistent. |
| Three bundles load with ordinary `joblib.load` and predict valid probabilities | FAIL | Each bundle raises `AttributeError: module '__main__' has no attribute 'FinalModelBundle'` in a fresh ordinary loader. A temporary `__main__` alias allowed each bundle to score two development-only rows, but that workaround does not satisfy the requirement. |
| Held-out boundary and T07 restriction | PASS | No held-out files were accessed; T07 was not run. |

At this verification stage, the bundle import-path defect blocked T06. The
generated artifacts were preserved. An importable bundle class and
fresh-process serialization regression test were added; the focused suite
passed 17 tests, and an independent code review passed. The user then approved
replacement of only the three bundle files, which were reserialized without
model fitting. The independent follow-up verification below resolves the
remaining criterion.

### T06 bundle serialization fix review (2026-11-06)

- Independent read-only code review: **PASS for the serialization fix**. Direct
  script execution imports `FinalModelBundle` from
  `mi_stage10_t06_bundle.py`; the freeze module re-exports it; and the
  regression test loads a script-style serialization in a fresh Python process
  with ordinary `joblib.load`, without a `__main__` alias.
- The focused suite passed 17 tests; syntax and whitespace checks passed.
- This review did not change the earlier generated-output verdict: at the time
  of that review the three existing bundle files still used the invalid
  `__main__` path. The user subsequently approved reserializing only those
  three files. See the independent follow-up verification below.

### T06 repaired-bundle independent follow-up (2026-11-06)

**Overall follow-up verdict: PASS.** This resolves the only failure in the
initial generated-output verification. Taken together, the initial verification
and this follow-up establish T06 acceptance.

| Follow-up criterion | Verdict | Evidence |
|---|---|---|
| Ordinary fresh-process loading, without `__main__` alias | PASS | Each bundle loaded in its own fresh subprocess as `mi_stage10_t06_bundle.FinalModelBundle`. |
| Bundle target/model/calibration/threshold metadata | PASS | Target, model, Platt method/parameters, threshold, and threshold status matched the manifest and selected-threshold row. |
| Valid prediction from development-only sample | PASS | Each bundle produced three finite `[0,1]` probabilities from the first three rows of `stage1_outputs/X_train.csv`. |
| Exact Stage 10 output membership | PASS | `OUTPUT_FILES` and the output directory contained the same 13 files; no missing, extra, non-file, or temporary repack entries. |
| Held-out boundary and no T07 | PASS | No held-out artifacts were accessed; T07 was not run. |

The independent reviewer ran the bundles sequentially with offline flags and
made no model fits or artifact changes. Example probabilities were
`DEATH_BINARY`: 0.024386, 0.031072, 0.000309;
`ZSN`: 0.682371, 0.076316, 0.075347;
`FIBR_PREDS`: 0.093561, 0.008772, 0.036415.

### T06 final status

**PASS — T06 complete.** The initial generated-output review passed all
criteria except ordinary loading of the three bundles. After the approved
serialization-only repair, independent fresh-process load and prediction
checks passed, exact output membership remained valid, and no other artifact
was regenerated. The focused test suite passes 17 tests. No test partition was
accessed. This does not authorize T07; that task remains gated by separate
explicit user approval.

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

## Additional Stage 7 — FIBR_PREDS diagnostics

1. Review `mi_stage7_fibr_diagnostics.py`, its dated run log, and only the
   explicitly named artifacts under `stage7_fibr_diagnostics_outputs/`.
   Do not enumerate Stage 1 files or access `X_test.csv`/`y_test.csv`.
2. Confirm all inputs are development artifacts; forbidden predictors are
   rejected; folds are deterministic, stratified, and matched against saved
   fold signatures; Stage 6 OOF IDs, labels, and fold memberships align.
3. Confirm group summaries, missingness rates, IQR outlier counts,
   validation-fold AP permutation scores, stability, and SHAP calculations.
   For SHAP, the reference sample must come from the matching training fold.
4. Confirm ablations are pre-specified, use saved fold parameters, fit
   preprocessing inside each fold, and are reported as exploratory rather
   than selected winners. Check AP, AUROC, Brier, and Log Loss across all five
   folds for each candidate.
5. Validate five folds per model, one fold metric per model/fold, expected
   output row counts, finite metric values, warning/failure visibility, and no
   record identifiers in diagnostic outputs. Inspect the report and plots.
6. Return a separate PASS/FAIL for Stage 7. Do not change implementation
   outputs or task status during independent verification.

## Additional Stage 8 — TabPFN experiment

1. Review `mi_stage8_tabpfn.py`, its dated run log, configuration and report,
   and only the explicitly named Stage 8 outputs. Do not inspect locked-test
   records or regenerate outputs.
2. Confirm official compatibility/license review and the recorded user
   approval; confirm installed package versions and that existing packages
   were not silently upgraded.
3. Confirm explicit development-only inputs, no locked-test access, the three
   required targets, seed-42 five-fold stratified CV, sequential execution,
   and matching-fold comparator checks. Check the documented limitation that
   prior comparator folds lack record-level IDs and were reconstructed.
4. Confirm native numerical/categorical/missing handling, fold-local category
   vocabulary behavior, no unnecessary imputation/scaling/encoding, no class
   weights or tuning, and fixed threshold 0.5.
5. Check all 15 fold rows and recompute AP, AUROC, Brier, Log Loss, precision,
   recall, F1, accuracy, and confusion counts from validation probabilities
   and labels. Confirm summaries include mean and sample SD.
6. Check OOF uniqueness and ID/label/fold alignment to development records,
   paired comparisons by matching fold, required output files, plot/report,
   memory/runtime evidence, limitations, and no significance claims.
7. Return a separate PASS/FAIL for Stage 8. Do not change implementation
   outputs or task status during independent verification.

## T07 implementation-only status (pre-run, 2026-11-06)

This records the state before the authorized recovery below. It is historical;
the later audited recovery and output-verification section supersedes its
statements that no production run or output directory existed.

- The approval-gated evaluator and synthetic-only tests are present in
  `mi_stage11_t07_evaluation.py` and
  `tests/test_stage11_t07_evaluation.py`. The focused command
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_stage11_t07_evaluation.py' -v`
  passed 11 tests. AST parsing and `git diff --check` passed. A no-flag CLI
  invocation exited with status 2 before input/bundle access; no T07 output
  directory was created.
- The synthetic end-to-end test uses temporary synthetic CSVs and mocked
  bundles only. No locked-test file was opened, hashed, loaded, or enumerated;
  no frozen bundle was loaded, and T07 was not run.
- The assignment guide's Brier Skill Score requirement is included. Ranking
  metrics (AUROC/AP) are separated from probability/calibration metrics in
  the report. Calibration-slope optimizer failure is recorded as unavailable
  without aborting primary evaluation metrics.
- The evaluator records the Stage 1 split source/hash and checks the held-out
  row count against `split_summary.csv` at an authorized run. Stage 1 builds
  X, y, and IDs using the same `test_idx`, but the existing CSVs contain no
  common per-row key/fingerprint; therefore exact X-to-ID correspondence
  cannot be independently established from those files alone. This limitation
  must be considered in independent verification; the split is not recreated.
- The explicit approval procedure in `AGENTS.md` was reviewed. The CLI
  acknowledgement flag does not independently establish approval; a separate
  explicit user approval remains required before any held-out access.
- At this implementation-only checkpoint, independent review was pending
  because the code-review agent failed to start. The later readiness audit
  records the successful review attempt and targeted follow-up below. T07
  remains incomplete and approval-gated; no test evaluation or T08 work is
  authorized by this implementation record.

## T07/T09 final-readiness checklist (pre-run, historical, 2026-11-06)

This was an implementation-readiness audit before the authorized recovery,
not a held-out evaluation or a PASS for T07 acceptance. Its pre-run status is
superseded by the audited recovery and output verification below. The
independent implementation-review result and targeted follow-up are recorded
here for history. At that checkpoint, evidence was inspected without reading
or hashing the locked test files.

| Requirement | Current implementation | Evidence available | Missing work | Proposed action |
|---|---|---|---|---|
| T07 authorization before test access; one evaluation only | CLI flag acknowledges separate approval but does not grant it; runner checks approval before path access; durable exclusive marker blocks retries after any attempt, including after output deletion | `AGENTS.md`; approval-refusal and marker synthetic tests; runner sequence; no T07 outputs or marker exist | Explicit approval is not present; failure after starting an approved attempt will intentionally require a new decision rather than retry | Obtain a separate explicit approval only after independent review; retain the marker and outputs unless a separately approved action changes that policy |
| Frozen T06 models, Platt calibrators, thresholds, features and ordering | Runner checks target/model identity, frozen Platt method and parameters, threshold metadata, and exact feature names/order; prediction path calls inference only | `mi_stage10_t06_bundle.py`; T06 manifest/freeze report; mocked bundle tests; initial independent reviewer found checks consistent | No authorized end-to-end inference against the actual bundles | Do not alter bundles, calibrators, or thresholds; actual bundle inference remains part of the one-time approved evaluation |
| Three binary targets, label definition, IDs and predictor safety | Validates each target, `DEATH_BINARY == (LET_IS > 0)`, unique IDs, forbidden predictors, and equal row counts | Synthetic input-validation tests; Stage 1 source | Exact saved test X-to-ID row alignment cannot be shown by equal counts/unique IDs; no shared row key exists | Keep the limitation explicit. Do not reconstruct the split downstream unless the user separately authorizes an exception to `AGENTS.md` |
| AUROC, AUPRC/AP, Brier, Brier Skill Score, Log Loss, threshold metrics and accuracy | Computed per target; confusion matrix stores TN/FP/FN/TP; report separates ranking from probability/calibration quality | Synthetic metric test; guide states BSS baseline `pi * (1 - pi)`; implementation uses `1 - Brier / (pi * (1 - pi))` with evaluated-partition prevalence; initial independent reviewer found metric calculations consistent | Actual held-out metric values are unavailable by design | Compute once only after approval; do not use results for selection or tuning |
| ECE, calibration intercept/slope, uncertainty and reliability diagrams | Equal-width ECE and equal-width/quantile reliability bins are generated; joint logistic slope is diagnostic-only; failed/unstable diagnostic does not discard primary metrics | Synthetic diagnostic test including optimizer failure; initial independent reviewer found calibration calculations consistent | Actual held-out diagnostics/plots | Generate after approval; do not refit calibration or tune from results |
| Before/after calibration comparison | Uses T06 outer-OOF development raw and fold-local Platt probabilities on the same IDs/labels; emits comparison metrics and development-only plots; does not fit a calibrator | `stage10_t06_non_tabpfn_outputs/cross_fitted_oof_predictions.csv`; T06 outputs and bundle API; synthetic output test | Final report has not yet been assembled; independent confirmation of OOF provenance | Reviewer checks paired same-row source and labels; later include the plots as development calibration evidence, distinct from test reliability diagrams |
| Run provenance and output safety | Manifest design records bundle/input hashes, frozen parameters/thresholds, versions, split source/configuration, and limitations; staging is no-overwrite | Code and synthetic end-to-end output test; no actual T07 directory created | Hashes and run details can only be recorded during authorized evaluation | Reviewer checks manifest completeness; preserve the one-time output directory and do not add labels to T08 export |
| Test-file/source-data provenance and alignment audit | No held-out file was opened, hashed, or loaded; no source-data hash was found in existing reviewed records | Stage 1 code applies one `test_idx` to X/y/IDs and writes a test row count to `split_summary.csv`; current CSVs lack a shared row key | Exact X-to-ID row matching and unchanged-source hash are unverified; AGENTS.md prohibits downstream split recreation | Do not perform the requested reconstruction audit under the current gate. A separately approved policy exception would be required before such source/test access |
| T09 split, preprocessing and comparable CV results | Existing Stage 1–10 scripts/reports contain split, fold-local preprocessing and model-comparison evidence; no final consolidated report exists | Stage reports, run records, and `TASKS.md` | Final guide-mapped report and an independent final cross-check | After T07 approval/evaluation, assemble from existing reports without changing model selection |
| T09 final test results and calibration reporting | T07 report is implemented but has not been generated; development OOF before/after plots are supported | Synthetic output test; guide deliverables; development OOF source | Authorized T07 outputs and final report | Complete only after the one-time approved T07 evaluation; keep ranking and calibration quality clearly distinguished |
| T09 mortality SHAP, four required questions, two literature questions, and extra-credit section | Not supplied by the T07 evaluator | Guide final-deliverable list; existing mortality-analysis evidence must be separately checked | Presence/completeness in the final assignment report has not been verified | Audit the relevant existing report/SHAP artifacts and answer the guide questions in T09; keep extra credit separate if applicable |
| Independent verification | Implementation review is clear after a targeted correction; held-out evaluation remains unverified | Initial code review returned FAIL for retry-after-output-deletion; targeted independent follow-up returned PASS after the durable-marker fix. The initial reviewer found the remaining approval, metric, and alignment descriptions consistent | No authorized evaluation/output review; no independent PASS exists for actual T07 acceptance | Keep T07 incomplete. After separate explicit approval, perform the one-time evaluation and request read-only verification of the outputs |

### Readiness status

- **Implementation:** present; final readiness changes include the explicit
  one-time attempt guard.
- **Synthetic tests:** PASS; `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B
  -m unittest discover -s tests -p 'test_stage11_t07_evaluation.py' -v`
  ran 12 tests successfully. AST parsing and `git diff --check` also passed.
  Pylance reported no diagnostics for the evaluator; its syntax endpoint did
  not recognize the tests file as a workspace user file.
- **Development-only provenance/calibration evidence:** PASS for the T06 OOF
  source check (1,360 rows and unique IDs per target; both label classes);
  Stage 1 split construction reviewed. No split reconstruction or source-data
  hash performed.
- **Independent review:** the initial implementation review found one medium
  guard issue; the targeted independent follow-up passed after the fix. This
  does not verify actual held-out results or complete T07.
- **Held-out evaluation:** not performed. No test file was accessed and no
  T07 outputs were created.
- **T07/T09:** remain incomplete. T08 and T09 execution are out of scope until
  T07 is approved and completed.

### T07 audited recovery and output verification (2026-11-06)

**Overall verdict: PASS — T07 evaluation complete.** The user reported explicit
TA authorization for one audited recovery on the existing split. The original
attempt marker and its metadata were preserved in
`logs/t07_recovery_20261106T192834Z/`; the evaluator's exclusive marker guard
was not changed, and the production run created a new active marker.

| Acceptance criterion | Verdict | Evidence |
|---|---|---|
| Final frozen model/calibration/threshold configuration | PASS | Preflight and run manifest confirm `non_tabpfn_v1`: CatBoost tuned / Random Forest tuned / Random Forest preliminary, frozen Platt calibration, and T06 thresholds. |
| One-time attempt control and audit trail | PASS | Original marker backup/archive match the recorded SHA-256 and metadata; new marker is present and its hash matches the manifest. Recovery decision is recorded. |
| Exact output membership and run manifest | PASS | `stage11_t07_outputs/` contains exactly the 19 declared files; manifest target order and model metadata match T06. |
| Target metrics, calibration, threshold metrics, DCA and bootstrap | PASS | Independent reviewer recomputed AUROC, AP, Brier, log loss, Brier Skill Score, ECE, threshold metrics/confusions, calibration diagnostics, bootstrap intervals, and DCA net benefit for all three targets; saved values matched. |
| Reliability and development calibration plots | PASS | All nine expected PNGs have valid signatures; development comparison and report claims match saved sources. |
| No tuning or model/calibrator changes | PASS | Evaluation uses current frozen bundles and thresholds; no fit/selection code was added or invoked by T07. |
| Held-out alignment limitation | DOCUMENTED | Existing CSVs lack a shared per-row key, so exact saved X/y/ID row correspondence cannot be independently established; report records this. |

The full report and outputs are in `stage11_t07_outputs/`. The production
command was invoked once; output publication and manifest validation confirm
completion. No additional T07 run was made.

## T08 — Assignment probability export

### T08 implementation review (pre-production)

The implementation and synthetic tests must pass independent code review
before any production export. Do not read test labels at any point.

1. Confirm the exporter never references or opens `y_test.csv`. For reuse of
   T07 predictions, verify its CSV reader selects only `ID`, `target`, and
   `calibrated_probability`; it must not materialize `true_label`.
2. Verify T07 probabilities are reused only if the T07 manifest matches the
   current `non_tabpfn_v1` models, Platt parameters, thresholds, and bundle
   hashes, and its recorded test-ID input hash matches the current saved IDs.
   Its recorded X input hash must also match current `X_test.csv`. Each target
   must have 340 probabilities, valid `[0,1]` values, and IDs in exact saved
   order. If any criterion fails or the T07 source is unavailable,
   verify the fallback uses only `X_test.csv`, saved IDs, and the frozen
   bundles' calibrated `predict_probability()` API. No `.fit()`, recalibration,
   thresholding, or test-label access is allowed.
3. Verify exact submission header `ID,LET_IS,ZSN,FIBR_PREDS`; the `LET_IS`
   column must be mapped from `DEATH_BINARY` calibrated probabilities. Confirm
   export row count is read from Stage 1 `split_summary.csv` (340), IDs are
   unique and unchanged in order from `test_record_ids.csv`, and all
   probabilities are finite and within `[0,1]`.
4. Run only the synthetic T08 unit tests before the code review. Confirm output
   staging refuses overwrite and the manifest records source provenance,
   frozen models/calibration/threshold verification, and `test_labels_loaded`
   false. Do not run the production exporter during code review.
5. After code review passes, the implementation agent may run the export once
   with explicit user authorization. A separate output verifier then checks
   the CSV header, 340 rows, ID order against saved test IDs, probability
   bounds, absence of label columns, manifest, output hashes, and no overwrite.
   Do not read `y_test.csv` or reconstruct the split. Record the known
   row-level predictor/ID alignment limitation because the existing artifacts
   have no shared row key.

### T08 production output verification (2026-11-06)

**Overall verdict: PASS — T08 complete.** The approved exporter reused the
validated calibrated probabilities already saved by T07. It loaded only
`ID`, `target`, and `calibrated_probability` from the T07 prediction file and
did not read `y_test.csv` or materialize its `true_label` column.

| Acceptance criterion | Verdict | Evidence |
|---|---|---|
| Exact assignment header and death-probability mapping | PASS | `submission.csv` header is `ID,LET_IS,ZSN,FIBR_PREDS`; `LET_IS` is mapped from calibrated `DEATH_BINARY`. |
| Expected row count, unique IDs and saved test-ID order | PASS | 340 rows; unique IDs match `test_record_ids.csv` exactly and in order. |
| Probability validity | PASS | All three probability columns are finite and within `[0,1]`. |
| T07 probability source provenance | PASS | T07 model/calibration/threshold metadata, current bundle hashes, current X input hash, saved ID hash/order, row counts, and probability values passed checks. |
| Test-label exclusion and frozen inference policy | PASS | No label input path is present; manifest records `test_labels_loaded: false` and `thresholds_applied: false`. No fitting, recalibration, or thresholding occurred. |
| Output safety and provenance | PASS | Exactly `submission.csv` and `t08_run_manifest.json` are present; submission hash matches the manifest; exporter refuses overwrite. |
| Synthetic tests and independent review | PASS | Nine synthetic tests passed. Independent pre-production code review and post-production output review both returned PASS. |
| Predictor-to-ID correspondence limitation | DOCUMENTED | Existing inputs have no shared row key, so exact row-level correspondence cannot be independently proven; saved ID order was preserved and checked. |

The production command was run once after code review. Files are in
`stage12_t08_outputs/`; the run record is
`logs/2026-11-06_t08_probability_export.log`.

### T08 independent verification follow-up (2026-11-06)

**Fresh read-only output review: PASS.** The independent reviewer confirmed
340 rows, exact header, unique saved-ID order, finite bounded probabilities,
and exact equality between submission columns and T07 calibrated probabilities
read using only `ID`, `target`, and `calibrated_probability`. T06 bundle
metadata, Platt parameters, thresholds, bundle hashes, T07/T08 manifests,
submission hash, and the exact two-file output set agree. No `y_test.csv` or
label values were accessed, and the exporter has no `y_test.csv` path. No
artifacts were modified. The known predictor-to-ID row correspondence limit
remains: the source artifacts contain no common row key.

### T06/T07 comprehensive implementation review (2026-11-06)

- T06 label validation initially failed independent review because a missing
  `LET_IS` column bypassed mortality-definition validation. The implementation
  now requires `LET_IS`, validates finite exact binary outcomes before row
  filtering/conversion, matches `LET_IS` and mortality missingness, and checks
  `DEATH_BINARY == (LET_IS > 0)`. The targeted independent re-review passed;
  the focused T06 suite passed 20 tests. The existing `Pandas4Warning` remains.
- T07 edge-case fixes ensure tied probabilities remain in the same quantile
  reliability bin, constant probabilities produce one bin, split counts must
  be finite integers, and calibration labels are validated. Its synthetic
  suite also tests decision-curve net benefit and references, seeded bootstrap
  AUROC/AP intervals, invalid single-class draw accounting, output completeness,
  frozen bundle/threshold consistency, approval refusal, and retry protection.
- The independent T07 implementation review passed. It confirmed the DCA
  formula, 5%–40% grid, treat-all/treat-none references, 1,000-replicate
  percentile bootstrap provenance, and no effect on frozen thresholds.
- Complete relevant suite command:
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -v`
  Result: **45 tests passed**. One existing `Pandas4Warning` occurred in the
  T06 category-handling test. `git diff --check` passed.
- No T06 model fit or production T07 run occurred. No locked test input was
  accessed, enumerated, read, hashed, or loaded; no attempt marker or T07
  output directory was created. T07 implementation review is not T07 output
  verification and does not authorize held-out access.
- Alignment limitation remains: the Stage 1 source uses one split index for
  X/y/IDs, but saved test CSVs have no common row key or alignment fingerprint.
  The split was not recreated. After separate explicit approval, run T07 once
  and request independent review of its generated outputs.

## T06 replacement verification checklist (2026-11-06)

This new freeze supersedes the earlier TabPFN-based T06 bundle set for active
T07. The earlier `stage10_t06_outputs/` is historical and must remain untouched.

1. Review the modified T06/T07 scripts and tests, new log, model-selection
evidence, and exact output set in `stage10_t06_non_tabpfn_outputs/`.
2. Verify the selected models: CatBoost tuned for `DEATH_BINARY`, Random
Forest tuned for `ZSN`, and Random Forest preliminary for `FIBR_PREDS`.
Recompute selected development AP means/SDs from the saved Stage 3/4/6
summaries and verify the strongest alternatives and FIBR trade-off.
3. Recompute the deterministic Hamming medoid from five saved Stage 6
CatBoost or Stage 3 RF tuned configurations. Verify earliest-fold tie
breaking, configuration provenance, and no retuning in T06.
4. Verify the successful T06 run used only Stage 1 development files and
generated fresh calibration and threshold outputs. Confirm every selected
threshold cites pooled outer Platt OOF and meets sensitivity >=0.80.
5. In separate fresh processes, load each bundle and predict only on
development rows; check target/model metadata, Platt transform, threshold
agreement, finite bounded probabilities, and ordinary joblib loading.
6. Run the full unit suite and `git diff --check`. Confirm the active T06/T07
pipeline does not import TabPFN/PyTorch and T07 rejects an older T06 manifest
revision. Do not open held-out inputs or run T07.

Implementation verification is pending. Passing this checklist verifies the
replacement development freeze only; it does not authorize or complete T07.

### T09 mortality SHAP focused review (2026-11-07)

**Verdict: PASS for the mortality SHAP deliverable only.** This does not
complete the full T09 report.

- Read-only review confirmed that `mi_stage13_t09_mortality_shap.py` uses the
  frozen `non_tabpfn_v1` DEATH_BINARY CatBoost bundle and exact ordered Stage 1
  development predictors. It computes CatBoost native Independent TreeSHAP
  with a seeded 64-row background drawn from development data; there is no
  fitting, recalibration, thresholding, or held-out/T07/T08 input path.
- The output covers 1,360 development rows and the 102 frozen features. The
  summary plot, feature CSV, explanation, and manifest agree. The explanation
  uses Stage 1 dictionary definitions, accurately describes positive/negative
  SHAP on the raw score axis, and states that the attributions are not causal.
- Provenance checks confirmed the manifest hashes match development features,
  development labels, the column dictionary, frozen bundle, and T06 manifest.
  The labels serve only as CatBoost Pool metadata for native SHAP.
- Implementation reported six focused synthetic tests passed after the
  explanation update; independent review verified the plot and documented
  interpretation without accessing held-out or T07/T08 row-level data.
- Limitation: these are development-row explanations of the frozen base
  model's raw output, not Platt-calibrated probability explanations. The
  beeswarm-style figure is monochrome and therefore does not assert feature
  value direction. No causal interpretation is supported.

### Development-only imputation leakage experiment (2026-11-07)

**Verdict: PASS.** This experiment uses only the Stage 1 development
`X_train.csv` and `y_train.csv`; the held-out partition and test IDs were not
accessed.

- Independent read-only review confirmed the Stage 2 feature groups,
  feature-specific imputation/encoding/scaling, logistic-regression
  parameters, and target list are reused. The global-imputation method learns
  medians/modes from development predictors before CV; the comparison method
  fits the Stage 2 pipeline only on each fold's training rows.
- Both methods use the same materialized per-target `StratifiedKFold`
  (`n_splits=5`, `shuffle=True`, `random_state=42`) splits. Validation checks
  cover schemas, missing and binary target labels, forbidden predictor
  columns, binary predictor codes, fold class counts/sizes, matching split
  digests, fit warnings, and output overwrite protection.
- Six focused synthetic tests passed, including checks that global imputation
  uses the development table and fold-local imputation excludes validation
  predictors. Pylance reported no syntax errors. The production run completed
  30 evaluations with no warnings; fold metrics and aggregate means/sample
  standard deviations were independently checked against the CSV outputs.
- Outputs are in `leakage_experiment_outputs/`; the input paths, versions,
  model settings, preprocessing, fold digests, output schemas, and warnings
  are recorded in the run manifest. Aggregate differences are mixed across
  metrics and targets; no claim is made that leakage must increase scores.

## Dependency and execution documentation review (2026-11-07)

**Independent review verdict: PASS for the dependency files and root README.**
This is a scoped documentation result; it does not complete T10 or substitute
for the remaining assignment deliverables.

- The reviewer checked `requirements.txt` and `requirements-optional.txt`
  against core and later-stage imports and the recorded environment. TabPFN is
  optional. The README states Python 3.14.7 as the existing tested environment
  and explicitly says clean installation and cross-platform compatibility
  have not been verified.
- The README separates core, additional experiments, and frozen evaluation
  and export instructions. It documents dependencies, commands, inputs,
  outputs, runtime and overwrite cautions, and T07/T08 restrictions.
- Review initially found an inaccurate statement about T07/T08 tests. The
  README was corrected to say those modules call evaluation/export functions
  with synthetic temporary fixtures and patched paths and do not access real
  project test data. Independent follow-up review passed.
- Safe implementation checks: installed versions match both requirement
  files; `pip check` reports no broken requirements; syntax compilation
  passed for 28 Python files; imports passed for 18 modules excluding T07/T08;
  all 20 README script references and 16 output-directory references exist.
- Six safe test modules passed (54 tests total): cost-sensitive threshold,
  leakage, undersampling probability, Stage 10 calibration helpers, T06
  synthetic freeze tests, and mortality SHAP helpers. T07/T08 test modules
  were skipped under the explicit no-execution restriction.
- No installation, Stage 1, T07, T08, model experiment, or locked test-file
  access occurred. Existing experimental outputs were not modified. A fresh
  environment installation remains untested. T10 remains open while other
  deliverables are outstanding.
