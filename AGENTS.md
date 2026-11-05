# Project instructions

## Sources of truth

- `AI-HW2-Guide.pdf` is the authoritative assignment specification. Preserve its requirements; do not replace them with a narrower interpretation from a stage report.
- `TASKS.md` is the task and acceptance checklist. `progress.md` is the current handoff. `verification.md` defines independent checks. Existing stage reports document completed work and results; update or extend them only when a task requires it.
- Keep each change focused on one task. Do not duplicate long methodology or results across these harness files and the stage reports.

## Project and data constraints

- The project uses `MI.data`, `mi_column_dictionary.csv`, five Python scripts (`mi_stage1_eda.py` through `mi_stage4_extratrees.py`, including `mi_stage3_fold_comparison.py`), and the existing `.venv`. There is no committed dependency manifest or test suite at harness creation.
- Preserve the assignment's fixed, stratified 80/20 development/test split and its deterministic seed. Keep `X_test.csv` and `y_test.csv` locked until model and calibration choices are frozen using development data. Do not recreate the split downstream.
- Predict the required binary outcomes `DEATH_BINARY` (derived from `LET_IS > 0`), `ZSN`, and `FIBR_PREDS` separately. Never include record IDs or outcome columns among predictors. Use only assignment-permitted admission-time features; retain the documented post-admission exclusions.
- Fit imputation, encoding, scaling, feature selection, resampling, and other learned preprocessing only on the corresponding training fold, inside the model pipeline. Do not let validation or test data influence fitting or selection.
- Compare the assignment's required prevalence baseline, regularized logistic regression, and tree ensemble. Use AUPRC/average precision as the primary selection metric; also report AUROC and Brier score. Accuracy alone is not an acceptable selection basis. Preserve the natural class prevalence when estimating probabilities and explain any class reweighting or resampling.
- Keep experiments reproducible: record seeds, split/CV design, parameters, package versions, commands, and outputs. Do not silently discard failed folds or warnings.
- Preserve source data and existing stage outputs. Do not overwrite or delete them as cleanup. Write new work to appropriately named stage output files and review the diff before completion.

## Approval gates

Obtain the user's explicit approval before:

1. Opening the locked test partition for the final evaluation. First freeze and record the selected model, calibration method, and threshold using development data. Once approved, evaluate the test partition once for the final report and outputs; do not use those results to retune.
2. Deleting, replacing, or broadly regenerating source data or established output artifacts.
3. Installing/downloading dependencies, accessing external services, or submitting/publishing project materials.
4. Starting compute-intensive experiments beyond the bounded, documented assignment workflow.

Routine, bounded execution of existing scripts against their documented development-only inputs may proceed when needed for an approved task. Log runs and inspect whether generated artifacts changed.

## Implementation and verification roles

- The implementation session makes the scoped change, records commands and evidence in `logs/`, and updates `progress.md` only with facts supported by that evidence.
- Independent verification is a separate, fresh Codex session. It follows `verification.md`, reviews the relevant diff and artifacts, and reports PASS/FAIL with evidence. It does not edit implementation code or outputs. Do not mark a task complete until the independent review passes.
- If verification fails, keep the task open, record the failure and next action, and return to implementation. Do not hide, overwrite, or relabel a failed run as a pass.

## Resuming work

At the start of each session, read this file, `TASKS.md`, `progress.md`, `verification.md`, the latest relevant run log, and `git status`. Continue the first unblocked task in `TASKS.md`; do not repeat completed work unless verification identifies a reason. Make each session's changes and log entry reviewable before handing off.

## Multi-Agent Workflow

The main Codex agent acts as the orchestrator.

### Responsibilities
- Read AGENTS.md, TASKS.md, progress.md, and verification.md before starting work.
- Select one clearly defined task at a time.
- Delegate implementation to the `implementer` subagent.
- After implementation, delegate an independent review to the `reviewer` subagent.
- Do not review your own implementation as a substitute for independent review.

### Verification Loop

1. Assign a bounded task to the implementer.
2. Inspect the implementation report.
3. Ask the reviewer to independently inspect the actual changes.
4. If the verdict is FAIL:
   - Send the findings to the implementer.
   - Request targeted fixes.
   - Ask the reviewer to check again.
5. If the verdict is PASS:
   - Check that all acceptance criteria are satisfied.
   - Update progress.md.
   - Mark the task complete in TASKS.md, if appropriate.
   - Continue to the next task.

### Rules
- Never mark a task complete without verification evidence.
- Do not allow the implementer to access the locked test set.
- Do not let multiple agents modify the same files simultaneously.
- If a task is blocked or requirements are unclear, stop and report the issue.
- Keep progress.md and verification.md consistent with actual project state.
