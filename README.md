# Myocardial Infarction (MI) prediction project

This project studies prediction of three binary outcomes from admission-time
features in the myocardial infarction dataset: `DEATH_BINARY` (derived from
`LET_IS > 0`), `ZSN`, and `FIBR_PREDS`. The assignment workflow reserves a
deterministic, mortality-stratified 80/20 development/test split (seed 42),
compares a prevalence baseline, regularized logistic regression, and tree
models using development cross-validation, and reports average precision
(AUPRC) as the primary selection metric alongside AUROC and Brier score.
Calibration and threshold selection use development data. The final held-out
evaluation is a separate, approval-gated one-time stage.

The existing staged scripts and saved outputs are the source of truth for the
implemented experiments. This README describes how they connect; it does not
change their configurations or results.

## Repository layout

- `MI.data`, `mi_column_dictionary.csv`: source dataset and variable dictionary.
- `mi_stage1_eda.py` through `mi_stage4_extratrees.py`, including
  `mi_stage3_fold_comparison.py`: required split/EDA and core model comparisons.
- `mi_stage5_audit_checks.py`: read-only audit of source and stage artifacts.
- `mi_stage6_catboost.py` through `mi_stage9_finalist_calibration.py` and
  `mi_stage10_calibration_comparison.py`: additional development experiments.
- `mi_stage10_t06_freeze.py`, `mi_stage10_calibrators.py`, and
  `mi_stage10_t06_bundle.py`: active final model, calibration, and threshold
  freeze implementation and support.
- `mi_stage11_t07_evaluation.py` and `mi_stage12_t08_probability_export.py`:
  approval-gated final evaluation and assignment probability export.
- `mi_stage13_t09_mortality_shap.py`: development-only mortality SHAP analysis.
- `mi_leakage_experiment.py`, `mi_undersampling_probability_experiment.py`,
  `mi_cost_sensitive_threshold_experiment.py`: additional development-only
  experiments.
- `stage1_outputs/`, `stage2_outputs/`, `stage3_outputs/`,
  `stage4_extratrees_outputs/`, and later `stage*_outputs/` directories:
  saved inputs, metrics, models, reports, and plots from their corresponding
  stages.
- `tests/`: synthetic/unit checks implemented with Python's standard
  `unittest` framework.
- `TASKS.md`, `progress.md`, `verification.md`, `logs/`: project handoff,
  acceptance, review, and run records.

Scripts resolve data and output paths relative to their own file location.
Run commands from the project root so local module imports and paths behave as
shown below.

## Python and dependencies

The versions below are pinned to the existing environment, which reports
Python **3.14.7**. The installed core package versions are NumPy 2.5.4, pandas
3.1.0, scikit-learn 1.9.1, and Matplotlib 3.11.2. Later-stage package versions
recorded in that environment include CatBoost 1.2.10, SciPy 1.18.1, joblib
1.6.0, TabPFN 9.2.0, and PyTorch 2.15.0. The project scripts and experiments
have been run in this existing environment, but a clean installation from
these files has **not** been tested; cross-platform compatibility is not
claimed.

Core Stages 1–4 need only the dependencies in `requirements.txt`:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use `requirements-optional.txt` only for later experiments that need CatBoost,
SciPy, joblib, or TabPFN:

```bash
python -m pip install -r requirements-optional.txt
```

The optional file includes the core requirements. TabPFN is not needed for the
core assignment. Stage 8 and historical calibration experiments may trigger a
model-weight download/authentication on first fit. The TabPFN 3.5 weights have
a separate license; review and accept its terms before using those stages.
On the environment used for the saved Stage 8 run, automatic device selection
used Apple MPS. Other platforms may select a different device or require
different PyTorch support.

## A. Reproducing the core assignment

### Inputs and order

For a new reproduction, place `MI.data` and `mi_column_dictionary.csv` in the
project root. The scripts expect `stage1_outputs/` and later stage directories
at the same level. Run the core stages in this order:

```bash
python mi_stage1_eda.py
python mi_stage2_baselines.py
python mi_stage3_random_forest.py
python mi_stage3_fold_comparison.py
python mi_stage4_extratrees.py
```

Stage 1 writes the split, training-only EDA, dictionary roles, and metadata to
`stage1_outputs/`. It uses an 80/20 split, stratified by mortality, with seed
42. Stage 2 compares prevalence and regularized logistic regression and writes
`stage2_outputs/`. Stage 3 uses Stage 1 development inputs and Stage 2 metrics;
it writes Random Forest results and tuning records to `stage3_outputs/`. The
fold-comparison script reads Stage 2 and Stage 3 fold records and adds paired
comparison files in `stage3_outputs/`. Stage 4 uses Stage 1 data plus Stage 2
and Stage 3 outputs and writes Extra Trees results to
`stage4_extratrees_outputs/`.

The core comparison keeps learned preprocessing inside each training fold.
The development/test split is created only by Stage 1; downstream stages use
the saved development artifacts. The scripts do not perform additional
splitting of the held-out partition for model selection.

The expected primary outputs include `stage1_outputs/split_summary.csv`,
`stage2_outputs/model_comparison.csv`, `stage3_outputs/aggregate_metrics.csv`,
`stage3_outputs/paired_fold_differences.csv`, and
`stage4_extratrees_outputs/model_comparison.csv`, together with fold metrics,
tuning results, reports, and plots in those stage directories.

To run the source/artifact audit after the relevant outputs exist:

```bash
python mi_stage5_audit_checks.py
```

Stage 5 checks expected project inputs, source rules, and outputs; it is not a
model experiment.

### Existing outputs and overwrite caution

The project already contains Stage 1–4 outputs. In particular, Stage 1 creates
the existing fixed split files, and its output directory is reused if present.
Stages 2–4 also reuse their output directories and can replace files. **Do not
run the commands above in this working copy if you need to preserve the
current artifacts.** Reproduce them in a separate, clean copy of the project
and retain the source dataset and dictionary at that copy's root. Do not delete
or regenerate current outputs as setup or cleanup.

## B. Additional experiments

Install `requirements-optional.txt` for the stages below. Execute only stages
whose input artifacts already exist, and run from the project root.

| Experiment | Command | Required prior artifacts | Main output |
|---|---|---|---|
| CatBoost comparison | `python mi_stage6_catboost.py` | Stages 1–4, including Stage 2/3/4 folds and tuning results | `stage6_catboost_outputs/` |
| FIBR_PREDS diagnostics | `python mi_stage7_fibr_diagnostics.py` | Stages 1, 3, 4, and 6 | `stage7_fibr_diagnostics_outputs/` |
| TabPFN comparison (optional) | `python mi_stage8_tabpfn.py` | Stages 1–4 and 6; TabPFN weights/license as applicable | `stage8_tabpfn_outputs/` |
| Provisional finalist calibration | `python mi_stage9_finalist_calibration.py` | Stages 1, 3, 4, and 6; TabPFN package/weights | `stage9_finalist_calibration_outputs/` |
| Historical calibration comparison | `python mi_stage10_calibration_comparison.py` | Stage 1, 3, 8, and 9 artifacts; TabPFN package | `stage10_calibration_comparison_outputs/` |
| Imputation leakage comparison | `python mi_leakage_experiment.py` | Stage 1 development predictors/outcomes and Stage 2 source | `leakage_experiment_outputs/` |
| Undersampling probability experiment | `python mi_undersampling_probability_experiment.py` | Stage 1 development data and Stage 2/calibrator code | `undersampling_calibration_outputs/` |
| Cost-sensitive threshold analysis | `python mi_cost_sensitive_threshold_experiment.py` | Active T06 development OOF and frozen threshold outputs | `mi_cost_sensitive_threshold_outputs_v2/` |

These stages are separate experiments, not replacements for the active T06
model choices. Several are computationally expensive. Stage 3/4 perform
cross-validated parameter search; Stage 6 performs CatBoost tuning; Stage 7
repeats diagnostic fitting across folds and ablations; Stage 8 performs 15
TabPFN fits and can require substantial memory and first-fit weight retrieval;
Stage 9 and historical calibration comparison perform nested cross-fitting.
Do not launch them solely to verify installation.

The scripts that write to existing output directories generally refuse to
overwrite those experiment results. Stage 1–4 are an exception, as noted
above. Check a script's output guard and the destination directory before any
rerun.

## C. Final freeze, held-out evaluation, and export

The active frozen choices and outputs are already recorded in
`stage10_t06_non_tabpfn_outputs/`. T06 selected the models using development
evidence, fitted Platt calibrators using development cross-fitting, and froze
sensitivity-constrained thresholds. Its documented development-only command
is:

```bash
python mi_stage10_t06_freeze.py --approve-development-only-run
```

This command fits models and refuses to overwrite the existing T06 output
directory. It is a compute-intensive development run; the flag is an explicit
run acknowledgment, not a reason to repeat the existing freeze.

The one-time T07 held-out evaluation has already been completed. Its aggregate
metrics, frozen-threshold confusion counts, calibration diagnostics, plots,
and run manifest are saved in `stage11_t07_outputs/`. It must not be repeated.
The T07 CLI requires `--acknowledge-separate-user-approval`; the flag does not
grant approval, and a durable attempt marker prevents retries after an attempt.
The locked test partition must not be used for model or threshold selection.

The validated assignment export is already in
`stage12_t08_outputs/submission.csv` (header `ID,LET_IS,ZSN,FIBR_PREDS`). The
T08 script uses the saved T07 probabilities when their provenance checks pass;
its CLI requires `--acknowledge-approved-test-feature-use`, and the flag does
not itself grant approval. T08 refuses to overwrite its existing output.

For reference only, the guarded program entry points are:

```bash
python mi_stage11_t07_evaluation.py --acknowledge-separate-user-approval
python mi_stage12_t08_probability_export.py --acknowledge-approved-test-feature-use
```

**Do not run these commands in this project copy.** T07 is one-time and has
already run; T08 has already produced the export. The existing test-result
outputs are authoritative and must not be used to retune or recalibrate.

The development-only mortality SHAP analysis uses the frozen CatBoost bundle
and existing development inputs. Its output is in
`stage13_t09_mortality_shap_outputs/`; the script refuses to overwrite it:

```bash
python mi_stage13_t09_mortality_shap.py
```

## Tests

Tests use `unittest` and are discovered from the project root. In the existing
environment, set writable temporary directories for Matplotlib and the TabPFN
Skrub cache before importing/running tests:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mi-project-mpl SKB_DATA_DIRECTORY=/private/tmp/mi-project-skrub .venv/bin/python -B -m unittest discover -s tests -v
```

The T07/T08 test modules call `run_evaluation` and `run_export` on synthetic
temporary fixtures with patched paths; they do not access the real project
test data. The test suite does not need the real locked partition. A full
combined test run should still be performed in the
environment intended for submission; historical logs are not a substitute
for that environment's result.

## Reproducibility notes

- Source-stage commands use paths relative to the script directory. Run from
  the project root to follow the commands and local import layout shown here.
- The recorded split is seed 42 and stratified by the mortality outcome. Do
  not recreate it in downstream analysis or access the held-out labels before
  the approved final evaluation.
- The final test evaluation is a single completed run. A known limitation is
  that saved test predictor/outcome/ID CSVs have no shared row key, so exact
  row-level predictor-to-ID alignment cannot be independently established
  from those artifacts.
- Package versions and hardware are recorded in stage reports/manifests. A
  fresh-environment install has not been tested as part of this documentation
  change.
