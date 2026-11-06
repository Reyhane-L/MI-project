# T09 report draft

This is a first-draft academic report. `main.tex` is the entry point; section text is under `sections/`, the bibliography is `references.bib`, and `figures/` contains copies of existing project plots. No existing artifact was moved or edited.

## Build

From the project root, run `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=t09_report t09_report/main.tex`. The build requires the installed LaTeX toolchain and standard packages listed in `main.tex`; no dependency was installed for this report.

## Source map

| Report content | Existing source |
|---|---|
| Dataset roles, split, feature definitions | `mi_stage1_eda.py`, `mi_column_dictionary.csv`, `stage1_outputs/split_summary.csv`, `t05_requirement_leakage_audit.md` |
| Comparable common-fold CV results and comparison figure | `stage6_catboost_outputs/stage6_report.md`, `model_comparison.png` |
| Frozen models, calibration, thresholds, selection rationale | `stage10_t06_non_tabpfn_outputs/t06_freeze_report.md`, `final_model_manifest.json`, `selected_thresholds.csv` |
| Final test metrics, calibration, CIs, confusion matrix, limits | `stage11_t07_outputs/t07_evaluation_report.md`, `test_metrics.csv`, `test_calibration_diagnostics.csv`, `test_metric_confidence_intervals.csv`, `test_confusion_matrices.csv` |
| Calibration plots | T06 development outer-OOF comparison plots and T07 test reliability plots copied here |
| Assignment export | `stage12_t08_outputs/submission.csv` and `t08_run_manifest.json` |
| Development mortality SHAP | `stage13_t09_mortality_shap_outputs/mortality_shap_explanation.md`, feature summary, manifest, and plot copied here |

The report does not read row-level held-out predictions. T07 aggregate reports/tables are its test-result sources. Verify the guide's formal bibliographic details and fill the six clearly labeled question placeholders before submission. The guide's optional two-patient local SHAP explanations are also not present.
