# Stage 8 — TabPFN experiment

## Scope and design

This controlled experiment uses development data only (seed 42; five shuffled stratified outer folds per target). No locked test data are read. TabPFN is evaluated once per outer training fold; no outer validation data are used for model selection. Prior model metrics come from saved matched-fold Stage 2, 3, 4 and 6 outputs. The fixed threshold is 0.5. Average Precision is the primary ranking metric; threshold precision/recall/F1 and confusion counts are separate operating-point measures.

Saved Stage 2–4 fold metrics and Stage 6 comparator summaries do not contain validation record IDs. This run reconstructs the seed-42 folds from the unchanged Stage 1 development row order and checks fold sizes, validation class counts, and training prevalence. Historical record-level membership therefore cannot be directly confirmed by ID; paired comparisons use the reconstructed same-numbered folds and matching signatures.

TabPFN receives native numeric data, pandas categorical types for nominal fields, and missing values without manual imputation or one-hot encoding. The categorical vocabulary is learned on each training fold only. No class weighting, resampling, calibration, or tuning is used. Full configuration and transformations are in `tabpfn_configuration.md`.

## Results

### DEATH_BINARY

TabPFN AP 0.750 ± 0.047; best saved comparator AP: CatBoost tuned (0.733 ± 0.059). TabPFN mean AP exceeded the best comparator mean.
TabPFN AUROC 0.915 ± 0.017; mean AUROC is higher (better) than the best saved comparator, CatBoost tuned, by 0.013984 (TabPFN 0.914654; comparator 0.900670). Brier 0.073 ± 0.007; mean Brier is lower (better) than the best saved comparator, CatBoost tuned, by 0.004678 (TabPFN 0.072747; comparator 0.077425). Log Loss 0.246 ± 0.021; mean Log Loss is lower (better) than the best saved comparator, CatBoost tuned, by 0.016234 (TabPFN 0.245669; comparator 0.261902). Against the comparator with the highest mean AP, TabPFN won 4/5 paired AP folds.
At threshold 0.5, mean fold precision 0.797, recall 0.567, and F1 0.660. Per-fold confusion counts are in `fold_level_metrics.csv`.

### ZSN

TabPFN AP 0.519 ± 0.097; best saved comparator AP: Random Forest tuned (0.502 ± 0.083). TabPFN mean AP exceeded the best comparator mean.
TabPFN AUROC 0.732 ± 0.056; mean AUROC is lower (worse) than the best saved comparator, Extra Trees tuned, by 0.001120 (TabPFN 0.731931; comparator 0.733051). Brier 0.143 ± 0.015; mean Brier is lower (better) than the best saved comparator, Random Forest tuned, by 0.004127 (TabPFN 0.143413; comparator 0.147540). Log Loss 0.453 ± 0.036; mean Log Loss is lower (better) than the best saved comparator, Random Forest tuned, by 0.011440 (TabPFN 0.453430; comparator 0.464870). Against the comparator with the highest mean AP, TabPFN won 4/5 paired AP folds.
At threshold 0.5, mean fold precision 0.824, recall 0.239, and F1 0.367. Per-fold confusion counts are in `fold_level_metrics.csv`.

### FIBR_PREDS

TabPFN AP 0.295 ± 0.068; best saved comparator AP: Extra Trees preliminary (0.301 ± 0.125). TabPFN mean AP did not exceed the best comparator mean.
TabPFN AUROC 0.740 ± 0.051; mean AUROC is higher (better) than the best saved comparator, Extra Trees tuned, by 0.014694 (TabPFN 0.740348; comparator 0.725654). Brier 0.083 ± 0.004; mean Brier is higher (worse) than the best saved comparator, Random Forest preliminary, by 0.000129 (TabPFN 0.082983; comparator 0.082854). Log Loss 0.292 ± 0.014; mean Log Loss is lower (better) than the best saved comparator, Random Forest preliminary, by 0.000801 (TabPFN 0.291619; comparator 0.292421). Against the comparator with the highest mean AP, TabPFN won 3/5 paired AP folds.
At threshold 0.5, mean fold precision 0.200, recall 0.037, and F1 0.062. Per-fold confusion counts are in `fold_level_metrics.csv`.

## Overall stability recommendation

Across targets, the descriptive screen requires TabPFN mean AP to exceed the best saved comparator and win at least 4 of 5 paired folds against it. This is a practical stability screen only, not a significance test.

DEATH_BINARY and ZSN meet the descriptive screen; FIBR_PREDS does not. This supports at most a limited controlled comparison of TabPFN on DEATH_BINARY and ZSN, if separately authorized; it does not support adopting TabPFN or establish significance. No further experiment is started.

## FIBR_PREDS follow-up recommendation

For FIBR_PREDS, the transparent descriptive rule is that TabPFN must exceed the best saved comparator's mean AP and win at least 4 of 5 paired outer folds against that comparator. This is a pragmatic stability screen, not a significance test.
The current folds do not meet the descriptive rule for a dedicated FIBR_PREDS follow-up: TabPFN mean AP difference versus the best saved comparator (Extra Trees preliminary) is -0.006, with 3/5 paired AP wins; the rule requires a positive mean AP difference and at least 4/5 wins. Against that comparator, Brier and Log Loss differences are +0.000 and -0.003, respectively. If either probability-quality difference favors TabPFN, that is a separate exploratory signal, not a consistent ranking improvement; retain TabPFN as a comparator unless a focused question justifies follow-up.

## Runtime, memory, and limitations

Runtime device status: CUDA available `False`, MPS built `True`, MPS available `True`; `device='auto'` outcome `mps`.
Across 15 TabPFN fold fits, fit-plus-predict runtime was 388.7 seconds total (mean 25.9s/fold; range 14.4–184.1s). The first fit may include license authentication and lazy checkpoint download if the weights were not already cached. The largest recorded process peak RSS was 2050 MiB. MPS driver allocation after folds ranged up to 3041 MiB.
Five folds provide descriptive paired evidence only; they do not establish statistical significance. Clinical prediction metrics do not establish clinical utility. The model weights use a separate non-commercial/non-production license; see `tabpfn_configuration.md`. Memory references from official docs are A100 VRAM estimates, not a guarantee for Apple unified memory. Comparator metrics are from the project’s saved seed-42 fold results, with Stage 6 comparator Log Loss refits on matching folds.

## Output files

- `fold_level_metrics.csv`: TabPFN fold scores, threshold metrics, warnings, runtime and memory readings.
- `aggregate_metrics.csv`, `model_comparison.csv`: mean and sample standard deviation across folds for TabPFN and saved comparison models.
- `oof_predictions.csv`: one outer-validation probability per eligible development ID and target.
- `paired_comparison.csv`: per-fold TabPFN-minus-comparator differences, with metric direction recorded by column names and booleans.
- `tabpfn_configuration.md`, `stage8_report.md`, `model_comparison.png`.
