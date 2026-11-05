# Stage 8 TabPFN configuration

- Package: `tabpfn==9.2.0` (latest stable at compatibility review: `9.2.0`).
- Model family: TabPFN-3.5, the default local classifier checkpoint for package 9.2.0.
- Python: `3.14.7`; scikit-learn: `1.9.1`; PyTorch: `2.15.0`; device selection: `auto`; observed CUDA available: `False`, MPS built: `True`, MPS available: `True`, expected auto-selected device: `mps`.
- Outer CV: shuffled stratified 5-fold, seed `42`, target-specific eligible development rows; one fold is fit and released before the next.
- Predictors: all 102 permitted Stage 1 admission-time fields. Numeric fields are passed as numeric; the declared categorical fields `SEX`, `ant_im`, `lat_im`, `inf_im`, `post_im` are pandas categorical dtype. Category vocabularies are learned from each training fold only; validation-only category values map to missing.
- Missing values remain missing. No scaling, imputation, one-hot encoding, class weights, resampling, calibration, or hyperparameter search is applied. Internal TabPFN preprocessing is part of the model fit on outer training data.
- Fixed classification threshold: probability `>= 0.5`; AP is the primary ranking metric; no threshold tuning.
- Classifier: `TabPFNClassifier(random_state=42, device='auto', n_preprocessing_jobs=1, memory_saving_mode='auto', show_progress_bar=False)`; other classifier parameters are defaults.
- Package code is Apache-2.0; TabPFN-3.5 weights are under a separate non-commercial/non-production license. The license says it is unavailable to consumers under German BGB §13 and requires acceptance by a person acting for professional purposes. The user must review and accept it before weight download/use.
- Official memory guidance reports about 3 GB VRAM for an A100 example with 1,000 training rows, 1,000 test rows, and 200 features. This is not an Apple MPS or 8 GB MacBook Air guarantee. Here folds are run sequentially with about 1,088 training and 272 validation rows and 102 predictors; measured runtime/memory are reported after execution.
- Memory reporting: process lifetime peak RSS plus MPS driver allocation after each fold when available; these are diagnostics, not a universal peak-memory guarantee. Runtime is fit plus validation prediction time.
- Official sources reviewed 2026-11-05: https://github.com/PriorLabs/TabPFN/releases (v9.2.0); https://docs.priorlabs.ai/models; https://docs.priorlabs.ai/troubleshooting/OOM-errors; https://docs.priorlabs.ai/models/accessing-model-weights; https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE.
- Fold signatures checked against saved Stage 2–6 records: `105` model-fold signatures (3 targets × 7 reference models × 5 folds).
