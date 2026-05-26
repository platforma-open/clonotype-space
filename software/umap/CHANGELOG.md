# @platforma-open/milaboratories.clonotype-space.umap

## 2.14.0

### Minor Changes

- dd55c8f: Positional k-mer encoder for short peptides

## 2.13.1

### Patch Changes

- 45681c5: Fix CPU SVD path computing the same TruncatedSVD twice when the 95% explained variance target is not reached within the component cap. The already-fit model is now reused via `transform()` instead of being re-fit via `fit_transform()`. Roughly halves runtime of the SVD step on inputs that hit the component cap (e.g. small datasets with high k-mer entropy).
- 52a2b3f: Refactor `software/umap/src/main.py` for readability without changing behavior:

  - Split the ~340-line `main()` into focused functions (`parse_args`, `validate_args`, `load_and_filter_input`, `run_gpu_pipeline`, `run_cpu_pipeline`, `write_outputs`).
  - Extract magic numbers (`RANDOM_STATE`, `SVD_TARGET_VARIANCE`, `SVD_MAX_COMPONENTS`, `UMAP_FIT_MAX_SAMPLE_SIZE`, `KMER_PARALLEL_THRESHOLD`) to module-level constants.
  - Replace the `_orig_print = print; def print(...)` shim with `print = functools.partial(print, flush=True)`.
  - Drop dead code (redundant `import os` inside `kmer_count_vectors`, redundant `(ImportError, Exception)` catches, dead UMAP backend choice check).
  - Surface previously-discarded `n_components_used` and `explained_var_sum` in the SVD completion log line.
  - Fix the skipped-clonotypes summary message hardcoding "amino acid" — now matches the configured alphabet.
  - Change `Error: Input file is empty`/`No non-empty sequences after filtering` (which exited 0) to `Warning: ...` to match the exit code.
  - Use `(arr).sum()` / `float(...)` on numpy arrays instead of Python `sum()`.
  - Update module and `compute_svd_embedding` docstrings; add a determinism note about CPU vs GPU SVD divergence.
  - Use `if backend in (...)` instead of `if backend == 'a' or backend == 'b'`.
  - Relax `umap_min_dist` validation upper bound (defer to umap-learn).

## 2.13.0

### Minor Changes

- 11dff2f: Improved performance on large datasets.

## 2.12.0

### Minor Changes

- b014947: Improved automatic detection and graceful handling of GPU availability

## 2.11.0

### Minor Changes

- 8b5acc7: fix scfv input, dependencies updates

## 2.10.0

### Minor Changes

- 45fcc5c: Allow to select feature for calculations

## 2.9.0

### Minor Changes

- 8f155eb: Fix possible hanging of python script

## 2.8.5

### Patch Changes

- 86cdd7d: Improve UMAP performance by using parallelization for kmer generation and gpu for svd

## 2.8.4

### Patch Changes

- 38d8642: Block metadata update

## 2.8.3

### Patch Changes

- a680286: Fix GPU usage

## 2.8.2

### Patch Changes

- ecb03c9: Install CUDA dependencies only in Linux

## 2.8.1

### Patch Changes

- 09d3ae6: Fix cuda error

## 2.8.0

### Minor Changes

- d2c5287: Use new python env with RAPIDS

## 2.7.6

### Patch Changes

- b5f75cb: technical release
- d7482f5: technical release
- afa2968: technical release
- b467126: technical release

## 2.7.5

### Patch Changes

- 8fca895: technical release

## 2.7.4

### Patch Changes

- 59192ec: Update python

## 2.7.3

### Patch Changes

- 27066d8: Full SDK update

## 2.7.2

### Patch Changes

- 22d11c3: Updated SDK.

## 2.7.1

### Patch Changes

- 611033a: Update used python package versions

## 2.7.0

### Minor Changes

- efcaca3: Add UMAP log to UI and fix UMAP error on insufficient clonotype sequences

## 2.6.0

### Minor Changes

- 7cd404b: Fix software issue

## 2.5.0

### Minor Changes

- a89c77d: try to update software

## 2.4.0

### Minor Changes

- 5142159: Updated to handle non valid aa sequences as input

## 2.3.0

### Minor Changes

- 009c475: Deal with empty inputs and implement batch system

## 2.2.4

### Patch Changes

- 280e8c4: [sdk/msa] seqlogo overlaps

## 2.2.3

### Patch Changes

- 888dd27: SDK Upgrade

## 2.2.2

### Patch Changes

- c25ef01: Update SDK

## 2.2.1

### Patch Changes

- 095aa9a: SDK update

## 2.2.0

### Minor Changes

- 934ea49: Improved parameter handling and dimension calculation.

## 2.1.0

### Minor Changes

- 2ef021c: Update umap script and plot defaults

## 2.0.1

### Patch Changes

- eaf8a85: Update scripts

## 2.0.0

### Major Changes

- 9d10697: MVB
