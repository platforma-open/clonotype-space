---
"@platforma-open/milaboratories.clonotype-space.umap": patch
"@platforma-open/milaboratories.clonotype-space.workflow": patch
"@platforma-open/milaboratories.clonotype-space": patch
---

Refactor `software/umap/src/main.py` for readability without changing behavior:

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
