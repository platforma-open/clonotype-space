# @platforma-open/milaboratories.clonotype-space.workflow

## 3.6.0

### Minor Changes

- 075792e: Embedding mode: stream the embedding matrix instead of eager-loading it, so large inputs no longer OOM.

  The `--encoding embedding` path in `main.py` now streams the long-format parquet in fixed 4M-row batches
  rather than materializing the full N×D matrix. The CPU path collects a bounded, seeded fit sample
  (~`--max-sequences` vectors), fits PCA + UMAP on it, then re-streams to project every clonotype in
  batches, so its peak is bounded by the fit sample, independent of the clonotype count. The GPU path
  streams through cuML IncrementalPCA (partial_fit on ~100k-clonotype batches) so the raw N×D matrix
  never lives in VRAM; its footprint scales with the reduced N×k array and the fit-on-all UMAP, not N×D.

  The workflow passes the embedding length as `--dims` (from the `pl7.app/embedding/length` annotation),
  and the embedding RAM request is lowered to match the streamed footprint. The pre-PCA global vector
  dedup is dropped; the GPU path still collapses exact duplicates, but after PCA on the reduced vectors
  (the CPU path does not dedup). The fit sample is drawn over all clonotypes. These shift coordinates slightly versus prior
  builds (a memory fix — output already varies across backends and machines). PCA/UMAP parameters and
  sample sizes are unchanged. Sequence-feature mode is untouched.

### Patch Changes

- Updated dependencies [075792e]
  - @platforma-open/milaboratories.clonotype-space.umap@2.17.0

## 3.5.1

### Patch Changes

- 852fb98: Just mock change to push new version

## 3.5.0

### Minor Changes

- 490ab69: Resources request with formulas + minor UI validations

## 3.4.4

### Patch Changes

- 4be629b: Publish new version

## 3.4.3

### Patch Changes

- b0db708: Support `synthetic-repertoire-profiler` variant datasets:

  - The variantKey-axis sequence matcher now distinguishes the profiler (axis domain `pl7.app/repertoire/extractionRunId`) from peptide-extraction (`pl7.app/peptide/extractionRunId`). For profiler input it matches `pl7.app/sequence` feature-agnostically, so the whole-variant sequence (`pl7.app/feature: "amplicon-sequence"`) and each region subsequence appear as selectable sequence columns.
  - The peptide-mode min-length computation no longer runs for profiler datasets (they don't emit a `pl7.app/sequenceLength` column), so the workflow no longer fails with a missing-column error. The peptide safety check is preserved for genuine peptide-extraction input.
  - The `modality` output now reports a distinct `"amplicon"` value for profiler input instead of mislabeling it as `"peptide"`.

## 3.4.2

### Patch Changes

- ccc4b64: Migrate block onto the structurer and refresh the SDK to latest (model/ui-vue/test 1.79.15, workflow-tengo 6.6.3, tengo-builder 4.0.9, block-tools 2.11.1). Tooling now fully managed by `block-tools structure`; removed retired toolchain deps (vite, eslint-config) and dead boilerplate workflow tests.

## 3.4.1

### Patch Changes

- 60bc347: Reduce CPU request to 1 when GPU is in use. On a GPU node, the heavy compute (SVD/PCA, UMAP) runs on the GPU and the CPU only orchestrates the run plus the small k-mer counting step — so the workflow now requests just 1 CPU when `requireGpu && exec.hasGpu`, both for the exec resource and the Python `--n-jobs` flag. Keeps GPU pools densely packed without meaningfully hurting wall-clock for typical clonotype counts.

## 3.4.0

### Minor Changes

- fcd3aa3: Request GPU explicitly via `exec.builder().gpuMemory(...)` instead of relying on the UMAP software's runtime CuPy/cuML probe. The block now ships a `gpu` tag, a "Require run on GPU" checkbox (default on), and a "GPU memory (GB)" number field that is left empty by default. When the field is empty and GPU is required, the workflow falls back to a `16 GiB` request — covers T4-class hardware and the dataset sizes most clonotype repertoires fall into (<1M unique sequences after dedup). `.gpuMemory()` is gated on `exec.hasGpu && requireGpu`, so the block still runs on backends without GPU support and falls back to the existing CPU code path. Unchecking the box forces the CPU path on any backend.

  The UMAP software now prints a banner-style `GPU STATUS:` log line at startup, definitive `>>> SVD/UMAP EXECUTED ON GPU|CPU` markers at the exact return site of each compute call, and a closing `COMPUTATION SUMMARY:` line — so users have a ground-truth record of which backend ran each stage.

### Patch Changes

- Updated dependencies [fcd3aa3]
  - @platforma-open/milaboratories.clonotype-space.umap@2.16.0

## 3.3.0

### Minor Changes

- 7991214: Adapt block to embeddings

### Patch Changes

- Updated dependencies [7991214]
  - @platforma-open/milaboratories.clonotype-space.umap@2.15.0

## 3.2.0

### Minor Changes

- dd55c8f: Positional k-mer encoder for short peptides

### Patch Changes

- Updated dependencies [dd55c8f]
  - @platforma-open/milaboratories.clonotype-space.umap@2.14.0

## 3.1.0

### Minor Changes

- 7ae3673: Refactor workflow for early spec export.

  The workflow previously used `wf.prepare()`, which makes the body wait for upstream PColumn data to materialize before exports are defined. Downstream blocks (e.g. anything reading the UMAP PFrame from the result pool) could not discover this block's outputs until UMAP computation finished — even though the output specs are deterministic from the input specs alone.

  Split the workflow into:

  - `main.tpl.tengo` — outer body with no `wf.prepare()`. Builds the PColumn bundle and delegates to the inner template.
  - `process.tpl.tengo` — inner ephemeral template awaiting `PColumnBundle` (specs only, not data). Builds the UMAP input TSV, runs the UMAP exec, imports results, and assembles the output PFrame. Specs are published to the result pool immediately; data references resolve when computation completes.

  No changes to inputs, outputs, exports, or computation. Downstream blocks can now configure their inputs while UMAP is still running.

## 3.0.2

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

- Updated dependencies [45681c5]
- Updated dependencies [52a2b3f]
  - @platforma-open/milaboratories.clonotype-space.umap@2.13.1

## 3.0.1

### Patch Changes

- 5d72a91: Fix CID issues

## 3.0.0

### Major Changes

- f4731cf: Support peptides

## 2.14.1

### Patch Changes

- Updated dependencies [11dff2f]
  - @platforma-open/milaboratories.clonotype-space.umap@2.13.0

## 2.14.0

### Minor Changes

- b014947: Improved automatic detection and graceful handling of GPU availability

### Patch Changes

- Updated dependencies [b014947]
  - @platforma-open/milaboratories.clonotype-space.umap@2.12.0

## 2.13.0

### Minor Changes

- 8b5acc7: fix scfv input, dependencies updates

### Patch Changes

- Updated dependencies [8b5acc7]
  - @platforma-open/milaboratories.clonotype-space.umap@2.11.0

## 2.12.0

### Minor Changes

- 45fcc5c: Allow to select feature for calculations

### Patch Changes

- Updated dependencies [45fcc5c]
  - @platforma-open/milaboratories.clonotype-space.umap@2.10.0

## 2.11.10

### Patch Changes

- Updated dependencies [8f155eb]
  - @platforma-open/milaboratories.clonotype-space.umap@2.9.0

## 2.11.9

### Patch Changes

- 86cdd7d: Improve UMAP performance by using parallelization for kmer generation and gpu for svd
- Updated dependencies [86cdd7d]
  - @platforma-open/milaboratories.clonotype-space.umap@2.8.5

## 2.11.8

### Patch Changes

- 38d8642: Block metadata update
- Updated dependencies [38d8642]
  - @platforma-open/milaboratories.clonotype-space.umap@2.8.4

## 2.11.7

### Patch Changes

- 0ccc5ba: Fix StdoutStream

## 2.11.6

### Patch Changes

- a680286: Fix GPU usage
- Updated dependencies [a680286]
  - @platforma-open/milaboratories.clonotype-space.umap@2.8.3

## 2.11.5

### Patch Changes

- Updated dependencies [ecb03c9]
  - @platforma-open/milaboratories.clonotype-space.umap@2.8.2

## 2.11.4

### Patch Changes

- Updated dependencies [09d3ae6]
  - @platforma-open/milaboratories.clonotype-space.umap@2.8.1

## 2.11.3

### Patch Changes

- Updated dependencies [d2c5287]
  - @platforma-open/milaboratories.clonotype-space.umap@2.8.0

## 2.11.2

### Patch Changes

- f3fc1fb: Support parquet format (update SDK)

## 2.11.1

### Patch Changes

- b5f75cb: technical release
- d7482f5: technical release
- afa2968: technical release
- b467126: technical release
- Updated dependencies [b5f75cb]
- Updated dependencies [d7482f5]
- Updated dependencies [afa2968]
- Updated dependencies [b467126]
  - @platforma-open/milaboratories.clonotype-space.umap@2.7.6

## 2.11.0

### Minor Changes

- 21d9baa: Fixed deduplication

## 2.10.4

### Patch Changes

- 8fca895: technical release
- Updated dependencies [8fca895]
  - @platforma-open/milaboratories.clonotype-space.umap@2.7.5

## 2.10.3

### Patch Changes

- Updated dependencies [59192ec]
  - @platforma-open/milaboratories.clonotype-space.umap@2.7.4

## 2.10.2

### Patch Changes

- Updated dependencies [27066d8]
  - @platforma-open/milaboratories.clonotype-space.umap@2.7.3

## 2.10.1

### Patch Changes

- Updated dependencies [22d11c3]
  - @platforma-open/milaboratories.clonotype-space.umap@2.7.2

## 2.10.0

### Minor Changes

- 1751f19: fix scfv support

## 2.9.1

### Patch Changes

- Updated dependencies [611033a]
  - @platforma-open/milaboratories.clonotype-space.umap@2.7.1

## 2.9.0

### Minor Changes

- efcaca3: Add UMAP log to UI and fix UMAP error on insufficient clonotype sequences

### Patch Changes

- Updated dependencies [efcaca3]
  - @platforma-open/milaboratories.clonotype-space.umap@2.7.0

## 2.8.0

### Minor Changes

- 336dd7c: Add informative trace

## 2.7.2

### Patch Changes

- f88ef84: Ability to set CPU and MEM requests

## 2.7.1

### Patch Changes

- Updated dependencies [7cd404b]
  - @platforma-open/milaboratories.clonotype-space.umap@2.6.0

## 2.7.0

### Minor Changes

- a89c77d: try to update software

### Patch Changes

- Updated dependencies [a89c77d]
  - @platforma-open/milaboratories.clonotype-space.umap@2.5.0

## 2.6.0

### Minor Changes

- 05d2f21: hot fix for issue not updating script

## 2.5.0

### Minor Changes

- 5142159: Updated to handle non valid aa sequences as input

### Patch Changes

- Updated dependencies [5142159]
  - @platforma-open/milaboratories.clonotype-space.umap@2.4.0

## 2.4.0

### Minor Changes

- 009c475: Deal with empty inputs and implement batch system

### Patch Changes

- Updated dependencies [009c475]
  - @platforma-open/milaboratories.clonotype-space.umap@2.3.0

## 2.3.5

### Patch Changes

- e36baf0: Added mem & cpu requests

## 2.3.4

### Patch Changes

- 280e8c4: [sdk/msa] seqlogo overlaps
- Updated dependencies [280e8c4]
  - @platforma-open/milaboratories.clonotype-space.umap@2.2.4

## 2.3.3

### Patch Changes

- 888dd27: SDK Upgrade
- Updated dependencies [888dd27]
  - @platforma-open/milaboratories.clonotype-space.umap@2.2.3

## 2.3.2

### Patch Changes

- c25ef01: Update SDK
- Updated dependencies [c25ef01]
  - @platforma-open/milaboratories.clonotype-space.umap@2.2.2

## 2.3.1

### Patch Changes

- 095aa9a: SDK update
- Updated dependencies [095aa9a]
  - @platforma-open/milaboratories.clonotype-space.umap@2.2.1

## 2.3.0

### Minor Changes

- f01f4e2: Add annotations for number formatting, hide visibility of UMAP coordinates in tables

## 2.2.0

### Minor Changes

- 934ea49: Improved parameter handling and dimension calculation.

### Patch Changes

- Updated dependencies [934ea49]
  - @platforma-open/milaboratories.clonotype-space.umap@2.2.0

## 2.1.1

### Patch Changes

- Updated dependencies [2ef021c]
  - @platforma-open/milaboratories.clonotype-space.umap@2.1.0

## 2.1.0

### Minor Changes

- 5761090: chore: update deps

## 2.0.1

### Patch Changes

- eaf8a85: Update scripts
- Updated dependencies [eaf8a85]
  - @platforma-open/milaboratories.clonotype-space.umap@2.0.1

## 2.0.0

### Major Changes

- 9d10697: MVB

### Patch Changes

- Updated dependencies [9d10697]
  - @platforma-open/milaboratories.clonotype-space.umap@2.0.0
