# @platforma-open/milaboratories.clonotype-space

## 3.4.1

### Patch Changes

- Updated dependencies [60bc347]
  - @platforma-open/milaboratories.clonotype-space.workflow@3.4.1

## 3.4.0

### Minor Changes

- fcd3aa3: Request GPU explicitly via `exec.builder().gpuMemory(...)` instead of relying on the UMAP software's runtime CuPy/cuML probe. The block now ships a `gpu` tag, a "Require run on GPU" checkbox (default on), and a "GPU memory (GB)" number field that is left empty by default. When the field is empty and GPU is required, the workflow falls back to a `16 GiB` request — covers T4-class hardware and the dataset sizes most clonotype repertoires fall into (<1M unique sequences after dedup). `.gpuMemory()` is gated on `exec.hasGpu && requireGpu`, so the block still runs on backends without GPU support and falls back to the existing CPU code path. Unchecking the box forces the CPU path on any backend.

  The UMAP software now prints a banner-style `GPU STATUS:` log line at startup, definitive `>>> SVD/UMAP EXECUTED ON GPU|CPU` markers at the exact return site of each compute call, and a closing `COMPUTATION SUMMARY:` line — so users have a ground-truth record of which backend ran each stage.

### Patch Changes

- Updated dependencies [fcd3aa3]
  - @platforma-open/milaboratories.clonotype-space.workflow@3.4.0
  - @platforma-open/milaboratories.clonotype-space.model@3.3.0
  - @platforma-open/milaboratories.clonotype-space.ui@3.3.0

## 3.3.2

### Patch Changes

- Updated dependencies [3e5fd18]
  - @platforma-open/milaboratories.clonotype-space.model@3.2.1
  - @platforma-open/milaboratories.clonotype-space.ui@3.2.1

## 3.3.1

### Patch Changes

- Updated dependencies [7991214]
  - @platforma-open/milaboratories.clonotype-space.workflow@3.3.0
  - @platforma-open/milaboratories.clonotype-space.model@3.2.0
  - @platforma-open/milaboratories.clonotype-space.ui@3.2.0

## 3.3.0

### Minor Changes

- 96c216b: Migrate block to BlockModelV3. Unified `BlockData` (UI-shaped persistence), `.args` lambda derives the workflow-visible shape and validates by throw. Persisted V1 state preserved via `DataModelBuilder.upgradeLegacy`. UI bindings move to `app.model.data`; `defineApp` → `defineAppV3`.

  `defaultBlockLabel` is no longer stored: the UI snapshots `sequenceLabels` into `data` on sequence-selection gesture, and the args lambda assembles `defaultBlockLabel` from `data`. Existing projects keep `customBlockLabel`; the sequence-name fragment of the default label is reseeded on the next interaction with the sequences dropdown.

### Patch Changes

- Updated dependencies [96c216b]
  - @platforma-open/milaboratories.clonotype-space.model@3.1.0
  - @platforma-open/milaboratories.clonotype-space.ui@3.1.0

## 3.2.0

### Minor Changes

- dd55c8f: Positional k-mer encoder for short peptides

### Patch Changes

- Updated dependencies [dd55c8f]
  - @platforma-open/milaboratories.clonotype-space.workflow@3.2.0

## 3.1.0

### Minor Changes

- 7ae3673: Refactor workflow for early spec export.

  The workflow previously used `wf.prepare()`, which makes the body wait for upstream PColumn data to materialize before exports are defined. Downstream blocks (e.g. anything reading the UMAP PFrame from the result pool) could not discover this block's outputs until UMAP computation finished — even though the output specs are deterministic from the input specs alone.

  Split the workflow into:

  - `main.tpl.tengo` — outer body with no `wf.prepare()`. Builds the PColumn bundle and delegates to the inner template.
  - `process.tpl.tengo` — inner ephemeral template awaiting `PColumnBundle` (specs only, not data). Builds the UMAP input TSV, runs the UMAP exec, imports results, and assembles the output PFrame. Specs are published to the result pool immediately; data references resolve when computation completes.

  No changes to inputs, outputs, exports, or computation. Downstream blocks can now configure their inputs while UMAP is still running.

### Patch Changes

- Updated dependencies [7ae3673]
  - @platforma-open/milaboratories.clonotype-space.workflow@3.1.0

## 3.0.4

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
  - @platforma-open/milaboratories.clonotype-space.workflow@3.0.2

## 3.0.3

### Patch Changes

- Updated dependencies [5d72a91]
  - @platforma-open/milaboratories.clonotype-space.workflow@3.0.1
  - @platforma-open/milaboratories.clonotype-space.ui@3.0.2

## 3.0.2

### Patch Changes

- Updated dependencies [d98b141]
  - @platforma-open/milaboratories.clonotype-space.ui@3.0.1

## 3.0.1

### Patch Changes

- 0c78394: Include peptides in documentation

## 3.0.0

### Major Changes

- f4731cf: Support peptides

### Patch Changes

- Updated dependencies [f4731cf]
  - @platforma-open/milaboratories.clonotype-space.workflow@3.0.0
  - @platforma-open/milaboratories.clonotype-space.model@3.0.0
  - @platforma-open/milaboratories.clonotype-space.ui@3.0.0

## 2.5.9

### Patch Changes

- b327178: Fix linker discovery by graph-maker update
- Updated dependencies [b327178]
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.8

## 2.5.8

### Patch Changes

- Updated dependencies [e6befc4]
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.7

## 2.5.7

### Patch Changes

- Updated dependencies [c6308f6]
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.6

## 2.5.6

### Patch Changes

- Updated dependencies [4f38fcc]
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.5

## 2.5.5

### Patch Changes

- Updated dependencies [2e1029c]
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.4

## 2.5.4

### Patch Changes

- Updated dependencies [080069a]
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.3

## 2.5.3

### Patch Changes

- Updated dependencies [51f21e6]
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.2

## 2.5.2

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.14.1

## 2.5.1

### Patch Changes

- 5e36132: update dependencies
- Updated dependencies [5e36132]
  - @platforma-open/milaboratories.clonotype-space.model@2.7.1
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.1

## 2.5.0

### Minor Changes

- b014947: Improved automatic detection and graceful handling of GPU availability

### Patch Changes

- Updated dependencies [b014947]
  - @platforma-open/milaboratories.clonotype-space.model@2.7.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.8.0
  - @platforma-open/milaboratories.clonotype-space.workflow@2.14.0

## 2.4.0

### Minor Changes

- 8b5acc7: fix scfv input, dependencies updates

### Patch Changes

- Updated dependencies [8b5acc7]
  - @platforma-open/milaboratories.clonotype-space.model@2.6.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.7.0
  - @platforma-open/milaboratories.clonotype-space.workflow@2.13.0

## 2.3.4

### Patch Changes

- Updated dependencies [fce50e5]
  - @platforma-open/milaboratories.clonotype-space.ui@2.6.4

## 2.3.3

### Patch Changes

- Updated dependencies [6bce79f]
  - @platforma-open/milaboratories.clonotype-space.model@2.5.1
  - @platforma-open/milaboratories.clonotype-space.ui@2.6.3

## 2.3.2

### Patch Changes

- Updated dependencies [82952a4]
  - @platforma-open/milaboratories.clonotype-space.ui@2.6.2

## 2.3.1

### Patch Changes

- Updated dependencies [b7466b8]
  - @platforma-open/milaboratories.clonotype-space.ui@2.6.1

## 2.3.0

### Minor Changes

- 45fcc5c: Allow to select feature for calculations

### Patch Changes

- Updated dependencies [45fcc5c]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.12.0
  - @platforma-open/milaboratories.clonotype-space.model@2.5.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.6.0

## 2.2.30

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.11.10

## 2.2.29

### Patch Changes

- Updated dependencies [86cdd7d]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.11.9

## 2.2.28

### Patch Changes

- 38d8642: Block metadata update
- Updated dependencies [38d8642]
  - @platforma-open/milaboratories.clonotype-space.model@2.4.4
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.10
  - @platforma-open/milaboratories.clonotype-space.workflow@2.11.8

## 2.2.27

### Patch Changes

- Updated dependencies [0ccc5ba]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.11.7

## 2.2.26

### Patch Changes

- Updated dependencies [9252b6a]
- Updated dependencies [1d396b1]
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.9

## 2.2.25

### Patch Changes

- Updated dependencies [a680286]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.11.6

## 2.2.24

### Patch Changes

- 82bb4ee: Update SDK

## 2.2.23

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.11.5

## 2.2.22

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.11.4

## 2.2.21

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.11.3

## 2.2.20

### Patch Changes

- Updated dependencies [ad99091]
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.8

## 2.2.19

### Patch Changes

- Updated dependencies [f3fc1fb]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.11.2

## 2.2.18

### Patch Changes

- b5f75cb: technical release
- d7482f5: technical release
- afa2968: technical release
- b467126: technical release
- Updated dependencies [b5f75cb]
- Updated dependencies [d7482f5]
- Updated dependencies [afa2968]
- Updated dependencies [b467126]
  - @platforma-open/milaboratories.clonotype-space.model@2.4.3
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.7
  - @platforma-open/milaboratories.clonotype-space.workflow@2.11.1

## 2.2.17

### Patch Changes

- Updated dependencies [907241c]
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.6

## 2.2.16

### Patch Changes

- Updated dependencies [21d9baa]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.11.0

## 2.2.15

### Patch Changes

- 8fca895: technical release
- Updated dependencies [8fca895]
  - @platforma-open/milaboratories.clonotype-space.model@2.4.2
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.5
  - @platforma-open/milaboratories.clonotype-space.workflow@2.10.4

## 2.2.14

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.10.3

## 2.2.13

### Patch Changes

- Updated dependencies [f5a9f34]
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.4

## 2.2.12

### Patch Changes

- Updated dependencies [27066d8]
  - @platforma-open/milaboratories.clonotype-space.model@2.4.1
  - @platforma-open/milaboratories.clonotype-space.workflow@2.10.2
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.3

## 2.2.11

### Patch Changes

- Updated dependencies [36adaab]
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.2

## 2.2.10

### Patch Changes

- Updated dependencies [ac1d325]
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.1

## 2.2.9

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.10.1

## 2.2.8

### Patch Changes

- Updated dependencies [1751f19]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.10.0

## 2.2.7

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.9.1

## 2.2.6

### Patch Changes

- Updated dependencies [efcaca3]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.9.0
  - @platforma-open/milaboratories.clonotype-space.model@2.4.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.5.0

## 2.2.5

### Patch Changes

- Updated dependencies [520fdef]
  - @platforma-open/milaboratories.clonotype-space.ui@2.4.5

## 2.2.4

### Patch Changes

- Updated dependencies [7379529]
- Updated dependencies [ed663b8]
  - @platforma-open/milaboratories.clonotype-space.ui@2.4.4

## 2.2.3

### Patch Changes

- Updated dependencies [336dd7c]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.8.0

## 2.2.2

### Patch Changes

- a1b70cf: Update SDK

## 2.2.1

### Patch Changes

- 87a8fa8: Bump version
- Updated dependencies [f88ef84]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.7.2
  - @platforma-open/milaboratories.clonotype-space.model@2.3.1
  - @platforma-open/milaboratories.clonotype-space.ui@2.4.3

## 2.2.0

### Minor Changes

- f0d07ac: allow prepare venv on Windows

## 2.1.7

### Patch Changes

- Updated dependencies [6fcbff7]
  - @platforma-open/milaboratories.clonotype-space.ui@2.4.2

## 2.1.6

### Patch Changes

- c4efe91: SDK update

## 2.1.5

### Patch Changes

- Updated dependencies [180728a]
  - @platforma-open/milaboratories.clonotype-space.ui@2.4.1

## 2.1.4

### Patch Changes

- @platforma-open/milaboratories.clonotype-space.workflow@2.7.1

## 2.1.3

### Patch Changes

- Updated dependencies [a89c77d]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.7.0

## 2.1.2

### Patch Changes

- Updated dependencies [05d2f21]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.6.0

## 2.1.1

### Patch Changes

- Updated dependencies [5142159]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.5.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.4.0

## 2.1.0

### Minor Changes

- 009c475: Deal with empty inputs and implement batch system

### Patch Changes

- Updated dependencies [009c475]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.4.0
  - @platforma-open/milaboratories.clonotype-space.model@2.3.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.3.0

## 2.0.11

### Patch Changes

- Updated dependencies [e36baf0]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.3.5

## 2.0.10

### Patch Changes

- 280e8c4: [sdk/msa] seqlogo overlaps
- Updated dependencies [280e8c4]
  - @platforma-open/milaboratories.clonotype-space.model@2.2.4
  - @platforma-open/milaboratories.clonotype-space.ui@2.2.4
  - @platforma-open/milaboratories.clonotype-space.workflow@2.3.4

## 2.0.9

### Patch Changes

- 888dd27: SDK Upgrade
- Updated dependencies [888dd27]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.3.3
  - @platforma-open/milaboratories.clonotype-space.ui@2.2.3
  - @platforma-open/milaboratories.clonotype-space.model@2.2.3

## 2.0.8

### Patch Changes

- c25ef01: Update SDK
- Updated dependencies [c25ef01]
  - @platforma-open/milaboratories.clonotype-space.model@2.2.2
  - @platforma-open/milaboratories.clonotype-space.ui@2.2.2
  - @platforma-open/milaboratories.clonotype-space.workflow@2.3.2

## 2.0.7

### Patch Changes

- 095aa9a: SDK update
- Updated dependencies [095aa9a]
  - @platforma-open/milaboratories.clonotype-space.model@2.2.1
  - @platforma-open/milaboratories.clonotype-space.ui@2.2.1
  - @platforma-open/milaboratories.clonotype-space.workflow@2.3.1

## 2.0.6

### Patch Changes

- Updated dependencies [f01f4e2]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.3.0

## 2.0.5

### Patch Changes

- Updated dependencies [934ea49]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.2.0
  - @platforma-open/milaboratories.clonotype-space.model@2.2.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.2.0

## 2.0.4

### Patch Changes

- Updated dependencies [2ef021c]
  - @platforma-open/milaboratories.clonotype-space.model@2.1.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.1.0
  - @platforma-open/milaboratories.clonotype-space.workflow@2.1.1

## 2.0.3

### Patch Changes

- Updated dependencies [5761090]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.1.0

## 2.0.2

### Patch Changes

- Updated dependencies [6cc86da]
  - @platforma-open/milaboratories.clonotype-space.ui@2.0.2

## 2.0.1

### Patch Changes

- Updated dependencies [eaf8a85]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.0.1
  - @platforma-open/milaboratories.clonotype-space.model@2.0.1
  - @platforma-open/milaboratories.clonotype-space.ui@2.0.1

## 2.0.0

### Major Changes

- 9d10697: MVB

### Patch Changes

- Updated dependencies [9d10697]
  - @platforma-open/milaboratories.clonotype-space.workflow@2.0.0
  - @platforma-open/milaboratories.clonotype-space.model@2.0.0
  - @platforma-open/milaboratories.clonotype-space.ui@2.0.0
