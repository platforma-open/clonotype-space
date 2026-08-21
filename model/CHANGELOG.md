# @platforma-open/milaboratories.clonotype-space.model

## 3.6.1

### Patch Changes

- 4cf69da: Recognise bare antibody sets on the variantKey axis

  Three producers key on `pl7.app/variantKey` and only the run-id in the axis domain separates
  them. import-vdj-data's bare antibody sets stamp `pl7.app/vdj/clonotypingRunId`; they were
  classified peptide, whose sequence matcher asks for `pl7.app/sequence` with feature "peptide"
  and so matched nothing — the sequence dropdown came up empty with nothing to say why. They now
  use the bulk matcher, which finds their `pl7.app/vdj/sequence` columns.

  The workflow made the same assumption one step later and would have failed the run with
  "Peptide input is missing required pl7.app/sequenceLength column", a column only
  peptide-extraction emits. The short-peptide mmseqs tuning is now keyed to that producer
  specifically.

  Peptide and amplicon inputs are unaffected.

## 3.6.0

### Minor Changes

- a2b8499: Add the mandatory block kind, and upgrade the SDK

  The block now declares a `kind/` package carrying its identity and its
  init-params contract — the fields a project template supplies to seed a new
  instance: the input anchor and embedding refs, the UMAP recipe (input mode,
  sequence columns, sequence type, neighbours, min distance), the label snapshots
  that travel with those selections, the per-process resource limits, and the
  custom block label. The model consumes them in `init` and projects the same set
  back out via `templateParams`, so export and apply are inverses.

  The UMAP graph state and the multi-sequence-alignment model stay out of the
  contract: they record how one user was looking at one result, not the recipe a
  template exists to reproduce.

  `SequenceType` and `InputMode` move to the kind and are re-exported from the
  model, so the contract's own shapes are declared once, by the package the model
  depends on.

### Patch Changes

- Updated dependencies [a2b8499]
  - @platforma-open/milaboratories.clonotype-space.kind@1.1.0

## 3.5.1

### Patch Changes

- 1002cad: Disable GPU requirement by default. New blocks now run on CPU unless "Require run on GPU" is checked in Performance Settings.

## 3.5.0

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

## 3.4.0

### Minor Changes

- 490ab69: Resources request with formulas + minor UI validations

## 3.3.4

### Patch Changes

- 4be629b: Publish new version

## 3.3.3

### Patch Changes

- b0db708: Support `synthetic-repertoire-profiler` variant datasets:

  - The variantKey-axis sequence matcher now distinguishes the profiler (axis domain `pl7.app/repertoire/extractionRunId`) from peptide-extraction (`pl7.app/peptide/extractionRunId`). For profiler input it matches `pl7.app/sequence` feature-agnostically, so the whole-variant sequence (`pl7.app/feature: "amplicon-sequence"`) and each region subsequence appear as selectable sequence columns.
  - The peptide-mode min-length computation no longer runs for profiler datasets (they don't emit a `pl7.app/sequenceLength` column), so the workflow no longer fails with a missing-column error. The peptide safety check is preserved for genuine peptide-extraction input.
  - The `modality` output now reports a distinct `"amplicon"` value for profiler input instead of mislabeling it as `"peptide"`.

## 3.3.2

### Patch Changes

- feac133: Default missing `requireGpu` in stored block data to `true`. Projects saved before the field was added to `V20260518` came back with `requireGpu === undefined`, which the args lambda rejected and disabled the Run button. The lambda now coerces `undefined → true` (matching the defaults used by `init()` and `upgradeLegacy`) instead of throwing.

## 3.3.1

### Patch Changes

- ccc4b64: Migrate block onto the structurer and refresh the SDK to latest (model/ui-vue/test 1.79.15, workflow-tengo 6.6.3, tengo-builder 4.0.9, block-tools 2.11.1). Tooling now fully managed by `block-tools structure`; removed retired toolchain deps (vite, eslint-config) and dead boilerplate workflow tests.

## 3.3.0

### Minor Changes

- fcd3aa3: Request GPU explicitly via `exec.builder().gpuMemory(...)` instead of relying on the UMAP software's runtime CuPy/cuML probe. The block now ships a `gpu` tag, a "Require run on GPU" checkbox (default on), and a "GPU memory (GB)" number field that is left empty by default. When the field is empty and GPU is required, the workflow falls back to a `16 GiB` request — covers T4-class hardware and the dataset sizes most clonotype repertoires fall into (<1M unique sequences after dedup). `.gpuMemory()` is gated on `exec.hasGpu && requireGpu`, so the block still runs on backends without GPU support and falls back to the existing CPU code path. Unchecking the box forces the CPU path on any backend.

  The UMAP software now prints a banner-style `GPU STATUS:` log line at startup, definitive `>>> SVD/UMAP EXECUTED ON GPU|CPU` markers at the exact return site of each compute call, and a closing `COMPUTATION SUMMARY:` line — so users have a ground-truth record of which backend ran each stage.

## 3.2.1

### Patch Changes

- 3e5fd18: SDK update

## 3.2.0

### Minor Changes

- 7991214: Adapt block to embeddings

## 3.1.0

### Minor Changes

- 96c216b: Migrate block to BlockModelV3. Unified `BlockData` (UI-shaped persistence), `.args` lambda derives the workflow-visible shape and validates by throw. Persisted V1 state preserved via `DataModelBuilder.upgradeLegacy`. UI bindings move to `app.model.data`; `defineApp` → `defineAppV3`.

  `defaultBlockLabel` is no longer stored: the UI snapshots `sequenceLabels` into `data` on sequence-selection gesture, and the args lambda assembles `defaultBlockLabel` from `data`. Existing projects keep `customBlockLabel`; the sequence-name fragment of the default label is reseeded on the next interaction with the sequences dropdown.

## 3.0.0

### Major Changes

- f4731cf: Support peptides

## 2.7.1

### Patch Changes

- 5e36132: update dependencies

## 2.7.0

### Minor Changes

- b014947: Improved automatic detection and graceful handling of GPU availability

## 2.6.0

### Minor Changes

- 8b5acc7: fix scfv input, dependencies updates

## 2.5.1

### Patch Changes

- 6bce79f: Improve block subtitle generation

## 2.5.0

### Minor Changes

- 45fcc5c: Allow to select feature for calculations

## 2.4.4

### Patch Changes

- 38d8642: Block metadata update

## 2.4.3

### Patch Changes

- b5f75cb: technical release
- d7482f5: technical release
- afa2968: technical release
- b467126: technical release

## 2.4.2

### Patch Changes

- 8fca895: technical release

## 2.4.1

### Patch Changes

- 27066d8: Full SDK update

## 2.4.0

### Minor Changes

- efcaca3: Add UMAP log to UI and fix UMAP error on insufficient clonotype sequences

## 2.3.1

### Patch Changes

- f88ef84: Ability to set CPU and MEM requests

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
