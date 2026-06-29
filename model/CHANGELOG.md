# @platforma-open/milaboratories.clonotype-space.model

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
