# @platforma-open/milaboratories.clonotype-space.ui

## 3.4.4

### Patch Changes

- Updated dependencies [4cf69da]
  - @platforma-open/milaboratories.clonotype-space.model@3.6.1

## 3.4.3

### Patch Changes

- Updated dependencies [a2b8499]
  - @platforma-open/milaboratories.clonotype-space.model@3.6.0

## 3.4.2

### Patch Changes

- Updated dependencies [1002cad]
  - @platforma-open/milaboratories.clonotype-space.model@3.5.1

## 3.4.1

### Patch Changes

- Updated dependencies [075792e]
  - @platforma-open/milaboratories.clonotype-space.model@3.5.0

## 3.4.0

### Minor Changes

- 490ab69: Resources request with formulas + minor UI validations

### Patch Changes

- Updated dependencies [490ab69]
  - @platforma-open/milaboratories.clonotype-space.model@3.4.0

## 3.3.4

### Patch Changes

- 4be629b: Publish new version
- Updated dependencies [4be629b]
  - @platforma-open/milaboratories.clonotype-space.model@3.3.4

## 3.3.3

### Patch Changes

- b0db708: Support `synthetic-repertoire-profiler` variant datasets:

  - The variantKey-axis sequence matcher now distinguishes the profiler (axis domain `pl7.app/repertoire/extractionRunId`) from peptide-extraction (`pl7.app/peptide/extractionRunId`). For profiler input it matches `pl7.app/sequence` feature-agnostically, so the whole-variant sequence (`pl7.app/feature: "amplicon-sequence"`) and each region subsequence appear as selectable sequence columns.
  - The peptide-mode min-length computation no longer runs for profiler datasets (they don't emit a `pl7.app/sequenceLength` column), so the workflow no longer fails with a missing-column error. The peptide safety check is preserved for genuine peptide-extraction input.
  - The `modality` output now reports a distinct `"amplicon"` value for profiler input instead of mislabeling it as `"peptide"`.

- Updated dependencies [b0db708]
  - @platforma-open/milaboratories.clonotype-space.model@3.3.3

## 3.3.2

### Patch Changes

- Updated dependencies [feac133]
  - @platforma-open/milaboratories.clonotype-space.model@3.3.2

## 3.3.1

### Patch Changes

- ccc4b64: Migrate block onto the structurer and refresh the SDK to latest (model/ui-vue/test 1.79.15, workflow-tengo 6.6.3, tengo-builder 4.0.9, block-tools 2.11.1). Tooling now fully managed by `block-tools structure`; removed retired toolchain deps (vite, eslint-config) and dead boilerplate workflow tests.
- Updated dependencies [ccc4b64]
  - @platforma-open/milaboratories.clonotype-space.model@3.3.1

## 3.3.0

### Minor Changes

- fcd3aa3: Request GPU explicitly via `exec.builder().gpuMemory(...)` instead of relying on the UMAP software's runtime CuPy/cuML probe. The block now ships a `gpu` tag, a "Require run on GPU" checkbox (default on), and a "GPU memory (GB)" number field that is left empty by default. When the field is empty and GPU is required, the workflow falls back to a `16 GiB` request — covers T4-class hardware and the dataset sizes most clonotype repertoires fall into (<1M unique sequences after dedup). `.gpuMemory()` is gated on `exec.hasGpu && requireGpu`, so the block still runs on backends without GPU support and falls back to the existing CPU code path. Unchecking the box forces the CPU path on any backend.

  The UMAP software now prints a banner-style `GPU STATUS:` log line at startup, definitive `>>> SVD/UMAP EXECUTED ON GPU|CPU` markers at the exact return site of each compute call, and a closing `COMPUTATION SUMMARY:` line — so users have a ground-truth record of which backend ran each stage.

### Patch Changes

- Updated dependencies [fcd3aa3]
  - @platforma-open/milaboratories.clonotype-space.model@3.3.0

## 3.2.1

### Patch Changes

- Updated dependencies [3e5fd18]
  - @platforma-open/milaboratories.clonotype-space.model@3.2.1

## 3.2.0

### Minor Changes

- 7991214: Adapt block to embeddings

### Patch Changes

- Updated dependencies [7991214]
  - @platforma-open/milaboratories.clonotype-space.model@3.2.0

## 3.1.0

### Minor Changes

- 96c216b: Migrate block to BlockModelV3. Unified `BlockData` (UI-shaped persistence), `.args` lambda derives the workflow-visible shape and validates by throw. Persisted V1 state preserved via `DataModelBuilder.upgradeLegacy`. UI bindings move to `app.model.data`; `defineApp` → `defineAppV3`.

  `defaultBlockLabel` is no longer stored: the UI snapshots `sequenceLabels` into `data` on sequence-selection gesture, and the args lambda assembles `defaultBlockLabel` from `data`. Existing projects keep `customBlockLabel`; the sequence-name fragment of the default label is reseeded on the next interaction with the sequences dropdown.

### Patch Changes

- Updated dependencies [96c216b]
  - @platforma-open/milaboratories.clonotype-space.model@3.1.0

## 3.0.2

### Patch Changes

- 5d72a91: Fix CID issues

## 3.0.1

### Patch Changes

- d98b141: dependencies update

## 3.0.0

### Major Changes

- f4731cf: Support peptides

### Patch Changes

- Updated dependencies [f4731cf]
  - @platforma-open/milaboratories.clonotype-space.model@3.0.0

## 2.8.8

### Patch Changes

- b327178: Fix linker discovery by graph-maker update

## 2.8.7

### Patch Changes

- e6befc4: Allow to select all sequences in MSA

## 2.8.6

### Patch Changes

- c6308f6: update graph-maker

## 2.8.5

### Patch Changes

- 4f38fcc: update graph-maker

## 2.8.4

### Patch Changes

- 2e1029c: update graph-maker version

## 2.8.3

### Patch Changes

- 080069a: update dependencies

## 2.8.2

### Patch Changes

- 51f21e6: update dependencies

## 2.8.1

### Patch Changes

- 5e36132: update dependencies
- Updated dependencies [5e36132]
  - @platforma-open/milaboratories.clonotype-space.model@2.7.1

## 2.8.0

### Minor Changes

- b014947: Improved automatic detection and graceful handling of GPU availability

### Patch Changes

- Updated dependencies [b014947]
  - @platforma-open/milaboratories.clonotype-space.model@2.7.0

## 2.7.0

### Minor Changes

- 8b5acc7: fix scfv input, dependencies updates

### Patch Changes

- Updated dependencies [8b5acc7]
  - @platforma-open/milaboratories.clonotype-space.model@2.6.0

## 2.6.4

### Patch Changes

- fce50e5: Fix default label derivation

## 2.6.3

### Patch Changes

- 6bce79f: Improve block subtitle generation
- Updated dependencies [6bce79f]
  - @platforma-open/milaboratories.clonotype-space.model@2.5.1

## 2.6.2

### Patch Changes

- 82952a4: update graph-maker version

## 2.6.1

### Patch Changes

- b7466b8: Labels migration

## 2.6.0

### Minor Changes

- 45fcc5c: Allow to select feature for calculations

### Patch Changes

- Updated dependencies [45fcc5c]
  - @platforma-open/milaboratories.clonotype-space.model@2.5.0

## 2.5.10

### Patch Changes

- 38d8642: Block metadata update
- Updated dependencies [38d8642]
  - @platforma-open/milaboratories.clonotype-space.model@2.4.4

## 2.5.9

### Patch Changes

- 9252b6a: update dependencies
- 1d396b1: use moved PlMultiSequenceAlignment

## 2.5.8

### Patch Changes

- ad99091: update graph-maker version

## 2.5.7

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

## 2.5.6

### Patch Changes

- 907241c: update graph-maker version

## 2.5.5

### Patch Changes

- 8fca895: technical release
- Updated dependencies [8fca895]
  - @platforma-open/milaboratories.clonotype-space.model@2.4.2

## 2.5.4

### Patch Changes

- f5a9f34: Update SDK

## 2.5.3

### Patch Changes

- Updated dependencies [27066d8]
  - @platforma-open/milaboratories.clonotype-space.model@2.4.1

## 2.5.2

### Patch Changes

- 36adaab: remove extra padding

## 2.5.1

### Patch Changes

- ac1d325: update graph-maker version

## 2.5.0

### Minor Changes

- efcaca3: Add UMAP log to UI and fix UMAP error on insufficient clonotype sequences

### Patch Changes

- Updated dependencies [efcaca3]
  - @platforma-open/milaboratories.clonotype-space.model@2.4.0

## 2.4.5

### Patch Changes

- 520fdef: update graph-maker version

## 2.4.4

### Patch Changes

- 7379529: MSA updates
- ed663b8: Only allow aa alignment

## 2.4.3

### Patch Changes

- f88ef84: Ability to set CPU and MEM requests
- Updated dependencies [f88ef84]
  - @platforma-open/milaboratories.clonotype-space.model@2.3.1

## 2.4.2

### Patch Changes

- 6fcbff7: update graph-maker version

## 2.4.1

### Patch Changes

- 180728a: update graph-maker version

## 2.4.0

### Minor Changes

- 5142159: Updated to handle non valid aa sequences as input

## 2.3.0

### Minor Changes

- 009c475: Deal with empty inputs and implement batch system

### Patch Changes

- Updated dependencies [009c475]
  - @platforma-open/milaboratories.clonotype-space.model@2.3.0

## 2.2.4

### Patch Changes

- 280e8c4: [sdk/msa] seqlogo overlaps
- Updated dependencies [280e8c4]
  - @platforma-open/milaboratories.clonotype-space.model@2.2.4

## 2.2.3

### Patch Changes

- 888dd27: SDK Upgrade
- Updated dependencies [888dd27]
  - @platforma-open/milaboratories.clonotype-space.model@2.2.3

## 2.2.2

### Patch Changes

- c25ef01: Update SDK
- Updated dependencies [c25ef01]
  - @platforma-open/milaboratories.clonotype-space.model@2.2.2

## 2.2.1

### Patch Changes

- 095aa9a: SDK update
- Updated dependencies [095aa9a]
  - @platforma-open/milaboratories.clonotype-space.model@2.2.1

## 2.2.0

### Minor Changes

- 934ea49: Improved parameter handling and dimension calculation.

### Patch Changes

- Updated dependencies [934ea49]
  - @platforma-open/milaboratories.clonotype-space.model@2.2.0

## 2.1.0

### Minor Changes

- 2ef021c: Update umap script and plot defaults

### Patch Changes

- Updated dependencies [2ef021c]
  - @platforma-open/milaboratories.clonotype-space.model@2.1.0

## 2.0.2

### Patch Changes

- 6cc86da: Fixed Content-Security-Policy issue

## 2.0.1

### Patch Changes

- eaf8a85: Update scripts
- Updated dependencies [eaf8a85]
  - @platforma-open/milaboratories.clonotype-space.model@2.0.1

## 2.0.0

### Major Changes

- 9d10697: MVB

### Patch Changes

- Updated dependencies [9d10697]
  - @platforma-open/milaboratories.clonotype-space.model@2.0.0
