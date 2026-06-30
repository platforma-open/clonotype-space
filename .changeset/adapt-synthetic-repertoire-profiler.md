---
'@platforma-open/milaboratories.clonotype-space.model': patch
'@platforma-open/milaboratories.clonotype-space.ui': patch
'@platforma-open/milaboratories.clonotype-space.workflow': patch
---

Support `synthetic-repertoire-profiler` variant datasets:

- The variantKey-axis sequence matcher now distinguishes the profiler (axis domain `pl7.app/repertoire/extractionRunId`) from peptide-extraction (`pl7.app/peptide/extractionRunId`). For profiler input it matches `pl7.app/sequence` feature-agnostically, so the whole-variant sequence (`pl7.app/feature: "amplicon-sequence"`) and each region subsequence appear as selectable sequence columns.
- The peptide-mode min-length computation no longer runs for profiler datasets (they don't emit a `pl7.app/sequenceLength` column), so the workflow no longer fails with a missing-column error. The peptide safety check is preserved for genuine peptide-extraction input.
- The `modality` output now reports a distinct `"amplicon"` value for profiler input instead of mislabeling it as `"peptide"`.