---
"@platforma-open/milaboratories.clonotype-space.model": minor
"@platforma-open/milaboratories.clonotype-space.ui": minor
"@platforma-open/milaboratories.clonotype-space": minor
"@platforma-open/milaboratories.clonotype-space.workflow": patch
---

Multiple Sequence Alignment: the Sequence Columns list now offers only text columns keyed by the input dataset's own clonotype or variant axis. With an Amplicon Profiling input, the known-variant sequence (keyed by the known-variant axis) was offered and selected by default next to the variant sequence, and the alignment failed with "some of axes sets are disjoint"; mutation counts (AA mutations, Mutations to known) were offered as sequences. A saved selection that is no longer offered falls back to the defaults. SDK upgrade.
