---
'@platforma-open/milaboratories.clonotype-space.model': patch
'@platforma-open/milaboratories.clonotype-space.workflow': patch
'@platforma-open/milaboratories.clonotype-space': patch
---

Recognise bare antibody sets on the variantKey axis

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
