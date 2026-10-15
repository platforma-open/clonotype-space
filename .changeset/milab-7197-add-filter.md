---
"@platforma-open/milaboratories.clonotype-space": minor
"@platforma-open/milaboratories.clonotype-space.kind": minor
"@platforma-open/milaboratories.clonotype-space.model": minor
"@platforma-open/milaboratories.clonotype-space.ui": minor
"@platforma-open/milaboratories.clonotype-space.workflow": minor
---

Update the SDK and accept a subset filter (e.g. a Repertoire Labeling label or Lead Selection picks) with the dataset. The dataset list is built with buildDatasetOptions (requireEnrichments: false). Only the subset's clonotypes are laid out, in both sequence and embedding mode, so the UMAP coordinates (and, for peptides, the short-peptide k-mer choice) are computed within the subset and differ from a full-data run. Outputs of a subset run carry pl7.app/inputSubset in their domain and a trace rooted in the subset, so their labels name it. Unfiltered runs are unchanged.
