---
"@platforma-open/milaboratories.clonotype-space.umap": patch
"@platforma-open/milaboratories.clonotype-space.workflow": patch
"@platforma-open/milaboratories.clonotype-space": patch
---

Fix CPU SVD path computing the same TruncatedSVD twice when the 95% explained variance target is not reached within the component cap. The already-fit model is now reused via `transform()` instead of being re-fit via `fit_transform()`. Roughly halves runtime of the SVD step on inputs that hit the component cap (e.g. small datasets with high k-mer entropy).
