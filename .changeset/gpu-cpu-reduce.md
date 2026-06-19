---
"@platforma-open/milaboratories.clonotype-space.workflow": patch
---

Reduce CPU request to 1 when GPU is in use. On a GPU node, the heavy compute (SVD/PCA, UMAP) runs on the GPU and the CPU only orchestrates the run plus the small k-mer counting step — so the workflow now requests just 1 CPU when `requireGpu && exec.hasGpu`, both for the exec resource and the Python `--n-jobs` flag. Keeps GPU pools densely packed without meaningfully hurting wall-clock for typical clonotype counts.
