---
'@platforma-open/milaboratories.clonotype-space.umap': minor
'@platforma-open/milaboratories.clonotype-space.workflow': minor
'@platforma-open/milaboratories.clonotype-space.model': minor
'@platforma-open/milaboratories.clonotype-space': minor
---

Embedding mode: stream the embedding matrix instead of eager-loading it, so large inputs no longer OOM.

The `--encoding embedding` path in `main.py` now streams the long-format parquet in fixed 4M-row batches
rather than materializing the full N×D matrix. The CPU path collects a bounded, seeded fit sample
(~`--max-sequences` vectors), fits PCA + UMAP on it, then re-streams to project every clonotype in
batches; the GPU path streams cuML `IncrementalPCA` over the same batches, then re-streams to transform
every clonotype into the reduced N×k space on device. Neither path holds the raw N×D matrix. CPU peak is
now the fit sample plus the O(N) output coordinates, dominated by the N-independent PCA fit.

The workflow passes the embedding length as `--dims` (from the `pl7.app/embedding/length` annotation),
and the embedding RAM request is lowered to match the streamed footprint. Global exact-duplicate dedup
is dropped and the fit sample is drawn over all clonotypes; both shift coordinates slightly versus prior
builds (a memory fix — output already varies across backends and machines). The GPU path still collapses
duplicate reduced vectors before its UMAP fit, since cuML scatters zero-distance points; the CPU path
relies on umap-learn collapsing them naturally and guards only the pathological case of fewer distinct
vectors than `--umap-neighbors` + 1. PCA/UMAP parameters and sample sizes are unchanged. Sequence-feature
mode is untouched.
