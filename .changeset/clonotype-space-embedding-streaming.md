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
batches, so its peak is bounded by the fit sample, independent of the clonotype count. The GPU path
streams through cuML IncrementalPCA (partial_fit on ~100k-clonotype batches) so the raw N×D matrix
never lives in VRAM; its footprint scales with the reduced N×k array and the fit-on-all UMAP, not N×D.

The workflow passes the embedding length as `--dims` (from the `pl7.app/embedding/length` annotation),
and the embedding RAM request is lowered to match the streamed footprint. The pre-PCA global vector
dedup is dropped; the GPU path still collapses exact duplicates, but after PCA on the reduced vectors
(the CPU path does not dedup). The fit sample is drawn over all clonotypes. These shift coordinates slightly versus prior
builds (a memory fix — output already varies across backends and machines). PCA/UMAP parameters and
sample sizes are unchanged. Sequence-feature mode is untouched.
