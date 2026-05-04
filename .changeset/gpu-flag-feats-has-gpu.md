---
"@platforma-open/milaboratories.clonotype-space": minor
"@platforma-open/milaboratories.clonotype-space.workflow": minor
"@platforma-open/milaboratories.clonotype-space.model": minor
"@platforma-open/milaboratories.clonotype-space.ui": minor
---

Add GPU acceleration for UMAP via the cuml backend when the platforma backend reports GPU availability.

- New `GPU memory (GB)` field in Performance Settings (default 16, range 1–192). The value is requested only when `feats.hasGpu` is true; on CPU-only backends it is ignored and the calculation falls back to the sklearn UMAP backend, identical to the previous behavior.
- Workflow gates `.gpuMemory()` and `--umap-backend cuml` on `feats.hasGpu` so the same block runs both on GPU clusters (e.g. AWS EKS with the `kueue.pools.gpu.enabled` pool) and on CPU-only/local installations.
- Bumps `@platforma-sdk/workflow-tengo` to 5.20.0 (which ships `feats.hasGpu`).
