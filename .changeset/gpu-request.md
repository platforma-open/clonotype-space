---
"@platforma-open/milaboratories.clonotype-space": minor
"@platforma-open/milaboratories.clonotype-space.workflow": minor
"@platforma-open/milaboratories.clonotype-space.model": minor
"@platforma-open/milaboratories.clonotype-space.ui": minor
---

Request GPU explicitly via `exec.builder().gpuMemory(...)` instead of relying on the UMAP software's runtime CuPy/cuML probe. The block now ships a `gpu` tag and exposes a `GPU memory (VRAM)` dropdown (default `16GiB`, T4-class). The workflow gates `.gpuMemory()` on `exec.hasGpu`, so the block still runs on backends without GPU support and falls back to the existing CPU code path. Picking "No GPU (CPU only)" in the UI forces the CPU path on any backend.
