---
"@platforma-open/milaboratories.clonotype-space": minor
"@platforma-open/milaboratories.clonotype-space.workflow": minor
"@platforma-open/milaboratories.clonotype-space.model": minor
"@platforma-open/milaboratories.clonotype-space.ui": minor
"@platforma-open/milaboratories.clonotype-space.umap": minor
---

Request GPU explicitly via `exec.builder().gpuMemory(...)` instead of relying on the UMAP software's runtime CuPy/cuML probe. The block now ships a `gpu` tag and exposes a "Require run on GPU" checkbox (default on) plus a `GPU memory (VRAM)` number field (default `16 GiB`, T4-class). The workflow gates `.gpuMemory()` on `exec.hasGpu && requireGpu`, so the block still runs on backends without GPU support and falls back to the existing CPU code path. Unchecking "Require run on GPU" forces the CPU path on any backend.

The UMAP software now prints a banner-style `GPU STATUS:` log line at startup that says whether the GPU pipeline is in use, and why if not — so users can confirm at a glance which path ran without waiting to compare timings.
