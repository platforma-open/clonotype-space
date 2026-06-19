---
"@platforma-open/milaboratories.clonotype-space": minor
"@platforma-open/milaboratories.clonotype-space.workflow": minor
"@platforma-open/milaboratories.clonotype-space.model": minor
"@platforma-open/milaboratories.clonotype-space.ui": minor
"@platforma-open/milaboratories.clonotype-space.umap": minor
---

Request GPU explicitly via `exec.builder().gpuMemory(...)` instead of relying on the UMAP software's runtime CuPy/cuML probe. The block now ships a `gpu` tag and exposes a single "Require run on GPU" checkbox (default on). The workflow auto-sizes the GPU memory request from the input row count (T4-class for ≤1M unique clonotypes, A10G/L4 for ≤2.5M, A6000/L40 for ≤6M) using a new `row-count` sub-template that emits a one-row aggregation file before the UMAP exec is defined. The `.gpuMemory()` call is gated on `exec.hasGpu && requireGpu`, so the block still runs on backends without GPU support and falls back to the existing CPU code path. Unchecking "Require run on GPU" forces the CPU path on any backend.

The UMAP software now prints a banner-style `GPU STATUS:` log line at startup, definitive `>>> SVD/UMAP EXECUTED ON GPU|CPU` markers at the exact return site of each compute call, and a closing `COMPUTATION SUMMARY:` line — so users have a ground-truth record of which backend ran each stage, not just intent-of-use logs.
