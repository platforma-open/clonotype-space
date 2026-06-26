---
"@platforma-open/milaboratories.clonotype-space.workflow": minor
---

UMAP exec: drive the GPU VRAM request from input size via the new `exec.gpuFormula()` (SDK workflow-tengo), and actually apply the RAM/CPU formulas to the builder in formula mode. On a GPU-capable backend the VRAM request now scales with the clonotype count (sequence-features) or matrix size (embedding); each formula keeps a static fallback for backends that cannot evaluate formulas. The sizing logic is extracted into `resource-formula.lib.tengo` and covered by tengo unit tests.

The GPU-host CPU/RAM are no longer a fixed lean `1` core / `4 GiB` floor — they are floored at the block's `cpu` / `mem` data fields (host CPU request and host RAM floor in GiB), so a silent cuML→CPU fallback still has a workable allocation instead of taking effectively forever. Those same data-driven values are used as the GPU-path fallback when the backend cannot evaluate formulas.
