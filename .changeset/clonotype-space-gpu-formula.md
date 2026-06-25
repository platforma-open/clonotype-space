---
"@platforma-open/milaboratories.clonotype-space.workflow": minor
---

UMAP exec: drive the GPU VRAM request from input size via the new `exec.gpuFormula()` (SDK workflow-tengo), and actually apply the RAM/CPU formulas to the builder in formula mode. On a GPU-capable backend the VRAM request now scales with the clonotype count (sequence-features) or matrix size (embedding); each formula keeps a static fallback for backends that cannot evaluate formulas. The sizing logic is extracted into `resource-formula.lib.tengo` and covered by tengo unit tests.
