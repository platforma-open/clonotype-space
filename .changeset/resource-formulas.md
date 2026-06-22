---
"@platforma-open/milaboratories.clonotype-space.workflow": patch
---

MILAB-6480: derive UMAP exec RAM and CPU from input file size via `exec.formula`.

RAM and CPU on the UMAP exec are now computed from the tagged input file
size at exec time (parquet for the embedding path, tsv for the
sequence-features path). The CPU path requests ~4× input size in RAM and
~1 core per 2 GiB of input; the GPU path stays at ~1× RAM and a single
host core for orchestration. The user's UI-set mem/cpu values become the
ceiling and the fallback for backends that can't evaluate resource
formulas (pl < v3.0.4). `--n-jobs` now reads `{system.cpu}` so the binary
matches the actually-allocated cores.
