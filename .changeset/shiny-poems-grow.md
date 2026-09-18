---
"@platforma-open/milaboratories.clonotype-space.umap": minor
"@platforma-open/milaboratories.clonotype-space": minor
---

Speed up the CPU transform phase, and cap SVD dimensionality on large inputs

The CPU pipeline transforms every sequence through the fitted models in chunks, and that phase
dominates the runtime of a large run. Chunks are now spread across worker processes on
fork-capable systems instead of being transformed one at a time — roughly 6x on 8 cores.
umap-learn rebuilds its transform RNG on every call and keeps no state between calls, so a
chunk's coordinates do not depend on which process handled it: parallel output is identical to
sequential, byte for byte.

Inputs above 1M unique sequences now keep fewer SVD components, because the nearest-neighbour
query that dominates the transform grows with the reduced dimensionality:

    > 1M: 400    > 2M: 300    > 5M: 200    > 10M: 150    > 20M: 100

Previously every input used up to 500. Fewer components retain less variance, so **UMAP
coordinates for inputs above 1M will differ from earlier versions** — an existing project
re-run on this version will not reproduce its previous layout. Inputs at or below 1M keep the
500-component ceiling and are unaffected, as are the GPU pipeline and the embedding path, which
size their own reduction.
