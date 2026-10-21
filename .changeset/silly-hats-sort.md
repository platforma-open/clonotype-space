---
"@platforma-open/milaboratories.clonotype-space.umap": minor
"@platforma-open/milaboratories.clonotype-space": minor
---

Embedding mode: clonotypes with identical embeddings now always share one point in the UMAP. Identical vectors are collapsed before the UMAP fit and each distinct vector is placed once, its copies reusing its coordinates; before, copies processed in different batches could land far apart. Only byte-identical embeddings are merged. SDK upgrade.
