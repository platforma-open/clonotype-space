# @platforma-open/milaboratories.clonotype-space.kind

## 1.1.0

### Minor Changes

- a2b8499: Add the mandatory block kind, and upgrade the SDK

  The block now declares a `kind/` package carrying its identity and its
  init-params contract — the fields a project template supplies to seed a new
  instance: the input anchor and embedding refs, the UMAP recipe (input mode,
  sequence columns, sequence type, neighbours, min distance), the label snapshots
  that travel with those selections, the per-process resource limits, and the
  custom block label. The model consumes them in `init` and projects the same set
  back out via `templateParams`, so export and apply are inverses.

  The UMAP graph state and the multi-sequence-alignment model stay out of the
  contract: they record how one user was looking at one result, not the recipe a
  template exists to reproduce.

  `SequenceType` and `InputMode` move to the kind and are re-exported from the
  model, so the contract's own shapes are declared once, by the package the model
  depends on.
