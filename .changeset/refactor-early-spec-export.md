---
"@platforma-open/milaboratories.clonotype-space.workflow": minor
"@platforma-open/milaboratories.clonotype-space": minor
---

Refactor workflow for early spec export.

The workflow previously used `wf.prepare()`, which makes the body wait for upstream PColumn data to materialize before exports are defined. Downstream blocks (e.g. anything reading the UMAP PFrame from the result pool) could not discover this block's outputs until UMAP computation finished — even though the output specs are deterministic from the input specs alone.

Split the workflow into:
- `main.tpl.tengo` — outer body with no `wf.prepare()`. Builds the PColumn bundle and delegates to the inner template.
- `process.tpl.tengo` — inner ephemeral template awaiting `PColumnBundle` (specs only, not data). Builds the UMAP input TSV, runs the UMAP exec, imports results, and assembles the output PFrame. Specs are published to the result pool immediately; data references resolve when computation completes.

No changes to inputs, outputs, exports, or computation. Downstream blocks can now configure their inputs while UMAP is still running.
