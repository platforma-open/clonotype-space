---
"@platforma-open/milaboratories.clonotype-space.model": minor
"@platforma-open/milaboratories.clonotype-space.ui": minor
"@platforma-open/milaboratories.clonotype-space": minor
---

Migrate block to BlockModelV3. Unified `BlockData` (UI-shaped persistence), `.args` lambda derives the workflow-visible shape and validates by throw. Persisted V1 state preserved via `DataModelBuilder.upgradeLegacy`. UI bindings move to `app.model.data`; `defineApp` → `defineAppV3`.

`defaultBlockLabel` is no longer stored: the UI snapshots `sequenceLabels` into `data` on sequence-selection gesture, and the args lambda assembles `defaultBlockLabel` from `data`. Existing projects keep `customBlockLabel`; the sequence-name fragment of the default label is reseeded on the next interaction with the sequences dropdown.
