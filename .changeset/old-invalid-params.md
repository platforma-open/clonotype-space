---
"@platforma-open/milaboratories.clonotype-space.model": patch
---

Default missing `requireGpu` in stored block data to `true`. Projects saved before the field was added to `V20260518` came back with `requireGpu === undefined`, which the args lambda rejected and disabled the Run button. The lambda now coerces `undefined → true` (matching the defaults used by `init()` and `upgradeLegacy`) instead of throwing.
