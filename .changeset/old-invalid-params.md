---
"@platforma-open/milaboratories.clonotype-space.model": patch
---

Treat missing `requireGpu` in stored block data as `false`. Projects saved before the field was added to `V20260518` came back with `requireGpu === undefined`, which the args lambda rejected and disabled the Run button. The lambda now coerces `undefined → false` instead of throwing.
