---
"@platforma-open/milaboratories.clonotype-space.workflow": patch
---

Run the resource-formula unit test in CI.

`resource-formula.test.tengo` was committed complete but had never executed: the
workflow package had no `test` script, so nothing invoked it. Run for the first
time, 2 of its 4 tests failed — both asserting `spec.onGPU.cpu` against the
`onGPU` block that 63052ff commented out of `resource-formula.lib.tengo`. Tengo
yields `undefined` for `undefined[key]` instead of erroring, so the assertions
compared 4 against `<undefined>`.

The stale GPU assertions are removed and the script added, so the four remaining
tests now guard the CPU formulas against drift. Restore the GPU assertions
alongside the lib block whenever GPU is re-enabled.
