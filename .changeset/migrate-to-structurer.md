---
'@platforma-open/milaboratories.clonotype-space.model': patch
'@platforma-open/milaboratories.clonotype-space.ui': patch
'@platforma-open/milaboratories.clonotype-space.workflow': patch
---

Migrate block onto the structurer and refresh the SDK to latest (model/ui-vue/test 1.79.15, workflow-tengo 6.6.3, tengo-builder 4.0.9, block-tools 2.11.1). Tooling now fully managed by `block-tools structure`; removed retired toolchain deps (vite, eslint-config) and dead boilerplate workflow tests.
