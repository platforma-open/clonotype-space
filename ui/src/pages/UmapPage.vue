<script setup lang="ts">
import type {
  PColumnIdAndSpec,
  PlRef,
  PlSelectionModel,
  SUniversalPColumnId,
} from '@platforma-sdk/model';
import { getRawPlatformaInstance } from '@platforma-sdk/model';
import { getDefaultBlockLabel } from '@platforma-open/milaboratories.clonotype-space.model';

import { PlMultiSequenceAlignment } from '@milaboratories/multi-sequence-alignment';
import strings from '@milaboratories/strings';
import {
  listToOptions,
  PlAccordionSection,
  PlAlert,
  PlBlockPage,
  PlBtnGhost,
  PlBtnGroup,
  PlDropdown,
  PlDropdownMulti,
  PlDropdownRef,
  PlLogView,
  PlMaskIcon24,
  PlNumberField,
  PlSlideModal,
  PlTextField,
} from '@platforma-sdk/ui-vue';

import type { PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import { asyncComputed } from '@vueuse/core';
import { computed, ref, watch } from 'vue';

import { useApp } from '../app';
import { isSequenceColumn } from '../util';

const app = useApp();

const sequenceType = listToOptions(['aminoacid', 'nucleotide']);

// Filter sequence options by selected sequence type
const filteredSequenceOptions = computed(() => {
  const allOptions = app.model.outputs.sequenceOptions;
  if (!allOptions) return undefined;
  const selectedType = app.model.data.sequenceType;
  return allOptions.filter((option) => option.alphabet === selectedType);
});

const defaultLabel = computed(() =>
  getDefaultBlockLabel({
    sequenceLabels: app.model.data.sequenceLabels,
    umap_neighbors: app.model.data.umap_neighbors,
    umap_min_dist: app.model.data.umap_min_dist,
  }),
);

// Snapshot labels for the chosen sequenceRefs from current options. V3 keeps
// derived facts (labels) in `data` only when written on user gesture, so the
// args lambda can compose `defaultBlockLabel` purely from `data`.
function labelsForRefs(refs: SUniversalPColumnId[]): string[] {
  const options = filteredSequenceOptions.value;
  if (!options || options.length === 0) return [];
  const lookup = new Map(options.map((o) => [o.value, o.label]));
  return refs.map((r) => lookup.get(r) ?? '').filter(Boolean).sort();
}

const inputAnchorModel = computed({
  get: () => app.model.data.inputAnchor,
  set: (ref: PlRef | undefined) => {
    app.model.data.inputAnchor = ref;
    // Sequence options derive from inputAnchor; the prior selection no longer
    // applies. Clear and let the auto-select watcher below re-fill from the
    // new dataset's defaults.
    app.model.data.sequencesRef = [];
    app.model.data.sequenceLabels = [];
  },
});

const sequencesRefModel = computed({
  get: () => app.model.data.sequencesRef,
  set: (refs: SUniversalPColumnId[]) => {
    app.model.data.sequencesRef = refs;
    app.model.data.sequenceLabels = labelsForRefs(refs);
  },
});

function setSequencesRef(refs: SUniversalPColumnId[]) {
  sequencesRefModel.value = refs;
}

const defaultOptions = computed((): PredefinedGraphOption<'scatterplot-umap'>[] | null => {
  const umapPcols = app.model.outputs.umapPcols;
  if (!umapPcols) return null;

  function getIndex(name: string, pcols: PColumnIdAndSpec[]): number {
    return pcols.findIndex((p) => p.spec.name === name);
  }

  // @TODO: drop the legacy fallback once 3.0.0 has propagated.
  let umap1 = getIndex('pl7.app/umap1', umapPcols);
  let umap2 = getIndex('pl7.app/umap2', umapPcols);
  if (umap1 === -1) {
    umap1 = getIndex('pl7.app/vdj/umap1', umapPcols);
    umap2 = getIndex('pl7.app/vdj/umap2', umapPcols);
  }

  if (umap1 === -1 || umap2 === -1) return null;
  return [
    { inputName: 'x', selectedSource: umapPcols[umap1].spec },
    { inputName: 'y', selectedSource: umapPcols[umap2].spec },
  ];
});

// Check if the UMAP file is empty
const isEmpty = asyncComputed(async () => {
  if (app.model.outputs.umapDim1Table === undefined) return undefined;
  return (await getRawPlatformaInstance().pFrameDriver.getShape(app.model.outputs.umapDim1Table)).rows === 0;
});

const selection = ref<PlSelectionModel>({
  axesSpec: [],
  selectedKeys: [],
});

const multipleSequenceAlignmentOpen = ref(false);
const umapLogOpen = ref(false);

// Clearing the selection on sequenceType change keeps the args lambda valid
// (mixed-alphabet refs would not match any option). The auto-select watcher
// below then re-fills with defaults of the new alphabet.
watch(
  () => app.model.data.sequenceType,
  () => {
    app.model.data.sequencesRef = [];
    app.model.data.sequenceLabels = [];
  },
);

// Auto-select default sequences whenever the current selection is empty or
// contains refs that don't match the current filtered options. Watcher-driven
// writes from outputs back into `data` normally risk a multi-client race
// (two desktop instances open on the same project each fire the watcher
// independently and interleave writes); here the computed defaults are a
// deterministic function of `filteredSequenceOptions` (main sequences first,
// stable order), so both instances would write the same value — racing
// writes are idempotent.
watch(
  () => [app.model.data.inputAnchor, app.model.outputs.sequenceOptions, filteredSequenceOptions.value] as const,
  ([anchor, allOptions, filteredOptions]) => {
    if (!anchor || !allOptions || !filteredOptions || filteredOptions.length === 0) return;

    const validValues = new Set(filteredOptions.map((o) => o.value));
    const currentSelection = app.model.data.sequencesRef;
    const hasInvalidValues = currentSelection.some((v) => !validValues.has(v));

    if (!hasInvalidValues && currentSelection.length > 0) {
      // Selection is valid as-is — keep it. But `sequenceLabels` may be empty
      // (legacy upgrade has no source for them) or out of sync; reseed without
      // touching the selection so `defaultBlockLabel` recovers its sequence-name
      // fragment on load instead of leaving the block stale. Deterministic over
      // options, so the same multi-client idempotency argument holds.
      if (app.model.data.sequenceLabels.length !== currentSelection.length) {
        app.model.data.sequenceLabels = labelsForRefs(currentSelection);
      }
      return;
    }

    const mainSequences = filteredOptions.filter((o) => o.isMain);
    const defaults = mainSequences.length > 0 ? mainSequences : [filteredOptions[0]];
    setSequencesRef(defaults.map((o) => o.value));
  },
  { immediate: true },
);

// Auto-close settings panel when the block transitions to running.
watch(
  () => app.model.outputs.isRunning,
  (isRunning, wasRunning) => {
    if (isRunning && !wasRunning) {
      app.model.data.graphStateUMAP.currentTab = null;
    }
  },
);
</script>

<template>
  <PlBlockPage no-body-gutters>
    <GraphMaker
      v-model="app.model.data.graphStateUMAP"
      v-model:selection="selection"
      chartType="scatterplot-umap"
      :p-frame="app.model.outputs.umapPf"
      :default-options="defaultOptions"
      :status-text="{ noPframe: { title: strings.callToActions.configureSettingsAndRun } }"
    >
      <template #titleLineSlot>
        <PlBtnGhost
          icon="dna"
          @click.stop="() => (multipleSequenceAlignmentOpen = true)"
        >
          {{ strings.titles.multipleSequenceAlignment }}
        </PlBtnGhost>
        <PlBtnGhost @click.stop="() => (umapLogOpen = true)">
          {{ strings.titles.logs }}
          <template #append>
            <PlMaskIcon24 name="file-logs" />
          </template>
        </PlBtnGhost>
      </template>
      <template #settingsSlot>
        <PlDropdownRef
          v-model="inputAnchorModel"
          :options="app.model.outputs.inputOptions"
          label="Select dataset"
          required
          :style="{ width: '320px' }"
        />

        <PlTextField
          v-model="app.model.data.customBlockLabel"
          label="Block title"
          :clearable="true"
          :placeholder="defaultLabel"
          :style="{ width: '320px' }"
        />

        <PlAccordionSection label="UMAP Parameters" :style="{ width: '320px' }">
          <PlBtnGroup
            v-model="app.model.data.sequenceType"
            label="Sequence type"
            :options="sequenceType"
            :compact="true"
          />
          <PlDropdownMulti
            v-model="sequencesRefModel"
            :options="filteredSequenceOptions"
            label="Select sequence column/s for UMAP"
            required
          />

          <div :style="{ display: 'flex', gap: '8px', width: '320px' }">
            <PlNumberField
              v-model="app.model.data.umap_neighbors"
              label="Neighbors"
              placeholder="15"
              :min="2"
              :max="500"
              :step="5"
              required
              :validate="(value) => value === undefined ? 'Neighbors is required' : (value < 2 ? 'UMAP requires at least 2 neighbors' : undefined)"
              :style="{ flex: 1 }"
            >
              <template #tooltip>
                <div>
                  <strong>Number of Neighbors for UMAP</strong><br>
                  Controls the balance between local and global structure in UMAP visualization.<br><br>
                  <strong>Default:</strong> 15 neighbors<br><br>
                  <strong>Recommended ranges:</strong><br>
                  • 10-30: Optimal for most datasets<br>
                  • 5-10: Emphasizes local structure (more clusters)<br>
                  • 30+: Emphasizes global structure (fewer clusters)<br><br>
                </div>
              </template>
            </PlNumberField>
            <PlNumberField
              v-model="app.model.data.umap_min_dist"
              label="Minimum Distance"
              placeholder="0.5"
              :min="0"
              :max="1"
              :step="0.1"
              required
              :validate="(value) => value === undefined ? 'Minimum Distance is required' : undefined"
              :style="{ flex: 1 }"
            >
              <template #tooltip>
                <div>
                  <strong>Minimum Distance for UMAP</strong><br>
                  Controls how tightly UMAP packs points together. Lower values create denser clusters, while higher values preserve broader structure.<br><br>
                  <strong>Default:</strong> 0.5<br><br>
                  <strong>Recommended ranges:</strong><br>
                  • 0.0 - 0.2: For creating tight clusters.<br>
                  • 0.2 - 0.5: A good balance for most datasets.<br>
                  • 0.5 - 1.0: For a more global view of the data.<br><br>
                </div>
              </template>
            </PlNumberField>
          </div>
        </PlAccordionSection>

        <PlAccordionSection label="Performance Settings" :style="{ width: '320px' }">
          <div :style="{ display: 'flex', gap: '8px', width: '320px' }">
            <PlNumberField
              v-model="app.model.data.mem"
              label="Memory (GB)"
              placeholder="64"
              :min="8"
              :max="1024"
              :step="1"
              required
              :validate="(value) => value === undefined ? 'Memory is required' : undefined"
              :style="{ flex: 1 }"
            >
              <template #tooltip>
                <div>
                  <strong>Memory Allocation for UMAP Calculation</strong><br>
                  Set the amount of memory (in GB) for the UMAP calculation. The right amount depends on the number of sequences in your dataset.<br><br>
                  <strong>Default:</strong> 64 GB<br><br>
                  <strong>Recommended Memory:</strong><br>
                  <strong>Small</strong> (&lt; 10k sequences): <strong>4-8 GB</strong><br>
                  <strong>Medium</strong> (10k - 100k sequences): <strong>8-32 GB</strong><br>
                  <strong>Large</strong> (&gt; 100k sequences): <strong>32+ GB</strong><br><br>

                  <hr>
                  ⚠️ Insufficient memory can cause the process to fail. If you run into errors, try increasing the allocated memory. <br>

                  <strong>Note:</strong> Larger values for the <code>neighbors</code> parameter can also increase memory usage.
                </div>
              </template>
            </PlNumberField>

            <PlNumberField
              v-model="app.model.data.cpu"
              label="CPU"
              placeholder="8"
              :min="1"
              :max="128"
              :step="1"
              required
              :validate="(value) => value === undefined ? 'CPU is required' : undefined"
              :style="{ flex: 1 }"
            >
              <template #tooltip>
                <div>
                  <strong>CPU Cores for UMAP Calculation</strong><br>
                  Number of CPU cores to allocate for the UMAP calculation. More cores can speed up computation, especially for larger datasets.<br><br>
                  <strong>Default:</strong> 8 cores<br><br>
                  <strong>Recommended:</strong><br>
                  • Small datasets: 2-4 cores<br>
                  • Medium datasets: 4-8 cores<br>
                  • Large datasets: 8+ cores<br>
                </div>
              </template>
            </PlNumberField>
          </div>
          <PlDropdown
            v-model="app.model.data.gpuMemory"
            label="GPU memory (VRAM)"
            :options="[
              { value: '', label: 'No GPU (CPU only)' },
              { value: '1GiB', label: 'Any GPU (1 GiB minimum)' },
              { value: '16GiB', label: '16 GiB (T4)' },
              { value: '24GiB', label: '24 GiB (A10G/L4)' },
              { value: '36GiB', label: '36 GiB' },
              { value: '48GiB', label: '48 GiB (A6000/L40)' },
              { value: '60GiB', label: '60 GiB' },
            ]"
          >
            <template #tooltip>
              <div>
                <strong>GPU Memory Request</strong><br>
                Requests a GPU node with at least this much VRAM. The UMAP software uses RAPIDS cuML on GPU when available and falls back to scikit-learn on CPU.<br><br>
                <strong>Default:</strong> 16 GiB (covers T4-class GPUs)<br><br>
                Pick <strong>No GPU</strong> to force the CPU path. The request is automatically dropped on backends without GPU support — the block then runs on CPU regardless of the selection.
              </div>
            </template>
          </PlDropdown>
        </PlAccordionSection>
        <PlAlert v-if="isEmpty === true" type="warn" :style="{ width: '320px' }">
          <template #title>Empty dataset selection</template>
          The input dataset you have selected is empty or has too few sequences.
          Please choose a different dataset.
        </PlAlert>
      </template>
    </GraphMaker>
    <PlSlideModal
      v-model="multipleSequenceAlignmentOpen"
      width="100%"
      :close-on-outside-click="false"
    >
      <template #title>{{ strings.titles.multipleSequenceAlignment }}</template>
      <PlMultiSequenceAlignment
        v-model="app.model.data.alignmentModel"
        :sequence-column-predicate="isSequenceColumn"
        :p-frame="app.model.outputs.msaPf"
        :selection="selection"
      />
    </PlSlideModal>
    <PlSlideModal v-model="umapLogOpen" width="80%">
      <template #title>UMAP Log</template>
      <PlLogView :log-handle="app.model.outputs.umapOutput"/>
    </PlSlideModal>
  </PlBlockPage>
</template>
