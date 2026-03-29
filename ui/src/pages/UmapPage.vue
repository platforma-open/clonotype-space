<script setup lang="ts">
import type {
  PColumnIdAndSpec,
  PlRef,
} from '@platforma-sdk/model';
import {
  getRawPlatformaInstance,
} from '@platforma-sdk/model';

import { PlAccordionSection, PlAlert, PlBlockPage, PlBtnGhost, PlBtnGroup, PlDropdownMulti, PlDropdownRef, PlLogView, PlMaskIcon24, PlNumberField, PlSlideModal, PlTextField, usePlugin } from '@platforma-sdk/ui-vue';
import { listToOptions } from '@platforma-sdk/ui-vue';
import { PlMultiSequenceAlignment } from '@milaboratories/multi-sequence-alignment';
import strings from '@milaboratories/strings';
import { useApp } from '../app';

import type { PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMakerPlugin } from '@milaboratories/graph-maker';
import { asyncComputed } from '@vueuse/core';

import { computed, ref, watch } from 'vue';
import { isSequenceColumn } from '../util';

const app = useApp();

const sequenceType = listToOptions(['aminoacid', 'nucleotide']);

// Filter sequence options by selected sequence type
const filteredSequenceOptions = computed(() => {
  const allOptions = app.model.outputs.sequenceOptions;
  if (!allOptions) return undefined;

  const selectedType = app.model.data.sequenceType;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return allOptions.filter((option: any) => option.alphabet === selectedType);
});

function setAnchorColumn(ref: PlRef | undefined) {
  app.model.data.inputAnchor = ref;
  // Reset sequence selection when dataset changes
  app.model.data.sequencesRef = [];
}

const defaultOptions = computed((): PredefinedGraphOption<'scatterplot-umap'>[] | null => {
  if (!app.model.outputs.umapPcols)
    return null;

  const umapPcols = app.model.outputs.umapPcols;
  function getIndex(name: string, pcols: PColumnIdAndSpec[]): number {
    return pcols.findIndex((p) => (p.spec.name === name
    ));
  }

  const defaults: PredefinedGraphOption<'scatterplot-umap'>[] = [
    {
      inputName: 'x',
      selectedSource: umapPcols[getIndex('pl7.app/vdj/umap1',
        umapPcols)].spec,
    },
    {
      inputName: 'y',
      selectedSource: umapPcols[getIndex('pl7.app/vdj/umap2',
        umapPcols)].spec,
    },
  ];
  return defaults;
});

// Check if the UMAP file is empty
const isEmpty = asyncComputed(async () => {
  if (app.model.outputs.umapDim1Table === undefined) return undefined;
  return (await getRawPlatformaInstance().pFrameDriver.getShape(app.model.outputs.umapDim1Table)).rows === 0;
});

const multipleSequenceAlignmentOpen = ref(false);
const umapLogOpen = ref(false);

// Clear selected sequences when sequence type changes
watch(
  () => app.model.data.sequenceType,
  () => {
    // Reset selection when sequence type changes
    app.model.data.sequencesRef = [];
  },
);

watch(() => app.plugins.umap.publicData.selection, (v) => {
  console.log('selection', v);
}, { immediate: true, deep: true });

const pluginInner = usePlugin(app.plugins.umap.handle);

watch(() => pluginInner.model.data.selection, (v) => {
  console.log('selection plugin inner', v);
}, { immediate: true, deep: true });
// Validate and auto-select sequences when options change
watch(
  () => [app.model.data.inputAnchor, app.model.outputs.sequenceOptions, filteredSequenceOptions.value] as const,
  ([anchor, allOptions, filteredOptions]) => {
    if (!anchor || !allOptions || !filteredOptions || filteredOptions.length === 0) {
      return;
    }

    // Create a set of valid option values for fast lookup
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const validValues = new Set(filteredOptions.map((option: any) => option.value));

    // Check if current selection contains invalid values (from previous dataset)
    const currentSelection = app.model.data.sequencesRef;
    const hasInvalidValues = currentSelection.some((value: string) => !validValues.has(value));

    // Clear selection if it contains invalid values or if it's empty
    if (hasInvalidValues || currentSelection.length === 0) {
      // Auto-select ALL main sequences (e.g., for single-cell datasets with multiple primary chains)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const mainSequences = filteredOptions.filter((option: any) => option.isMain);
      if (mainSequences.length > 0) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        app.model.data.sequencesRef = mainSequences.map((option: any) => option.value);
      } else {
        // Fallback: if no main sequences found, select the first option
        app.model.data.sequencesRef = [filteredOptions[0].value];
      }
    }
  },
  { immediate: true },
);
</script>

<template>
  <PlBlockPage no-body-gutters>
    <GraphMakerPlugin
      :handle="app.plugins.umap.handle"
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
          v-model="app.model.data.inputAnchor"
          :options="app.model.outputs.inputOptions"
          label="Select dataset"
          required
          :style="{ width: '320px' }"
          @update:model-value="setAnchorColumn"
        />

        <PlTextField
          v-model="app.model.data.customBlockLabel"
          label="Block title"
          :clearable="true"
          :placeholder="app.model.data.defaultBlockLabel"
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
            v-model="app.model.data.sequencesRef"
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
                  Set the amount of memory (in GB) for the UMAP calculation. The right amount depends on the number of clonotypes in your dataset.<br><br>
                  <strong>Default:</strong> 64 GB<br><br>
                  <strong>Recommended Memory:</strong><br>
                  <strong>Small</strong> (&lt; 10k clonotypes): <strong>4-8 GB</strong><br>
                  <strong>Medium</strong> (10k - 100k clonotypes): <strong>8-32 GB</strong><br>
                  <strong>Large</strong> (&gt; 100k clonotypes): <strong>32+ GB</strong><br><br>

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
        </PlAccordionSection>
        <PlAlert v-if="isEmpty === true" type="warn" :style="{ width: '320px' }">
          <template #title>Empty dataset selection</template>
          The input dataset you have selected is empty or has too few clonotypes.
          Please choose a different dataset.
        </PlAlert>
      </template>
    </GraphMakerPlugin>
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
        :selection="app.plugins.umap.publicData.selection"
      />
    </PlSlideModal>
    <PlSlideModal v-model="umapLogOpen" width="80%">
      <template #title>UMAP Log</template>
      <PlLogView :log-handle="app.model.outputs.umapOutput"/>
    </PlSlideModal>
  </PlBlockPage>
</template>
