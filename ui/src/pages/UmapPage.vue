<script setup lang="ts">
import type {
  PColumnIdAndSpec,
  PlRef,
} from '@platforma-sdk/model';
import {
  getRawPlatformaInstance,
  plRefsEqual,
} from '@platforma-sdk/model';

import '@milaboratories/graph-maker/styles';
import { PlAccordionSection, PlAlert, PlBlockPage, PlBtnGhost, PlDropdownRef, PlLogView, PlMaskIcon24, PlMultiSequenceAlignment, PlNumberField, PlSlideModal } from '@platforma-sdk/ui-vue';
import { useApp } from '../app';

import type { GraphMakerProps, PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import type { PlSelectionModel } from '@platforma-sdk/model';
import { asyncComputed } from '@vueuse/core';

import { computed, ref, watch } from 'vue';
import { isLabelColumnOption, isLinkerColumn, isSequenceColumn } from '../util';

const app = useApp();

function setAnchorColumn(ref: PlRef | undefined) {
  app.model.args.inputAnchor = ref;
  app.model.ui.title = 'Clonotype Space - ' + (ref
    ? app.model.outputs.inputOptions?.find((o) =>
      plRefsEqual(o.ref, ref),
    )?.label
    : '');
}

const defaultOptions = computed((): PredefinedGraphOption<'scatterplot-umap'>[] | undefined => {
  if (!app.model.outputs.umapPcols)
    return undefined;

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

const selection = ref<PlSelectionModel>({
  axesSpec: [],
  selectedKeys: [],
});

const multipleSequenceAlignmentOpen = ref(false);
const umapLogOpen = ref(false);

// Auto-close settings panel when block starts running
watch(
  () => app.model.outputs.isRunning,
  (isRunning, wasRunning) => {
    // Close settings when block starts running (false -> true transition)
    if (isRunning && !wasRunning) {
      // Close the settings tab by setting currentTab to null
      app.model.ui.graphStateUMAP.currentTab = null;
    }
  },
);
</script>

<template>
  <PlBlockPage>
    <GraphMaker
      v-model="app.model.ui.graphStateUMAP"
      v-model:selection="selection"
      chartType="scatterplot-umap"
      :data-state-key="app.model.outputs.umapPf"
      :p-frame="app.model.outputs.umapPf"
      :default-options="defaultOptions"
    >
      <template #titleLineSlot>
        <PlBtnGhost
          icon="dna"
          @click.stop="() => (multipleSequenceAlignmentOpen = true)"
        >
          Multiple Sequence Alignment
        </PlBtnGhost>
        <PlBtnGhost @click.stop="() => (umapLogOpen = true)">
          Logs
          <template #append>
            <PlMaskIcon24 name="progress" />
          </template>
        </PlBtnGhost>
      </template>
      <template #settingsSlot>
        <PlDropdownRef
          v-model="app.model.args.inputAnchor"
          :options="app.model.outputs.inputOptions"
          label="Select dataset"
          required
          :style="{ width: '320px' }"
          @update:model-value="setAnchorColumn"
        />

        <PlAccordionSection label="UMAP Parameters" :style="{ width: '320px' }">
          <div :style="{ display: 'flex', gap: '8px', width: '320px' }">
            <PlNumberField
              v-model="app.model.args.umap_neighbors"
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
              v-model="app.model.args.umap_min_dist"
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
              v-model="app.model.args.mem"
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
              v-model="app.model.args.cpu"
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
    </GraphMaker>
    <PlSlideModal
      v-model="multipleSequenceAlignmentOpen"
      width="100%"
      :close-on-outside-click="false"
    >
      <template #title>Multiple Sequence Alignment</template>
      <PlMultiSequenceAlignment
        v-model="app.model.ui.alignmentModel"
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
