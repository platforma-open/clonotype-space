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
import { PlAccordionSection, PlAlert, PlBlockPage, PlBtnGhost, PlDropdownRef, PlMultiSequenceAlignment, PlNumberField, PlSlideModal } from '@platforma-sdk/ui-vue';
import { useApp } from '../app';

import type { GraphMakerProps, PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import type { PlSelectionModel } from '@platforma-sdk/model';
import { asyncComputed } from '@vueuse/core';
import { computed, ref } from 'vue';
import { isSequenceColumn } from '../util';

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
        <PlAccordionSection label="UMAP parameters" :style="{ width: '320px' }">
          <PlNumberField
            v-model="app.model.args.umap_neighbors"
            label="N Neighbors"
            :min="5"
            :max="500"
            :step="5"
            required
            :style="{ width: '320px' }"
          >
            <template #tooltip>
              <div>
                <strong>Number of Neighbors for UMAP</strong><br>
                Controls the balance between local and global structure in UMAP visualization.<br><br>
                <strong>Recommended ranges:</strong><br>
                • 10-30: Optimal for most datasets<br>
                • 5-10: Emphasizes local structure (more clusters)<br>
                • 30+: Emphasizes global structure (fewer clusters)<br><br>
              </div>
            </template>
          </PlNumberField>
          <PlNumberField
            v-model="app.model.args.umap_min_dist"
            label="Min Distance"
            :min="0"
            :max="1"
            :step="0.1"
            required
            :style="{ width: '320px' }"
          >
            <template #tooltip>
              <div>
                <strong>Minimum Distance for UMAP</strong><br>
                Controls how tightly UMAP packs points together. Lower values create denser clusters, while higher values preserve broader structure.<br><br>
                <strong>Recommended ranges:</strong><br>
                • 0.0 - 0.2: For creating tight clusters.<br>
                • 0.2 - 0.5: A good balance for most datasets.<br>
                • 0.5 - 1.0: For a more global view of the data.<br><br>
              </div>
            </template>
          </PlNumberField>

          <PlNumberField
            v-model="app.model.args.mem"
            label="Memory (GB)"
            :min="8"
            :max="1024"
            :step="1"
            :style="{ width: '320px' }"
          >
            <template #tooltip>
              <div>
                <strong>Memory (GB) for UMAP Calculation</strong><br>
                Set the amount of memory (in GB) for the UMAP calculation. The right amount depends on the number of clonotypes in your dataset.<br><br>
                <strong>Recommended Memory:</strong><br>
                <strong>Small</strong> (&lt; 10k clonotypes): <strong>4-8 GB</strong><br>
                <strong>Medium</strong> (10k - 100k clonotypes): <strong>8-32 GB</strong><br>
                <strong>Large</strong> (&gt; 100k clonotypes): <strong>32+ GB</strong><br><br>

                <hr>
                ⚠️ Insufficient memory can cause the process to fail. If you run into errors, try increasing the allocated memory. <br>

                <strong>Note:</strong> Larger values for the <code>n_neighbors</code> parameter can also increase memory usage.
              </div>
            </template>
          </PlNumberField>

          <PlNumberField
            v-model="app.model.args.cpu"
            label="CPU"
            :min="1"
            :max="128"
            :step="1"
            :style="{ width: '320px' }"
          >
            <template #tooltip>
              Amount of CPU cores to request for the UMAP calculation.
            </template>
          </PlNumberField>
        </PlAccordionSection>
        <PlAlert v-if="isEmpty === true" type="warn" :style="{ width: '320px' }">
          <template #title>Empty dataset selection</template>
          The input dataset you have selected is empty.
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
  </PlBlockPage>
</template>
