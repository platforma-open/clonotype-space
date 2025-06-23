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
import { PlAlert, PlBlockPage, PlBtnGhost, PlDropdownRef, PlMultiSequenceAlignment, PlNumberField, PlSectionSeparator, PlSlideModal } from '@platforma-sdk/ui-vue';
import { useApp } from '../app';

import type { GraphMakerProps, PredefinedGraphOption } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import type { PlSelectionModel } from '@platforma-sdk/model';
import { asyncComputed } from '@vueuse/core';
import { computed, ref } from 'vue';
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

const defaultOptions = computed((): GraphMakerProps['defaultOptions'] => {
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
        <PlSectionSeparator>UMAP parameters</PlSectionSeparator>
        <PlNumberField
          v-model="app.model.args.umap_neighbors"
          label="N Neighbors"
          :min="5"
          :max="500"
          :step="5"
          required
          :style="{ width: '320px' }"
        />
        <PlNumberField
          v-model="app.model.args.umap_min_dist"
          label="Min Distance"
          :min="0"
          :max="1"
          :step="0.1"
          required
          :style="{ width: '320px' }"
        />
        <PlAlert v-if="isEmpty === true" type="warn" :style="{ width: '320px' }">
          <template #title>Empty dataset selection</template>
          The input dataset you have selected is empty.
          Please choose a different dataset.
        </PlAlert>
      </template>
    </GraphMaker>
    <PlSlideModal v-model="multipleSequenceAlignmentOpen" width="100%">
      <template #title>Multiple Sequence Alignment</template>
      <PlMultiSequenceAlignment
        v-model="app.model.ui.alignmentModel"
        :label-column-option-predicate="isLabelColumnOption"
        :sequence-column-predicate="isSequenceColumn"
        :linker-column-predicate="isLinkerColumn"
        :p-frame="app.model.outputs.msaPf"
        :selection="selection"
      />
    </PlSlideModal>
  </PlBlockPage>
</template>
