<script setup lang="ts">
import type {
  PlRef,
} from '@platforma-sdk/model';
import {
  plRefsEqual,
} from '@platforma-sdk/model';

import '@milaboratories/graph-maker/styles';
import { PlBlockPage, PlDropdownRef, PlMultiSequenceAlignment } from '@platforma-sdk/ui-vue';
import { useApp } from '../app';

import type { GraphMakerProps } from '@milaboratories/graph-maker';
import { GraphMaker } from '@milaboratories/graph-maker';
import type { PlSelectionModel } from '@platforma-sdk/model';
import { ref } from 'vue';
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

const defaultOptions: GraphMakerProps['defaultOptions'] = [
  {
    inputName: 'x',
    selectedSource: {
      kind: 'PColumn',
      name: 'pl7.app/vdj/umap1',
      valueType: 'Double',
      axesSpec: [
        {
          name: 'pl7.app/clonotypeKey',
          type: 'String',
        },
      ],
    },
  },
  {
    inputName: 'y',
    selectedSource: {
      kind: 'PColumn',
      name: 'pl7.app/vdj/umap2',
      valueType: 'Double',
      axesSpec: [
        {
          name: 'pl7.app/clonotypeKey',
          type: 'String',
        },
      ],
    },
  },
];

const selection = ref<PlSelectionModel>({
  axesSpec: [],
  selectedKeys: [],
});

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
        <PlMultiSequenceAlignment
          v-model="app.model.ui.alignmentModel"
          :label-column-option-predicate="isLabelColumnOption"
          :sequence-column-predicate="isSequenceColumn"
          :linker-column-predicate="isLinkerColumn"
          :p-frame="app.model.outputs.msaPf"
          :selection="selection"
        />
      </template>
      <template #settingsSlot>
        <PlDropdownRef
          v-model="app.model.args.inputAnchor"
          :options="app.model.outputs.inputOptions"
          label="Select dataset"
          required
          @update:model-value="setAnchorColumn"
        />
      </template>
    </GraphMaker>
  </PlBlockPage>
</template>
