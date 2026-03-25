import { isJsonEqual } from '@milaboratories/helpers';
import { getDefaultBlockLabel, platforma } from '@platforma-open/milaboratories.clonotype-space.model';
import { defineAppV3 } from '@platforma-sdk/ui-vue';
import { computed, watchEffect } from 'vue';
import UmapPage from './pages/UmapPage.vue';

export const sdkPlugin = defineAppV3(platforma, (app) => {
  app.model.data.customBlockLabel ??= '';

  syncDefaultBlockLabel(app.model);

  return {
    progress: () => {
      return app.model.outputs.isRunning;
    },
    showErrorsNotification: true,
    routes: {
      '/': () => UmapPage,
    },
  };
});

export const useApp = sdkPlugin.useApp;

type AppModel = ReturnType<typeof useApp>['model'];

function syncDefaultBlockLabel(model: AppModel) {
  const sequenceLabels = computed(() => {
    return model.data.sequencesRef
      .map((r) => {
        const label = model.outputs.sequenceOptions
          ?.find((o) => isJsonEqual(o.value, r))
          ?.label;
        return label ?? '';
      })
      .filter(Boolean)
      .sort();
  });

  watchEffect(() => {
    model.data.defaultBlockLabel = getDefaultBlockLabel({
      sequenceLabels: sequenceLabels.value,
      umap_neighbors: model.data.umap_neighbors,
      umap_min_dist: model.data.umap_min_dist,
    });
  });
}
