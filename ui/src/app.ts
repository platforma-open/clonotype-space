import { isJsonEqual } from '@milaboratories/helpers';
import { getDefaultBlockLabel, model } from '@platforma-open/milaboratories.clonotype-space.model';
import { defineApp } from '@platforma-sdk/ui-vue';
import { computed, watchEffect } from 'vue';
import UmapPage from './pages/UmapPage.vue';

export const sdkPlugin = defineApp(model, (app) => {
  app.model.args.customBlockLabel ??= '';

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
    return model.args.sequencesRef
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
    model.args.defaultBlockLabel = getDefaultBlockLabel({
      sequenceLabels: sequenceLabels.value,
      umap_neighbors: model.args.umap_neighbors,
      umap_min_dist: model.args.umap_min_dist,
    });
  });
}
