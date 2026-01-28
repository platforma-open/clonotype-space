import { getDefaultBlockLabel, model } from '@platforma-open/milaboratories.clonotype-space.model';
import { defineApp } from '@platforma-sdk/ui-vue';
import { watchEffect } from 'vue';
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
  watchEffect(() => {
    const sequenceLabels = model.args.sequencesRef
      .map((r) => {
        const label = model.outputs.sequenceOptions
          ?.find((o) => o.value === r)
          ?.label;
        return label ?? '';
      })
      .filter(Boolean)
      .sort();
    model.args.defaultBlockLabel = getDefaultBlockLabel({
      sequenceLabels,
      umap_neighbors: model.args.umap_neighbors,
      umap_min_dist: model.args.umap_min_dist,
    });
  });
}
