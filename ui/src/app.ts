import { model } from '@platforma-open/milaboratories.clonotype-space.model';
import { defineApp } from '@platforma-sdk/ui-vue';
import UmapPage from './pages/UmapPage.vue';

export const sdkPlugin = defineApp(model, (app) => {
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
