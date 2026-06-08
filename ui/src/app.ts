import { platforma } from '@platforma-open/milaboratories.clonotype-space.model';
import { defineAppV3 } from '@platforma-sdk/ui-vue';
import UmapPage from './pages/UmapPage.vue';

export const sdkPlugin = defineAppV3(
  platforma,
  (app) => {
    return {
      progress: () => app.model.outputs.isRunning,
      showErrorsNotification: true,
      routes: {
        '/': () => UmapPage,
      },
    };
  },
);

export const useApp = sdkPlugin.useApp;
