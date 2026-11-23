import { graphMakerPlugin } from "@milaboratories/graph-maker/plugin";

/**
 * The UMAP chart is a graph-maker plugin instance: it owns its own view state
 * and builds its own PFrame from the columns handed to it as anchors.
 *
 * Declared apart from the block model because the data model's `.transfer()`
 * names it, and the block model imports the data model.
 */
export const umapPlugin = graphMakerPlugin("scatterplot-umap").create({
  pluginId: "umap",
  // Where a chart saved before the plugin enters the chain.
  transferAt: "v1",
  config: {
    initialTitle: "Sequence Space UMAP",
    initialTemplate: "dots",
    initialState: {
      currentTab: "settings",
      layersSettings: { dots: { dotFill: "#99E099" } },
    },
  },
});
