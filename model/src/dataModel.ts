import { kind } from "@platforma-open/milaboratories.clonotype-space.kind";
import { DataModelBuilder } from "@platforma-sdk/model";
import { transferableGraphState } from "@milaboratories/graph-maker/plugin";
import type { BlockData, LegacyBlockArgs, LegacyBlockUiState } from "./types";
import { umapPlugin } from "./plugins";

const defaultGraphState = (): BlockData["graphStateUMAP"] => ({
  title: "Sequence Space UMAP",
  template: "dots",
  currentTab: "settings",
  layersSettings: {
    dots: {
      dotFill: "#99E099",
    },
  },
});

export const blockDataModel = new DataModelBuilder({ kind })
  .from<BlockData>("V20260518")
  .upgradeLegacy<LegacyBlockArgs, LegacyBlockUiState>(({ args, uiState }) => ({
    customBlockLabel: args?.customBlockLabel ?? "",
    inputAnchor: args?.inputAnchor,
    // Existing projects load in sequence mode (embedding mode is opt-in).
    inputMode: "sequence-features",
    sequencesRef: args?.sequencesRef ?? [],
    // No V1 source: the previous build derived labels live via a UI hairpin.
    // Seeded empty here; the UI's auto-select watcher reseeds them from the
    // current options on load (no user interaction required).
    sequenceLabels: [],
    sequenceType: args?.sequenceType ?? "aminoacid",
    embeddingRef: undefined,
    selectedEmbeddingLabel: "",
    umap_neighbors: args?.umap_neighbors ?? 15,
    umap_min_dist: args?.umap_min_dist ?? 0.5,
    directPerformanceSettings: false,
    cpu: args?.cpu ?? 8,
    mem: args?.mem ?? 64,
    requireGpu: false,
    gpuMemory: undefined,
    graphStateUMAP: uiState?.graphStateUMAP ?? defaultGraphState(),
    alignmentModel: uiState?.alignmentModel ?? {},
  }))
  // A chart saved before the UMAP became a plugin instance: hand the plugin
  // what survives the move. The data mapping does not — the old ids name
  // columns in a frame built the old way — so it is left to `defaultOptions`.
  .transfer(umapPlugin, (data) => ({
    state: transferableGraphState(data.graphStateUMAP),
    chartType: "scatterplot-umap" as const,
    readonlyInputs: [],
    allowChartDeleting: false,
    allowTitleEditing: false,
  }))
  // `params` is absent when a block is created by hand rather than from a
  // template, so every field the contract carries keeps its own default.
  // `graphStateUMAP` and `alignmentModel` are outside the contract — they are
  // how one user was looking at one result, not the recipe to reproduce it.
  .init(({ params }) => ({
    customBlockLabel: params?.customBlockLabel ?? "",
    inputAnchor: params?.inputAnchor,
    filterRef: params?.filterRef,
    inputMode: params?.inputMode ?? "sequence-features", // embedding mode is opt-in
    sequencesRef: params?.sequencesRef ?? [],
    sequenceLabels: params?.sequenceLabels ?? [],
    sequenceType: params?.sequenceType ?? "aminoacid",
    embeddingRef: params?.embeddingRef,
    selectedEmbeddingLabel: params?.selectedEmbeddingLabel ?? "",
    umap_neighbors: params?.umap_neighbors ?? 15,
    umap_min_dist: params?.umap_min_dist ?? 0.5,
    directPerformanceSettings: params?.directPerformanceSettings ?? false,
    cpu: params?.cpu ?? 8,
    mem: params?.mem ?? 64,
    requireGpu: params?.requireGpu ?? false,
    gpuMemory: params?.gpuMemory,
    graphStateUMAP: defaultGraphState(),
    alignmentModel: {},
  }));
