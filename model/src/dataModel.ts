import { DataModelBuilder } from '@platforma-sdk/model';
import type {
  BlockData,
  LegacyBlockArgs,
  LegacyBlockUiState,
} from './types';

const defaultGraphState = (): BlockData['graphStateUMAP'] => ({
  title: 'Sequence Space UMAP',
  template: 'dots',
  currentTab: 'settings',
  layersSettings: {
    dots: {
      dotFill: '#99E099',
    },
  },
});

export const blockDataModel = new DataModelBuilder()
  .from<BlockData>('V20260518')
  .upgradeLegacy<LegacyBlockArgs, LegacyBlockUiState>(({ args, uiState }) => ({
    customBlockLabel: args?.customBlockLabel ?? '',
    inputAnchor: args?.inputAnchor,
    // Existing projects load in sequence mode (embedding mode is opt-in).
    inputMode: 'sequence-features',
    sequencesRef: args?.sequencesRef ?? [],
    // No V1 source: the previous build derived labels live via a UI hairpin.
    // Seeded empty here; the UI's auto-select watcher reseeds them from the
    // current options on load (no user interaction required).
    sequenceLabels: [],
    sequenceType: args?.sequenceType ?? 'aminoacid',
    embeddingRef: undefined,
    selectedEmbeddingLabel: '',
    umap_neighbors: args?.umap_neighbors ?? 15,
    umap_min_dist: args?.umap_min_dist ?? 0.5,
    cpu: args?.cpu ?? 8,
    mem: args?.mem ?? 64,
    graphStateUMAP: uiState?.graphStateUMAP ?? defaultGraphState(),
    alignmentModel: uiState?.alignmentModel ?? {},
  }))
  .init(() => ({
    customBlockLabel: '',
    inputAnchor: undefined,
    inputMode: 'sequence-features', // embedding mode is opt-in
    sequencesRef: [],
    sequenceLabels: [],
    sequenceType: 'aminoacid',
    embeddingRef: undefined,
    selectedEmbeddingLabel: '',
    umap_neighbors: 15,
    umap_min_dist: 0.5,
    cpu: 8,
    mem: 64,
    graphStateUMAP: defaultGraphState(),
    alignmentModel: {},
  }));
