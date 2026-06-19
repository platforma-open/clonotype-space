import type { GraphMakerState } from "@milaboratories/graph-maker";
import type {
  PlMultiSequenceAlignmentModel,
  PlRef,
  SUniversalPColumnId,
} from "@platforma-sdk/model";

export type SequenceType = "aminoacid" | "nucleotide";

/**
 * UMAP feature source.
 */
export type InputMode = "sequence-features" | "embedding";

/** Unified V3 data: persisted state, shaped on the UI's terms. */
export type BlockData = {
  customBlockLabel: string;
  inputAnchor?: PlRef;
  /**
   * UMAP feature source. Defaulted to `sequence-features` in `init()` and the
   * `upgradeLegacy` path so existing projects load in sequence mode.
   */
  inputMode: InputMode;
  sequencesRef: SUniversalPColumnId[];
  /**
   * Snapshot of human-readable labels for `sequencesRef`, written by the UI in
   * the same gesture that picks/changes the selection. Used by the args lambda
   * to assemble `defaultBlockLabel`; not consumed by the workflow directly.
   */
  sequenceLabels: string[];
  sequenceType: SequenceType;
  /**
   * Embedding column.
   */
  embeddingRef?: PlRef;
  /**
   * Snapshot of the picked embedding column's native label, written by the UI on
   * the selection gesture.
   */
  selectedEmbeddingLabel: string;
  umap_neighbors: number;
  umap_min_dist: number;
  cpu: number;
  mem: number;
  /**
   * When false, the workflow skips the `.gpuMemory()` call and the software
   * runs on CPU regardless of `gpuMemory`. When true, the workflow gates the
   * request on `exec.hasGpu` so backends without GPU support fall back to CPU
   * anyway.
   */
  requireGpu: boolean;
  /**
   * GPU memory request in GiB. Empty/undefined leaves the field blank in the
   * UI; the workflow falls back to 16 GiB when `requireGpu` is true and this
   * is undefined. Only used when `requireGpu` is true.
   */
  gpuMemory?: number;
  graphStateUMAP: GraphMakerState;
  alignmentModel: PlMultiSequenceAlignmentModel;
};

/**
 * Projected args consumed by the workflow. The `.args` lambda branches on
 * `inputMode` and returns only the active mode's fields plus the shared ones
 * (V3 conditional suppression): `sequencesRef`/`sequenceType` are present
 * only in sequence mode, `embeddingRef` only in embedding mode.
 */
export type BlockArgs = {
  defaultBlockLabel: string;
  customBlockLabel: string;
  inputAnchor: PlRef;
  inputMode: InputMode;
  // Sequence-features mode (omitted in embedding mode).
  sequencesRef?: SUniversalPColumnId[];
  sequenceType?: SequenceType;
  // Embedding mode (omitted in sequence mode).
  embeddingRef?: PlRef;
  umap_neighbors: number;
  umap_min_dist: number;
  cpu: number;
  mem: number;
  requireGpu: boolean;
  gpuMemory?: number;
};

/** Pre-V3 args shape, frozen snapshot for `upgradeLegacy`. */
export type LegacyBlockArgs = {
  defaultBlockLabel: string;
  customBlockLabel: string;
  inputAnchor?: PlRef;
  sequencesRef: SUniversalPColumnId[];
  sequenceType: SequenceType;
  umap_neighbors: number;
  umap_min_dist: number;
  cpu: number;
  mem: number;
};

/** Pre-V3 UI state shape, frozen snapshot for `upgradeLegacy`. */
export type LegacyBlockUiState = {
  graphStateUMAP: GraphMakerState;
  alignmentModel: PlMultiSequenceAlignmentModel;
};
