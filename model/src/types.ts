import type { GraphMakerState } from "@milaboratories/graph-maker";
import type {
  PlMultiSequenceAlignmentModel,
  PlRef,
  SUniversalPColumnId,
} from "@platforma-sdk/model";

export type SequenceType = "aminoacid" | "nucleotide";

/** Unified V3 data: persisted state, shaped on the UI's terms. */
export type BlockData = {
  customBlockLabel: string;
  inputAnchor?: PlRef;
  sequencesRef: SUniversalPColumnId[];
  /**
   * Snapshot of human-readable labels for `sequencesRef`, written by the UI in
   * the same gesture that picks/changes the selection. Used by the args lambda
   * to assemble `defaultBlockLabel`; not consumed by the workflow directly.
   */
  sequenceLabels: string[];
  sequenceType: SequenceType;
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
  /** GPU memory request in GiB. Only used when `requireGpu` is true. */
  gpuMemory: number;
  graphStateUMAP: GraphMakerState;
  alignmentModel: PlMultiSequenceAlignmentModel;
};

/** Projected args consumed by the workflow. */
export type BlockArgs = {
  defaultBlockLabel: string;
  customBlockLabel: string;
  inputAnchor: PlRef;
  sequencesRef: SUniversalPColumnId[];
  sequenceType: SequenceType;
  umap_neighbors: number;
  umap_min_dist: number;
  cpu: number;
  mem: number;
  requireGpu: boolean;
  gpuMemory: number;
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
