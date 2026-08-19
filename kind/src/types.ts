import type { PlRef, SUniversalPColumnId } from "@platforma-sdk/model";

export type SequenceType = "aminoacid" | "nucleotide";

/** UMAP feature source: sequence features, or a precomputed embedding column. */
export type InputMode = "sequence-features" | "embedding";

/**
 * This block's init-params contract — the shape a block of this kind receives
 * at creation, and exactly what a project template serializes for it.
 *
 * Every field is optional. A block with no input picked and no sequence columns
 * selected is an ordinary state the UI reaches, so export has to be able to
 * write it and apply has to be able to take it back; a contract that demanded
 * `inputAnchor` would make export and apply stop being inverses. Whether a
 * configuration is runnable is settled by the model's `args` lambda, not here.
 *
 * Plot and view state — the UMAP graph state and the alignment model — is
 * absent on purpose: it is how one user was looking at one result, not the
 * recipe a template exists to reproduce.
 */
export type BlockParams = {
  // Input wiring — PlRefs a template engine fills from an earlier entry's output.
  inputAnchor?: PlRef;
  embeddingRef?: PlRef;

  // Analysis configuration — the recipe a template exists to reproduce.
  inputMode?: InputMode;
  sequencesRef?: SUniversalPColumnId[];
  sequenceType?: SequenceType;
  umap_neighbors?: number;
  umap_min_dist?: number;

  // Label snapshots the UI writes in the same gesture that picks a column.
  // They travel with the selection they describe — without them an applied
  // template shows a blank subtitle until the user re-picks.
  sequenceLabels?: string[];
  selectedEmbeddingLabel?: string;

  // Per-process resource limits, and the switch that decides whether they are
  // used at all: when off, the workflow sizes CPU and RAM from input size.
  directPerformanceSettings?: boolean;
  cpu?: number;
  mem?: number;
  requireGpu?: boolean;
  gpuMemory?: number;

  // Display naming.
  customBlockLabel?: string;
};
