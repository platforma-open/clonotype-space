import strings from "@milaboratories/strings";
import { kind } from "@platforma-open/milaboratories.clonotype-space.kind";
import type {
  DataInfo,
  InferOutputsType,
  PColumn,
  PColumnIdAndSpec,
  PColumnValues,
  PFrameHandle,
  RenderCtxBase,
  TreeNodeAccessor,
} from "@platforma-sdk/model";
import { BlockModelV3, createPFrameForGraphs, isPColumnSpec } from "@platforma-sdk/model";
import { blockDataModel } from "./dataModel";
import { getDefaultBlockLabel } from "./label";
import type { BlockArgs, BlockData } from "./types";

type Column = PColumn<DataInfo<TreeNodeAccessor> | TreeNodeAccessor | PColumnValues | undefined>;

const inputAnchorSpecs = [
  {
    axes: [{ name: "pl7.app/sampleId" }, { name: "pl7.app/vdj/clonotypeKey" }],
    annotations: { "pl7.app/isAnchor": "true" },
  },
  {
    axes: [{ name: "pl7.app/sampleId" }, { name: "pl7.app/vdj/scClonotypeKey" }],
    annotations: { "pl7.app/isAnchor": "true" },
  },
  {
    axes: [{ name: "pl7.app/sampleId" }, { name: "pl7.app/variantKey" }],
    annotations: { "pl7.app/isAnchor": "true" },
  },
];

function computeDefaultLabel(data: BlockData): string {
  if (data.inputMode === "embedding") {
    return getDefaultBlockLabel({
      inputMode: "embedding",
      embeddingLabel: data.selectedEmbeddingLabel ?? "",
      umap_neighbors: data.umap_neighbors,
      umap_min_dist: data.umap_min_dist,
    });
  }
  return getDefaultBlockLabel({
    inputMode: "sequence-features",
    sequenceLabels: data.sequenceLabels,
    umap_neighbors: data.umap_neighbors,
    umap_min_dist: data.umap_min_dist,
  });
}

function getAnchoredClonotypeProps(
  ctx: Pick<RenderCtxBase<BlockArgs, BlockData>, "data" | "resultPool">,
): Column[] | undefined {
  const anchor = ctx.data.inputAnchor;
  if (!anchor) return undefined;

  const anchorSpec = ctx.resultPool.getPColumnSpecByRef(anchor);
  if (!anchorSpec) return undefined;

  return (
    ctx.resultPool.getAnchoredPColumns({ main: anchor }, [
      { axes: [{ anchor: "main", idx: 1 }] },
    ]) ?? []
  ).filter((p) => p.spec.annotations?.["pl7.app/sequence/isAnnotation"] !== "true");
}

export const platforma = BlockModelV3.create({ dataModel: blockDataModel, kind })

  // Inverse of `init` — the same fields, projected back out for template export.
  // `graphStateUMAP` and `alignmentModel` are view state and never cross the
  // boundary: they say how one result was being looked at, not how to produce
  // it. Both modes' fields go out as they stand; the off-mode ones are inert
  // until the mode is switched back, and dropping them would lose a
  // configuration the user actually set.
  .templateParams((data) => ({
    inputAnchor: data.inputAnchor,
    embeddingRef: data.embeddingRef,

    inputMode: data.inputMode,
    sequencesRef: data.sequencesRef,
    sequenceType: data.sequenceType,
    umap_neighbors: data.umap_neighbors,
    umap_min_dist: data.umap_min_dist,

    sequenceLabels: data.sequenceLabels,
    selectedEmbeddingLabel: data.selectedEmbeddingLabel,

    directPerformanceSettings: data.directPerformanceSettings,
    cpu: data.cpu,
    mem: data.mem,
    requireGpu: data.requireGpu,
    gpuMemory: data.gpuMemory,

    customBlockLabel: data.customBlockLabel,
  }))

  .args<BlockArgs>((data) => {
    if (data.inputAnchor === undefined) throw new Error("Input dataset is required");
    if (data.umap_neighbors === undefined) throw new Error("UMAP neighbors is required");
    if (data.umap_neighbors < 2) throw new Error("UMAP requires at least 2 neighbors");
    if (data.umap_min_dist === undefined) throw new Error("UMAP min distance is required");
    if (data.umap_min_dist < 0 || data.umap_min_dist > 1)
      throw new Error("UMAP min distance must be between 0 and 1");
    if (data.cpu === undefined) throw new Error("CPU is required");
    if (data.cpu < 1) throw new Error("CPU count must be at least 1");
    if (data.mem === undefined) throw new Error("Memory is required");
    if (data.mem < 1) throw new Error("Memory must be at least 1 GB");
    data.requireGpu = data.requireGpu ?? false;
    // Only when a GPU is asked for: the field is disabled otherwise, so a stale
    // value behind an unchecked box is not something to refuse a run over.
    if (data.requireGpu && (data.gpuMemory === undefined || data.gpuMemory < 1))
      throw new Error("GPU memory must be at least 1 GB when a GPU is required");

    // Shared by both modes. The lambda branches on inputMode and returns ONLY the active mode's
    // fields, so a stale off-mode value can't affect the run.
    const shared = {
      defaultBlockLabel: computeDefaultLabel(data),
      customBlockLabel: data.customBlockLabel,
      inputAnchor: data.inputAnchor,
      inputMode: data.inputMode,
      umap_neighbors: data.umap_neighbors,
      umap_min_dist: data.umap_min_dist,
      directPerformanceSettings: data.directPerformanceSettings,
      cpu: data.cpu,
      mem: data.mem,
      requireGpu: data.requireGpu,
      gpuMemory: data.gpuMemory,
    };

    if (data.inputMode === "embedding") {
      if (!data.embeddingRef)
        throw new Error(
          "Connect a Sequence Embeddings output and pick an embedding column to project by embedding",
        );
      // The embedding model is read from the column spec in the workflow, not snapshotted here (R10).
      return { ...shared, embeddingRef: data.embeddingRef };
    }

    if (data.sequencesRef.length === 0) throw new Error("At least one sequence column is required");
    return { ...shared, sequencesRef: data.sequencesRef, sequenceType: data.sequenceType };
  })

  .output("inputOptions", (ctx) => ctx.resultPool.getOptions(inputAnchorSpecs))

  .output(
    "modality",
    (ctx) => {
      const spec = ctx.data.inputAnchor
        ? ctx.resultPool.getPColumnSpecByRef(ctx.data.inputAnchor)
        : undefined;
      if (!spec) return undefined;
      for (const ax of spec.axesSpec) {
        if (ax.name === "pl7.app/variantKey") {
          // peptide-extraction and synthetic-repertoire-profiler share this axis;
          // the domain distinguishes them (see sequenceOptions for the same check).
          return (ax.domain ?? {})["pl7.app/repertoire/extractionRunId"] !== undefined
            ? "amplicon"
            : "peptide";
        }
        if (ax.name === "pl7.app/vdj/clonotypeKey" || ax.name === "pl7.app/vdj/scClonotypeKey")
          return "antibody_tcr";
      }
      return "antibody_tcr";
    },
    { retentive: true },
  )

  .output("sequenceOptions", (ctx) => {
    const ref = ctx.data.inputAnchor;
    if (ref === undefined) return undefined;

    const keyAxis = ctx.resultPool.getPColumnSpecByRef(ref)?.axesSpec[1];
    const axis1Name = keyAxis?.name;
    const keyAxisDomain = keyAxis?.domain ?? {};

    const inputKind: "peptide" | "amplicon" | "singleCell" | "bulk" =
      axis1Name === "pl7.app/variantKey"
        ? keyAxisDomain["pl7.app/repertoire/extractionRunId"] !== undefined
          ? "amplicon"
          : "peptide"
        : axis1Name === "pl7.app/vdj/scClonotypeKey"
          ? "singleCell"
          : "bulk";

    const sequenceMatchers = [];

    switch (inputKind) {
      case "peptide":
        // Peptide: peptide-extraction emits both nt and aa sequences with
        // name 'pl7.app/sequence' and domain.pl7.app/feature: 'peptide'.
        sequenceMatchers.push({
          axes: [{ anchor: "main", idx: 1 }],
          name: "pl7.app/sequence",
          domain: { "pl7.app/feature": "peptide" },
        });
        break;
      case "amplicon":
        // synthetic-repertoire-profiler keys sequences on pl7.app/variantKey but tags
        // each with a feature: the whole-variant sequence (feature "amplicon-sequence")
        // and one per region (feature = region name)
        sequenceMatchers.push({
          axes: [{ anchor: "main", idx: 1 }],
          name: "pl7.app/sequence",
        });
        break;
      case "singleCell":
        // Single-cell: per-chain sequences (primary chain) + scFv construct
        sequenceMatchers.push({
          axes: [{ anchor: "main", idx: 1 }],
          name: "pl7.app/vdj/sequence",
          domain: { "pl7.app/vdj/scClonotypeChain/index": "primary" },
        });
        sequenceMatchers.push({
          axes: [{ anchor: "main", idx: 1 }],
          name: "pl7.app/vdj/scFv-sequence",
        });
        break;
      case "bulk":
        // Bulk: regular VDJ sequences (all features × alphabets)
        sequenceMatchers.push({
          axes: [{ anchor: "main", idx: 1 }],
          name: "pl7.app/vdj/sequence",
        });
        break;
    }

    const options = ctx.resultPool.getCanonicalOptions({ main: ref }, sequenceMatchers, {
      ignoreMissingDomains: true,
      labelOps: {
        includeNativeLabel: true,
      },
    });

    if (!options) return undefined;

    // Pre-compute all necessary fields for UI filtering and sorting
    const optionsWithMetadata = options.map((option) => {
      const colId = JSON.parse(option.value) as never;
      const columns = ctx.resultPool.getAnchoredPColumns({ main: ref }, [colId]);
      const spec = columns?.[0]?.spec;
      const alphabet = spec?.domain?.["pl7.app/alphabet"] as "aminoacid" | "nucleotide" | undefined;
      // Read both annotation forms: VDJ uses 'pl7.app/vdj/isMainSequence',
      // peptide-extraction uses the modality-neutral 'pl7.app/isMainSequence'.
      const isMain =
        spec?.annotations?.["pl7.app/vdj/isMainSequence"] === "true" ||
        spec?.annotations?.["pl7.app/isMainSequence"] === "true";

      return {
        label: option.label,
        value: option.value,
        alphabet,
        isMain,
      };
    });

    // Sort: main sequences first, then keep result-pool order
    return optionsWithMetadata.sort((a, b) => {
      if (a.isMain && !b.isMain) return -1;
      if (b.isMain && !a.isMain) return 1;
      return 0;
    });
  })

  .output("embeddingOptions", (ctx) => {
    const ref = ctx.data.inputAnchor;
    if (ref === undefined) return undefined;
    // PlRef-based options (NOT getCanonicalOptions): the embedding's producer must be wired as an
    // upstream via wf.resolve(PlRef), so the picker binds a PlRef.
    const datasetSpec = ctx.resultPool.getPColumnSpecByRef(ref);
    const cloneAxis = datasetSpec?.axesSpec?.[1];
    if (cloneAxis === undefined) return undefined;
    const sameClonotypeAxis = (embAxis?: { name?: string; domain?: Record<string, string> }) => {
      if (embAxis === undefined || embAxis.name !== cloneAxis.name) return false;
      const datasetDomain = cloneAxis.domain ?? {};
      const embDomain = embAxis.domain ?? {};
      return Object.keys(datasetDomain).every((k) => embDomain[k] === datasetDomain[k]);
    };
    const options = ctx.resultPool.getOptions(
      (spec) =>
        isPColumnSpec(spec) &&
        spec.name === "pl7.app/embedding" &&
        sameClonotypeAxis(spec.axesSpec?.[0]),
      { label: { includeNativeLabel: true } },
    );
    return options.map((o) => {
      const spec = ctx.resultPool.getPColumnSpecByRef(o.ref);
      return {
        ref: o.ref,
        label: o.label,
        feature: spec?.domain?.["pl7.app/feature"],
        chain: spec?.domain?.["pl7.app/vdj/scClonotypeChain"],
      };
    });
  })

  .output("msaPf", (ctx) => {
    const props = getAnchoredClonotypeProps(ctx);
    if (!props) return undefined;
    return createPFrameForGraphs(ctx, props);
  })

  .outputWithStatus("umapPf", (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve("umapPf")?.getPColumns();
    if (pCols === undefined) return undefined;
    return createPFrameForGraphs(ctx, pCols);
  })

  .output("umapOutput", (ctx) => ctx.outputs?.resolve("umapOutput")?.getLogHandle())

  // Single-column PTable used by the UI to detect empty UMAP output
  // (the input dataset has too few sequences to embed).
  .output("umapDim1Table", (ctx) => {
    const pCols = ctx.outputs?.resolve("umapPf")?.getPColumns();
    if (pCols === undefined) return undefined;
    const dim1Column = pCols.find((p) => p.spec.name === "pl7.app/umap1");
    if (dim1Column === undefined) return undefined;
    return ctx.createPTable({ columns: [dim1Column] });
  })

  .output("umapPcols", (ctx) => {
    const pCols = ctx.outputs?.resolve("umapPf")?.getPColumns();
    if (pCols === undefined || pCols.length === 0) return undefined;
    return pCols.map(
      (c) =>
        ({
          columnId: c.id,
          spec: c.spec,
        }) satisfies PColumnIdAndSpec,
    );
  })

  .output("isRunning", (ctx) => ctx.outputs?.getIsReadyOrError() === false)

  .title(() => "Sequence Space")

  .subtitle((ctx) => ctx.data.customBlockLabel || computeDefaultLabel(ctx.data))

  .sections((_ctx) => [{ type: "link" as const, href: "/" as const, label: strings.titles.main }])

  .done();

export type Platforma = typeof platforma;
export type BlockOutputs = InferOutputsType<typeof platforma>;

export { getDefaultBlockLabel } from "./label";
export * from "./types";
