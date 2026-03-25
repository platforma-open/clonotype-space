import type { GraphMakerState } from '@milaboratories/graph-maker';
import { graphMakerPlugin } from '@milaboratories/graph-maker/plugin';
import type {
  BlockRenderCtx,
  DataInfo,
  InferOutputsType,
  PColumn,
  PColumnIdAndSpec,
  PColumnValues,
  PlMultiSequenceAlignmentModel,
  PlRef,
  SUniversalPColumnId,
  TreeNodeAccessor,
} from '@platforma-sdk/model';
import {
  BlockModelV3,
  DataModelBuilder,
  createPFrameForGraphs,
} from '@platforma-sdk/model';
import strings from '@milaboratories/strings';
import { getDefaultBlockLabel } from './label';

// ---------------------------------------------------------------------------
// Block data versions
// ---------------------------------------------------------------------------

type OldArgs = {
  defaultBlockLabel: string;
  customBlockLabel: string;
  inputAnchor?: PlRef;
  sequencesRef: SUniversalPColumnId[];
  sequenceType: 'aminoacid' | 'nucleotide';
  umap_neighbors: number;
  umap_min_dist: number;
  cpu: number;
  mem: number;
};

type OldUiState = {
  graphStateUMAP: GraphMakerState;
  alignmentModel: PlMultiSequenceAlignmentModel;
};

/** v1 block data — includes graph state that will be transferred to plugin */
type BlockDataV1 = {
  defaultBlockLabel: string;
  customBlockLabel: string;
  inputAnchor?: PlRef;
  sequencesRef: SUniversalPColumnId[];
  sequenceType: 'aminoacid' | 'nucleotide';
  umap_neighbors: number;
  umap_min_dist: number;
  cpu: number;
  mem: number;
  graphStateUMAP: GraphMakerState;
  alignmentModel: PlMultiSequenceAlignmentModel;
};

/** v2 block data — graph state lives in plugin */
export type BlockData = {
  defaultBlockLabel: string;
  customBlockLabel: string;
  inputAnchor?: PlRef;
  sequencesRef: SUniversalPColumnId[];
  sequenceType: 'aminoacid' | 'nucleotide';
  umap_neighbors: number;
  umap_min_dist: number;
  cpu: number;
  mem: number;
  alignmentModel: PlMultiSequenceAlignmentModel;
};

// ---------------------------------------------------------------------------
// Plugin instances
// ---------------------------------------------------------------------------

const umapPlugin = graphMakerPlugin.create({
  pluginId: 'umap',
  transferAt: 'v1',
  config: {
    chartType: 'scatterplot-umap',
    initialTitle: 'Clonotype Space UMAP',
    initialTemplate: 'dots',
    initialState: {
      currentTab: 'settings',
      layersSettings: {
        dots: {
          dotFill: '#99E099',
        },
      },
    },
  },
});

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

const dataModel = new DataModelBuilder()
  .from<BlockDataV1>('v1')
  .upgradeLegacy<OldArgs, OldUiState>(({ args, uiState }) => ({
    defaultBlockLabel: args.defaultBlockLabel,
    customBlockLabel: args.customBlockLabel,
    inputAnchor: args.inputAnchor,
    sequencesRef: args.sequencesRef,
    sequenceType: args.sequenceType,
    umap_neighbors: args.umap_neighbors,
    umap_min_dist: args.umap_min_dist,
    cpu: args.cpu,
    mem: args.mem,
    graphStateUMAP: uiState.graphStateUMAP,
    alignmentModel: uiState.alignmentModel,
  }))
  .transfer(umapPlugin, (v1) => ({
    state: v1.graphStateUMAP,
    selection: undefined,
    chartType: 'scatterplot-umap' as const,
    readonlyInputs: [],
    allowChartDeleting: false,
    allowTitleEditing: false,
  }))
  .migrate<BlockData>('v2', ({ graphStateUMAP: _, ...rest }) => rest)
  .init(() => ({
    defaultBlockLabel: getDefaultBlockLabel({
      sequenceLabels: [],
      umap_neighbors: 15,
      umap_min_dist: 0.5,
    }),
    customBlockLabel: '',
    sequenceType: 'aminoacid',
    sequencesRef: [],
    umap_neighbors: 15,
    umap_min_dist: 0.5,
    mem: 64,
    cpu: 8,
    alignmentModel: {},
  }));

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

type Column = PColumn<DataInfo<TreeNodeAccessor> | TreeNodeAccessor | PColumnValues>;

type Columns = {
  props: Column[];
};

function getColumns(ctx: Pick<BlockRenderCtx<unknown, BlockData>, 'data' | 'resultPool'>): Columns | undefined {
  const anchor = ctx.data.inputAnchor;
  if (anchor === undefined)
    return undefined;

  const anchorSpec = ctx.resultPool.getPColumnSpecByRef(anchor);
  if (anchorSpec === undefined)
    return undefined;

  // all clone properties
  const props = (ctx.resultPool.getAnchoredPColumns(
    { main: anchor },
    [
      {
        axes: [{ anchor: 'main', idx: 1 }],
      },
    ]) ?? [])
    .filter((p) => p.spec.annotations?.['pl7.app/sequence/isAnnotation'] !== 'true');

  return {
    props: props,
  };
}

// ---------------------------------------------------------------------------
// Block model
// ---------------------------------------------------------------------------

export const platforma = BlockModelV3.create(dataModel)

  .args((data) => {
    if (!data.inputAnchor) throw new Error('Dataset is required');
    if (!data.sequencesRef.length) throw new Error('Sequence columns are required');
    if (data.umap_neighbors === undefined) throw new Error('Neighbors is required');
    if (data.umap_min_dist === undefined) throw new Error('Minimum distance is required');
    if (data.mem === undefined) throw new Error('Memory is required');
    if (data.cpu === undefined) throw new Error('CPU is required');
    return {
      inputAnchor: data.inputAnchor,
      sequencesRef: data.sequencesRef,
      sequenceType: data.sequenceType,
      umap_neighbors: data.umap_neighbors,
      umap_min_dist: data.umap_min_dist,
      cpu: data.cpu,
      mem: data.mem,
    };
  })

  .output('inputOptions', (ctx) =>
    ctx.resultPool.getOptions([{
      axes: [
        { name: 'pl7.app/sampleId' },
        { name: 'pl7.app/vdj/clonotypeKey' },
      ],
      annotations: { 'pl7.app/isAnchor': 'true' },
    }, {
      axes: [
        { name: 'pl7.app/sampleId' },
        { name: 'pl7.app/vdj/scClonotypeKey' },
      ],
      annotations: { 'pl7.app/isAnchor': 'true' },
    }]),
  )

  .output('sequenceOptions', (ctx) => {
    const ref = ctx.data.inputAnchor;
    if (ref === undefined) return undefined;

    const isSingleCell = ctx.resultPool.getPColumnSpecByRef(ref)?.axesSpec[1].name === 'pl7.app/vdj/scClonotypeKey';

    const sequenceMatchers = [];

    if (isSingleCell) {
      // Single-cell: get per-chain sequences (all types)
      sequenceMatchers.push({
        axes: [{ anchor: 'main', idx: 1 }],
        name: 'pl7.app/vdj/sequence',
        domain: {
          'pl7.app/vdj/scClonotypeChain/index': 'primary',
        },
      });
      // Single-cell: include scFv construct sequences
      sequenceMatchers.push({
        axes: [{ anchor: 'main', idx: 1 }],
        name: 'pl7.app/vdj/scFv-sequence',
      });
    } else {
      // Bulk: get regular sequences (all types)
      sequenceMatchers.push({
        axes: [{ anchor: 'main', idx: 1 }],
        name: 'pl7.app/vdj/sequence',
      });
    }

    const options = ctx.resultPool.getCanonicalOptions(
      { main: ref },
      sequenceMatchers,
      {
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
      const alphabet = spec?.domain?.['pl7.app/alphabet'] as 'aminoacid' | 'nucleotide' | undefined;
      const isMain = spec?.annotations?.['pl7.app/vdj/isMainSequence'] === 'true';

      return {
        label: option.label,
        value: option.value,
        alphabet,
        isMain,
      };
    });

    // Sort: main sequences first, then alphabetically
    return optionsWithMetadata.sort((a, b) => {
      // Main sequences first
      if (a.isMain && !b.isMain) return -1;
      if (b.isMain && !a.isMain) return 1;
      return 0;
    });
  })

  .output('msaPf', (ctx) => {
    const columns = getColumns(ctx);
    if (!columns) return undefined;

    return createPFrameForGraphs(ctx, columns.props);
  })

  .output('umapOutput', (ctx) => ctx.outputs?.resolve('umapOutput')?.getLogHandle())

  // Create a PTable with the first dimension of the UMAP to test if file is empty
  // output file will only be empty in cases where input data was empty
  .output('umapDim1Table', (ctx) => {
    const pCols = ctx.outputs?.resolve('umapPf')?.getPColumns();
    if (pCols === undefined) {
      return undefined;
    }
    const dim1Column = pCols.find((p) => p.spec.name === 'pl7.app/vdj/umap1');
    if (dim1Column === undefined) {
      return undefined;
    }
    return ctx.createPTable({ columns: [dim1Column] });
  })

  // Return a list of Pcols for plot defaults
  .output('umapPcols', (ctx) => {
    const pCols = ctx.outputs?.resolve('umapPf')?.getPColumns();

    if (pCols === undefined || pCols.length === 0) {
      return undefined;
    }

    return pCols.map(
      (c) =>
        ({
          columnId: c.id,
          spec: c.spec,
        } satisfies PColumnIdAndSpec),
    );
  })

  .title(() => 'Clonotype Space')

  .subtitle((ctx) => ctx.data.customBlockLabel || ctx.data.defaultBlockLabel)

  .sections((_ctx) => ([
    { type: 'link', href: '/', label: strings.titles.main },
  ]))

  .plugin(umapPlugin, {
    blockColumns: (ctx) => ctx.outputs?.resolve('umapPf')?.getPColumns(),
  })

  .done();

export type BlockOutputs = InferOutputsType<typeof platforma>;

export { getDefaultBlockLabel } from './label';
