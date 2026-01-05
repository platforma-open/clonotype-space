import type { GraphMakerState } from '@milaboratories/graph-maker';
import type {
  DataInfo,
  InferOutputsType,
  PColumn,
  PColumnIdAndSpec,
  PColumnValues,
  PFrameHandle,
  PlMultiSequenceAlignmentModel,
  PlRef,
  RenderCtx,
  SUniversalPColumnId,
  TreeNodeAccessor,
} from '@platforma-sdk/model';
import {
  BlockModel,
  createPFrameForGraphs,
} from '@platforma-sdk/model';

export type BlockArgs = {
  inputAnchor?: PlRef;
  sequenceFeatureRef?: SUniversalPColumnId;
  umap_neighbors: number;
  umap_min_dist: number;
  cpu: number;
  mem: number;
};

export type UiState = {
  title?: string;
  graphStateUMAP: GraphMakerState;
  alignmentModel: PlMultiSequenceAlignmentModel;
};

type Column = PColumn<DataInfo<TreeNodeAccessor> | TreeNodeAccessor | PColumnValues>;

type Columns = {
  props: Column[];
};

function getColumns(ctx: RenderCtx<BlockArgs, UiState>): Columns | undefined {
  const anchor = ctx.args.inputAnchor;
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

export const model = BlockModel.create()

  .withArgs<BlockArgs>({
    umap_neighbors: 15,
    umap_min_dist: 0.5,
    mem: 64,
    cpu: 8,
  })

  .withUiState<UiState>({
    title: 'Clonotype Space',
    graphStateUMAP: {
      title: 'Clonotype Space UMAP',
      template: 'dots',
      currentTab: 'settings',
      layersSettings: {
        dots: {
          dotFill: '#99E099',
        },
      },
    },
    alignmentModel: {},
  })

  .argsValid((ctx) =>
    ctx.args.inputAnchor !== undefined
    && ctx.args.sequenceFeatureRef !== undefined
    && ctx.args.umap_neighbors !== undefined
    && ctx.args.umap_min_dist !== undefined
    && ctx.args.mem !== undefined
    && ctx.args.cpu !== undefined,
  )

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
    }], { refsWithEnrichments: true }),
  )

  .output('sequenceOptions', (ctx) => {
    const ref = ctx.args.inputAnchor;
    if (ref === undefined) return undefined;

    const isSingleCell = ctx.resultPool.getPColumnSpecByRef(ref)?.axesSpec[1].name === 'pl7.app/vdj/scClonotypeKey';

    const sequenceMatchers = [];

    if (isSingleCell) {
      // Single-cell: get per-chain sequences
      sequenceMatchers.push({
        axes: [{ anchor: 'main', idx: 1 }],
        name: 'pl7.app/vdj/sequence',
        domain: {
          'pl7.app/vdj/scClonotypeChain/index': 'primary',
        },
      });
    } else {
      // Bulk: get regular sequences
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

    // Sort options to prioritize main protein sequences first
    // Priority: 1) Main aminoacid sequences, 2) Other aminoacid sequences, 3) Main nucleotide, 4) Other nucleotide
    const sortedOptions = [...options].sort((a, b) => {
      // Parse the universal column IDs to get the column specs
      const aColId = JSON.parse(a.value) as never;
      const bColId = JSON.parse(b.value) as never;

      const aColumns = ctx.resultPool.getAnchoredPColumns({ main: ref }, [aColId]);
      const bColumns = ctx.resultPool.getAnchoredPColumns({ main: ref }, [bColId]);

      if (!aColumns || aColumns.length === 0 || !bColumns || bColumns.length === 0) return 0;

      const aSpec = aColumns[0].spec;
      const bSpec = bColumns[0].spec;

      const aIsMain = aSpec.annotations?.['pl7.app/vdj/isMainSequence'] === 'true';
      const bIsMain = bSpec.annotations?.['pl7.app/vdj/isMainSequence'] === 'true';
      const aIsAA = aSpec.domain?.['pl7.app/alphabet'] === 'aminoacid';
      const bIsAA = bSpec.domain?.['pl7.app/alphabet'] === 'aminoacid';

      // Main AA sequences first
      if (aIsMain && aIsAA && !(bIsMain && bIsAA)) return -1;
      if (bIsMain && bIsAA && !(aIsMain && aIsAA)) return 1;

      // Then other AA sequences
      if (aIsAA && !bIsAA) return -1;
      if (bIsAA && !aIsAA) return 1;

      // Then main NT sequences
      if (aIsMain && !bIsMain) return -1;
      if (bIsMain && !aIsMain) return 1;

      return 0;
    });

    return sortedOptions;
  })

  .output('msaPf', (ctx) => {
    const columns = getColumns(ctx);
    if (!columns) return undefined;

    return createPFrameForGraphs(ctx, columns.props);
  })

  .outputWithStatus('umapPf', (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve('umapPf')?.getPColumns();
    if (pCols === undefined) {
      return undefined;
    }

    return createPFrameForGraphs(ctx, pCols);
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

  .output('isRunning', (ctx) => ctx.outputs?.getIsReadyOrError() === false)

  .title((ctx) => ctx.uiState.title ?? 'Clonotype Space')

  .sections((_ctx) => ([
    { type: 'link', href: '/', label: 'Main' },
  ]))

  .done(2);

export type BlockOutputs = InferOutputsType<typeof model>;
