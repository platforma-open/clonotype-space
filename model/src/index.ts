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
import strings from '@milaboratories/strings';
import { getDefaultBlockLabel } from './label';

export type BlockArgs = {
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

export type UiState = {
  graphStateUMAP: GraphMakerState;
  alignmentModel: PlMultiSequenceAlignmentModel;
};

type Column = PColumn<DataInfo<TreeNodeAccessor> | TreeNodeAccessor | PColumnValues>;

type Columns = {
  props: Column[];
};

function getColumns(ctx: Pick<RenderCtx<BlockArgs, UiState>, 'args' | 'resultPool'>): Columns | undefined {
  const anchor = ctx.args?.inputAnchor;
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
  })

  .withUiState<UiState>({
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
    && ctx.args.sequencesRef.length > 0
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
    }]),
  )

  .output('sequenceOptions', (ctx) => {
    const ref = ctx.args.inputAnchor;
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

  .title(() => 'Clonotype Space')

  .subtitle((ctx) => ctx.args.customBlockLabel || ctx.args.defaultBlockLabel)

  .sections((_ctx) => ([
    { type: 'link', href: '/', label: strings.titles.main },
  ]))

  .done(2);

export type BlockOutputs = InferOutputsType<typeof model>;

export { getDefaultBlockLabel } from './label';
