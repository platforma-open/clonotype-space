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
  TreeNodeAccessor,
} from '@platforma-sdk/model';
import {
  BlockModel,
  createPFrameForGraphs,
} from '@platforma-sdk/model';

export type BlockArgs = {
  inputAnchor?: PlRef;
  umap_neighbors: number;
  umap_min_dist: number;
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

  .argsValid((ctx) => ctx.args.inputAnchor !== undefined)

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

  .output('msaPf', (ctx) => {
    const columns = getColumns(ctx);
    if (!columns) return undefined;

    return createPFrameForGraphs(ctx, columns.props);
  })

  .output('umapPf', (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve({ field: 'umapPf', allowPermanentAbsence: true })?.getPColumns();
    if (pCols === undefined) {
      return undefined;
    }

    return createPFrameForGraphs(ctx, pCols);
  })

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

  .done();

export type BlockOutputs = InferOutputsType<typeof model>;
