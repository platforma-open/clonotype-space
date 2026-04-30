import type { PColumnPredicate, PColumnSpec } from '@platforma-sdk/model';
import { type PTableColumnSpec, Annotation, Domain, PAxisName, readAnnotationJson, readDomain } from '@platforma-sdk/model';

export const isSequenceColumn: PColumnPredicate = ({ spec }) => {
  const isBulkSequence = (spec: PColumnSpec) =>
    spec.name !== 'pl7.app/vdj/sequenceLength'
    && spec.name !== 'pl7.app/vdj/sequence/annotation'
    && readDomain(spec, Domain.Alphabet) === 'aminoacid'
    // Peptide specific
    && spec.name !== 'pl7.app/sequenceLength'
    // Reject cluster-centroid sequences
    && spec.axesSpec[0]?.name !== 'pl7.app/clusterId';

  const isSingleCellSequence = (spec: PColumnSpec) =>
    spec.name !== 'pl7.app/vdj/sequenceLength'
    && spec.name !== 'pl7.app/vdj/sequence/annotation'
    && readDomain(spec, Domain.VDJ.ScClonotypeChain.Index) === 'primary'
    && readDomain(spec, Domain.Alphabet) === 'aminoacid'
    && spec.axesSpec[0].name === PAxisName.VDJ.ScClonotypeKey;

  return (isBulkSequence(spec) || isSingleCellSequence(spec))
    && {
      // VDJ uses 'pl7.app/vdj/isAssemblingFeature'; peptide-extraction emits
      // the modality-neutral 'pl7.app/isAssemblingFeature' (no `vdj/` prefix).
      default: readAnnotationJson(spec, Annotation.VDJ.IsAssemblingFeature)
        ?? readAnnotationJson(spec, 'pl7.app/isAssemblingFeature' as never)
        ?? false,
    };
};

export function defaultFilters(tSpec: PTableColumnSpec): (unknown | undefined) {
  console.log('defaultFilters spec', tSpec);
  if (tSpec.type !== 'column') {
    return undefined;
  }

  const spec = tSpec.spec;

  if (spec.annotations?.['pl7.app/isScore'] !== 'true')
    return undefined;

  const valueString = spec.annotations?.['pl7.app/score/defaultCutoff'];
  if (valueString === undefined)
    return undefined;

  if (spec.valueType === 'String') {
    const value = JSON.parse(valueString);
    // should be an array of strings
    if (!Array.isArray(value)) {
      console.error('defaultFilters: invalid string filter', valueString);
      return undefined;
    }
    console.log('defaultFilters: string filter', value);
    return {
      type: 'string_equals',
      reference: value[0], // @TODO: support multiple values
    };
  } else {
    // Assuming non-String valueType implies a number for 'number_greaterThan'
    const numericValue = parseFloat(valueString);
    if (isNaN(numericValue)) {
      console.error('defaultFilters: invalid numeric value', valueString);
      return undefined;
    }

    const direction = spec.annotations?.['pl7.app/score/rankingOrder'] ?? 'increasing';
    if (direction !== 'increasing' && direction !== 'decreasing') {
      console.error('defaultFilters: invalid ranking order', direction);
      return undefined;
    }

    console.log('defaultFilters: number filter', numericValue, direction);
    return {
      type: direction === 'increasing' ? 'number_greaterThanOrEqualTo' : 'number_lessThanOrEqualTo',
      reference: numericValue,
    };
  }
};
