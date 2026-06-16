import type { PColumnPredicate } from "@platforma-sdk/model";
import {
  type PTableColumnSpec,
  Annotation,
  Domain,
  PAxisName,
  readAnnotationJson,
  readDomain,
} from "@platforma-sdk/model";

export const isSequenceColumn: PColumnPredicate = ({ spec }) => {
  // Length / annotation columns are not sequence data.
  if (
    spec.name === "pl7.app/vdj/sequenceLength" ||
    spec.name === "pl7.app/sequenceLength" ||
    spec.name === "pl7.app/vdj/sequence/annotation"
  )
    return false;
  // Reject cluster-centroid sequences (their axis is clusterId, not the
  // input's clonotype/variant axis).
  if (spec.axesSpec[0]?.name === "pl7.app/clusterId") return false;
  // Only amino-acid sequences belong in MSA.
  if (readDomain(spec, Domain.Alphabet) !== "aminoacid") return false;
  // Single-cell sequences: reject non-primary chains (e.g. light), but keep
  // chain-less constructs like scFv where the domain field is absent entirely.
  if (spec.axesSpec[0]?.name === PAxisName.VDJ.ScClonotypeKey) {
    const chainIndex = readDomain(spec, Domain.VDJ.ScClonotypeChain.Index);
    if (chainIndex !== undefined && chainIndex !== "primary") return false;
  }

  // Default-select the assembling-feature sequence.
  const isAssemblingFeature =
    readAnnotationJson(spec, Annotation.VDJ.IsAssemblingFeature) ??
    spec.annotations?.["pl7.app/isAssemblingFeature"] === "true";
  return { default: isAssemblingFeature };
};

export function defaultFilters(tSpec: PTableColumnSpec): unknown | undefined {
  console.log("defaultFilters spec", tSpec);
  if (tSpec.type !== "column") {
    return undefined;
  }

  const spec = tSpec.spec;

  if (spec.annotations?.["pl7.app/isScore"] !== "true") return undefined;

  const valueString = spec.annotations?.["pl7.app/score/defaultCutoff"];
  if (valueString === undefined) return undefined;

  if (spec.valueType === "String") {
    const value = JSON.parse(valueString);
    // should be an array of strings
    if (!Array.isArray(value)) {
      console.error("defaultFilters: invalid string filter", valueString);
      return undefined;
    }
    console.log("defaultFilters: string filter", value);
    return {
      type: "string_equals",
      reference: value[0], // @TODO: support multiple values
    };
  } else {
    // Assuming non-String valueType implies a number for 'number_greaterThan'
    const numericValue = parseFloat(valueString);
    if (isNaN(numericValue)) {
      console.error("defaultFilters: invalid numeric value", valueString);
      return undefined;
    }

    const direction = spec.annotations?.["pl7.app/score/rankingOrder"] ?? "increasing";
    if (direction !== "increasing" && direction !== "decreasing") {
      console.error("defaultFilters: invalid ranking order", direction);
      return undefined;
    }

    console.log("defaultFilters: number filter", numericValue, direction);
    return {
      type: direction === "increasing" ? "number_greaterThanOrEqualTo" : "number_lessThanOrEqualTo",
      reference: numericValue,
    };
  }
}
