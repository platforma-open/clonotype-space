import { assertParamsObject } from "@platforma-sdk/block-kind";
import {
  isAnchoredPColumnId,
  isColumnUniversalId,
  isPlRef,
  parseJsonSafely,
  type SUniversalPColumnId,
} from "@platforma-sdk/model";
import { isBoolean, isString } from "es-toolkit";
import type { BlockParams, InputMode, SequenceType } from "./types";

/**
 * The contract at runtime, for params that arrive from a template file rather
 * than from typed code.
 *
 * Each field the contract names is read and checked; a key it does not name is
 * dropped by never being read, so it needs no rejection here. Params written
 * against a different version of the contract are caught by the version in the
 * template entry's `{name}@{selector}` reference, not by a key-set check.
 */
export function parseInitializationParams(value: unknown): BlockParams {
  assertParamsObject(value);

  const params: Record<string, unknown> = {};
  for (const [field, { is, must }] of Object.entries(CONTRACT)) {
    const raw = value[field];
    if (raw === undefined) continue;
    if (!is(raw)) throw new Error(`'${field}' must be ${must}.`);
    params[field] = raw;
  }
  // Every value placed here passed its own field's guard, and `CONTRACT` is
  // proven exhaustive over `BlockParams` by the `satisfies` below.
  return params as BlockParams;
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

type Guard<T> = (value: unknown) => value is T;

/** A guard plus how to finish the sentence "'field' must be …". */
type Check<T> = { readonly is: Guard<T>; readonly must: string };

function check<T>(is: Guard<T>, must: string): Check<T> {
  return { is, must };
}

/** `Number.isInteger` already rejects non-numbers; this only adds the narrowing. */
const isInteger: Guard<number> = (v): v is number => Number.isInteger(v);

/** `umap_min_dist` is a fraction, so integrality is the wrong bar for it. */
const isFiniteNumber: Guard<number> = (v): v is number =>
  typeof v === "number" && Number.isFinite(v);

function oneOf<T extends string>(...allowed: readonly T[]): Guard<T> {
  return (v): v is T => allowed.includes(v as T);
}

function arrayOf<T>(item: Guard<T>): Guard<T[]> {
  return (v): v is T[] => Array.isArray(v) && v.every((e) => item(e));
}

/**
 * A column identifier as this block stores it: a canonically serialized JSON
 * key. `isColumnUniversalId` covers the four key forms the SDK's id encoding
 * uses, but every column id here comes from `resultPool.getCanonicalOptions`,
 * which mints an *anchored* key — a shape none of those four recognizes even
 * though the SDK types it `SUniversalPColumnId`. Both forms are accepted, or
 * the kind would refuse ids the block itself writes.
 */
const isColumnId: Guard<SUniversalPColumnId> = (v): v is SUniversalPColumnId =>
  isString(v) && (isColumnUniversalId(v) || isAnchoredPColumnId(parseJsonSafely(v)));

const REF = "a reference to another block's output";

/**
 * The contract, field by field, at runtime.
 *
 * The `satisfies` clause is the drift guard: it demands an entry for every key
 * `BlockParams` declares, and types each guard against that key's own type. Add
 * a field to the contract and this stops compiling until the check exists —
 * which matters here because every field is optional, so a parser that simply
 * forgot one would otherwise return a valid `BlockParams` and say nothing.
 */
const CONTRACT = {
  inputAnchor: check(isPlRef, REF),
  embeddingRef: check(isPlRef, REF),

  inputMode: check(
    oneOf<InputMode>("sequence-features", "embedding"),
    "one of: sequence-features, embedding",
  ),
  sequencesRef: check(arrayOf(isColumnId), "an array of column ids"),
  sequenceType: check(
    oneOf<SequenceType>("aminoacid", "nucleotide"),
    "one of: aminoacid, nucleotide",
  ),
  umap_neighbors: check(isInteger, "an integer"),
  umap_min_dist: check(isFiniteNumber, "a number"),

  sequenceLabels: check(arrayOf(isString), "an array of strings"),
  selectedEmbeddingLabel: check(isString, "a string"),

  directPerformanceSettings: check(isBoolean, "a boolean"),
  cpu: check(isInteger, "an integer"),
  mem: check(isInteger, "an integer"),
  requireGpu: check(isBoolean, "a boolean"),
  gpuMemory: check(isInteger, "an integer"),

  customBlockLabel: check(isString, "a string"),
} satisfies { [K in keyof BlockParams]-?: Check<NonNullable<BlockParams[K]>> };
