// Single source of truth for the auto-subtitle, one variant per input mode.
export function getDefaultBlockLabel(
  data:
    | {
        inputMode: "sequence-features";
        sequenceLabels: string[];
        umap_neighbors: number;
        umap_min_dist: number;
      }
    | {
        inputMode: "embedding";
        embeddingLabel: string;
        umap_neighbors: number;
        umap_min_dist: number;
      },
) {
  const parts: string[] = [];
  if (data.inputMode === "embedding") {
    parts.push(data.embeddingLabel || "Embedding");
    parts.push("UMAP");
  } else if (data.sequenceLabels.length > 0) {
    parts.push(data.sequenceLabels.join("+"));
  }
  parts.push(`nbrs: ${data.umap_neighbors}`);
  parts.push(`dist: ${data.umap_min_dist}`);
  return parts.filter(Boolean).join(", ");
}
