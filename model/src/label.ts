export function getDefaultBlockLabel(data: {
  sequenceLabels: string[];
  umap_neighbors: number;
  umap_min_dist: number;
}) {
  const parts: string[] = [];
  if (data.sequenceLabels.length > 0) {
    parts.push(data.sequenceLabels.join('+'));
  }
  parts.push(`nbrs: ${data.umap_neighbors}`);
  parts.push(`dist: ${data.umap_min_dist}`);
  return parts.filter(Boolean).join(', ');
}
